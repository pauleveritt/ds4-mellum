"""CPU-only synthetic tests: python3 -I tests/test_mellum_ssd_replay.py."""

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "mellum_ssd_replay.py"


def tool():
    assert SCRIPT.is_file(), "offline replay tool is not implemented"
    if "mellum_ssd_replay" not in sys.modules:
        spec = importlib.util.spec_from_file_location("mellum_ssd_replay", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["mellum_ssd_replay"]


def header(*, layers=2, experts=4, top_k=2, expert_bytes=10):
    return {
        "schema_version": 1,
        "capture_kind": "synthetic",
        "model_id": "synthetic-tiny",
        "engine_revision": "synthetic-revision",
        "numerical_mode": "exact-prefill",
        "workload_id": "unit-test",
        "geometry": {
            "layers": layers,
            "experts_per_layer": experts,
            "top_k": top_k,
            "expert_bytes": expert_bytes,
        },
    }


def forward(number, routes, *, phase="decode", segment="initial", position=None):
    start = number if position is None else position
    return [
        {
            "forward": number,
            "segment": segment,
            "phase": phase,
            "layer": layer,
            "positions": list(range(start, start + len(rows))),
            "experts": rows,
        }
        for layer, rows in enumerate(routes)
    ]


def records():
    return [header(), *forward(0, [[[1, 0]], [[2, 3]]])]


def load(items):
    return tool().load_trace(json.dumps(item) for item in items)


@contextmanager
def invalid(line, reason):
    try:
        yield
    except tool().TraceError as exc:
        assert f"line {line}:" in str(exc), str(exc)
        assert reason in str(exc), str(exc)
    else:
        raise AssertionError("invalid trace was accepted")


def test_preserves_routes_and_nonzero_start():
    trace = load([header(), *forward(0, [[[1, 0]], [[2, 3]]], position=42)])
    assert trace.requests[0].experts == ((1, 0),)
    assert trace.requests[0].positions == (42,)
    assert trace.header.geometry.expert_bytes == 10


def test_valid_multiphase_forward_sequence():
    trace = load([
        header(),
        *forward(0, [[[0, 1], [1, 2]], [[2, 3], [0, 1]]], phase="prefill"),
        *forward(1, [[[1, 2]], [[0, 1]]], position=2, segment="after-tool"),
    ])
    assert len(trace.requests) == 4
    assert trace.requests[-1].segment == "after-tool"


def test_header_requires_provenance_and_valid_geometry():
    for field in ("model_id", "engine_revision", "workload_id", "capture_kind", "numerical_mode"):
        items = records()
        del items[0][field]
        with invalid(1, field):
            load(items)
    for field, value in (("schema_version", 2), ("capture_kind", "benchmark"), ("numerical_mode", "unknown")):
        items = records()
        items[0][field] = value
        with invalid(1, field):
            load(items)
    for field in ("layers", "experts_per_layer", "top_k", "expert_bytes"):
        for value in (True, 0, -1, 2.5, "2"):
            items = records()
            items[0]["geometry"][field] = value
            with invalid(1, field):
                load(items)
    items = records()
    items[0]["geometry"]["top_k"] = 5
    with invalid(1, "top_k"):
        load(items)


def test_measured_trace_requires_mellum_geometry():
    items = records()
    items[0]["capture_kind"] = "measured"
    with invalid(1, "Mellum geometry"):
        load(items)


def test_rejects_bad_rows_and_integer_types():
    for field, value, reason in (
        ("forward", True, "forward"), ("layer", False, "layer"),
        ("positions", [True], "positions"), ("positions", [-1], "positions"),
        ("positions", [], "positions"), ("experts", [[0, True]], "expert"),
        ("experts", [[0, 0]], "duplicate"), ("experts", [[0, 4]], "expert"),
        ("experts", [[0]], "top_k"), ("experts", [], "row count"),
        ("phase", "idle", "phase"), ("segment", "", "segment"),
    ):
        items = records()
        items[1][field] = value
        with invalid(2, reason):
            load(items)


def test_rejects_incomplete_or_inconsistent_forwards():
    with invalid(3, "truncated"):
        load(records()[:2])
    with invalid(2, "no requests"):
        load([header()])
    with invalid(1, "header"):
        load([])
    for field, value in (("forward", 1), ("layer", 0), ("phase", "prefill"),
                         ("segment", "other"), ("positions", [1])):
        items = records()
        items[2][field] = value
        with invalid(3, field):
            load(items)
    for position in (0, 2):
        with invalid(4, "positions"):
            load([*records(), *forward(1, [[[0, 1]], [[0, 1]]], position=position)])


def test_rejects_gaps_and_non_decode_shapes():
    items = [header(), *forward(0, [[[0, 1], [2, 3]], [[0, 1], [2, 3]]], phase="prefill")]
    for record in items[1:]:
        record["positions"] = [0, 2]
    with invalid(2, "positions"):
        load(items)
    items = [header(), *forward(0, [[[0, 1], [2, 3]], [[0, 1], [2, 3]]])]
    with invalid(2, "decode"):
        load(items)


def test_rejects_non_objects_bad_json_and_duplicate_keys():
    for text, reason in (("{", "JSON"), ("[]", "object"),
                         ('{"schema_version":1,"schema_version":1}', "duplicate"),
                         ('{"value":NaN}', "JSON")):
        with invalid(1, reason):
            tool().load_trace([text])
    for bad in (None, [], {"positions": []}):
        with invalid(2, "object" if not isinstance(bad, dict) else "forward"):
            load([header(), bad])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )


if __name__ == "__main__":
    unittest.main()
