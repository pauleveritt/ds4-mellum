"""CPU-only synthetic tests: python3 -I tests/test_mellum_ssd_replay.py."""

from contextlib import contextmanager
from functools import lru_cache
import importlib.util
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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


def single_layer(groups, *, expert_bytes=10):
    """One row per group, for hand-computable decode scenarios."""
    return load([
        header(layers=1, experts=4, top_k=len(groups[0]), expert_bytes=expert_bytes),
        *(forward(i, [[list(group)]])[0] for i, group in enumerate(groups)),
    ])


def test_cold_then_warm_layer_identity_and_byte_accounting():
    trace = load([*records(), *forward(1, [[[1, 0]], [[2, 3]]])])
    report = tool().replay(trace, 45, "lru")
    assert report["cache"] == {"requested_bytes": 45, "effective_bytes": 40, "slots": 4, "peak_bytes": 40}
    totals = report["totals"]
    assert totals["demand_loads"] == 4
    assert totals["demand_bytes"] == 40
    assert totals["row_references"] == 8
    assert totals["row_reference_hits"] == 4
    assert totals["group_hits"] == 2
    assert totals["group_misses"] == 2
    assert report["cold_start"]["demand_bytes"] == 40
    assert [token["blocked_layers"] for token in report["decode_tokens"]] == [2, 0]
    other = load([header(top_k=1), *forward(0, [[[0]], [[0]]])])
    assert tool().replay(other, 10, "lru")["totals"]["demand_loads"] == 2


def test_group_pinning_and_router_order_independence():
    # Group 2 needs resident 0, although 0 is oldest; loading 2 must evict 1.
    for groups in (([0, 1], [2, 0], [0, 2]), ([1, 0], [0, 2], [2, 0])):
        report = tool().replay(single_layer(groups), 20, "lru")
        assert [t["demand_bytes"] for t in report["decode_tokens"]] == [20, 10, 0]
        assert report["totals"]["evictions"] == 1
        assert report["cache"]["peak_bytes"] == 20


def test_prefill_union_is_counted_once_and_excluded_from_decode_distribution():
    trace = load([
        header(layers=1),
        *forward(0, [[[0, 1], [0, 1]]], phase="prefill"),
        *forward(1, [[[0, 1]]], position=2, segment="answer"),
    ])
    report = tool().replay(trace, 20, "lru")
    assert report["totals"]["row_references"] == 6
    assert report["totals"]["row_reference_hits"] == 2
    assert report["totals"]["demand_loads"] == 2
    assert report["by_phase"]["prefill"]["demand_bytes"] == 20
    assert report["decode_distribution"]["tokens"] == 1
    assert report["decode_distribution"]["demand_bytes"] == {"p50": 0, "p95": 0, "worst": 0}


def test_segments_retain_cache_and_preserve_post_append_recovery():
    trace = load([
        header(layers=1, top_k=1),
        *forward(0, [[[0]]]),
        *forward(1, [[[0]]]),
        *forward(2, [[[1]]], phase="prefill", segment="tool"),
        *forward(3, [[[0]]], segment="after-tool"),
        *forward(4, [[[0]]], segment="after-tool"),
    ])
    report = tool().replay(trace, 10, "lru")
    assert [(t["position"], t["segment"], t["demand_bytes"]) for t in report["decode_tokens"]] == [
        (0, "initial", 10), (1, "initial", 0), (3, "after-tool", 10), (4, "after-tool", 0),
    ]
    assert report["by_segment"]["after-tool"]["demand_bytes"] == 10
    assert report["decode_distribution"]["demand_bytes"] == {"p50": 0, "p95": 10, "worst": 10}
    assert report["decode_distribution"]["blocked_layers"] == {"p50": 0, "p95": 1, "worst": 1}


def test_oversized_group_and_invalid_capacity_fail_for_both_policies():
    trace = single_layer(([0, 1],))
    for policy in ("lru", "relaxed-optimal"):
        with invalid(2, "group"):
            tool().replay(trace, 10, policy)
        for capacity in (0, -1, True, 5, 20.5):
            try:
                tool().replay(trace, capacity, policy)
            except ValueError as exc:
                assert "capacity" in str(exc), str(exc)
            else:
                raise AssertionError("invalid capacity accepted")


def test_fractional_mellum_budget_and_empty_decode_distribution():
    trace = load([
        header(layers=1, expert_bytes=6_580_224),
        *forward(0, [[[0, 1]]], phase="prefill"),
    ])
    report = tool().replay(trace, 3 * 2**30, "lru")
    assert report["cache"]["slots"] == 489
    assert report["cache"]["effective_bytes"] == 3_217_729_536
    assert report["decode_distribution"] == {"tokens": 0, "demand_bytes": None, "blocked_layers": None}


def optimal_misses(sequence, capacity):
    """Enumerate legal resident sets, including bypass; no Belady logic."""
    @lru_cache(maxsize=None)
    def solve(index, resident):
        if index == len(sequence):
            return 0
        expert = sequence[index]
        if expert in resident:
            return solve(index + 1, resident)
        candidates = [resident]  # A miss can be served without being retained.
        if len(resident) < capacity:
            candidates.append(tuple(sorted((*resident, expert))))
        else:
            candidates.extend(
                tuple(sorted((set(resident) - {victim}) | {expert}))
                for victim in resident
            )
        return 1 + min(solve(index + 1, state) for state in candidates)
    return solve(0, ())


def test_relaxed_optimal_matches_exhaustive_search_and_never_exceeds_lru():
    for sequence in product(range(3), repeat=5):
        trace = single_layer(tuple([expert] for expert in sequence))
        for slots in (1, 2, 3):
            bound = tool().replay(trace, slots * 10, "relaxed-optimal")
            causal = tool().replay(trace, slots * 10, "lru")
            assert bound["totals"]["demand_loads"] == optimal_misses(sequence, slots), sequence
            assert bound["totals"]["demand_loads"] <= causal["totals"]["demand_loads"], sequence
            assert bound["cache"]["peak_bytes"] <= slots * 10
    for groups in product(([0, 1], [0, 2], [1, 2]), repeat=3):
        sequence = tuple(expert for group in groups for expert in sorted(group))
        trace = single_layer(groups)
        bound = tool().replay(trace, 20, "relaxed-optimal")
        assert bound["totals"]["demand_loads"] == optimal_misses(sequence, 2)
        assert bound["totals"]["demand_loads"] <= tool().replay(trace, 20, "lru")["totals"]["demand_loads"]


def test_relaxed_bound_can_bypass_and_is_not_a_grouped_optimum():
    # A single slot retains 0 across a one-use 1 by bypassing 1.
    report = tool().replay(single_layer(([0], [1], [0])), 10, "relaxed-optimal")
    assert [t["demand_bytes"] for t in report["decode_tokens"]] == [10, 10, 0]
    assert report["totals"]["bypasses"] == 1
    assert report["totals"]["evictions"] == 0
    # Relaxation may discard an already served member of the current group.
    trace = single_layer(([0, 1], [0, 2], [0, 1]))
    bound = tool().replay(trace, 20, "relaxed-optimal")
    assert bound["totals"]["demand_loads"] == 3
    assert tool().replay(trace, 20, "lru")["totals"]["demand_loads"] == 4


def test_sweep_preserves_provenance_and_labels_unmodeled_memory():
    trace = single_layer(([0], [0]))
    report = tool().run_sweep(trace, [20])
    assert report["trace"]["capture_kind"] == "synthetic"
    assert report["trace"]["engine_revision"] == "synthetic-revision"
    assert report["assumptions"]["cache_starts_cold"] is True
    assert report["assumptions"]["staging_bytes_modeled"] is False
    assert {r["policy"] for r in report["results"]} == {"lru", "relaxed-optimal"}
    assert report["results"][0]["totals"]["distinct_experts"] == 1


def cli(items, *args, raw=None):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl"
        path.write_text(raw if raw is not None else "\n".join(json.dumps(item) for item in items), encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-I", str(SCRIPT), str(path), *args],
            capture_output=True, text=True,
        )


def test_cli_default_sweep_is_one_provenance_labeled_json_document():
    result = cli(records())
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "CLI did not emit a replay report"
    report = json.loads(result.stdout)
    assert report["trace"]["capture_kind"] == "synthetic"
    assert sorted({r["cache"]["requested_bytes"] for r in report["results"]}) == [
        2_147_483_648, 2_684_354_560, 3_221_225_472, 3_758_096_384,
    ]
    assert len(report["results"]) == 8
    assert not result.stderr


def test_cli_fractional_gib_is_floored_exactly_to_bytes():
    result = cli(records(), "--cache-gib", "0.00000002", "0.5")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "CLI did not emit a replay report"
    caches = [item["cache"] for item in json.loads(result.stdout)["results"]]
    assert [cache["requested_bytes"] for cache in caches] == [21, 21, 536_870_912, 536_870_912]
    assert [cache["effective_bytes"] for cache in caches[:2]] == [20, 20]


def test_cli_errors_never_publish_partial_results_or_tracebacks():
    cases = [
        cli(records(), "--cache-gib", "1", "0.00000001"),
        cli(records()[:2]),
        cli([], raw="{bad JSON"),
        cli([], raw=""),
        cli([], raw="\n"),
    ]
    cases.extend(cli(records(), "--cache-gib", value) for value in ("0", "-1", "NaN", "Infinity", "no", "1e-40"))
    cases.append(subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(SCRIPT / "absent.jsonl")], capture_output=True, text=True,
    ))
    for result in cases:
        assert result.returncode != 0, (result.stdout, result.stderr)
        assert not result.stdout, result.stdout
        assert result.stderr, "failure did not explain why"
        assert "Traceback" not in result.stderr, result.stderr


def test_measured_mellum_provenance_is_preserved_by_cli():
    metadata = header(layers=28, experts=64, top_k=8, expert_bytes=6_580_224)
    metadata["capture_kind"] = "measured"
    metadata["model_id"] = "declared-fixture-artifact"
    result = cli([metadata, *forward(0, [[[0, 1, 2, 3, 4, 5, 6, 7]]] * 28)])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "CLI did not emit a replay report"
    report = json.loads(result.stdout)
    assert report["trace"] == metadata
    assert report["results"][0]["totals"]["demand_bytes"] == 1_473_970_176


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )


if __name__ == "__main__":
    unittest.main()
