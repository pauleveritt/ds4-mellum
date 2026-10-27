"""CPU-only synthetic tests: python3 -I tests/test_mellum_ssd_replay.py."""

from copy import deepcopy
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import platform
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
    for field, value in (("schema_version", 3), ("capture_kind", "benchmark"), ("numerical_mode", "unknown")):
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


def test_bound_interpretation_and_phase_counterexample():
    ids = [0, 1, 2, 1, 2] + [0] * 19
    items = [header(layers=1, experts=3, top_k=1)]
    for number, expert in enumerate(ids):
        items.extend(forward(number, [[[expert]]], phase="prefill" if number < 5 else "decode"))
    trace = load(items)
    result = tool().replay(trace, 20, "relaxed-optimal")
    assert result["totals"]["demand_bytes"] == 40
    assert result["by_phase"]["prefill"]["demand_bytes"] == 30
    assert result["by_phase"]["decode"]["demand_bytes"] == 10
    assert result["decode_distribution"]["demand_bytes"]["p95"] == 10
    # Causal witness: always pin expert 0, rotate the only other slot.
    cache, misses = set(), []
    for expert in ids:
        miss = expert not in cache
        misses.append(10 if miss else 0)
        if miss:
            if len(cache) == 2:
                cache.remove(next(item for item in cache if item != 0))
            cache.add(expert)
    assert sum(misses) == 50
    assert misses[5:] == [0] * 19
    assert sorted(misses[5:])[18] == 0  # nearest-rank p95 for 19 tokens
    for policy, kind, scope in (
        ("lru", "causal-policy", "atomic-full-group-residency"),
        ("relaxed-optimal", "lower-bound", "atomic-full-group-residency-or-fixed-sorted-sequence"),
    ):
        assert tool().replay(trace, 20, policy)["interpretation"] == {
            "kind": kind, "objective": "whole-trace-demand-loads-and-bytes",
            "schedule_scope": scope, "diagnostics_are_bounds": False,
            "order_flexible_tiling_covered": False,
        }


def test_sorted_bound_excludes_flexible_group_order():
    trace = single_layer([[0, 2], [1, 2], [0, 1]])
    result = tool().replay(trace, 20, "relaxed-optimal")
    assert result["totals"]["demand_loads"] == 4
    assert optimal_misses((0, 2, 1, 2, 0, 1), 2) == 4
    assert optimal_misses((0, 2, 2, 1, 0, 1), 2) == 3
    assert result["interpretation"]["order_flexible_tiling_covered"] is False


def test_prefill_union_exceeds_capacity_even_when_each_row_fits():
    trace = load([header(layers=1, experts=3),
                  *forward(0, [[[0, 1], [1, 2]]], phase="prefill")])
    for policy in ("lru", "relaxed-optimal"):
        with invalid(2, "group requires 3"):
            tool().replay(trace, 20, policy)
        with invalid(2, "unsupported by atomic group residency"):
            tool().replay(trace, 20, policy)


def manifest_fixture(metadata, *, forwards=1, positions=1, first=0, last=0):
    return {
        "manifest_version": 1, "capture_id": "fixture-capture", "complete": True,
        "artifacts": {"model_id": metadata["model_id"], "tokenizer_id": "fixture-tokenizer",
                      "template_id": "fixture-template", "token_stream_sha256": "0" * 64},
        "build": {"engine_revision": metadata["engine_revision"], "source_state_id": "fixture-source",
                  "binary_sha256": "0" * 64, "recorder_revision": "fixture-recorder"},
        "execution": {"context_tokens": 8192, "prefill_chunk_tokens": 1024,
                      "numerical_mode": metadata["numerical_mode"], "requested_kernel": "fixture-requested",
                      "effective_kernel": "fixture-effective", "device": "fixture-device",
                      "os": "fixture-os", "capabilities": {}},
        "generation": {"mode": "generated", "settings": {"seed": None}},
        "workload": {"protocol_id": "fixture-protocol", "conversation_id": "fixture-conversation",
                     "repository_id": "fixture-repository", "split": "tuning",
                     "continuation_kind": "fixture-continuation"},
        "extent": {"forwards": forwards, "positions": positions,
                   "first_position": first, "last_position": last},
    }


def test_manifest_preserves_extensions():
    metadata = header()
    manifest = manifest_fixture(metadata)
    manifest["future"] = {"nested": [None, True, 1.5, {"anything": "kept"}]}
    manifest["execution"]["capabilities"]["e1"] = [False, 2]
    expected = deepcopy(manifest)
    assert hasattr(tool(), "validate_capture_manifest"), "capture manifest validation is missing"
    assert tool().validate_capture_manifest(manifest, metadata) == expected
    assert manifest == expected


def test_manifest_rejects_missing_wrong_and_inconsistent_fields():
    metadata = header()
    base = manifest_fixture(metadata)
    assert hasattr(tool(), "validate_capture_manifest"), "capture manifest validation is missing"
    cases = []
    # Each declared field is required, and every scalar/section has a type contract.
    for key, value in base.items():
        paths = [(key,)] + ([(key, field) for field in value] if isinstance(value, dict) else [])
        for path in paths:
            for replacement in ("REMOVE", None, []):
                cases.append((path, replacement))
    cases += [
        (("manifest_version",), value) for value in (True, 2, "1")
    ] + [(("complete",), value) for value in (False, 1, "true")]
    for section, fields in (
        ("execution", ("context_tokens", "prefill_chunk_tokens")),
        ("extent", ("forwards", "positions", "first_position", "last_position")),
    ):
        for field in fields:
            bad = (True, -1, "1", 1.5) + ((0,) if field not in ("first_position", "last_position") else ())
            cases += [((section, field), value) for value in bad]
    for section in ("artifacts", "build", "execution", "workload"):
        for field, value in base[section].items():
            if isinstance(value, str):
                cases.append(((section, field), " "))
    cases += [
        (("generation", "mode"), "invalid"), (("workload", "split"), "invalid"),
        (("artifacts", "model_id"), "other"), (("build", "engine_revision"), "other"),
        (("execution", "numerical_mode"), "other"), (("capture_id",), " "),
    ]
    for path in (("artifacts", "token_stream_sha256"), ("build", "binary_sha256")):
        cases += [(path, value) for value in ("a" * 63, "A" * 64, "g" * 64, True)]
    for path, replacement in cases:
        manifest = deepcopy(base)
        parent = manifest if len(path) == 1 else manifest[path[0]]
        if replacement == "REMOVE":
            del parent[path[-1]]
        else:
            parent[path[-1]] = replacement
        with invalid(1, "capture_manifest." + ".".join(path)):
            tool().validate_capture_manifest(manifest, metadata)
    for value in (float("inf"), float("-inf"), float("nan"), {1: "bad"}, {"set"}, (1,)):
        manifest = deepcopy(base)
        manifest["generation"]["settings"]["extension"] = [value]
        with invalid(1, "capture_manifest.generation.settings.extension"):
            tool().validate_capture_manifest(manifest, metadata)


def test_v2_manifest_round_trip_and_v1_compatibility():
    assert tool().run_sweep(load(records()), [40])["trace"] == header()
    for kind in ("synthetic", "measured"):
        metadata = header() if kind == "synthetic" else header(layers=28, experts=64, top_k=8, expert_bytes=6580224)
        metadata.update(schema_version=2, capture_kind=kind)
        manifest = manifest_fixture(metadata)
        manifest["extension"] = {"nested": [None, {"kept": True}]}
        metadata["capture_manifest"] = manifest
        routes = [[list(reversed(range(metadata["geometry"]["top_k"])))]] * metadata["geometry"]["layers"]
        trace = load([metadata, *forward(0, routes)])
        assert trace.requests[0].experts == (tuple(routes[0][0]),)
        assert tool().run_sweep(trace)["trace"] == metadata
    for version in (1, 2, 3, True):
        metadata = header()
        metadata["schema_version"] = version
        if version == 1 and type(version) is int:
            metadata["capture_manifest"] = manifest_fixture(metadata)
            reason = "version 2"
        else:
            reason = "capture_manifest" if version == 2 else "schema_version"
        with invalid(1, reason):
            load([metadata, *records()[1:]])


def test_v2_capture_extent_and_execution_checks():
    metadata = header()
    metadata.update(schema_version=2, capture_manifest=manifest_fixture(metadata, forwards=2, positions=3, first=42, last=44))
    items = [metadata,
             *forward(0, [[[0, 1], [1, 2]], [[0, 1], [2, 3]]], phase="prefill", position=42),
             *forward(1, [[[1, 2]], [[0, 1]]], position=44)]
    assert len(load(items).requests) == 4
    with invalid(4, "extent.forwards"):
        load(items[:-2])  # Truncated at a complete-forward boundary.
    for field, value in (("forwards", 3), ("positions", 4), ("first_position", 41), ("last_position", 45)):
        broken = deepcopy(items)
        broken[0]["capture_manifest"]["extent"][field] = value
        with invalid(6, "extent." + field):
            load(broken)
    for field in ("forwards", "positions", "first_position", "last_position"):
        broken = deepcopy(items)
        broken[0]["capture_manifest"]["extent"][field] = True
        with invalid(1, "extent." + field):
            load(broken)
    for field, bad, good, line in (("context_tokens", 44, 45, 4), ("prefill_chunk_tokens", 1, 2, 2)):
        broken = deepcopy(items)
        broken[0]["capture_manifest"]["execution"][field] = bad
        with invalid(line, "execution." + field):
            load(broken)
        broken[0]["capture_manifest"]["execution"][field] = good
        assert len(load(broken).requests) == 4
    with invalid(5, "truncated final forward"):
        load(items[:-1])


def test_nested_json_exponent_overflow_is_rejected():
    for version in (1, 2):
        metadata = header()
        metadata["schema_version"] = version
        extension = metadata
        if version == 2:
            metadata["capture_manifest"] = manifest_fixture(metadata)
            extension = metadata["capture_manifest"]
        extension["extension"] = {"nested": ["OVERFLOW"]}
        raw = json.dumps(metadata).replace('"OVERFLOW"', '1e999')
        with invalid(1, "finite JSON"):
            tool().load_trace([raw, *(json.dumps(item) for item in records()[1:])])
    raw = json.dumps(header())[:-1] + ', "extension": {"nested": {"x": 1, "x": 2}}}'
    with invalid(1, "duplicate JSON key"):
        tool().load_trace([raw])


def test_report_identity_without_raw_input():
    for kind, version, status in (
        ("synthetic", 1, "synthetic-declared"), ("synthetic", 2, "synthetic-declared"),
        ("measured", 1, "legacy-declared"), ("measured", 2, "manifest-declared"),
    ):
        metadata = header() if kind == "synthetic" else header(layers=28, experts=64, top_k=8, expert_bytes=6580224)
        metadata.update(schema_version=version, capture_kind=kind)
        if version == 2:
            metadata["capture_manifest"] = manifest_fixture(metadata)
        routes = [[list(range(metadata["geometry"]["top_k"]))]] * metadata["geometry"]["layers"]
        trace = load([metadata, *forward(0, routes)])
        assert tool().run_sweep(trace)["evidence"] == {
            "trace_sha256": None, "replay_source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "python_version": platform.python_version(), "provenance_status": status,
        }
        assert tool().run_sweep(trace, trace_sha256="a" * 64)["evidence"]["trace_sha256"] == "a" * 64
        for bad in (True, "", "A" * 64, "a" * 63, "g" * 64):
            try:
                tool().run_sweep(trace, trace_sha256=bad)
            except ValueError as exc:
                assert "trace_sha256" in str(exc)
            else:
                raise AssertionError(f"accepted invalid digest {bad!r}")


def cli_bytes(raw):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl"
        path.write_bytes(raw)
        return subprocess.run([sys.executable, "-I", str(SCRIPT), str(path)], capture_output=True, text=True)


def test_cli_hashes_exact_bytes_and_preserves_manifest():
    items = records()
    items[0].update(schema_version=2, capture_manifest=manifest_fixture(items[0]))
    items[0]["capture_manifest"]["extension"] = {"nested": [None, {"kept": True}]}
    reports, digests = [], []
    for newline in (b"\n", b"\r\n"):
        raw = newline.join(json.dumps(item).encode("utf-8") for item in items) + newline
        result = cli_bytes(raw)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["trace"] == items[0]
        assert report["evidence"]["trace_sha256"] == hashlib.sha256(raw).hexdigest()
        assert report["evidence"]["replay_source_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        reports.append(report)
        digests.append(report["evidence"]["trace_sha256"])
    assert digests[0] != digests[1]
    assert reports[0]["results"] == reports[1]["results"]
    report = json.loads(cli_bytes(("\n".join(json.dumps(item) for item in records()) + "\n").encode()).stdout)
    assert report["trace"] == header()


def test_cli_invalid_utf8_emits_no_report():
    result = cli_bytes(b"\xff\n")
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr and "Traceback" not in result.stderr


def test_cli_preserves_unicode_separators_inside_json_strings():
    for char in ("\u2028", "\u2029", "\u0085"):
        for version in (1, 2):
            items = records()
            text = "before" + char + "after"
            items[0].update(schema_version=version, workload_id=text)
            if version == 2:
                items[0]["capture_manifest"] = manifest_fixture(items[0])
                items[0]["capture_manifest"]["extension"] = {"nested": [text]}
            raw = ("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n").encode("utf-8")
            result = cli_bytes(raw)
            assert result.returncode == 0, (version, hex(ord(char)), result.stderr)
            report = json.loads(result.stdout)
            assert report["trace"] == items[0]
            assert report["evidence"]["trace_sha256"] == hashlib.sha256(raw).hexdigest()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )


if __name__ == "__main__":
    unittest.main()
