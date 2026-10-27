#!/usr/bin/env python3
"""Validate and replay Mellum router traces without loading weights or using a GPU."""

import argparse
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import heapq
import io
import json
import math
import re
from pathlib import Path
import platform


class TraceError(ValueError):
    """A trace violates the versioned capture contract."""


@dataclass(frozen=True, kw_only=True)
class Geometry:
    layers: int
    experts_per_layer: int
    top_k: int
    expert_bytes: int


MELLUM = Geometry(layers=28, experts_per_layer=64, top_k=8, expert_bytes=6_580_224)


@dataclass(frozen=True, kw_only=True)
class Header:
    schema_version: int
    capture_kind: str
    model_id: str
    engine_revision: str
    numerical_mode: str
    workload_id: str
    geometry: Geometry
    capture_manifest: dict | None = None


@dataclass(frozen=True, kw_only=True)
class Request:
    forward: int
    segment: str
    phase: str
    layer: int
    positions: tuple[int, ...]
    experts: tuple[tuple[int, ...], ...]
    line: int


@dataclass(frozen=True, kw_only=True)
class Trace:
    header: Header
    requests: tuple[Request, ...]


DEFAULT_CAPACITIES = (2 * 2**30, 5 * 2**29, 3 * 2**30, 7 * 2**29)
COUNTERS = (
    "group_requests", "group_hits", "group_misses", "group_expert_hits",
    "row_references", "row_reference_hits", "row_reference_misses",
    "demand_loads", "demand_bytes", "evictions", "bypasses",
)


def fail(line: int, reason: str) -> None:
    raise TraceError(f"line {line}: {reason}")


def object_value(value: object, line: int, name: str) -> dict:
    if not isinstance(value, dict):
        fail(line, f"{name} must be an object")
    return value


def integer(value: object, line: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(line, f"{name} must be an integer >= {minimum}")
    return value


def string_field(data: dict, key: str, line: int) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(line, f"{key} must be a nonempty string")
    return value


def unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant {value}")


def parse_object(text: str, line: int) -> dict:
    try:
        raw = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except ValueError as exc:
        fail(line, f"invalid JSON: {exc}")
    finite_json(raw, line, "record")
    return object_value(raw, line, "record")


def finite_json(value: object, line: int, path: str) -> None:
    """Validate JSON values, including exponent overflow and nested extensions."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, line, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                fail(line, f"{path} keys must be strings")
            finite_json(item, line, f"{path}.{key}")
        return
    fail(line, f"{path} must contain only finite JSON values")


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_capture_manifest(raw: object, header_fields: dict) -> dict:
    """Preserve declared provenance; validate its shape, not its authenticity."""
    root = "capture_manifest"
    manifest = object_value(raw, 1, root)
    finite_json(manifest, 1, root)
    if integer(manifest.get("manifest_version"), 1, f"{root}.manifest_version", 1) != 1:
        fail(1, f"{root}.manifest_version must be 1")
    if manifest.get("complete") is not True:
        fail(1, f"{root}.complete must be true")
    fields = {
        "": ("capture_id",),
        "artifacts": ("model_id", "tokenizer_id", "template_id", "token_stream_sha256"),
        "build": ("engine_revision", "source_state_id", "binary_sha256", "recorder_revision"),
        "execution": ("numerical_mode", "requested_kernel", "effective_kernel", "device", "os"),
        "generation": ("mode",),
        "workload": ("protocol_id", "conversation_id", "repository_id", "split", "continuation_kind"),
        "extent": (),
    }
    sections = {}
    for section, names in fields.items():
        prefix = f"{root}.{section}" if section else root
        data = object_value(manifest.get(section), 1, prefix) if section else manifest
        sections[section] = data
        for name in names:
            value = data.get(name)
            if not isinstance(value, str) or not value.strip():
                fail(1, f"{prefix}.{name} must be a nonempty string")
            if name.endswith("_sha256") and not is_sha256(value):
                fail(1, f"{prefix}.{name} must be a lowercase SHA-256 digest")
    for section, name, minimum in (
        ("execution", "context_tokens", 1), ("execution", "prefill_chunk_tokens", 1),
        ("extent", "forwards", 1), ("extent", "positions", 1),
        ("extent", "first_position", 0), ("extent", "last_position", 0),
    ):
        integer(sections[section].get(name), 1, f"{root}.{section}.{name}", minimum)
    for section, name in (("execution", "capabilities"), ("generation", "settings")):
        object_value(sections[section].get(name), 1, f"{root}.{section}.{name}")
    for section, name, allowed in (
        ("generation", "mode", ("generated", "teacher-forced")),
        ("workload", "split", ("tuning", "evaluation")),
    ):
        if sections[section][name] not in allowed:
            fail(1, f"{root}.{section}.{name} must be one of {allowed}")
    for section, name in (
        ("artifacts", "model_id"), ("build", "engine_revision"), ("execution", "numerical_mode"),
    ):
        if sections[section][name] != header_fields[name]:
            fail(1, f"{root}.{section}.{name} must match header {name}")
    return manifest


def parse_header(raw: dict) -> Header:
    version = integer(raw.get("schema_version"), 1, "schema_version", 1)
    if version not in (1, 2):
        fail(1, "unsupported schema_version")
    geometry = object_value(raw.get("geometry"), 1, "geometry")
    shape = Geometry(**{
        key: integer(geometry.get(key), 1, key, 1)
        for key in ("layers", "experts_per_layer", "top_k", "expert_bytes")
    })
    if shape.top_k > shape.experts_per_layer:
        fail(1, "top_k exceeds experts_per_layer")
    fields = {
        key: string_field(raw, key, 1)
        for key in ("capture_kind", "model_id", "engine_revision", "numerical_mode", "workload_id")
    }
    if fields["capture_kind"] not in ("synthetic", "measured"):
        fail(1, "capture_kind must be synthetic or measured")
    if fields["numerical_mode"] not in ("release", "exact-prefill"):
        fail(1, "numerical_mode must be release or exact-prefill")
    if fields["capture_kind"] == "measured" and shape != MELLUM:
        fail(1, "measured traces require Mellum geometry (28/64/8, 6580224 bytes)")
    manifest = None
    if version == 2:
        manifest = validate_capture_manifest(raw.get("capture_manifest"), fields)
    elif "capture_manifest" in raw:
        fail(1, "capture_manifest requires schema version 2")
    return Header(schema_version=version, geometry=shape, capture_manifest=manifest, **fields)


def parse_request(raw: dict, line: int, geometry: Geometry) -> Request:
    number = integer(raw.get("forward"), line, "forward")
    layer = integer(raw.get("layer"), line, "layer")
    segment = string_field(raw, "segment", line)
    phase = raw.get("phase")
    if phase not in ("prefill", "decode"):
        fail(line, "phase must be prefill or decode")
    positions = raw.get("positions")
    if not isinstance(positions, list) or not positions:
        fail(line, "positions must be a nonempty array")
    for position in positions:
        integer(position, line, "positions entry")
    if any(b != a + 1 for a, b in zip(positions, positions[1:])):
        fail(line, "positions must increase consecutively")
    if phase == "decode" and len(positions) != 1:
        fail(line, "decode must have exactly one position")
    rows = raw.get("experts")
    if not isinstance(rows, list) or len(rows) != len(positions):
        fail(line, "expert row count must equal positions count")
    for row in rows:
        if not isinstance(row, list) or len(row) != geometry.top_k:
            fail(line, "each expert row must contain top_k IDs")
        for expert in row:
            integer(expert, line, "expert ID")
            if expert >= geometry.experts_per_layer:
                fail(line, "expert ID out of range")
        if len(set(row)) != len(row):
            fail(line, "duplicate expert ID in row")
    return Request(
        forward=number, segment=segment, phase=phase, layer=layer,
        positions=tuple(positions), experts=tuple(tuple(row) for row in rows), line=line,
    )


def load_trace(lines: Iterable[str]) -> Trace:
    """Validate the entire capture before returning any requests for replay."""
    source = iter(lines)
    try:
        header_text = next(source)
    except StopIteration:
        fail(1, "missing header")
    header = parse_header(parse_object(header_text, 1))
    requests = []
    anchor = None
    previous_end = None
    line = 1
    for line, text in enumerate(source, 2):
        request = parse_request(parse_object(text, line), line, header.geometry)
        expected_forward, expected_layer = divmod(len(requests), header.geometry.layers)
        if request.forward != expected_forward:
            fail(line, f"forward must be {expected_forward}")
        if request.layer != expected_layer:
            fail(line, f"layer must be {expected_layer}")
        if request.layer == 0:
            if previous_end is not None and request.positions[0] != previous_end + 1:
                fail(line, "positions must continue from previous forward")
            anchor = request
            previous_end = request.positions[-1]
        else:
            for name, current, original in (
                ("positions", request.positions, anchor.positions),
                ("segment", request.segment, anchor.segment),
                ("phase", request.phase, anchor.phase),
            ):
                if current != original:
                    fail(line, f"{name} differs within forward")
        requests.append(request)
    if not requests:
        fail(line + 1, "no requests after header")
    if len(requests) % header.geometry.layers:
        fail(line + 1, "truncated final forward")
    if header.capture_manifest is not None:
        manifest = header.capture_manifest
        starts = requests[::header.geometry.layers]
        actual = {
            "forwards": len(starts), "positions": sum(len(item.positions) for item in starts),
            "first_position": starts[0].positions[0], "last_position": starts[-1].positions[-1],
        }
        for key, value in actual.items():
            if manifest["extent"][key] != value:
                fail(line + 1, f"capture_manifest.extent.{key} must equal captured value {value}")
        config = manifest["execution"]
        for item in starts:
            if item.positions[-1] >= config["context_tokens"]:
                fail(item.line, "capture_manifest.execution.context_tokens must exceed captured positions")
            if item.phase == "prefill" and len(item.positions) > config["prefill_chunk_tokens"]:
                fail(item.line, "capture_manifest.execution.prefill_chunk_tokens is smaller than prefill")
    return Trace(header=header, requests=tuple(requests))


def group_ids(request: Request) -> tuple[tuple[int, int], ...]:
    return tuple((request.layer, expert) for expert in sorted({
        expert for row in request.experts for expert in row
    }))


class LRU:
    """Whole-group residency, with router-order-independent refresh ties."""

    def __init__(self, slots: int):
        self.slots = slots
        self.cache: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.peak = 0

    def serve(self, required: tuple[tuple[int, int], ...]) -> tuple[set, int, int, int]:
        protected = set(required)
        hits = protected.intersection(self.cache)
        missing = [key for key in required if key not in hits]
        evictions = 0
        for key in missing:
            if len(self.cache) == self.slots:
                victim = next(item for item in self.cache if item not in protected)
                del self.cache[victim]
                evictions += 1
            self.cache[key] = None
        for key in required:
            self.cache.move_to_end(key)
        self.peak = max(self.peak, len(self.cache))
        return hits, len(missing), evictions, 0


class RelaxedOptimal:
    """Sequential farthest-next-use eviction, including bypass on a miss.

    Selected experts cease to be protected once their sequential access is
    served. This relaxation is an optimistic bound, not a native cache policy.
    """

    def __init__(self, slots: int, groups: Sequence[tuple[tuple[int, int], ...]]):
        self.slots = slots
        self.future: dict[tuple[int, int], deque[int]] = defaultdict(deque)
        index = 0
        for group in groups:
            for key in group:
                self.future[key].append(index)
                index += 1
        self.never = index
        self.cache: dict[tuple[int, int], int] = {}
        self.heap: list[tuple[int, tuple[int, int]]] = []
        self.peak = 0

    def serve(self, required: tuple[tuple[int, int], ...]) -> tuple[set, int, int, int]:
        hits = set(required).intersection(self.cache)
        loads = evictions = bypasses = 0
        for key in required:
            remaining = self.future[key]
            remaining.popleft()
            next_use = remaining[0] if remaining else self.never
            if key not in self.cache:
                loads += 1
                if len(self.cache) == self.slots:
                    # Remove obsolete heap versions before consulting the farthest use.
                    while self.heap and self.cache.get(self.heap[0][1]) != -self.heap[0][0]:
                        heapq.heappop(self.heap)
                    farthest, victim = self.heap[0]
                    if next_use >= -farthest:
                        bypasses += 1
                        continue
                    heapq.heappop(self.heap)
                    del self.cache[victim]
                    evictions += 1
            self.cache[key] = next_use
            heapq.heappush(self.heap, (-next_use, key))
            self.peak = max(self.peak, len(self.cache))
            # A hot resident set must not grow an unbounded lazy-version heap.
            if len(self.heap) > 2 * len(self.cache):
                self.heap = [(-future, item) for item, future in self.cache.items()]
                heapq.heapify(self.heap)
        return hits, loads, evictions, bypasses


def empty_counts() -> dict[str, int]:
    return dict.fromkeys(COUNTERS, 0)


def add_counts(target: dict, counts: dict[str, int]) -> None:
    for key in COUNTERS:
        target[key] += counts[key]


def distribution(values: Sequence[int]) -> dict[str, int] | None:
    """Nearest-rank percentiles; no invented zero measurements for an empty phase."""
    if not values:
        return None
    ordered = sorted(values)
    return {
        "p50": ordered[(50 * len(ordered) + 99) // 100 - 1],
        "p95": ordered[(95 * len(ordered) + 99) // 100 - 1],
        "worst": ordered[-1],
    }


def replay(trace: Trace, capacity_bytes: int, policy: str) -> dict:
    """Replay a complete trace. Only expert-cache residency counts toward this budget."""
    expert_bytes = trace.header.geometry.expert_bytes
    if type(capacity_bytes) is not int or capacity_bytes < expert_bytes:
        raise ValueError(f"capacity must be integer bytes >= one expert ({expert_bytes})")
    slots = capacity_bytes // expert_bytes
    groups = tuple(group_ids(request) for request in trace.requests)
    for request, required in zip(trace.requests, groups):
        if len(required) > slots:
            fail(request.line, f"group requires {len(required)} experts, capacity holds {slots}; "
                 "unsupported by atomic group residency; prefill tiling is not modeled")
    if policy == "lru":
        cache = LRU(slots)
    elif policy == "relaxed-optimal":
        cache = RelaxedOptimal(slots, groups)
    else:
        raise ValueError(f"unknown cache policy {policy!r}")
    totals = empty_counts()
    by_phase = {}
    by_segment = {}
    forwards = []
    distinct = set()
    for request, required in zip(trace.requests, groups):
        distinct.update(required)
        hits, loads, evictions, bypasses = cache.serve(required)
        references = sum(len(row) for row in request.experts)
        row_hits = sum((request.layer, expert) in hits for row in request.experts for expert in row)
        counts = dict(
            group_requests=1, group_hits=int(loads == 0), group_misses=int(loads > 0),
            group_expert_hits=len(hits), row_references=references,
            row_reference_hits=row_hits, row_reference_misses=references - row_hits,
            demand_loads=loads, demand_bytes=loads * expert_bytes,
            evictions=evictions, bypasses=bypasses,
        )
        add_counts(totals, counts)
        add_counts(by_phase.setdefault(request.phase, empty_counts()), counts)
        add_counts(by_segment.setdefault(request.segment, empty_counts()), counts)
        if request.layer == 0:
            forwards.append(dict(
                forward=request.forward, phase=request.phase, segment=request.segment,
                positions=list(request.positions), **empty_counts(),
            ))
        add_counts(forwards[-1], counts)
    tokens = [dict(
        forward=item["forward"], position=item["positions"][0], segment=item["segment"],
        demand_bytes=item["demand_bytes"], demand_loads=item["demand_loads"],
        blocked_layers=item["group_misses"],
    ) for item in forwards if item["phase"] == "decode"]
    totals["distinct_experts"] = len(distinct)
    return {
        "policy": policy,
        "interpretation": {
            "kind": "lower-bound" if policy == "relaxed-optimal" else "causal-policy",
            "objective": "whole-trace-demand-loads-and-bytes",
            "schedule_scope": ("atomic-full-group-residency-or-fixed-sorted-sequence"
                               if policy == "relaxed-optimal" else "atomic-full-group-residency"),
            "diagnostics_are_bounds": False,
            "order_flexible_tiling_covered": False,
        },
        "cache": dict(requested_bytes=capacity_bytes, effective_bytes=slots * expert_bytes,
                      slots=slots, peak_bytes=cache.peak * expert_bytes),
        "totals": totals,
        "cold_start": forwards[0],
        "by_phase": by_phase,
        "by_segment": by_segment,
        "forwards": forwards,
        "decode_tokens": tokens,
        "decode_distribution": dict(
            tokens=len(tokens), demand_bytes=distribution([t["demand_bytes"] for t in tokens]),
            blocked_layers=distribution([t["blocked_layers"] for t in tokens]),
        ),
    }


def report_identity(trace: Trace, trace_sha256: str | None) -> dict:
    if trace_sha256 is not None and not is_sha256(trace_sha256):
        raise ValueError("trace_sha256 must be a lowercase SHA-256 digest or null")
    if trace.header.capture_kind == "synthetic":
        status = "synthetic-declared"
    else:
        status = "manifest-declared" if trace.header.schema_version == 2 else "legacy-declared"
    return {
        "trace_sha256": trace_sha256,
        "replay_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "provenance_status": status,
    }


def run_sweep(trace: Trace, capacities: Sequence[int] = DEFAULT_CAPACITIES, *,
              trace_sha256: str | None = None) -> dict:
    if not capacities:
        raise ValueError("at least one capacity is required")
    metadata = asdict(trace.header)
    if trace.header.capture_manifest is None:
        del metadata["capture_manifest"]
    return {
        "trace": metadata,
        "evidence": report_identity(trace, trace_sha256),
        "assumptions": {
            "cache_starts_cold": True,
            "capacity_scope": "complete expert weights only",
            "staging_bytes_modeled": False,
            "other_process_and_system_bytes_modeled": False,
            "prefetch_modeled": False,
            "group_model": "atomic per-layer union; oversized groups rejected",
            "row_hits": "references resident before the group begins",
            "relaxed-optimal": "whole-trace loads/bytes bound for atomic residency or fixed sorted order; "
                               "future-aware bypass; not a grouped optimum or a bound for flexible tiling",
            "diagnostics": "phase/segment/forward/token/cold-start/tail and blocked layers are policy outcomes, not bounds",
            "percentiles": "nearest rank over all decode tokens, including cold start",
            "throughput_prediction": None,
        },
        "results": [replay(trace, capacity, policy) for capacity in capacities
                    for policy in ("lru", "relaxed-optimal")],
    }


def gib_bytes(text: str) -> int:
    """Convert a decimal GiB value exactly, flooring only fractional bytes."""
    try:
        value = Decimal(text)
        if not value.is_finite() or value <= 0:
            raise ValueError("not a positive finite value")
        numerator, denominator = value.as_integer_ratio()
        result = numerator * 2**30 // denominator
        if not result:
            raise ValueError("smaller than one byte")
        return result
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"cache GiB must be positive and finite: {text!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="complete version 1 or 2 routing trace, JSON Lines")
    parser.add_argument(
        "--cache-gib", type=gib_bytes, nargs="+", metavar="GIB", default=DEFAULT_CAPACITIES,
        help="expert-cache capacities in GiB (default: 2 2.5 3 3.5); excludes staging and other memory",
    )
    args = parser.parse_args()
    try:
        raw = args.trace.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        trace = load_trace(io.StringIO(raw.decode("utf-8"), newline=None))
        report = run_sweep(trace, args.cache_gib, trace_sha256=digest)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    # No partial success document if a later capacity or policy fails.
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
