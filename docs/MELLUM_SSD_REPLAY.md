# Mellum routing trace replay

This CPU-only tool measures cache demand in an existing routing trace. It
does not load model weights, run inference, benchmark an SSD, or predict
tokens/second. No real Mellum routing traces have been collected by this work.

Run with Python 3.10+ and no third-party dependencies:

```sh
python3 tools/mellum_ssd_replay.py trace.jsonl > replay.json
python3 tools/mellum_ssd_replay.py trace.jsonl --cache-gib 2 2.5 3 3.5 > replay.json
make test-mellum-ssd-replay
```

The four capacities shown are the default sweep. GiB means 2**30 bytes;
decimal capacity arguments are converted exactly, flooring fractional bytes.
Then the cache size is floored to complete-expert slots. A 3 GiB Mellum cache
holds 489 experts (3,217,729,536 effective bytes), out of 1,792 total experts.
The cache budget excludes fixed tensors, KV/scratch, staging, allocation
rounding, the host application and the OS. It is not a total-memory limit.

## Produce a synthetic example

This example fabricates two identical decode tokens with Mellum's geometry.
It exercises the format and cold/warm accounting, not actual model locality:

```sh
python3 - <<'PY' > /tmp/mellum-synthetic.jsonl
import json
print(json.dumps({
    "schema_version": 1,
    "capture_kind": "synthetic",
    "model_id": "synthetic-mellum-shape",
    "engine_revision": "synthetic-no-engine",
    "numerical_mode": "exact-prefill",
    "workload_id": "two-identical-decode-tokens",
    "geometry": {"layers": 28, "experts_per_layer": 64,
                 "top_k": 8, "expert_bytes": 6580224},
}))
for token in range(2):
    for layer in range(28):
        print(json.dumps({
            "forward": token, "segment": "synthetic-answer", "phase": "decode",
            "layer": layer, "positions": [token], "experts": [list(range(8))],
        }))
PY
python3 tools/mellum_ssd_replay.py /tmp/mellum-synthetic.jsonl > /tmp/mellum-synthetic-replay.json
```

Both policies load 224 experts on the first token and none on the second.
That is 1,473,970,176 demand bytes cold. This result is constructed, and is
not a measured hit rate or a warm steady-state performance claim.

## Version 1 JSONL contract

The first line is the metadata object shown above. Required nonempty strings
identify the model artifact, engine revision, numerical mode and workload.
`capture_kind` is `synthetic` or `measured`; this label is declared by the
producer, not independently verified. `numerical_mode` is `release` or
`exact-prefill`. A model identifier is not proof of a file hash.

Geometry fields are positive integers and `top_k <= experts_per_layer`.
Measured Mellum traces require exactly 28 layers, 64 experts/layer, top-8
and 6,580,224 bytes/complete expert. Synthetic tests may use tiny geometries.
Q8 bytes and trained router decisions are not changed by this tool.

Each later line describes one layer of one forward:

| Field | Meaning |
| --- | --- |
| `forward` | Consecutive zero-based forward number |
| `segment` | Workload label, e.g. `initial`, `tool-append`, `after-tool` |
| `phase` | `prefill` or `decode` |
| `layer` | Consecutive zero-based layer within the forward |
| `positions` | Nonempty array of consecutive absolute token positions |
| `experts` | Ordered top-k ID row for each position |

Every layer must be present, in order. All layers of one forward have the
same phase, segment and positions. Decode has exactly one position. Prefill
can have multiple rows. IDs are distinct within each row and in range.
The ordered rows are retained; duplicate IDs across prefill rows are legal.

A trace may begin at a nonzero position, but forwards must then advance
without gaps or overlaps. Every trace starts with an empty expert cache;
there is no seed-cache or independently warmed-session claim. Independent
workloads use separate traces. Idle intervals cannot be inferred.

Malformed JSON, duplicate keys, non-finite JSON constants, invalid integer
types (including booleans), missing provenance, incomplete forwards and
inconsistent records fail with a line number. Empty lines are not records.
End-of-file truncation errors identify the next expected line. The entire
trace and all replay runs must succeed before any JSON is written to stdout;
errors go to stderr with nonzero exit status. A shell may still create an
empty output file when the command fails.

The GPU recorder is a subsequent deliverable. It should buffer exact router
IDs on the GPU and export at normal synchronization points rather than add
per-layer host waits. Existing hidden-state trace timings are not a product
throughput baseline. Record the integrated engine revision when capturing.

## Policies and request groups

Expert identity includes both layer and expert ID. Each cache slot represents
the complete Q8 gate/up/down expert, not a matrix fragment.

`lru` is causal global LRU. Each record's union of expert IDs is one atomic
request group. Resolve hits before loading misses, protect every required
expert until the group finishes, and evict only outside that group. Refresh
required entries in ascending expert-ID order, making recency ties independent
of router slot order. Cache persists across forward, phase and segment changes.

If a group exceeds the available slots, the tool rejects that capacity for
both policies. This says the atomic residency model cannot execute it; it
does not say streaming is impossible. Prefill tiling could execute such a
group, but version 1 has no tile schedule or staging-occupancy model.

`relaxed-optimal` expands each group into ascending expert-ID accesses and
uses future-aware farthest-next-use eviction with optional bypass. A fetched
expert can be served without retention. Group protection is removed, including
after an expert has been served but while its group is still executing.
This is an optimistic lower bound for this equal-size expert model, not an
implementable predictor or an exact optimum under group residency. Tests
compare it to exhaustive tiny-state search. Sequential relaxation may
understate both memory requirements and blocked layers.

## Reading the JSON report

`trace` preserves the required header fields and geometry. `assumptions`
explicitly records cold starts, unmodeled allocations and absent prefetch.
`results` has both policies for each requested capacity, in argument order.

Each result contains:

- `cache`: requested bytes, floored slot count/effective bytes, and peak cached
  expert bytes; temporary loading/staging memory is excluded.
- `totals`: whole-trace demand loads/bytes, evictions, bypasses, distinct
  experts, group counts and row-reference counts.
- `cold_start`: the first complete forward's metrics, explicitly included
  in the totals and distributions. It is not a separate timing measurement.
- `by_phase` and `by_segment`: counters aggregated by their declared labels;
  recurring segment labels are combined, without resetting the cache.
- `forwards`: ordered forward summaries, with positions and phase labels.
- `decode_tokens`: ordered position, forward, segment, demand loads/bytes and
  `blocked_layers`. A layer is considered blocked if it has a demand miss;
  this is a demand-count proxy, not an actual scheduler wait measurement.
- `decode_distribution`: token count and nearest-rank p50/p95/worst demand
  bytes and blocked layers. With no decode tokens, distributions are null.

`group_hits` counts all-hit groups; `group_misses` counts groups with at least
one demand load. `group_expert_hits` counts distinct required experts that
were resident before their group. `row_references` counts all selected IDs
across token rows, while `row_reference_hits` counts references resident
before the group; `row_reference_misses` is the remainder. These are separate
from `demand_loads`: repeated cold references within a prefill group are loaded
only once. No row-weighted metric substitutes for actual demand bytes.

All distributions include cold start and every captured decode token. Choose
and disclose measurement windows before drawing warm steady-state conclusions.
The per-token series supports inspecting post-append recovery without imposing
an arbitrary recovery threshold. No latency, SSD bandwidth or tok/s is
inferred; 10 tok/s remains a proposed, unagreed feasibility checkpoint.

## Next measurements

The development laptop can validate functionality and collect traces when GPU
access is available. Physical 8 GB memory-pressure behavior still needs an
8 GB machine. Measure realistic expert reads while controlling OS page-cache
effects before turning replay demand into an I/O budget. Implement streaming
only after locality and storage measurements justify it, with equality checks
against the matching resident arithmetic path. Integrate `../ds4-engine` later.
