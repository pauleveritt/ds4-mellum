# Mellum SSD: CPU-only preparation

Date: 2026-10-27. Status: approved in chat and implemented on
`codex/mellum-ssd-replay`; independent code review found no correctness blockers.

Builder check: an independent reviewer read this design and reported
"nothing important left". Preparation baseline: `env -u DS4_TEST_MELLUM_MODEL
make -j4 test-mellum` passed in this worktree, including 14 logits-comparison
tests. These are existing CPU-only tests, not new replay tests or GPU results.
The completed implementation adds 21 synthetic replay/CLI tests; both the
dedicated replay target and the full model-free Mellum suite pass. No GPU,
model or storage experiment was performed. The design below records the
approved scope; its implementation choices were approved with that scope.

## Purpose and boundary

Prepare the first decisive locality experiment while GPU access is unavailable.
Build an offline replay tool and synthetic correctness tests before collecting
Mellum routing traces or implementing Metal streaming. The development laptop
is acceptable for this stage; it cannot establish physical 8 GB feasibility.

The user supplied the feasibility brief and authorized using this laptop.
Another agent owns rebasing `main` onto the approved upstream `ee7e840`.
This work starts at Mellum revision `2d0756d3a355f9b99a45adc159992a9ac4221296`
on `codex/mellum-ssd-replay` in an isolated worktree. Do not update upstream,
edit inference code, load the GGUF, benchmark storage, or use the GPU here.
After the upstream work lands, integrate this branch with the resulting main
and record that revision before any model measurements.

The first implementation is deliberately limited to trace validation,
causal LRU replay, a clearly labeled optimistic lower bound, and byte/miss
statistics. Prediction, phase-aware eviction and storage packing are later
comparisons. All routes and Q8 bytes remain authoritative and unchanged.

## Implementation choices proposed by this agent

Use standard-library Python, matching the repository's standalone analysis
scripts, without adding a package or dependencies. Put the tool at
`tools/mellum_ssd_replay.py`, its tests at `tests/test_mellum_ssd_replay.py`,
and its usage/trace contract at `docs/MELLUM_SSD_REPLAY.md`. Add a dedicated
CPU-only `make test-mellum-ssd-replay` target and include it in the model-free
Mellum tests. Keep integration edits small to ease the upstream rebase.

## Version 1 trace contract

JSON Lines: one metadata header followed by complete, ordered forward passes.
The header contains schema version, capture kind (`synthetic` or `measured`),
model identity, engine revision, numerical mode (release or exact prefill),
workload identity, and geometry: layer count, experts per layer, selected
experts per row, and complete-expert byte size. Measured Mellum traces use
28 layers, 64 experts/layer, top-8, and 6,580,224 bytes/expert. Synthetic tests
can use smaller geometries; output must retain their synthetic label.
Model identity is a declared artifact identifier, not proof of a weight hash.

Each subsequent record is one layer of one forward pass, with fields:

- `forward`: zero-based consecutive forward number;
- `segment`: workload phase label such as initial prompt, tool append or topic switch;
- `phase`: `prefill` or `decode`;
- `layer`: layer index, starting at zero for each forward;
- `positions`: increasing nonnegative absolute token positions;
- `experts`: one ordered top-k expert-ID row for each position.

All layers of a forward have identical positions, segment and phase. A decode
forward has one position; prefill may have several. Forward positions advance
without overlap or gaps within a trace; a trace may start at a nonzero position
but always begins with an empty expert cache. A new independent workload uses
a new trace. Idle time is not inferred from this format.

Reject unknown versions, missing provenance, invalid integer types (including
booleans), out-of-range or duplicate expert IDs in a row, incorrect top-k or
row counts, missing/duplicate/out-of-order layers, inconsistent forwards,
empty traces and truncated final forwards. Fail with the source line number
and reason. Preserve expert slot order even though placement decisions use
sets. The future native recorder must export from existing synchronization
points; this design does not implement or claim such a recorder exists.

## Cache model and policies

Each identity is `(layer, expert)`. One slot contains the complete gate/up/down
expert. Capacity is specified in bytes; floor it by expert bytes and report
both requested bytes and effective slots/bytes. Default sweeps are 2, 2.5,
3 and 3.5 GiB, where GiB is 2**30 bytes. At 3 GiB Mellum fits 489 experts.

Each record's union of selected experts is one atomic request group. Resolve
hits against cache contents before loading any misses. Protect every required
expert until the group finishes; admit misses only after evicting entries
outside the group. This prevents slot ordering from evicting an expert that
the same request still needs. Reject a group larger than the available slots
as unsupported by this residency model, rather than report a valid execution.
Such rejection is not a verdict against streaming: prefill expert tiling could
execute that group, but requires a later, explicit tile schedule.

Implement global LRU first. Refresh all required entries after a group in
ascending expert-ID order for a deterministic tie break, independent of
router slot order. All policies start cold and retain cache across segment
and prefill/decode transitions. No arbitrary warm-up records are discarded.

For a lower bound, expand each group into a deterministic sequence of distinct
expert identities and simulate future-aware optimal eviction with optional
bypass (a fetched expert need not be retained). Remove group protection for
this comparison. This relaxed sequential model can only be more optimistic
than the grouped resident-expert model; label it `relaxed-optimal`, never an
implementable policy or an exact grouped optimum. Use an exhaustive tiny-state
oracle in tests to verify optimal misses, including repeated-access and
bypass cases. Do not predict token rate from this bound.

Version 1 models expert-cache bytes only. No prefetch or external protected
entries exist. Staging and non-cache allocations are excluded and explicitly
reported as unmodeled; any future comparison claiming a total native memory
budget must reserve those bytes and track their simultaneous occupancy.

## Reports and interpretation

Emit JSON for each capacity/policy, preserving input provenance and model
assumptions. Report requested/effective cache capacity, peak cached expert
bytes, evictions, distinct experts, selected row references, group requests,
group hits/misses, demand expert loads and demand bytes. Repeated experts in
one prefill group are fetched once; report row-reference hit statistics
separately so prefill reuse cannot masquerade as low I/O.

Include summaries by phase and segment. For decode, aggregate demand bytes
and blocked layers by forward/token; report p50/p95/worst bytes and blocked
layers, using nearest-rank percentiles. Expose the ordered per-decode-token
series with segment labels, so post-append recovery can be inspected without
inventing a recovery threshold. Prefill is reported by grouped forward and
must not be divided into fictional independent decode tokens.

Report cold-start and whole-trace totals explicitly. No synthetic result is
evidence of Mellum locality, SSD bandwidth, model quality or achieved tok/s.
10 tok/s remains an unagreed feasibility checkpoint. Warm, steady-state and
post-append measurements require real traces and explicit windows later.

## Tests and acceptance

Write failing tests before implementation. Use tiny hand-computable traces
and a separate exhaustive reference for the optimistic eviction bound.

1. Cold then repeated decode: exact misses/hits, bytes and blocked layers;
   same expert ID at different layers is a different cache entry.
2. Group pinning: selected resident experts survive loads for the same group;
   reordering router slots does not alter I/O totals.
3. Prefill union: repeated IDs across rows are loaded once; oversized groups
   fail explicitly; prefill does not enter decode percentiles.
4. Phase/segment changes retain cache and expose post-append misses in token
   order; p50/p95/worst agree with hand-computed series.
5. Byte capacities floor correctly: 3 GiB equals 489 Mellum slots; fewer than
   one expert and invalid capacities fail. Peak residency never exceeds budget.
6. Relaxed optimum equals exhaustive search on small traces, never exceeds
   causal grouped misses, and handles bypass without an extra future load.
7. Trace validation rejects malformed/truncated/inconsistent captures with
   line-specific diagnostics and never publishes partial successful results.
8. CLI smoke test reads a synthetic trace and emits parseable, labeled JSON
   for all default sweep capacities; errors return nonzero without a traceback.

Run the dedicated CPU suite and `make test-mellum` without a model environment
variable. Do not run Metal kernel tests, inference or storage benchmarks.

## Subsequent work after GPU availability

Integrate the completed upstream baseline, then implement a native buffered
router recorder and verify it preserves routes/logits without per-layer host
waits. Capture representative coding and tool-use workloads, replay real
traces, and choose further policies from the observed miss curves. Measure
actual expert-size SSD access before selecting a streaming backend design.
`../ds4-engine` integration follows bounded, correct Mellum streaming.
