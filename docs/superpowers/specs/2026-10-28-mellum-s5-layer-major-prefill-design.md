# S5 step 2: grouped layer-major prefill for streamed Mellum

Date: 2026-10-28. Branch `codex/mellum-ssd-replay`, from `9b8515d`. Owner decisions are marked
**(owner)**; everything else is the designer's call and may be revisited in planning. Evidence:
[options note](../research/2026-10-28-mellum-s5-layer-major-prefill-options.md),
[S5 measurement 01](../research/2026-10-28-mellum-s5-measurement-01.md).

## Intent

Streamed Mellum's cold first token on a 30,620-token prompt is 60.6 s (median, n=3), against resident's
29.6 s. About 31 s of the gap is expert re-reads: every 1,024-row prefill chunk loads nearly all 1,792
experts again, 50,806 loads (334 GB) per prompt. A layer's 64 experts (421 MB) fit the 6 GiB cache many
times over. Only the loop order, all 28 layers per chunk, forces the re-reads.

The change runs **groups of K chunks per layer pass (owner)**: each layer processes the group's chunks in
order before the next layer starts. A layer's experts are then loaded about once per group.

**Success (owner): closer to the anticipated speed without losing quality.**
- Speed: the cold first token moves toward resident's. By arithmetic, at K = 8 and this laptop's about
  10.8 GB/s, it is about 34 s (4 passes, about 47 GB). A turn's append of two chunks becomes one group and
  reads about half as much.
- Quality: output identical to resident, logit for logit, the S4 bar.

## Goals

1. **Grouped prefill when streaming.** A streamed Mellum sync runs its prompt suffix in groups of K
   chunks. Within a group, layers run outer and chunks inner.
2. **K is a setting, default 8 (owner).** K = 8 adds about 75 MB (8 × 1,024 rows × 2,304 × 4 B), keeping
   streamed Metal allocation near 7.94 GiB, under the about 8 GiB budget. The expert cache stays at 979
   slots (owner). K = 1 must reproduce today's chunk-major order and loads exactly, so it doubles as the
   comparison arm.
3. **Identical output.** Grouped streamed prefill gives resident's logits word for word, for prompts
   spanning several groups and a partial last group.
4. **No resident change.** Resident Mellum keeps today's chunk-major path and code. The 2% A/B rule from
   S4 applies.
5. **Fewer loads, measured.** One preregistered measurement at the end, judged like S5 measurement 01.

## Non-goals

- Fewer host syncs. Each (layer, chunk) still syncs after its router, as in S4; only the loads drop.
- Cache-size or eviction-policy changes (S5 step 3), prefetch, lower-bit experts.
- Choosing K from a memory-budget formula. K is a plain setting; a later measurement may try others.
- The M1 Pro or 16 GB machines.

## Anti-goals

These count as failure even if everything else works:
- any logit that differs from resident;
- any resident slowdown or numeric change;
- memory that grows with prompt length rather than with K.

## How

**The loop.** For a sync suffix of n chunks (chunk size `cap_tokens`, today 1,024):

    for each group g of up to K consecutive chunks:
        embed each chunk's tokens into its slot of the group's hidden buffer
        for layer l in 0..27:
            for chunk c in g, in order:
                run layer l on chunk c at its own position, reading and writing c's hidden rows
        run the head on the group's last row
        the group is now a checkpoint

**Why it stays exact (to be proven, not assumed).** Each (layer, chunk) pair sees the same inputs as
today:
- its hidden rows come from the same layer below;
- a full-attention layer's KV for earlier positions is already written, because chunks run in order
  within the layer and earlier groups finished every layer;
- a sliding layer's ring holds the previous chunk's rows at that layer, written just before.

Chunk boundaries, row counts and kernels are unchanged. The full-attention rows table is written by the
CPU at encode time (`ds4.c:17619-17621`). Reusing it within a layer is safe only because every streamed
(layer, chunk) waits on the GPU after its router (`ds4.c:60663`), so that wait must stay.

**The group hidden buffer.** It holds K × `cap_tokens` × n_embd floats and is allocated only for
streamed Mellum sessions. It counts in Mellum's memory plan (`mellum_requests`, `ds4.c:17582`), so the
startup plan line and the session test's allocated-equals-planned check include it (designer's call).
Nothing in the engine enforces the about 8 GiB budget: K = 8 is justified by this laptop's measured
7.87 GiB plus 75 MB, and S5's measurements judge it. Per-chunk scratch (Q, K, the attention residual, FFN input, selected experts,
MoE output) is reused per (layer, chunk) as today.

**The MoE needs no change for chunks of two or more rows.** The S4 streamed prefill already syncs per
layer, loads the chunk's unique experts with the protect set, and re-checks residency. In a group, the first
chunk at a layer loads, and later chunks mostly hit.

**A one-row chunk stays out of the group (designer's call).** At T = 1, `mellum_graph_moe` takes the decode
branch (`ds4.c:60617`), which needs the stop/resume driver's gate and loads. Only a suffix's last chunk can
have one row. When it does, the group ends before it, and that row runs through today's
`mellum_stream_decode`, exactly as it does now.

**Checkpoints, cancellation and failure (designer's call).**
- A group is the checkpoint unit. Progress callbacks report per group.
- Cancellation is checked between groups only.
- A failure inside a group, such as a failed load, invalidates the session's checkpoint, and the next sync
  rebuilds. That is today's behavior for a failed chunk (`ds4.c:78428-78433`), so nothing changes. It is
  also necessary: the group's earlier layers have overwritten sliding rings past the last checkpoint.
- Only cancellation keeps a partial checkpoint today (`ds4.c:78417-78421`). Under groups, it keeps the last
  finished group.
- **Progress inside a group (designer's call).** The server's SSE keepalive fires only on progress callbacks
  (`ds4_server.c:12453-12485`, checked every 5 s), and a group of 8 takes about 8 s here and longer on
  slower machines. So the loop also emits `prefill_display` events, which are not checkpoints (`ds4.h:438`),
  after each layer pass.

## Left to the builder

- **Where the loop lives.** Probably beside `mellum_graph_forward_tokens`, called from the session sync's
  chunk loop (`ds4.c:78416`) for streamed Mellum. Every prefill entry point that can reach a streamed
  Mellum graph needs it or must refuse; audit the `qwen4_graph_forward_tokens` callers.
- **How chunk rows reach the layer:** separate per-chunk hidden tensors or offset views into one buffer.
- **The scratch audit.** Confirm, with a test rather than by reading, that no scratch state carries from
  layer l to layer l+1 of the same chunk except the hidden rows. `full_slot` is derived from the layer
  index; anything else stays to be checked.
- **The setting's form:** an engine option or an environment variable, consistent with
  `DS4_QWEN4_PREFILL_CHUNK`.
- **Test placement and names,** following the repository's tiers.

## Tests (designer's call)

- **Live identity.** Grouped streamed prefill against resident logits:
  - with small chunks (`DS4_QWEN4_PREFILL_CHUNK`) and K = 2 so a short prompt spans several groups and a
    partial last group;
  - at 6 GiB and at 64 slots;
  - and a warm append across a group boundary.
- **K = 1 equals today:** identical logits and an identical load count to the current build.
- **Fewer loads:** for the same prompt at 6 GiB, K > 1 loads fewer experts than K = 1.
- **Failure:** a new prefill fault test. A load forced to fail mid-group fails the sync, and the next
  sync rebuilds and matches resident. S4's faults test fails loads only on the decode path, so it is
  unchanged.
- **One-row tail:** a suffix whose last chunk has one row matches resident.
- **Cancellation:** between groups, the checkpoint is the last finished group.
- **Every new test is shown to fail first,** under the SDD rules.

## The measurement (designer's call)

The same frozen prompt and appended text as S5 measurement 01, a new preregistration, and these arms:
- R, resident, n=1;
- C, streamed at 6 GiB with K = 8, n=3, each from a fresh copy with device counters.

The A/B arms rerun only if the build touches resident code paths.
- **Reported:** cold first token, the append's first token, loads in prefill and in the append, decode t/s
  and memory peaks.
- **Comparisons:** against R and against S5 measurement 01's C.
- **The prediction to test:** cold first token near 34 s, and prefill loads at most 7,168 (4 passes ×
  1,792). That is a ceiling, not a point: eviction is frequency-based (`prune_global`,
  `ds4_metal.m:16089`), so some experts may survive between groups. A count above the ceiling would
  contradict the design.

## Constraints

- Superpowers SDD: a plan next, separate RED and GREEN commits, about 150 product lines per commit, one
  concern per commit.
- Respect `/tmp/ds4.lock`. Each live or GPU window and the measurement need owner authorization.
- No change to weights, routing, Q8_0 arithmetic or expert reduction order.

## Done when

- Goals 1–4 are proven by the tests above, with live runs owner-authorized.
- The measurement is recorded with its n and limits, and the roadmap's S5 step 2 is updated.
- An independent review has run.
