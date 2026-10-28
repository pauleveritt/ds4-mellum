# S5 step 2: layer-major prefill for streamed Mellum, options from the code

Date: 2026-10-28. Desk study for S5 step 2 in the [roadmap](../../../ROADMAP.md), read from `ds4.c` at
`4d0349c`. Input to a brainstorm, not a design: the owner chooses.

**Honesty header.**
- **Read from the code:** where streamed prefill re-reads experts, and what the session's chunk loop
  promises callers.
- **Measured elsewhere:** 334 GB of prefill loads and about 31 s over resident's time to first token, in
  [measurement 02](2026-10-28-mellum-s4-measurement-02.md) (n=3).
- **Arithmetic, not measured:** every saving, memory figure and exactness claim below. No layer-major
  Mellum code exists.

## Where the reads come from

- **The chunk loop.** The session sync splits a prompt into chunks of `cap_tokens` rows: 1,024, the sliding
  window, by `mellum_prefill_cap` (`ds4.c:17568`). It runs all 28 layers per chunk
  (`ds4_session_sync`'s loop, `ds4.c:78416`; `mellum_graph_forward_tokens`, `ds4.c:61015`).
- **The streamed MoE.** Each layer syncs after its router, then loads that chunk's unique experts. A
  1,024-row chunk touches nearly all 64 of a layer's experts, 1,792 in all, about 11.8 GB, against the
  6 GiB cache's 979 slots. So every chunk evicts what the next chunk needs. 30 chunks gave 50,806 loads.
- **A layer's working set fits.** 64 experts are 421 MB, so one layer's union fits the cache many times
  over. Only the loop order forces the re-reads.

## The change: layer outer, chunk inner

For each layer, run every chunk of the prompt through it in order, then move to the next layer. The
layer's experts are loaded about once per prompt, not once per chunk.

- **Reads:** at most 28 × 421 MB = 11.8 GB per prompt, against 334 GB. At 02's about 10.8 GB/s, that is
  about 1 s instead of about 31 s, so streamed time to first token falls near resident's 29 s on this
  laptop.
- **Same arithmetic, so exact by construction.** The chunk boundaries, row counts, kernels and per-layer
  rings stay the same. A layer's KV and ring are still written in chunk order, so chunk *c* at layer *l*
  sees exactly the inputs it sees today. Only the order in which (layer, chunk) pairs run changes. This
  must be proven by the existing identity tests, not assumed.
- **New memory: the residual stream for the whole prompt.** Today `MELLUM_S_HIDDEN` holds one chunk
  (`ds4.c:17603`). Layer-major needs every row's hidden state between layers: n_embd 2,304 × 4 B = 9.2 KB
  per token, 282 MB at 30,620 tokens. Metal allocation would rise from 7.81 GiB to about 8.1 GiB, past the
  about 8 GiB budget, unless the cache gives up about 43 slots.

## Options

| Option | Reads per 30K prompt | Extra memory | Notes |
|---|---:|---:|---|
| A. Whole prompt per layer pass | ≤ 11.8 GB | 9.2 KB/token (282 MB at 30K) | simplest order; memory grows with the prompt |
| B. Groups of K chunks per pass | ≤ ⌈30/K⌉ × 11.8 GB | K × 9.4 MB | K = 8: 75 MB and about 47 GB; bounded memory |
| C. A, with the hidden stream in f16 | ≤ 11.8 GB | 141 MB | changes the residual's precision; breaks exactness, so ruled out |

**Recommendation: B, with K chosen from the memory budget.** It caps the new memory independent of prompt
length, and at K = 8 it still removes about 86% of the reads. A is B with K = all chunks, so one
implementation covers both, and K can be a session setting.

## What the session's chunk loop promises, and what changes

- **Per-chunk checkpoints.** Today every finished chunk is a valid checkpoint with its own logits
  (`ds4.c:78427-78437`), for progress callbacks and cancellation. Under layer-major, no chunk is finished
  until the last layer. A group becomes the checkpoint unit: cancelling mid-group drops back to the last
  finished group. `ds4.h` already says this of upstream's layer-major prefill: "a partial layer-major pass is
  invalidated and the next sync rebuilds it."
- **Logits per chunk.** Only the group's last row needs the head; the per-chunk head runs go away.
- **Streaming only.** Resident Mellum keeps the chunk-major path, so the no-regression rule is untouched.

## Also helps the per-turn append

A 2K-token append is two chunks, so it reads each layer's experts twice today. As one group, it reads them
once: about 11.8 GB less per turn at the cost of 19 MB of extra memory.

## What would falsify this

- **The prefill time doesn't fall near resident's.** That would mean the reads were not what cost the 31 s,
  for example because of per-layer syncs or load latency. Measure loads and time to first token before and
  after.
- **The identity tests show different logits.** That would mean some state depends on the (layer, chunk)
  order, such as a ring or position written outside the per-layer path.

## Risks

- `mellum_graph_layer` and its attention may read state that assumes the previous layer of the same chunk
  just ran, such as shared scratch reused across layers. Each scratch tensor needs checking.
- Upstream's DeepSeek layer-major prefill (`metal_graph_prefill_layer_major`, `ds4.c:38556`) is a reference
  for the loop and its cancellation, not code to reuse: different model, attention and caches.
- The cache's eviction state after a layer-major prefill differs: it ends holding the last layers'
  experts, not the last chunk's working set. That may change decode's first tokens; it is a performance
  effect only.
