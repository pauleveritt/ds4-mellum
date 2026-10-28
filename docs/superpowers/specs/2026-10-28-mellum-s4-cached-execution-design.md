# S4 Mellum cached expert execution and miss recovery

Date: 2026-10-28. Branch `codex/mellum-ssd-replay`, baseline `63d7ee9`.
Owner decisions are marked **(owner)**. Everything else is the designer's call
and may be revisited in planning.

## Intent

Mellum 2.1 is 28 layers × 64 routed Q8_0 experts, top-8, no shared expert,
6,580,224 bytes per expert. It runs fully resident today at about 12.4 GiB.
The first streaming target is **a 16 GB laptop with 32K context, fitting
comfortably (owner)**. The budget **(owner, accepting the designer's
estimate)** leaves about 8 GiB for macOS, an editor, a browser and a couple of
apps. That means a Mellum process at or under about 8 GiB peak, and a
**6 GiB expert cache** at 32K. The arithmetic:

- 6 GiB of cache;
- 1.06 GiB of non-expert tensors;
- about 0.73 GiB of session at 32K: 314.8 MB plus 14,336 B/token, from MELLUM.md's three measured plans;
- about 0.3 GiB of buffers and staging.

S4 ends when **performance is measured with those cache settings (owner)**,
on this 128 GiB M5 Max. Underneath that it must prove that cached execution
is correct.

Evidence behind the choices:

- [retarget](../research/2026-10-28-mellum-ssd-retarget.md): miss curve,
  16 GB paper fit;
- [layer-sync spike](../research/2026-10-28-mellum-layer-sync-spike.md):
  a host sync costs about 164 µs, n=4 per arm;
- the -02 trace (descriptive, n=1), at 6 GiB: about 12.6 misses per token,
  9.4 layers per token with a miss, 1.3% of tokens all-hit.

## Goals

1. **Opt-in streaming for Mellum (owner).** `--ssd-streaming` with
   `--ssd-streaming-cache-experts 6GB` runs Mellum from a bounded slot cache.
   Without the flag, Mellum runs exactly as today. For Mellum, `6GB` means
   6 GiB of expert slots, 979 experts of 6,580,224 B (designer's call). Prefill
   loads into those same slots, so upstream's carve-out of two prefill layers
   from NGB is bypassed. That keeps the miss rate comparable with the 6 GiB
   replay.
2. **No regression with streaming off (owner).** Resident decode and prefill
   speed stay within noise of the pre-S4 baseline. The preregistered test
   (designer's call):
   - six processes per arm, in A/B/B/A order with 30 s rests;
   - B's median decode t/s and median prefill t/s each within 2% of A's;
   - startup time reported, not gated.

   Resident *behavior* must not change: byte-identical logits and greedy
   output, plus this speed test. Resident Metal source may be refactored, for
   example templated on how the expert base is obtained, if the resident
   instantiation keeps the same arithmetic (designer's call). Streaming code
   paths, including any guarded non-MoE kernels, run only when streaming is
   on.
3. **Byte-identical results.** For the same prompt and settings, streamed
   greedy output and logits equal resident output. Weights, routing, Q8_0
   arithmetic and expert reduction order are unchanged.
4. **Decode syncs only at layers that miss (owner).** Syncing every layer
   would cost about 4.4 ms per token. On an all-hit token, the host waits no
   more than it does resident.
5. **Safe miss recovery** under forced misses, eviction, cancellation and read
   failure: no wrong output, no use of a slot being refilled, and a clean
   error rather than a hang.
6. **The 32K measurement** (below) at a 6 GiB cache, with streamed reads
   coming from the device, not macOS's page cache. Peak memory is *reported*
   in S4, not gated (designer's call):
   - `phys_footprint`, the primary metric;
   - Metal allocated size;
   - RSS.

   S5 gates the about 8 GiB budget.

## Non-goals

- Layer-major prefill, which reads each layer once per prompt rather than per
  chunk. It is recorded as an S5 hypothesis.
- Prefetch or routing prediction, per-layer cache splits, hotlists tuned on
  any trace, and lower-bit experts.
- Validation on real 16 GB or 8 GB hardware, which is S3b, S5 and S6.
- CUDA/ROCm, `ds4-server`/`ds4-agent` streaming, and `../ds4-engine`
  integration.
- Changing upstream DeepSeek/GLM behavior.

## Anti-goals

These count as failure even if everything else works:

- any measurable resident slowdown, or any resident numeric change;
- streamed output that differs from resident;
- choosing a cache size or policy from the held-out -02 trace or from ac-demo;
- any claim of 16 GB or 8 GB feasibility, or of streamed speed on other
  hardware.

## How: reuse upstream, add a Mellum path

Upstream's Metal streaming for DeepSeek and GLM supplies the cache machinery.
Reuse it as it is, or adapt it where noted:

- the budget and flags in `ds4_ssd.c`;
- 4 GiB shared-storage slabs with per-slot `mlock` (`ds4_metal.m:1017-1033`,
  `13872-13925`);
- the `[layer][expert]` entry table with in-flight sequence ownership
  (`ds4_metal.m:964-986`). Its limits cover 28 × 64 × 8.
- the per-layer `uint64` GPU-address tables (`ds4_metal.m:14676-14783`) and
  `useResource` residency;
- the `pread` thread pool and `F_RDADVISE` (`ds4_metal.m:13232-13420`);
- global hotness and LRU eviction (`ds4_metal.m:15328-15458`);
- the model-free eviction-oracle test pattern in
  `tests/test_metal_ssd_experts.c`.

Gates hard-coded to upstream's models must admit Mellum: `n_expert == 6`,
`n_total_expert >= 128`, and the quant-type lists. No Mellum hotlist exists;
the cache starts empty and fills on demand.

New for Mellum:

1. **Q8_0 address-table kernels.** Mellum's kernels compute `base + selected
   × expert_bytes` into one model view (`metal/qwen4.metal:2793`, also 3072,
   3127; tiled and E1 at 3682-3683, 3805, 3985-3986, 4142). The streaming
   variants take each expert's base from the layer's address table, one
   address per expert for gate, up and down. The arithmetic and the
   per-threadgroup order stay the same. Every variant gets its own pipeline:
   decode mid/down, tiled mid/down, and E1's tensor-tile paths.
2. **A streaming dispatch in `mellum_graph_moe`** (`ds4.c:60521-60581`)
   selected at session open, not per kernel.
3. **The decode miss protocol: stop at the first miss, resume at its MoE.**
   - A token's 28 layers stay in one command buffer.
   - After a layer's router and top-8, a check finds whether all eight
     addresses are present. On the first miss, it records that layer and
     suppresses all later work in the token.
   - At the buffer's end the host reads the marker. It loads that layer's
     missing experts, waits for those reads, updates the table and encodes the
     rest of the token. It resumes from that layer's expert compute, not the
     start of the layer.
   - Attention and that layer's KV write ran once and must not run again.
   - Routing for the resumed layer must equal the original, by reusing the
     saved top-8 or recomputing it identically.
4. **Prefill: sync per layer after the router.** A 1,024-row chunk touches
   nearly all 64 experts of a layer, about 421 MB. Before each layer's expert
   compute, the host waits, reads the chunk's selected set, and makes every
   needed expert resident. It then runs the E1 or tiled address variant. A
   host wait per layer per chunk is negligible next to the reads. Chunks
   under 32 rows keep the exact path, which also needs its address variant.
5. **Device-true reads.** Streamed reads use an `F_NOCACHE` descriptor for the
   model file, with no `F_RDADVISE` read-ahead on that path. Upstream opens the
   model without it; the only `F_NOCACHE` is the n-gram table at
   `ds4.c:2902`. `F_NOCACHE` does not evict pages that are already cached, and
   the resident arms read the whole file. So each streaming measurement process
   reads from a fresh copy of the GGUF, written with `F_NOCACHE` just before it
   and checked by size and a byte comparison (designer's call). This
   follows the storage-service evidence's fresh-file method and needs no
   `sudo purge`. Experts are not 4 KiB-aligned in the GGUF, and edge pages
   may be served from cache. The device/logical ratio is therefore reported
   per run.

## Left to the builder

- **How "suppress later work" is done in Metal.** Candidates:
  - indirect dispatch with threadgroup counts the check zeroes;
  - a guard in each streaming pipeline;
  - splitting the buffer at MoE boundaries, only if it adds no per-layer
    host wait on all-hit tokens.

  Settle this with a model-free proof first: an all-hit token must not add
  host waits, and resident pipelines must not change.
- Slot layout (combined or per-tensor), table format, thread-pool size and
  hotness parameters.
- Exact cancellation and read-failure semantics, within goal 5.
- Test names and placement, following the repository's tiers: model-free,
  engine, live.

## Test tiers (designer's call)

- **Model-free and engine tiers** prove the protocol and kernels:
  - kernel-level equality between resident and address-table variants, on
    synthetic, distinct expert buffers (`test-mellum-kernels` style);
  - the stop/resume state machine;
  - eviction, ownership, cancellation and read failure, with fakes.

  The sparse `tests/mellum_fixture.h`, whose experts all alias offset 0,
  cannot catch a wrong address, so it is not used for that.
- **Live tier, each run owner-authorized,** proves whole-model byte identity
  and stop/resume on real weights. It compares resident and streamed logits
  and greedy output, including a tiny forced-miss cache, such as 64 slots.
  No new full-size weights fixture is built.

## Constraints

- Superpowers SDD: plan next, separate RED and GREEN commits, about 150
  product lines per commit, one concern per commit. Every new test is shown
  to fail when the behavior it pins is broken.
- Respect the shared `/tmp/ds4.lock`. Never bypass it, and never kill another
  owner.
- Each live or GPU run needs owner authorization and a stated time window.
- One-shot measurements are preregistered and not retried.
- Model-free tests need no model, GPU or subprocess. Engine and live tests are
  opt-in.

## The measurement

**Workload (designer's call).** A synthetic prompt of about 30K tokens is built
deterministically from this repository's own C and Markdown sources, never from
ac-demo. It is followed by greedy generation of about 2K tokens, so the run
reaches about 32K context. The prompt bytes and token IDs are frozen before any
timing.

**Read volume (designer's call).** A 1,024-row chunk touches nearly all 1,792
experts, about 11.8 GB, which is more than the 979 slots. So each chunk
re-reads almost everything: about 350 GB of logical reads per streaming
process, and roughly 35 s of prefill on this SSD. That is accepted and
reported, not trimmed.

Rules:
- A per-process bound of 400 GB of logical reads is preregistered; exceeding
  it ends the run as a recorded failure.
- Prefill loads update eviction recency normally, so decode starts from the
  cache prefill leaves.
- Fresh GGUF copies add about 13 GB of writes per streaming process.

**Classification (designer's call).** Each streaming run is classified by its
device/logical read ratio:
- 99% or more: device-observed;
- below 99%: reported as cache-contaminated.

Neither outcome is retried.

**Arms (designer's call),** run in rotated order with rests, three processes
per arm for C and six for A and B (goal 2):

- A: the pre-S4 baseline build, resident;
- B: the S4 build with streaming off;
- C: the S4 build with streaming on at 6 GiB, starting cold.

**Correctness:** C's output must be byte-identical to B's, and to A's.

**Report, with n:**

- time to first token;
- decode tok/s at early and late context;
- misses, stops and bytes read per token;
- device read bytes against logical bytes, reusing the storage-service
  counter method;
- peak process memory against the about 8 GiB budget;
- B against A, for goal 2.

**Limits:** this M5 Max's SSD and compute only. Nothing here establishes 16 GB
behavior.

## Done when

- Goals 1–5 are proven by model-free and engine tests, with owner-authorized
  live runs for whole-model identity.
- The 32K measurement is recorded as research with its n and limits.
- The roadmap's S4 status is updated.
- An independent review has run.
