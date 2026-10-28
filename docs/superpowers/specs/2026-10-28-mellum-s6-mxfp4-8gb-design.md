# S6 Mellum on an 8 GB Mac with MXFP4 experts

Date: 2026-10-28. Branch `research/mellum-8gb-q4`, based on
`codex/mellum-ssd-replay` at `3af0a3f`. Owner decisions are marked
**(owner)**. Everything else is the designer's call and may be revisited in
planning.

## Intent

The user is on an 8 GB Apple Silicon Mac with a browser and an editor open.
They do routine, small-scale Python work through a Pi coding agent: edits
across a few files, a step up from autocomplete, and some lightweight worker
tasks that an SDD roadmap has already shaped **(owner)**. Their expectations
are low. They accept a smaller model because it is fast, and their reference
point is a large local model such as Qwen3.8 27B, which is slow for them
**(owner)**.

The rest of the experience comes from levers outside this phase.
`../mellum-worker` explores them: a `/mellum` extension, a smaller context, a
smaller worker system prompt, a low thinking setting **(owner)**. This phase
makes the engine able to run Mellum on 8 GB at all, at a speed those levers
can build on.

The route is lower-bit streamed experts **(owner, 2026-10-28)**: the S6
reopen condition in [ROADMAP.md](../../../ROADMAP.md). Q8 cannot fit; the
[retarget](../research/2026-10-28-mellum-ssd-retarget.md) put an 8 GB
expert cache at 2–2.5 GiB, where Q8 misses cost 372–651 MB per token.

### The artifact

JetBrains' `Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf`, 7,029,537,056 B,
Hugging Face snapshot `4159d684` **(owner)**. Its tensor table, read from
the header:

- every non-expert tensor is Q8_0 or F32, the same tensors as the Q8_0 file
  (1,133,656,352 B in both files);
- all 84 routed-expert tensors are MXFP4 (GGUF type 39, 32-wide blocks of 17
  B): 3,290,112 B per expert, exactly half of Q8_0's 6,580,224.

JetBrains reports KLD 0.115 and 85.6% top-token match against BF16
(Q8_0: 0.008, 96.1%; Wikitext-2, `n_ctx=512`). Their figures, not ours.

The [code read](../research/2026-10-28-mellum-mxfp4-metal-path.md) found
that Mellum's routed experts already run on the Qwen4 MoE kernels, whose row
dot branches on type 39 (`metal/qwen4.metal:2614`). The Q8_0 restriction
is in host-side gates: about 8 resident, about 7 streamed, and the constant
`DS4_MELLUM_EXPERT_BYTES`. No production model sends MXFP4 gate/up rows
through those kernels yet.

## Goals

1. **Resident MXFP4.** The engine loads the MXFP4 file and runs it fully
   resident on the M5 Max. The file chooses the expert type; one binary runs
   both files (designer's call). Expert tensors must be all Q8_0 or all MXFP4
   with one type across the 84; anything else is refused at open, naming the
   tensor.
2. **Not broken (owner).** "Reliable" means the logits are not broken, not
   that answers are good. Evidence, in order:
   - a model-free test of the MXFP4 expert path (gate, up, SwiGLU, down,
     weighted sum) against a double-precision CPU reference, for decode and
     prefill shapes;
   - our MXFP4 logits against llama.cpp's on the same file (Homebrew
     `llama.cpp` 0.7.0 is installed), on fixed prompts: top-1 agreement and
     KLD, thresholds preregistered before the run;
   - a KLD read of our MXFP4 against our Q8_0 on the same prompts, reported
     beside JetBrains' 0.115, not gated;
   - finite logits throughout, and no repetition collapse in 2,048 greedy
     tokens on the S4 prompts.

   Kernel selection depends on chip and OS: Metal 4 only on macOS 26 or later
   (`ds4_metal.m:2888-2893`), and M5-only fast paths. So the model-free
   kernel tests also run on the M5 Max with `DS4_METAL_DISABLE_METAL4=1`, and
   again on the iMac (designer's call).
3. **Streamed MXFP4 matches resident MXFP4.** With `--ssd-streaming`, the
   same prompt and settings give the same token ids as resident MXFP4, as S4
   required for Q8. The slot count comes from the file's expert size, not
   from a Q8_0 constant.
4. **The 8 GB footprint, measured on an 8 GB Mac (owner).** The owner has
   an 8 GB Apple Silicon iMac. On it, with a browser and an editor open,
   streamed MXFP4 runs without swapping that slows decode, inside the measured
   Metal working-set limit. It needs no `iogpu.wired_limit_mb` change.
   The background load is fixed and scripted: PyCharm with a small Python
   project open, and a browser with ten tabs (designer's call). Pass means
   no growth in swap-outs (`vm_stat`) during the timed decode window, and
   decode t/s within a preregistered margin of the same run with the apps
   closed (designer's call).
5. **Tolerable speed, reported (owner).** Decode t/s and time to first token
   on the iMac, each with its `n`. No speed target; the report is the exit.
   The workload is Pi-shaped: a captured mellum-worker first request (system
   prompt, tools, a few files), not S4's 30K synthetic prompt, which would
   re-read nearly every expert per prefill chunk (designer's call).
6. **The bigger-Mac measurement (owner: measured, not promised).** On the
   M5 Max, resident Q8_0 against resident MXFP4: decode t/s, time to first
   token, and energy if `powermetrics` can read it, each with its `n`. It
   feeds a later product decision about MXFP4 on 16 GB and larger Macs.
   S6 makes no claim about them.

## Non-goals

- Answer quality, tool-call success, agent competence.
- Q4_K_M, IQ4, Q4_0 or any other quantization. JetBrains' Q4_K_M needs a
  Q5_0 down kernel and Q4_K/Q6_K attention projections Mellum lacks.
- Changing the 16 GB plan (S3b–S5 keep Q8), or making MXFP4 a default
  anywhere.
- The mellum-worker levers themselves; native host integration in
  `../ds4-engine`.
- New cache policy. S6 uses the S4/S5 policy at a smaller cache.

## Anti-goals

- Any change to Q8_0 behavior. Resident and streamed Q8_0 stay
  byte-identical in logits and greedy output, and resident speed stays within
  noise, by S4's preregistered A/B test.
- A footprint result taken only on the 128 GB laptop and presented as 8 GB.
- A speed or quality figure without its `n`, or JetBrains' numbers reported as
  ours.

## Constraints

- **No GPU until the SSD work merges (owner).** Before that, work is CPU-only
  and model-free. This includes no Metal tests and no llama.cpp GPU runs.
  The SSD session's device-true read measurements also run on this laptop's
  SSD, so large reads or writes wait for them as well.
- `ds4-mellum`'s working rules: one concern per commit, a failing behavioral
  test committed before its fix, about 150 product lines per commit,
  Superpowers design and plan per milestone.
- Evidence notes open with an honesty header and end with "What would falsify
  this" and "Risks". Screens may run freely; a decision gets a
  preregistration.

## The 8 GB memory plan, on paper

The S4 accounting with MXFP4 slots. An MXFP4 slot is the expert rounded up to
a 16 KiB page: 3,293,184 B (Q8_0: 6,586,368 B).

| Item | 2 GiB cache | 2.5 GiB cache | 3 GiB cache |
|---|---:|---:|---:|
| Slots | 652 | 815 | 978 |
| Non-expert tensors | 1.06 GiB | 1.06 GiB | 1.06 GiB |
| Session at 32K | 0.73 GiB | 0.73 GiB | 0.73 GiB |
| Buffers and staging (S4 estimate) | 0.3 GiB | 0.3 GiB | 0.3 GiB |
| **Process total** | **4.09 GiB** | **4.59 GiB** | **5.09 GiB** |
| LRU demand, -02 trace (approx.) | ~114 MB/token | ~69 MB/token | ~42 MB/token |

The session is 314.8 MB plus 14,336 B per token (MELLUM.md): 0.51 GiB at
16K, 1.04 GiB at 56K, mellum-worker's measured profile. Demand is the
retarget curve at the Q8 cache with the same slot count, times 3,290,112 B.
It is descriptive: n=1, a thinking conversation, routing assumed unchanged
by quantization.

The cache size is chosen on the iMac, from its measured Metal limit and
the room left with other apps open (designer's call). Planning starts at
2.5 GiB and 32K.

## Order

**A. Before the merge, CPU-only.**
1. The expert size as a parameter in `tools/mellum_ssd_replay.py` and the
   `mellum_read_*` tools, then the miss curve re-run at 3,290,112 B (a
   replay, no model).
2. An MXFP4 dequantizer in the double CPU reference
   (`tests/mellum_moe_ref.inc`), tested against hand-built blocks.
3. Host-side gates and the slot count derived from the file. These are
   written and covered by model-free CPU tests where those exist. Anything
   needing Metal waits.
4. A preregistration for B's correctness and speed runs.

**B. After the merge, on the M5 Max.** Rebase onto the merged branch. Then:
Metal kernel tests; resident MXFP4 with the Goal 2 evidence; the Q8_0
no-regression A/B; Goal 6; streamed MXFP4 against resident (Goal 3).

B starts when `codex/mellum-ssd-replay` (S4) has merged into `mellum` and
the GPU is free. S6 does not wait for S5, and builds on S4's streamed code
(designer's call).

**C. On the iMac.** The target is an **M1 iMac, 8 GB, 1 TB SSD (owner)**.
`docs/METAL.md` claims only M3 and M5 Macs, and `ds4_metal.m` has no Apple
GPU-family gate beyond Metal 4 (`:2893`). So M1 is untested, not refused.
C opens with a bring-up: build, model-free kernel tests, and a short streamed
MXFP4 run. If the M1 needs a fallback kernel path, that is engineering inside
S6, planned when it is found. Fully resident MXFP4 (about 6.5 GiB of
weights) does not fit there, so the iMac's correctness check is our streamed
MXFP4 logits against the M5 Max's resident MXFP4 logits on the same prompts,
by KLD and top-1 agreement within a preregistered bound. It is not token
identity, because the kernel paths differ by chip (designer's call). Then:
the Metal working-set limit; a miss-order SSD read bench at 3,290,112 B; and
streamed MXFP4 with other apps open (Goals 4 and 5). The owner runs the
iMac by hand from a written runbook (designer's call).

## Left to the builder

- Where the type check lives and how the expert type reaches the kernels,
  as long as it is chosen per file.
- The exact prompts and token counts for Goal 2, within S4's set.
- How the iMac runs are driven and captured, as long as each figure is
  recomputable from its capture.

## Open

- Whether the M1 reports `MTLGPUFamilyMetal4`. It runs macOS 27.1
  **(owner)**, so the Metal 4 runtime is present; the family is queried at
  run time, and bring-up records it.
- The context for the 8 GB plan: 32K here (designer's call);
  mellum-worker's measured profile is 56K, and its levers include a smaller
  context.

## What would falsify this

- The MXFP4 gate/up path disagrees with the CPU reference, or our logits
  disagree with llama.cpp's on the same file beyond the preregistered bound.
- The iMac's Metal limit leaves under about 2 GiB for the cache at the
  plan's context.
- Decode on the iMac at the chosen cache is no faster than the 27B on the
  owner's 36 GB M4 Max, the slowest machine that runs the 27B today, measured
  with `ds4-engine` (designer's call: the 27B cannot run on 8 GB).
- MXFP4 routing diverges from Q8_0 so much that the replay's demand estimate
  is off by more than 2×.

## Risks

- One thinking trace sets every demand figure; non-thinking locality is
  unmeasured.
- The Metal limit on 8 GB is hearsay until measured.
- MXFP4 gate/up through the Qwen4 kernels is new in production.
- The engine has never run on an M1. The target machine may need kernel
  work this design cannot size yet.
- The SSD branch is still moving; rebasing after the merge may surface
  conflicts in the streamed entry points this phase edits.
- An 8 GB machine under memory pressure compresses and swaps. A single clean
  run will not show that.
