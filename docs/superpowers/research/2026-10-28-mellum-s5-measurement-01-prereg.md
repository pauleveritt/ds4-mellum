# S5 measurement 01 (32K at the current build, plus a per-turn append): preregistration

Date: 2026-10-28, committed before any timing. Branch `codex/mellum-ssd-replay`. S5 step 1 in the
[roadmap](../../../ROADMAP.md).
- **Why:** [S4 measurement 02](2026-10-28-mellum-s4-measurement-02.md) is spent and predates `9e9757f`,
  which suppresses the Q8_0 projections and f16 copies after a streamed stop. That change raised
  short-context streamed decode from 30.6 to 39.6 t/s (n=1,
  [decode profile](2026-10-28-mellum-s4-decode-profile-spike.md)). No 32K figure exists for it.
- **New case:** the time to first token after a turn appends about 2K tokens to a warm session. Nothing
  has measured it.

**Honesty header.**
- **Measured:** this 128 GiB M5 Max's SSD and compute only, one frozen prompt and one frozen appended
  text, n as stated per arm.
- **One run:** no rerun or retry follows any outcome.
- **Not claimed:** speed on any other machine. S5's targets are judged on the owner's M1 Pro; this run
  reports against them for orientation only.

## Prerequisite: the driver's append phase

`tools/mellum_s4_measure` gains `--append FILE --append-gen N`, committed after this file and before any
timing, with a RED test first. After the 2,048 decode tokens it:
1. pushes the generated ids onto the prompt's token list;
2. appends the file's text as one tool result (`ds4_chat_append_tool_results`, which Mellum renders as a
   user turn with a `<tool_response>` block), then the assistant prefix with thinking on;
3. times `ds4_session_sync` over that list, up to its logits, as `append_ttft_ns`;
4. decodes N greedy tokens, recorded like the first 2,048.

The output adds `append_tokens` (the suffix length), `ds4_session_pos` before and after the sync, and the
stops and loads during it. The generated text is used as it stands, usually mid-thinking, not as a
finished turn: the case times the appended prefill, not the model's behavior on that turn.

The measured build is the commit that completes the append phase. It must not change `ds4.c`,
`ds4_metal.m` or `metal/` (`git diff 8ecb058..<build>` of those paths is empty).

## Frozen inputs

- **Model:** `~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf` (Q8_0, 12,925,417,760 B).
- **Prompt:** 01's [`prompt.txt`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-28-mellum-s4-measurement-prereg/prompt.txt), SHA-256
  `2164763c…6344da`, 30,620 tokens rendered.
- **Appended text:** [`append.txt`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-28-mellum-s5-measurement-01-prereg/append.txt), a copy of
  `ds4_layer_pack.c` at `36ee8c1`.
  - SHA-256 `ed93c59f8f7579e03ee74ffda8e4677653d408b6f2fc53db54270825909f23ce`, 5,975 B.
  - The file is not in the prompt.
  - Its rendered length is recorded by each run as `append_tokens`, not estimated here.
- **Context:** 36,864, raised from 02's 32,768 to hold the append. The positions decoded match 02's: 30,620
  to 32,668. The larger context adds about 59 MB of KV (4,096 × 14,336 B) to the memory peaks.
- **Generation:** 2,048 greedy tokens, then the append, then `--append-gen 256`.
- **Streamed cache:** `--stream-bytes 6442450944` (979 slots).

## Arms

| Arm | Build | Tool | Runs |
|---|---|---|---|
| A | pre-S4 baseline `b1994c6`, in a detached worktree | `ds4` CLI: `-c 8192 --temp 0 -n 256 -p "Explain how a hash map works, with an example in C."` | 6 |
| B | measured build, streaming off | the same CLI command | 6 |
| R | measured build, resident | the driver with `--append` | 1 |
| C | measured build, streamed at 6 GiB | the driver with `--append` and `--stream-bytes`, each from a fresh write-only `F_NOCACHE` copy | 3 |

- **A/B returns** because `9e9757f` routed resident dispatch sites through `ds4_gpu_dispatch_gated`. That
  call is plain when no gate is bound, but it is resident-path code changed since 01's A/B.
- **Order:** A and B in A/B/B/A blocks (`regression_order(6)`) with 30 s rests, then R, then C, with 30 s
  rests between processes.
- **Each C process:** bracketed by block-driver read counters; its copy is verified against the model
  after the run (`verify_copy`) and then deleted.
- **The shared model lock:** every run respects it. A held lock before the first run postpones the whole
  measurement; nothing is created.
- **Raw outputs:** `~/.codex/visualizations/2026/10/28/mellum-s5-measure-01`, an exclusive
  root.

## Decisions fixed now

- **No regression:** B's median prefill t/s and median generation t/s are each at least 0.98 of A's
  (`regression_verdict`, one-sided).
- **Identity:** every C run's 2,048 ids and 256 post-append ids equal R's.
- **Suffix reuse:**
  - The append sync must start at `ds4_session_pos` = prompt + 2,048 and end at that + `append_tokens`.
  - R's `append_ttft_ns` must be under a tenth of R's cold time to first token. A rebuild of about 32.7K
    tokens could not be.
  - Failing either makes the per-turn case invalid for that run, as a recorded failure.
- **Read class per C run:** device-observed when device bytes are at least 0.99 × (loads × 6,580,224);
  otherwise cache-contaminated. Over 600 GB of expert bytes is a recorded failure. 02 loaded 480 GB, and
  the append and 256 tokens are estimated to add under 50 GB.
- **Reported, not gated:**
  - cold time to first token;
  - decode t/s over all 2,048 tokens, the first 256 and the last 256;
  - `append_tokens`, the append time to first token, and the append's loads;
  - post-append decode t/s over 256 tokens;
  - stops and loads per decode token;
  - logical and device bytes;
  - peak `phys_footprint`, RSS and Metal allocation, against the about 8 GiB budget.

  C is shown as a range and a median, against R, against 02's C (decode 26.1–27.2 t/s, first token
  58.6–60.9 s), and against S5's targets (15 t/s decode; 60 s cold and 10 s per turn to first token) for
  orientation only.
- **Speed claims** come only from runs classed device-observed, and only for this SSD.

## What would change the plan

Nothing in the run. Any result outside these rules is reported as it is. A second measurement needs its own
preregistration.
