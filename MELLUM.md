# Mellum 2.1 on ds4

Branch `mellum` ports JetBrains' Mellum 2.1 (12B-A2.5B, thinking) onto
antirez/ds4, starting at `5379c1c`, in small commits. This file records what
works, what was measured (each number with its command and n), what diverges
from the model's training, and what is not covered.

## Status

The separate SSD investigation is tracked in [ROADMAP.md](ROADMAP.md).

Phases 1 (recognize and load), 2 (text) and 3a up to the oracle (steps 15-20)
are done: a `mellum` GGUF is detected, validated, bound and planned; the
`mellum2` pre-tokenizer, the stop classes, the chat prefix and the sampling
defaults match the model's own tokenizer and template; the Metal forward pass
gives logits byte-identical to the old port's on three prompts in prefill and
decode, and 64 equal greedy tokens (Checkpoint 2). Since step 22 the `ds4` CLI
runs Mellum chats on Metal.

**Prefill no longer equals decode exactly** (steps 25 and 27): a prefill chunk
of 32 rows or more runs its experts as tiled GEMMs and its Q8_0 projections as
batched GEMMs, both on half-staged activations; together about 6x faster than
the exact path (2,200 tokens: about 2,000 t/s against 350). Decode, and any
chunk under 32 rows, keep the exact arithmetic. Over 64 positions of each long
test prompt, max |d logit| against the exact path is 2.0 (1,100 tokens) and 8.5
(2,200), mean KL divergence 2.1e-3 and 1.7e-3, largest probability change 0.06,
with the same top-1 token at every position (the experts alone: 0.65 and 0.87).
On held-out prompts the top-1 token is not always the same. Router top-8 flips at near-ties change it
at 4 of 64 positions of a Unicode prompt and 1 of 64 of a tool-call prompt ("Oracle caps" below).
Since phase 8a such chunks also run their attention in tiles with half operands (T2, below): 1,100
tokens mean KL 2.4e-3 against the old engine, top-1 63/65 with both changes at near-ties.
Long prefills are exact only under the diagnostic `DS4_MELLUM_EXACT_PREFILL=1`
(read at session start), which sends every chunk back through the exact paths;
that is what the oracle uses to check long prefills byte for byte.

Speed on the M5 Max (Q8_0, Metal), step 29, through the CLI after a rest, n = 3
each: a short chat prompt prefills at 440-450 t/s and generates at 132.6-133.1
t/s; a 3,235-token prompt prefills at 1,842-1,871 t/s and then generates at
116.2-117.8 t/s. The trace driver's 2,200-token prefill: about 2,000 t/s. Since E1 (routed Q8_0
experts on the Metal 4 tensor tiles), about 2,880 t/s at 2,200 tokens and 2,060 at 6,000. Through the CLI
since phase 8a (the attention tiles), medians of 6: 4,055 t/s at 2,056 tokens, 3,528 at 8,195 and 1,908 at
32,778, against 2,968, 2,132 and 939 with `DS4_MELLUM_ATTN_TILED=0`. Decode (phase 8b, D1): 133.2 t/s
after 100 prompt tokens, 119.1 after 2,056, 116.7 after 8,195, 98.2 after 32,778; GPU-bound, at 53-60% of the
measured 575 GB/s read ("D: decode research"); since phase 8c-8d the sliding decode kernel's reordered loads
(1.035x at 2,056 tokens, measured) and fused small kernels (about 1.03x in the screen; the landed head was
not re-measured) are on. The M4
Max was not measured. Session save/load refuses by name. Since step 34,
ds4-agent runs Mellum with the Hermes syntax and Pi's four tools. Since phase
6 (steps 41-43), ds4-server serves Mellum as `mellum-2.1`, with the Hermes
syntax and one session: `--batched-session` above 1 and `--kv-disk-dir` are
refused at startup by name. Since phase 7 (steps 44-47), the agent repairs bare
quotes in `write` and `edit` values, reports backtick-delimited values, says
where a missed edit departs, and caps one tool result at a quarter of the
context. The next user turn after a thinking answer reuses the live KV (the
rewind frontier).

**Speed run, 2026-10-29** (M5 Max, merge `633fd23`). The workload is S6 phase C's flattened coding-agent
request: 2,686 prompt tokens, then 512 greedy, context 32,768, through `tools/mellum_s4_measure`; n=3 per
row, rotated.

| Model file and mode | Decode t/s (ms/token) | First token | Metal peak |
|---|---|---|---|
| Q8_0, resident | 122.1 (8.19) | 0.97 s | 12.78 GiB |
| MXFP4 experts, resident | 142.8 (7.00) | 0.95 s | 7.29 GiB |
| Q8_0, streamed, 6 GiB cache | 43.8 (22.84) | 1.41 s | 7.88 GiB |
| MXFP4, streamed, 3 GiB cache | 53.1 (18.82) | 1.60 s | 4.88 GiB |
| MXFP4, streamed, 2.5 GiB cache | 40.7 (24.59) | 1.19 s | 4.37 GiB |

- In every row the three runs give the same 512 ids, and the rows of each file agree with each other.
- Streamed decode was 23 t/s at 2.5 GiB before that day's cheaper stops
  ([result](docs/superpowers/research/2026-10-29-mellum-stream-decode.md)).
- Streaming is opt-in (`--ssd-streaming`). These rows are the M5's; streamed MXFP4 on an 8 GB M1 iMac is in S6
  phase C's [result](docs/superpowers/research/2026-10-28-mellum-s6-c-imac.md).
- The table, its ranges and its commands are in
  [`2026-10-29-mellum-speed-run/`](docs/superpowers/research/2026-10-29-mellum-speed-run/summary.md).

Model-free tests: `make test-mellum`. With `DS4_TEST_MELLUM_MODEL` set,
`./tests/test_mellum_config` also validates, binds and renders against the real
GGUF, `./tests/test_mellum_tokenizer` checks token ids through its vocabulary,
and `./ds4_test --mellum-session` checks session allocation against the plan.

Generated test data (each generator's command is in its docstring; they ran on
the local Mellum2.1-12B-A2.5B-Thinking MLX snapshot, whose `tokenizer.json`
and `chat_template.jinja` are byte-identical to the Hugging Face snapshot
`6f239041` the old port used, by the sha256 its fixtures recorded):
`ds4_mellum_unicode.inc` (`tests/gen_mellum_unicode.py`),
`tests/mellum_tokenizer_cases.inc` (`tests/gen_mellum_tokenizer_cases.py`),
`tests/mellum_chat_ids.inc` (`tests/gen_mellum_chat_ids.py`).

## Changes to upstream code other models run

- The Qwen3.8 attention decode kernels (`metal/qwen4.metal`) gained an
  `ungated` word in place of `pad0` and a branch in the decode tile and the two
  merge kernels, for Mellum's ungated full layers (step 17). The gated path is
  unchanged: a probe hashing the raw output of `ds4_gpu_qwen4_attn_decode_tensor`
  and `ds4_gpu_qwen4_attn_decode_rows_tensor` on fixed inputs (head dims 128
  and 256, 26 to 2,100 keys, one and two rows, both merge kernels, with and
  without partials: 60 cases) gave identical bytes before and after, and
  `make test-qwen4-kernels` passes. Flash Next's decode speed was not
  re-measured: there is no Flash Next GGUF on this machine.
- `ds4_json.h` (step 31): the server's static JSON readers moved to a shared
  header, word for word apart from `static inline` and a small string buffer
  of its own (`json_sbuf`) in place of the server's `buf`; `./ds4_test
  --server` passes unchanged.
- `ds4_chat_append_tool_results` (step 30) is new public API for every model;
  for models other than Mellum it appends one tool message per result, exactly
  as `ds4_chat_append_message` did. Flash Next's tool messages still write
  `<tool_response>` as plain text (fork commit `14212a9` changed that for
  every qwen4 model; this branch does not), and a tiny-vocabulary test checks
  that it is unchanged.
- The `ds4` CLI's `-n` cap message (step 28, `a6c2eb0`): for every model, a
  turn that generated as many tokens as `-n` allows now ends with
  `ds4: stopped at the -n cap of N tokens` on stderr. This applies to the
  one-shot greedy and sampled paths and to a REPL turn. A turn that ends on
  its own prints nothing new, and stdout is unchanged.
- The DeepSeek max-effort preamble (`chat_push_think_prefix`, review fix R6) is skipped for Mellum by name.
  Other models' max thinking is unchanged; a tiny-vocabulary sibling test
  checks that Flash Next still pushes it.
- Q8_0 on the tensor-tile expert kernels (E1). `metal/qwen4.metal`'s `qwen4_load_raw16` and
  `qwen4_dequant_raw16` gain a Q8_0 branch.
  - `qwen4_moe_mm_nax` admits Q8_0 only at the level Mellum sets through the new
    `ds4_gpu_qwen4_moe_q8_tiles`.
  - `ds4_gpu_tensor_tiles_available` is new, for tests.
  - Other weight types and other models' Q8_0 experts are unchanged, and a model-free kernel test checks
    that Q8_0 experts without Mellum's level give the same bytes with and without `DS4_QWEN4_MOE_MM_NAX=2`.
- ds4-server (phase 6). The Hermes renderer, parser, suffixes and live tail are Mellum-only
  branches, chosen by `SERVER_MODEL_SYNTAX_HERMES`. Four places are shared code:
  - The OpenAI, Responses and Anthropic streams now call one helper, `stream_skip_think_open`,
    to skip a leading `<think>`. That behaviour is unchanged for other models; only Mellum's
    newline after the tag is new.
  - `server_engine_refusal` refuses no model now.
  - `server_session_option_refusal` is new, and refuses only Mellum's options.
  - `send_models` names Mellum before the qwen4 check.
- The rewind frontier (after phase 6). `ds4_session_rewind_frontier` is new public API; it
  returns -1 for every model but Mellum, so the server's new last tier in
  `slot_probe_reuse_locked` never fires for them. The memory-rewind materialization's log
  names the model; GLM's lines read as before.
- `ds4_prompt_prefix_append` (`ds4_prompt_prefix.h`, review fix R4): for Mellum
  it no longer pushes EOS after an imported assistant turn, because
  `ds4_chat_append_message` has already closed the turn with `<|im_end|>\n`.
  It also drops the reasoning of imported Mellum assistant turns, as the
  template does (R5). Other models are unchanged. **Possible upstream issue:** Flash Next takes the same path and
  probably renders `answer<|im_end|>\n<|im_end|><|im_start|>user` too; a
  tiny-vocabulary sibling test pins its current render (three `<|im_end|>`
  for a user and an assistant turn). It was not checked against Flash Next's
  template: there is no Flash Next GGUF or template on this machine.

## Commits

Phase 0: `15824be` baseline, `57278db` dependency audit.

Phase 1:
- `e8ba0ac` recognize `general.architecture = mellum`; refuse it by name.
- `5109a6d` validate the hyperparameters, each refusal naming its key.
- `34b5d44` bind the tensors by name and check their Q8_0 layout.
- `f4d1ef1` memory plan and session allocation; the forward entries refuse.

Phase 2:
- `e398212` generated Unicode classes, with the generator (step 7).
- `5e4884d`, `a7f0b4f` tokenizer cases from the model's own tokenizer; the
  `mellum2` pre-tokenizer (step 8). The old port's batch drivers
  (`dump_segments_batch`, `dump_tokens_batch`) are not ported: a generated
  fixture and `tests/test_mellum_tokenizer` replace them, so step 10 (the
  drivers refusing a signed record count) has nothing to apply to; upstream's
  tokenizer entry points take C strings.
- `79922e5` special tokens classed, corpus markers are content (step 9; it
  also covers step 12: `<|im_end|>` is EOS, id 0 EOS-alt, `</tool_call>`
  stops nothing).
- `8021751`, `20a696c` chat ids from the model's own template; Mellum's own
  system turn and assistant prefix (step 11).
- `dd0d997` sampling defaults from the engine, read by the CLI, agent and
  server (step 13).
- `a74bafe` session save and load refuse Mellum by name (step 14).

Phase 3a (forward pass on Metal, identical to the old engine):
- `f58b776` q/k norm and RoPE kernel, YaRN on full layers only (step 15).
- `bb2a2f3` ring commit and sliding-window attention kernels (step 16).
- `f0332e3` ungated full attention on upstream's qwen4 decode kernels (step 17).
- `2712e56` MoE, top-8 of 64 routed experts, no shared expert (step 18).
- `b3f9897` the forward pass behind `DS4_MELLUM_TRACE`, and its trace driver (step 19).
- `f86c6b4`, `46d7d79`: the oracle's token files and comparison script (step 20).
- `48296f7` the Unicode table ends with one newline; `51e898c` the oracle shown failing.
- `9d2ff02` prefill equals decode, op by op (step 21).
- `a27783e` the forward-pass refusal lifted; the CLI runs Mellum chats (step 22).
- `322389e` compaction resumes an open reasoning block with `<think>` (step 23; the
  rest of step 23 is in `5109a6d` and `a74bafe`).
- `63c647f` session rewind keeps both caches one token back (step 24).

Phase 3b (speed):
- `e4a6595` experts as tiled GEMMs from 32 rows (step 25).
- Step 26 not done: `kernel_qwen4_attn_mm` is fixed at head width 256 and gated
  (`constexpr uint D = 256`); Mellum's full layers are 128 and ungated, so it
  would need a kernel change, and the brief says to stop and report instead.
- `219ac0a` Q8_0 projections batched from 32 rows (step 27).

Phase 4 (CLI):
- `c5c4e57` (red), `a6c2eb0` the CLI says when a turn stops at the -n cap (step 28).
- `907094d` what `--nothink` does with Mellum (step 28).
- `be82f23` (red), `b9dbf42` the speed line and session errors name Mellum (step 29).
- `36fe842` the step-29 speed measurement.

Exact-prefill diagnostic (after Checkpoint 3, asked for with keeping step 27):
- `5aa742c` (red), `4bdb7b5` `DS4_MELLUM_EXACT_PREFILL=1` keeps long prefills exact.
- `9e77c2d` the oracle checks every prefill both ways; the next commit records the run.

Phase 5 (tool calls in ds4-agent):
- `4c5ef02` (red), `98a5bc6` a tool message's `<tool_response>` wrapper is
  tokens 31 and 32 (step 30; the idea of fork commit `14212a9`, here Mellum's
  alone).
- `a821dd7` (red), `cdc5b10` the same on the multimodal path ds4-agent uses
  for text-only observations.
- `7514d1f` `ds4_chat_append_tool_results`, which takes the count.
- `11684de` (red), `ba1c0e0` one turn's tool results are one user turn.
- `b2134ce`, `d5cb032` the server's JSON readers move to `ds4_json.h`
  (step 31; a refactor, `./ds4_test --server` unchanged).
- `a81de8d` `AGENT_TOOL_SYNTAX_HERMES`, not yet selected (step 32).
- `534ef56`, `e2958cc` (red), `a440f79` the Hermes parser: the call object
  and string arguments.
- `6ddb1b2` (red), `42e4717` numbers, literals and nested values kept raw,
  through a strict JSON validator.
- `91f5be0` (red), `53b9441` the call object read as `json.loads` reads it.
- `aaebdde` (red), `d2d2b32` prose around calls.
- `9b6aa77` (red), `c98441c` Pi's four tools in the system turn render as
  2.1's template does: a Mellum system message matches special-token text
  (step 33); `a59b281` (red), `0cc60c9` the CLI's `-sys` path the same.
- `80ae163` (red), `5482d43` ds4-agent's one Hermes system turn (intro, -sys,
  date, working directory, tools block).
- `24e2ce5` (red), `285cfdf` think state, turn end, stop classes, `[upto]` off.
- `3ad12c3` (red), `5ffbfe3` a Mellum engine gets the Hermes syntax; the agent
  still refuses it until step 34.
- `e38b18d`, `7c44f0c` the earlier-reasoning divergence, at the code.
- Step 34, Pi's four tools (ported from the old port's `a5a7e82`, split into
  pairs):
  - `96289c4` (red), `c80bdfe` a round's results are one user turn, a raw
    `<tool_response>` block per call;
  - `3dd86ba` (red), `8c85dae` `read`;
  - `315c411` (red), `d9b6ecf` `write`;
  - `45f77b0` (red), `4d062f9` `edit`, exact matching;
  - `627672d` (red), `d3c562d` `bash`;
  - `194dbf8` (red), `5a0b4b0` other tool names get "Tool X not found",
    and the agent no longer refuses Mellum.
- Step 35: `f9d006a` (red), `a15284c` `read` clamps absurd offsets and limits
  (a 1e30 limit crashed); `e138b1f` (red), `6f61a9a` fuzzy `edit` (Pi's
  normalizer without NFKC, occurrences counted on normalized text, unedited
  lines kept).
- Step 36: `022822a` (red), `1a75f9d` the terminal shows a Hermes call's
  arguments JSON-decoded, and Pi's read as `path offset:limit`.
- Step 37: `63866ed` (red), `184a0f2` a turn the token cap cut off is
  reported, not accepted. Abnormal stops were already Hermes-only (step 33).
- Step 38: `e8563e3` (red), `d4d6fb7` a headless run reads the worker's error
  again after its loop, so an abnormal end exits 1 with the line.
- Step 39: `c441f17` (red), `dc66028` a compaction summary that stops on id 0
  or a protocol id fails the compaction.
- Step 40: `379aa54` (red), `cfc9eab` a Pi read shows only the range it asked
  for (found on the live screen).

Phase 6, ds4-server (steps 41-43), a red test before each fix:

- `55df178`, `8960fc6` the Hermes syntax selected for Mellum, ahead of the
  qwen4 check.
- `f69bf7a`, `7f6fbc5` Mellum chats rendered with its own template (with R5's
  server half); `31b19d6` the renders tokenize to the template's ids.
- `ce2cc56`, `3f09160` Hermes output parsed into reasoning, content and
  calls.
- `0cc8574`, `3e06a17`, `5f233d4` a finished turn remembered as the next
  request renders it.
- `16901b0`, `2d81330` the live tool tail and the malformed-call error turn.
- `1c262ce`, `85629bc` reasoning and content streamed as they come.
- `14ab0f4`, `f5aed20` batched sessions and disk KV refused by name.
- `bb5bbf5`, `beed2c7` the server serves Mellum. The model-id check
  landed with the fix, because a failed edit kept it out of the red commit.
- From the live check: `7ee0e8f`, `8accdcc` a turn renders its sampled call
  text, which fixed a KV cache miss; `521e72a`, `462574c` `/v1/models` lists
  `mellum-2.1`.

Phase 6 review fixes (the outside review of `9bc817a`, P6-R1 to P6-R13), a red
test before each fix; nothing Mellum-only changes another model's behavior:

- P6-R1: `bcccc19` the Responses visible key in its own function;
  `ab59a5e` (red), `ad051f0` a Mellum visible key is the KV's own text, or
  none, so a reused prompt keeps the sampled `<|im_end|>`.
- P6-R2: `98fbc30` (red), `3960896` a developer message is the system turn.
- P6-R4: `eff00cc` (red), `13a8830` an Anthropic `system` is the initial
  system message.
- P6-R12: `fcb7600` (red), `ad59ad0` the generation prompt after a history
  that ends in an assistant turn.
- P6-R3: `70a5e85` (red), `fcc07b3` Anthropic tool results are `<tool_response>`
  turns, with the cycle's reasoning kept.
- P6-R5: `dc86d84` (red), `db349e5` reasoning ends at the first `</think>`.
- P6-R8: `ab616d2` (red), `a48921e` string-encoded arguments are one whole
  object.
- P6-R6: `8949e9a` (red), `d65e370` the generation tracker follows Hermes
  calls, so a malformed call is recovered.
- P6-R7: `e694f27` (red), `8a3eae0` restored call text only for the exact
  calls.
- P6-R9: `2857b54` (red), `6e3895f` `\u0000` refused anywhere in a call.
- P6-R10: `bde5001` (red), `bc62016` Pi's string arguments must be strings.
- P6-R11: `eaade46` the deadline test and Pi's timeout named; `1f3cb77` (red),
  `0b2bc9b` Pi's bash has no deadline without a timeout, as in Pi.
- P6-R13: `eecd799` (red), `d0629bc` a repeated `arguments` replaces the
  earlier one.

Upstream cases these fixes leave as they are (recorded, not changed):

- P6-R1, Qwen: `build_thinking_visible_text` and the Responses visible
  suffix still build Qwen's key as the rendered turn closed by
  `<|im_end|>\n`, while generation stops before evaluating the sampled stop
  token. A reuse at those tiers would continue from a KV that lacks it.
  Mellum's keys are now the KV's own text; Qwen's were not probed on a Qwen
  model here.
- P6-R7: `tool_memory_attach_to_messages` predates Mellum and restores a
  sampled block whenever every call id in the request maps to it. The DSML
  and Qwen renderers still trust that block when the client sends a subset
  of its calls or edits one. Only the Hermes renderer now checks the block
  against the sent calls.
- P6-R11: the change is in the shared watchdog, but only Pi's bash passes a
  timeout of 0. The agent's own bash tool always passes at least a second,
  so its behavior is unchanged.

Live re-check after these fixes (`ds4-server -c 16384`, Q8_0, 2026-10-27):

- A plain thinking chat: "What is 17 + 25?" gave `42`. The second user turn,
  "Now multiply that by 2", gave `84`, with 0 cached tokens out of 49.
- The same on Responses: `Paris`, then `Rome`, with 0 cached tokens out
  of 36.
- The bash tool round trip: a `wc -l hello.py` call, then "The file hello.py
  has 6 lines.", with 480 of 499 tokens cached.

The plain second turns miss the cache by design now. The KV holds the first
answer's reasoning, which the template drops from the next render (the log
shows `common=24`, the whole first prompt). Mellum has no tier that rewinds
the KV to a common prefix (upstream rewinds only for GLM), so the server
prefills the template's render. Before P6-R1 it continued from a KV that
lacked the `<|im_end|>`. Each new user turn after a thinking answer now
prefills the whole history. (The rewind frontier, below, restores the reuse.)

The rewind frontier (after phase 6, before phase 7). A rewind more than one
token back cannot just truncate Mellum's KV: the sliding rings hold only the
last 1,024 positions, and a rewind to a common prefix would have reset the
session and prefilled the kept tokens anyway. So the session saves its rings
(K, V and tags) and logits where each sync ends, a copy of 44,126,208 B
counted in the memory plan as KV, and a rewind to exactly there restores
them at any length. A reset, or a rewind below the frontier, makes the copy
stale. The server's last reuse tier rewinds a Mellum slot to its frontier
when the prompt keeps that prefix and goes past it, as the next user turn
after a thinking answer does.

- `7280304` red, `61c9654` fix: the engine. `--mellum-rewind` on the real
  model: sync 1,100 tokens (past the window), decode 40, rewind to 1,100.
  The graph stays at 1,100, 0 of 98,304 logit words differ from the sync's,
  and a next sync of 5 equals a fresh session's sync and extension word for
  word. A rewind below the frontier and a different token there make a later
  rewind to the old frontier reset.
- `e480f46` red, `2825298` fix: the server tier, with siblings (no frontier,
  a prompt leaving below it, a prompt stopping at it). An exact replay of
  the previous prompt, with no token past the frontier, still prefills it.

Live check with the frontier (same server and requests, 2026-10-27):

- Plain chat: `42`, then `84` with 24 of 49 tokens cached (log: "rewound
  Mellum live prefix from 56 to 24, where its last sync ended").
- Responses: `Paris`, then `Rome` with 17 of 36 cached (rewound from 129
  to 17).
- Bash tool round trip: unchanged, 480 of 499 cached.
- Past the window: a system prompt of 150 facts (2,953 tokens), "What is the
  code word for item 12?" gave `w444`. The second user turn, "And for item
  40?", gave `w480` with 2,953 of 2,980 cached (rewound from 3,066 to 2,953).
  Both answers are correct.

Phase 7, hardening (steps 44-47, 2026-10-27), each a red test and its fix.
The evidence is ds4-engine's pathologies 140 and 148 (Mellum 2.1 on that
host: unescaped quotes break a call's JSON, and backtick-delimited values).

- Step 44, quote repair. `ef5e1fa` red, `2bf606a` fix: a `write` whose
  arguments fail the strict parse no longer ends the turn at the first bad
  byte. The error waits for `</tool_call>`, and when the body is exactly
  `{"name": "write", "arguments": {"path": <a valid string>, "content":
  "..."}}`, the bare quotes in the content are escaped and the result parsed
  strictly again. Otherwise the strict parse's error stands: a bare quote in
  the path, a content that folded another member into itself, a turn cut off
  before `</tool_call>`. `0a45b7d` red, `6661089` fix: the same for `edit`'s
  `edits[].oldText` and `newText`, cut at the boundaries between them, which
  must alternate and end in a `newText`. `edits` is a raw JSON value, so an
  odd count of bare quotes kept it open past `</tool_call>` with no error at
  all; at the turn's end such a call gets the same repair. The legacy
  top-level `oldText`/`newText` form is not repaired. `0e90f56` red, `e9f5afb`
  fix: a value delimited with backticks is reported at once, naming the
  argument and saying how a JSON string is written; it is not repaired.
- Step 45, `6fcc9ab` red, `82dfeb2` fix: after Pi's own message, a missed
  edit names the longest start of its `oldText` the file holds, the line and
  column where it departs, and what each side has there. The search runs on
  the exact text, so a miss that only the fuzzy pass would bridge (smart
  quotes) departs at the first such character.
- Step 46, `57da1dc` red, `646829e` fix: in the Mellum round, a result
  longer than a quarter of the context in tokens keeps its head and tail,
  with a note between them (how many tokens were cut, the cap, how to ask for
  less). Other models' rounds are unchanged.
- Step 47: antirez `main` is still `ee7e840` (`git ls-remote`, 2026-10-27),
  which `mellum` already sits on, so there was nothing to rebase: no new
  upstream commits, no conflicts, no SHA changes, and no tag or SHA map
  needed. On the head the full gate passes with the CPU build, and the
  oracle passes: exact prefill, decode and the 64 greedy tokens byte-equal to
  the old engine; the release prefill within tolerance (1,100 tokens: mean KL
  1.6e-3, top-1 64/65 with the one change at a near-tie; 2,200: 1.3e-3,
  64/65).

Phase 7b, the gaps left at Checkpoint 5 (steps 48-50, 2026-10-27):

- Step 48, `129f74b` red, `303c698` fix: the quote repair takes `edit`'s
  older single-edit form, `oldText` and `newText` after the path, cut at their
  one `", "newText": "` boundary; any other boundary leaves it unrepaired.
- Step 49, `a711ace` red, `0d3b8f2` fix: the missed-edit departure is found
  under Pi's fuzzy normalisation and mapped back to the file's bytes, so a
  curly quote typed straight or dropped trailing spaces are no departure.
- Step 50, NFKC in Pi's fuzzy edit: skipped by Paul's decision at
  Checkpoint 5b, on its table's size. NFKC stays a recorded divergence from
  Pi (Known divergences, below). The generator and its table are kept in
  `misc/step50-nfkc/`, which is git-ignored, so they are not in the
  repository. The generator takes the decompositions
  and compositions from node (Pi's runtime, Unicode 17.0) and the combining
  classes from Python (Unicode 16.0) matches node's NFKC on all 1,112,063 code
  points and 8,519 probes. Its table is 5,914 decompositions (9,193 code
  points), 961 compositions and 393 class ranges: 2,313 lines and 219 KB, or
  about 167 KB packed, against 192 KB for the largest generated table here.
  Pi's fuzzy edit still runs without NFKC, so a full-width or ligature
  difference is a miss here where Pi would match.

Phase 8a, tiled prefill attention (T1 and T2, 2026-10-27). After Checkpoint 6 Paul said "Proceed and
continue through all of it until done"; both proposed variants then landed, the f32 tiles first:

- T1, CLI prefill with E1: `518ee06` the runner, `16b52ed` its state lines, `6cbf14e` raw logs,
  `a06c490` arms from the environment, `fc6dece` the row.
- T2, f32 tiles (probe variant 1b): `df8d1bb` (red), `5c2af4f` the kernel; `853087d` (red), `218bfef`
  chunks of 32 rows or more use it, `DS4_MELLUM_ATTN_TILED=0` keeps the decode kernels.
- T2, half operands (probe variant h32, a precision change): `e967e4b` (red), `52af643` the kernel and
  the corrected staged reference; `339dded` (red), `81b5f61` the default, `DS4_MELLUM_ATTN_TILED=1`
  keeps the f32 tiles. The f32 tiles were removed on 2026-10-31 (`0a0c9c2` red, `0ea3328`); `=1` is
  now the default, and `=0` still selects the decode kernels.
- `905b15e` the runner's empty arm, `1860c51` T2's raw logs.

Phase 8b, decode research (D1-D3, 2026-10-28; Paul: "Proceed with next step, I believe it is decode
research"). Measurement tooling only; nothing in the product changed:

- D1: `98217a8` bytes per step from the GGUF, `f7694b7` as functions; `ea9d418` the bandwidth tool, `a23ad07`
  its output; `1019023` the decode runner, `421bc35` its summary, `f0e270b` raw logs; `eee2c6f` the bytes and
  prompt-lookup outputs.
- D2, B0: `6b64bae` (red), `bc716e4` the test hook; `ea0f37b` the runner, `ec8860f` raw logs.
- D3: `f4a0432` prompt-lookup acceptance from token files; screen S1 in `misc/d-decode/` (not committed).
- `d8cb20f`, `4a50270`, `dc5943e` the record.

Phase 8c, decode (2026-10-28; Paul at Checkpoint 7: "Land S1, screen 2-3"):

- S1, the sliding decode kernel's loads reordered, its arithmetic unchanged: `0b19b01` (red), `f160af7` the
  reference arithmetic kept for the kernel tests, `6604b19` the kernel; `385fbd2` the decode A/B/B/A
  runner, `8d316ea` raw logs.
- S2 and S3: screens only, `misc/d-decode/s2-s3-screens.diff` and its README (not committed).

Phase 8d, S2 landed (2026-10-28; Paul at Checkpoint 8: "Land S2"; then, before the speed run: "I think I want
to skip a final speed run. I have a branch for SSD streaming that might impact it."). Decode only (T = 1);
chunks keep the separate kernels:

- G1, q and k norm and rope in one dispatch: `f029488` (red), `8fa20e4` the kernel, `4bda83b` decode uses it.
- G2, the K and V f16 copies in one dispatch: `caee78f` (red), `c06d9c1` the kernel, `cd66454` decode uses it.
- G3, the MoE reduce and residual add in one dispatch: `9090219` (red), `baa9451` the kernel, `3a215f3`
  decode's MoE uses it.
- G4a, upstream's fused add and RMS norm: `75e9da6` the test (not red: the kernel exists; the commit records
  the breaks planted in it and which the test caught), `0bb422a` decode uses it.
- `6538f80` the decode A/B/B/A runner takes a binary per arm (for the skipped run).

Phase 9, the README (2026-10-28; Paul: "README is good", "Let's just make everything on README.md file", "New
ds4-mellum repo", "Let's wait to do the speed runs until after the go/no-go on SSD."):

- RM1, the runners' single-arm mode: `e08c4ac` (red), `49bf9eb` `ARMS`; `fff1b44` (red), `ba83406` a
  single-arm prefill run sets no variables.
- RM5, `README.md`: `c692779` the guide on top, upstream below a divider; `ebc93df` Pi, settings, thinking;
  `942c6e9` speed and measuring; `5c84246` a sentence about the coming tables.
- RM3 and RM4: `98da6cb` raw logs of the checks.

Review fixes and experiment E1 (after step 33, before step 34; the outside
review of `5aa742c`):

- R1, the oracle rejects NaN and Inf: `a6ce058` the comparison moves to
  `tests/mellum_logits_compare.py` with tests; `c99ac87` (red), `fc16a75`
  every row of both runs finite first; `807c7be` (red), `28fdbd5` the trace
  driver checks every listed position.
- R2, per-position gates: `381d402` (red), `a3a7962` four held-out token
  files, `fff47fc` worst-position KL and |d log p(top-1)| caps, the session
  row counted once; `62dce64` the caps' source and the router-flip finding.
- R4: `c95e0e8` (red), `5d75ebe` one `<|im_end|>` after an imported Mellum
  assistant turn; `890da18` Flash Next's probable double close recorded.
- R5: `6d60625` (red), `544d3d5` imported Mellum histories drop earlier
  reasoning; `a468bfb` the divergence restated.
- R6: `98f40bf`, `9bbf9fe` the deep-rewind contract (a fresh sync's logits),
  tested and stated; `6eeade3` (red), `c479351` no DeepSeek max-effort
  preamble for Mellum; `4e5fbf7` (red), `119001b` head and width shapes
  refused at validation; `8fbd9b3` the `-n` cap message listed.
- R3, fast-path coverage: `6bd685e` the trace driver's `prefill:N` and
  greedy modes; `033dbf0` `tests/test_mellum_fastpath.sh`; `a8e5b4b` the record.
- R7: `767a29a` the trace writes each layer's MoE input row; `11ed730` the
  router and experts on real activations against this port's double
  reference, with a half-staged reference for the tiled rows; `99c259d`.
- R8: `f16a1dd` the Metal 4 record reconciled.
- E1: `86fcd05`, `8e7b158` measured and reverted (no product code).

After the rebase onto `ee7e840` (`9986668` the SHA map):

- E1a: `3864941` (red), `6f933ac` the near-tie top-1 rule; `08e39fd` its margin.
- E1b: `849ee5b` held-out log loss.
- E1c: `face2b0` Q8_0 raw words; `db073ac` (red), `816d062` Mellum's opt-in;
  `2d0756d` the Flash Next boundary test; `5e3b53f` the measurements.

From step 28 on, commits follow smaller rules: one concern each, the failing
test committed red on its own, at most about 150 lines of product code.

## Smoke runs

Step 22, M5 Max, Q8_0, `--temp 0`, n = 1 each:
- `./ds4 -m $MODEL -c 4096 --temp 0 -n 400 -p "What is the capital of France? Answer in one sentence."`
  reasons briefly ("...The capital of France is Paris. So answer...") then answers
  "The capital of France is Paris." and stops.
- The same with `--nothink` and "Write a Python function that returns the n-th
  Fibonacci number.": the prompt ends with the template's empty think block
  (`--dump-tokens` shows `<think>\n\n</think>\n\n`), yet the model still
  reasons in plain text before its answer, and 300 tokens ran out before the
  code. So with thinking off Mellum 2.1 does not answer directly; the CLI's
  handling of that is step 28.

Step 40, the live agent screen, at cfc9eab. Setup: M5 Max, Q8_0, Mellum's default sampling (temperature 1.0,
top-p 1.0). Each run used a fresh scratch directory holding only `hello.py` (a `greet` function; it prints
"Hello, Mellum!"), and ran `ds4-agent -m $MODEL --non-interactive --seed N -c 16384 --trace FILE -p PROMPT`
from that directory, with the Metal sources named through their `DS4_METAL_*_SOURCE` variables. One
process at a time. Wall time is the whole process, model load included.

| prompt | seed | calls, in order | how it ended | wall | exit |
|---|---|---|---|---|---|
| "Read hello.py and say what it prints." | 1 | `read {path}`; `bash python hello.py` (127: no `python` here); `bash python3 hello.py` | correct ("Hello, Mellum!"), clean stop | 9.8 s | 0 |
| the same | 2 | `read {path}`; `bash python3 hello.py` | correct, clean stop | 2.9 s | 0 |
| the same | 3 | `read {path}`; `bash python hello.py` (127) | correct from the code, clean stop | 3.0 s | 0 |
| "Change hello.py so it prints Hello, World! instead, then run it to check." | 1 | `read`; `edit` with single-quoted strings, refused by the parser as `json.loads` would refuse it, with the Hermes syntax reminder; `edit {path, edits[{oldText, newText}]}`, "Successfully replaced 1 block(s)"; `bash python hello.py` (127); `bash which python3 \|\| which python`; `bash python3 hello.py` ("Hello, World!") | file edited correctly, clean stop | 28.8 s | 0 |

Every tool round stopped on the call's closing `</tool_call>` (id 30), and every final answer on a stop
token with no abnormal end traced. Since step 37 any stop other than EOS is traced and reported, and
exits 1. The trace records the tokens but not the sampled stop token itself. The first run's 9.8 s against
about 3 s for the next two is probably Metal compiling its shaders once; the edit's 28.8 s is mostly 954
tokens of reasoning before its first, malformed, call. Each run printed "failed to save system prompt KV:
Mellum 2.1 session save and load are not supported yet": the save is refused by name, and the run goes on.
The screen found one display bug, a Pi read shown with the agent's own default range ("hello.py 1:240"),
fixed in `379aa54` (red) and `cfc9eab`.

## Known divergences from training

- `--nothink` (the template's `enable_thinking=false`, an empty think block in
  the prompt): simple questions get a direct answer ("Paris", "Buongiorno",
  "17 times 23 is 391."), but asked to write a Fibonacci function the model
  still reasoned in plain text, wrote `</think>` as ordinary text (byte pieces,
  not the special token, so nothing stops), then answered. The prompt equals
  the template's; this is the model's own behavior, handled as for other
  families: the CLI prints it as it comes (`tests/test_mellum_cli.sh` checks a
  direct answer on the simple question).

- Earlier reasoning in live transcripts (the ds4-agent and the CLI's chat):
  2.1's template drops the reasoning of the assistant turns before the last
  user turn when it renders a new one. A live transcript is its KV cache and
  keeps every turn as generated; dropping the reasoning would mean
  prefilling the conversation again at every user message. Within one user
  turn (a tool loop) the two agree. Rendered histories follow the template
  (review fix R5): a prompt-prefix file's assistant turns all come before
  the live prompt, so their reasoning is dropped (`ds4_prompt_prefix_append`,
  pinned against the template's ids). When this was written ds4-server still
  refused Mellum; it has served Mellum since phase 6 (steps 41-43).
- Special-token text in user turns: the HF tokenizer would encode a
  `<tool_call>` typed in a user message as token 29; ds4 keeps user text plain
  BPE, as for every model. System text (the tools block, `-sys`) and assistant
  text match special tokens, as the template's tokenizer does.

- Pi's tool texts by version. The training contract (Paul's ds4-engine research notes) puts training
  on Pi 0.84.4. Two copies of Pi are installed here: 0.80.10 (bundled by pi-xcode) and 1.1.0. The
  agent's texts are 0.80.10's, which the contract says 1.1.0 changed only in `write`. `write` says
  "Successfully wrote N bytes to P", with N being JavaScript's `content.length` (UTF-16 units); 1.1.0
  says "Successfully wrote to P". `bash`'s "Command terminated without an exit code" is 1.1.0's:
  0.80.10 has no such case. An unknown tool gets Pi's agent-loop text, "Tool X not found". `edit`'s
  texts are the same in both versions. Since step 35, `edit` matches as Pi does, except that its
  fuzzy pass has no NFKC step, by Paul's decision at Checkpoint 5b (step 50): a verified generator
  exists (`misc/step50-nfkc/`, git-ignored), but its table, 219 KB, was not taken into the
  repository. So compatibility forms such as full-width letters and ligatures don't match their
  ASCII spelling.
- `read` with an offset or limit past the file's end reads as Pi does. A negative limit reads
  nothing and reports how many lines remain; Pi's own arithmetic would count from before the
  start.

- ds4-server with Mellum: a request's earlier reasoning follows the template (R5): dropped before
  the latest real user message, kept within the current tool cycle. Two things still differ from
  what the template alone would give:
  - A turn whose calls the server sampled renders those calls as sampled (`raw_tool_text`, kept
    by tool memory). The client's arguments come back minified, as the server returns them for
    every model, and rendering those would leave the KV cache's bytes.
  - A streamed answer's content begins with the template's blank lines after `</think>`; the
    non-streamed answer trims them.
  The server does not yet treat Mellum's id-0 or protocol-token stops as errors the way the agent
  does (steps 33 and 37).

## Measured

Scripts and outputs this table names that are no longer in the tree (the decode-research scripts, the B0
runner, `tests/mellum_measurements/`) are on the branch `mellum-evidence-2026-10-31`, as of `6f8d91b`.

| what | command | result | n |
|---|---|---|---|
| tokenizer ids vs the model's own tokenizer, 30 strings | `DS4_TEST_MELLUM_MODEL=... ./tests/test_mellum_tokenizer` | 30/30 equal | 1 |
| chat renders vs the model's own template, 6 conversations (3 with tool messages) + 2 prefixes | `DS4_TEST_MELLUM_MODEL=... ./tests/test_mellum_config` | all equal | 1 |
| forward pass vs the old port (`refs/old/mellum21-stage1`), session logits | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64` | byte-equal (`cmp`) on 29, 1,100 and 2,200 templated tokens, prefill and decode; prefill equals decode in this engine on all three; 64 of 64 greedy tokens equal, with equal logits at each step | 1 |
| prefill speed, exact experts vs tiled (step 25), 2,200 templated tokens, one chunk of 1,024 at a time | `MELLUM_TRACE_REPEAT=6 tests/mellum_trace $MODEL tests/mellum_oracle_2200.tokens prefill` (prototype switch, then the committed code) | runs 3-6 (steady clocks; run 1 is ~20% faster, then the GPU settles): exact 349-364 t/s, tiled 813-886 t/s, two alternated sessions each | 4 per arm, x2 |
| tiled-experts crossover, one chunk | the same with 9- to 256-token prompts, runs 3-8 of 8 | 9 rows: exact 0.031 s, tiled 0.044 s; 16: 0.046 vs 0.051; 32: 0.081 vs 0.067; 48: 0.114 vs 0.080; 72: 0.164 vs 0.102; 256: 0.61 vs 0.29. Threshold: 32 rows | 6 per point |
| batched Q8_0 projections (step 27), 2,200 tokens prefill | `MELLUM_TRACE_REPEAT=6 tests/mellum_trace ...2200.tokens prefill`, prototype switch alternated with the step-25 code, 30 s rests | runs 3-6, two rounds: step 25 805-884 t/s; batched with the Metal 4 tensor API on 1,973-2,065 t/s; batched with `DS4_METAL_DISABLE_METAL4=1` 1,971-2,066 t/s (the tensor tile adds nothing measurable here; the batching does). **Wrong, kept as recorded:** that arm never disabled Metal 4 ("Metal 4 record" below) | 4 per arm, x2 |
| batched projections crossover, one chunk | runs 3-8 of 8, projections exact vs batched (experts tiled from 32 in both) | 16 rows 0.048 vs 0.043 s; 32: 0.071 vs 0.059; 96: 0.127 vs 0.076; 256: 0.28 vs 0.13; 1,024: 1.21-1.31 vs 0.51-0.61. Threshold kept at 32 (one rule with the experts) | 6 per point |
| committed code after step 27, 2,200 tokens prefill, after a 2-minute rest | `MELLUM_TRACE_REPEAT=6 tests/mellum_trace ...2200.tokens prefill` | 1,977-2,030 t/s, all six runs. Heat-soaked (many runs just before): 930-1,005 t/s, so rest before timing | 6 |
| generation | `./ds4 -m $MODEL -c 8192 --temp 0 -n 512 -p "Explain how a hash map works, with an example in C."` (CLI's own line) | 133.3-133.5 t/s generation; prefill of the ~25-token prompt 439-511 t/s | 3 |
| step 29, short prompt, after a 2-minute rest, 30 s between runs | the same command | prefill 439.6, 440.6, 450.4 t/s; generation 132.6, 133.1, 133.1 t/s (each turn stopped at the 512 cap) | 3 |
| step 29, 3,235-token prompt, 30 s between runs | `./ds4 -m $MODEL -c 8192 --temp 0 -n 256 --prompt-file PROMPT` (9,000 characters of promessi_sposi.txt and "Summarize the text above in three sentences.") | prefill 1,842, 1,871, 1,846 t/s; generation 116.2, 117.8, 116.6 t/s | 3 |
| long-prompt prefill and generation | `./ds4 ... -n 256 --prompt-file` (9,000 characters of promessi_sposi.txt and a request: 3,235 tokens) | prefill 1,900 t/s, generation 119 t/s | 1 |
| Metal 4 tensor API | the startup line of any run | "ds4: Metal 4 tensor API enabled for Tensor kernels" on the M5 Max; `tensor_matmul=off` with `DS4_METAL_DISABLE_METAL4=1`. M4: not measured (no M4 here) | 1 |
| both prefill paths vs the old engine (after the exact-prefill switch) | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64` | under `DS4_MELLUM_EXACT_PREFILL=1`: byte-equal to the old engine and to decode on 29, 1,100 and 2,200 tokens (30, 65 and 65 logits files); release path: within tolerance (2.008 / 2.1e-3 and 8.471 / 1.7e-3, top-1 65/65); decode byte-equal; 64/64 greedy equal; PASS | 1 |
| oracle with per-position caps (R2), fff47fc | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64` | release prefill vs the old engine and vs decode: 1,100 tokens mean KL 2.1e-3, worst KL 4.2e-2 (p802), worst \|d log p(top-1)\| 0.268; 2,200: 1.7e-3, 2.2e-2 (p803), 0.161; top-1 65/65 both; exact prefill and decode byte-equal; 64/64 greedy equal; PASS | 1 |
| fast paths at chunk and window edges (R3), 033dbf0 | `DS4_TEST_MELLUM_MODEL=... make test-mellum-fastpath` (prefixes of `tests/mellum_heldout_long.tokens`: 31, 32, 1023, 1024, 1025, 2048 tokens; chunks 1024, 512, 100) | 45 of 45 pass. Untraced = traced, byte for byte, in all 18 count and chunk pairs. Release vs exact at the edges (0, 31, 99, 100, 511, 512, 1022-1024, last): byte-equal at 31 tokens; otherwise top-1 equal everywhere, worst KL 4.7e-3, worst \|d log p(top-1)\| 0.136. Eight decode steps after a release prefill of 32, 1023, 1024, 1025: worst KL 4.9e-3, top-1 9/9 each. 32 greedy tokens from a release-prefilled 1,024-token cache equal the exact path's, and each is the top token under the exact path. Chunks of 512 and 1024 gave byte-identical logits on 1,023 tokens (every row tiled either way; observed, not asserted); chunk 100 differs (its last chunk is 23 exact rows) | 1 |
| deep rewind contract (R6) | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-rewind` | sync 31, eval 3, rewind to 32, eval: 0 of 98,304 logit words differ from a fresh sync of the 32 kept tokens, 98,304 from the straight run (the replay's 32-row chunk is tiled; the original was 31 exact rows and a decode step) | 1 |
| rewind frontier, 61c9654 | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-rewind` | sync 1,100, decode 40, rewind to 1,100: graph at 1,100, 0 of 98,304 logit words differ from the sync's; the next sync of 5 tokens differs in 0 from a fresh session's; a stale frontier resets (graph at 0). Prompt 1,100, k 3, now back to the sync's end: 0 of 294,912 words from the straight decode. The session line: 417,902,848 B allocated = planned at 4,096 context (was 373,776,640) | 1 |
| router and experts on real activations vs this port's double reference (R7), 11ed730 | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-moe` (MoE input rows of `tests/mellum_heldout_unicode.tokens` at 64 positions, layers 0/6/10/13/14/27, 384 rows per path) | selection equals the reference in every row; 1 near-tie (8th/9th router logit gap < 1e-4), broken the same way. Exact rows: output 1.2e-6, weights 3.2e-7. Tiled rows: 1.1e-4 from a reference that stages as the tiled kernels do (input, half(d * q) weights, SwiGLU output as half), and 2.5e-3 from the unstaged one at layer 27 (max \|x\| 21.8): about 96% of the tiled path's error on real rows is the half staging. The random-row check's 1.0e-3 understated that cost | 1 |
| E1 landed (tensor-tile experts on by default) vs `DS4_MELLUM_MOE_NAX=0`, 2d0756d, 2,200 tokens | `abba.sh` (session scratchpad; copy in `misc/e1-q8-tensor-tiles/`): A/B/B/A, arm A `DS4_MELLUM_MOE_NAX=0`, arm B default, 6 runs per process, first dropped, 45 s between processes, AC, Automatic, no thermal or performance warning before or after any process | second round: A median 2,045 t/s (1,897-2,057), B 2,881 (2,814-2,893), **+41%**. A first round gave A 1,628 and B 2,742 (+68%), but its first A process ran at 1,253-1,423 against 1,832-1,994 for the same arm at the round's end, so that round is not used. Each arm's two processes gave byte-identical logits in both rounds | 10 per arm, x2 |
| the same, 6,000 tokens (`mellum_heldout_long`) | the same, 4 runs per process, first dropped | A median 1,596 t/s (1,389-1,719), B 2,057 (1,828-2,220), **+29%** | 6 per arm |
| CLI prefill, E1 on (default) vs `DS4_MELLUM_MOE_NAX=0`, phase 8a T1, 518ee06 | `PER=4 COOL=45 GAP=10 tests/mellum_cli_prefill_abba.sh OUT $MODEL 2048 8192 32768`: `ds4 -c CTX --temp 0 -n 1 --prompt-file` on the start of `speed-bench/promessi_sposi.txt` (Italian prose) plus one request; blocks NAX=0, E1, E1, NAX=0 of 4 processes, each block's first process dropped (ds4 runs one prefill per process, after its GPU warmup), 45 s before a block, 10 s between processes; M5 Max, 128 GB (`sysctl`), AC, Automatic, no thermal or performance warning before or after any block. Raw logs: `mellum-evidence-2026-10-31:tests/mellum_measurements/t1_cli_prefill/` | 2,056 tokens (ctx 3,072): NAX=0 median 1,966 t/s (1,957-1,982), E1 2,792 (2,781-2,852), **+42%**. 8,195 tokens (ctx 9,216): 1,623 (1,599-1,638), 2,108 (2,056-2,122), **+30%**. 32,778 tokens (ctx 33,792 is enough): 737 (681-797), 850 (806-881), **+15%** | 6 per arm per length |
| CLI prefill, attention tiles (T2), 81b5f61 | T1's runner, prompts and conditions (`A_ENV=DS4_MELLUM_ATTN_TILED=0`, B the default, then B `DS4_MELLUM_ATTN_TILED=1`); raw logs `mellum-evidence-2026-10-31:tests/mellum_measurements/t2_attention_tiles/` | half tiles (default) vs decode kernels: 2,056 tokens 4,055 (4,039-4,078) vs 2,968 (2,964-2,977) t/s, **1.37x**; 8,195: 3,528 (3,522-3,529) vs 2,132 (2,131-2,133), **1.65x**; 32,778: 1,908 (1,802-2,026) vs 939 (870-1,000), **2.03x**. f32 tiles vs decode kernels: 3,331 vs 2,975 (1.12x); 2,576 vs 2,133 (1.21x); 1,260 vs 936 (1.35x) | 6 per arm per length |
| oracle and fast paths with the half tiles, 81b5f61 | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64`; `DS4_TEST_MELLUM_MODEL=... ./tests/test_mellum_fastpath.sh` | oracle PASS: 1,100 tokens top-1 63/65 (2 changes at near-ties), mean KL 2.35e-3, worst 4.4e-2; 2,200: 65/65, 1.96e-3, worst 2.1e-2; exact prefill, decode and 64 greedy tokens byte-equal. Fast paths 45/45, worst release-vs-exact KL 4.8e-3, 32/32 greedy. At 218bfef (the f32 tiles default): oracle PASS, 1,100 64/65 (1 near-tie), mean KL 2.52e-3; 2,200 64/65, 1.65e-3; fast paths 45/45 | 1 |
| memory bandwidth, phase 8b D1, ea0f37b | `tests/mellum_copy_bandwidth 1024 20`, `... 4096 20` (a Metal copy kernel and a grid-stride read kernel on private buffers, each dispatch its own command buffer, 3 warmups), 30 s idle before each, AC, Automatic; raw output `mellum-evidence-2026-10-31:tests/mellum_measurements/d1_bandwidth/` | 1 GiB: read median 575.4 GB/s (495.6-579.4), copy (read + written) 534.9 (497.7-557.2). 4 GiB: read 547.9 (525.8-575.7), copy 542.6 (530.7-554.5) | 20 per kernel per size |
| bytes one decode step reads, D1 | `python3 -I tests/mellum_decode_bytes.py $MODEL 100 2048 8192 32768` (the GGUF's tensor table; refuses if a tensor is uncounted); output `mellum-evidence-2026-10-31:tests/mellum_measurements/d1_decode_bytes/` | weights 2,363.4 MB per step: the 8 selected experts 1,474.0, attention projections 631.7, output head 240.6, F32 router 16.5, norms 0.6, embedding row 0.002. K/V read: 5.7 MB at 100 ids of context, 73.5 at 2,048, 161.6 at 8,192, 513.9 at 32,768 (17.9% of the step) | arithmetic |
| CLI decode, D1, ea0f37b | `PER=4 COOL=45 GAP=10 GEN=256 tests/mellum_decode_profile.sh OUT $MODEL 100 2048 8192 32768` (T1's prompts; each length's first process dropped), then `tests/mellum_decode_profile_summary.py OUT $MODEL 575.4`; raw logs and `anatomy.txt` in `mellum-evidence-2026-10-31:tests/mellum_measurements/d1_decode_profile/` | generation: 100 tokens 133.15 t/s (132.82-134.45); 2,056: 119.08 (118.80-119.14); 8,195: 116.67 (116.51-117.24); 32,778: 98.15 (96.02-98.90). Per step: one command buffer, 535 dispatches, one wait; GPU 92-94% of the wall step; 303-344 GB/s, 53-60% of 575.4. Section "D: decode research" has the split | 3 per length (+1 CB_TIMES, +1 timeline process) |
| B0, a two-row verify against two decode steps, D2 | `N=16 tests/mellum_b0.sh OUT $MODEL 100,2048,8192,32768` (`./ds4_test --mellum-b0`: D V V D rounds at a fixed context, the first round dropped; V = one two-row forward and the row-0 head in its own command buffer); raw logs `mellum-evidence-2026-10-31:tests/mellum_measurements/d2_b0/` | verify / one decode step: 1.318 at 100 ids, 1.290 at 2,048, 1.315 at 8,192, 1.439 at 32,768 (two steps 14.52, 16.75, 17.26, 20.87 ms; verify 9.57, 10.80, 11.35, 15.02 ms, of which the row-0 head 0.61-0.68). Both rows' logits equal the decode steps' bytes in every run | 30 per arm per context |
| prompt-lookup acceptance on token files, D3 (model-free) | `python3 -I tests/mellum_prompt_lookup.py FILES` (the four held-out and three oracle files); output `mellum-evidence-2026-10-31:tests/mellum_measurements/d3_prompt_lookup/` | drafts from the last 1-3 ids, right on the file's own next id: tool 34.2% of positions (49.3% of drafts), long C source 49.2% (55.2%), code 13.5% (27.5%), Unicode 12.1% (42.1%), 2,200-id prose 7.5% (12.4%). Walked with drafts of 1 id: 1.265, 1.426, 1.116, 1.088 and 1.066 ids per step, a draft verified on 61%, 85%, 43%, 23% and 58% of steps | every position |
| screen S1, the sliding decode attention loading two keys per simdgroup before using them (scratch, `misc/d-decode/`) | kernel tests and the oracle in a scratch clone at f0e270b with the patch, `DS4_MELLUM_SWA_KERNEL=pf2`; `misc/d-decode/abba_decode.sh` A/B/B/A (PER=4, COOL=30, GAP=10, GEN=256, each block's first process dropped) on D1's prompts | kernel tests 584/584; oracle PASS with decode byte-equal to the old engine and 64/64 greedy equal; the kernel 52.9 to 39.7 us at 2,056 tokens (timeline, n = 1,344 dispatches). Decode: 100 tokens 137.47 vs 136.88 t/s, **1.004x**; 2,056: 126.20 vs 121.53, **1.038x**; 8,195: 117.73 vs 114.27, **1.030x**. Every process's generated text identical | 6 per arm per length |
| S1 landed (the sliding decode kernel loads two keys per simdgroup before using them), phase 8c, 6604b19 | `A_ENV="DS4_METAL_MELLUM_SOURCE=<git show f160af7:metal/mellum.metal>" tests/mellum_cli_decode_abba.sh OUT $MODEL mellum-evidence-2026-10-31:tests/mellum_measurements/d1_decode_profile 100 2048 8192` (PER=4, COOL=30, GAP=10, GEN=256; each block's first process dropped; arm A the kernel before S1, arm B the landed head); `./tests/test_mellum_kernels`; `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64`; `./tests/test_mellum_fastpath.sh`; raw logs `mellum-evidence-2026-10-31:tests/mellum_measurements/s1_decode_abba/` | decode: 100 tokens 134.08 (133.32-134.42) vs 134.13 (133.84-137.31) t/s, **1.000x**; 2,056: 126.15 (126.08-126.21) vs 121.85 (121.12-121.95), **1.035x**; 8,195: 120.91 (120.48-121.16) vs 117.00 (116.85-117.46), **1.033x**; generated text identical in every process. Kernel tests 696/696, the kernel word for word equal to `kernel_mellum_attn_swa_ref` at 1-48 and 1,040-1,047 keys; oracle PASS, decode, exact prefill and 64 greedy tokens byte-equal; fast paths 45/45 | 6 per arm per length |
| screen S2, the small per-step kernels fused (scratch, `misc/d-decode/`, phase 8c) | a scratch clone at 385fbd2 (S1 landed) with `s2-s3-screens.diff`, decode only, `DS4_MELLUM_FUSE` bits: G1 Q and K head norm with RoPE in one dispatch, G2 the K and V f16 copies in one, G3 the MoE reduce with its residual add, G4a the attention residual add with the FFN norm, G4b the MoE add with the next layer's attention norm (G4a and G4b through upstream's `kernel_add_rms_norm_mul_f32_4`); kernel tests, oracle, fast paths; `tests/mellum_cli_decode_abba.sh` against the S1 head (PER=4, COOL=30, GAP=10, GEN=256) | byte-equal: oracle PASS with bits 23 and 15 (decode, short prefill and 64 greedy tokens equal to the old engine), kernel tests 696/696, fast paths 45/45 (bits 23). Bits 23 (G1 G2 G4a G4b): 100 tokens 135.74 vs 131.18 t/s, **1.035x**; 2,056: 124.25 vs 120.84, **1.028x**; 8,195: 118.60 vs 115.54, **1.026x**. Bits 15 (G1 G2 G3 G4a): 136.55 vs 132.95, **1.027x**; 126.76 vs 122.62, **1.034x**; 121.42 vs 117.31, **1.035x**. 535 dispatches per step down to 424 and 423; generated text identical in every process | 6 per arm per length |
| screen S3, K and V in the Q projection's dispatch (scratch, phase 8c) | the same clone, `DS4_MELLUM_QKV`: `kv` and `qk` upstream's pair matvec (each threadgroup both banks), `conc` one concurrent encoder section, `kvcat` and `qkvcat` one bank per threadgroup (upstream's `kernel_qwen4_q8_concat`, a three-bank copy of it); probe `-n 64` at 2,056 tokens, median decode-buffer GPU time; A/B/B/A for `qkvcat` | probe, n = 1 process each: base 7,625 us; `kv` 7,767; `qk` 8,076; `conc` 7,809; second round base 7,878, `kvcat` 7,891, `qkvcat` 7,885. `qkvcat`: oracle PASS byte-equal; A/B/B/A 132.27 vs 131.17 t/s, **1.008x**; 122.69 vs 121.35, **1.011x**; 117.82 vs 117.85, **1.000x** (535 to 479 dispatches) | 6 per arm per length |
| S2 landed (decode's small kernels fused: G1 G2 G3 G4a), phase 8d, 0bb422a | `./tests/test_mellum_kernels`; `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64` after each decode change; `DS4_TEST_MELLUM_MODEL=... ./tests/test_mellum_fastpath.sh`; `./ds4_test --mellum-exact-prefill`, `--mellum-b0`, `--mellum-rewind`, `--mellum-attn-tiled` | kernel tests 832/832: each fused call word for word equal to the calls it replaces (q/k rope, K/V copies, MoE reduce + add, upstream's add + RMS norm); oracle PASS after each of the four decode commits, with decode, exact prefill and 64 greedy tokens byte-equal to the old engine; fast paths 45/45; exact prefill 0 of 98,304 words from decode; B0 logits equal decode (0 words); rewind and attention tiles OK. Decode dispatches per step 535 to 423, counted from the code (28 layers, four fewer each), not traced. **Speed not re-measured at the landed head:** Paul skipped the final run ("I think I want to skip a final speed run. I have a branch for SSD streaming that might impact it."). The screen of the same four groupings (`DS4_MELLUM_FUSE=15`, the S2 screen row above) measured 1.027x, 1.034x and 1.035x at 100, 2,056 and 8,195 tokens, n 6 per arm, against 385fbd2, whose product code is 89e4018's; that was a scratch build, not this code | 1 |
| oracle, fast paths, exact switch at 2d0756d | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64`, `make test-mellum-fastpath`, `./ds4_test --mellum-exact-prefill`, `--mellum-moe` | oracle PASS under the near-tie rule (1,100 and 2,200 tokens: top-1 64/65 each, the one change at a near-tie, mean KL 1.6e-3 and 1.3e-3; exact prefill, decode and 64 greedy tokens byte-equal); fast paths 45/45; `DS4_MELLUM_EXACT_PREFILL=1` 0 of 98,304 words from decode; `--mellum-moe` OK, the default 64-row MoE differs from `DS4_MELLUM_MOE_NAX=0` in 147,181 of 147,456 words, both within 4.3e-5 of the staged reference | 1 |
| ds4-server renders vs the template (steps 41-42) | `./ds4_test --server`; `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-server-ids` | 11 requests (`tests/gen_mellum_server_cases.py`): text and ids equal 2.1's template. Parse, round trip, remembered text, live tail, error turn and stream checks pass | 1 |
| ds4-server live (step 43), 462574c | `ds4-server -m $MODEL -c 16384 --port 8099`, then `curl /v1/chat/completions`: a system message, "How many lines does hello.py have? Use the bash tool.", one `bash` tool, seed 1, `max_tokens` 1024 | 1st request: one `bash` call, `{"command":"wc -l hello.py"}`, with `reasoning_content` split out, `finish_reason` "tool_calls", 274 tokens, 2.4 s. 2nd request (the turn sent back, plus the tool result "6 hello.py"): "The file hello.py has 6 lines.", "stop", 1.2 s, `cached_tokens` 480 of 499: the live KV was reused. Before `8accdcc` the same 2nd request missed the cache (token mismatch at 469). Streamed (seed 2, "List the files here"): 107 reasoning chunks, the first "Ok", then one `bash` `ls` call in `tool_calls`, "tool_calls", `[DONE]`. `/v1/models` lists `mellum-2.1` | 1 each |
| exact-prefill switch | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-exact-prefill` | 1,100-token prefill vs decode: 0 of 98,304 logit words differ with the switch, 98,303 without | 1 |
| batched + tiled prefill vs the old engine / exact decode, 64 positions | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64` (after step 27; rule: top-1 everywhere, max\|d\| <= 10, mean KL <= 0.01) | 1,100: max\|d logit\| 2.008, mean KL 2.1e-3, top-1 65/65; 2,200: 8.471, 1.7e-3, 65/65; short prompt and every decode byte-equal; 64/64 greedy equal; PASS | 1 |
| tiled prefill vs the exact path / the old engine, logits at 64 positions | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64` (after step 25) | 1,100 tokens: max\|d logit\| 0.649, top-1 65/65; 2,200: 0.874, top-1 65/65; short prompt and every decode still byte-equal; 64 of 64 greedy tokens equal; PASS | 1 |
| MoE driver on real layers, 64 rows tiled vs double | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-moe` | same experts selected; output within 1.0e-3 (16 rows exact: 3.4e-7, row 0 alone bit-equal) | 1 |
| the oracle script can fail: the same comparison against an old tree whose sliding attention splits keys over 4 simdgroups instead of 8 (same math, another summation order; scratch, not committed) | `tests/mellum_oracle_compare.sh ORACLE_BAD $MODEL 8` | exit 1, `mellum-oracle: FAILED`: all six cmp checks fail (max\|d logit\| 7.3e-3 at 29 tokens, 9.1e-2 at 1,100, 5.4e-1 at 2,200, argmax still 23), the greedy check fails at token 0; this engine's prefill-equals-decode checks still pass | 1 |
| Metal kernels vs double references | `make test-mellum-kernels` | 379 checks, 0 failures | 1 |
| MoE on real layers 0/13/27 vs double | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-moe` | same experts selected; weights 1.7e-7, output 3.4e-7 | 1 |
| session bytes allocated vs planned, Q8_0 GGUF | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-session` | equal at ctx 20,000 / 65,536 / 131,072 for chunks 1,024 and 512 (601,513,984 / 1,254,318,080 / 2,193,842,176 bytes at chunk 1,024) | 1 |
| real use: plain Pi 1.1.0 on AgentClinic easy Phase 1, published Q8_0 and MXFP4, M5 Max, High Power, engine 0364a95 | `tools/mellum_pi_run.py --model GGUF --out DIR` per run; `tools/mellum_pi_report.py docs/superpowers/research/2026-10-31-mellum-pi-real-use/q8-{1,2,3}` (and `mx-{1,2,3}`) | passed: Q8_0 2/3, MXFP4 0/3 (2 foreground-server stalls, 1 unterminated tool call, 1 context overflow). Median decode by context at start: Q8_0 124.4 / 114.1 / 109.3 t/s at 0-4K / 4-8K / 8-16K (n 25 / 23 / 7 requests); MXFP4 147.1 / 128.7 at 0-4K / 4-8K (n 22 / 6). Cold prefill (1,593-1,594 tokens) 3,832-3,850 t/s Q8_0, 3,953-4,074 MXFP4. [note](docs/superpowers/research/2026-10-31-mellum-pi-real-use.md) | 3 runs per quant |
| what a k-row forward costs (k 1-8; the last row's LM head only, a verify needs all k), contexts 2K / 8K / 32K, Q8_0 and MXFP4 | `tests/mellum_bk.sh OUT $MODEL 2048,8192,32768` (`./ds4_test --mellum-bk`: D C C D rounds per k, the first dropped); anatomy `tools/mellum_bk_anatomy.py`; distinct experts `tools/mellum_expert_union.py`; [note](docs/superpowers/research/2026-10-31-mellum-verify-cost.md) | k-row forward / one decode step, Q8_0: 1.256 / 1.910 / 3.173 at k 2 / 4 / 8 (2K), 1.414 / 2.132 / 3.754 (32K); MXFP4 1.313 / 2.026 / 3.446 (2K), 1.453 / 2.260 / 3.889 (32K). At 32K the decode step's wall time rises with the larger forwards run before it (unexplained); with each k in its own process or k descending, 8 rows are 3.522 / 3.510 (Q8_0) and 3.751 / 3.854 (MXFP4). Last-row logits equal decode's (0 words differ) at every point. Two consecutive tokens select 11.1-11.8 distinct experts per layer (of 8 for one), 15.3-17.2 for 4, 20.5-23.8 for 8 (five token files, 64 positions each) | 30 per arm per point |

### Oracle caps

The oracle's tolerance rule for the release prefill (`tests/mellum_logits_compare.py`) holds each traced
position to two caps beside the mean KL over positions: KL(old || new) <= 1.0
(`MELLUM_ORACLE_KLMAX`) and |d log p| of the old engine's top-1 token <= 1.0 (`MELLUM_ORACLE_DLOGP`).
The session row, which is the last position's row again, is held to the per-row rules and left out of the
mean. Every logit of both runs must be finite before anything is compared.

The caps come from the release prefill against the exact prefill (`DS4_MELLUM_EXACT_PREFILL=1`, byte-equal
to decode and to the old engine) at 64 traced positions of four held-out prompts. The positions are spread
evenly, so they fall in earlier assistant text as well as in the prompts. The prompts are written by
`tests/gen_mellum_heldout_tokens.py` and are unlike the oracle's own. The oracle's three files were
measured too. Revision a3a7962, M5 Max, one process at a time; per file, `tests/mellum_trace $MODEL FILE
prefill` with `DS4_MELLUM_TRACE_AT` set to the 64 positions, with and without the exact switch:

| file | ids | n | mean KL | worst KL | worst \|d log p(top-1)\| | max \|d logit\| | top-1 equal |
|---|---|---|---|---|---|---|---|
| `mellum_heldout_code` | 164 | 64 | 6.2e-4 | 2.2e-2 | 0.143 | 4.6 | 64/64 |
| `mellum_heldout_tool` (Pi's tools, a tool call) | 966 | 64 | 2.8e-3 | 9.6e-2 | 0.288 | 3.0 | 63/64 |
| `mellum_heldout_unicode` | 199 | 64 | 1.9e-2 | 0.467 | 0.456 | 13.2 | 60/64 |
| `mellum_heldout_long` (6,000 ids) | 6,000 | 64 | 3.2e-4 | 1.0e-2 | 0.174 | 4.2 | 64/64 |
| `mellum_oracle_short` | 29 | 29 | 0 | 0 | 0 | 0 | 29/29 |
| `mellum_oracle_1100` | 1,100 | 64 | 2.1e-3 | 4.2e-2 | 0.268 | 2.0 | 64/64 |
| `mellum_oracle_2200` | 2,200 | 64 | 1.7e-3 | 2.2e-2 | 0.161 | 8.5 | 64/64 |

Across the 413 rows the worst KL is 0.467 and the worst |d log p(top-1)| 0.456, both at one position of the
Unicode prompt. The caps are about twice those.

The large single-position departures are router flips, not drift that builds up gradually.
A scratch probe (not committed) dumped the router logits and the selected experts at four Unicode positions in
both paths. At every position whose hidden state departs from the exact path's, the layer where the
departure jumps 10-100x (from about 1e-3 relative error to 1e-2 to 1e-1) is the first layer whose top-8
experts differ. The exact path's gap between its 8th and 9th router logits there is 1.2e-4 (position 157,
layer 10), 4.2e-4 (195, layer 14) and 1.5e-2 (60, layer 6). After the first flip more follow, and a
flipped row reaches later rows through attention. A position with no flip stays near 3e-3 at every layer.
Any arithmetic that is not bit-identical will flip near-ties like these, so per-position caps can't tell
flips from a broken kernel. This has three consequences:

- The per-position caps catch only gross errors at single positions. The review's case (one position
  changed from 0.99/0.01 to 0.51/0.49, KL 0.618) passes them, because routing flips alone reach 0.467 on
  held-out text.
- The rest of the tolerance rule fails on held-out text. Top-1 equal at every position and mean KL <= 0.01
  hold on the oracle's three prompts. They do not hold on the Unicode prompt (4 top-1 changes, mean KL
  0.019) or the tool prompt (1 change). The rule is a regression gate set on those three files, not a
  bound on how far the release prefill may move any prompt.
- Per-position correctness rests on the exact path. (The second point is partly changed by the near-tie
  margin below: no top-1 change on held-out text is now out of rule, but the Unicode prompt's mean KL still
  exceeds 0.01.) Under `DS4_MELLUM_EXACT_PREFILL=1` the oracle checks
  long prefills byte for byte against the old engine and decode. A change that reorders work without
  changing arithmetic can be held to byte equality with the release path it replaces, which is a
  stronger gate than any cap.

**Near-tie margin (E1a, Paul's decision after Checkpoint 3.5).** A top-1 change now passes the
tolerance rule only where the reference's top two log-probabilities are within 0.75 nats
(`MELLUM_ORACLE_NEARTIE`). The per-position caps still apply to that row. The margin was set on the four
held-out files, at the same 64 traced positions each (n = 256), with the release prefill against the exact
one at rebased head 9986668. The current path changed the exact path's top-1 at five of those positions;
E1's tensor-tile experts (the scratch probe) changed it at four of the same five. Their margins were 0.002,
0.174, 0.369, 0.524 and 0.718 nats. The widest is Unicode position 60, where an earlier layer's router flip
spread forward. 0.75 is that maximum rounded up. It covers 31 of the 256 held-out positions and 92 of all
413 traced positions; the oracle's prose files have more low-margin positions. The oracle's own files were
not used to set it. There E1 changes the top-1 at 1,100 p1082 (margin 0.032) and 2,200 p1675 (0.156); the
current path changes none.

### Held-out log loss (E1b)

The per-position caps can't tell router flips from a broken kernel, so the quality check is log loss. For
each of the four held-out files, the trace driver recorded the mean negative log-likelihood of the file's
own next token, at every position: all of them for the code, tool and Unicode files, and every fourth for
the 6,000-id file (64 positions per run, 47 runs per path, n = 2,826 tokens). It was measured under three
paths:

- the exact prefill, `DS4_MELLUM_EXACT_PREFILL=1`;
- the current release prefill, with the simdgroup tiles;
- E1, the tensor-tile experts, measured on the scratch probe build with `DS4_QWEN4_MOE_MM_NAX=2`. Without
  that variable, the probe build gives the clean build's logits on all 413 traced rows.

Rebased head 9986668, one process at a time. The release paths are deterministic (two processes give
the same bytes), so the spread below is that of the paired per-token difference: its standard error
and a 2,000-sample bootstrap 95% interval.

| file | n | exact | current | E1 | E1 - current | se | 95% interval |
|---|---|---|---|---|---|---|---|
| code | 163 | 2.11981 | 2.11405 | 2.11403 | -0.00002 | 0.00032 | -0.0007 to +0.0006 |
| tool | 965 | 1.79379 | 1.78756 | 1.79014 | +0.00258 | 0.00827 | -0.0130 to +0.0198 |
| unicode | 198 | 2.94366 | 2.95096 | 2.95309 | +0.00212 | 0.00230 | -0.0023 to +0.0063 |
| long (every 4th) | 1,500 | 0.61689 | 0.61669 | 0.61641 | -0.00028 | 0.00065 | -0.0016 to +0.0010 |
| all | 2,826 | 1.26848 | 1.26642 | 1.26730 | +0.00088 | 0.00285 | -0.0045 to +0.0065 |

E1 is not worse than the current path beyond the spread: +0.0009 nats per token, with an interval
that spans zero. Both release paths are within 0.0021 nats per token of the exact path; the current path
is 0.0021 below it and E1 0.0012 below it.

### Metal 4 record

Three measurements of the Metal 4 tensor API (`DS4_METAL_DISABLE_METAL4=1` off), 2,200 tokens prefill,
`tests/mellum_trace`, M5 Max:

| build | design | on | off | on vs off |
|---|---|---|---|---|
| step-27 prototype (`MELLUM_Q8_BATCHED=1` switch, before 219ac0a) | runs 3-6 of 6, two rounds, 30 s rests, arms rotated per process | 1,973-2,065 t/s | 1,971-2,066 t/s (see below) | "nothing measurable" |
| 5aa742c (the outside review's round, its `automatic-benchmark-round-20261027`) | A/B/B/A, 6 runs per process, first dropped, 45 s between processes, AC, Automatic | median 2,051 | median 1,859 | +10.3%, n = 10 per arm |
| 99c259d (review fix R8) | the same design; `abba.sh` in the session scratchpad, arms on, off, off, on; AC, Automatic (`powermode 0`), no thermal or performance warning recorded before or after any process | median 1,962, range 1,821-2,047 | median 1,759, range 1,636-1,864 | +11.5%, n = 10 per arm |

The step-27 off arm did not run with Metal 4 off. Its loop set
`env="MELLUM_Q8_BATCHED=1 DS4_METAL_DISABLE_METAL4=1"` and ran `env $env ...` under zsh, which does not
split an unquoted parameter into words. That defined one variable, `MELLUM_Q8_BATCHED`, holding the
whole string, and never set `DS4_METAL_DISABLE_METAL4`. The "off" arm was a second run of the batched
tensor arm, which is why their ranges match. The prototype's code was otherwise the code 219ac0a committed,
so its "nothing measurable" is withdrawn. At 99c259d the arms really differ: the two processes of
each arm give byte-identical final logits, and on against off gives max |d logit| 1.3445 and
KL(on || off) 1.06e-4 with the same argmax, the same distance the review recorded at 5aa742c.
Metal 4 stays on. The absolute medians are about 4-5% below the review's. Runs within a process
slow down (the fourth process's first run was 1,942 against the first process's 2,006), and the A/B/B/A
order balances that drift between the arms but not its level.

### E1: routed Q8_0 experts on the tensor tiles (landed after the rebase)

`qwen4_moe_mm_nax` admits Q4_K, MXFP4, IQ2_XXS and Q2_K experts to the Metal 4 tensor tiles, not Mellum's Q8_0
experts. A scratch probe added Q8_0 raw words to the tensor kernels' weight fetch (`qwen4_load_raw16` and
`qwen4_dequant_raw16`), staged as half(d * q) exactly as the simdgroup tiles stage them, and admitted type
8 at level 2, the half-operand tiles. The operands keep their precision and only the summation order
changes, so this is a scheduling change. No precision variant (levels 3-6) was tried. The A/B/B/A runs were
on the same scratch build, with `DS4_QWEN4_MOE_MM_NAX=2` as arm B. AC power, Automatic, 45 s between
processes, no thermal or performance warning before or after any process:

| prompt | design | simdgroup tiles (now) | tensor tiles | per forward |
|---|---|---|---|---|
| `mellum_oracle_2200` (2,200 ids, prose) | 6 runs per process, first dropped, n = 10 per arm | median 1,933 t/s (1,772-1,941) | median 2,707 t/s (2,649-2,712) | +40%: 1.14 s to 0.81 s |
| `mellum_heldout_long` (6,000 ids, code) | 4 runs per process, first dropped, n = 6 per arm | median 1,505 t/s (1,444-1,604) | median 1,992 t/s (1,913-2,145) | +32%: 3.99 s to 3.01 s |

Only the routed expert GEMMs changed (the mid and down kernels), so the 0.33 s and 1.0 s saved are
expert time. At 2,200 tokens that is about half of the review's 53.8% MoE share, so the experts run about
twice as fast. The gain is smaller at 6,000 tokens, where attention takes a larger share. Each arm's two
processes gave byte-identical logits.

Numbers with the tensor tiles on:
- `ds4_test --mellum-moe`: every selection equals the double reference on random and real rows. Real
  rows are within 7.7e-5 of the staged reference, against the simdgroup tiles' 1.1e-4.
- `make test-mellum-fastpath`: 45 of 45 pass, including 32 of 32 greedy tokens equal to the exact path's.
- Against the exact prefill on the seven held-out and oracle files, mean KL is lower on four, equal on
  the short prompt, and slightly higher on the tool prompt (3.1e-3 against 2.8e-3) and the Unicode
  prompt (1.93e-2 against 1.89e-2). Worst KL is lower on four. The tool prompt's one top-1 change goes
  away. The Unicode prompt keeps its four router-flip changes (worst KL 0.480 against 0.467). Overall
  it is about as close to the exact path as the current tiles, not uniformly closer.
- `tests/mellum_oracle_compare.sh` **fails**. At 1,100 and 2,200 tokens the release prefill changes the top-1 at one of
  65 positions: p1082, where the exact path's top token has p = 0.139 and KL is 2.5e-4, and p1675,
  where it has p = 0.043 and KL is 3.8e-3. Means and maxima improve: 1,100 tokens mean KL 1.6e-3
  against 2.1e-3; 2,200 tokens 1.3e-3 against 1.7e-3, with max |d logit| 1.5 against 8.5. Exact
  prefill, decode and the 64 greedy tokens stay byte-equal.

At first E1 was reverted (`86fcd05`). The two changes are near-ties of the kind "Oracle caps" shows the
current path making on held-out prompts, but the oracle's top-1 rule is the gate, and changing it was
Paul's decision. After Checkpoint 3.5 he chose to revise the rule (E1a: a top-1 change passes only at a
near-tie of 0.75 nats) and to check quality with log loss (E1b: not worse beyond the spread). E1 then
landed on the rebased branch:

- `face2b0` adds the Q8_0 raw words.
- `db073ac` (red) and `816d062` add the Mellum-only opt-in. `mellum_graph_moe` sets
  `ds4_gpu_qwen4_moe_q8_tiles(2)` around its tiled calls and clears it after.
- `DS4_MELLUM_MOE_NAX=0` keeps the simdgroup tiles, as a diagnostic.
- `2d0756d` tests that other callers' Q8_0 experts, such as Flash Next's, keep the simdgroup tiles even under
  `DS4_QWEN4_MOE_MM_NAX=2`.

At the landed head prefill is +41% at 2,200 tokens and +29% at 6,000 (the Measured table). The patches
and runner are also in `misc/e1-q8-tensor-tiles/`, which is git-ignored.

### T2: tiled prefill attention (phase 8a, landed)

T1 (the Measured table) put E1's gain at +42% at 2,056 tokens, falling to +15% at 32,778. A scratch stage
profile (a wait after each stage group of each layer, `DS4_MELLUM_STAGE_PROFILE=1`, n = 1, on T1's prompts)
shows why: attention (f16 staging, the kernel, the ring commit) took 46.8% of the layer time at 2,056 tokens,
64.2% at 8,195 and 82.0% at 32,778 (29.2 s of attention, 5.4 s of FFN, 1.0 s of projections). The build has
no such split; the profile is in the probe patches. The sliding kernel ran one (head, row) per threadgroup, and
the full layers ran the decode-row kernels per row. So each row read its keys from memory on its own.

`kernel_mellum_attn_tiled` takes one KV head and 8 consecutive tokens per threadgroup. Simdgroup s holds
token t0 + s's eight query heads as one 8-row tile, so a tile has one query position and one mask. The eight
simdgroups share each staged tile of keys. A sliding layer reads the ring below pos0 with the tag check (a
stale slot makes the rows that need it NaN) and the staged rows above. A full layer reads its cache. The ring
commit stays after attention, and the 1,024 chunk cap is unchanged. Chunks of 32 rows or more use it on the
release path; decode, short chunks and `DS4_MELLUM_EXACT_PREFILL=1` keep the kernels they had.

The probe (`misc/t2-tiled-attention/`, git-ignored; its README has every variant and log) tried, at 8,195
tokens, attention profiled, n = 1 (the decode kernels: 2.71 s):

| variant | change | attention |
|---|---|---|
| 1 | f32 operands, 16 keys per tile, scores and rescale through threadgroup memory | 2.17 s |
| 1b (landed, `DS4_MELLUM_ATTN_TILED=1`; removed 2026-10-31) | 1, softmax on the fragments in registers | 2.10 s |
| 1c | 1b with K/V staged as half and widened per fragment, 16 or 32 keys | 4.22 s, 2.49 s |
| h16, h32 (h32 landed, the default) | precision: the scaled queries and the probabilities rounded to half, f32 accumulation, 16 or 32 keys | 1.44 s, 1.11 s |

1c shows that h32's speed comes from the half-input matrix operations, not from staging K and V as half
(which is exact; they are f16 already). Only 1b and h32 were gated.

Kernel tests (`test_mellum_kernels`): every f32-tile row is within 6.5e-6 of the double reference (tolerance
1e-5), on sliding sessions of 26 to 2,100 keys in three chunkings and on full-layer rows to 4,100 keys. The
half tiles are within 3.1e-4 of a reference that stages as the kernel does, and 7.5e-4 to 9.1e-4 from the f32
one. The staging explains part of the distance; the rest is not explained (rounding the scores, the partial
products or the tile products to half, or flushing subnormal probabilities, each made it larger), so the
tolerance is 5e-4, the worst seen with margin. Both pass the stale-tag control (63 of 64 rows NaN).

Held-out log loss by E1b's method (four files, n = 2,826 tokens, 47 trace runs per path, one process at a
time; variant minus the decode kernels per token, se and a 2,000-sample bootstrap 95% interval). At 81b5f61
each level's per-token file is byte-identical to the probe's, and level 0 reproduces E1b's E1 column:

| file | n | decode kernels | f32 tiles | half tiles | f32 - decode | half - decode |
|---|---|---|---|---|---|---|
| code | 163 | 2.11403 | 2.12039 | 2.11888 | +0.00635 [-0.0021, +0.0204] | +0.00484 [-0.0005, +0.0118] |
| tool | 965 | 1.79014 | 1.77991 | 1.78278 | -0.01023 [-0.0276, +0.0064] | -0.00736 [-0.0226, +0.0076] |
| unicode | 198 | 2.95309 | 2.95005 | 2.93287 | -0.00304 [-0.0109, +0.0049] | -0.02022 [-0.0584, +0.0135] |
| long (every 4th) | 1,500 | 0.61641 | 0.61530 | 0.61595 | -0.00111 [-0.0023, +0.0002] | -0.00046 [-0.0016, +0.0007] |
| all | 2,826 | 1.26730 | 1.26337 | 1.26341 | -0.00393, se 0.00300 [-0.0098, +0.0016] | -0.00389, se 0.00291 [-0.0103, +0.0014] |

Neither is worse than the decode kernels beyond the spread. The A/B/B/A at the landed head (the Measured
table) gives the half tiles 1.37x at 2,056 tokens, 1.65x at 8,195 and 2.03x at 32,778; the probe's own rounds
gave 1.37x, 1.65x and 2.05x. A 32,778-token summary request (`./ds4 -c 34816 --temp 0 -n 700 --prompt-file
.../prompt_32768.txt`, n = 1) prefilled at 1,873 t/s and answered with three sentences drawn from the text.

### D: decode research (phase 8b, Checkpoint 7; nothing landed)

Paul, 2026-10-28: "Proceed with next step, I believe it is decode research". D1 and D2 measured decode at
ea0f37b. Commands, n and raw logs are in the Measured table.

**D1.** Each decode step is one command buffer of 535 dispatches, then one wait, the logits read and the CPU's
sampling. Wall time is the CLI's generation rate. GPU time and the gap between steps come from
`DS4_METAL_CB_TIMES` (255 or 256 decode buffers per length). Bandwidth is bytes per step, at the middle of
the 256 generated tokens, over the median GPU time:

| prompt tokens | t/s | wall per step | GPU per step | GPU idle between steps | encode | bytes per step | GB/s (of 575.4) | read floor |
|---|---|---|---|---|---|---|---|---|
| 100 | 133.15 | 7.510 ms | 6.908 ms (92.1%) | 0.591 ms | 0.3 ms | 2,376.5 MB | 344.0 (59.8%) | 4.130 ms, 242 t/s |
| 2,056 | 119.08 | 8.398 ms | 7.920 ms (93.2%) | 0.598 ms | 0.3 ms | 2,438.8 MB | 307.9 (53.5%) | 4.238 ms, 236 t/s |
| 8,195 | 116.67 | 8.571 ms | 8.314 ms (92.4%) | 0.620 ms | 0.3 ms | 2,526.8 MB | 303.9 (52.8%) | 4.391 ms, 228 t/s |
| 32,778 | 98.15 | 10.188 ms | 9.495 ms (94.0%) | 0.626 ms | 0.3 ms | 2,879.3 MB | 303.2 (52.7%) | 5.004 ms, 200 t/s |

So decode is GPU-bound: launch, encode and the wait cost the 0.6 ms of GPU idle per step, 7-8%.

The kernel split comes from the encoder timeline, `anatomy.txt`. Each dispatch group there is its own pass,
so durations are upper bounds and add up to about 12% more than the buffer's GPU span. At 2,056 tokens, per
step:

| kernel (threadgroups) | per step | each | data each | GB/s |
|---|---|---|---|---|
| experts gate + up, `moe_mid` (112x8) | 28 | 78.2 us | 35.1 MB | 449 |
| experts down, `moe_down` (288x8) | 28 | 39.6 us | 17.5 MB | 443 |
| Q projection (2,048) | 28 | 26.2 us | 10.0 MB | 383 |
| output projection (1,152) | 28 | 21.9 us | 10.0 MB | 458 |
| K and V projections (256) | 56 | 17.0 us | 1.25 MB | 74 |
| output head (49,152) | 1 | 414.3 us | 240.6 MB | 581 |
| F32 router, `multi_gemv` (8) | 28 | 15.1 us | 0.59 MB | 39 |
| sliding attention, `mellum_attn_swa` (32) | 21 | 52.9 us | 2.1 MB | 40 |
| full attention, decode + merge (64x4, 32) | 7 | 38.1 us | 4.5 MB | 117 |
| norms, q/k norm and RoPE, adds, f16 copies, ring commit, MoE reduce, top-k | 361 | 2.0-7.0 us | KB | - |

At 32,778 tokens, full attention is 254.8 + 22.0 us per layer (the decode kernel at 264 GB/s), and attention as a whole takes 28%
of the summed durations, up from 7% at 100 tokens and 16% at 2,056.

**D2, B0.** A two-row verify costs 1.318 decode steps at 100 ids of context, 1.290 at 2,048, 1.315 at 8,192
and 1.439 at 32,768. That includes the row-0 head in its own command buffer, 0.61-0.68 ms. In the same
buffer it would be about the head's 0.41 ms, giving about 1.26 at 2,048. Every run's two logit rows equal the
decode steps' bytes, so a verify accepts exactly what greedy decode would. A verify that rejects its draft
at depth one needs no restore: the rejected row's ring slot is outside the next position's window, and the
next step overwrites it. Phase 8's B0 budget is a draft-and-verify cycle of at most 1.28 steps for 1.5x at
92% acceptance, and 1.07 for 1.8x. The two-row verify alone is at or over 1.28 at every context, so B0 fails
as the code stands. Not measured: why two rows cost 1.29 steps. The two-row projections run each row as its
own threadgroups (the exact rows kernel), and the experts run per token, up to 16 distinct experts per
layer.

**D3, candidates ranked by ceiling.** A ceiling is the per-step saving if the kernels reached their read
floor at 575.4 GB/s (timeline durations, so upper bounds), taken out of the CLI's wall step. All of them
together are capped by the read floor: 1.98x at 2,056 tokens.

| rank | candidate | ceiling at 2,056 tokens (others) | byte-equal decode? |
|---|---|---|---|
| 1 | sliding decode attention, 32 threadgroups at 40 GB/s: more parallel keys | 1.034 ms, **1.140x** (100: 1.043x; 8,195: 1.137x; 32,778: 1.123x) | only as a reordering-free change: screen S1, 1.038x measured; a split over more partitions changes the merge and needs a stated tolerance |
| 2 | fuse the small kernels (norms, RoPE, adds, copies, commit, reduce, top-k: 1.24 ms) into neighbours | 1.239 ms, **1.173x** if all vanished | yes, if each value's arithmetic is kept |
| 3 | K and V projections into the Q dispatch (and Q, output projections to the floor) | 0.828 ms, **1.109x** (all four: 1.197 ms, 1.166x) | yes: each output row is computed by the same kernel |
| 4 | experts at the floor (449 and 443 GB/s now) | 0.738 ms, **1.096x** | yes, if the per-row order is kept; already the closest to peak |
| 5 | GPU idle between steps (logits read, sampling, encode) | 0.598 ms, **1.077x** | yes; greedy needs an on-GPU argmax and the next step encoded first; sampling needs it on the GPU too |
| 6 | full attention at long context (264 GB/s at 32,778) | 32,778: 1.118 ms, **1.123x**; 2,056: 1.026x | the decode kernel is upstream's; a change is not byte-equal |
| 7 | F32 router with 8 threadgroups (39 GB/s) | 0.394 ms, **1.049x** (+0.196 ms if top-k fuses in) | yes |
| 8 | prompt-lookup drafts, depth one, at B0's 1.290 | tool 1.074x, long C source 1.145x, code 0.992x, Unicode 1.021x, prose 0.912x (last 1-3 ids); last 3 ids only: 1.087x, 1.112x, 1.019x, 1.023x, 1.001x | yes (B0's verify equals decode) |
| 9 | a cheaper two-row verify (B0 at 1.05 instead of 1.29) | prompt lookup then 1.227x (tool), 1.368x (long), 1.092x (code), 1.036x (prose); MTP below | depends on the kernels |
| 10 | an MTP head | at 92% acceptance and depth one, with only the head's BF16 output read as draft cost (453 MB, 0.094 step): 1.388x at B0's 1.29, 1.252x at 32,768's 1.44, 1.679x at 1.05. The GGUF has no MTP tensors (all 339 counted), so a head comes from outside | yes for the verify |
| 11 | the output head | 0: already at 581 GB/s | - |

The prompt-lookup figures are teacher-forced on the files' own text, not the model's greedy output, and use
ids per step / (share of steps verifying x B0 + the rest). Drafting every step breaks even at 29% acceptance
at B0's 1.29. Deeper drafts need a (k + 1)-row verify, which was not measured.

**Screen S1** (`misc/d-decode/s1-swa-prefetch.diff`, scratch, not landed). Each simdgroup of the sliding
decode kernel loads its next P keys before folding them in, in the same order with the same arithmetic.
Timeline at 2,056 tokens, n = 1 process each: P = 2, 3, 4 gave 39.7, 38.4 and 39.0 us against 52.9. P = 8 and
16 gave 64.8 and 65.0 us, more registers held. A first version whose load loop could break out early gave
68.1, 96.3 and 69.0 us for P = 4, 8, 16; those logs were overwritten by the predicated rerun, and the
numbers are from the session's output. P = 2 passed:
- the kernel tests;
- the oracle: decode byte-equal to the old engine, and 64 equal greedy tokens;
- the CLI A/B/B/A: 1.004x at 100 tokens, 1.038x at 2,056, 1.030x at 8,195, with identical generated text.

S1 takes about a quarter of candidate 1's 1.14x ceiling at 2,056 tokens. The rest would need more partitions
per head, which changes the merge.

### README checks (phase 9, 2026-10-28)

`README.md` is now the Mellum guide, with upstream's text below a divider (`c692779`, `ebc93df`, `942c6e9`,
`5c84246`). Each of its command and config blocks was run as written on the M5 Max, at `49bf9eb`; the prefill
runner was rerun at `ba83406`, which changes only that runner. Raw logs are in
`mellum-evidence-2026-10-31:tests/mellum_measurements/readme_checks/`.

The checks ran under a scratch home: `HOME` and `PI_CODING_AGENT_DIR` were scratch directories and `HF_HOME`
was the real cache. So `~/models` and `~/.pi/agent` were scratch paths, and Paul's were never touched. His
`~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf` has sha256 `ea1e4ec7…`, not the published file, so the
download as written would have replaced it. A local clone of `mellum` stood in for the GitHub clone; the repository was published as
`pauleveritt/ds4-mellum` on 2026-10-30.

| check | command (as in the README) | result | n |
|---|---|---|---|
| download | `hf download JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf --local-dir ~/models` | 3.1 s from the cache; 12,925,417,760 bytes; sha256 `69e0ac02a16586c2c9e991f1e3b8a1eb403882e7ab9e69ea01eb23dbbbc6841f` | 1 |
| build | `cd ds4-mellum`, `make` | exit 0, 0 warnings | 1 |
| smoke test | `./ds4 -m ~/models/… -c 32768 -p "Write a Python function that checks whether a string is a palindrome."` | reasoning in plain text, then a function, an example and an explanation; 35 prompt tokens (its timing line is a check, not a measurement) | 1 |
| server | `./ds4-server -m ~/models/… -c 32768 --port 8099`; `curl -s http://127.0.0.1:8099/v1/models` | listening; context buffers 790.54 MiB; `mellum-2.1`, `context_length` 32768 | 1 |
| Pi entry | the README's `models.json`; `pi --list-models ds4` | `mellum-2.1`, context 32.8K, max-out 16.4K, thinking "no" | 1 |
| Pi tasks | `echo "<task>" \| pi --model ds4/mellum-2.1` (print mode; the README's command with the task piped in, not the interactive UI) in a scratch project | 3 of 3 done and checked by hand: `is_palindrome` and 5 tests, the suite run, 7 OK (11.0 s); `mean([])` fixed to 0.0 with a test, by `read` then `edit` (74.9 s); `unittest -v` run with `bash` and its 8 tests listed (4.9 s) | 3 |
| reasoning in Pi | `pi --model ds4/mellum-2.1 --mode json "Read stats.py and tell me what median([4, 1, 3, 2]) returns." < /dev/null` | `thinking` blocks (from `reasoning_content`), a `read` tool call, more thinking, then the answer, 2.5 (correct), without `"reasoning": true` in the entry. Print mode shows only the final text. Without `< /dev/null` and with stdin not a terminal, Pi waits on stdin and sends nothing (10 minutes, stopped) | 1 |
| sampling and context take effect | the server command plus `--trace FILE`, one task | both requests: temperature 0.600, top_k 20, top_p 0.950 (raw JSON: `"temperature":0.6`, `"top_p":0.95`, `"top_k":20`), `max_tokens` 16384 (half of `contextWindow`), think_mode high; the second request reused 1,877 of 1,896 tokens. The trace's `min_p: 0.050` is the request struct's default; the sampler starts from the engine's defaults (`server_sampling_for_request`), where Mellum's `min_p` is 0.0 | 1 task, 2 requests |
| memory at 32K | `/usr/bin/time -l ./ds4 -m MODEL -c 32768 -p "…" -n 64` | the engine's plan: KV 0.52 GiB + buffers 0.25 GiB + resident model 12.03 GiB = 12.81 GiB; session allocated 828,944,640 bytes (equal to the plan); `time`: peak memory footprint 1,393,181,584 bytes, maximum resident set 75,022,336 (neither counts the mapped weights) | 1 |
| measure on your own Mac | `ARMS=AA tests/mellum_cli_prefill_abba.sh ~/ds4-runs/prefill ~/models/… 2048`; `ARMS=AA tests/mellum_cli_decode_abba.sh ~/ds4-runs/decode ~/models/… tests/mellum_prompts 2048` | both run and print a summary. Prefill 3,744 t/s (3,727-3,753), decode 123.30 t/s (122.78-124.47), as checks of the commands, not the README's numbers. The first prefill run, before `ba83406`, measured 2,447 t/s, because arm A still took T1's `DS4_MELLUM_MOE_NAX=0` (red `fff1b44`, fix `ba83406`) | 6 per runner |

Waiting then: the clone command (published 2026-10-30); RM2, the speed tables, until
Paul's SSD-streaming decision.

Overlap with another session's work (Paul's SSD-streaming work in the Codex worktree `mellum-ssd-replay`; times
local, from that worktree's logs and this session's file times; the GPU was not monitored continuously):
- the memory measurement (11:19:03-11:19:06) ran before that session's collect run (11:20:26-11:22:24);
- the first smoke-test attempt was refused by ds4 because the collect run's model process
  (`tools/mellum_generate collect`) was running. The check waited until no Mellum model process was left
  (11:22:16), then ran the smoke test (about 11:22:20-11:22:43), during the collect script's last seconds after
  its model process had exited. Its timing line (generation 88.89 t/s, against 131.12 for the same prompt at
  11:19) is not reported;
- `ds4-server` (11:22:49-11:35:31) was up, idle, while that session's validation ran (11:24:54-11:25:06); no
  timed Pi task overlapped it (task 3 ended at 11:24:52; the JSON run started at 11:25:00 but sent no request
  until it was stopped);
- during the runner checks (11:36:27-11:47:36) that session wrote notes and a source-identity check
  (11:38:10, file hashes); no model process of its was seen. The runner numbers above are checks of the
  commands, not results.

## Rebase onto ee7e840 (review step RB, 2026-10-27)

`mellum` was rebased from upstream `5379c1c` onto antirez `ee7e840`, upstream `main` that day. That adds two
upstream commits off Mellum's path: `a101246`, MXFP4 downloads and a shard-merge script, and `ee7e840`, CUDA
SSD prefetch made opt-in inside a `!__APPLE__` guard in `ds4.c`. The rebase had no conflicts. The rebased
tree differs from the old head by exactly those two commits' changed lines. The full gate and the CPU build
pass, and the oracle passes, with exact prefill, decode and the 64 greedy tokens byte-equal.

The SHAs above this section were written before the rebase. The tag `mellum-pre-ee7e840` (the old head,
`afaf7c8`) keeps them resolving, and measurements cited at an old SHA were taken on that tree. The commits
cited here map to the rebased branch as follows (all 108 commits kept their order and subjects):

<details><summary>old SHA to new SHA</summary>

| before | after | subject |
|---|---|---|
| `15824be` | `0bb2432` | MELLUM.md: the Mellum port's notes, with the baseline at 5379c1c |
| `57278db` | `4720ffe` | MELLUM.md: dependency audit of the old Mellum port against 5379c1c |
| `e8ba0ac` | `83e39b6` | mellum: recognize general.architecture = mellum, refuse it by name |
| `5109a6d` | `0428a4c` | mellum: validate the hyperparameters, each refusal naming its key |
| `34b5d44` | `06b13f8` | mellum: bind the tensors by name and check their Q8_0 layout |
| `f4d1ef1` | `5b6758e` | mellum: memory plan and session allocation; the forward entries refuse |
| `e398212` | `b5f0530` | mellum: generated Unicode classes for the pre-tokenizer, with the generator |
| `5e4884d` | `fd46752` | test: Mellum 2.1 tokenizer cases from its own tokenizer, with the generator |
| `a7f0b4f` | `70fb4d5` | mellum: the mellum2 pre-tokenizer |
| `79922e5` | `5cdf14c` | mellum: special tokens classed, corpus markers are content |
| `8021751` | `9710f50` | test: Mellum 2.1 chat ids from its own template, with the generator |
| `20a696c` | `835c6b0` | mellum: its own system turn and assistant prefix |
| `dd0d997` | `1d9b628` | mellum: sampling defaults from the engine, read by the CLI, agent and server |
| `a74bafe` | `d3892ee` | mellum: session save and load refuse Mellum by name |
| `f58b776` | `5c3b117` | mellum: q/k norm and RoPE kernel, YaRN on full layers only |
| `bb2a2f3` | `d1c144d` | mellum: ring commit and sliding-window attention kernels |
| `f0332e3` | `9df140f` | mellum: ungated full attention on upstream's qwen4 decode kernels |
| `2712e56` | `c6cb6fb` | mellum: MoE, top-8 of 64 routed experts, no shared expert |
| `b3f9897` | `68f818a` | mellum: the forward pass, behind DS4_MELLUM_TRACE, and its trace driver |
| `f86c6b4` | `b7e3a9f` | test: the oracle's three token files, from the model's own template |
| `46d7d79` | `b547dda` | mellum: oracle comparison script; the forward pass matches the old engine |
| `48296f7` | `ac8cf98` | mellum: the Unicode table ends with one newline |
| `51e898c` | `f83b888` | MELLUM.md: the oracle script, shown failing against a perturbed old tree |
| `9d2ff02` | `b62ebc9` | test: Mellum prefill equals decode, op by op |
| `a27783e` | `e1885c7` | mellum: lift the forward-pass refusal; the CLI runs Mellum chats |
| `322389e` | `17de1a4` | mellum: compaction resumes an open reasoning block with <think> |
| `63c647f` | `36c8ea9` | mellum: session rewind keeps both caches one token back |
| `e4a6595` | `f975919` | mellum: prefill experts as tiled GEMMs from 32 rows (2.4x prefill) |
| `219ac0a` | `66c40c1` | mellum: Q8_0 prefill projections batched from 32 rows (another 2.4x) |
| `c5c4e57` | `f9c1229` | test: the ds4 CLI with Mellum (red: the -n cap is not reported) |
| `a6c2eb0` | `0d5f2e7` | ds4: the CLI says when a turn stops at the -n cap |
| `907094d` | `4a9f5ee` | MELLUM.md: what --nothink does with Mellum 2.1 |
| `be82f23` | `2f0217f` | test: the CLI's speed line names Mellum (red: it says Qwen3.8) |
| `b9dbf42` | `d356669` | mellum: the CLI's speed line and session errors name Mellum |
| `36fe842` | `2642d76` | MELLUM.md: step 29 speed, and the commit list through phase 4 |
| `5aa742c` | `6f7e22a` | test: a long Mellum prefill equals decode under DS4_MELLUM_EXACT_PREFILL=1 (red) |
| `4bdb7b5` | `73544c9` | mellum: DS4_MELLUM_EXACT_PREFILL=1 keeps long prefills exact (a diagnostic) |
| `9e77c2d` | `24e0ec1` | test: the oracle checks every prefill both ways |
| `4c5ef02` | `6b377f2` | test: a Mellum tool message renders <tool_response> as tokens 31 and 32 (red) |
| `98a5bc6` | `d1a98ca` | mellum: a tool message's <tool_response> wrapper renders as tokens 31 and 32 |
| `a821dd7` | `006983d` | test: Mellum's text-only tool observation renders the same tokens 31 and 32 (red) |
| `cdc5b10` | `0df1620` | mellum: the multimodal tool path uses the same <tool_response> tags |
| `7514d1f` | `81050b3` | ds4_chat_append_tool_results: append one assistant turn's n tool results |
| `11684de` | `19054eb` | test: Mellum's two tool results are one user turn (red) |
| `ba1c0e0` | `05cddb2` | mellum: one turn's tool results render as one user turn |
| `b2134ce` | `41bfdd8` | ds4_json.h: the server's JSON string readers move to a shared header |
| `d5cb032` | `0bcdc29` | ds4_json.h: the number readers and value skippers join the shared header |
| `a81de8d` | `9d70d86` | ds4-agent: AGENT_TOOL_SYNTAX_HERMES, Mellum's tool-call syntax, not yet selected |
| `534ef56` | `e1c8a6f` | test: the Hermes parser's call object, string arguments and errors (red) |
| `e2958cc` | `29b06f3` | test: the Hermes markers test expects the call at the final byte (still red) |
| `a440f79` | `a5caaf4` | ds4-agent: the Hermes parser, call object and string arguments |
| `6ddb1b2` | `305fdf2` | test: the Hermes parser keeps numbers, literals and nested values raw (red) |
| `42e4717` | `b1c667d` | ds4-agent: the Hermes parser keeps numbers, literals and nested values raw |
| `91f5be0` | `0409aab` | test: the Hermes parser reads what json.loads reads, and refuses what it refuses (red) |
| `53b9441` | `c579913` | ds4-agent: the Hermes parser reads the call object as json.loads does |
| `aaebdde` | `9eff81e` | test: prose around Hermes calls neither ends the turn early nor drops a call (red) |
| `d2d2b32` | `dafd5a3` | ds4-agent: the Hermes parser tolerates prose around calls |
| `9b6aa77` | `cf63c43` | test: Mellum's tool-bearing renders, Pi's tools in the system turn (red) |
| `c98441c` | `2eda52f` | mellum: a system message renders special-token text as its tokens |
| `a59b281` | `892bd7b` | test: the CLI's Mellum system prompt renders as a system message does (red) |
| `0cc60c9` | `88cb9ac` | mellum: the CLI's system prompt uses the same tokenizer as a system message |
| `80ae163` | `0bf4fa5` | test: ds4-agent's Hermes system turn equals the template's (red) |
| `5482d43` | `d44f1fa` | ds4-agent: the Hermes system turn, Pi's tools block as 2.1's template writes it |
| `24e2ce5` | `e6ee4fd` | test: Hermes think state, turn end, stop classes and the [upto] forcer (red) |
| `285cfdf` | `295db94` | ds4-agent: Hermes think state, turn end and stop classes |
| `3ad12c3` | `24a5e9b` | test: ds4-agent selects the Hermes syntax for a Mellum engine (red) |
| `5ffbfe3` | `a8bd1d0` | ds4-agent: a Mellum engine gets the Hermes syntax, ahead of the qwen4 check |
| `e38b18d` | `a5c6401` | ds4-agent: say where Mellum's earlier reasoning diverges from the template |
| `7c44f0c` | `4189df7` | ds4-agent: drop an unchecked claim about the Qwen templates from that comment |
| `a6ce058` | `822749e` | test: move the oracle's logits comparison into a script the tests can run |
| `c99ac87` | `4485d0d` | test: the oracle comparison rejects NaN and Inf in every row (red) |
| `fc16a75` | `9af4e3f` | test: the oracle comparison rejects NaN and Inf before anything else |
| `807c7be` | `72f6a5c` | test: the trace driver checks every listed position's logits finite (red) |
| `28fdbd5` | `322cb52` | test: the trace driver checks every listed position's logits finite |
| `381d402` | `95dd91f` | test: the oracle comparison caps each position, and counts the session row once (red) |
| `a3a7962` | `044594e` | test: four held-out token files for setting the oracle's per-position caps |
| `fff47fc` | `2ffd0ba` | test: the oracle caps each position's KL and top-1 log-probability change |
| `62dce64` | `d9fec50` | MELLUM.md: the oracle's per-position caps, where they come from, and router flips |
| `c95e0e8` | `4ba0969` | test: an imported history closes each assistant turn once for Mellum (red) |
| `5d75ebe` | `77b6ac8` | ds4_prompt_prefix: no second <\|im_end\|> after an imported Mellum assistant turn |
| `890da18` | `f1c7f75` | MELLUM.md: the prompt-prefix close fix, and Flash Next's probable double close |
| `6d60625` | `1d8af56` | test: an imported Mellum history drops earlier reasoning, as the template does (red) |
| `544d3d5` | `c8f8198` | ds4_prompt_prefix: drop the reasoning of imported Mellum assistant turns |
| `a468bfb` | `ed63238` | MELLUM.md: rendered histories drop earlier reasoning; live transcripts keep it |
| `98f40bf` | `fbd4ec4` | test: a deep Mellum rewind gives a fresh sync's logits, not the straight run's |
| `9bbf9fe` | `181e3a7` | ds4: say what a deep Mellum rewind's replay gives |
| `6eeade3` | `910fc7b` | test: max thinking prefixes no DeepSeek preamble for Mellum (red) |
| `c479351` | `aef990e` | ds4: no DeepSeek max-effort preamble in a Mellum prompt |
| `4e5fbf7` | `c9d63e0` | test: Mellum metadata refuses head and width shapes the kernels refuse (red) |
| `119001b` | `7eef70b` | ds4: Mellum metadata refuses the head and width shapes its kernels refuse |
| `8fbd9b3` | `5876d68` | MELLUM.md: list the -n cap message and the skipped max-effort preamble as upstream changes |
| `6bd685e` | `5374e1c` | test: the trace driver syncs a prefix and decodes the rest, and decodes greedy tokens |
| `033dbf0` | `225d981` | test: the release prefill's fast paths at chunk and window edges, with the model |
| `a8e5b4b` | `fe15699` | MELLUM.md: fast-path coverage at chunk and window edges, and the deep-rewind contract |
| `767a29a` | `0b7f707` | ds4: the Mellum trace writes each layer's MoE input row |
| `11ed730` | `efd8981` | test: the Mellum router and experts on real activations against the double reference |
| `99c259d` | `54b41f4` | MELLUM.md: the router and experts on real activations, and what half staging costs there |
| `f16a1dd` | `53cfc5e` | MELLUM.md: reconcile the Metal 4 record (R8): on is +11.5% at 99c259d |
| `86fcd05` | `d1be955` | MELLUM.md: experiment E1, routed Q8_0 experts on the tensor tiles, measured and reverted |
| `8e7b158` | `59fa06c` | MELLUM.md: E1's held-out drift is about the same as now, not lower on five files |

</details>

## Backends covered

Metal only, on Apple Silicon. CUDA, ROCm, the CPU reference path and
tensor-parallel/distributed inference are not built or tested for Mellum here,
and Mellum is not expected to run on them until someone ports and checks it.

## Baseline at 5379c1c (2026-10-26)

Machine: Apple M5 Max, 128 GiB (`sysctl -n hw.memsize machdep.cpu.brand_string`),
macOS SDK 27.0 (`xcrun --show-sdk-version`).

`make -j8` builds `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval` and `ds4-agent`
with no warnings (10.9 s wall). The model-free checks, each run once (n = 1),
all exit 0:

| command | result |
|---|---|
| `./ds4-eval --validate-cases` | 92 core and 50 hard cases validated |
| `./ds4-eval --self-test-extractors` | passed |
| `./ds4_agent_test` | ok |
| `./ds4_test --server` | ok |
| `./ds4_test --metal-kernels` | ok |
| `./tests/test_layer_pack` | 97/97 |
| `./tests/test_engine_mgpu_placement` | 114/114 |
| `./tests/test_gpu_args` | passed |
| `./tests/test_gpu_args_cli.sh` | 90 passed, 0 failed |
| `./tests/test_prompt_prefix` | passed |
| `./tests/test_sampling` | OK |
| `./tests/test_deepseek4_vision_image` | exit 0 |
| `./tests/test_image_decode` | exit 0 |

The rest of `make test` (`./ds4_test` with no arguments) needs a DeepSeek V4
Flash model and was not run.

## Dependency audit (step 2, 2026-10-26)

The old port lives on three local-only branches, fetched by path into
`refs/old/*` (no remote added):

```sh
git fetch --no-tags <ds4-engine>/.git/worktrees/p42-2d-mellum-fit/modules/external/ds4 \
  'refs/heads/mellum21-*:refs/old/mellum21-*'
```

`refs/old/mellum21-on-upstream` (`7e8d9e6`) is antirez `0aaea5a` + 40 fork
commits (Qwen3.5 dense, DFlash, verify, host extensions, ending at `ebdcb0f`)
+ 21 Mellum engine commits; `refs/old/mellum21-stage1` (`f5cc405`) adds 22
agent/host commits; `refs/old/mellum21-minimal` (`e7b77ed`) is the engine
squashed onto `ebdcb0f`. So the Mellum work is `git diff ebdcb0f
refs/old/mellum21-stage1`, and everything it needs is either in that diff, in
upstream `5379c1c`, or in the fork commits `0aaea5a..ebdcb0f`. The audit lists
the third group. Method: every identifier in the added lines of that diff,
checked against trees of `5379c1c` and `ebdcb0f` (found in the fork, absent
upstream), then each hit read in context.

Addendum (found at step 4): the old port also relied on the fork raising
`DS4_MAX_HEAD_KV` from 2 to 4 (`6a015c6`, for Qwen3.5's 27B). Mellum has 4 KV
heads; upstream reads the limit nowhere else, so `5109a6d` raises it.

### Fork-only Metal code to re-home under Mellum names

| old name (where, in `refs/old/mellum21-stage1`) | from fork commit | Mellum use | step |
|---|---|---|---|
| `kernel_dflash_ring_write` (`metal/dflash.metal:51-82`), host `ds4_gpu_dflash_ring_write_tensor` (`ds4_metal.m:51309`) | `8f3af76` | commits a sliding layer's chunk to its f16 K/V ring and tags each slot with its position, after attention has read the ring (`ds4.c:60726`) | 16 |
| `kernel_dflash_head_norm_rope` (`metal/dflash.metal:86-145`), host `dflash_head_norm_rope_dispatch` (`ds4_metal.m:51333`) | `e2d05e2`; Mellum's `ba07f9e` changed `pad0` to `mscale` | per-head q/k RMS norm + NeoX RoPE; `ds4_gpu_mellum_qk_norm_rope_tensor` (`ds4_metal.m:51381`) fills the 64 per-pair frequencies (YaRN blend on full layers) and the mscale in double | 15 |
| `dflash_dispatch` (`ds4_metal.m:51269`), a named-pipeline dispatch helper | `e2d05e2` | dispatches `kernel_mellum_attn_swa` (`ds4_metal.m:51476`) | 16 |
| the `DS4_METAL_DFLASH_SOURCE` entry in `ds4_gpu_full_source` (`ds4_metal.m:4804`) | `e2d05e2` | `metal/mellum.metal` is appended after it; upstream's list is at `ds4_metal.m:4754-4800` | 15 |

`kernel_mellum_attn_swa` itself (`metal/mellum.metal`) is already Mellum's
own and self-contained: it uses no identifier from `dflash.metal` or
`qwen4.metal`.

Upstream cannot serve the q/k norm + RoPE: its `ds4_gpu_qwen4_attn_prep_tensor`
(`ds4_gpu.h:3556`) takes rotary frequencies from one process-wide table of at
most 32 pairs (`ds4_gpu_qwen4_set_rope`, `ds4_metal.m:49051-49063`), while
Mellum rotates all 128 dims (64 pairs) with two bases, `freq_base_swa` on
sliding layers and `freq_base` + YaRN on full ones.

### Host extensions to replace

- `ds4_engine_sampling_defaults` (fork `8b70f74`, Mellum values added in
  `3e31173`): the old CLI (`ds4_cli.c:545`) and agent read it. Upstream has no
  per-engine table: `cli_apply_model_sampling_defaults` (`ds4_cli.c:528-536`)
  and `agent_apply_model_sampling_defaults` (`ds4_agent.c:1006-1014`) set
  T 1.0 / top_p 0.95 / min_p 0 for GLM-DSA only; every other model, Mellum
  included, gets `DS4_DEFAULT_*` (`ds4.h:59-61`: T 1.0, top_p 1.0, min_p
  0.05). The server starts from the same defaults (`ds4_server.c:992-994`)
  and forces them again in thinking mode for knobs the request left out
  (`ds4_server.c:13950-13958`). Step 13 adds the narrow engine call; the
  server's thinking-mode override has to read it too.
- `ds4_session_supports_payload` (fork `8b70f74`): in the old branches only
  tests call it (`tests/test_mellum_config.c:928,957,1217`); no host does.
  Upstream's contract already covers it: `ds4_session_payload_bytes` returns 0
  when a session cannot be saved (`ds4.c:63511`), and stage/save/load/snapshot
  return an error. Step 14 refuses Mellum by name in those functions. It must:
  a Mellum session sits in the QWEN4_EXP family, so as is it would take the
  `ds4_session_is_qwen4` branch (`ds4.c:63512`) and size a Qwen3.8 payload.

### Other fork-isms in the old code (dropped, nothing to replace)

- `ds4_model_is_qwen35()` appears only in guards (`!ds4_model_is_qwen35() &&
  !ds4_model_is_mellum()`); upstream has no Qwen3.5, so the clause goes.
  `DS4_SHAPE_QWEN35` is the "sibling model" in `tests/test_mellum_config.c`
  and `tests/test_mellum_frontends.c`; `DS4_SHAPE_QWEN4_EXP` takes its place.
- `g_ds4_qwen_reasoning_effort` (fork `591c6a0`): the old Mellum config turns
  it off because 2.1's template has no reasoning effort. Upstream's
  `qwen4_chat_system` (`ds4.c:44450`) always prepends
  `ds4_qwen4_reasoning_effort_text`, and `qwen4_chat_assistant_prefix`
  (`ds4.c:44438`) opens `<think>\n` when thinking is on, so Mellum needs its
  own system turn and assistant prefix (steps 11 and 33), not a flag.
- The Metal 4 language-version fix `ebdcb0f` is not carried (CPython only).

### Upstream code the old port changed

These are not fork dependencies, but they touch other models' paths:

- The Qwen3.8 attention decode kernels gain an `ungated` word in place of
  `pad0` in `ds4_metal_args_qwen4_attn_decode` (`metal/qwen4.metal:2090`) and
  a branch in `qwen4_attn_decode_tile` (2115), `kernel_qwen4_attn_merge`
  (2254) and `qwen4_attn_merge_wide_head` (2294); the host wrappers
  `ds4_gpu_qwen4_attn_decode_tensor` and `ds4_gpu_qwen4_attn_decode_rows_tensor`
  become thin calls into `_impl` versions. This is step 17. `make
  test-qwen4-kernels` covers the gated path numerically; there is no Qwen3.8
  Flash Next GGUF on this machine, so its decode speed cannot be re-measured
  here.
- `ds4_qwen4_layer_is_linear/_ple/_nextn` need a Mellum guard (a Mellum shape
  has `n_full_attn_interval` 0 and arm64 `x % 0 == x`), and session creation
  must not give Mellum a shared prefill workspace (`ds4.c:74337`).

### Upstream functions Mellum will use

Each exists at `5379c1c` with the same prototype as at the fork base `ebdcb0f`
(and at `0aaea5a`). The kernels behind them are unchanged too: the fork's
edits to existing Metal code are Qwen3.5 GDN only (`git diff 0aaea5a ebdcb0f
-- metal/` deletes 6 lines, all in `kernel_qwen4_gdn_scan`/`_front`), and
upstream's only Metal change since `0aaea5a` is a new DSV4 kernel in
`metal/dsv4_misc.metal` (`git diff --stat 0aaea5a 5379c1c -- ds4_metal.m
metal/`: 2 files, 49 lines). So the oracle's byte-equality premise holds for
the reused kernels.

| function | `ds4_gpu.h` | `ds4_metal.m` | Mellum use |
|---|---|---|---|
| `ds4_gpu_embed_tokens_q8_0_tensor` | 588 | 12244 | token embedding |
| `ds4_gpu_rms_norm_weight_rows_tensor` | 1187 | 22301 | attn, ffn and output norms |
| `ds4_gpu_matmul_q8_0_tensor` | 727 | 19611 | one-row projections and the LM head |
| `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor` | 847 | 19637 | T-row projections, bit-equal to T one-row calls |
| `ds4_gpu_tensor_copy_f32_to_f16` | 61 | 9379 | K/V to f16 (full cache rows, sliding staging) |
| `ds4_gpu_add_tensor` | 2549 | 31163 | residuals |
| `ds4_gpu_qwen4_attn_decode_tensor` | 3584 | 49229 | full-layer decode attention (with `ungated`) |
| `ds4_gpu_qwen4_attn_decode_rows_tensor`, `ds4_gpu_qwen4_attn_rows_stage` | 387, 360 | 49491, 49323 | full-layer prefill rows, row by row |
| `ds4_gpu_qwen4_attn_part_floats` | 3583 | 49225 | split partials sizing |
| `ds4_gpu_qwen4_multi_gemv_tensor` | 3542 | 50183 | F32 router logits |
| `ds4_gpu_qwen4_router_topk_tensor` | 3548 | 49016 | top-8 of 64 |
| `ds4_gpu_qwen4_moe_mid_tensor`, `_down_tensor`, `_reduce_tensor` | 3593, 3599, 3620 | 49530, 49589, 49692 | Q8_0 experts, per token, no shared expert (`shared_type = UINT32_MAX`) |
| `ds4_gpu_qwen4_moe_mm_mid_tensor`, `_down_tensor` | 3629, 3634 | 49875, 49957 | step 25 only: Q8_0 (type 8) is accepted on simdgroup tiles; tensor tiles are types 12, 39, 16, 10 only (`qwen4_moe_mm_nax`, `ds4_metal.m:49809`) |
| `glm_graph_begin_commands_if_needed`, `ds4_gpu_end_commands`, `ds4_gpu_tensor_view/read/write/alloc` | | | one command buffer per chunk (`ds4.c:56437`) |

Upstream's speculative loop is where the brief says: `ds4.h:549-558`
(`ds4_sessions_eval_batch_speculative_argmax`) and `ds4.h:604`
(`ds4_session_rewind_speculative`). The model id and syntax selection that
step 3 tests are `ds4_server.c:1215-1225` and `ds4_agent.c:412-417`.

### Cherry-pick probe

`git diff ebdcb0f refs/old/mellum21-stage1 -- <file>` applied to a `5379c1c`
tree with `patch -p1 --dry-run`: `ds4_agent.c` 52 of 53 hunks apply (the
failing one is `agent_apply_model_sampling_defaults`, the host-extension
caller); `ds4.h` 2 of 3; `Makefile` 7 of 9; `ds4_cli.c` 0 of 1 (the sampling
call again); `ds4.c` 30 of 54. The agent cherry-picks; the engine is re-ported.

### Where the brief and the code disagree (for Checkpoint 1)

1. "Three fork-only kernels": the fork code Mellum uses is two kernels
   (ring write, head norm + RoPE) and one host dispatch helper. The sliding
   attention kernel is already Mellum's own.
2. Step 13, "upstream's agent defaults top_p to 0.95": only for GLM-DSA.
   Mellum would get T 1.0 / top_p 1.0 / min_p 0.05, so the divergence
   Mellum's table replaces is min_p 0.05 (top_p already matches).
3. Steps 25-27 and prefill-equals-decode: `ds4_gpu_matmul_q8_0_tensor` does
   take the Metal 4 tile (rows a multiple of 32, or the multiple-of-32 prefix
   when n_tok >= 192 outside quality mode, `ds4_metal.m:19422-19440`), but the
   old forward never calls it for more than one row. `mellum_proj`
   (`refs/old/mellum21-stage1:ds4.c:60688-60700`) uses the decode-rows-exact
   kernel so a prefill chunk equals its decode steps bit for bit, and
   `ds4_gpu_mellum_moe_tensor` avoids `moe_mm_*` because they stage half
   operands. So step 27 has nothing to confirm in the oracle-matched forward:
   the tile, like the `moe_mm` and `attn_mm` paths of steps 25-26, is a
   kernel swap with a tolerance, and it gives up the exact prefill = decode
   that step 21 builds and the backlog's B5 relies on.
4. Step 6, ring of window + 1 "so a speculative write never overwrites an
   entry a later token still reads": the old engine attends first and commits
   the chunk to the ring afterwards. After a forward ending at position P-1 a
   W-slot ring holds P-W..P-1, and after a rewind to P-k the next query needs
   P-k-W+1..P-k-1, which is in the ring when k <= 1. A depth-1 cycle only ever
   drops the draft row (k = 1), so W slots already suffice; W + 1 buys a
   rewind of 2. The kernel also uses one number for both (`slot = pos %
   args.window`, `metal/mellum.metal:62`), so W + 1 means a ring-size argument
   separate from the window: still the same logits, but not a verbatim port.
5. The brief lists 11 commits of `mellum21-on-upstream`; the branch has 21
   Mellum commits (on 40 fork commits). The list reads as a selection.
6. Not in the brief's list: `g_ds4_qwen_reasoning_effort` and the qwen4
   system/assistant renders above, and the old port's change to upstream's
   Qwen3.8 attention kernels (`ungated`), whose speed this machine can only
   check model-free.
