# Mellum 2.1 on ds4

Branch `mellum` ports JetBrains' Mellum 2.1 (12B-A2.5B, thinking) onto
antirez/ds4, starting at `5379c1c`, in small commits. This file records what
works, what was measured (each number with its command and n), what diverges
from the model's training, and what is not covered.

## Status

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
Long prefills are exact only under the diagnostic `DS4_MELLUM_EXACT_PREFILL=1`
(read at session start), which sends every chunk back through the exact paths;
that is what the oracle uses to check long prefills byte for byte.

Speed on the M5 Max (Q8_0, Metal), step 29, through the CLI after a rest, n = 3
each: a short chat prompt prefills at 440-450 t/s and generates at 132.6-133.1
t/s; a 3,235-token prompt prefills at 1,842-1,871 t/s and then generates at
116.2-117.8 t/s. The trace driver's 2,200-token prefill: about 2,000 t/s. The M4
Max was not measured. Session save/load refuses by name, and ds4-server
and ds4-agent refuse a Mellum engine at startup (no Hermes tool syntax yet).

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
  pinned against the template's ids). ds4-server still refuses Mellum; its
  request renderer (step 42) is to drop earlier reasoning and keep it within
  the current tool cycle, as the template does.
- Special-token text in user turns: the HF tokenizer would encode a
  `<tool_call>` typed in a user message as token 29; ds4 keeps user text plain
  BPE, as for every model. System text (the tools block, `-sys`) and assistant
  text match special tokens, as the template's tokenizer does.

## Measured

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
| router and experts on real activations vs this port's double reference (R7), 11ed730 | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-moe` (MoE input rows of `tests/mellum_heldout_unicode.tokens` at 64 positions, layers 0/6/10/13/14/27, 384 rows per path) | selection equals the reference in every row; 1 near-tie (8th/9th router logit gap < 1e-4), broken the same way. Exact rows: output 1.2e-6, weights 3.2e-7. Tiled rows: 1.1e-4 from a reference that stages as the tiled kernels do (input, half(d * q) weights, SwiGLU output as half), and 2.5e-3 from the unstaged one at layer 27 (max \|x\| 21.8): about 96% of the tiled path's error on real rows is the half staging. The random-row check's 1.0e-3 understated that cost | 1 |
| exact-prefill switch | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-exact-prefill` | 1,100-token prefill vs decode: 0 of 98,304 logit words differ with the switch, 98,303 without | 1 |
| batched + tiled prefill vs the old engine / exact decode, 64 positions | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64` (after step 27; rule: top-1 everywhere, max\|d\| <= 10, mean KL <= 0.01) | 1,100: max\|d logit\| 2.008, mean KL 2.1e-3, top-1 65/65; 2,200: 8.471, 1.7e-3, 65/65; short prompt and every decode byte-equal; 64/64 greedy equal; PASS | 1 |
| tiled prefill vs the exact path / the old engine, logits at 64 positions | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64` (after step 25) | 1,100 tokens: max\|d logit\| 0.649, top-1 65/65; 2,200: 0.874, top-1 65/65; short prompt and every decode still byte-equal; 64 of 64 greedy tokens equal; PASS | 1 |
| MoE driver on real layers, 64 rows tiled vs double | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-moe` | same experts selected; output within 1.0e-3 (16 rows exact: 3.4e-7, row 0 alone bit-equal) | 1 |
| the oracle script can fail: the same comparison against an old tree whose sliding attention splits keys over 4 simdgroups instead of 8 (same math, another summation order; scratch, not committed) | `tests/mellum_oracle_compare.sh ORACLE_BAD $MODEL 8` | exit 1, `mellum-oracle: FAILED`: all six cmp checks fail (max\|d logit\| 7.3e-3 at 29 tokens, 9.1e-2 at 1,100, 5.4e-1 at 2,200, argmax still 23), the greedy check fails at token 0; this engine's prefill-equals-decode checks still pass | 1 |
| Metal kernels vs double references | `make test-mellum-kernels` | 379 checks, 0 failures | 1 |
| MoE on real layers 0/13/27 vs double | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-moe` | same experts selected; weights 1.7e-7, output 3.4e-7 | 1 |
| session bytes allocated vs planned, Q8_0 GGUF | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-session` | equal at ctx 20,000 / 65,536 / 131,072 for chunks 1,024 and 512 (601,513,984 / 1,254,318,080 / 2,193,842,176 bytes at chunk 1,024) | 1 |

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
- Per-position correctness rests on the exact path. Under `DS4_MELLUM_EXACT_PREFILL=1` the oracle checks
  long prefills byte for byte against the old engine and decode. A change that reorders work without
  changing arithmetic can be held to byte equality with the release path it replaces, which is a
  stronger gate than any cap.

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

### E1: routed Q8_0 experts on the tensor tiles (reverted)

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

Reverted. The two changes are near-ties of the kind "Oracle caps" shows the current path making on
held-out prompts, but the gate is the oracle's top-1 rule, and changing that rule is Paul's decision.
The patches (Q8_0 raw words, a Mellum-only opt-in through `DS4_MELLUM_MOE_NAX`, a red test) and the
runner are in the session notes folder, `e1-q8-tensor-tiles/`, with a README.

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
