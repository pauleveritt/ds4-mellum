# S3b first generated ac-demo Phase 1 conversation

Date: 2026-10-28. Execution authority: Paul's direct overnight instruction to
continue reviewed SSD roadmap work without waiting for him, 02:00–08:00 Eastern.
Baseline `a86b0bc`, existing `codex/mellum-ssd-replay` worktree. Choices below are
the executor's conservative choices within that authority, not user-specified
experimental parameters. Use Superpowers planning, separate RED/GREEN commits,
full CPU checks and independent review before collection.

## Intent and bounded result

Obtain the first self-generated evaluation routing trace for the user's actual
ac-demo Phase 1 task, then replay unchanged global LRU at roughly 3 GiB. Existing
qualified captures are teacher-forced diagnostics; the publisher cannot produce
generated multi-turn evidence. Build only the missing bounded two-turn research
collector and analysis. Do not build an agent harness, edit ac-demo, execute model
output, run a server or claim its tests passed. This is a coding conversation,
not a completed Phase 1 implementation or a full representative benchmark.

Reuse the immutable nine-file bundle at
`.../01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/ac-demo-source-s3a/artifact`, content ID
`e14b1b120a8563fd1dcb316ea6e6d815b593d2ba45605fb34b55c3b238834861`, original
revision `bfb6e1f6e2ef141a1ac88f5c25466d3a076caab3`. Check all nine file digests.
Never change or re-freeze that source. Entire ac-demo remains evaluation data.

Prompt uses exact UTF-8 bodies of AGENTS.md, pyproject.toml, .gitignore and all
three specs in sorted path-delimited blocks; list the two empty placeholders.
Identify uv.lock by its exact size/hash, without copying its irrelevant 240 KB
body into context. Retain the complete source bundle linkage. Never pad input.
Freeze exact system, first-user and second-user bytes in a published input
artifact before native tokenization. First user asks for Phase 1 only, complete
file contents including home-page tests, no server, following preserved project
rules. Second user asks to review that answer against Phase 1, single quotes,
Request import and the specified test command, without claiming execution or
implementing Phase 2. It is a scripted user review, not fabricated tool output.

## Frozen schedule and budgets

Exactly **one conversation**, two generated responses, release numerical mode,
resident Q8 model, context8192, E1 chunk1024, thinking enabled, greedy temperature0
with top-k1/top-p1/min-p0 and deterministic lowest-ID tie breaking. Caps:
first response4096 generated tokens, second1024; no retry/seed sweep. Policy
is the existing grouped global LRU, unmodified; no policy selection/tuning after
opening this evaluation trace. Analyze3GiB only for this first result.

CPU-only inspect preparation uses native model tokenizer/template helpers,
never initializes Metal or allocates an inference session. Retain initial prompt,
append-user/assistant-prefix, newline and host-termination token sequences and
rendered token text. Native generated collection must recreate these exact token
sequences and reject a mismatch before capture begins. Require the worst-case
prompt+both caps+host termination+append+64 reserve <=8192; refuse rather than
truncate input. Model is the existing12925417760-byte Q8 artifact, SHA
`ea1e4ec737bb259ac2ffe1f20d412f458fd52dd230f42dce9e944f111093c29b`.
Use existing recorded full verification provenance plus unchanged stat identity;
no new whole-model hash pass is necessary for this study.

Sync initial prompt once (segment0); repeatedly copy logits, reject any nonfinite
logit, choose greedy token, and eval every generated token including a terminal
stop (segment1). Retain emitted IDs/text, per-operation timings and stop reason.
At EOS/EOS-alt append only template separator newline; protocol stop is a retained
unsupported-stop failure, with no invented tool result. If capped, append exact
host-injected `<|im_end|>\n` token sequence, explicitly labeled truncation and
host intervention. Append the frozen second user turn and assistant prefix,
sync the full unchanged-plus-extended token history (segment2), then generate
up to1024 tokens (segment3). Second-turn cap ends the recording without inventing
an EOS. Never reset, rewind or re-tokenize generated assistant text. Preserve
all generated control IDs in the actual committed token stream.

Capture through existing owned handle and observed kernel profile catalog.
Do not change routing, weights, math/reduction order or production engine code.
Use a separate ordinary session to teacher-force the exact collected tokens and
same sync/eval schedule once, checking logits byte-for-byte after every operation;
never call that replay a second generated conversation. This matching resident
parity gate runs before trace publication, including after append and host end.
If parity fails, retain failure and stop; no replacement trace.

Reserve one conversation plus one parity replay before GPU launch. Total native
GPU child wall timeout900seconds, aggregate <=900seconds for this attempt,
max5120 generated and5120 parity decode tokens plus two bounded prompt syncs
per arm. Include native CPU token preparation in a separate120second timeout.
At every command require deadline to fit before12:00UTC. Existing shared model
lock remains mandatory, no bypass/concurrent model process. A busy read-only
preflight postpones launch to a later overnight wake; once a GPU reservation is
created it is spent even on lock-race/failure. No automatic retry afterward.
Output root is exclusive and remains on failure. Preserve all partial files,
command output, source/binary/input identities and exception identity. No cleanup
of evidence. No downloads, dependencies or external communication. Python3.14+
stdlib and C99/POSIX only. Small commits, no push/merge/main/engine integration.

## Publication and analysis

Use existing exporter/spool grammar and profile validation. Add generated header
completion separately; existing teacher-forced publisher behavior stays unchanged.
Retain source/binary forced-build identities, actual host/backend/context/chunk,
exact prompt and consumed/generated tokens, operation ledger, profiles, spool,
parity verdict, bounded native receipt, commands and source/binary/input closing
checks. Derive token stream SHA from actual little-endian u32 IDs. Require exact
agreement between operation ledger and every spool frame's phase/segment/token
position/row count; no claimed trace completeness on failure. Full provenance
must precede the first GPU collection, while generated IDs necessarily become
observed outputs afterward. Freeze input protocol/settings/policy first.

Replay entire trace from cold with existing atomic per-layer group semantics;
never restart cache at a window. Report whole trace and windows selected before
seeing routes: all initial prefill; first32 decode positions of each response;
remaining decode positions after32; last256 decode positions of each response
(if fewer available, explicitly shortened); all append prefill. Include positions,
token counts, layer groups, demand/load/hit bytes and per-token misses/bursts.
Warm and tail windows overlap by design and are never added together. Empty
windows remain empty; a short/capped response cannot supply missing coverage.
Keep whole-trace existing replay output and trace hash. No relaxed-optimal/window
lower-bound claim, predictive/prefetch timing or inference-speed conversion.

Output is one small real-repository evaluation case, with descriptive locality
and resident recorder timing only. It does not settle real tool-result appends,
medium/8K context, other repositories, S3 original-range/concurrent storage,
S4 all-hit GPU cache path, native SSD inference or physical8GB feasibility.
Do not combine p95 storage service and miss counts into a claimed tok/s result.
S3 remains open. A report must state whether caps interrupted reasoning/code.

## Review focus

Check template separator/stop handling, capped-thinking intervention, exact
history append, parity at each operation, no prompt/output substitution,
nonfinite logits, reservation/timeout/interrupt retention, actual frame schedule,
evaluation-policy freeze and whole-prefix replay before window summaries.
