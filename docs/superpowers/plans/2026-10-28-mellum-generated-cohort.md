# First Generated Phase 1 Cohort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One source-checked generated ac-demo Phase1 routing conversation, matching resident logit parity and held-out3GiB LRU windows.

**Architecture:** Reuse owned capture, native template/tokenizer, spool/export/profile validators, source identities and existing LRU. Add a bounded native callback-driven generation/parity loop, native inspect/collect executable and Python frozen input/publisher/window analysis. Persist raw outputs outside publisher cleanup. No production engine changes.

**Tech Stack:** Python3.14+ stdlib, C99/POSIX, existing Metal resident engine; no dependencies.

**Spec:** [Generated cohort design](../specs/2026-10-28-mellum-generated-cohort-design.md). Fresh builder check: nothing important left. Paul's overnight instruction explicitly delegates routine planning/execution without waiting for questions/approval; all experimental parameters remain executor choices.

## Global Constraints

- Existing branch/worktree, baseline19c7231; no push/merge/main/rebase or ds4-engine/ac-demo edits, execution of generated code, server, downloads/dependencies.
- One evaluation conversation only, release/E1 chunk1024/context8192/thinking on, greedy caps4096 and1024. One resident parity replay; aggregate GPU child <=900seconds and before12:00UTC. CPU inspect <=120seconds.
- Preserve exact routing/Q8/math, native templates/generated token IDs and complete prefix. Initial LRU code/policy frozen before evaluation; no tuning from results.
- Fresh exclusive inputs/run roots, reservation before GPU launch, no retry/replace/delete after partial launch. Busy preflight may postpone before reservation. Capture all failures, PID/exit/exception/cause and partial raw files.
- No storage attempt rerun; prior CPU storage reservation spent. No individual SSD attribution, inference tok/s conversion or8GB conclusion.
- Native logit parity tape bounded by8194 operations*393216B (<4GiB), plus <=32MiB spool/text; retained-output hard cap4GiB. Require32GiB free. Gold tape is part of retained experiment, no unbounded in-memory logits tape.
- Small commits, separate observed RED before GREEN, roughly150 product lines per implementation commit. Full model-free gate and one fresh full-range review before GPU collection; only that review, then TDD fixes.

## Review Focus

1. EOS/EOS-alt versus protocol stop and capped thinking must preserve generated IDs and label host termination; native CPU fake-backend stop tests pin this in Task2.
2. Same-history append must not re-tokenize output; mismatched prepared token/template/source/binary inputs must refuse before collection/publication (Tasks1/3).
3. Parity must cover every sync/eval including terminal and append, reject nonfinite/mismatched logits and extra/truncated tape or operations (Task2).
4. Cancellation, lock race, timeout, write/close failure or native nonzero exit preserves evidence and spent reservation, with no claimed complete trace (Task3).
5. Window summaries must replay full prefix once, report shortened/empty windows and preserve evaluation policy freeze; test prefix reuse versus restart and phase overlap (Task3).

## Task 1: Frozen exact text and native inspect token preparation

**Files:** create `tools/mellum_generated_inputs.py`, `tools/mellum_generate_io.h`, `tools/mellum_generate_io.c`, `tests/test_mellum_generated_inputs.py`, `tests/test_mellum_generate_io.c`; extend Makefile/ignore.

**Interfaces:** Python `messages(bundle:Path)->dict[str,bytes]` verifies all9 frozen file hashes, renders system/user/followup files and repository linkage. `prepare(bundle,model,binary,container)->Path` exclusively creates root/raw, invokes native CPU inspect <=120seconds, freezes text/tokens/geometry/model-stat/source/binary/initial-LRU identities, publishes root/result/artifact by hardlink; failures retain raw and failure.json. `verify(prepared)->dict` metadata/content checks only, never whole model hashes. Native I/O `mellum_generated_read_text(path)` bounded64KiB, no NUL; `read_tokens(path,out)` <=8192 decimal IDs<98304; `write_tokens(path,tokens)` exclusive checked; `write_rendered(path,engine,tokens)` exact token pieces exclusive checked. Native CLI preparation implemented with executable in Task2; Task1 Python tests inject bounded native command outputs and assert frozen mismatch/exclusivity/failure retention. Native I/O tests use tiny local files.

- [ ] Write tests: all9 source hashes, exact sorted prompt text (uv.lock hash only), mutation refusal, no repetition/padding; token parser invalid/overflow/empty/truncated, no overwrite; prepared budgets/policy/source/token mutation and failed command preserve raw. Run to intended RED; commit tests separately.
- [ ] Implement bounded native I/O and Python manifest contracts in focused GREEN commits. Verify fixture tests and existing workload-source tests.

## Task 2: Native generated/ordinary parity loop and CLI

**Files:** create `tools/mellum_generate_loop.h`, `tools/mellum_generate_loop.c`, `tools/mellum_generate.c`, `tests/test_mellum_generate_loop.c`; extend Makefile/ignore.

**Interfaces:** callback backend `{context,sync(history,segment),eval(token,segment),logits(out,n),stop_class(token)}`. `mellum_generated_sync` records one S operation and gold logits after successful sync; `mellum_generated_turn` greedily consumes current logits, evals/appends IDs once, writes one E operation/gold logits per token and returns count/stop_class, refusing protocol/nonfinite. `mellum_generated_parity` reads bounded S/E ledger and gold tape against complete final token history, calls ordinary backend, memcmp after each operation, refuses mismatch/truncation/extra bytes. FILE sinks checked; all helpers return failure, no exit on errors.

CLI `tools/mellum_generate prepare MODEL INPUT_DIR OUTPUT_DIR` uses CPU inspect only to derive initial/append/close/newline tokens and rendered text from system/user/followup; context worst-case check <=8192. `collect MODEL INPUT_DIR OUTPUT_DIR` uses Metal resident engine with explicit chunk1024/context8192, recreates native tokens and compares prepared sequences before owned handle; captures initial sync, turn0, host separator+followup sync, turn1, finish/catalog, ordinary parity, checked receipt. All native outputs exclusive, no cleanup; close every opened stream even after failure. Files: `owned.routes`, `owned.profiles`, `gold.f32`, `operations.txt`, `tokens.txt`, `emitted.txt`, `response0.txt`, `response1.txt`, `runtime.json`, `result.json`. Finish/footer alone doesn't publish if parity/close fails. Cap/stop and injected host token count explicit.

- [ ] Fake-backend CPU tests cover greedy ties, NaN anywhere, EOS/protocol/cap, append prefix, every-operation parity, mismatch/short/extra tape and failed sinks. Observe RED, commit. Native CLI CPU build includes DS4_NO_GPU; prepare smoke uses tiny/fake boundaries only. Compile Metal binary without running before final review.
- [ ] Implement helpers and CLI in <=150-product-line pieces, build/execute CPU loop/I/O checks and existing capture API/publisher tests. Add no-GPU-generated target to full gate.

## Task 3: Checked publisher, frame schedule and whole-prefix windows

**Files:** create `tools/mellum_generated_publish.py`, `tools/mellum_generated_analysis.py`, `tests/test_mellum_generated_publish.py`, `tests/test_mellum_generated_analysis.py`; extend usage/Makefile/roadmap.

**Interfaces:** `run(prepared,model,binary,container)->Path` does busy/host/free-space/deadline preflight before exclusive reservation; verifies frozen source/LRU/inputs and force-build receipt, launches one bounded native collect<=900seconds, retains command outputs/raw on every outcome. Standard shared engine lock unchanged. Source/binary/input/model-stat checks before/after, native success receipt, exact rendered/token identities, stop/cap policy, consumed S/E schedule, runtime/profile validation. Generated header completion is separate from existing teacher-forced helper; use schema mode generated, split evaluation, exact generation/input/policy settings. Export only after native parity and all closing gates, using hardlinks for retained files; failure never publishes trace success. Deadline supplied as frozen12:00UTC, no CLI budget overrides in production.

`check_schedule(spool,operations,tokens,cap)` rederives every expected forward from S prefix extension or E token, validates segment/phase/position/rows, then exact footer totals/token digest. `analyze(trace,operations,capacity=3GiB)` runs existing LRU over all groups from cold, assigns only frozen segment windows, emits demand/hit/load byte and per-token burst summaries; preserve original whole-trace replay report and source/trace/policy hashes. Never open evaluation routes until manifests/settings/LRU hash frozen. Empty windows explicit.

- [ ] CPU tests fabricated actual spool/receipts: generated schedule accepted; substituted token/segment/cap/source/plan rejects; native failure/cancellation keeps reservation/raw and no success; no second command/retry; fake command honors900second/combined1MiB bounds. Nonfinite/parity receipt false rejects. Tests causal full-prefix warm windows, short/empty/tail overlap and atomic group oversized refusal. RED commit then focused GREEN pieces.
- [ ] Full serial model-free gate including new tests, existing synthetic-fixture lock isolation only if unrelated live inference blocks model-free config test. Record exact commands and no live GPU tests in this gate.

## Task 4: Review, one bounded capture, artifact audit, report

- [ ] Fresh whole-range review baseline19c7231..implementationHEAD, spec/plan/ledger and five focus items. Fix Important/Critical findings via separately committed RED/GREEN and full CPU gate; no extra reviewer.
- [ ] Freeze source/LRU/binaries/input text, CPU inspect native tokens under one CPU preparation, all settings and output roots before GPU launch. Use root `/Users/pauleveritt/.codex/visualizations/2026/10/28/mellum-generated-phase1/` with new `inputs`, `run`, `qualification` children; stop on existing paths. Retain CPU command120second budget separately. Busy before GPU reservation may postpone to another wake; no GPU reservation/launch until free and within deadline. One900second native collect includes generation and parity; no retries after reservation, even on failure.
- [ ] Artifact-only audit: source/input/policy/native/output identities, rendered/token/operation/frame agreement, whole-prefix LRU/windows, profile/runtime/caps, stop/cap/truncation labels, parity logit tape length and native verdict, retained failures. No model/scratch rehash or storage experiment. Report actual n=1 coding conversation limits and recorder resident timing; preserve raw/failed evidence.
- [ ] Commit evidence/roadmap/usage status, retain ledger/commands/test/review/audit artifacts before plan-workspace cleanup. Keep branch/worktree for authorized overnight work, do not ask to merge/push. Advance only to preceding-measurement-supported next roadmap work within remaining time.
