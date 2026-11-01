# Mellum in real use (Pi through ds4-server) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure Mellum's per-request prefill and decode rates, and pass/fail, when plain Pi 1.1.0 drives
`ds4-server` through a copied AgentClinic "easy" task, and publish them beside the README's engine ceiling.

**Architecture:** The task is copied into `tests/agentclinic_easy/`. A Python runner starts a fresh `ds4-server` per
run, runs Pi headless in a scratch home against a profile it writes, grades the project, and keeps every raw file. A
parser turns the server's existing per-request log lines into rates; a report bands them by context.

**Tech Stack:** Python 3 standard library (`unittest`, `subprocess`, `dataclasses`, `statistics`), run as
`uv run --no-project python -I`; bash only where the Makefile calls it; Pi 1.1.0; `ds4-server` from this branch.

**Spec:** `docs/superpowers/specs/2026-10-31-mellum-pi-real-use-design.md`

## Global Constraints

- One concern per commit, about 150 product lines, never over 300 lines per commit; the copied suite is one commit of
  its own. A red test goes in its own commit before the code it pins.
- Python through `uv run --no-project python -I`; standard library only in `tools/` and the tests.
- Pi 1.1.0, no upgrade; no `mellum-worker` code is imported or run.
- `ds4-server ... -c 32768 --port 8099`, thinking on; never `--raw-prompt`; no `iogpu.wired_limit_mb` change.
- Pi profile: provider `ds4`, `baseUrl` `http://127.0.0.1:<port>/v1`, `api` `openai-completions`, model `mellum-2.1`,
  `contextWindow` 32768, `maxTokens` 16384, `reasoning` true, `samplingParams` temperature 0.6, top_p 0.95, top_k 20.
- Pi flags: `-p --session-id <id> --session-dir <dir> --provider ds4 --model mellum-2.1 --no-approve --no-extensions
  --no-skills --no-prompt-templates --no-themes --no-context-files`, with `PI_CODING_AGENT_DIR` and `HOME` scratch.
- Never write `~/.pi/agent` or `~/models`. Deadline 1,800 s per run. Prefill medians skip requests with < 256 new
  prompt tokens. Context bands: < 4096, 4096-8191, 8192-16383, 16384-32768 (adjust only from the screen, in the note).
- Measurements: AC power, High Power mode (`pmset -g` shows `powermode 2`), state recorded before and after a run.

## Review Focus

1. `ds4-server` exits or crashes mid-run: the runner must stop Pi, keep the log, and record the run as failed with the
   server's exit code, not hang until the deadline. (Task 5, `test_server_death_fails_the_run`.)
2. The server never prints `listening on`: the runner gives up after 300 s with a clear error and kills it.
   (Task 5, `test_server_that_never_listens_is_refused`.)
3. A fully cached request (`ctx=P..P:0`) or one whose prefill took 0 s: rates are `None`, never a division by zero.
   (Task 3, `test_fully_cached_request_has_no_prefill_rate`.)
4. Pi leaves children behind (the task's `uv run python app.py` starts uvicorn with reload): at the deadline or at
   exit the runner kills Pi's whole process group, so no server holds port 8000 or the GPU into the next run.
   (Task 5, `test_deadline_kills_the_process_group`.)
5. Log lines with ANSI colour codes and the `MMDD HH:MM:SS ` prefix, and lines from other categories in between:
   the parser ignores what it does not know and strips escapes. (Task 3, `test_parses_coloured_interleaved_log`.)

---

### Task 1: Copy the AgentClinic "easy" task

**Files:**
- Create: `tests/agentclinic_easy/` from `../mellum-worker/learn/code/suites/easy` at `06dcd66`: `base/`,
  `prompts/01.md`, `grader/` (including `canary/` and `_contract.py`), `expected.json`, `fixtures/known-good/`,
  `fixtures/known-broken/`; not `layout.json`.
- Create: `tests/agentclinic_easy/SOURCE.md`: one paragraph naming the source path and commit, and that the files are
  copied unchanged.

- [ ] **Step 1:** Copy with `git -C ../mellum-worker archive 06dcd66 learn/code/suites/easy | tar -x` into a scratch
  directory, then move the listed paths; check `diff -r` against the archive shows only `layout.json` missing.
- [ ] **Step 2:** Commit: `tests: copy AgentClinic "easy" from mellum-worker 06dcd66 (unchanged)`.

### Task 2: Grade a project, and prove the grader both ways

**Files:**
- Create: `tools/mellum_pi_grade.py`
- Test: `tests/test_mellum_pi_grade.py`
- Modify: `Makefile` (targets below)

**Interfaces:**
- Produces: `GradeResult(passed: bool, checks_ran: bool, missing: list[str], output_tail: str)` (frozen dataclass);
  `missing_checks(output: str, expected: list[str]) -> list[str]`; `grade(workspace: Path, suite: Path) -> GradeResult`
  (copies `suite/grader` to `workspace/grader_tests`, runs `uv run --with pytest pytest -q -rA grader_tests` in
  `workspace` with `PYTHONPATH=workspace`; `checks_ran` is exit code 0 or 1 and no `errors during collection`;
  `passed` is `checks_ran` and no missing check); `qualify(suite: Path) -> dict[str, bool]` (base plus each fixture in a
  temp directory, graded); CLI `python tools/mellum_pi_grade.py qualify SUITE` exits 0 only if known-good passes and
  known-broken fails.

- [ ] **Step 1: Failing tests (model-free, no uv):** `test_missing_checks_lists_expected_ids_not_passed`
  (output with `PASSED a` and `FAILED b`, expected `[a, b, c]` gives `[b, c]`) and
  `test_collection_error_is_not_checks_ran` (`grade` on a workspace whose `app.py` raises at import: `checks_ran`
  False, `passed` False). Commit red.
- [ ] **Step 2:** Implement; run `uv run --no-project python -I tests/test_mellum_pi_grade.py`. Expected: OK.
- [ ] **Step 3:** Makefile: `test-mellum-pi-grader` runs `python3 -I tools/mellum_pi_grade.py qualify
  tests/agentclinic_easy` (needs `uv` and, the first time, the network; not in `test-mellum`). Run it. Expected:
  known-good True, known-broken False, exit 0. Commit.

### Task 3: Parse ds4-server's log into requests

**Files:**
- Create: `tools/mellum_server_requests.py`
- Test: `tests/test_mellum_server_requests.py`

**Interfaces:**
- Consumes: the server's existing lines (`ds4_server.c` 14172 and 15040-15090):
  `<MMDD HH:MM:SS> ds4-server: chat ctx=C..P:N[ flags] prompt done S.SSSs` and
  `... chat ctx=C..P:N gen=G[ flags] finish=F[ error="E"] T.TTTs`; `ctx` is cached `C`, prompt `P`, new `N`.
- Produces: `Request(index: int, cached: int, prompt: int, new: int, prefill_s: float | None, generated: int,
  total_s: float, finish: str, error: str | None)` with properties `prefill_tps` (`new / prefill_s`, `None` if
  `new == 0` or no `prefill_s` or `prefill_s <= 0`) and `decode_tps` (`generated / (total_s - prefill_s)`, `None` if
  either is missing or the span is `<= 0`); `parse(lines: Iterable[str]) -> list[Request]` (a `finish=` line closes
  the request opened by the latest `prompt done` with the same `ctx`; a `finish=` with no `prompt done` gives
  `prefill_s=None`); CLI prints one tab-separated row per request.

- [ ] **Step 1: Failing tests:** `test_one_request_rates` (prompt done 2.000 s at `ctx=0..4096:4096`, finish
  `gen=256` at 4.000 s: prefill 2048.0, decode 128.0), `test_fully_cached_request_has_no_prefill_rate`,
  `test_error_finish_keeps_the_error` (`finish=error error="boom"`), `test_finish_without_prompt_done`,
  `test_parses_coloured_interleaved_log` (escape codes, a decoding progress line and a KV-cache line between the two).
  Commit red.
- [ ] **Step 2:** Implement; run the tests. Expected: OK. Commit.

### Task 4: Band requests across runs

**Files:**
- Create: `tools/mellum_pi_report.py`
- Test: `tests/test_mellum_pi_report.py`

**Interfaces:**
- Consumes: `parse`, `Request` (Task 3); each run directory's `server.log` and `run.json` (Task 5).
- Produces: `Band(low: int, high: int, prefill_median: float | None, prefill_n: int, decode_median: float | None,
  decode_n: int, runs: int)`; `bands(requests: list[Request], edges=(4096, 8192, 16384, 32769), min_new=256) ->
  list[Band]` (a request belongs to the band of `cached`, its context at the start); CLI
  `python tools/mellum_pi_report.py RUN_DIR...` prints a markdown table per run (each request, run's pass/fail and
  `deadline_hit`) and one band table over all runs, each row with its `n`. In the per-run table, request 0 is marked
  `cold`, and requests listed in `run.json`'s `compaction_requests` (Task 6 fills it) are marked `compaction`.

- [ ] **Step 1: Failing tests:** `test_small_prefills_leave_the_median` (requests of 100 and 3,000 new tokens:
  prefill_n 1), `test_band_by_context_at_start`, `test_empty_band_has_none_not_zero`,
  `test_first_request_is_marked_cold`. Commit red.
- [ ] **Step 2:** Implement; run. Expected: OK. Commit.

### Task 5: Run one session

**Files:**
- Create: `tools/mellum_pi_run.py`
- Test: `tests/test_mellum_pi_run.py` (fake `ds4-server` and fake `pi` as small scripts written by the test)
- Modify: `Makefile`: `test-mellum` runs the three model-free test files of Tasks 3-5 and the two of Task 2's
  Step 1.

**Interfaces:**
- Consumes: `grade` (Task 2), `parse` (Task 3).
- Produces: CLI `python tools/mellum_pi_run.py --model PATH --out DIR [--port 8099] [--deadline 1800]
  [--server ./ds4-server] [--pi pi]`. `DIR` gets: `server.log`, `pi.stdout`, `pi.stderr`, `session/` (Pi's),
  `workspace/` (the project after the run), `grade.txt`, `run.json` with `model`, `model_sha256`, `engine_commit`
  (`git rev-parse HEAD`), `pi_version`, `power_before`/`power_after` (`pmset -g` and `pmset -g therm` text),
  `deadline_hit`, `pi_exit`, `server_exit`, `passed`, `checks_ran`, `missing`, `server_requests`,
  `pi_assistant_messages`, `count_mismatch`, `started`, `ended`. Refuses an existing non-empty `DIR`.

- [ ] **Step 1: Failing tests:** `test_profile_has_the_readme_settings` (written `models.json` has every Global
  Constraints value), `test_scratch_home_and_profile_dir` (fake `pi` records `HOME` and `PI_CODING_AGENT_DIR`; both
  under `DIR`, neither under the real home), `test_server_death_fails_the_run`,
  `test_server_that_never_listens_is_refused` (listen timeout as a parameter, 1 s in the test),
  `test_deadline_kills_the_process_group` (fake `pi` starts a sleeping child; after the deadline neither lives,
  `deadline_hit` true), `test_count_mismatch_is_flagged`. Commit red.
- [ ] **Step 2:** Implement: start the server in its own process group with stderr to `server.log`, wait for
  `listening on`, copy `tests/agentclinic_easy/base` to `workspace/`, run Pi there in its own process group with the
  prompt from `prompts/01.md`, kill both groups at exit or deadline, then grade. `pi_assistant_messages` counts
  assistant messages in Pi's session file; Task 6 confirms that against a real session and adjusts if Pi 1.1.0 writes
  them differently.
- [ ] **Step 3:** Run all model-free tests: `make test-mellum`. Expected: pass. Commit.

### Task 6: A live check, then the screen

- [ ] **Step 1: One live run (a check, not a measurement):** published Q8_0 from the Hugging Face cache (sha256
  `69e0ac02…`), `tools/mellum_pi_run.py --model ... --out ~/ds4-runs/pi/check-1`. Expected: Pi edits the project,
  `server_requests` equals `pi_assistant_messages`, `tools/mellum_pi_report.py` prints rates. Read Pi's session file
  for how 1.1.0 records a compaction; if it does, `mellum_pi_run.py` writes those requests' indexes to `run.json`'s
  `compaction_requests` (empty list otherwise), with a test from a trimmed real session. Fix and commit what the
  check finds, each with a test.
- [ ] **Step 2: The screen:** confirm AC and `powermode 2`, `omlx-server` stopped or idle (recorded); 3 runs with the
  published Q8_0 and 3 with the published MXFP4_MOE (sha256 `cf09584a…`), alternating, 2 minutes apart. Keep every
  run directory; copy them into `docs/superpowers/research/2026-10-31-mellum-pi-real-use/` without `workspace/.venv`.

### Task 7: The note and the README

**Files:**
- Create: `docs/superpowers/research/2026-10-31-mellum-pi-real-use.md` (honesty header; per-run tables and band
  table from `tools/mellum_pi_report.py`, with the command that recomputes them; pass counts with `n`; what would
  falsify this; risks)
- Modify: `README.md`: a "Real use in Pi" subsection under the M5 Max section with the band table's medians and `n`,
  linked to the note; one sentence near the summary table's note pointing at it. `MELLUM.md`: a Measured row.

- [ ] **Step 1:** Write the note from the report output; every figure has its command and `n`.
- [ ] **Step 2:** README and MELLUM.md rows; check every link resolves. Commit.
