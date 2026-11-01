# Mellum in real use: Pi, ds4-server and an AgentClinic task (design)

Date: 2026-10-31. Branch `research/pi-real-use` in `ds4-mellum`, cut from `mellum` at `d2346d7`.

## Why

The README publishes one kind of speed: an engine ceiling. Its note under the summary table says so: measured
inside the engine, not over the server to Pi, on a 2K-token prompt, with no thermal load and one simple request.
A reader who wants to know what Mellum does when they actually code with it has nothing to go on. This work adds
the second number: what Mellum does when Pi drives it through `ds4-server` on a real coding task, and how that
trends down from the ceiling as a session's context grows and tool results pile up.

## What gets published (the owner's ask)

- **(a) The top line, synthetic.** The existing ceiling stays as it is (README summary table and Speed work).
- **(b) Real-world rates.** Prefill and generation tokens per second for each request of a real session, against
  that request's context depth, so the trend down from (a) is visible. Next to the rates, whether the task passed,
  so a fast run that failed is not read as a good one.

The deliverable is a research note in `docs/superpowers/research/` with the measurement, and new README rows (and a
small table of rate against context) that cite it with their `n`.

## The workload: copied, not run from mellum-worker

The owner's decision: `ds4-mellum` does not use `../mellum-worker` at all. Its AgentClinic "easy" suite
(`learn/code/suites/easy` at `06dcd66`) is the source to copy, and its `learn/code/clinic` script shows how to run
Pi headless. Copied into `ds4-mellum` (a directory under `tests/`, its name the builder's), with a line naming the
source commit:

- the starting project: `base/` (`AGENTS.md`, `pyproject.toml`, `uv.lock`, `specs/mission.md`, `specs/roadmap.md`,
  `specs/tech-stack.md`);
- the prompt: `prompts/01.md`;
- the grader: `grader/` and `expected.json` (four Phase 1 checks), with the `known-good` and `known-broken`
  fixtures that prove the grader passes a good project and fails a broken one.

Nothing of `mellum-worker`'s harness, extension, recorder, arms or nudges comes along. Pi runs plain.

## The stack under measurement

- Pi **1.1.0**, as installed (the owner's decision: no upgrade), headless: `pi -p` with a session id and session
  directory, `--no-approve --no-extensions --no-skills --no-prompt-templates --no-themes --no-context-files`, as the
  `clinic` script runs it, and `PI_CODING_AGENT_DIR` pointing at a profile this repository holds.
- That profile: one provider `ds4` (`http://127.0.0.1:8099/v1`, `openai-completions`), model `mellum-2.1`,
  `contextWindow` 32,768 (equal to the server's `-c`), `maxTokens` 16,384, `reasoning: true`, `samplingParams`
  temperature 0.6, top-p 0.95, top-k 20 (the README's settings).
- `ds4-server` from this branch, Mellum 2.1, `-c 32768`, thinking on.
- Model files: the published Q8_0 and MXFP4_MOE (sha256 as in the README). The M5's local Q8_0 is not the published
  file (`ea1e4ec7…`); the runs use the published one, and the note records the sha256 of the file each run used.

## Where the rates come from

`ds4-server` already prints, per request, on stderr: prefill progress with the context size and a running t/s, and
decode `chunk=`/`avg=` t/s with `ctx=`. The run keeps the server's stderr whole, and a small parser (Python, standard
library, `uv run --no-project`) turns it into one row per request: context at the start, new prompt tokens
prefilled, cached tokens, prefill t/s, generated tokens, decode t/s. New versus cached prompt tokens come from the
server's live-cache hit/miss lines and the `ctx=A..B` span; if those cannot give them for every request, the server
gets one per-request summary line (prompt, cached and generated tokens, prefill and decode seconds) as its own small
commit with a test, and the note says so.

Pi's own session file (from `--session-dir`) is kept too, as the record of what was asked and which tools ran; the
parser checks its request count against the server's and flags a mismatch rather than guessing.

Capture is separate from analysis: the raw server log, Pi's session file, the project directory and the grader's
output are kept, so every number can be recomputed without running a model.

## Goals

1. The copied task, with its grader shown to pass the known-good fixture and fail the known-broken one.
2. One command that runs the task against a fresh `ds4-server` and keeps the raw records.
3. A parser, with tests, that turns a server log into per-request rates.
4. A measurement on the M5 Max, Q8_0 and MXFP4: rates against context and pass/fail, with `n` stated.
5. A research note and README rows citing it.

## Non-goals

- Upgrading Pi; any `mellum-worker` code or change to it.
- Changing the engine or the server for speed. A server change to report rates is the one exception (above).
- The medium and hard suites; other machines (the M1 Pro and the iMac are follow-ups once the M5 run is settled).
- A pass-rate claim about Mellum's coding ability beyond what `n` supports.

## Anti-goals (failure even if it runs)

- A rate with no `n`, or a single run reported as typical.
- Rates that cannot be recomputed from kept raw files.
- Mixing a failed run's speed into a headline without saying it failed.
- Touching the owner's real Pi configuration (`~/.pi/agent`) or `~/models` contents.

## Constraints

- One concern per commit, about 150 product lines, never over 300 lines per commit (copied fixtures excepted, in a
  commit of their own); a red test in its own commit before the code it pins.
- No `--raw-prompt`; no `iogpu.wired_limit_mb` change; Python through `uv run`.
- Measurements on AC power in High Power mode, with power and thermal state recorded before and after each run.
  Nothing else using the GPU (the owner's `omlx-server` stopped or confirmed idle, and recorded).
- Evals are screens first: a few runs and a short note; preregister arms and `n` only to decide something.

## Decisions made here, as my calls (the owner can override)

- **n.** A screen of 3 runs per quant first; the note reports each run's per-request rates and the medians by
  context band, with the run count, and says whether that is enough to publish.
- **Context bands.** By context at the start of a request (for example under 4K, 4-8K, 8-16K, 16-32K); the exact
  bands follow from what the screen shows.
- **Deadline.** 30 minutes per run. A run that hits it counts as a failure in the pass column; its requests' rates are
  kept and marked as from a run that did not finish.
- **Server lifecycle.** One fresh `ds4-server` per run, started by the runner and stopped after Pi exits, stderr at
  its default level. No warm-up: a run's first request is a cold prefill and is marked as such.
- **Compaction.** If Pi compacts, those requests are part of real use: kept and marked.
- **Thinking.** Decode rates count every generated token, reasoning and answer alike. Prefill rates for requests
  with fewer than 256 new prompt tokens are reported per request but left out of the band medians: fixed costs
  dominate them, and they are not comparable to the 2K ceiling.
- **Scratch home.** As in the README checks: `HOME` and `PI_CODING_AGENT_DIR` are scratch directories, `HF_HOME` is
  the real cache, and `~/.pi/agent` and `~/models` are never written.

## Left to the builder

The copied suite's directory name, the runner's flags and language (shell or Python), the parser's structure, file
layout under `tests/` or `tools/`, and the note's prose.
