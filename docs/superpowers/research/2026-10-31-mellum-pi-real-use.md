# Mellum 2.1 in real use: plain Pi on AgentClinic "easy"

**Honesty header.** Measured: 6 screen runs (3 per quant) and 1 earlier check, on one M5 Max (128 GB, AC
power, High Power mode), engine `0364a95` (screen) and `0aa1fde` (check), Pi 1.1.0. The per-request rates are
computed from ds4-server's own log lines; the pass/fail verdicts come from the copied task's grader. Nothing here
is a rate of how often Mellum passes the task: n = 3 per quant is a screen, and its pass counts say only which
failure modes exist. Conjecture: why each failure happened, beyond what the session file shows.

## What ran

`tools/mellum_pi_run.py` starts a fresh `ds4-server -c 32768` (thinking on), copies
`tests/agentclinic_easy/base` (from mellum-worker `06dcd66`, see its `SOURCE.md`) into a workspace, and runs plain
Pi 1.1.0 headless (`-p --no-approve --no-extensions --no-skills ...`) with the prompt "Implement Phase 1 — Home
Page." in a scratch HOME, with the README's Pi profile (temperature 0.6, top_p 0.95, top_k 20). A run ends at Pi's
exit, at the 30-minute deadline, or after 300 s in which Pi is alive but the server logs nothing (`stalled`).
Finished runs are graded with the task's four checks.

Order: q8-1, mx-1, q8-2, mx-2, q8-3, mx-3, 2 minutes apart (`screen.sh`, `screen.log`). Q8_0 is sha256
`69e0ac02…`, MXFP4_MOE `cf09584a…`, both the published files (each `run.json` has the full hash). `omlx-server`
was running and idle (0.7% CPU, `screen-env.txt`). Evidence: `2026-10-31-mellum-pi-real-use/`, one directory per
run (`server.log`, `run.json`, `grade.txt`, Pi's session and output, the workspace without `.venv`).

## Outcomes

| run | verdict | requests | wall time | how it ended |
|---|---|---|---|---|
| q8-1 | passed | 21 | 51 s | Pi finished; 4/4 checks |
| mx-1 | failed | 5 | 7 s | `finish=error`: "unterminated tool call (stop token ...)"; Pi exits on the provider error |
| q8-2 | failed (stalled) | 17 | 343 s | Mellum ran `uv run --frozen python app.py` (foreground uvicorn); no traffic for 300 s |
| mx-2 | failed | 10 | 18 s | Mellum read `uv.lock`; the next prompt was 38,348 tokens of 32,768 and the server answered 400 |
| q8-3 | passed | 17 | 59 s | Pi finished; 4/4 checks |
| mx-3 | failed (stalled) | 13 | 337 s | the same foreground `app.py` as q8-2 |

Pass counts: **Q8_0 2 of 3, MXFP4 0 of 3** (n = 3 each). Wall time is `run.json`'s `started` to `ended`
(server start to grading, after the model hash). The check run (`check-1`, before the stall stop existed) hit the
foreground server too and idled to the 30-minute deadline.

mx-2's `count_mismatch` (10 server requests, 11 Pi assistant messages) is that 400: the server refused the
request before prefill, so it wrote no request line, and Pi recorded an error message. Pi exited 0 after it.

Recompute: `python3 -c` over `*/run.json` for verdicts; the Pi session files under `*/session/` for the last tool
calls.

## Rates by context

```
python3 tools/mellum_pi_report.py docs/superpowers/research/2026-10-31-mellum-pi-real-use/q8-{1,2,3}
python3 tools/mellum_pi_report.py docs/superpowers/research/2026-10-31-mellum-pi-real-use/mx-{1,2,3}
```

A request is placed by its context at the start (tokens reused from the cache). Prefill medians leave out requests
of fewer than 256 new tokens; n is the number of requests.

**Q8_0** (3 runs, 55 requests):

| context at start | prefill t/s (median) | n | decode t/s (median) | n |
|---|---|---|---|---|
| 0-4,095 | 3,105 | 7 | 124.4 | 25 |
| 4,096-8,191 | 2,558 | 1 | 114.1 | 23 |
| 8,192-16,383 | 2,395 | 3 | 109.3 | 7 |

**MXFP4 experts** (3 runs, 28 requests):

| context at start | prefill t/s (median) | n | decode t/s (median) | n |
|---|---|---|---|---|
| 0-4,095 | 4,008 | 4 | 147.1 | 22 |
| 4,096-8,191 | 2,436 | 2 | 128.7 | 6 |

No run reached 16K tokens, and no session compacted.

What the per-run tables show (each run's table is the first part of the report's output):

- **Cold prefill is within 5% of the synthetic figures.** Request 0 (1,593-1,594 new tokens on an empty cache):
  Q8_0 3,832-3,850 t/s, MXFP4 3,953-4,074 t/s (n = 3 each), against the README's 4,026 / 4,170 at 2,056 tokens.
- **Most real prefills are small, and small prefills are slow.** After request 0, a turn adds a tool result:
  often 18-44 tokens, which run at 280-830 t/s, because each request has a fixed cost. Of the 77 non-cold requests,
  66 added fewer than 256 tokens. In wall time this cost is small (0.05-0.12 s per request).
- **Decode trends down with context.** In the band table above, Q8_0's median falls from 124.4 to 114.1 to 109.3
  t/s (n = 25, 23, 7) and MXFP4's from 147.1 to 128.7 (n = 22, 6). Below 4K that is at or slightly above the
  README's synthetic 2K figures (123.3 / 138.5); at 8-9.6K, Q8_0's 109.3 is below the synthetic 124.5 at 8,195
  tokens, which is one fresh request rather than a growing session.

## Findings

1. **The foreground server is the most common failure** (2 of 6 screen runs, plus the check). The task's
   `AGENTS.md` names `uv run --frozen python app.py` as how to run the app; Mellum runs it through Pi's bash to
   check its work, and plain Pi has no timeout, so the session waits forever. mellum-worker's extension guards
   against this; plain Pi does not. This is a property of plain Pi plus this task, not of the engine.
2. **MXFP4 emitted a stop token inside a tool call once** (mx-1, request 4). The server reports it and Pi ends the
   session. Not seen with Q8_0 here (0 of 55 requests; n is too small to call it a quant difference).
3. **One tool result can overflow the context.** Reading `uv.lock` (mx-2) pushed the next prompt past 32,768
   tokens. Pi does not trim a tool result to fit, and the server refuses instead of truncating.
4. **When a run does finish, it finishes in about a minute** (51 s and 59 s, 17-21 requests).

## What would falsify this

- Rates: a rerun of `tools/mellum_pi_report.py` on these server logs giving other figures, or the same request
  shapes running materially faster or slower in a fresh screen on the same machine and commit.
- Findings 1-3: the session files not showing the tool calls and errors named above.
- The decode trend: decode at 8-9K context in a longer screen not falling below the 0-4K band.

## Risks

- n = 3 per quant. The pass counts and the failure mix will move with more runs; don't read 2/3 vs 0/3 as a quant
  difference.
- One task, one prompt, one Mac. Context never passed 9,583 tokens, so the 16-32K band is unmeasured in real use.
- The 300 s stall rule ends a run that is waiting on a slow tool step with no model traffic; none of the stalled
  runs here was doing anything but serving `app.py` (their last tool call says so).
- `omlx-server` was resident and idle, not stopped.
