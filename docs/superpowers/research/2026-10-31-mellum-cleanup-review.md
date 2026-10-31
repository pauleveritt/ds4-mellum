# ds4-mellum deep-cleaning review (read-only)

Repository: `/Users/pauleveritt/projects/pauleveritt/ds4-mellum`, branch `mellum`, HEAD `6f8d91b`.
Merge base with `upstream/main`: `ee7e8405`. `git diff --shortstat ee7e8405 mellum`: 1,636 files, +162,528 / -608.
Every count below was recomputed from that checkout on 2026-10-31; the command is given where it is not obvious.
Nothing in the repository was edited or run that writes (only `git diff`, `git log`, `git ls-files`, `git merge-tree`
old-style, `grep`, `sed`, and a hashing script in the scratchpad).

## Where the 162K lines are

| Area | Files | Lines (approx) | Bytes | Note |
|---|---|---|---|---|
| `docs/superpowers/research/` | 1,013 | ~101K | ~9.6 MB | 43 top-level `.md` notes; the rest are raw captures in 10 big subdirs (see table below) |
| `tests/mellum_measurements/` | 398 | ~16K | 6.9 MB | 6.1 MB is `d1_decode_profile/` (four `.xz` timelines = 5.6 MB); only the 4 `prompt_*.txt` are used by anything |
| `docs/superpowers/plans` + `specs` | 19 + 12 | 3,078 + 2,363 | | closed-phase process docs; 11 personal absolute paths |
| Product C/ObjC/Metal | | ~9,700 | | `ds4.c` +2,318/-48, `ds4_agent.c` +2,447/-140, `ds4_metal.m` +1,299/-94, `ds4_server.c` +671/-294, `metal/mellum.metal` 689, `ds4_mellum_unicode.inc` 902 (generated), `metal/qwen4.metal` +232/-13, `ds4_gpu.h` +275, `ds4_json.h` 279, `ds4_mellum_{profile,capture_state,observer,routes,stream}.c/.h` ~700, `ds4_mellum_kernels.inc` 61, `ds4_mellum_capture.h` 24 |
| `tools/` | 62 | ~7,100 | | 5 tools are product-relevant (below); the rest is S0-S4 research tooling |
| `tests/` (non-measurement) | ~110 | ~16,500 | | `test_mellum_kernels.c` 3,250, `ds4_test.c` +1,976, `test_mellum_config.c` 1,155, `test_mellum_frontends.c` 1,027, `ds4_agent_test.c` +721, `test_mellum_ssd_replay.py` 726, ... |
| `Makefile` | | +455 | | ~60 Mellum targets |
| `MELLUM.md`, `ROADMAP.md`, `README.md`, `docs/MELLUM_*.md` | | 1,323 + 350 + 312 + 953 | | |

Research subdirs by size (`git ls-files` + `cat | wc -c`):

| dir | files | bytes | has summary.md | referenced from README/MELLUM/ROADMAP |
|---|---|---|---|---|
| `2026-10-29-mellum-imac-stop-profile/` | 56 | 2.46 MB | yes | no |
| `2026-10-28-mellum-s6-b-evidence/` | 43 | 1.71 MB | no | no |
| `2026-10-28-mellum-s6-c-evidence/` | 211 | 1.07 MB | yes | no |
| `2026-10-29-mellum-readme-speed/` | 360 | 0.99 MB | no | README (captures links) |
| `2026-10-29-mellum-stream-decode-ab/` | 82 | 0.68 MB | no | no |
| `2026-10-29-mellum-stream-decode-screen/` | 55 | 0.67 MB | no | no |
| `2026-10-28-mellum-s4-decode-profile-spike/` | 5 | 0.42 MB | no | ROADMAP |
| `2026-10-29-mellum-speed-run/` | 50 | 0.39 MB | yes | ROADMAP, MELLUM |
| `2026-10-29-mellum-s6-d-evidence/` | 29 | 0.18 MB | no | no |
| `2026-10-29-mellum-s6-d-sync-screen/` | 59 | 0.13 MB | no | no |

Commit shape: 765 commits since the base; 227 touch only docs/measurements; 164 touch `ds4.c`, `ds4_metal.m`,
`ds4_agent.c`, `ds4_server.c` or `metal/`.

## Findings that drive the list

1. **Evidence is the bulk.** ~117K of the 162K added lines are captures (research subdirs + `tests/mellum_measurements`),
   kept byte for byte under the project's binding rule 5. They belong in git history, not in the working tree of a
   public fork. Keeping them *reachable* (binding rule 8, "nothing discarded") only needs a tag.
2. **Byte-identical duplicates** (`scratchpad/dups.py`, files >= 4 KB): 760 KB. The 32K prompt (90,008 B) is in 6
   places, the 8K prompt in 6, the 2K prompt in 10: `tests/mellum_measurements/{d1_decode_profile,t1_cli_prefill}/`,
   `tests/mellum_measurements/readme_checks/runs/{prefill,prefill-nax0-before-ba83406}/`, and
   `docs/superpowers/research/2026-10-29-mellum-readme-speed/{m5,m1pro}/{q8,mx}-prefill/` and `imac-static-lock/*/`.
   Only `tests/mellum_measurements/d1_decode_profile/prompt_{100,2048,8192,32768}.txt` are read by anything
   (`tests/mellum_cli_decode_abba.sh` takes the directory as `PROMPTDIR`; the README names it).
3. **Personal paths/hosts.** Product code (`*.c *.h *.m *.metal`, `Makefile`, `tools/`, `tests/` minus measurements)
   is clean. Hits are in docs: `docs/MELLUM_STORAGE_EVIDENCE.md:12` (`/Users/pauleveritt/projects/pauleveritt/ac-demo`),
   8 plan files (`2026-10-27-mellum-{device-observation,storage-evidence,storage-service,workload-source}.md`,
   `2026-10-28-mellum-{generated-cohort,s6-a-cpu,s6-b-m5}.md`, `2026-10-29-mellum-s6-d-prefetch.md`, the last with
   `pauleveritt@herveline.local`), 2 specs (`2026-10-27-mellum-storage-evidence-design.md:35`,
   `2026-10-29-mellum-stream-prefetch-design.md:91`, the latter with `herveline.local`), and 50 files under
   research/measurements. (`tests/long_context_security_prompt.txt` has `/Users/antirez/...`: upstream's, leave.)
4. **Research-tooling layers for closed phases.** ROADMAP S0-S4 ("complete") produced: route capture/spool
   (`ds4_mellum_routes`, `capture_state`, `profile`, `observer`, `capture.h`, `kernels.inc`), an offline replay
   (`tools/mellum_ssd_replay.py`), storage evidence (`tools/mellum_read_*`, `mellum_stage_*`, `mellum_storage*.py`,
   `mellum_workload_bundle.py`, `mellum_service_*.py`, `mellum_device_*.py`, `mellum_observe_*.py`,
   `mellum_expert_{index,pack}.py`, `mellum_gguf*.py`), a generated cohort (`tools/mellum_generate*.c`,
   `mellum_generated_*.py`), plus their tests and four `docs/MELLUM_*.md`. The *product* streaming path
   (`ds4_mellum_stream.c`, the gate in `ds4_metal.m`, slabs) does not use any of it. The capture layer is the one
   that touches shared code: `ds4_metal.m` includes `ds4_mellum_observer.h` and calls `mellum_observer_encode()` at
   13 dispatch sites inside upstream's Q8 matmul / get_rows / rms_norm paths, which is why every Metal test binary
   (`test_qwen4_kernels`, `test_glm53_kda`, `test_mxfp4_metal`, `test_glm_attention`) now links
   `ds4_mellum_observer.o ds4_mellum_profile.o` (Makefile lines 171, 885, 900, 907, and the `test_glm_attention` rule).
   `ds4.c` holds the adapter (`struct ds4_mellum_capture` at line 160, `ds4_mellum_capture_*` at 89669-89800, hooks at
   61034-61037, 61184, 61211-61280, 61315, and `qwen4_graph.mellum_capture` at 59389).
5. **Dead experiment paths still in product code** (all env-switched, none user-facing):
   - `DS4_MELLUM_ATTN_TILED=1`: the f32-operand tiled attention (`kernel_mellum_attn_tiled`, `metal/mellum.metal`
     393-544; `ds4_gpu_mellum_attn_tiled_tensor`, `ds4_metal.m` ~51355-51400; the ternary at `ds4.c:60974`).
     README "Didn't work": twice the attention time. The landed path is `_tiled_half`.
   - B0 two-row verify (decode research D2, "B0 fails as the code stands", nothing landed):
     `ds4_test_mellum_b0_arm` (`ds4.c:89635`), `--mellum-b0` in `tests/ds4_test.c` (1561-1643, table row 9288),
     `tests/mellum_b0.sh`, `DS4_TEST_MELLUM_B0_CTX`.
   - Decode-research scripts with nothing landed behind them: `tests/mellum_decode_profile.sh`,
     `tests/mellum_decode_profile_summary.py`, `tests/mellum_decode_bytes.py`, `tests/mellum_copy_bandwidth.m`,
     `tests/mellum_prompt_lookup.py` (D1/D3; the landed S1/S2 kernels are pinned by `test_mellum_kernels` and the oracle).
   - Phase D prefetch left **no** code: the prefetch is `prefetch-sync.patch` in the research dir. What landed from D
     (slot-return on failed reads, stop-scratch gating, `load_begin`/`discard` cheaper stops) is product and tested.
   - `DS4_MELLUM_MOE_NAX=0` and `DS4_MELLUM_EXACT_PREFILL=1` are diagnostics with tests behind them
     (`--mellum-exact-prefill`, the runners' arm B); keep.
6. **Stale statements** (will mislead a reader now):
   - `ROADMAP.md:323` "`ds4-server` does not serve Mellum, so the workload runs flattened" (false since phase 6).
   - `MELLUM.md:568` "ds4-server still refuses Mellum" (same).
   - `MELLUM.md:957-985` "README checks (nothing pushed)", "waits for Paul's repository and push", "Waiting: the clone
     command, until Paul creates `pauleveritt/ds4-mellum`" (the repository is public).
   - `README.md:290` "about 24,000 added engine lines across 91 files today" (recompute after the cleanup).
   - `docs/MELLUM_ROUTE_CAPTURE.md:3-5` "its live gates are still unrun" and `docs/MELLUM_SSD_REPLAY.md:4` "No real
     Mellum routing traces have been collected" (both superseded by S2-S4; both docs go with the tooling, item A5).
7. **Upstream's 13 commits merge cleanly.** `git merge-tree ee7e8405 mellum upstream/main` (old style, writes nothing)
   produced **zero** conflict markers. Six files overlap: `.gitignore`, `Makefile`, `ds4.c`, `ds4_metal.m`,
   `ds4_server.c`, `metal/qwen4.metal`. Semantic re-checks are listed under Q4.
8. **Shared-code divergence that is legitimately product** (leave, but each is a candidate upstream PR):
   `ds4_gpu_dispatch_gated()` at 13 sites (streamed stops suppress later dispatches), the expert-cache scan bound
   `g_stream_expert_cache_{layer,expert}_end` (generic speedup), `ds4_gpu_stream_expert_pending_load_abandon` (a slab-slot
   leak fix, generic), `ds4_json.h` (server's JSON readers shared with the agent, -294 in `ds4_server.c`),
   `ds4_chat_append_tool_results`, `ds4_session_rewind_frontier`, `stream_skip_think_open`, the `ungated` word in
   `metal/qwen4.metal` (4 added lines mention it) and the Q8_0 branch in `qwen4_load_raw16`/`qwen4_dequant_raw16`.
9. `.gitignore` is missing `/tests/test_prompt_prefix` (an upstream `make test` binary; shows as untracked). One line.
10. `tests/mellum_read_plan_v1.bin` (192 B) is the only tracked binary the fork added outside evidence; it goes with A5.

## Commit rule for the executor (from `ROADMAP.md` "Working rules" and the project memory)

One concern per commit; about 150 product lines; never over 300 lines per commit **excluding generated files and pure
deletions**; a red test goes in its own commit before the fix it motivates; commit messages name the concrete behavior.
Pure moves (A1-A4, B1-B4) are reviewed with `git diff --color-moved=dimmed-zebra`; split a move by concern so each
commit's non-move hunks stay under 300 lines. Gate every commit with the fast checks in "Build and test" (never `;`
before `git commit`; use `&&`).

Before the first deletion, make the evidence reachable:

```sh
git tag mellum-pre-cleanup-2026-10-31 6f8d91b
git branch mellum-evidence-2026-10-31 6f8d91b
```

Every removed evidence directory is then cited as `mellum-evidence-2026-10-31:<path>` from the note that owned it.
(Push of the tag and branch is the owner's; do not push.)

## (a) Safe removals and moves, ranked

Lines are from `git ls-files | xargs wc -l` on the named paths. "Behavior" means user-visible product behavior.

**A1. Move raw research captures out of the tree.** Remove every non-`.md` file under `docs/superpowers/research/`
(keep the 43 top-level notes and each subdir's `summary.md`/`README.md`/`KIT.md`; `s6-c-prereg/KIT.md` is cited by
ROADMAP). Then in each top-level note and in `README.md:142,165` (the two "captures" links), `ROADMAP.md:285,315,323`
and `MELLUM.md:69-71,80`, replace the directory link with the tag path. Drop the eight `-whitespace` rules in
`.gitattributes` (lines 4-27) and the four research `raw/` and `*.logits.f32` ignores at the end of `.gitignore`.
- Removes: ~970 files, ~98K lines, ~9.5 MB. Pure deletion: one commit per research subdir is fine, or one commit for all
  (deletions are exempt from the cap); the link edits are a separate commit (~30 lines).
- Risk: a dangling link. Check: `git grep -n 'superpowers/research/.*/' -- README.md MELLUM.md ROADMAP.md docs/superpowers/research/*.md`
  and confirm each target exists or names the tag. No test touches these files (`git grep -l 'superpowers/research' -- tests tools Makefile` is empty today).
- Behavior: none.

**A2. Move `tests/mellum_measurements/` out; keep the four prompts.** `git mv tests/mellum_measurements/d1_decode_profile/prompt_{100,2048,8192,32768}.txt tests/mellum_prompts/`,
delete the rest. Update `tests/mellum_cli_decode_abba.sh:3-4` (comment), `README.md:212` (the runner command's third
argument), `MELLUM.md` (its `tests/mellum_measurements/` mentions: `grep -n mellum_measurements MELLUM.md`), and the
links in `MELLUM.md:961` to the `readme_checks/` logs (tag path).
- Removes: 394 files, ~16K lines, 6.9 MB (the four prompts stay, 124 KB). Also removes 6 of the 10 duplicate-prompt copies;
  the other 4 go with A1.
- Risk: a runner that cannot find its prompts. Check: `grep -rn mellum_measurements tests tools Makefile README.md`
  returns nothing, and `tests/mellum_cli_decode_abba.sh` still resolves `$PD/prompt_$target.txt` for the four lengths
  (dry read: `ls tests/mellum_prompts/`).
- Behavior: none.

**A3. Move the closed-phase plans out of the tree.** Delete `docs/superpowers/plans/2026-10-2{7,8,9}-mellum-*.md`
(19 files, 3,078 lines) after confirming every one is for a phase ROADMAP marks complete/merged/paused (S0-S6). Where
`ROADMAP.md` links a plan (lines 300, 310, 328), point at the tag path. This also removes 8 of the 11 personal paths and
one `herveline.local`.
- Removes 3,078 lines. Pure deletion + ~5 link edits.
- Risk: none to code. Check `git grep -n 'superpowers/plans' -- README.md MELLUM.md ROADMAP.md docs` afterwards.
- Behavior: none.
- If the owner wants the SDD record kept in-tree, keep the plans but edit the 8 personal paths (see A8). Marked as the
  owner's taste; the report recommends removal because the plans are checklists for work already merged.

**A4. Decode-research scripts with nothing landed.** Delete `tests/mellum_decode_profile.sh` (94), `tests/mellum_decode_profile_summary.py` (156),
`tests/mellum_decode_bytes.py` (150), `tests/mellum_copy_bandwidth.m` (107), `tests/mellum_prompt_lookup.py` (101),
`tests/mellum_b0.sh` (31), and `tools/mellum_background_load.sh` (20), `tools/mellum_machine_facts.sh` (21) (both
referenced only from plans). Drop `/tests/mellum_copy_bandwidth` from `.gitignore`. Their MELLUM.md "Measured" rows stay
(they carry commands and n); add "(script at `mellum-evidence-2026-10-31:tests/...`)" where a row names one.
- Removes ~680 lines. No Makefile target builds any of these (checked: none appear in `Makefile`).
- Risk: none. Check `make test-mellum` still resolves.
- Behavior: none.

**A5. Remove the S0-S4 research tooling (storage evidence, service, generated cohort, replay, observe, device).**
These are the Python/C tools and tests under `tools/mellum_*` and `tests/test_mellum_*` that ROADMAP S0-S4 produced and
that the product does not call. Delete, with their Makefile targets:

- tools: `mellum_ssd_replay.py` (586), `mellum_replay_request.py` (79), `mellum_routes_to_trace.py` (261),
  `mellum_capture_publish.py` (127), `mellum_capture_identity.py` (72), `mellum_capture_metadata.py` (153),
  `mellum_artifact.py` (29), `mellum_workload_bundle.py` (128), `mellum_storage.py` (68), `mellum_storage_common.py` (183),
  `mellum_read.h` (44), `mellum_read_bench.c` (128), `mellum_read_plan.c` (99), `mellum_read_io.c` (55),
  `mellum_read_run.c` (116), `mellum_read_gate.c` (43), `mellum_read_plan.py` (109), `mellum_read_batch.py` (304),
  `mellum_stage.h` (8), `mellum_stage_bench.c` (101), `mellum_stage_io.c` (66), `mellum_service_plan.py` (103),
  `mellum_service_run.py` (208), `mellum_service_verify.py` (75), `mellum_device_classify.py` (68),
  `mellum_device_observer.py` (197), `mellum_observe_batch.py` (164), `mellum_observe_manifest.py` (141),
  `mellum_observe_process.py` (106), `mellum_expert_index.py` (134), `mellum_expert_pack.py` (161),
  `mellum_gguf.py` (112), `mellum_gguf_reader.py` (68), `mellum_generate.c` (58), `mellum_generate_driver.c/.h` (147),
  `mellum_generate_inspect.c` (16), `mellum_generate_io.c/.h` (58), `mellum_generate_loop.c/.h` (76),
  `mellum_generated_analysis.py` (40), `mellum_generated_inputs.py` (154), `mellum_generated_publish.py` (68),
  `mellum_generated_schedule.py` (63), `mellum_generated_validation.py` (96). About 5,300 lines.
- tests: `test_mellum_ssd_replay.py` (726), `test_mellum_replay_request.py` (120), `test_mellum_routes_to_trace.py` (418),
  `test_mellum_capture_publish.py` (345), `test_mellum_capture_build.py` (45), `test_mellum_capture_state_export.py` (93),
  `test_mellum_artifact.py` (109), `test_mellum_workload_bundle.py` (196), `test_mellum_read_*.{c,py}` (~700 with the
  `mellum_read_*_faults.*` headers), `test_mellum_stage.{c,py}` + `mellum_stage_faults.*` (152), `test_mellum_service_*.py` (363),
  `test_mellum_device_*.py` (221), `test_mellum_observe_*.py` (414), `test_mellum_expert_*.py` (473),
  `test_mellum_gguf_reader.py` (110), `test_mellum_generate_*.c` + `mellum_generate_driver_fault.h` (199),
  `test_mellum_generated_*.py` (412), `test_mellum_s4_device_reads.py` (54, referenced by nothing),
  `mellum_read_plan_v1.bin`, `mellum_routes_compare.py` (39). About 5,200 lines.
- docs: `docs/MELLUM_ROUTE_CAPTURE.md` (395), `docs/MELLUM_SSD_EVALUATION.md` (69), `docs/MELLUM_SSD_REPLAY.md` (254),
  `docs/MELLUM_STORAGE_EVIDENCE.md` (235) = 953 lines (and the one personal path at `MELLUM_STORAGE_EVIDENCE.md:12`).
  Remove the "SSD streaming research tools, for contributors" bullet in `README.md:299-303`; add one line to `ROADMAP.md`
  S0-S4 saying the tooling lives at the tag.
- Makefile: the `test-mellum-storage`, `test-mellum-device-observation`, `test-mellum-service`, `test-mellum-stage`,
  `test-mellum-read-*`, `test-mellum-generated-*`, `test-mellum-ssd-replay`, `test-mellum-replay-request`,
  `test-mellum-workload-source`, `test-mellum-capture-publish`, `test-mellum-capture-build`, `test-mellum-routes` (Python
  half) rules, their `clean-mellum-*` rules and their `test-mellum:` prerequisites (Makefile 1096-1101, 1104-1107,
  1612-1631, 1633-1760, 1793-1840 region). About 250 Makefile lines.
- **Keep** (product-relevant tooling): `tools/mellum_s4_measure.c/.py` + `tests/test_mellum_s4_measure.py`,
  `tests/test_mellum_s4_driver.py` (the measurement tool behind README's speed rows and the iMac result),
  `tools/mellum_metal_limit.m` (device facts for the 8 GB plan), `tools/mellum_llama_logits.c`,
  `tools/mellum_logits_agree.py`, `tests/mellum_logits_compare.py`, `tests/test_mellum_logits_agree.py`,
  `tests/test_mellum_logits_compare.py` (the quality check against llama.cpp for MXFP4/Q8; it is how a future quant
  gets accepted), `tests/test_mellum_runners.sh`, the oracle (`tests/mellum_trace.c`, `mellum_trace_context.h`,
  `mellum_oracle_*.tokens`, `mellum_oracle_compare.sh`, `mellum_heldout_*.tokens`, `gen_mellum_*.py`).
- Dependency check before deleting any file F: `git grep -n "$(basename F .py)" -- Makefile tests tools README.md MELLUM.md ROADMAP.md`
  must show only files in this same list. Known cross-links: `mellum_s4_measure.py` and `mellum_gguf.py` (verify with the
  grep; if `mellum_s4_measure.py` imports `mellum_gguf`, keep `mellum_gguf.py` + its test).
- Removes ~11,700 lines. Do it in slices by phase (storage evidence; service; generated cohort; replay/routes Python;
  observe/device; docs) so each commit's Makefile edit stays reviewable; file deletions are exempt from the cap.
- Risk: `make test-mellum` prerequisites that no longer exist (make fails loudly); a kept tool importing a removed module
  (the Python tests under `make test-mellum` catch it). Run `make test-mellum` after each slice.
- Behavior: none.

**A6. Remove the route-capture instrumentation from the product (the C modules and their hooks).** This is the one
removal that touches product files; it is the tail of A5 and should follow it.
- Delete `ds4_mellum_routes.c/.h` (114), `ds4_mellum_profile.c/.h` (215), `ds4_mellum_capture_state.c/.h` (165),
  `ds4_mellum_observer.c/.h` (67), `ds4_mellum_capture.h` (24), `ds4_mellum_kernels.inc` (61).
- `ds4.c`: remove `#include "ds4_mellum_capture.h"` (45), `"ds4_mellum_capture_state.h"` (57), `"ds4_mellum_observer.h"` (58),
  `struct ds4_mellum_capture` (160-16x), the `mellum_capture` field at 59389, the hooks at 61034-61037 (`mellum_observer_projection_role`),
  the `g->mellum_capture ||` halves of the conditions at 61184 and 61315, the capture/observer calls at 61211-61214,
  61226, 61276-61280, and the adapter 89669-89800. About 190 lines.
- `ds4_metal.m`: remove `#include "ds4_mellum_observer.h"` (27) and the 13 `(void)mellum_observer_encode(...)` lines
  (`grep -n mellum_observer_encode ds4_metal.m`). 14 lines; this is what un-touches upstream's dispatch paths.
- Makefile: `MELLUM_CAPTURE_OBJS`/`MELLUM_CAPTURE_HEADERS` and the `override CORE_OBJS`/`CPU_CORE_OBJS` lines (73-77);
  `ds4_mellum_observer.o ds4_mellum_profile.o` on lines 171, 885, 900, 907 and the `test_glm_attention` rule (restoring
  upstream's link lines); the `ds4.o`/`ds4_cpu.o`/`ds4_metal.o` header deps (759, 825, 846); targets
  `test-mellum-profile`, `test-mellum-capture-state`, `test-mellum-capture-api`, `test-mellum-observer`,
  `test-mellum-capture-receipt`, `test-mellum-routes-c`, `test-mellum-routes-gpu`, `test-mellum-capture-failures-gpu`
  and their `test-mellum:` prerequisites. Keep `ds4_mellum_stream.o` in `CORE_OBJS` (product).
- tests: delete `test_mellum_profile.c` (188), `test_mellum_capture_state.c` (239), `test_mellum_capture_api.c` (118),
  `test_mellum_observer.c` (72), `test_mellum_capture_receipt.c` + `mellum_capture_receipt.h` (28), `test_mellum_routes.c` (173),
  `test_mellum_routes_gpu.sh` (18), `mellum_routes_gpu.c` (273), `mellum_capture_failures_gpu.c` (131). About 1,240 lines.
  **Keep** `tests/test_metal_tensor_lifetime.m` (it pins `ds4_gpu_tensor_contents/read/write` not retaining freed buffers,
  a product change in `ds4_metal.m` 9326-9368) and its `test-metal-tensor-lifetime` target.
- Removes ~2,100 lines, of which ~210 are product lines in shared files. Two commits: (1) the hook removal in `ds4.c` +
  `ds4_metal.m` + Makefile link lines (~230 lines, product); (2) module and test deletions (exempt).
- Risk: a build break (make catches it); a changed forward pass (impossible if only `(void)` calls and the
  `g->mellum_capture` branches go, but prove it): `make test-mellum-kernels`, model-gated `./ds4_test --mellum-exact-prefill
  --mellum-attn-tiled --mellum-rewind --mellum-moe` and `make test-mellum-trace` must pass unchanged, and
  `make test-qwen4-kernels` / `make test-mxfp4-metal` (the upstream Metal tests that lose the extra objects) must build and pass.
- Behavior: none (the capture was only reachable through `ds4_mellum_capture_create`, which no front end calls:
  `git grep -l ds4_mellum_capture_create -- ds4_cli.c ds4_agent.c ds4_server.c` is empty).

**A7. Remove the f32 tiled-attention variant and the B0 verify arm (dead experiments).**
- f32 tiles: delete `kernel_mellum_attn_tiled` (`metal/mellum.metal` 393-544, ~150 lines; keep `_tiled_half`),
  `ds4_gpu_mellum_attn_tiled_tensor` in `ds4_metal.m` (~51355-51400, the dispatch helper shared with `_half` stays) and
  its declaration in `ds4_gpu.h`; in `ds4.c:60974` make the tiled path unconditionally `_tiled_half` and collapse
  `g->mellum_attn_tiled` to on/off (`DS4_MELLUM_ATTN_TILED=0` stays as the runners' arm B; `=1` goes). Remove the
  `test_tiled_swa_session`, `test_tiled_full_rows`, `test_tiled_refusals` f32 cases in `tests/test_mellum_kernels.c`
  (the `_half` siblings stay) and any `"1"` arm in `test_mellum_attn_tiled` (`tests/ds4_test.c` 1451-1560). Update
  `MELLUM.md:24-27` and the `DS4_MELLUM_ATTN_TILED` sentence (line 55). About 300 lines removed, ~40 product lines changed.
  Behavior: changes only under `DS4_MELLUM_ATTN_TILED=1`, which README lists under "Didn't work". Check:
  `make test-mellum-kernels`, model-gated `./ds4_test --mellum-attn-tiled`, `make test-mellum-trace`.
- B0: delete `ds4_test_mellum_b0_arm` (`ds4.c:89635`-end of function, ~60 lines; it is a test-only export), the
  `--mellum-b0` test (`tests/ds4_test.c` 1561-1643 and row 9288), `tests/mellum_b0.sh` (A4), and `DS4_TEST_MELLUM_B0_CTX`
  mentions. MELLUM.md's D2 row stays (measurement record). ~160 lines. Behavior: none. Check: `make ds4_test && ./ds4_test`.
- Marked **guess**: whether `ds4_gpu_mellum_attn_tiled_tensor`'s dispatch helper is shared with `_half` (the header
  comment at `ds4_metal.m:51355` says "f32 or half", so likely one helper with a kernel-name parameter).

**A8. Scrub the remaining personal paths and hosts.** After A3/A5: `docs/superpowers/specs/2026-10-27-mellum-storage-evidence-design.md:35`
and `2026-10-29-mellum-stream-prefetch-design.md:91` (`/Users/pauleveritt/...`, `pauleveritt@herveline.local`) become
`<your checkout>` / "the 8 GB iMac, over SSH". If A3 is declined, do the same in the 8 plan files. Re-run the check:
`git grep -n -I -E '/Users/(pauleveritt|koudai)|herveline|\.local\b' -- . ':!tests/long_context_security_prompt.txt'` must be empty.
~12 lines. Behavior: none.

**A9. Fix the stale statements** (one docs commit): `ROADMAP.md:323`, `MELLUM.md:568`, `MELLUM.md:957-985` (rewrite the
"README checks" header to "phase 9, 2026-10-28" and drop the "waiting for the push" sentences; the repository is public),
`README.md:290` (recompute: `git diff --shortstat <new base> mellum -- '*.c' '*.h' '*.m' '*.metal' '*.inc' Makefile`
after the cleanup and write that number with its command). ~15 lines.

**A10. Housekeeping.** Add `/tests/test_prompt_prefix` to `.gitignore` (upstream's omission; 1 line). Collapse the
dozen `clean-mellum-*` rules into one `clean-mellum` that removes the surviving Mellum test binaries (~40 Makefile lines
fewer). Behavior: none.

Running total for (a): about 130K lines removed (A1 ~98K, A2 ~16K, A5 ~11.7K, A3 ~3.1K, A6 ~2.1K, A4-A10 ~1.2K),
of which about 250 are product lines in shared files (A6, A7).

## (b) Refactors of product code (smaller, more reviewable divergence; no behavior change)

Ordered by payoff per risk. Each is a pure move or a narrowing; run the full gate after each.

**B1. Pull the Mellum engine code out of `ds4.c` into `ds4_mellum.inc`** (included from `ds4.c` the way
`ds4_mellum_unicode.inc` already is at line 44737). Candidates, by the function list the diff adds (all `static`, all
Mellum-only): config/validate/bind (`ds4_model_is_mellum`, `ds4_mellum_pattern_rule`, `ds4_mellum_layer_is_sliding`,
`weights_mellum_layer_has_required`, `mellum_expert_type`, `weights_validate_mellum_layout`, `config_refuse_mellum_key`,
`config_mellum_range`, `config_mellum_positive_f32`, `config_validate_mellum_model`, `weights_bind_mellum_layer`,
`weights_model_map_mellum_static_spans`), memory plan (`mellum_clamp_chunk`, `mellum_prefill_cap`, `mellum_group_k`,
`mellum_group_rows`, `mellum_requests`, `mellum_context_memory`, `ds4.c` 17491-17705), tokenizer (`mellum_char_at`,
`mellum_piece_end`, `mellum_segment_pieces`, `mellum_emit_piece`, `bpe_tokenize_text_mellum`, 44726-44850), chat
(`mellum_chat_assistant_prefix`, `mellum_chat_system`, `qwen4_tool_response_tag`), and the graph block 59497-61700
(`mellum_graph_*`, `mellum_stream_*`, `mellum_trace_*`, `mellum_rope_params`, `mellum_proj`) plus the test exports at
89513-89800 (`ds4_test_mellum_*`). What stays in `ds4.c` are the ~54 `ds4_model_is_mellum()` branch sites in shared
functions (`ds4_engine_open_internal` 7 hunks, `ds4_session_create` 6, `ds4_chat_append_message` 4, `vocab_load`,
`ds4_session_rewind`, ...), which a reviewer wants to see in place.
- Moves ~1,500 lines; `ds4.c`'s diff against upstream drops from +2,318 to roughly +600. Do it in five move commits
  (config/bind, memory plan, tokenizer+chat, graph+stream, test exports), each verified with `--color-moved` showing
  only moved blocks. The include must sit after the static helpers each block uses (the graph block uses
  `qwen4_graph_*` statics; keep the include at the same position the graph code occupies today, ~line 59497, and a
  second include for the tokenizer block if forward declarations get awkward).
- Risk: a static used before its definition (compile error, caught by `make`); nothing else, since C `#include` of
  `static` code is textual. Gate: `make` with 0 warnings, `make test-mellum`, `make test-mellum-kernels`, model-gated
  `make test-mellum-trace` and `./ds4_test --mellum-exact-prefill --mellum-attn-tiled --mellum-rewind --mellum-moe --mellum-session`
  (byte-equal logits are what `--mellum-exact-prefill` and the trace oracle check).
- Behavior: none.

**B2. Same for the front ends.** `ds4_agent.c`: the Hermes parser (`agent_hermes_*`, `agent_qwen_tool_parse` region,
2522-3160, ~630 lines) to `ds4_agent_hermes.inc`, and Pi's tools (`pi_*`, 10336-10930, ~590 lines) to
`ds4_agent_pi_tools.inc`. `ds4_server.c`: the Hermes renderer/parser (`append_hermes_*`, `render_hermes_*`,
`parse_hermes_generated_message_ex`, `hermes_*`, `build_*hermes*`, 3416-3690 and 6752-6900, ~430 lines) to
`ds4_server_hermes.inc`. `tests/ds4_test.c`: the Mellum tests 540-1960 and 8826-9280 (~1,900 lines) to
`tests/ds4_test_mellum.inc`, leaving the table rows 9286-9302 in place.
- Moves ~3,550 lines; leaves `ds4_agent.c` at roughly +900 (the shared hooks: `agent_tool_syntax_*`, sampling defaults,
  observation caps, `worker_run_turn`'s 10 hunks), `ds4_server.c` at roughly +250.
- Risk/gate as B1 plus `./ds4_test --server`, `./ds4_agent_test`, `make test-mellum` (the three `test_mellum_frontends_*`
  binaries `#include` `ds4_cli.c`, `ds4_agent.c`, `ds4_server.c` and will fail to compile if an include path is wrong).
- Behavior: none.

**B3. Narrow `ds4_metal.m`.** The Mellum block is already contiguous (51087-~51760, +657) and could move to
`ds4_mellum_metal.inc` for symmetry (low value, ~660 lines). More useful: after A6, re-check the 16 hunks in
`ds4_gpu_matmul_q8_0_legacy_tensor`; what remains is `ds4_gpu_dispatch_gated` at 13 sites (product). Leave those.
Priority: low.

**B4. Make `ds4_mellum_stream.c` the only home of streaming policy.** Today the slot/group arithmetic is in
`ds4_mellum_stream.c` (124 lines, model-free tested by `tests/test_mellum_stream.c`, 415 lines), the protocol
(`mellum_stream_encode/finish/load/load_begin/discard/cancelled/decode`) is in `ds4.c`, and the gate/slab mechanics in
`ds4_metal.m`. That split is reasonable (CPU-testable policy vs. GPU mechanics); do not merge them. Only move the `ds4.c`
`mellum_stream_*` ops with B1's graph slice. No separate action.

**B5. Trim `tests/test_mellum_kernels.c` (3,250) only if a case duplicates the oracle.** Its 257 helper lines plus
`tests/mellum_moe_ref.inc` (179, shared with `test_mellum_moe_ref.c` and `ds4_test.c`) are reference arithmetic the
model-free gate depends on; the 47 `test_*` functions each pin a kernel contract or a refusal (the project's binding
rule 4: every refusal test has a sibling success test, which is why `test_*_refusals` come in pairs). After A7 removes the
three f32-tile tests, leave the file. Do not "shrink without losing coverage" here; the coverage is the point.

**B6. Hoist the three Mellum env switches into one place.** `DS4_MELLUM_EXACT_PREFILL`, `DS4_MELLUM_MOE_NAX`,
`DS4_MELLUM_ATTN_TILED` are read in `mellum_graph_alloc` (`ds4.c:59545-59557`), `DS4_MELLUM_TRACE[_AT]` at 60871-60872
and again at 89680-89681, `DS4_MELLUM_STREAM_PREFILL_GROUP` at 74588. A `mellum_diagnostics_from_env()` returning a
small struct (~30 lines) replaces five `getenv` sites and documents in one comment which are diagnostics. Optional;
~0 net lines. Behavior: none (same strings, same defaults). Gate: `./ds4_test --mellum-exact-prefill --mellum-attn-tiled
--mellum-stream-group`.

## (c) Leave alone, with the reason

- **Hermes parser, renderer, Pi tool implementations, bare-quote repair, result cap, rewind frontier** (`ds4_agent.c`,
  `ds4_server.c`, `ds4.c`): user-facing behavior, each pinned by a red-then-green pair in `tests/ds4_agent_test.c`,
  `tests/test_mellum_frontends.c`, `tests/ds4_test.c --server` and `--mellum-server-*`. Move (B2), never edit.
- **The oracle:** `DS4_MELLUM_TRACE[_AT]`, `tests/mellum_trace.c`, `mellum_trace_context.h`, `mellum_oracle_*.tokens`,
  `mellum_oracle_compare.sh`, `mellum_heldout_*.tokens`, `make test-mellum-trace`, `make test-mellum-fastpath`. This is
  how "prefill equals decode byte for byte" and "logits equal the old engine" are re-checked; MELLUM.md cites it for
  every landed kernel.
- **`DS4_MELLUM_EXACT_PREFILL`, `DS4_MELLUM_MOE_NAX=0`, `DS4_MELLUM_ATTN_TILED=0`:** diagnostics with tests and
  runners behind them (`--mellum-exact-prefill`; the ABBA runners' arm B). `=0` arms also matter for pre-M5 chips
  (the simdgroup path is the product path there).
- **Generated fixtures and their generators:** `ds4_mellum_unicode.inc` (902), `tests/mellum_tokenizer_cases.inc`,
  `tests/mellum_chat_ids.inc`, `tests/mellum_server_cases.inc`, `tests/gen_mellum_*.py`. Generated files are exempt from
  the cap and are the tokenizer/template contract.
- **Streaming product code:** `ds4_mellum_stream.c/.h`, the gate (`ds4_metal.m` 11481-11600), `ds4_gpu_dispatch_gated`
  (13 sites), slabs, `g_mellum_stream_protect_*`, `pending_load_abandon`, the `layer_end/expert_end` scan bound,
  `ds4_gpu_test_stream_fail_read/slots` (test hooks that back `--mellum-stream-read-fault`). The scan bound and the
  abandon fix look generic and upstreamable; note them in README "Offer the patches upstream", do not revert.
- **`ds4_json.h`:** a move of the server's readers so the agent can share them; `./ds4_test --server` unchanged. An
  upstream PR candidate. Reverting would mean duplicating ~280 lines into `ds4_agent.c`.
- **`metal/qwen4.metal` changes:** the `ungated` word (4 lines) and the Q8_0 branch in `qwen4_load_raw16`/
  `qwen4_dequant_raw16`, gated by `ds4_gpu_qwen4_moe_q8_tiles`. Minimal already; a model-free kernel test checks other
  models' bytes are unchanged (`test_moe_q8_tiles_mellum_only`).
- **Public API additions in `ds4.h`** (`ds4_engine_is_mellum`, `ds4_engine_sampling_defaults`,
  `ds4_chat_append_tool_results`, `ds4_token_stop_class`, `ds4_session_rewind_frontier`,
  `ds4_session_set_test_rewind_frontier`): each has callers in two or three front ends.
- **`kernel_mellum_attn_swa_ref`** (a reference kernel in the product shader, used only by `test_mellum_kernels`):
  the shader source is loaded from `metal/` at run time, so a test cannot supply its own kernel without a second source
  mechanism. 100 lines; leave.
- **`tests/test_metal_tensor_lifetime.m`:** pins a product fix in `ds4_gpu_tensor_contents/read/write`.
- **`tools/mellum_s4_measure.c/.py`, `tools/mellum_metal_limit.m`, the llama.cpp logits agreement tools:** the
  measurement and quality tooling README's tables and any future quant acceptance depend on.
- **`MELLUM.md` and `ROADMAP.md`:** the record; edit only the stale lines (A9) and the evidence links (A1-A3).
- **The specs (12 files, 2,363 lines):** the SDD record of decisions; scrub two paths (A8). Owner's taste whether to
  move them with the plans.
- **Upstream content the fork did not touch** (`gguf-tools/imatrix/dataset/*` 46 MB, `tests/test-vectors` duplicates,
  `dir-steering/out/verbosity.f32`, `misc/*.md`): not this fork's business.

## Q4. Merge upstream's 13 commits? Yes, first, with gates.

`git merge-tree ee7e8405 mellum upstream/main` shows **no conflicts**. Merge `upstream/main` (`c97964b`) as the first
commit of the cleanup, run the whole gate (below) immediately, and only then start A1. Reasons: every line count,
`README.md:290`, the README's "as of the commit this branch is based on" divider text and the oracle/measurement
"at commit X" references then refer to one base; the later move commits (B1, B2) are not re-done on a second base; and
a textual no-conflict merge is cheapest to review on its own. If any gate fails after the merge, `git reset --hard
mellum-pre-cleanup-2026-10-31`, write the failure into `MELLUM.md` under a dated note, and proceed with the cleanup
without the merge.

Semantic re-checks the gate must cover (upstream commits that touch code Mellum shares):
- `a8ce74a server: render blank reasoning as absent` touches the reasoning rendering next to `stream_skip_think_open`
  and `reasoning_content`: `./ds4_test --server`, `./ds4_test --mellum-server-ids` (model), the three
  `test_mellum_frontends_*`.
- `e73e516 metal: reuse Q4_K raw loads in Qwen NAX prefill` touches `qwen4_load_raw16`/`qwen4_dequant_raw16`, where
  Mellum added the Q8_0 branch: `make test-mellum-kernels`, `make test-qwen4-kernels`, model-gated `./ds4_test --mellum-moe`.
- `a4c922c Fix a Metal session memory leak on /ctx` and `381808d`: Mellum sessions own extra ring buffers
  (`mellum_graph_forget_ring`, `mellum_graph_alloc`): `./ds4_test --mellum-session` (allocation equals the plan) and
  `tests/test_metal_tensor_lifetime`.
- `347c2e1`/`64baaa0`/`c97964b` (Qwen PLE prefetch and token validation): Mellum has its own forward
  (`mellum_graph_forward_tokens`) but shares `qwen4_graph_forward_tokens`'s entry (3 hunks there): `make test-mellum-trace`.

## Build and test commands (run before the first commit and after every commit; model-gated ones after A6, A7, B1, B2 and the merge)

From `Makefile` and `MELLUM.md`. Model path for the gated tests: `DS4_TEST_MELLUM_MODEL=$HOME/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf`
(the MXFP4 rows use the `MXFP4_MOE` file with the same variable where a test says so; `tests/test_mellum_runners.sh`
and `tools/mellum_s4_measure.py` document theirs).

Fast gate, no model (every commit):

```sh
make 2>&1 | tee /tmp/build.log; grep -c warning /tmp/build.log   # expect 0 warnings
make test                                   # upstream's model-free suite (ds4_test, ds4_agent_test, ds4-eval, ...)
make test-mellum                            # Mellum model-free: config, tokenizer, three frontends, stream, moe-ref, trace-stream, logits-agree, s4-measure, ... (after A5/A6 the list shrinks; it must still build and pass)
make test-mellum-kernels                    # Metal kernels, no weights
make test-qwen4-kernels && make test-mxfp4-metal   # the sibling kernel tests that share metal/qwen4.metal and the Metal link lines
./tests/test_mellum_stream                  # built by test-mellum; the streaming policy unit tests
```

Model-gated (after the merge, A6, A7, B1, B2, and at the end):

```sh
export DS4_TEST_MELLUM_MODEL=$HOME/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf
make test-mellum                            # config/tokenizer now also validate, bind, render and tokenize against the GGUF
./ds4_test --mellum-session --mellum-exact-prefill --mellum-attn-tiled --mellum-rewind --mellum-moe --mellum-server-ids
./ds4_test --mellum-stream-identity --mellum-stream-prefill --mellum-stream-faults --mellum-stream-close \
           --mellum-stream-group --mellum-stream-group-events --mellum-stream-group-memory --mellum-stream-slabs \
           --mellum-stream-stop-scratch --mellum-stream-read-fault
make test-mellum-trace                      # the trace driver's model-gated test
make test-mellum-cli                        # ./tests/test_mellum_cli.sh on the real CLI
make test-mellum-fastpath                   # about 60 model runs; run once at the end
tests/test_mellum_runners.sh                # the README's two runners, as checks of the commands
```

Omit `--mellum-b0` once A7 lands. The oracle comparison (`tests/mellum_oracle_compare.sh`) needs the old engine's logit
files and is not part of the gate; `--mellum-exact-prefill` and `test-mellum-trace` carry the byte-equality check.

Doc checks after A1-A3, A5, A8, A9:

```sh
git grep -n -I -E '/Users/(pauleveritt|koudai)|herveline|\.local\b' -- . ':!tests/long_context_security_prompt.txt'   # empty
git grep -n 'superpowers/research/[^ )]*/' -- README.md MELLUM.md ROADMAP.md docs/superpowers/research/*.md | while IFS=: read f l rest; do :; done   # every linked path exists or names the tag
grep -rn mellum_measurements tests tools Makefile README.md   # empty
make -n test-mellum >/dev/null             # every prerequisite still exists
```

## Order of execution (one concern per commit)

1. Tag and branch the pre-cleanup state. 2. Merge `upstream/main`; full gate. 3. A1 (per research subdir), A2, A3,
A4 (deletions; link edits as their own commits). 4. A5 by phase slice, `make test-mellum` after each. 5. A6 in two
commits (hooks, then modules/tests); fast gate + model-gated gate. 6. A7 (f32 tiles; B0) with red/green where a test
changes; model-gated gate. 7. A8, A9, A10. 8. B1 in five move commits; B2 in four; gate after each. 9. Recompute
`README.md:290` with its command; `git diff --shortstat` against the new base for the final MELLUM.md note.

## What I am unsure of (guesses, verify before acting)

- Whether `ds4_gpu_mellum_attn_tiled_tensor` and `_half` share one dispatch helper (affects A7's line count only).
- Whether `tools/mellum_s4_measure.py` imports `tools/mellum_gguf.py` (keep `mellum_gguf.py` if so; the grep in A5 decides).
- Whether every `docs/superpowers/plans` file is for a closed phase: S5 is "in progress" and S5b/S3b are "waiting on
  hardware", so `2026-10-28-mellum-s5-layer-major-prefill.md` may still be live; keep that one if the owner is mid-phase.
- The exact hunk ranges in `ds4.c`/`ds4_agent.c` are from `git diff -U0` headers and `grep -n` on HEAD `6f8d91b`; they
  shift after the upstream merge. Re-derive with `grep -n` before editing.
