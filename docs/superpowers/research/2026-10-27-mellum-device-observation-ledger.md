# SDD ledger — plan: docs/superpowers/plans/2026-10-27-mellum-device-observation.md
Execution baseline: 72da50b. Existing linked worktree retained. Inline implementation.
Pre-flight 1→5: native optional callbacks and R/G/D/L pipes consumed by owned-child runner; matched.
Pre-flight 2→3/5: snapshots carry command start/end, raw bytes and driver/media counters; matched.
Pre-flight 4→6: metadata-only prepare, exactly two full verify_packed passes, one new reservation; matched.
Pre-flight 5→6: partial child evidence/failure propagates without retry; matched.
Todo: Tasks 1–7 pending. No real diagnostic until full gate and final review.
Baseline full model-free suite passed (baseline.log); initial overlapping baseline/test edit was rerun with original baseline bytes restored, no baseline defect.
Task 1 hooks: RED missing observed API at 804b9ac; GREEN lifecycle/content/join suite passed. Remaining gate protocol/CLI.
Task 1: complete (commits 72da50b..460beea, tests: sh -c 'make -f Makefile test-mellum-read-run test-mellum-read-gate test-mellum-read-bench && python3 -I tests/test_mellum_read_observed.py' → OK)
Task 2 mapping uses /dev block-node st_rdev matched to file st_dev, then diskutil DeviceNode verification. diskutil does not accept an arbitrary model-directory path on this host. No payload read occurred.
Task 2 tests consolidate same-behavior cases; native/Python clock containment is additionally checked in Task 5 actual-native integration.
Task 2: complete (commits 460beea..e07ac50, tests: python3 -I tests/test_mellum_device_observer.py → OK)
Task 3: complete (commits e07ac50..c610838, tests: python3 -I tests/test_mellum_device_classify.py → OK)
Task 4: Ruling: enforce optional read ceilings in shared storage reader and route source hashes through it while budgeted — forecasts alone cannot stop changed inputs from consuming excess reads — cost: shared helpers changed, existing expert index/pack tests revalidated; ordinary unbudgeted behavior retained.
Task 4 forecast: exactly 2*(M+3P) payload base, 128 MiB total directory/prefix upper allowance, 8 GiB metadata/source/report upper allowance. Actual reads hard-capped before read requests. Preparation reads only metadata/plans/current source, no model or payload.
Task 4: complete (commits c610838..bed8efa, tests: python3 -I tests/test_mellum_observe_manifest.py → OK)
Task 5: complete (commits bed8efa..08fd545, tests: sh -c 'python3 -I tests/test_mellum_observe_process.py && python3 -I tests/test_mellum_read_observed.py' → OK)
Task 6: complete (commits 08fd545..15c7952, tests: make -f Makefile test-mellum → OK)
Final review: 6 Important, no Critical; one deferred Minor. RED regressions in faa206a.
Final: fixed reservation-before-payload, closing metadata verification, actual Makefile compute targets and previous-cell raw lifetime — batch regressions RED→GREEN (11 tests).
Final: fixed XML/schema/discovery evidence loss and metadata interruption cleanup — parser/cancellation/discovery and actual-child regressions RED→GREEN (8 observer tests, 4 process tests).
Final: fixed peak-memory accounting — retained-container sizes include bytearray capacity, 128 MiB transient reserve, XML decoding capped at 1 MiB/16384 nodes/depth64, previous cell raw references freed before next capture; RED→GREEN capture and batch regressions.
Final: minor (deferred): stderr emitted while a failing snapshot runs can remain undrained during native child cleanup; failure refusal and captured evidence remain, late native diagnostics may be absent.
Final: Ruling: reject non-XML or >1 MiB/16384-node/depth64 metadata before decoding — fixed Apple commands emit XML and bounded structure makes capture budget defensible — cost: larger/alternate metadata fails safely instead of running.
Final: Ruling: actual host/counter behavior deferred to the single approved real diagnostic — read-only reviewer could not measure it — cost: model-free tests alone do not establish compatibility or physical traffic.
Final: Ruling: full-suite status remains executor-owned and freshly rerun after fixes — reviewer ran focused probes only — cost: review does not independently attest the entire suite.
Final: Ruling: memory cap is conservative captured-data/decoder accounting, not a measured whole-process peak RSS claim — no large-allocation stress run or physical 8 GB proof — cost: runtime/allocator/OS overhead remains unmeasured.
Final: Ruling: unchanged S3a code relies on its prior reviews, with changed shared helpers/consumers covered now — keeps review scoped to this milestone — cost: unchanged defects outside this range may remain.
Final: Ruling: no power-loss durability or signed compiler attestation added — retain existing publication/provenance guarantees — cost: crash recovery or compiler substitution is outside this evidence.
Final reviewer fix pass: all six Important findings have passing regressions; final-gate.log full serial make test-mellum passed. No second independent review.
Real attempt at 17ee04b: observe-prepare succeeded; observe-run refused during discovery on diskutil WholeDisk vs expected Whole. Zero started cells, zero native timed/prewarm reads, zero model/payload verification reads. Metadata Python counts: prepare 18094888, run 19494975, cumulative37589863. No reservation was created; no retry performed.
Task 7: Ruling: repair the demonstrated WholeDisk compatibility defect using saved metadata/model-free tests only, while ending this real attempt — the approved design requires stopping on refusal and a fresh explicit execution decision for resumption — cost: device deltas remain unmeasured and the old source-bound manifest is unusable with repaired code.
WholeDisk repair: e996153 RED, subsequent GREEN; fixture now mirrors real diskutil WholeDisk and IOKit Whole independently. Final full gate follows.
Task 7: complete (commits 15c7952..72f3bc9, tests: python3 -I /Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/mellum-device-observation-execution/verify_attempt.py → Saved refusal identities, planned budgets, zero started cells and CLI accounting verified; no device experiment deltas.)
