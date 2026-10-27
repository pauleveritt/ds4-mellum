# SDD ledger — plan: docs/superpowers/plans/2026-10-27-mellum-storage-evidence.md

Tasks: 1 bounded reader; 2 directory/index; 3 pack; 4 jobs/wire; 5 native parser; 6 I/O; 7 workers/report; 8 frozen batches; 9 gate/review/real evidence.
Pre-flight: source plan -> all publications: publish returns container/artifact, not container; matches.
Pre-flight: 1 -> 2: bounded Reader(source,size) methods match directory consumer.
Pre-flight: 2 -> 3/4/8: validated ordered expert ranges plus verified source binding; schema alone is insufficient; matches.
Pre-flight: 3 -> 4/8: contiguous offsets and full source/payload verification remain distinct; matches.
Pre-flight: 4 -> 5: 64-byte header and 32-byte records, jobs 1..4096, ranges 1..3, max 6580224; exact match.
Pre-flight: 5/6 -> 7: native plan/job/count types and pread boundary match; caller owns read fd/buffers.
Pre-flight: 4/7 -> 8: frozen IDs, timestamps and counts support strict validation and fixed 72-run budget.
Pre-flight: 8 -> 9: optional real commands are explicit, never model-free gate; matches.
Task 1: complete (commits f05bbab..c0a8fb8, tests: python3 -I tests/test_mellum_gguf_reader.py → OK)
Task 2: Ruling: add mellum_storage_common.py for strict bounded JSON and opened-file identity/hash operations — all storage modules require identical checks; one shared implementation avoids drift — cost if wrong: extra module boundary to maintain.
Task 2: named free-function groups run via runpy, not unittest name arguments (FunctionTestCase load_tests owns discovery).
Task 2: complete (commits c0a8fb8..1de9566, tests: python3 -I tests/test_mellum_expert_index.py → OK)
Task 3: complete (commits 1de9566..ecfcc46, tests: python3 -I tests/test_mellum_expert_pack.py → OK)
Task 4: complete (commits ecfcc46..ae88055, tests: python3 -I tests/test_mellum_read_plan.py → OK)
Task 5: complete (commits ae88055..7c759a5, tests: ./tests/test_mellum_read_plan → native range plan checks: PASS)
Task 6: complete (commits 7c759a5..388bfe2, tests: make -f Makefile test-mellum-read-io test-mellum-read-plan → native range plan checks: PASS)
Task 7: complete (commits 388bfe2..8f41d13, tests: make -f Makefile test-mellum-read-bench test-mellum-read-run test-mellum-read-io test-mellum-read-plan → native range plan checks: PASS)
Task 8: complete (commits 8f41d13..0b8c384, tests: python3 -I tests/test_mellum_read_batch.py → OK)
Task 9: full serial make test-mellum passed; strict C99 build recipe links only four storage sources with pthreads; diff check passed. Fresh gpt-6-astra final review dispatched for f05bbab..a20dd09. No real-file preparation or timed batch yet.
Final review: fresh gpt-6-astra of f05bbab..a20dd09; no Critical, two Important, one Minor. No second review.
Final: fixed detached/replaced packed payload — test_replaced_payload_path_cannot_publish_detached_descriptor RED→GREEN, including content mutation during verification; fixes 5e15e6c/9d0a185.
Final: fixed transient consumed wire substitution — test_transient_native_plan_substitution_never_publishes RED→GREEN; native parser now checks/emits fd/path identity and Python compares frozen identity; fix 5d6e156.
Final: minor (deferred): worker failure stderr discards detailed pread error/job ID and retains a generic complete-run refusal; failure classification remains correct.
Final: Ruling: unrelated source-freeze changes excluded from storage review — their independent review/fix is retained in source evidence — cost if wrong: cross-plan defects could remain beyond this review's focus.
Final: Ruling: real layout/space/full-size runs remain Task 9 executor evidence — run explicitly after green fix gate; a specific refusal is retained without claiming S3 completion — cost if wrong: synthetic tests cannot establish full-file compatibility/performance.
Final: Ruling: physical SSD traffic/cold-cache/inference/8 GB claims remain excluded — flags and logical storage timing do not establish those claims — cost if wrong: results cannot answer physical-target feasibility yet.
Final: Ruling: compiler provenance is declared recipe plus independent binary/source hashes — actual executor build log is retained, not signed attestation — cost if wrong: external binary replacement cannot be cryptographically tied to compiler invocation.
Final: Ruling: no new power-loss durability guarantee — existing complete-child/no-overwrite/checked-close contract stands — cost if wrong: crash may require fresh preparation.
Final: Ruling: privileged metadata falsification/kernel corruption excluded — normal fd/path/stat identities plus full content checks are supported — cost if wrong: a privileged adversary can evade integrity checks.
Final: Ruling: full gate and omitted subprocess test are executor-owned — execute complete serial gate after fix pass — cost if wrong: review alone does not prove integration checks.
Final fix gate: full serial make test-mellum passed after 9d0a185; 36 new Python storage tests plus three native suites and existing Mellum CPU gate green. No unresolved Important findings.
Task 9: real index/pack/prepare/run succeeded once at code/source revision 89369ea. Model SHA ea1e4ec737bb259ac2ffe1f20d412f458fd52dd230f42dce9e944f111093c29b; 1792 experts, packed bytes 11791761408 verified exact. Approved trace SHA matched. 72 frozen runs completed 242573377536 timed logical bytes; prewarming 121286688768 bytes; no repeated batch. All raw stdout hashes/samples/controls independently checked after publication.
Task 9: real report SHA 45133f3a2e3ac9448c71fd05e3b880a52474a5ed802d64e1b4bdfe5a3c2773dd. Packed warm throughput 13.6–30.5% higher; both labels 29–78 GB/s, physical_io_verified=false. No inference miss penalty/tok/s claim. Device metadata supplemented read-only after run, without changing report.
Task 9: evidence/roadmap committed 2a986fc. Preserve branch/worktree and all artifacts; S3 remains open for device-observed SSD traffic and generated held-out S3b collection.
Task 9: complete (commits 0b8c384..2a986fc, tests: make -f Makefile test-mellum → OK)
