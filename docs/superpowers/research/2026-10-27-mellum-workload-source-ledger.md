# SDD ledger — plan: docs/superpowers/plans/2026-10-27-mellum-workload-source.md

Baseline: 1acc18e; codex/mellum-ssd-replay; existing isolated worktree clean; full model-free make test-mellum passed.
Tasks: 1 publication; 2 checked source bundle; 3 real freeze, review, retained handoff.
Pre-flight: Task 1 -> Task 2: publish(container, build) returns container/artifact; matching producer/consumer.
Pre-flight: Tasks 1–2 -> Task 3: CLI output is workload-source, never generated collection; interfaces agree.
Pre-flight: source -> storage plan: storage consumes same generic publication helper without changing capture publication.
Task 1: complete (commits 1acc18e..ab09321, tests: python3 -I tests/test_mellum_artifact.py → OK)
Task 2: complete (commits ab09321..f05bbab, tests: make -f Makefile test-mellum-workload-source → OK)

Task 3: real freeze verified nine exact Git blobs; content e14b1b120a8563fd1dcb316ea6e6d815b593d2ba45605fb34b55c3b238834861; full serial gate and diff check passed. Final fresh review dispatched.
Final: fixed hidden tracked changes — test_hidden_tracked_changes_cannot_pass_cleanliness_check RED→GREEN; covers flags before and during copy; full model-free suite rerun.
Final review: fresh gpt-6-astra, 1acc18e..f05bbab; one Important fixed in 7cda48d/26d027d; additional mid-copy coverage a60dfc1. No minors.
Final: Ruling: hostile reserved-directory replacement is outside consistency trust model — cooperating publishers are protected by exclusive mkdir; deliberate tampering requires stronger filesystem isolation — cost if wrong: adversarial interference can invalidate publication.
Final: Ruling: corrupted Git object storage is outside consistency trust model — Git supplies committed blobs and SHA identities; no hostile-repository attestation promised — cost if wrong: compromised Git storage could supply untrustworthy bytes.
Final: Ruling: power-loss durability not promised — complete-child visibility only, as documented — cost if wrong: crash/power loss may require fresh preparation.
Final: Ruling: CPU gate/evidence and generated/GPU work set aside by reviewer remain executor-owned — full CPU gate and nine-blob verification performed; generated collection/GPU explicitly deferred to S3b — cost if wrong: source freeze alone cannot establish workload/inference performance.
Task 3: complete (commits f05bbab..1de9566, tests: make -f Makefile test-mellum → OK)
