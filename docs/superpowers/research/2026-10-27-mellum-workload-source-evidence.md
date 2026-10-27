# S3a evaluation source freeze

The approved source-freeze plan ran inline on `codex/mellum-ssd-replay`, starting
at `1acc18e`. This is CPU-only preparation; no GPU, inference, ac-demo phase,
server or dependency installation ran.

## Frozen source

Command (Python 3.14.7):

```sh
python3 -I tools/mellum_workload_bundle.py \
  --repo /Users/pauleveritt/projects/pauleveritt/ac-demo \
  --revision bfb6e1f6e2ef141a1ac88f5c25466d3a076caab3 \
  --output /Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/ac-demo-source-s3a
```

Published artifact: the command's output container plus `/artifact`.
Content ID: `e14b1b120a8563fd1dcb316ea6e6d815b593d2ba45605fb34b55c3b238834861`.
Manifest SHA-256: `a4c86abcb347290baded655250c662b23a4836eda2157a8c1e7d234468f067dc`.
Producer implementation: `6e51b74`; gate/usage commit: `f05bbab`.

All nine tracked files were compared byte-for-byte with their recorded Git
blobs, independently of the copying loop; individual SHA-256 values also match.
The exact clean HEAD was checked before and after copying. The original checkout
was left unchanged. No ignored/untracked contents were copied or enumerated.

The manifest reserves the full repository for evaluation, proposes Phase 1
only (including its tests, without a server), and says
`collection_complete=false`. AGENTS.md and roadmap bytes are preserved. No
ac-demo tests ran, because no phase was implemented. S3b still needs the exact
rendered/tokenized prompt and generated operation schedule; this source freeze
does not supply those or complete S3.

## Verification

Publication tests: 4/4 passed after an observed missing-implementation RED.
Source-bundle tests: 6/6 passed after an observed missing-implementation RED.
The standalone gate and full serial `make -f Makefile test-mellum` passed;
`git diff --check` passed. Tests use temporary Git repositories, never the real
model or ac-demo checkout. Tests cover exclusive reservation, failure cleanup,
source movement, redirect/replacement isolation, path/mode/blob budgets and
untracked-secret exclusion.

Independent final review and retained execution ledger are recorded below
before this plan is closed.

## Final review

Fresh read-only gpt-6-astra review of `1acc18e..f05bbab` found one Important
issue: Git status can hide tracked changes under assume-unchanged/skip-worktree.
The regression failed before the fix (`7cda48d`); `26d027d` rejects both flags,
and `a60dfc1` covers introducing them during copying. Source tests now pass 7/7.
No minors were identified. The full CPU gate was rerun after the fix.

Reviewer exclusions were assessed and retained: deliberate reserved-directory
tampering and corrupted Git object storage are outside the consistency trust
model (cost: hostile interference needs stronger isolation/attestation);
power-loss durability is not promised (cost: interrupted preparation may need
a fresh output); full CPU/evidence checks remain executor-owned and passed,
while generated/tokenized/GPU work remains S3b (cost: this source freeze alone
establishes no inference performance). There was no second review.

The original artifact remains valid: it contains the approved commit's exact
bytes, and the source checkout had neither hidden-index flag. The fix strengthens
future refusal behavior; it does not change the frozen content ID.
