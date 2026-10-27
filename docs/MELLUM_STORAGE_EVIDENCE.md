# Mellum storage evidence tools

These Python 3.14+ standard-library tools prepare CPU-only evidence. They do
not initialize the model or GPU, generate conversations, or implement streaming.

## Freeze the evaluation source

Run from this worktree with a fresh output container whose parent exists:

```sh
python3 -I tools/mellum_workload_bundle.py \
  --repo /Users/pauleveritt/projects/pauleveritt/ac-demo \
  --revision bfb6e1f6e2ef141a1ac88f5c25466d3a076caab3 \
  --output /absolute/existing-parent/ac-demo-source
```

The result is `ac-demo-source/artifact`, containing `manifest.json` and
`source/` with exact committed regular-file bytes and executable modes. The
manifest records the commit, tree, per-file hashes, tool/source identity and
machine-independent content ID. The checkout must remain at the exact clean
tracked revision before and after copying. Untracked and ignored files are
excluded; `.venv` is never enumerated or read.

The entire repository is reserved for evaluation. The proposed first task is
Phase 1 only, including home-page tests, without starting a server. The bundle
preserves AGENTS.md, roadmap and stack instructions. `collection_complete=false`
means this is source material, not rendered/tokenized prompts, generated
conversations or a completed phase. Those need the later S3b design. No ac-demo
tests run during freezing because no phase is implemented.

Publication reserves a new container exclusively. Only its complete `artifact`
child is valid; the container can exist earlier. Existing files, directories
and symlinks are refused. A failed build exposes no complete artifact. An
interrupted reservation must not be reused. Builders check writes/flush/close
before promotion; this is complete-directory visibility, not power-loss durability.

Run synthetic tests without inspecting the real repository or model:

```sh
make -f Makefile test-mellum-workload-source
```
