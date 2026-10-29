# Mellum workload source freeze implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the approved ac-demo commit as an immutable evaluation-source artifact without changing that repository.

**Architecture:** A small standard-library publication helper reserves a fresh output container and atomically promotes a complete private directory inside it. The standalone source-bundle command reads committed Git blobs, checks the source checkout before/after, and retains exact content identities and the proposed Phase 1 task. This source bundle is not a tokenized workload or completed collection manifest.

**Tech Stack:** Python 3.14+ standard library; existing Git and Makefile.

**Spec:** `docs/superpowers/specs/2026-10-27-mellum-storage-evidence-design.md` (approved by the user's subsequent “Proceed”; Workload source and publication requirements).

## Global Constraints

- Python 3.14+ standard library; isolated `python3 -I` execution, no dependency installation.
- No GPU, inference, server, generated capture, streaming, port integration, push or merge.
- Keep `codex/mellum-ssd-replay`; do not modify the ac-demo checkout or its Python constraint.
- Source: `/Users/pauleveritt/projects/pauleveritt/ac-demo`, exact commit `bfb6e1f6e2ef141a1ac88f5c25466d3a076caab3`.
- Never inspect `.venv`, copy untracked files/credentials, or claim an ac-demo phase was implemented.
- Reserve all ac-demo conversations/phases for evaluation; existing Mellum fixtures remain tuning.
- Preserve the roadmap/AGENTS instructions, including one phase per request, single quotes, Request import and `uv run --frozen python -m pytest tests`.
- Small focused commits; commit failing behavioral tests separately before fixes; roughly 150 new product lines per implementation commit.
- Execute this plan before the storage plan, which consumes its artifact helper. One independent final review covers this plan; no per-task reviewer.

## File structure and interfaces

- `tools/mellum_artifact.py`: generic publication only; no repository/model logic.
  `publish(container: Path, build: Callable[[Path], None]) -> Path` returns `container/artifact`.
  Parent must already exist. Reserve `container` with exclusive `mkdir`; stage at a private child; after the callback closes all files, rename that child to `artifact`.
  Never reuse an existing container (including an empty directory or symlink). On failure remove only children created by this invocation, then its empty reservation. An interrupted reservation is incomplete and cannot be mistaken for `artifact`.
  Atomicity refers to the complete `artifact` child; the reservation container may exist earlier. This avoids a non-atomic exists/rename replacement race without adding platform-specific rename APIs.
- `tools/mellum_workload_bundle.py`: `bundle(repo: Path, revision: str, container: Path) -> Path`; CLI `--repo --revision --output`.
  Bootstrap existing `mellum_capture_identity` as current isolated tools do; reuse its sibling loading/canonical source identity where applicable, without broadening the capture publisher contract.
- `tests/test_mellum_artifact.py`, `tests/test_mellum_workload_bundle.py`: standalone `unittest` entry points, temporary repositories only.
- `docs/MELLUM_STORAGE_EVIDENCE.md`: artifact contract/source-bundle usage, extended by the storage plan.
- Makefile: `test-mellum-workload-source`, included in `test-mellum`; no real source inspection in that gate.

## Review Focus

1. Competing publishers or an existing empty/symlink output must not be overwritten (Task 1).
2. Failure after writing data or during checked close must expose no complete artifact (Task 1).
3. Inherited Git redirects, replacement objects and source movement must not substitute another repository/tree (Task 2).
4. Tracked symlinks/submodules/path traversal and oversized blobs must be rejected, with ignored/untracked secrets excluded (Task 2).
5. The source bundle must not claim that generated/tokenized Phase 1 evaluation has occurred (Tasks 2–3).

## Task 1: Atomic complete-artifact publication

**Files:** create `tools/mellum_artifact.py`, `tests/test_mellum_artifact.py`.
**Consumes:** only stdlib pathlib/tempfile/os/shutil and caller-owned builder callback.
**Produces:** `publish(container: Path, build: Callable[[Path], None]) -> Path` above. Caller must check flush/close inside the callback; publication itself must propagate rename and cleanup errors without deleting pre-existing content.

- [ ] Add tests `test_complete_directory_visible_only_after_builder_returns`, `test_existing_outputs_untouched`, `test_competing_reservation_has_one_winner`, `test_builder_close_and_rename_failures_never_publish`. Assertions: before callback completion `(container/'artifact').exists()` is false; successful files match exact bytes; a losing writer preserves a sentinel; injected callback/close/rename failure leaves no artifact; no unrelated sibling is removed.
  Core success assertions (builder writes `stage/'x'` as `b'complete'`):
  ```python
  artifact = publish(container, build)
  assert artifact == container / 'artifact'
  assert (artifact / 'x').read_bytes() == b'complete'
  ```
- [ ] Run `python3 -I tests/test_mellum_artifact.py`; expect assertion/import RED because helper is absent. Commit `test: specify complete artifact publication boundaries`.
- [ ] Implement `publish` with an exclusive reservation and private same-filesystem stage, no overwrite fallback. Keep callback exceptions intact; document interruption semantics. Run the same command, expect all tests PASS. Commit `feat: publish complete artifacts inside exclusive containers`.

## Task 2: Checked Git source bundle

**Files:** create `tools/mellum_workload_bundle.py`, `tests/test_mellum_workload_bundle.py`; extend usage document.
**Consumes:** Task 1 `publish`, existing identity/sibling helpers.
**Produces:** `bundle(repo: Path, revision: str, container: Path) -> Path` and CLI. Output `artifact/source/<tracked paths>` plus `artifact/manifest.json`.

Manifest `schema_version=1`, `kind='workload-source'`, `repository_id='ac-demo'`, `split='evaluation'`, source absolute path/commit/tree, tool/source identity, sorted `files` map `{path: {bytes, sha256, git_blob}}`, and `content_id` (SHA-256 of canonical JSON containing the revision/tree/file hashes, excluding machine-specific paths). Include `proposed_task={phase:1, request:'Implement Phase 1 only, including its home-page tests, without starting a server.'}` and `collection_complete=false`.

Require exact HEAD revision and no staged/unstaged tracked changes before and after. Enumerate `git ls-tree -rz -l` at the immutable commit and read only those blobs. Sanitize inherited `GIT_*`, set `GIT_NO_REPLACE_OBJECTS=1`, use explicit `git -C` argv and full object IDs. Allow untracked/ignored content to exist but never enumerate/copy it. Reject modes other than regular files, any tracked `.venv` path before reading its blob, unsafe/absolute/non-UTF-8 paths, a destination within the source checkout, more than 256 files, any blob over 16 MiB or aggregate over 64 MiB before capturing blob output. Copy the actual modes for regular executable files. Git is a consistency source, not hostile-repository attestation.

- [ ] Add temporary-repo tests `test_bundle_matches_git_bytes_and_content_id`, `test_wrong_revision_and_tracked_changes_refused`, `test_git_redirects_and_replacements_cannot_substitute_source`, `test_unsafe_modes_paths_and_blob_limits`, `test_untracked_secrets_excluded`, `test_mid_bundle_change_and_output_collision_fail`. Assertions include exact original bytes, stable content ID across output locations, no `.venv`/secret in manifest/output, no source changes, and `collection_complete is False`.
  Pin these manifest assertions against the temporary repo's known blob:
  ```python
  assert manifest['collection_complete'] is False
  assert manifest['split'] == 'evaluation'
  assert manifest['proposed_task']['phase'] == 1
  assert manifest['files']['AGENTS.md']['sha256'] == hashlib.sha256(original).hexdigest()
  ```
- [ ] Run `python3 -I tests/test_mellum_workload_bundle.py`; expect missing-module RED. Commit `test: pin immutable workload source bundle identities`.
- [ ] Implement validation, bounded blob copying and manifest/publication in focused GREEN commits (split source validation and bundle assembly if needed). Same command must PASS before committing each completed behavior; final message `feat: freeze repository inputs for held-out workload collection`.
- [ ] Add the model-free target and usage invocation. Run `make -f Makefile test-mellum-workload-source`; expect both new suites PASS with no real checkout/model access.

## Task 3: Real source freeze, review and retained handoff

**Files:** Makefile, `docs/MELLUM_STORAGE_EVIDENCE.md`, ROADMAP.md and a dated retained evidence record.
**Consumes:** Tasks 1–2 public APIs. Real output goes under this task's visualization/artifact directory, not ac-demo and not a tracked model-data directory.
**Produces:** a checked ac-demo source artifact and evidence path/content ID for later S3b.

- [ ] Run the real CLI against the exact approved revision, a fresh output container and the existing clean checkout. Expect nine tracked files, correct per-file hashes, `split='evaluation'`, Phase 1 request and `collection_complete=false`. Compare every published file with its Git blob. If the checkout has advanced/changed, stop this real freeze with a clear record; do not substitute its current revision or alter it.
- [ ] Run `make -f Makefile test-mellum` serially with no model/GPU process; expect all model-free suites PASS. `git diff --check` must pass. Do not run ac-demo tests: no phase has been implemented.
- [ ] One fresh final review of this plan's range/spec/ledger using executing-plans and its Review Focus. Fix Critical/Important findings once with separate RED/GREEN commits; record minors without implementing them, and preserve rulings/evidence.
- [ ] Commit the evidence/usage/roadmap update separately. Retain branch/worktree and source bundle for the storage/S3b plans; remove only this plan's completed scratch after retaining its ledger.

## Self-review

Source preservation and holdout semantics map to Task 2 and real verification in Task 3; publication/failure safety maps to Task 1. All five Review Focus entries have named tests. The only cross-plan interface is `publish`; the storage plan consumes the returned `container/artifact`, never a partially reserved container. No generated workload, GPU result or roadmap phase completion is implied. This plan is ready for user plan review; implementation has not started.
