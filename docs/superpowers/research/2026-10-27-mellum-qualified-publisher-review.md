# Mellum qualified publisher: CPU preparation and review

Date: 2026-10-27. Branch: `codex/mellum-ssd-replay`.
Stage baseline: `01603cc`. Reviewed head: `972ef0a`. Final code: `064ac84`.
CPU implementation and independent review are complete. No GPU initialization,
model process, SSD experiment or live qualified artifact occurred in this stage.
S2b remains open pending the approved live gates and a real v2 artifact round trip.

## Result

The publisher force-builds the existing parity harness using the repository
Makefile, brackets collection with source/binary/input identities, validates
native receipts and complete canonical profiles, compares ordered routes and
the requested token/operation schedule, then uses the unchanged atomic exporter.
The CLI supports only the harness's teacher-forced protocol. Caller labels and
model stat checks are explicit limits; source consistency is not attestation.

## Independent review

One fresh read-only gpt-6-astra review found no Critical issues, three Important
issues and one Minor coverage gap. All Important findings were accepted and
fixed in one pass, with RED commit `454f3f7` and GREEN commit `064ac84`:

1. Prefix-only profile validation accepted missing/garbage descriptors. Validation
   now checks complete native grammar, canonical role/function/value ordering,
   known functions/constants from the native X-macro catalog, mandatory roles and
   head consistency. The success fixture now uses a complete native-format profile.
2. Inherited Git repository/index settings could omit tracked ignored sources.
   The publisher now rejects inherited GIT_* settings before build and the identity
   helper also sanitizes its Git subprocess environment. The regression force-tracks
   an ignored C file and reproduces the alternate-index omission scenario.
3. Make could select an unhashed GNUmakefile or lowercase makefile. The build now
   explicitly passes -f Makefile and snapshots both alternative primary names.
   The regression checks the selected command and rejects an alternate-file mutation.

No second reviewer was dispatched. Minor deferred: explicit capture-stage
subprocess failure after complete outputs and checked-close failure injection.
Current propagation paths were inspected, but those matrix cases are not tested.

## Verification

All eleven Python publisher tests pass, including source/environment controls,
partial/conflicting artifacts, token/schedule identity and atomic no-overwrite.
The full model-free Mellum suite and native harness compile/link passed:

```sh
env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum tests/mellum_routes_gpu
```

The native receipt test passed with explicit C99, -Wall -Wextra -Werror.
A CPU-only C fixture used the actual observer/registry and serializer to emit
head and no-head profiles; both passed the Python catalog validator. This checks
format compatibility, not actual Metal observation. The fixture reused the
existing tests/test_mellum_observer.c registry/observed helpers, called observed
with nsg=4 and each head flag, then emitted both profiles with mellum_profiles_emit.
The real worktree source snapshot covered 300 inputs and was stable across reads
before the review fixes. Isolated CLI --help passed without executing inference.

## Next gates

In a separately authorized device window, validate the owned adapter's real
route/logit/profile and lifecycle behavior, allocation/path/fallback matrix and
capture/default overhead, then run the publisher and round-trip the actual v2
artifact through replay. This publisher does not by itself establish qualified
representative workloads, model quality, warm locality, SSD throughput or 8 GB
feasibility. Generated cohorts need a later driver and separate workload labels.

## Retained execution ledger

The ledger below preserves all scope decisions, their costs, review findings
and excluded behaviors before this plan's scratch directory is removed.

```text
# SDD ledger — plan: docs/superpowers/plans/2026-10-27-mellum-qualified-publisher.md
Authorization: explicit Proceed continues the approved session-capture design's CPU publisher stage, using the existing inline method and isolated codex/mellum-ssd-replay branch at 01603cc. No GPU permission.
Ruling: implement the detailed publisher plan within the approved design without repeating design approval — user explicitly continued the named next CPU stage — cost if wrong: implementation choices may need revision.
Pre-flight: Task 1 source/file identities -> Task 3 compares complete identity dictionaries across build/capture, metadata receives source_state_id and binary hash.
Pre-flight: Task 2 receipt -> Task 3 parses receipt_version/context_tokens/cap_tokens/capabilities; actual profiles still arrive via existing finish callback file.
Pre-flight: Task 3 -> exporter consumes unchanged header/segments/execution_profiles metadata; only final trace publication is public, private files close before export.
Task 1: complete (commits baacd10..329b8dd, tests: python3 -I tests/test_mellum_capture_publish.py → OK)
Task 2: complete (commits 329b8dd..e2f7313, tests: make test-mellum-capture-receipt tests/mellum_routes_gpu → make: `tests/mellum_routes_gpu' is up to date.)
Task 3: Ruling: separate bounded metadata parsing/derivation into mellum_capture_metadata.py — keeps subprocess policy and JSON validation independently readable — cost if wrong: one additional internal module to maintain.
Task 3: Ruling: reject inherited make controls and require a freshly rebuilt executable — force-build success alone can be a no-op under inherited flags/compiler shims — cost if wrong: some custom build environments require explicit cleanup before capture.
Task 3: verified token stream and actual phase/segment/chunk schedule against requested input independently of reference parity; RED accepted wrong input/schedule at offsets 60 and 1884, GREEN rejected both. Synthetic subprocesses establish only publisher behavior, no GPU or provenance artifact was collected.
Task 3: complete (commits e2f7313..b7d4176, tests: python3 -I tests/test_mellum_capture_publish.py → OK)
Task 4: source-map conflict regression RED→GREEN (223a8de, 0e2894d); caller-added unobserved source files are rejected. CLI --help passed in isolated Python without GPU.
Task 4: complete (commits b7d4176..972ef0a, tests: env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum tests/mellum_routes_gpu → OK)
Final review: fresh read-only gpt-6-astra review of 01603cc..972ef0a found three Important issues, no Critical, and one Minor coverage gap. Accepted all Important grades; fixed them in one pass, no second reviewer.
Final: fixed malformed profile acceptance — test_incomplete_or_noncanonical_profiles_never_publish RED→GREEN (454f3f7, 064ac84); complete descriptor grammar/order/function/constants/roles/head validation now required. Real C observer/serializer head and no-head profiles passed Python validation.
Final: fixed inherited Git redirection — test_git_redirection_cannot_hide_tracked_source_mutation RED→GREEN; inherited GIT_* fails before build, source Git subprocesses also sanitize GIT_*.
Final: fixed omitted primary makefile identity — test_primary_makefile_selection_is_explicit_and_source_changes_rejected RED→GREEN; force build pins -f Makefile and source snapshot includes GNUmakefile/makefile. Full model-free suite and harness compile/link passed.
Final: Ruling: validate profile text against the existing native X-macro catalog — avoids guessed descriptors and duplicated function/constant definitions — cost if wrong: future catalog formatting changes may require a Python parser update.
Final: Ruling: reject all inherited GIT_* and pin the ordinary Makefile — qualified identity must refer to this worktree/build — cost if wrong: custom Git/build environments need cleaning before capture.
Final: minor (deferred): add explicit capture-stage subprocess failure after complete outputs and checked-close failure injection; current failure matrix tests build failure but lacks those cases, although propagation paths are present.
Final: Ruling: Physical GPU parity and observer accuracy — retain live gates; CPU publisher checks cannot establish either — cost if wrong: live capture may fail parity/provenance.
Final: Ruling: GPU allocation/cancellation/cleanup/fallback/no-head/sequential-handle matrix — retain the broader approved live matrix — cost if wrong: device lifecycle/path defects may remain.
Final: Ruling: Capture/default overhead and memory peaks — require later physical measurements — cost if wrong: timing and memory budgets remain unknown.
Final: Ruling: Generated capture cohorts — keep this publisher restricted to its explicit teacher-forced schedule — cost if wrong: generated workloads need another driver/cohort.
Final: Ruling: Representative quality/locality/SSD feasibility and physical 8 GB suitability — defer conclusions until qualified representative evidence and target measurements — cost if wrong: feasibility remains unknown.
Final: Ruling: Model-byte authenticity and caller model/tokenizer/template/device labels — retain explicit declared limitations and model-file stat checks — cost if wrong: wrong labels or same-stat content substitution are not attested.
Final: Ruling: Hostile compilers, restored-between-check changes and reproducible-build attestation — retain the nonadversarial source-consistency contract — cost if wrong: those substitutions/reproducibility issues may go undetected.
Final: Ruling: External toolchain/system-header provenance — keep the approved repository-source snapshot scope; binary hash is retained — cost if wrong: external build inputs are not fingerprinted.
Final: Ruling: Unchanged standalone-link/checkpoint methods and unrelated backends — retain baseline implementation and model-free regression gates without another baseline review — cost if wrong: unexercised backend behavior remains unproved.
Final: Ruling: Exhaustive standalone exporter correctness — reuse the baseline exporter and its tests, plus publisher round trips — cost if wrong: no new exhaustive proof of exporter correctness is claimed.
Task 4: complete (commits b7d4176..064ac84, tests: env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum tests/mellum_routes_gpu → OK)
```
