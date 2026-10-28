# S6-A: MXFP4 Mellum, the CPU-only work before the merge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything S6 can do before the SSD work merges, without the GPU: a replay at MXFP4 expert size, an MXFP4 CPU reference, a loader that accepts an all-MXFP4 expert file, a slot count taken from the file, and the preregistration for phase B.

**Architecture:** Host-pure and CPU-tested only. The Metal gates (`mellum_graph_moe`, `mellum_moe`, the four `_addr` entry points) stay Q8_0-only until phase B, where Metal tests can show them red then green. Until then an MXFP4 file loads, then is refused at its first MoE dispatch.

**Tech Stack:** C99 (`ds4.c`, `ds4_mellum_stream.c`, tests built with `-DDS4_NO_GPU`), Python 3 stdlib (`tools/mellum_ssd_replay.py`, unittest-style tests run with `python3 -I`), `make`.

**Spec:** [`docs/superpowers/specs/2026-10-28-mellum-s6-mxfp4-8gb-design.md`](../specs/2026-10-28-mellum-s6-mxfp4-8gb-design.md), its "Order", section A.

**Scope.** This plan is phase A only. Phases B (M5 Max, after the merge) and C (the M1 iMac) get their own plans. Per `ROADMAP.md`'s working rules, later implementation is planned only when the preceding steps settle its choices. Deviation from the design's A.1, as a designer's call: the `mellum_read_*` C tools need no change for MXFP4, because their `6580224` checks are upper bounds and 3,290,112 B fits under them. `tools/mellum_expert_index.py` and `mellum_expert_pack.py` are Q8_0-pinned, and the read bench that needs them runs on the iMac, so they move to plan C.

## Global Constraints

- **No GPU, no model run (owner).** Never run `make test-mellum-kernels`, `tests/ds4_test --mellum-moe`, `./ds4`, `ds4-bench`, llama.cpp, or anything that opens Metal or a GGUF. Compiling, including `.metal` sources, is allowed.
- Avoid large SSD reads or writes while the SSD session may be measuring. The one trace this plan reads is small.
- Work in `/Users/pauleveritt/projects/pauleveritt/ds4-mellum/.worktrees/mellum-8gb-q4`, branch `research/mellum-8gb-q4`. Use absolute paths, or `git -C` with that path.
- One concern per commit. The failing test is committed **before** its fix, as a separate commit. About 150 product lines per commit at most. Messages name the behavior, e.g. `test: ...` and `feat: ...`, and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Q8_0 behavior must not change. Every existing CPU test passes unchanged, except the Q8_0-constant renames in Task 4.
- Exact values: Q8_0 expert 6,580,224 B. MXFP4 expert 3,290,112 B (each gate, up or down slice 1,096,704 B). MXFP4 block = 1 B E8M0 scale + 16 B nibbles for 32 values, GGUF type 39. E8M0 `e` decodes to 2^(e−127), and `e == 0` to 2^−127, as `ds4_e8m0_to_f32` (`ds4.c`, just above `ds4_mxfp4_values`). E2M1 values: {0, .5, 1, 1.5, 2, 3, 4, 6, −0, −.5, −1, −1.5, −2, −3, −4, −6}. The low nibble of `qs[j]` is element j; the high nibble is element j+16.
- `tests/test_mellum_config` takes `/tmp/ds4.lock`. If another session holds it, run with `DS4_LOCK_FILE=<private path>`, as the 10-27 and -02 evidence did. Never touch the holder.
- Research notes open with an honesty header and end with "What would falsify this" and "Risks".

## Review Focus

1. **No flag, no change.** A replay without `--expert-bytes` must produce exactly today's report: same keys, same values. Pinned in Task 1.
2. **Mixed experts inside one layer** (MXFP4 gate/up, Q8_0 down): refused at open, naming the tensor. Pinned in Task 3.
3. **A too-small MXFP4 streaming cache:** the refusal quotes MXFP4's 64-slot minimum, 210,567,168 B, not Q8_0's. Pinned in Task 4.
4. **An unknown expert type** in the stream configuration: refused with a message, never a divide by zero. Pinned in Task 4.
5. **E8M0 edge scales** (`e = 0`, `e = 254`): the reference decodes them as the engine's CPU dot does. Pinned in Task 2.

---

### Task 1: Replay at a given expert size, and the MXFP4 miss curve

**Files:**
- Modify: `tools/mellum_ssd_replay.py` (`replay`, `run_sweep`, `main`)
- Test: `tests/test_mellum_ssd_replay.py`
- Create: `docs/superpowers/research/2026-10-28-mellum-mxfp4-miss-curve.md` and `docs/superpowers/research/2026-10-28-mellum-mxfp4-miss-curve/{curve.py,curve-output.txt}`

**Interfaces:**
- Produces: `replay(trace, capacity_bytes, policy, *, expert_bytes: int | None = None) -> dict`; `run_sweep(trace, capacities=DEFAULT_CAPACITIES, *, trace_sha256=None, expert_bytes: int | None = None) -> dict`; CLI `--expert-bytes N`.

- [ ] **Step 1: Write the failing tests** in `tests/test_mellum_ssd_replay.py`, with the file's `header`/`forward`/`load` helpers:

```python
def test_expert_bytes_override_counts_mxfp4_slots():
    trace = load([header(layers=1, expert_bytes=6_580_224), *forward(0, [[[0, 1]]], phase="prefill")])
    report = tool().replay(trace, 3 * 2**30, "lru", expert_bytes=3_290_112)
    assert report["cache"]["slots"] == 979
    assert report["cache"]["effective_bytes"] == 3_221_019_648
    assert report["expert_bytes"] == {"value": 3_290_112, "source": "override"}

def test_no_override_report_is_unchanged():
    trace = load([header(layers=1, expert_bytes=6_580_224), *forward(0, [[[0, 1]]], phase="prefill")])
    plain = tool().replay(trace, 3 * 2**30, "lru")
    assert plain["cache"]["slots"] == 489 and "expert_bytes" not in plain

def test_expert_bytes_override_must_be_a_positive_int():
    trace = load([header(layers=1), *forward(0, [[[0, 1]]], phase="prefill")])
    for bad in (0, -1, True, 2.5):
        try:
            tool().replay(trace, 100, "lru", expert_bytes=bad)
        except ValueError as exc:
            assert "expert_bytes" in str(exc)
        else:
            raise AssertionError(f"expert_bytes {bad!r} accepted")
```

Also add a CLI test, beside the existing subprocess tests: `--expert-bytes 3290112` reaches every result, `report["assumptions"]["routing"] == "unchanged by quantization (expert_bytes override)"`, and with no flag that key is absent.

- [ ] **Step 2: Run them, expecting failure**

Run: `python3 -I tests/test_mellum_ssd_replay.py`
Expected: the four new tests FAIL (`unexpected keyword argument 'expert_bytes'` / unknown CLI flag); the 33 existing pass.

- [ ] **Step 3: Commit the red tests**

`git commit -m "test: the SSD replay can count slots at another expert size"`, after `git add tests/test_mellum_ssd_replay.py`.

- [ ] **Step 4: Implement the override** in `tools/mellum_ssd_replay.py`. `expert_bytes=None` means the trace's geometry, and the report must then be byte-identical to today's. An override is validated as `type(x) is int and x >= 1`; otherwise raise `ValueError` naming `expert_bytes`. It replaces `trace.header.geometry.expert_bytes` everywhere `replay` multiplies or divides by it. With an override, `replay` adds `"expert_bytes": {"value": N, "source": "override"}`, and `run_sweep` adds `assumptions["routing"]`. The measured-geometry check in `parse_header` is unchanged: the trace stays Q8_0, and only the costing changes.

- [ ] **Step 5: Run the tests, expecting a pass**

Run: `python3 -I tests/test_mellum_ssd_replay.py` → all pass. Then `git commit -m "feat: replay a Q8_0 trace at another expert size"`.

- [ ] **Step 6: Run the MXFP4 curve (descriptive, CPU only).** Copy the retarget's `curve.py` into the new evidence directory. Fix its `sys.path` depth for the new location, add an `expert_bytes` argument passed to `R.replay`, and keep the GiB rows 2, 2.5, 3, 4, 5, 6. Run it twice over `/Users/pauleveritt/.codex/visualizations/2026/10/28/mellum-generated-phase1-02/validated/raw/trace.jsonl` (read only; record its sha256 and confirm it starts `26b05a0c`): at 6,580,224, where the 2, 3, 4, 5, 6 GiB rows must equal `2026-10-28-mellum-ssd-retarget/curve-output.txt`, and at 3,290,112. Save both outputs to `curve-output.txt`.

- [ ] **Step 7: Write the note** `2026-10-28-mellum-mxfp4-miss-curve.md`, under 80 lines. Its honesty header says this is a replay of one thinking trace (n=1), with routing assumed unchanged by quantization and no SSD read. Include a table of LRU and relaxed-optimal loads/token and MB/token at both expert sizes, the command, the trace sha256, and "What would falsify this" and "Risks". Commit: `docs: the -02 miss curve at MXFP4 expert size`.

### Task 2: MXFP4 in the double CPU reference

**Files:**
- Modify: `tests/mellum_moe_ref.inc`; `tests/test_mellum_kernels.c` and `tests/ds4_test.c`, only to set the new field
- Create: `tests/test_mellum_moe_ref.c`
- Modify: `Makefile` (a `test-mellum-moe-ref` target, added to `test-mellum`, built like `tests/test_mellum_config`: `-DDS4_NO_GPU`, the test `#include`s `ds4.c`)

**Interfaces:**
- Produces: `mref_weights.expert_type` (`uint32_t`: 8 = Q8_0, 39 = MXFP4; anything else is a test bug, caught by `assert`); `static inline double mref_mxfp4_dot(const uint8_t *row, const double *x, uint32_t n)`; `#define MREF_MXFP4_BLOCK_BYTES 17u`. `mref_moe_row` dispatches on `expert_type`, with row strides 1,224 (gate/up) and 476 (down) for MXFP4. `mref_moe_row_staged` asserts `expert_type == 8`; staged MXFP4 waits for phase B, which reads the tile kernels' MXFP4 staging.

- [ ] **Step 1: Write the failing test** `tests/test_mellum_moe_ref.c`, with the `CHECK` style of `tests/test_mellum_stream.c`:
  - `test_mxfp4_hand_block`: `e = 127`, `qs[0] = 0x71`, every other nibble 0. With `x[0] = x[16] = 1` and the rest 0, `mref_mxfp4_dot == 6.5`. With `e = 128`, it is 13.0. With `qs[0] = 0xF8`: `x[0] = 1` gives 0.0 (nibble 8 is −0), and `x[16] = 1` gives −6.0.
  - `test_mxfp4_edge_scales`: `e = 0` and `e = 254`, one nibble 7, compared with `ds4_vec_dot_mxfp4_f32` on the same block. Equal as doubles after the engine's float result is widened.
  - `test_mxfp4_row_matches_engine_dot`: a 2,304-wide row of 72 blocks from a fixed LCG (seed 1), `e` in 120..134 and `x` in [−1, 1). `|ref − engine| <= 1e-5 * max(1, |ref|)`.
  - `test_moe_row_mxfp4_runs`: a full-size MXFP4 weight buffer (router F32, gate/up/down MXFP4, from the same LCG). `mref_moe_row` with `expert_type = 39` gives finite outputs. Its expert-0 gate value equals `mref_mxfp4_dot` of that row.

- [ ] **Step 2: Run it, expecting failure.** `make test-mellum-moe-ref` fails to compile (`mref_mxfp4_dot` undeclared). Commit the test and the Makefile target: `test: an MXFP4 double reference for Mellum experts`.

- [ ] **Step 3: Implement** `mref_mxfp4_dot` and the `expert_type` dispatch in `tests/mellum_moe_ref.inc`, with the decode in Global Constraints. Set `.expert_type = 8` at the existing initializers in `tests/test_mellum_kernels.c` and `tests/ds4_test.c`.

- [ ] **Step 4: Verify.** `make test-mellum-moe-ref` passes. `make tests/test_mellum_kernels.o ds4_test.o` compiles, which is compile only; do **not** run either. Commit: `feat: the Mellum double reference reads MXFP4 experts`.

### Task 3: The loader accepts all-MXFP4 experts and refuses mixtures

**Files:**
- Modify: `tests/mellum_fixture.h` (add `MF_MXFP4 = 39`, mutations `MF_EXPERTS_MXFP4`, `MF_LAYER2_MXFP4`, `MF_DOWN2_MXFP4`)
- Test: `tests/test_mellum_config.c` (next to the refusals at about `:1120`)
- Modify: `ds4.c` (the Mellum layout check, about `:5775-5777`; the comment at about `:5730`; the `(Q8_0 blocks)` wording at about `:7364`)

**Interfaces:**
- Produces: `static uint32_t mellum_expert_type(const ds4_model *m)`, or the codebase's model type at that site, in `ds4.c`. It returns `DS4_TENSOR_Q8_0` or `DS4_TENSOR_MXFP4`, read from `blk.0.ffn_gate_exps`. Any other type exits 1 with `ds4: Mellum routed experts must be q8_0 or mxfp4: <tensor> is <type>`. Plan B and Task 4 use it.

- [ ] **Step 1: Write the failing tests.** The fixture mutations:
  - `MF_EXPERTS_MXFP4`: all 84 `*_exps` tensors are `MF_MXFP4`;
  - `MF_LAYER2_MXFP4`: only layer 2's three are MXFP4;
  - `MF_DOWN2_MXFP4`: only layer 2's `ffn_down_exps` is MXFP4.

  Then in `test_mellum_config.c`:

```c
expect_accepted(MF_EXPERTS_MXFP4, "all experts MXFP4");
expect_refused(MF_LAYER2_MXFP4, "blk.2.ffn_gate_exps", "one layer MXFP4, the rest Q8_0");
expect_refused(MF_DOWN2_MXFP4, "blk.2.ffn_down_exps", "MXFP4 down beside Q8_0 gate/up");
```

  The existing `MF_DOWN_Q4K` and `MF_UP_Q4K` refusals stay as they are, as the sibling refusals.

- [ ] **Step 2: Run, expecting failure.** `make tests/test_mellum_config && ./tests/test_mellum_config` reports `FAIL all experts MXFP4` (exit 1, `has type mxfp4, expected q8_0`). Commit: `test: a Mellum file whose experts are all MXFP4 opens; mixtures do not`.

- [ ] **Step 3: Implement.** Before the per-layer loop, call `mellum_expert_type`, then pass its result instead of `Q8` for the three `*_exps` checks. `tensor_expect_layout`'s own message names a mismatching tensor. Reword `(Q8_0 blocks)` to `(32-value quantization blocks)`, and the layout comment to "Q8_0 or MXFP4 experts, uniform across all 84".

- [ ] **Step 4: Verify.** `./tests/test_mellum_config` passes, including the old refusals. Commit: `feat: open Mellum files whose routed experts are all MXFP4`.

### Task 4: Streaming slots come from the file's expert size

**Files:**
- Modify: `ds4_mellum_stream.h`, `ds4_mellum_stream.c`
- Test: `tests/test_mellum_stream.c`
- Modify: `ds4.c` (the S4 configuration at about `:74154-74172`)

**Interfaces:**
- Consumes: `mellum_expert_type` (Task 3).
- Produces: `#define DS4_MELLUM_Q8_EXPERT_BYTES 6580224ull`, `#define DS4_MELLUM_MXFP4_EXPERT_BYTES 3290112ull` (the old `DS4_MELLUM_EXPERT_BYTES` is removed, so the compiler finds every user); `uint64_t ds4_mellum_expert_bytes(uint32_t expert_type)` (8 → Q8, 39 → MXFP4, else 0); `uint32_t ds4_mellum_stream_slots(uint64_t cache_bytes, uint64_t expert_bytes)` (0 when `expert_bytes == 0`); `int ds4_mellum_stream_config(bool streaming, uint64_t cache_bytes, uint64_t expert_bytes, uint32_t *slots, char *err, size_t n)`.

- [ ] **Step 1: Write the failing tests.** Update the existing calls to pass `DS4_MELLUM_Q8_EXPERT_BYTES`; the Q8 expectations, 979 at 6 GiB, stay. Add:

```c
CHECK(ds4_mellum_expert_bytes(8) == 6580224ull && ds4_mellum_expert_bytes(39) == 3290112ull &&
      ds4_mellum_expert_bytes(12) == 0ull, "expert bytes by GGUF type");
CHECK(ds4_mellum_stream_slots(6ull << 30, DS4_MELLUM_MXFP4_EXPERT_BYTES) == 1958u, "6 GiB MXFP4");
CHECK(ds4_mellum_stream_slots(2684354560ull, DS4_MELLUM_MXFP4_EXPERT_BYTES) == 815u, "2.5 GiB MXFP4");
CHECK(ds4_mellum_stream_slots(1ull << 30, 0) == 0u, "unknown type holds none");
/* config: 63 MXFP4 experts refused naming 210567168; 64 accepted; expert_bytes 0 refused with
   "unsupported Mellum expert type" in err and rc -1 */
```

- [ ] **Step 2: Run, expecting failure.** `make test-mellum-stream` fails to compile. Commit: `test: Mellum streaming slots follow the expert size`.

- [ ] **Step 3: Implement** the new signatures. The minimum-slot refusal prints `DS4_MELLUM_STREAM_MIN_SLOTS * expert_bytes`. `expert_bytes == 0` → rc −1, `unsupported Mellum expert type`. In `ds4.c`, take the size from `ds4_mellum_expert_bytes(mellum_expert_type(...))` for `want`, the configure call, and `ssd_streaming_cache_bytes`.

- [ ] **Step 4: Verify.** `make test-mellum-stream` passes, `./tests/test_mellum_config` still passes, and `make ds4` compiles (compile only). `grep -n DS4_MELLUM_EXPERT_BYTES *.c *.h tests/*.c` is empty. Commit: `feat: size Mellum's streamed expert cache from the file's expert type`.

### Task 5: Preregister phase B, gate the branch, update the roadmap

**Files:**
- Create: `docs/superpowers/research/2026-10-28-mellum-s6-b-preregistration.md`
- Modify: `ROADMAP.md` (S6)

- [ ] **Step 1: Write the preregistration**, at most 120 lines, decided before any GPU run:
  - **(a) Model-free Metal tests:** Q8_0 unchanged; MXFP4 mid/down, decode and tiled, against `mref` within the Q8_0 tests' tolerances; repeated with `DS4_METAL_DISABLE_METAL4=1`.
  - **(b) Ours against llama.cpp** on the MXFP4 file: Homebrew `llama.cpp` 0.7.0; the S4 prompts; the bound for top-1 agreement and mean KLD, stated as numbers. Then our MXFP4 against our Q8_0 KLD, reported only.
  - **(c) Q8_0 no regression:** S4's A/B, six processes per arm, A/B/B/A, ±2% median decode and prefill.
  - **(d) Goal 6:** resident Q8_0 against MXFP4 on the M5 Max, with `n` per arm, decode t/s, time to first token, and the `powermetrics` method if any.
  - **(e) Streamed MXFP4 against resident:** token ids at the S4 settings.

  Each item names its pass condition and what it decides. Then add "What would falsify this" and "Risks". Commit: `docs: preregister S6 phase B`.

- [ ] **Step 2: Run the CPU gate.** `make test-mellum` (the model-free suite; take the lock as in Global Constraints if needed), then `python3 -I tests/test_mellum_ssd_replay.py`, then `git diff --check 3af0a3f..HEAD`. All pass; record the counts.

- [ ] **Step 3: Update `ROADMAP.md` S6**, at most 6 added lines: phase A done, with the commit range, the MXFP4 curve's 2.5 GiB LRU figure with its `n`, and "B waits for the S4 merge and a free GPU". Commit: `docs: S6 phase A complete, CPU only`.
