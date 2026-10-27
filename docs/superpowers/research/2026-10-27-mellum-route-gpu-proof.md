# Mellum routing GPU proof and exploratory replay

**Date:** 2026-10-27. **Question:** Can ordered resident routes be captured at
normal forward completion without changing logits? What does preliminary
3 GiB LRU replay show? **Scope:** diagnostic correctness/overhead on this
laptop; no streamed inference, storage benchmark or physical 8 GB result.

## Evidence identity

Project experiment, accessed 2026-10-27. Engine/harness source `0ebc1c4`, based
on `59932c9` / upstream `ee7e840`. Apple M5 Max, Mac17,7, 128 GiB RAM;
macOS 27.0.1 (26A434), Apple clang 21.0.0, Python 3.14.7. The dedicated C99
test translation unit omits `-ffast-math`; Metal keeps the ordinary build flags.
Shipping builds do not compile the observer hooks or link the native writer.

Model: `Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf`, SHA-256
`ea1e4ec737bb259ac2ffe1f20d412f458fd52dd230f42dce9e944f111093c29b`.
Model and source identities, input-byte identities, raw spools, full logs,
per-token replay and reproduction scripts are retained at:

`/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/mellum-gpu-proof/`

Sources: `manifest.json`, `summary.json`, `exploratory-raw-replay.json` and
each case's `run.log`, `buffered.routes`, `reference.routes` in that directory.
The manifest was written before matrix collection. The replay is explicitly
exploratory and does not constitute a qualified S1 trace. No measured v2 trace
was published with guessed kernel metadata.

## Findings

1. **Ordered capture and logits parity pass on the tested paths.**
   Evidence: eight cases, seven fresh sessions each (56 sessions); every
   session operation's 98,304 final logits compare byte-for-byte against the
   matching uninstrumented run. Complete buffered/reference spools match,
   covering 13,191 positions and 743 forwards per capture mode across cases.
   Source: per-case logs and spool hashes in `summary.json`; harness
   [mellum_routes_gpu.c](../../../tests/mellum_routes_gpu.c). Assessment: fact.

2. **Buffered capture adds no per-layer host wait in its hooks.**
   Evidence: the hook copies before scratch reuse using command-buffer blits;
   CPU readback follows the existing successful forward completion. The
   independent reference waits and reads original scratch per layer. All
   buffered hook wait counters are zero; reference waits are 28 per forward.
   At cap=1024 each ID buffer is 917,504 bytes; GPU + host total 1,835,008
   bytes, excluding model/session/global scratch, logits and file buffering.
   Source: graph hooks in `ds4.c`, `ds4_gpu_tensor_copy` in `ds4_metal.m`, logs.
   Assessment: fact; this is logical observer storage, not peak process RAM.

3. **Observer overhead is measurable; no-wait does not mean free.**
   Evidence: mean of warm baseline runs 3/5 versus buffered runs 4/6, with
   cold run 0 and synchronous reference run 2 excluded. Forward timings
   include observer serialization. Only two warm samples per mode; no
   confidence interval or cross-machine conclusion. Source: `summary.json`.
   Assessment: fact about this run, insufficient for a stable overhead estimate.

| Case | Initial prefill / total positions | Captured forwards | Total overhead | Decode overhead |
| --- | ---: | ---: | ---: | ---: |
| one-row | 1 / 29 | 29 | +1.2% | +0.7% |
| threshold31 | 31 / 164 | 134 | +4.5% | +5.0% |
| threshold32 | 32 / 164 | 133 | +6.2% | +6.8% |
| threshold33 | 33 / 164 | 132 | +9.5% | +10.1% |
| release2200 | 2200 / 2270 | 73 | +4.0% | +4.7% |
| release6000 | 5900 / 6000 | 106 | +2.2% | +6.2% |
| exact2200 requested | 2135 / 2200 | 68 | +0.2% | +4.7% |
| SIMD-group requested | 2135 / 2200 | 68 | +10.7% | +5.0% |

4. **These fixtures do not achieve 90% LRU hits at 3 GiB.**
   Evidence: existing replay code, unchanged policies, complete prefix replayed
   cold before every case. 3 GiB holds 489 experts (3,217,729,536 effective
   bytes), assuming 6,580,224 Q8 bytes per gate/up/down expert bundle. The two
   long release cases yield the following outcomes across *all* captured
   decode tokens, including post-prefill recovery. No steady-state window was
   selected afterward. Source: `exploratory-raw-replay.json`,
   [replay implementation](../../../tools/mellum_ssd_replay.py). Assessment:
   deterministic policy outcomes under the declared model; not a feasibility bound.

| Case | Decode tokens | LRU reference hits | Mean demand MB/token | Whole-trace LRU / relaxed GB |
| --- | ---: | ---: | ---: | ---: |
| release2200 | 69 | 79.92% | 296.01 | 51.944 / 32.447 |
| release6000 | 99 | 74.38% | 377.66 | 103.395 / 66.184 |

MB/GB here are decimal. Whole-trace relaxed loads/bytes are an optimistic
bound only for atomic full-group residency or the fixed sorted sequential
schedule. Decode/phase/tail outcomes are not bounds. No I/O, prefetch, staging,
fixed tensors, host reservation or read/compute overlap is modeled.

## Open questions and limits

- All continuations are teacher-forced existing port-test fixtures; the final
  token is an explicit one-token prompt append, not a realistic tool transcript.
  Names containing `heldout` do not establish a frozen SSD evaluation cohort.
- Requested exact/NAX controls are recorded; actual encoded projection,
  expert/tail and output-head paths remain unobserved. Raw routes are retained
  for debugging, not a decisive locality verdict or portable throughput claim.
- Public per-session attachment, failure/cancellation, rewind/replay and
  concurrent-session ownership remain open. The harness is single-session and
  append-only; it checks operation results, not every intermediate chunk's logits.
- Blits end compute encoders. This may contribute to overhead; its isolated
  contribution has not been measured. No physical SSD activity was benchmarked.
- Full `make test` needs the absent default `ds4flash.gguf`. Its first attempt
  also encountered a separate server's instance lock; neither failure is a Mellum
  parity failure. One smoke run was invalidated by editing its active shell
  file; the frozen wrapper rerun passed, including inherited chunk=1/exact=1.

## Recommendation

Confidence is high in this bounded capture primitive on the tested laptop.
Finish S2b with narrow session attachment, lifecycle tests and actual encoding
provenance, then freeze representative generated/tool workloads before S3.
Keep the 3 GiB investigation open: these preliminary LRU outcomes motivate
measuring locality and scheduling rather than assuming 90% hits. No latency
prediction follows; 10 tok/s remains an unagreed proposed checkpoint.

Verification: the model-free Mellum suite (33 replay tests, 12 exporter tests,
native/config/frontend tests and 14 logits tests), live route smoke (21/21
sessions under hostile inherited chunk/exact controls), resident trace gate,
Metal kernel gate and full Mellum fast-path gate pass. Independent code review
found only smoke coverage/environment gaps; the corrected wrapper passes.
