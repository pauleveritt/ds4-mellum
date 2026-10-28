"""D1's table from tests/mellum_decode_profile.sh's raw logs (phase 8b).

    python3 -I tests/mellum_decode_profile_summary.py OUTDIR MODEL.gguf READ_GBPS

Per prompt length:
  - t/s: the median of the speed processes' CLI generation lines (the first process of each length dropped);
    wall ms per token is its inverse;
  - from the DS4_METAL_CB_TIMES process: the decode command buffers, which are the trailing run of buffers whose
    GPU time is under 3x the median of the last 32 (prefill chunks and warmups are far longer); their count, and
    the medians of GPU time and of the gap from the previous buffer's GPU end (the GPU idle between steps: the
    logits read, sampling, output and encoding the next step);
  - bytes per step (tests/mellum_decode_bytes.py's arithmetic) over the median GPU time, against READ_GBPS;
  - from the same buffers' second CB_TIMES line: the median time from one decode buffer's completion to the next
    (a whole step, wall), its encode time and its commit-to-done time;
  - from the DS4_METAL_ENCODER_TIMELINE process (its log xz-compressed): per decode buffer, the encoders and
    dispatches, the summed gaps between encoders, each kernel group's share of the summed encoder durations and
    the 20 costliest kernel and grid pairs (durations are an upper bound, since each dispatch group gets its own
    pass, and passes overlap, so their sum exceeds the buffer's GPU span).
The timeline's decode buffers are its last N, N the count found in the CB_TIMES process (both runs are greedy on
the same prompt, so they decode the same tokens).
"""

import importlib.util
import lzma
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mellum_decode_bytes", HERE / "mellum_decode_bytes.py")
_bytes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bytes)

CB = re.compile(r"ds4: cb (.*?): driver ([\d.]+) us, queue-wait (-?[\d.]+) us, gpu ([\d.]+) us, "
                r"gap-from-prev-gpu-end (-?[\d.]+) us")
# the second line DS4_METAL_CB_TIMES prints per buffer: a monotonic stamp at completion, the encode time (from the
# buffer's creation to its commit) and commit to done
CB2 = re.compile(r"ds4: cb '(.*?)': mono ([\d.]+) ms: encode ([\d.]+) ms, commit->done ([\d.]+) ms")

# kernel name -> stage, first match wins
STAGES = [
    ("attention", r"attn|flash|swa|ring"),
    ("router + top-k", r"topk|router|argsort|multi_gemv"),
    ("experts", r"moe|mul_mv_id|expert"),
    ("norms, rope, adds, copies", r"norm|rope|add|cpy|copy|blit|get_rows|embed"),
    ("output head", r"^head:"),
    ("projections (Q8_0 matvec)", r"mul_mv|matvec|q8_0"),
]


def stage_of(kernel):
    for name, pat in STAGES:
        if re.search(pat, kernel, re.I):
            return name
    return "other"


def step_bytes(kv, tensors, context):
    return sum(v[0] for v in _bytes.weight_groups(kv, tensors).values()) + _bytes.kv_read(kv, context)


def decode_run(gpu):
    if len(gpu) < 8:
        return 0
    ref = statistics.median(gpu[-32:])
    n = 0
    for g in reversed(gpu):
        if g >= 3 * ref:
            break
        n += 1
    return n


def main(out, model, read_gbps):
    out = Path(out)
    lengths = {}
    for line in open(out / "setup.txt"):
        m = re.match(r"length (\d+): (\d+) tokens, \d+ bytes, ctx (\d+)", line)
        if m:
            lengths[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    runs = [l.split() for l in open(out / "runs.txt")]
    _version, kv, tensors = _bytes.read_gguf(model)
    print("read bandwidth %.1f GB/s (tests/mellum_copy_bandwidth)\n" % read_gbps)
    for target, (ntok, ctx) in lengths.items():
        gen = [float(r[4]) for r in runs if r[0] == target and r[2] != "1"]
        tps = statistics.median(gen)
        lines = open(out / ("l%s_cbtimes.err" % target)).read().splitlines()
        cbs = [m for m in map(CB.search, lines) if m]
        cb2 = [m for m in map(CB2.search, lines) if m]
        assert len(cbs) == len(cb2), "the two CB_TIMES lines do not pair"
        gpu = [float(m.group(4)) for m in cbs]
        n_dec = decode_run(gpu)
        dec = cbs[len(cbs) - n_dec:]
        g_ms = statistics.median(float(m.group(4)) for m in dec) / 1e3
        gap_ms = statistics.median(float(m.group(5)) for m in dec[1:]) / 1e3
        drv_ms = statistics.median(float(m.group(2)) for m in dec) / 1e3
        dec2 = cb2[len(cb2) - n_dec:]
        mono = [float(m.group(2)) for m in dec2]
        step_ms = statistics.median(b - a for a, b in zip(mono, mono[1:]))
        enc_ms = statistics.median(float(m.group(3)) for m in dec2)
        done_ms = statistics.median(float(m.group(4)) for m in dec2)
        ctx_mid = ntok + 128  # the middle of 256 generated tokens
        b = step_bytes(kv, tensors, ctx_mid)
        print("length %s: %d prompt tokens, ctx %d" % (target, ntok, ctx))
        print("  speed: n %d, generation median %.2f t/s (%.2f-%.2f), wall %.3f ms per token" %
              (len(gen), tps, min(gen), max(gen), 1e3 / tps))
        print("  cb times: %d command buffers, %d decode (trailing); per decode buffer median GPU %.3f ms, gap "
              "from the previous GPU end %.3f ms, driver %.3f ms; GPU / (GPU + gap) %.1f%%" %
              (len(cbs), n_dec, g_ms, gap_ms, drv_ms, 100 * g_ms / (g_ms + gap_ms)))
        print("  per decode step (medians): completion to completion %.3f ms; encode %.1f ms; commit to done %.1f "
              "ms; GPU %.3f ms = %.1f%% of the step; the rest %.3f ms (logits read, sampling, output, waits)" %
              (step_ms, enc_ms, done_ms, g_ms, 100 * g_ms / step_ms, step_ms - enc_ms - done_ms))
        print("  bytes per step at context %d: %.1f MB; at the median GPU time %.1f GB/s, %.1f%% of %.1f; "
              "the read floor %.3f ms (%.0f t/s)" %
              (ctx_mid, b / 1e6, b / g_ms / 1e6, 100 * b / g_ms / 1e6 / read_gbps, read_gbps,
               b / read_gbps / 1e6, read_gbps * 1e9 / b))
        tl = out / ("l%s_timeline.txt.xz" % target)
        bufs, encs = [], {}
        for line in lzma.open(tl, "rt"):
            f = line.split()
            if f and f[0] == "B":
                bufs.append((int(f[1]), int(f[2]), float(f[3]), float(f[4])))
            elif f and f[0] == "E":
                encs.setdefault(int(f[1]), []).append((float(f[5]), float(f[6]), int(f[8]), f[11] if len(f) > 11 else "?",
                                                       f[9]))
        dec_bufs = bufs[len(bufs) - n_dec:]
        stage_us, kern_us, disp, nenc, gaps, spans, shape_us = {}, {}, [], [], [], [], {}
        for seq, n_e, start, end in dec_bufs:
            e = encs.get(seq, [])
            nenc.append(len(e))
            disp.append(sum(x[2] for x in e))
            gaps.append(sum(x[1] for x in e))
            spans.append((end - start) / 1e6)
            for dur, _gap, _nd, kernel, tg in e:
                if "mul_mv_q8_0" in kernel and tg.startswith("49152x"):
                    kernel = "head:" + kernel  # the LM head: 98,304 rows, two per threadgroup
                stage_us[stage_of(kernel)] = stage_us.get(stage_of(kernel), 0.0) + dur
                kern_us[kernel] = kern_us.get(kernel, 0.0) + dur
                shape_us.setdefault((kernel, tg), []).append(dur)
        total = sum(stage_us.values())
        nb = len(dec_bufs)
        print("  timeline: %d decode buffers; per buffer median %d encoders, %d dispatches, GPU span %.3f ms, "
              "encoder durations %.3f ms, gaps between encoders %.3f ms" %
              (nb, statistics.median(nenc), statistics.median(disp), statistics.median(spans),
               total / nb / 1e3, statistics.median(gaps) / 1e3))
        for name, us in sorted(stage_us.items(), key=lambda x: -x[1]):
            print("    %-34s %6.1f%%  %.3f ms per step" % (name, 100 * us / total, us / nb / 1e3))
        print("    kernel and grid (threadgroups): dispatches per step, median us each, us per step")
        for (k, tg), v in sorted(shape_us.items(), key=lambda x: -sum(x[1]))[:20]:
            print("      %-44s %-10s %5.1f  %8.2f  %8.1f" % (k, tg, len(v) / nb, statistics.median(v), sum(v) / nb))
        print()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
