"""Where a streamed decode stop's time goes, from run_imac_profile.sh's output directory.
    python3 -I analyze.py OUT
Per run: decode ms/token by phase C's metric and swap-out growth.  Per profiled run: the per-token split the patch
prints, the per-stop medians (read start, resumed-pass encode, read wait after the encode), and a fit of each
round's GPU wait (`finish`) against the layers it executed.  The reads run: pread wall per call and the command
buffers' commit-to-done against their GPU span, decode only (lines after the first profiled round).
"""
import glob, json, os, re, statistics as st, sys

out = sys.argv[1]


def ms_per_token(f):
    t = json.load(open(f))['tokens']
    return sum(x['ns'] for x in t[1:]) / (len(t) - 1) / 1e6


def swap(f):
    kv = dict(l.split() for l in open(f) if l.startswith('swapouts_'))
    return int(kv['swapouts_after']) - int(kv['swapouts_before'])


def med(xs):
    return st.median(xs) if xs else float('nan')


print('| run | decode ms/token | swap-out growth |\n|---|---|---|')
for f in sorted(glob.glob(os.path.join(out, '*.measure.json'))):
    n = os.path.basename(f).split('.')[0]
    print(f'| {n} | {ms_per_token(f):.1f} | {swap(os.path.join(out, n + ".vmstat"))} |')

for f in sorted(glob.glob(os.path.join(out, 'prof-*.stderr')) + glob.glob(os.path.join(out, 'reads.stderr'))):
    text = open(f).read()
    print(f'\n## {os.path.basename(f)}')
    m = re.search(r'mellum_stream_profile .*', text)
    print(m.group(0) if m else 'no profile line')
    stops = [tuple(map(float, g)) for g in
             re.findall(r'msp_stop layer \d+ loaded (\d+) begin_ms ([\d.]+) encode_ms ([\d.]+) wait_ms ([\d.]+)', text)]
    rounds = [tuple(map(float, g)) for g in
              re.findall(r'msp_round from (\d+) to (\d+) stopped (\d) fin_ms ([\d.]+)', text)]
    if stops:
        print('stops %d: median experts loaded %.1f, read start %.3f ms, encode %.3f ms, read wait %.3f ms'
              % (len(stops), med([s[0] for s in stops]), med([s[1] for s in stops]), med([s[2] for s in stops]),
                 med([s[3] for s in stops])))
        # the reads start in load_begin and are done when load returns: their span is start + encode + wait
        for k in sorted({int(s[0]) for s in stops}):
            w = [s[3] for s in stops if s[0] == k]
            span = med([s[1] + s[2] + s[3] for s in stops if s[0] == k])
            gbs = f'{k * 3.290112 / span:.2f} GB/s' if k else 'no reads'
            print(f'  {k} loaded: n {len(w)}, median read wait {med(w):.3f} ms, read span {span:.3f} ms ({gbs})')
    if rounds:
        L = max(r[1] for r in rounds)
        ex = [r[1] - r[0] for r in rounds]
        fin = [r[3] for r in rounds]
        mx, my = st.mean(ex), st.mean(fin)
        b = sum((x - mx) * (y - my) for x, y in zip(ex, fin)) / sum((x - mx) ** 2 for x in ex)
        print('rounds %d (n_layer %d): finish median %.3f ms; fit finish = %.3f ms + %.3f ms per executed layer; '
              'mean layers executed %.2f' % (len(rounds), L, med(fin), my - b * mx, b, mx))
    i = text.find('msp_round')
    dec = text[i:] if i >= 0 else ''
    pr = [(int(e), float(w)) for e, w in re.findall(r'parallel pread layer=\d+ experts=(\d+) .*?wall=([\d.]+) ms', dec)]
    if pr:
        for k in sorted({p[0] for p in pr}):
            w = [p[1] for p in pr if p[0] == k]
            print(f'  pread {k} experts: n {len(w)}, median wall {med(w):.3f} ms, '
                  f'{k * 3.290112 / med(w):.2f} GB/s')
    cb = [(float(a), float(b)) for a, b in re.findall(r"cb 'command batch'.*?commit->done ([\d.]+) ms, gpu span ([\d.]+) ms", dec)]
    if cb:
        print('  decode command buffers %d: median commit->done %.3f ms, gpu span %.3f ms, gap %.3f ms'
              % (len(cb), med([c[0] for c in cb]), med([c[1] for c in cb]), med([c[0] - c[1] for c in cb])))
