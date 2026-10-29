"""The preregistered verdict from the A/B captures: per run, the median per-token ns over decode tokens 1-511;
per arm, the median of run medians with the range; identity of all 512 ids across every streamed and resident run.
    python3 -I analyze.py DIR
"""
import json, os, statistics as st, sys
d = sys.argv[1]
def load(name):
    j = json.load(open(os.path.join(d, name + '.measure.json')))
    t = j['tokens']
    assert len(t) == 512, (name, len(t))
    n = len(t) - 1
    return dict(ms=st.median(x['ns'] for x in t[1:]) / 1e6, ids=[x['id'] for x in t], ttft=j['ttft_ns'] / 1e9,
                loads=(t[-1]['loads'] - t[0]['loads']) / n, stops=(t[-1]['stops'] - t[0]['stops']) / n,
                alloc=j['metal_allocated_peak'] / 2**30)
runs = {n: load(n) for n in ['A1', 'B1', 'B2', 'A2', 'A3', 'B3', 'B-resident', 'A-3g', 'B-3g']}
for n, r in runs.items():
    print('%-11s %6.2f ms/token (%.1f t/s)  ttft %.2f s  loads/tok %.2f  stops/tok %.2f  metal peak %.2f GiB'
          % (n, r['ms'], 1e3 / r['ms'], r['ttft'], r['loads'], r['stops'], r['alloc']))
A = [runs[n]['ms'] for n in ('A1', 'A2', 'A3')]
B = [runs[n]['ms'] for n in ('B1', 'B2', 'B3')]
ma, mb = st.median(A), st.median(B)
print('arm A median %.2f ms (%.2f-%.2f), n=3' % (ma, min(A), max(A)))
print('arm B median %.2f ms (%.2f-%.2f), n=3' % (mb, min(B), max(B)))
ref = runs['A1']['ids']
same = {n: r['ids'] == ref for n, r in runs.items()}
print('ids equal to A1:', same)
print('rule 1 (identity, every A and B run at 2.5 GiB):', 'PASS' if all(same[n] for n in ('A2', 'A3', 'B1', 'B2', 'B3')) else 'FAIL')
print('rule 2 (B <= 0.85 A): B/A = %.3f ->' % (mb / ma), 'PASS' if mb <= 0.85 * ma else 'FAIL')
