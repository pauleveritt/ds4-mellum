"""Summarize a speed-run directory: per configuration, the median (and range) over its runs of decode ms/token
(median of tokens 1-511), decode t/s, time to first token and prefill t/s, loads and stops per token, the Metal
peak, and whether every run's 512 ids equal the configuration's first.
    python3 -I summarize.py DIR
"""
import collections, glob, json, os, re, statistics as st, sys

d = sys.argv[1]
by = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, '*.measure.json'))):
    name = re.sub(r'-\d+\.measure\.json$', '', os.path.basename(f))
    j = json.load(open(f))
    t = j['tokens']
    n = len(t) - 1
    by[name].append(dict(ms=st.median(x['ns'] for x in t[1:]) / 1e6, ttft=j['ttft_ns'] / 1e9,
                         prefill=j['prompt_tokens'] / (j['ttft_ns'] / 1e9), ids=[x['id'] for x in t],
                         loads=(t[-1]['loads'] - t[0]['loads']) / n, stops=(t[-1]['stops'] - t[0]['stops']) / n,
                         alloc=j['metal_allocated_peak'] / 2**30, prompt=j['prompt_tokens']))
print('| Configuration | n | Decode ms/token, median (range) | Decode t/s | First token s, median (range) | '
      'Loads/token | Stops/token | Metal peak GiB | Ids equal |')
print('|---|---|---|---|---|---|---|---|---|')
for name, rs in by.items():
    ms = [r['ms'] for r in rs]
    tt = [r['ttft'] for r in rs]
    same = all(r['ids'] == rs[0]['ids'] for r in rs)
    print('| %s | %d | %.2f (%.2f-%.2f) | %.1f | %.2f (%.2f-%.2f) | %.2f | %.2f | %.2f | %s |'
          % (name, len(rs), st.median(ms), min(ms), max(ms), 1e3 / st.median(ms), st.median(tt), min(tt), max(tt),
             st.median(r['loads'] for r in rs), st.median(r['stops'] for r in rs), max(r['alloc'] for r in rs),
             'yes' if same else 'NO'))
mx = [rs[0]['ids'] for name, rs in by.items() if name.startswith('mx-')]
q8 = [rs[0]['ids'] for name, rs in by.items() if name.startswith('q8-')]
print('MXFP4 configurations agree with each other:', all(x == mx[0] for x in mx))
print('Q8_0 configurations agree with each other:', all(x == q8[0] for x in q8))
