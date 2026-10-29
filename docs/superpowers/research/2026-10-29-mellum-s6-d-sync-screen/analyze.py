"""The synchronous prefetch screen's figures, from run_sync_screen.sh's output directory.
    python3 -I analyze.py OUT
Per run: decode ms/token (phase C's metric: summed ns of tokens 1..n-1 over n-1), stops and experts loaded per token
(the stop profile's line), prefetch calls and experts the prefetch read (per token), and swap-out growth.  Then the
arms' means, stops avoided, the measured change, and whether every run's ids are equal.
"""
import glob, json, os, re, statistics as st, sys

out = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(out, '*.measure.json'))):
    n = os.path.basename(f).split('.')[0]
    j = json.load(open(f))
    t = j['tokens']
    err = open(os.path.join(out, n + '.stderr')).read()
    m = re.search(r'mellum_stream_profile tokens (\d+) .*?stops (\d+) .*?experts_loaded (\d+)', err)
    p = re.search(r'screen_prefetch calls (\d+) loaded (\d+)', err)
    kv = dict(l.split() for l in open(os.path.join(out, n + '.vmstat')) if l.startswith('swapouts_'))
    tok = int(m.group(1))
    rows[n] = dict(ms=sum(x['ns'] for x in t[1:]) / (len(t) - 1) / 1e6, stops=int(m.group(2)) / tok,
                   demand=int(m.group(3)) / tok, calls=int(p.group(1)) / tok if p else 0.0,
                   pre=int(p.group(2)) / tok if p else 0.0,
                   swap=int(kv['swapouts_after']) - int(kv['swapouts_before']), ids=[x['id'] for x in t])
print('| run | decode ms/token | stops/token | demand experts/token | prefetch calls/token | prefetched experts/token | swap-out growth |')
print('|---|---|---|---|---|---|---|')
for n, r in rows.items():
    print(f"| {n} | {r['ms']:.1f} | {r['stops']:.2f} | {r['demand']:.2f} | {r['calls']:.2f} | {r['pre']:.2f} | {r['swap']} |")
arm = lambda a, k: st.mean(r[k] for n, r in rows.items() if n.startswith(a))
off_ms, on_ms = arm('off', 'ms'), arm('on', 'ms')
avoided = arm('off', 'stops') - arm('on', 'stops')
print(f"\nmeans: off {off_ms:.1f} ms/token, on {on_ms:.1f}; measured change {off_ms - on_ms:+.1f} ms/token (positive: on is faster)")
print(f"stops avoided per token {avoided:.2f}; prefetched experts per token {arm('on', 'pre'):.2f}; "
      f"demand experts per token off {arm('off', 'demand'):.2f}, on {arm('on', 'demand'):.2f}")
ref = next(iter(rows.values()))['ids']
print('every run\'s ids equal:', all(r['ids'] == ref for r in rows.values()))
