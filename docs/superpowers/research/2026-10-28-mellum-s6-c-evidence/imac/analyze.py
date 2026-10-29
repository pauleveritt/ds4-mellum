"""Phase C's figures, recomputed from the iMac captures alone.
    python3 -I analyze.py            (from this directory)
Per workload run: decode t/s by the preregistered metric, (n-1) / the summed ns of tokens 1..n-1; time to first
token; loads and stops per token over the same tokens; the Metal and footprint peaks; swap-outs before and after;
and whether the 512 ids equal the M5's resident MXFP4 ids from the speed run at the merge.
"""
import glob, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
M5 = os.path.join(HERE, '..', '..', '2026-10-29-mellum-speed-run', 'mx-resident-1.measure.json')
m5_ids = [t['id'] for t in json.load(open(M5))['tokens']]


def vmstat(path):
    if not os.path.exists(path):
        return None, None
    kv = dict(line.split() for line in open(path) if line.startswith('swapouts_'))
    return int(kv.get('swapouts_before', -1)), int(kv['swapouts_after']) if 'swapouts_after' in kv else None


def run(f):
    j = json.load(open(f))
    t = j['tokens']
    n = len(t) - 1
    b, a = vmstat(f.replace('.measure.json', '.vmstat'))
    return dict(tps=n / (sum(x['ns'] for x in t[1:]) / 1e9), ttft=j['ttft_ns'] / 1e9, tokens=len(t),
                loads=(t[-1]['loads'] - t[0]['loads']) / n, stops=(t[-1]['stops'] - t[0]['stops']) / n,
                metal=j['metal_allocated_peak'], foot=j['phys_footprint_peak'], slots=j['slots'],
                swap=None if a is None else a - b, ids=[x['id'] for x in t] == m5_ids)


def table(stage):
    files = sorted(glob.glob(os.path.join(HERE, stage, '*.measure.json')))
    print(f'\n## {stage}')
    print('| run | slots | tokens | decode t/s | first token s | loads/tok | stops/tok | Metal peak B | '
          'footprint peak B | swap-outs grown | ids = M5 resident |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    rs = []
    for f in files:
        r = run(f)
        rs.append(r)
        print('| %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %d | %d | %s | %s |' % (
            os.path.basename(f).split('.')[0], r['slots'], r['tokens'], r['tps'], r['ttft'], r['loads'],
            r['stops'], r['metal'], r['foot'], r['swap'], r['ids']))
    for f in sorted(glob.glob(os.path.join(HERE, stage, '*.vmstat'))):
        if not os.path.exists(f.replace('.vmstat', '.measure.json')):
            print(f'| {os.path.basename(f).split(".")[0]} | no measure.json (run did not finish) |')
    if len(rs) > 1:
        tps = [r['tps'] for r in rs]
        tt = [r['ttft'] for r in rs]
        print(f'median decode {st.median(tps):.2f} t/s (range {min(tps):.2f}-{max(tps):.2f}), median first token '
              f'{st.median(tt):.2f} s (range {min(tt):.2f}-{max(tt):.2f}), n={len(rs)}')


def agree(stage):
    print(f'\n## {stage}: logits agreement against the M5')
    for f in sorted(glob.glob(os.path.join(HERE, stage, 'agree.*.json'))):
        j = json.load(open(f))
        s = j.get('summary', j)
        keys = [k for k in s if not isinstance(s[k], (list, dict))]
        print(os.path.basename(f), {k: s[k] for k in keys})


for stage in ['s6-c-imac-2/cache', 's6-c-imac-3/cache', 's6-c-imac-2/footprint-closed',
              's6-c-imac-3/footprint-closed-interrupted', 's6-c-imac-3/footprint-closed-interrupted/footprint-closed',
              's6-c-imac-3/footprint-closed']:
    table(stage)
for stage in ['s6-c-imac-2/correct', 's6-c-imac-3/correct']:
    agree(stage)
