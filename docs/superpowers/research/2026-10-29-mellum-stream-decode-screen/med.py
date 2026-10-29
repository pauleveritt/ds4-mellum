import json, statistics as st, sys
ref = None
for f in sys.argv[1:]:
    d = json.load(open(f)); t = d['tokens']; ns = [x['ns'] for x in t[1:]]; ids = [x['id'] for x in t]
    ref = ref or ids
    n = len(t) - 1
    print('%-28s median %.2f ms  loads/tok %.2f  stops/tok %.2f  ttft %.2f s  ids==first %s' % (f.split('/')[-1], st.median(ns) / 1e6,
          (t[-1]['loads'] - t[0]['loads']) / n, (t[-1]['stops'] - t[0]['stops']) / n, d['ttft_ns'] / 1e9, ids == ref))
