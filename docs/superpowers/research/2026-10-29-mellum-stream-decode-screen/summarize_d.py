"""Summarize d.pread-cb.txt: pread wall and rate by experts per load, and command-buffer spans."""
import re, statistics as st, sys
t = open(sys.argv[1]).read()
pr = [(int(e), int(th), float(w)) for e, th, w in re.findall(r'parallel pread layer=\d+ experts=(\d+) tensors=\d+ threads=(\d+) bytes=[\d.]+ GiB wall=([\d.]+) ms', t)]
print('pread events', len(pr))
for k in sorted({p[0] for p in pr}):
    ws = [p[2] for p in pr if p[0] == k]
    print(' experts', k, 'n', len(ws), 'median wall ms', round(st.median(ws), 3), 'GB/s', round(k * 3.290112 / st.median(ws), 2),
          'threads', st.median([p[1] for p in pr if p[0] == k]))
cb = [(float(a), float(b), float(c)) for a, b, c in re.findall(r"cb 'command batch'.*?encode ([\d.]+) ms, commit->done ([\d.]+) ms, gpu span ([\d.]+) ms", t)]
print('cbs', len(cb), 'sum commit->done', round(sum(c[1] for c in cb), 1), 'sum gpu span', round(sum(c[2] for c in cb), 1),
      'median gap', round(st.median([c[1] - c[2] for c in cb]), 3))
