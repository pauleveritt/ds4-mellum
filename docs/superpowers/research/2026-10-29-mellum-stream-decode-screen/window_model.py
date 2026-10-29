"""Predict GPU-wait ms per token for windowed passes from b.rounds.txt and regress.py's fit: a window of W layers
pays the fixed round cost, its executed layers, and, when a stop ends it, the layers of that window after the stop."""
import re, sys
FIXED, EXEC, SUPP, L = 0.2587, 0.3139, 0.0253, 28
rows = [tuple(map(int, m.groups()[:3])) for m in re.finditer(r'msp_round from (\d+) to (\d+) stopped (\d)', open(sys.argv[1]).read())][17:]
tokens = sum(1 for r in rows if not r[2])
for W in [1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 28]:
    ms = 0.0
    for f, t, stopped in rows:
        span = t - f + (1 if stopped else 0)          # layers this round reaches, the stop layer's attention included
        windows = max(1, -(-span // W))
        tail = (windows * W - span) if stopped else 0  # suppressed layers left in the last window
        tail = min(tail, L - t) if stopped else 0
        ms += windows * FIXED + (t - f) * EXEC + tail * SUPP
    print('W %2d  predicted GPU wait %.2f ms/token' % (W, ms / tokens))
