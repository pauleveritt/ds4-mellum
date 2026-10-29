import re, sys
rows = [tuple(map(float, m.groups())) for m in re.finditer(r'msp_round from (\d+) to (\d+) stopped (\d) fin_ms ([\d.]+)', open(sys.argv[1]).read())]
L = max(r[1] for r in rows)
# skip the first token's rounds (prefill-adjacent): drop the first 17 rows
rows = rows[17:]
import numpy as np
X = np.array([[1, r[1]-r[0], L-r[1] if r[2] else 0] for r in rows]); y = np.array([r[3] for r in rows])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
print('n_layer', L, 'rounds', len(rows))
print('fixed_ms %.4f per_executed_layer_ms %.4f per_suppressed_layer_ms %.4f' % tuple(coef))
full = [r[3] for r in rows if r[0]==0 and not r[2]]
print('no-stop full tokens', len(full), 'median', np.median(full) if full else None)
st = [r for r in rows if r[2]]
print('mean suppressed layers per stop %.2f, mean executed per round %.2f' % (np.mean([L-r[1] for r in st]), np.mean([r[1]-r[0] for r in rows])))
pred = X @ coef; print('r2 %.3f' % (1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()))
