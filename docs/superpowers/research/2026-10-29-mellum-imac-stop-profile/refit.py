"""Four-term fit of each GPU round's wait (`finish`) from a profiled run's msp_round lines (Fable review, 2026-10-29):
    finish = fixed + a * executed layers + b * suppressed layers + c * [final round: the head runs]
A stopped round runs `to - from` layers and suppresses `n_layer - to`; the final round runs `n_layer - from` and the head.
    python3 -I refit.py STDERR...
"""
import re, statistics as st, sys


def solve(A, y):
    # least squares through the normal equations, Gaussian elimination
    n = len(A[0])
    M = [[sum(r[i] * r[j] for r in A) for j in range(n)] + [sum(r[i] * v for r, v in zip(A, y))] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


for f in sys.argv[1:]:
    rows = [tuple(map(float, m)) for m in re.findall(r'msp_round from (\d+) to (\d+) stopped (\d) fin_ms ([\d.]+)', open(f).read())]
    L = max(r[1] for r in rows)
    A = [[1.0, r[1] - r[0], (L - r[1]) if r[2] else 0.0, 0.0 if r[2] else 1.0] for r in rows]
    y = [r[3] for r in rows]
    c = solve(A, y)
    pred = [sum(a * b for a, b in zip(row, c)) for row in A]
    my = st.mean(y)
    r2 = 1 - sum((a - b) ** 2 for a, b in zip(y, pred)) / sum((a - my) ** 2 for a in y)
    fin = [r[3] for r in rows if not r[2]]
    stopped = [r[3] for r in rows if r[2]]
    print('%s: rounds %d; fixed %.3f ms, %.3f ms per executed layer, %.4f ms per suppressed layer, head %.3f ms; '
          'r2 %.3f; median final round %.3f ms, stopped round %.3f ms'
          % (f.split('/')[-2] + '/' + f.split('/')[-1], len(rows), *c, r2, st.median(fin), st.median(stopped)))
