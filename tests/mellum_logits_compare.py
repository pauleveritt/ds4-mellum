"""Compare two traced Mellum runs' logits, for tests/mellum_oracle_compare.sh.

    python3 -I tests/mellum_logits_compare.py A B POSITIONS EXACT TOL KLTOL KLMAX DLOGP

A and B are trace prefixes: the logits at position N are A.pN.logits.f32 (little-endian f32) and the
session's last-row logits are A.session.logits.f32.  POSITIONS is a comma-separated list.  B is the
reference.  EXACT 1: every file must be byte-equal; 0, for every row: the same top-1, max|d logit| <=
TOL, KL(B || A) <= KLMAX and |log p_A(t) - log p_B(t)| <= DLOGP for B's top-1 token t; and over the
traced positions, mean KL <= KLTOL.  The session row is the last position's row again, so it is held
to the per-row rules but left out of the mean.  Under either rule every logit of both runs must be
finite first.  Prints one summary line; exit 0 when the rules hold, 1 otherwise.
"""

import math
import struct
import sys


def log_softmax(v):
    m = max(v)
    s = math.log(sum(math.exp(t - m) for t in v))
    return [t - m - s for t in v]


def main(argv):
    a, b, positions, exact = argv[1], argv[2], argv[3].split(','), argv[4] == '1'
    tol, kltol, klmax, dlogp = float(argv[5]), float(argv[6]), float(argv[7]), float(argv[8])
    names = ['p%s' % p for p in positions] + ['session']
    rows = []
    for name in names:
        x = open('%s.%s.logits.f32' % (a, name), 'rb').read()
        y = open('%s.%s.logits.f32' % (b, name), 'rb').read()
        u = struct.unpack('<%df' % (len(x) // 4), x)
        v = struct.unpack('<%df' % (len(y) // 4), y)
        for prefix, row in ((a, u), (b, v)):
            bad = sum(not math.isfinite(t) for t in row)
            if bad:
                print('FAIL %s.%s.logits.f32: %d of %d logits are not finite' % (prefix, name, bad, len(row)))
                return 1
        rows.append((name, x, y, u, v))
    worst, agree, equal, kl_sum = 0.0, 0, 0, 0.0
    kl_worst, kl_at, dl_worst, dl_at = 0.0, '-', 0.0, '-'
    for name, x, y, u, v in rows:
        equal += x == y
        n = len(u)
        worst = max(worst, max(abs(i - j) for i, j in zip(u, v)))
        top = max(range(n), key=v.__getitem__)
        agree += max(range(n), key=u.__getitem__) == top
        if x == y:
            continue
        lq, lp = log_softmax(u), log_softmax(v)
        kl = sum(math.exp(pi) * (pi - qi) for pi, qi in zip(lp, lq) if pi > -27.6)
        dl = abs(lq[top] - lp[top])
        if name != 'session':
            kl_sum += kl
        if kl > kl_worst:
            kl_worst, kl_at = kl, name
        if dl > dl_worst:
            dl_worst, dl_at = dl, name
    kl_mean = kl_sum / len(positions)
    if exact:
        ok = equal == len(rows)
        rule = 'byte-equal'
    else:
        ok = agree == len(rows) and worst <= tol and kl_mean <= kltol and kl_worst <= klmax and dl_worst <= dlogp
        rule = 'top-1 equal, max|d| <= %g, mean KL <= %g, worst KL <= %g, |d log p(top-1)| <= %g' % (
            tol, kltol, klmax, dlogp)
    print('%s (%s): %d of %d logits files byte-equal, top-1 agree %d/%d, max|d logit| %.3e, mean KL %.2e, '
          'worst KL %.2e at %s, max |d log p(top-1)| %.2e at %s' %
          ('ok  ' if ok else 'FAIL', rule, equal, len(rows), agree, len(rows), worst, kl_mean,
           kl_worst, kl_at, dl_worst, dl_at))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
