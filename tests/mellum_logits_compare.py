"""Compare two traced Mellum runs' logits, for tests/mellum_oracle_compare.sh.

    python3 -I tests/mellum_logits_compare.py A B POSITIONS EXACT TOL KLTOL

A and B are trace prefixes: the logits at position N are A.pN.logits.f32 (little-endian f32) and the
session's last-row logits are A.session.logits.f32.  POSITIONS is a comma-separated list.  EXACT 1:
every file must be byte-equal; 0: the same top-1 everywhere, max|d logit| <= TOL and mean
KL(B || A) <= KLTOL.  Under either rule every logit of both runs must be finite first.  Prints one
summary line; exit 0 when the rule holds, 1 otherwise.
"""

import math
import struct
import sys


def softmax(v):
    m = max(v)
    e = [math.exp(t - m) for t in v]
    s = sum(e)
    return [t / s for t in e]


def main(argv):
    a, b, positions, exact = argv[1], argv[2], argv[3].split(','), argv[4] == '1'
    tol, kltol = float(argv[5]), float(argv[6])
    files = ['p%s.logits' % p for p in positions] + ['session.logits']
    rows = []
    for f in files:
        x = open('%s.%s.f32' % (a, f), 'rb').read()
        y = open('%s.%s.f32' % (b, f), 'rb').read()
        u = struct.unpack('<%df' % (len(x) // 4), x)
        v = struct.unpack('<%df' % (len(y) // 4), y)
        for prefix, row in ((a, u), (b, v)):
            bad = sum(not math.isfinite(t) for t in row)
            if bad:
                print('FAIL %s.%s.f32: %d of %d logits are not finite' % (prefix, f, bad, len(row)))
                return 1
        rows.append((x, y, u, v))
    worst, agree, equal, kl = 0.0, 0, 0, 0.0
    for x, y, u, v in rows:
        equal += x == y
        n = len(u)
        worst = max(worst, max(abs(i - j) for i, j in zip(u, v)))
        agree += max(range(n), key=u.__getitem__) == max(range(n), key=v.__getitem__)
        if x != y:
            p, q = softmax(v), softmax(u)
            kl += sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q) if pi > 1e-12)
    kl /= len(files)
    ok = equal == len(files) if exact else (agree == len(files) and worst <= tol and kl <= kltol)
    rule = 'byte-equal' if exact else 'top-1 equal, max|d| <= %g, mean KL <= %g' % (tol, kltol)
    print('%s (%s): %d of %d logits files byte-equal, top-1 agree %d/%d, max|d logit| %.3e, mean KL %.2e' %
          ('ok  ' if ok else 'FAIL', rule, equal, len(files), agree, len(files), worst, kl))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
