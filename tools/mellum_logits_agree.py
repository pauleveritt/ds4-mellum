"""How far two traced Mellum runs' logits agree, for S6 phase B's preregistered comparison (item b).

    python3 -I tools/mellum_logits_agree.py A B POSITIONS [NEARTIE]

A and B are trace prefixes: the logits at position N are A.pN.logits.f32 (little-endian f32), as
tests/mellum_trace and tools/mellum_llama_logits write them.  B is the reference.  POSITIONS is a
comma-separated list.  For each position: KL(B || A) over the softmaxes, whether the top-1 tokens are equal,
and whether a different top-1 is a near-tie (B's top two log-probabilities within NEARTIE nats, default 0.75,
the oracle's rule in tests/mellum_logits_compare.py).  Prints one JSON object: the per-position rows, n, mean
and worst KL, top1_equal, agree (equal or near-tie) and agree_rate.  It judges nothing; the bounds are the
preregistration's.  Exit 0, or 1 on a missing, unequal-length or non-finite file.
"""

import json
import math
import struct
import sys


def read_row(prefix, pos):
    with open('%s.p%d.logits.f32' % (prefix, pos), 'rb') as f:
        data = f.read()
    if not data or len(data) % 4:
        raise ValueError('%s.p%d.logits.f32: empty or not f32' % (prefix, pos))
    row = struct.unpack('<%df' % (len(data) // 4), data)
    if not all(math.isfinite(v) for v in row):
        raise ValueError('%s.p%d.logits.f32: a logit is not finite' % (prefix, pos))
    return row


def log_softmax(row):
    top = max(row)
    log_z = top + math.log(sum(math.exp(v - top) for v in row))
    return [v - log_z for v in row]


def top_two(values):
    first = max(range(len(values)), key=lambda i: (values[i], -i))
    second = max((i for i in range(len(values)) if i != first), key=lambda i: (values[i], -i))
    return first, second


def compare(a_row, b_row, neartie):
    if len(a_row) != len(b_row):
        raise ValueError('rows of %d and %d logits' % (len(a_row), len(b_row)))
    la, lb = log_softmax(a_row), log_softmax(b_row)
    kl = sum(math.exp(q) * (q - p) for p, q in zip(la, lb))
    a1 = top_two(la)[0]
    b1, b2 = top_two(lb)
    equal = a1 == b1
    return {'kl': max(kl, 0.0), 'top1_equal': equal, 'near_tie': (not equal) and lb[b1] - lb[b2] < neartie,
            'top1_a': a1, 'top1_b': b1}


def main(argv):
    if len(argv) not in (4, 5):
        sys.stderr.write(__doc__)
        return 2
    a, b, positions = argv[1], argv[2], [int(p) for p in argv[3].split(',') if p]
    neartie = float(argv[4]) if len(argv) == 5 else 0.75
    rows = []
    try:
        for pos in positions:
            row = compare(read_row(a, pos), read_row(b, pos), neartie)
            row['position'] = pos
            rows.append(row)
    except (OSError, ValueError) as exc:
        sys.stderr.write('mellum_logits_agree: %s\n' % exc)
        return 1
    n = len(rows)
    worst = max(rows, key=lambda r: r['kl']) if rows else None
    agree = sum(r['top1_equal'] or r['near_tie'] for r in rows)
    print(json.dumps({
        'a': a, 'b': b, 'neartie': neartie, 'n': n, 'positions': rows,
        'mean_kl': sum(r['kl'] for r in rows) / n if n else None,
        'worst_kl': worst['kl'] if worst else None, 'worst_position': worst['position'] if worst else None,
        'top1_equal': sum(r['top1_equal'] for r in rows), 'agree': agree, 'agree_rate': agree / n if n else None,
    }, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
