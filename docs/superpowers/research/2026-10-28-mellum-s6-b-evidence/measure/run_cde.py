"""S6 phase B, preregistered items (c), (d) and (e), run as the preregistration fixes them
(docs/superpowers/research/2026-10-28-mellum-s6-b-preregistration.md).  From the repo root:

    Q8=... MX=... ARM_A=<a 1500320 checkout with ds4 and tests/mellum_trace built> \\
        python3 -I docs/superpowers/research/2026-10-28-mellum-s6-b-evidence/measure/run_cde.py [c] [d] [e]

Each process runs once, in the preregistered order, with 30 s rests, and with its own checkout as the working
directory (the engine compiles its Metal sources from there).  Every process's stdout, stderr and argv are kept
under raw/ (git-ignored); summary-<item>.json holds the figures and verdicts computed from them.
"""

import filecmp
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'tools'))
import mellum_s4_measure as M  # noqa: E402

E = ROOT / 'docs/superpowers/research/2026-10-28-mellum-s6-b-evidence/measure'
RAW = E / 'raw'
Q8, MX, ARM_A = os.environ['Q8'], os.environ['MX'], Path(os.environ['ARM_A'])
LONG = ROOT / 'docs/superpowers/research/2026-10-28-mellum-s4-measurement-prereg/prompt.txt'
SHORT = ['-c', '8192', '--temp', '0', '-n', '256', '-p', 'Explain how a hash map works, with an example in C.']
REST = 30


def log(msg):
    line = '%s %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(E / 'progress.log', 'a') as f:
        f.write(line + '\n')


def run(name, argv, cwd, env=None, timeout=4 * 3600):
    t = time.time()
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                       env=dict(os.environ, **(env or {})))
    (RAW / f'{name}.out').write_text(r.stdout)
    (RAW / f'{name}.err').write_text(r.stderr)
    (RAW / f'{name}.json').write_text(json.dumps(dict(name=name, argv=argv, cwd=str(cwd), env=env or {},
                                                      returncode=r.returncode, seconds=round(time.time() - t, 1))))
    log('%s rc=%d %.0fs' % (name, r.returncode, time.time() - t))
    return r


def cli(name, binary_root, model):
    r = run(name, [str(binary_root / 'ds4'), '-m', model, *SHORT], cwd=binary_root)
    rates = M.parse_cli_rates(r.stderr) if r.returncode == 0 else (None, None)
    time.sleep(REST)
    return dict(name=name, rc=r.returncode, prefill=rates[0], gen=rates[1], text=r.stdout)


def positions(tokens):
    n = len([w for line in open(tokens) for w in line.split('#')[0].split()])
    k = min(64, n)
    return ','.join(str(round(i * (n - 1) / (k - 1))) for i in range(k))


def measure(name, model, stream_bytes=None):
    out = RAW / f'{name}.measure.json'
    argv = [str(ROOT / 'tools/mellum_s4_measure'), '--model', model, '--prompt', str(LONG), '--gen', '2048',
            '--context', '32768', '--out', str(out)]
    if stream_bytes:
        argv += ['--stream-bytes', str(stream_bytes)]
    r = run(name, argv, cwd=ROOT)
    time.sleep(REST)
    return json.loads(out.read_text()) if r.returncode == 0 and out.exists() else None


def long_figures(m):
    ns = [t['ns'] for t in m['tokens']]
    loads = [t['loads'] for t in m['tokens']]
    return dict(prompt_tokens=m['prompt_tokens'], ttft_s=m['ttft_ns'] / 1e9,
                prefill_tps=m['prompt_tokens'] / (m['ttft_ns'] / 1e9),
                decode_tps_first256=256 / (sum(ns[:256]) / 1e9), decode_tps_last256=256 / (sum(ns[-256:]) / 1e9),
                decode_loads_per_token=(loads[-1] - loads[0]) / (len(loads) - 1), stops=m['stops'], loads=m['loads'],
                slots=m['slots'], phys_footprint_peak=m['phys_footprint_peak'],
                metal_allocated_peak=m['metal_allocated_peak'])


def collapsed(ids, span=512, max_period=64):
    """A span of at least `span` tokens made of one block of period <= max_period repeated back to back."""
    for p in range(1, max_period + 1):
        run_len = 0
        for i in range(p, len(ids)):
            run_len = run_len + 1 if ids[i] == ids[i - p] else 0
            if run_len + p >= span:
                return dict(period=p, ends_at=i)
    return None


def item_c():
    runs = [cli(f'c-{i:02d}-{arm}', ARM_A if arm == 'A' else ROOT, Q8)
            for i, arm in enumerate(M.regression_order(6))]
    for r, arm in zip(runs, M.regression_order(6)):
        r['arm'] = arm
    ok = [r for r in runs if r['rc'] == 0 and r['prefill'] is not None]
    verdict = M.regression_verdict(ok) if len(ok) == len(runs) else None
    same_text = len({r['text'] for r in runs}) == 1
    traces = {}
    for t in ('short', '1100', '2200'):
        tok = ROOT / f'tests/mellum_oracle_{t}.tokens'
        pos = positions(tok)
        for arm, root in (('A', ARM_A), ('B', ROOT)):
            run(f'c-trace-{arm}-{t}', [str(root / 'tests/mellum_trace'), Q8, str(tok), 'prefill'], cwd=root,
                env={'DS4_MELLUM_TRACE': str(RAW / f'c-trace-{arm}.{t}'), 'DS4_MELLUM_TRACE_AT': pos})
            time.sleep(REST)
        names = [f'p{p}' for p in pos.split(',')] + ['session']
        differ = [n for n in names if not filecmp.cmp(RAW / f'c-trace-A.{t}.{n}.logits.f32',
                                                       RAW / f'c-trace-B.{t}.{n}.logits.f32', shallow=False)]
        traces[t] = dict(files=len(names), differ=differ)
    summary = dict(runs=[{k: v for k, v in r.items() if k != 'text'} for r in runs], verdict=verdict,
                   generated_text_identical=same_text, traced_logits=traces,
                   pass_=bool(verdict and verdict['pass'] and same_text and all(not v['differ'] for v in traces.values())))
    (E / 'summary-c.json').write_text(json.dumps(summary, indent=1))
    log('c: pass=%s' % summary['pass_'])


def item_d():
    long_runs = []
    for i, arm in enumerate(['Q8', 'MX', 'MX', 'Q8', 'Q8', 'MX']):
        m = measure(f'd-long-{i}-{arm}', Q8 if arm == 'Q8' else MX)
        long_runs.append(dict(arm=arm, figures=long_figures(m) if m else None))
    short_runs = [cli(f'd-short-{i:02d}-{arm}', ROOT, Q8 if arm == 'A' else MX)
                  for i, arm in enumerate(M.regression_order(6))]
    for r, arm in zip(short_runs, M.regression_order(6)):
        r['arm'] = 'Q8' if arm == 'A' else 'MX'

    def med(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return dict(n=len(vals), median=statistics.median(vals) if vals else None,
                    min=min(vals) if vals else None, max=max(vals) if vals else None)

    keys = ('ttft_s', 'prefill_tps', 'decode_tps_first256', 'decode_tps_last256', 'phys_footprint_peak')
    summary = dict(
        long={arm: {k: med([r['figures'] for r in long_runs if r['arm'] == arm and r['figures']], k) for k in keys}
              for arm in ('Q8', 'MX')},
        short={arm: {k: med([r for r in short_runs if r['arm'] == arm and r['rc'] == 0], k) for k in ('prefill', 'gen')}
               for arm in ('Q8', 'MX')},
        long_runs=long_runs, short_runs=[{k: v for k, v in r.items() if k != 'text'} for r in short_runs],
        energy='not measured: powermetrics needs root')
    (E / 'summary-d.json').write_text(json.dumps(summary, indent=1))
    log('d: done')


def item_e():
    resident = measure('e-resident', MX)
    ref = [t['id'] for t in resident['tokens']] if resident else None
    runs = []
    for name, b in (('e-stream-2.5g-1', 2684354560), ('e-stream-2.5g-2', 2684354560), ('e-stream-3g-1', 3221225472)):
        m = measure(name, MX, stream_bytes=b)
        ids = [t['id'] for t in m['tokens']] if m else None
        first = next((i for i, (a, b2) in enumerate(zip(ids or [], ref or [])) if a != b2), None)
        runs.append(dict(name=name, stream_bytes=b, ok=m is not None, ids_equal=ids is not None and ids == ref,
                         first_difference=first, figures=long_figures(m) if m else None))
    summary = dict(resident=long_figures(resident) if resident else None, collapse=collapsed(ref) if ref else 'no run',
                   streamed=runs, pass_=bool(ref) and all(r['ids_equal'] for r in runs))
    (E / 'summary-e.json').write_text(json.dumps(summary, indent=1))
    log('e: pass=%s collapse=%s' % (summary['pass_'], summary['collapse']))


if __name__ == '__main__':
    RAW.mkdir(parents=True, exist_ok=True)
    for item in sys.argv[1:] or ['c', 'd', 'e']:
        log('start %s' % item)
        {'c': item_c, 'd': item_d, 'e': item_e}[item]()
