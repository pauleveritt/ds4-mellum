"""Real-use rates from one or more run directories of tools/mellum_pi_run.py, as markdown.

    python3 tools/mellum_pi_report.py RUN_DIR...

Prints each run's requests (context at the start, new prompt tokens, prefill and decode t/s, finish) with the run's
pass/fail, then one table over all runs: the median prefill and decode rates by context band, each with its n.  A
request belongs to the band of its context at the start (the tokens it reused).  Prefill medians leave out requests
of fewer than 256 new prompt tokens, which fixed costs dominate.  Request 0 of a run is a cold prefill on a fresh
server and is marked `cold`; requests Pi sent to compact the session are marked `compaction`.
"""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import statistics
import sys

_spec = importlib.util.spec_from_file_location('mellum_server_requests',
                                               Path(__file__).with_name('mellum_server_requests.py'))
_requests = importlib.util.module_from_spec(_spec)
sys.modules.setdefault('mellum_server_requests', _requests)
_spec.loader.exec_module(_requests)
Request = _requests.Request
parse = _requests.parse

EDGES = (4096, 8192, 16384, 32769)
MIN_NEW = 256


@dataclass(frozen=True)
class Band:
    low: int
    high: int
    prefill_median: float | None
    prefill_n: int
    decode_median: float | None
    decode_n: int
    runs: int


def bands(runs: list[list[Request]], edges: tuple[int, ...] = EDGES, min_new: int = MIN_NEW) -> list[Band]:
    result = []
    for low, high in zip((0,) + edges[:-1], edges):
        prefill, decode, contributing = [], [], 0
        for requests in runs:
            inside = [r for r in requests if low <= r.cached < high]
            contributing += bool(inside)
            prefill += [r.prefill_tps for r in inside if r.new >= min_new and r.prefill_tps is not None]
            decode += [r.decode_tps for r in inside if r.decode_tps is not None]
        result.append(Band(low=low, high=high,
                           prefill_median=statistics.median(prefill) if prefill else None, prefill_n=len(prefill),
                           decode_median=statistics.median(decode) if decode else None, decode_n=len(decode),
                           runs=contributing))
    return result


def marks(index: int, compaction_requests: list[int]) -> list[str]:
    found = ['cold'] if index == 0 else []
    return found + (['compaction'] if index in compaction_requests else [])


def _fmt(value: float | None) -> str:
    return '-' if value is None else f'{value:,.1f}'


def run_table(run_dir: Path, run: dict, requests: list[Request]) -> str:
    verdict = 'passed' if run.get('passed') else 'failed'
    if run.get('deadline_hit'):
        verdict += ' (deadline)'
    if run.get('stalled'):
        verdict += ' (stalled)'
    lines = [f'### {run_dir.name}: {verdict}, {len(requests)} requests', '',
             '| # | context at start | new tokens | prefill t/s | generated | decode t/s | finish | marks |',
             '|---|---|---|---|---|---|---|---|']
    for r in requests:
        lines.append(f'| {r.index} | {r.cached:,} | {r.new:,} | {_fmt(r.prefill_tps)} | {r.generated:,} | '
                     f'{_fmt(r.decode_tps)} | {r.finish} | {" ".join(marks(r.index, run.get("compaction_requests", [])))} |')
    return '\n'.join(lines)


def band_table(found: list[Band]) -> str:
    lines = ['| context at start | prefill t/s (median) | n | decode t/s (median) | n | runs |',
             '|---|---|---|---|---|---|']
    for b in found:
        lines.append(f'| {b.low:,}-{b.high - 1:,} | {_fmt(b.prefill_median)} | {b.prefill_n} | '
                     f'{_fmt(b.decode_median)} | {b.decode_n} | {b.runs} |')
    return '\n'.join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    all_runs = []
    for arg in argv[1:]:
        run_dir = Path(arg)
        run = json.loads((run_dir / 'run.json').read_text())
        with open(run_dir / 'server.log', errors='replace') as f:
            requests = parse(f)
        all_runs.append(requests)
        print(run_table(run_dir, run, requests), end='\n\n')
    print(f'### All runs (n = {len(all_runs)})\n')
    print(band_table(bands(all_runs)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
