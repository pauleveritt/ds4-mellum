"""S4 measurement runner (docs/superpowers/plans/2026-10-28-mellum-s4-cached-execution.md, Task 8).

Builds the frozen synthetic prompt, makes fresh uncached model copies for streamed runs, classifies device
reads, and runs a preregistered list of steps once each into an exclusive root, keeping every raw output.
The arms themselves are commands: the ds4 CLI for the resident A/B regression, tools/mellum_s4_measure for
the streamed and resident detail runs.  Stdlib only; never retries."""
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

EXPERT_BYTES = 6_580_224
READ_CAP = int(400e9)
RATE = re.compile(r'Mellum prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s')
F_NOCACHE = getattr(fcntl, 'F_NOCACHE', 48)


class Busy(RuntimeError):
    """The shared model lock is held: postpone, nothing was created."""


def build_prompt(root: Path, budget_bytes: int) -> dict:
    """Sorted tracked *.c and *.md files, whole, as path-delimited blocks, until the next would pass the
    budget.  Research notes and anything under ac-demo stay out (held-out evaluation data)."""
    listed = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], check=True, capture_output=True).stdout
    names = sorted(n for n in listed.decode().split('\0') if n.endswith(('.c', '.md')) and
                   not n.startswith('docs/superpowers/research/') and 'ac-demo' not in n.split('/'))
    head = 'Review the following source files from this repository and describe how they fit together.\n\n'
    text, files = head, []
    for name in names:
        block = f'--- {name} ---\n{(root / name).read_text()}\n--- end {name} ---\n'
        if len((text + block).encode()) > budget_bytes:
            continue
        text += block
        files.append(name)
    return dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest(), files=files)


def fresh_copy(src: Path, dest: Path, block: int = 64 << 20) -> dict:
    """An exclusive copy written through F_NOCACHE descriptors and fsynced, and never read back: reading it
    would leave it in the unified buffer cache (cached-reads spike).  verify_copy checks it after the run."""
    out = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        fcntl.fcntl(out, F_NOCACHE, 1)
        with open(src, 'rb', buffering=0) as f:
            fcntl.fcntl(f.fileno(), F_NOCACHE, 1)
            while chunk := f.read(block):
                view = memoryview(chunk)
                while view:
                    view = view[os.write(out, view):]
        os.fsync(out)
    finally:
        os.close(out)
    if os.stat(dest).st_size != src.stat().st_size:
        raise ValueError('copy size differs')
    return dict(path=str(dest), bytes=src.stat().st_size)


def verify_copy(src: Path, dest: Path, block: int = 64 << 20) -> dict:
    """Byte comparison of a copy with its source; run it after any timed read of the copy."""
    with open(src, 'rb') as a, open(dest, 'rb') as b:
        while True:
            x, y = a.read(block), b.read(block)
            if x != y:
                raise ValueError('copy bytes differ')
            if not x:
                break
    return dict(path=str(dest), bytes=src.stat().st_size)


def classify_reads(logical: int, device: int) -> str:
    """logical: expert bytes the run read into slots (loads x EXPERT_BYTES); device: block-driver read bytes
    over the run (static weights only add to it)."""
    if logical > READ_CAP:
        return 'over-read-cap'
    return 'device-observed' if device >= 0.99 * logical else 'cache-contaminated'


def regression_order(n: int) -> list[str]:
    """A/B/B/A blocks, n processes per arm."""
    if n % 2:
        raise ValueError('n must be even')
    return list('ABBA' * (n // 2))


def parse_cli_rates(stderr: str) -> tuple[float, float]:
    m = RATE.search(stderr)
    if not m:
        raise ValueError('no Mellum rate line')
    return float(m[1]), float(m[2])


def regression_verdict(runs: list[dict]) -> dict:
    """Goal 2: B's median prefill and generation t/s each within 2% of A's."""
    med = {arm: {k: statistics.median(r[k] for r in runs if r['arm'] == arm) for k in ('prefill', 'gen')} for arm in 'AB'}
    ratio = {k: med['B'][k] / med['A'][k] for k in ('prefill', 'gen')}
    return {'medians': med, 'ratio': ratio, 'pass': all(r >= 0.98 for r in ratio.values())}


def _serial(result: dict) -> dict:
    out = {}
    for k, v in result.items():
        out[k] = dict(encoding='base64', data=base64.b64encode(v).decode()) if isinstance(v, bytes) else v
    return out


def run_all(root: Path, plan: dict, runner, lock_held) -> dict:
    """Runs plan['steps'] (name, argv, optional timeout) once each, in order, into an exclusive root.  A held
    lock postpones before anything is created; a failed step is recorded and the next runs; nothing retries."""
    if lock_held():
        raise Busy('shared model lock held')
    root.mkdir(mode=0o700)
    raw = root / 'raw'
    raw.mkdir(mode=0o700)
    steps = []
    for step in plan.get('steps', []):
        result = runner(step['argv'], timeout=step.get('timeout', 3600))
        record = dict(name=step['name'], argv=step['argv'], **_serial(result))
        with open(raw / f"{step['name']}.json", 'x') as f:
            json.dump(record, f)
        steps.append(dict(name=step['name'], returncode=result['returncode']))
    summary = dict(steps=steps)
    with open(root / 'summary.json', 'x') as f:
        json.dump(summary, f)
    return summary
