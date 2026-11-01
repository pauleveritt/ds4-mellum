"""One real-use session: plain Pi drives a fresh ds4-server through the copied AgentClinic task.

    python3 tools/mellum_pi_run.py --model GGUF --out DIR [--port 8099] [--deadline 1800]
                                   [--server ./ds4-server] [--pi pi]

Starts ds4-server (-c 32768, thinking on) with its stderr in DIR/server.log, waits for it to listen, copies
tests/agentclinic_easy/base to DIR/workspace and runs Pi 1.1.0 there headless with the task's prompt, in a scratch
HOME and a Pi profile this runner writes (the README's settings).  At Pi's exit, the deadline, 300 s with no line
from the server while Pi is alive (stalled, e.g. on a foreground app server), or the server's death it
stops both process groups (Pi's children too), grades the project, and writes DIR/run.json.  Nothing is written
outside DIR; uv's package cache is the user's, so the task's dependencies are not downloaded again per run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import importlib.util
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'tests/agentclinic_easy'

_spec = importlib.util.spec_from_file_location('mellum_pi_grade', Path(__file__).with_name('mellum_pi_grade.py'))
_grade = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grade)
GradeResult = _grade.GradeResult
_spec = importlib.util.spec_from_file_location('mellum_server_requests',
                                               Path(__file__).with_name('mellum_server_requests.py'))
_requests = importlib.util.module_from_spec(_spec)
sys.modules.setdefault('mellum_server_requests', _requests)
_spec.loader.exec_module(_requests)

PI_FLAGS = ['-p', '--no-approve', '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-themes',
            '--no-context-files']


def profile(port: int) -> dict:
    return {'providers': {'ds4': {
        'baseUrl': f'http://127.0.0.1:{port}/v1', 'api': 'openai-completions', 'apiKey': 'ds4',
        'models': [{'id': 'mellum-2.1', 'contextWindow': 32768, 'maxTokens': 16384, 'reasoning': True,
                    'samplingParams': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20}}]}}}


def power_state() -> str:
    out = []
    for args in (['pmset', '-g', 'batt'], ['pmset', '-g'], ['pmset', '-g', 'therm']):
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        out.append(result.stdout.strip())
    return '\n'.join(out)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def _stop(proc: subprocess.Popen) -> None:
    """Stop a process group started with start_new_session: TERM, then KILL after 10 s."""
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _assistant_messages(session_dir: Path) -> int:
    count = 0
    for path in sorted(session_dir.glob('**/*.jsonl')):
        for line in path.read_text(errors='replace').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            count += row.get('type') == 'message' and (row.get('message') or {}).get('role') == 'assistant'
    return count


def run(model: Path, out: Path, port: int = 8099, deadline: float = 1800, server: str = './ds4-server',
        pi: str = 'pi', listen_timeout: float = 300, idle_timeout: float = 300, suite: Path = SUITE, grade_fn=None,
        power_fn=power_state) -> dict:
    model, out = Path(model).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'{out} is not empty')
    out.mkdir(parents=True, exist_ok=True)
    grade_fn = grade_fn or (lambda workspace: _grade.grade(workspace, suite))
    record = {'model': str(model), 'model_sha256': _sha256(model), 'port': port, 'deadline_s': deadline,
              'engine_commit': subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True,
                                              text=True, check=False).stdout.strip(),
              'pi_version': subprocess.run([pi, '--version'], capture_output=True, text=True,
                                           check=False).stdout.strip(),
              'power_before': power_fn(), 'started': datetime.now(timezone.utc).isoformat(),
              'compaction_requests': []}
    agent_dir, home, workspace, session = out / 'pi-agent', out / 'home', out / 'workspace', out / 'session'
    agent_dir.mkdir()
    home.mkdir()
    (agent_dir / 'models.json').write_text(json.dumps(profile(port), indent=2) + '\n')

    log = open(out / 'server.log', 'w')
    srv = subprocess.Popen([server, '-m', str(model), '-c', '32768', '--port', str(port)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        waited = time.monotonic()
        while 'listening on' not in (out / 'server.log').read_text(errors='replace'):
            if srv.poll() is not None or time.monotonic() - waited > listen_timeout:
                raise RuntimeError(f'ds4-server did not print "listening on" within {listen_timeout:g} s '
                                   f'(exit {srv.poll()}); see {out / "server.log"}')
            time.sleep(0.2)

        shutil.copytree(suite / 'base', workspace)
        prompt = (suite / 'prompts/01.md').read_text()
        env = {**os.environ, 'HOME': str(home), 'PI_CODING_AGENT_DIR': str(agent_dir),
               'UV_CACHE_DIR': os.environ.get('UV_CACHE_DIR', os.path.expanduser('~/.cache/uv'))}
        args = [pi, *PI_FLAGS, '--session-id', 'real-use', '--session-dir', str(session),
                '--provider', 'ds4', '--model', 'mellum-2.1', prompt]
        with open(out / 'pi.stdout', 'w') as so, open(out / 'pi.stderr', 'w') as se:
            agent = subprocess.Popen(args, cwd=workspace, env=env, stdout=so, stderr=se, start_new_session=True)
            started, deadline_hit, stalled = time.monotonic(), False, False
            log_size, log_grew = -1, time.monotonic()
            try:
                while agent.poll() is None:
                    if srv.poll() is not None:
                        break
                    now = time.monotonic()
                    if now - started > deadline:
                        deadline_hit = True
                        break
                    size = (out / 'server.log').stat().st_size
                    if size != log_size:
                        log_size, log_grew = size, now
                    elif now - log_grew > idle_timeout:
                        stalled = True   # Pi is alive and the model has done nothing (a foreground server, say)
                        break
                    time.sleep(0.5)
            finally:
                _stop(agent)   # also on Ctrl-C: Pi's own session never sees the terminal's SIGINT
        record.update(pi_exit=agent.returncode, deadline_hit=deadline_hit, stalled=stalled, server_exit=srv.poll())
    finally:
        _stop(srv)
        log.close()

    if record['deadline_hit'] or record['stalled'] or record['server_exit'] is not None:
        result = GradeResult(passed=False, checks_ran=False, missing=[], output_tail='not graded: run did not finish')
    else:
        result = grade_fn(workspace)
    (out / 'grade.txt').write_text(result.output_tail + '\n')
    with open(out / 'server.log', errors='replace') as f:
        server_requests = len(_requests.parse(f))
    pi_messages = _assistant_messages(session)
    record.update(passed=result.passed, checks_ran=result.checks_ran, missing=result.missing,
                  server_requests=server_requests, pi_assistant_messages=pi_messages,
                  count_mismatch=server_requests != pi_messages, power_after=power_fn(),
                  ended=datetime.now(timezone.utc).isoformat())
    (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--port', type=int, default=8099)
    p.add_argument('--deadline', type=float, default=1800)
    p.add_argument('--server', default=str(ROOT / 'ds4-server'))
    p.add_argument('--pi', default='pi')
    a = p.parse_args(argv[1:])
    record = run(a.model, a.out, port=a.port, deadline=a.deadline, server=a.server, pi=a.pi)
    print(json.dumps({k: record[k] for k in ('passed', 'deadline_hit', 'stalled', 'server_exit', 'server_requests',
                                             'pi_assistant_messages', 'count_mismatch')}))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
