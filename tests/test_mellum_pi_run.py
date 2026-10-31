"""CPU tests of the real-use runner (tools/mellum_pi_run.py) with a fake ds4-server and a fake pi: no model, GPU,
network or Pi.  The fakes are small Python scripts the test writes; FAKE_* variables choose their behaviour."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]

FAKE_SERVER = r'''#!/usr/bin/env python3
import os, sys, time
mode = os.environ.get('FAKE_SERVER', 'ok')
if mode != 'silent':
    print('1031 12:00:00 ds4-server: listening on http://127.0.0.1:8099', file=sys.stderr, flush=True)
if mode == 'die':
    time.sleep(0.5)
    sys.exit(3)
if mode == 'ok':
    for ctx in ('0..3000:3000', '3000..3400:400'):
        print(f'1031 12:00:01 ds4-server: chat ctx={ctx} TOOLS prompt done 1.000s', file=sys.stderr)
        print(f'1031 12:00:02 ds4-server: chat ctx={ctx} gen=50 TOOLS finish=stop 2.000s', file=sys.stderr, flush=True)
while True:
    time.sleep(1)
'''

FAKE_PI = r'''#!/usr/bin/env python3
import json, os, subprocess, sys, time
from pathlib import Path
if sys.argv[1:] == ['--version']:
    print('1.1.0'); sys.exit(0)
Path('fake-pi.json').write_text(json.dumps({'argv': sys.argv[1:], 'HOME': os.environ.get('HOME'),
                                            'PI_CODING_AGENT_DIR': os.environ.get('PI_CODING_AGENT_DIR')}))
session = Path(sys.argv[sys.argv.index('--session-dir') + 1])
session.mkdir(parents=True, exist_ok=True)
rows = [{'type': 'session'}] + [{'type': 'message', 'message': {'role': r}}
                                for r in ['user'] + ['assistant'] * int(os.environ.get('FAKE_PI_ASSISTANTS', '2'))]
(session / 's.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
if os.environ.get('FAKE_PI') == 'hang':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    Path('fake-pi-child').write_text(str(child.pid))
    time.sleep(600)
time.sleep(float(os.environ.get('FAKE_PI_SLEEP', '0')))
'''


def load():
    path = ROOT / 'tools/mellum_pi_run.py'
    assert path.exists(), 'tools/mellum_pi_run.py missing'
    spec = importlib.util.spec_from_file_location('mellum_pi_run', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['mellum_pi_run'] = m
    spec.loader.exec_module(m)
    return m


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='mellum-pi-run-test-'))
        for name, body in (('ds4-server', FAKE_SERVER), ('pi', FAKE_PI)):
            p = self.tmp / name
            p.write_text(body)
            p.chmod(0o755)
        self.model = self.tmp / 'model.gguf'
        self.model.write_bytes(b'GGUF fake')
        self.out = self.tmp / 'run'
        self.env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith('FAKE_')}

    def tearDown(self):
        for k in [k for k in os.environ if k.startswith('FAKE_')]:
            del os.environ[k]
        os.environ.update(self.env)

    def run_once(self, **kw):
        m = load()
        grade = m.GradeResult(passed=True, checks_ran=True, missing=[], output_tail='4 passed')
        args = dict(model=self.model, out=self.out, server=str(self.tmp / 'ds4-server'), pi=str(self.tmp / 'pi'),
                    grade_fn=lambda workspace: grade, power_fn=lambda: 'AC Power; powermode 2', listen_timeout=5)
        args.update(kw)
        return m.run(**args)

    def test_profile_has_the_readme_settings(self):
        self.run_once(port=8123)
        models = json.loads((self.out / 'pi-agent/models.json').read_text())
        provider = models['providers']['ds4']
        self.assertEqual((provider['baseUrl'], provider['api']), ('http://127.0.0.1:8123/v1', 'openai-completions'))
        model = provider['models'][0]
        self.assertEqual((model['id'], model['contextWindow'], model['maxTokens'], model['reasoning']),
                         ('mellum-2.1', 32768, 16384, True))
        self.assertEqual(model['samplingParams'], {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20})

    def test_scratch_home_and_profile_dir(self):
        run = self.run_once()
        seen = json.loads((self.out / 'workspace/fake-pi.json').read_text())
        self.assertEqual(Path(seen['HOME']).parent, self.out.resolve())
        self.assertEqual(Path(seen['PI_CODING_AGENT_DIR']), self.out.resolve() / 'pi-agent')
        self.assertNotEqual(seen['HOME'], os.path.expanduser('~'))
        for flag in ('-p', '--no-approve', '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-themes',
                     '--no-context-files'):
            self.assertIn(flag, seen['argv'])
        self.assertEqual(seen['argv'][seen['argv'].index('--model') + 1], 'mellum-2.1')
        self.assertEqual(seen['argv'][-1].strip(), 'Implement Phase 1 — Home Page.')
        self.assertEqual((run['server_requests'], run['pi_assistant_messages'], run['count_mismatch']), (2, 2, False))
        self.assertEqual(run['compaction_requests'], [])
        self.assertTrue(run['passed'])

    def test_server_death_fails_the_run(self):
        os.environ['FAKE_SERVER'] = 'die'
        os.environ['FAKE_PI_SLEEP'] = '30'
        started = time.monotonic()
        run = self.run_once(deadline=60)
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(run['server_exit'], 3)
        self.assertFalse(run['passed'])
        self.assertFalse(run['deadline_hit'])
        self.assertTrue((self.out / 'server.log').read_text().count('listening'))

    def test_server_that_never_listens_is_refused(self):
        m = load()
        os.environ['FAKE_SERVER'] = 'silent'
        with self.assertRaisesRegex(RuntimeError, 'listening'):
            self.run_once(listen_timeout=1)
        self.assertFalse((self.out / 'workspace/fake-pi.json').exists())

    def test_deadline_kills_the_process_group(self):
        os.environ['FAKE_PI'] = 'hang'
        run = self.run_once(deadline=3)
        self.assertTrue(run['deadline_hit'])
        self.assertFalse(run['passed'])
        child = int((self.out / 'workspace/fake-pi-child').read_text())
        time.sleep(0.5)
        self.assertFalse(alive(child))

    def test_count_mismatch_is_flagged(self):
        os.environ['FAKE_PI_ASSISTANTS'] = '3'
        run = self.run_once()
        self.assertEqual((run['server_requests'], run['pi_assistant_messages'], run['count_mismatch']), (2, 3, True))

    def test_refuses_a_non_empty_out(self):
        self.out.mkdir()
        (self.out / 'x').write_text('x')
        with self.assertRaisesRegex(RuntimeError, 'not empty'):
            self.run_once()


if __name__ == '__main__':
    unittest.main()
