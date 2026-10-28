"""CPU tests of the S4 measurement runner (tools/mellum_s4_measure.py) with fake commands: no model, GPU or
device access."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / 'tools/mellum_s4_measure.py'
    assert path.exists(), 'tools/mellum_s4_measure.py missing'
    spec = importlib.util.spec_from_file_location('mellum_s4_measure', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def repo(root: Path) -> Path:
    for name, body in (('a.c', 'int a;\n'), ('b.md', '# b\n' * 50), ('docs/superpowers/research/r.md', 'secret\n'),
                       ('ac-demo/x.md', 'eval\n'), ('c.h', 'skip\n'), ('z.c', 'int z;\n' * 40)):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    return root


class Prompt(unittest.TestCase):
    def test_deterministic_and_excludes_research_and_ac_demo(self):
        m = load()
        with tempfile.TemporaryDirectory() as d:
            r = repo(Path(d))
            a, b = m.build_prompt(r, 10_000), m.build_prompt(r, 10_000)
            self.assertEqual(a, b)
            self.assertEqual(a['files'], ['a.c', 'b.md', 'z.c'])
            self.assertNotIn('secret', a['text'])
            self.assertNotIn('eval', a['text'])
            self.assertEqual(a['sha256'], m.hashlib.sha256(a['text'].encode()).hexdigest())

    def test_budget_never_splits_a_file(self):
        m = load()
        with tempfile.TemporaryDirectory() as d:
            r = repo(Path(d))
            small = m.build_prompt(r, 300)
            self.assertLessEqual(len(small['text'].encode()), 300)
            for name in small['files']:
                self.assertIn((r / name).read_text(), small['text'])


class Copy(unittest.TestCase):
    def test_fresh_copy_round_trip_and_exclusive(self):
        m = load()
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / 'm.gguf', Path(d) / 'copy.gguf'
            src.write_bytes(bytes(range(256)) * 5000)
            info = m.fresh_copy(src, dst, block=4096)
            self.assertEqual(dst.read_bytes(), src.read_bytes())
            self.assertEqual(info['bytes'], src.stat().st_size)
            with self.assertRaises(FileExistsError):
                m.fresh_copy(src, dst)


class Reads(unittest.TestCase):
    def test_classification_threshold(self):
        m = load()
        self.assertEqual(m.classify_reads(1000, 990), 'device-observed')
        self.assertEqual(m.classify_reads(1000, 985), 'cache-contaminated')
        self.assertEqual(m.classify_reads(int(400e9) + 1, int(400e9) + 1), 'over-read-cap')


class Regression(unittest.TestCase):
    def test_abba_order(self):
        m = load()
        self.assertEqual(m.regression_order(6), list('ABBAABBAABBA'))

    def test_two_percent_rule_on_medians(self):
        m = load()
        runs = [dict(arm='A', prefill=100.0, gen=50.0)] * 6 + [dict(arm='B', prefill=98.5, gen=49.2)] * 6
        v = m.regression_verdict(runs)
        self.assertTrue(v['pass'])
        slow = runs[:6] + [dict(arm='B', prefill=97.0, gen=50.0)] * 6
        self.assertFalse(m.regression_verdict(slow)['pass'])

    def test_cli_line_parse(self):
        m = load()
        self.assertEqual(m.parse_cli_rates('x\nds4: Mellum prefill: 504.11 t/s, generation: 138.36 t/s\n'), (504.11, 138.36))
        with self.assertRaises(ValueError):
            m.parse_cli_rates('no rates')


class Runner(unittest.TestCase):
    def test_lock_held_postpones_without_root(self):
        m = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'run'
            with self.assertRaises(m.Busy):
                m.run_all(root, plan={}, runner=lambda argv, **k: dict(returncode=0, stdout=b'', stderr=b''),
                          lock_held=lambda: True)
            self.assertFalse(root.exists())

    def test_every_step_runs_once_and_raw_is_kept_even_on_failure(self):
        m = load()
        calls = []

        def runner(argv, **k):
            calls.append(argv)
            return dict(returncode=1 if 'fail' in argv else 0, stdout=b'out', stderr=b'err')

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'run'
            plan = dict(steps=[dict(name='one', argv=['x']), dict(name='two', argv=['fail']), dict(name='three', argv=['y'])])
            result = m.run_all(root, plan=plan, runner=runner, lock_held=lambda: False)
            self.assertEqual([c for c in calls], [['x'], ['fail'], ['y']])
            self.assertEqual([s['returncode'] for s in result['steps']], [0, 1, 0])
            for name in ('one', 'two', 'three'):
                self.assertTrue((root / 'raw' / f'{name}.json').exists())
            with self.assertRaises(FileExistsError):
                m.run_all(root, plan=plan, runner=runner, lock_held=lambda: False)


if __name__ == '__main__':
    unittest.main()
