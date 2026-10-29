"""Live smoke for tools/mellum_s4_measure (DS4_TEST_MELLUM_MODEL): resident and streamed runs of a short chat
prompt report every field the S4 measurement reads, and greedy decode gives the same token ids; the append
phase extends the warm session.  Skipped without the model."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'tools' / 'mellum_s4_measure'
FIELDS = {'streaming', 'slots', 'context', 'prompt_tokens', 'ttft_ns', 'tokens', 'stops', 'loads', 'phys_footprint_peak',
          'rss_peak', 'metal_allocated_peak'}
APPEND_FIELDS = {'append_tokens', 'append_ttft_ns', 'append_pos_before', 'append_pos_after', 'append_common_prefix',
                 'append_stops', 'append_loads', 'append_decode'}
PROMPT = 'Explain how a hash map works, with an example in C.'
TOOL = 'def add(a, b):\n    return a + b\n' * 20


def command(model, prompt, out, *extra, context=8192):
    return [str(DRIVER), '--model', model, '--prompt', str(prompt), '--gen', '8', '--context', str(context),
            '--out', str(out), *extra]


def run(model, prompt, out, *extra):
    subprocess.run(command(model, prompt, out, *extra), check=True, cwd=ROOT, timeout=300, capture_output=True)
    return json.loads(out.read_text())


class DriverAppend(unittest.TestCase):
    """--append FILE --append-gen N: after decode, one tool result and the assistant prefix are synced onto the
    warm session, timed, then N more greedy tokens (S5 measurement 01's per-turn case)."""

    def setUp(self):
        self.model = os.environ.get('DS4_TEST_MELLUM_MODEL')
        if not self.model:
            self.skipTest('DS4_TEST_MELLUM_MODEL unset')
        self.assertTrue(DRIVER.exists(), 'tools/mellum_s4_measure is not built')

    def test_append_extends_the_warm_session(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            prompt, tool = d / 'prompt.txt', d / 'tool.txt'
            prompt.write_text(PROMPT)
            tool.write_text(TOOL)
            a = run(self.model, prompt, d / 'a.json', '--append', str(tool), '--append-gen', '4')
            b = run(self.model, prompt, d / 'b.json', '--append', str(tool), '--append-gen', '4', '--stream-experts', '64')
        for r in (a, b):
            self.assertTrue(APPEND_FIELDS <= set(r), sorted(APPEND_FIELDS - set(r)))
            # the sync starts at the warm frontier, whose whole checkpoint prefixes the turn, so only the suffix runs
            self.assertEqual(r['append_pos_before'], r['prompt_tokens'] + 8)
            self.assertEqual(r['append_common_prefix'], r['append_pos_before'])
            self.assertGreater(r['append_tokens'], 0)
            self.assertEqual(r['append_pos_after'], r['append_pos_before'] + r['append_tokens'])
            self.assertGreater(r['append_ttft_ns'], 0)
            self.assertEqual(len(r['append_decode']), 4)
            self.assertTrue(all(t['ns'] > 0 for t in r['append_decode']))
        self.assertEqual(a['append_tokens'], b['append_tokens'])
        self.assertEqual([t['id'] for t in a['append_decode']], [t['id'] for t in b['append_decode']])
        self.assertEqual(a['append_loads'], 0)
        self.assertGreater(b['append_loads'], 0)

    def test_append_refuses_a_turn_past_the_context(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            prompt, tool, out = d / 'prompt.txt', d / 'tool.txt', d / 'out.json'
            prompt.write_text(PROMPT)
            tool.write_text(TOOL * 8)
            p = subprocess.run(command(self.model, prompt, out, '--append', str(tool), '--append-gen', '4', context=1024),
                               cwd=ROOT, timeout=300, capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('append', p.stderr)
            self.assertIn('exceed', p.stderr)
            self.assertFalse(out.exists())


class DriverSmoke(unittest.TestCase):
    def test_resident_and_streamed_reports_agree(self):
        model = os.environ.get('DS4_TEST_MELLUM_MODEL')
        if not model:
            self.skipTest('DS4_TEST_MELLUM_MODEL unset')
        self.assertTrue(DRIVER.exists(), 'tools/mellum_s4_measure is not built')
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            prompt = d / 'prompt.txt'
            prompt.write_text(PROMPT)
            a = run(model, prompt, d / 'a.json')
            b = run(model, prompt, d / 'b.json', '--stream-experts', '64')
            c = run(model, prompt, d / 'c.json', '--stream-bytes', str(6 << 30))
        for r in (a, b, c):
            self.assertTrue(FIELDS <= set(r), sorted(FIELDS - set(r)))
            self.assertEqual(len(r['tokens']), 8)
            self.assertGreater(r['prompt_tokens'], 0)
            self.assertGreater(r['ttft_ns'], 0)
            self.assertGreater(r['phys_footprint_peak'], 0)
            self.assertGreater(r['rss_peak'], 0)
            self.assertGreater(r['metal_allocated_peak'], 0)
            self.assertTrue(all(t['ns'] > 0 for t in r['tokens']))
        self.assertFalse(a['streaming'])
        self.assertTrue(b['streaming'] and b['slots'] == 64)
        # the engine's effective slot count, not the flag as given
        self.assertTrue(c['streaming'] and c['slots'] == 979, c['slots'])
        self.assertEqual([t['id'] for t in a['tokens']], [t['id'] for t in c['tokens']])
        self.assertEqual([t['id'] for t in a['tokens']], [t['id'] for t in b['tokens']])
        self.assertGreater(b['stops'], 0)
        self.assertGreaterEqual(b['loads'], b['stops'])
        self.assertEqual(a['stops'], 0)


if __name__ == '__main__':
    unittest.main()
