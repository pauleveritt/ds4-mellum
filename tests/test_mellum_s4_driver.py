"""Live smoke for tools/mellum_s4_measure (DS4_TEST_MELLUM_MODEL): resident and streamed runs of a short chat
prompt report every field the S4 measurement reads, and greedy decode gives the same token ids.  Skipped
without the model."""
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


def run(model, prompt, out, *extra):
    cmd = [str(DRIVER), '--model', model, '--prompt', str(prompt), '--gen', '8', '--context', '8192', '--out', str(out), *extra]
    subprocess.run(cmd, check=True, cwd=ROOT, timeout=300, capture_output=True)
    return json.loads(out.read_text())


class DriverSmoke(unittest.TestCase):
    def test_resident_and_streamed_reports_agree(self):
        model = os.environ.get('DS4_TEST_MELLUM_MODEL')
        if not model:
            self.skipTest('DS4_TEST_MELLUM_MODEL unset')
        self.assertTrue(DRIVER.exists(), 'tools/mellum_s4_measure is not built')
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            prompt = d / 'prompt.txt'
            prompt.write_text('Explain how a hash map works, with an example in C.')
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
