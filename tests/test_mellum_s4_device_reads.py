"""Live (DS4_TEST_MELLUM_MODEL): every expert a streamed Mellum session loads is read from the SSD, not the
page cache, even when it was loaded before (64 slots: constant reloads).  A write-only fresh copy of the
model, block-driver read counters around one driver run, and device bytes >= 0.99 x loads x expert bytes.
Skipped without the model."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class DeviceReads(unittest.TestCase):
    def test_reloaded_experts_come_from_the_device(self):
        model = os.environ.get('DS4_TEST_MELLUM_MODEL')
        if not model:
            self.skipTest('DS4_TEST_MELLUM_MODEL unset')
        meas, obs, cls = load('mellum_s4_measure'), load('mellum_device_observer'), load('mellum_device_classify')
        # the copy lives beside the model (same volume, not a RAM-backed temporary)
        with tempfile.TemporaryDirectory(dir=Path(model).parent) as d:
            d = Path(d)
            copy = d / 'copy.gguf'
            meas.fresh_copy(Path(model), copy)
            prompt = d / 'prompt.txt'
            prompt.write_text('Explain how a hash map works, with an example in C.')
            f = dict(path=str(copy), dev=os.stat(copy).st_dev)
            topo = obs.discover({'model': f, 'payload': dict(f)})
            before = obs.snapshot(topo)
            subprocess.run([str(ROOT / 'tools/mellum_s4_measure'), '--model', str(copy), '--prompt', str(prompt),
                            '--gen', '32', '--context', '8192', '--stream-experts', '64', '--out', str(d / 'r.json')],
                           check=True, cwd=ROOT, timeout=900, capture_output=True)
            after = obs.snapshot(topo)
            report = json.loads((d / 'r.json').read_text())
            logical = report['loads'] * meas.EXPERT_BYTES
            device = cls.delta(before, after)['Bytes (Read)']
            print(f'device-reads: {report["loads"]} loads, {logical} logical bytes, {device} device bytes, '
                  f'ratio {device / logical:.3f}')
            self.assertGreater(report['loads'], 1792, 'expected reloads beyond one pass over the experts')
            self.assertGreaterEqual(device, 0.99 * logical)
            meas.verify_copy(Path(model), copy)


if __name__ == '__main__':
    unittest.main()
