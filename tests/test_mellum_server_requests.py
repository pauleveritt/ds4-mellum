"""CPU tests of the ds4-server log parser (tools/mellum_server_requests.py) on log lines in ds4_server.c's formats."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / 'tools/mellum_server_requests.py'
    assert path.exists(), 'tools/mellum_server_requests.py missing'
    spec = importlib.util.spec_from_file_location('mellum_server_requests', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DONE = '1031 12:00:01 ds4-server: chat ctx={ctx} TOOLS prompt done {s}s'
FINISH = '1031 12:00:03 ds4-server: chat ctx={ctx} gen={gen} TOOLS THINKING finish={fin} {s}s'


class Parse(unittest.TestCase):
    def test_one_request_rates(self):
        m = load()
        reqs = m.parse([DONE.format(ctx='0..4096:4096', s='2.000'),
                        FINISH.format(ctx='0..4096:4096', gen=256, fin='stop', s='4.000')])
        self.assertEqual(len(reqs), 1)
        r = reqs[0]
        self.assertEqual((r.index, r.cached, r.prompt, r.new, r.generated, r.finish, r.error),
                         (0, 0, 4096, 4096, 256, 'stop', None))
        self.assertAlmostEqual(r.prefill_tps, 2048.0)
        self.assertAlmostEqual(r.decode_tps, 128.0)

    def test_fully_cached_request_has_no_prefill_rate(self):
        m = load()
        r = m.parse([DONE.format(ctx='5000..5000:0', s='0.000'),
                     FINISH.format(ctx='5000..5000:0', gen=10, fin='stop', s='0.100')])[0]
        self.assertIsNone(r.prefill_tps)
        self.assertAlmostEqual(r.decode_tps, 100.0)

    def test_error_finish_keeps_the_error(self):
        m = load()
        r = m.parse([DONE.format(ctx='0..100:100', s='0.500'),
                     '1031 12:00:03 ds4-server: chat ctx=0..100:100 gen=3 finish=error error="boom" 0.700s'])[0]
        self.assertEqual((r.finish, r.error, r.generated), ('error', 'boom', 3))

    def test_finish_without_prompt_done(self):
        m = load()
        r = m.parse([FINISH.format(ctx='0..100:100', gen=0, fin='error', s='0.010')])[0]
        self.assertIsNone(r.prefill_s)
        self.assertIsNone(r.prefill_tps)
        self.assertIsNone(r.decode_tps)

    def test_parses_coloured_interleaved_log(self):
        m = load()
        lines = [
            '1031 12:00:00 \x1b[36mds4-server: chat ctx=1000..3000:2000 TOOLS prompt done 1.000s\x1b[0m',
            '1031 12:00:01 ds4-server: chat ctx=3000..3050:50 gen=50 TOOLS decoding chunk=51.00 t/s avg=50.00 t/s 1.000s',
            '1031 12:00:01 ds4-server: thinking live checkpoint remembered ctx=1000..3000:2000 live=3050 visible=12',
            '1031 12:00:02 \x1b[32mds4-server: chat ctx=1000..3000:2000 gen=100 TOOLS finish=tool_calls 3.000s\x1b[0m',
        ]
        reqs = m.parse(lines)
        self.assertEqual(len(reqs), 1)
        self.assertEqual((reqs[0].cached, reqs[0].new, reqs[0].finish), (1000, 2000, 'tool_calls'))
        self.assertAlmostEqual(reqs[0].prefill_tps, 2000.0)
        self.assertAlmostEqual(reqs[0].decode_tps, 50.0)

    def test_requests_are_numbered_in_order(self):
        m = load()
        lines = []
        for ctx in ('0..100:100', '100..400:300'):
            lines += [DONE.format(ctx=ctx, s='0.100'), FINISH.format(ctx=ctx, gen=5, fin='stop', s='0.200')]
        self.assertEqual([r.index for r in m.parse(lines)], [0, 1])


if __name__ == '__main__':
    unittest.main()
