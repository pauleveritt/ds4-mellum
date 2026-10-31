"""CPU tests of the AgentClinic grader's verdict (tools/mellum_pi_grade.py): no uv, model or network.  The grader
itself runs under `make test-mellum-pi-grader`, which proves it both ways on the copied fixtures."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / 'tools/mellum_pi_grade.py'
    assert path.exists(), 'tools/mellum_pi_grade.py missing'
    spec = importlib.util.spec_from_file_location('mellum_pi_grade', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Verdict(unittest.TestCase):
    def test_missing_checks_lists_expected_ids_not_passed(self):
        m = load()
        output = 'PASSED grader_tests/t.py::a\nFAILED grader_tests/t.py::b - assert 0\n'
        expected = ['grader_tests/t.py::a', 'grader_tests/t.py::b', 'grader_tests/t.py::c']
        self.assertEqual(m.missing_checks(output, expected), ['grader_tests/t.py::b', 'grader_tests/t.py::c'])

    def test_collection_error_is_not_checks_ran(self):
        m = load()
        self.assertFalse(m.checks_ran(2, 'ERROR grader_tests/t.py\n1 error in 0.1s\n'))
        self.assertFalse(m.checks_ran(1, 'Interrupted: 1 error during collection\n! errors during collection !'))
        self.assertFalse(m.checks_ran(5, 'no tests ran in 0.01s\n'))
        self.assertFalse(m.checks_ran(1, '!!! Interrupted: 1 error during collection !!!\n'))

    def test_checks_ran_when_tests_pass_or_fail(self):
        m = load()
        self.assertTrue(m.checks_ran(0, '4 passed in 0.5s\n'))
        self.assertTrue(m.checks_ran(1, '1 failed, 3 passed in 0.5s\n'))


if __name__ == '__main__':
    unittest.main()
