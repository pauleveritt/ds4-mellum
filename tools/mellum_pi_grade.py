"""Grade a project against the copied AgentClinic task's hidden checks, and prove the grader both ways.

    python3 tools/mellum_pi_grade.py grade WORKSPACE [SUITE]
    python3 tools/mellum_pi_grade.py qualify [SUITE]

SUITE defaults to tests/agentclinic_easy.  Grading copies SUITE/grader into WORKSPACE as grader_tests and runs
`uv run --with pytest pytest -q -rA grader_tests` there, as mellum-worker's harness does; the project passes when the
checks ran and every id in SUITE/expected.json passed.  qualify grades SUITE/base with each fixture laid over it and
exits 0 only if known-good passes and known-broken fails.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'tests/agentclinic_easy'


@dataclass(frozen=True)
class GradeResult:
    passed: bool
    checks_ran: bool
    missing: list[str]
    output_tail: str


def missing_checks(output: str, expected: list[str]) -> list[str]:
    """The expected check ids that pytest's -rA summary does not list as PASSED, in expected order."""
    passed = {line.split()[1] for line in output.splitlines() if line.startswith('PASSED ') and len(line.split()) > 1}
    return [check for check in expected if check not in passed]


def checks_ran(returncode: int, output: str) -> bool:
    """pytest ran the checks (all passed, or some failed): not a collection error and not an empty run."""
    return returncode in (0, 1) and 'during collection' not in output


def grade(workspace: Path, suite: Path = SUITE) -> GradeResult:
    workspace, suite = Path(workspace), Path(suite)
    target = workspace / 'grader_tests'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(suite / 'grader', target)
    command = ['uv', 'run', '--with', 'pytest', 'pytest', '-q', '-rA', 'grader_tests']
    result = subprocess.run(command, cwd=workspace, capture_output=True, text=True, check=False,
                            env={**os.environ, 'PYTHONPATH': str(workspace)})
    output = (result.stdout + result.stderr).strip()
    ran = checks_ran(result.returncode, output)
    missing = missing_checks(output, json.loads((suite / 'expected.json').read_text()))
    return GradeResult(passed=ran and not missing, checks_ran=ran, missing=missing,
                       output_tail='\n'.join(output.splitlines()[-20:]))


def qualify(suite: Path = SUITE) -> dict[str, bool]:
    results = {}
    for name in ('known-good', 'known-broken'):
        with tempfile.TemporaryDirectory(prefix=f'mellum-pi-{name}-') as tmp:
            workspace = Path(tmp)
            shutil.copytree(Path(suite) / 'base', workspace, dirs_exist_ok=True)
            shutil.copytree(Path(suite) / 'fixtures' / name, workspace, dirs_exist_ok=True)
            results[name] = grade(workspace, suite).passed
    return results


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == 'qualify':
        results = qualify(Path(argv[2]) if len(argv) > 2 else SUITE)
        print(json.dumps(results))
        return 0 if results['known-good'] and not results['known-broken'] else 1
    if len(argv) >= 3 and argv[1] == 'grade':
        result = grade(Path(argv[2]), Path(argv[3]) if len(argv) > 3 else SUITE)
        print(json.dumps(result.__dict__, indent=2))
        return 0 if result.passed else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
