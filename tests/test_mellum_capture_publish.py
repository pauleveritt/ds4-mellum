"""CPU-only publication tests; synthetic subprocesses never establish GPU behavior."""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    if name not in sys.modules:
        path = ROOT / 'tools' / f'{name}.py'
        assert path.is_file(), f'{name} is not implemented'
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
    return sys.modules[name]


@contextmanager
def repository():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        (root / 'ds4.c').write_bytes(b'initial source\n')
        (root / 'metal').mkdir()
        (root / 'metal/mellum.metal').write_bytes(b'initial shader\n')
        subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-qm', 'fixture'], check=True)
        yield root


def test_source_and_binary_changes_are_detected():
    identity = module('mellum_capture_identity')
    with repository() as root:
        original = identity.source_identity(root)
        assert original == identity.source_identity(root)
        assert original['files']['ds4.c']['sha256'] == hashlib.sha256(b'initial source\n').hexdigest()
        (root / 'ds4.c').write_bytes(b'changed source\n')
        assert original != identity.source_identity(root)
        (root / 'new.h').write_bytes(b'new source')
        assert 'new.h' in identity.source_identity(root)['files']
        (root / 'ds4.c').unlink()
        with raises('source'):
            identity.source_identity(root)
        binary = root / 'binary'
        binary.write_bytes(b'first')
        first = identity.file_identity(binary)
        binary.write_bytes(b'other')
        assert first != identity.file_identity(binary)
        (root / 'ds4.c').symlink_to(binary)
        with raises('symlink'):
            identity.source_identity(root)


@contextmanager
def raises(reason):
    try:
        yield
    except (ValueError, OSError) as exc:
        assert reason in str(exc), str(exc)
    else:
        raise AssertionError(f'expected rejection: {reason}')


def test_inherited_controls_are_rejected_before_collection():
    identity = module('mellum_capture_identity')
    for name in ('DS4_METAL_MELLUM_SOURCE', 'DS4_MOE_REPLAY_SELECTED_IDS', 'DS4_MELLUM_TRACE',
                 'MELLUM_TRACE_CONTEXT', 'DS4_QWEN4_PREFILL_CHUNK'):
        for value in ('', '0', 'other'):
            with raises(name):
                identity.checked_environment({'PATH': '/usr/bin', name: value})
    env = {'PATH': '/usr/bin', 'DS4_TEST_MELLUM_MODEL': '/model'}
    assert identity.checked_environment(env) == env
    assert identity.checked_environment(env) is not env


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith('test_') and callable(value))


if __name__ == '__main__':
    unittest.main()
