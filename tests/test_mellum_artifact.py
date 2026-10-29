"""Publication failures must never expose a complete artifact or replace an owner."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


def publisher():
    path = Path(__file__).resolve().parents[1] / 'tools/mellum_artifact.py'
    assert path.is_file(), 'mellum_artifact is not implemented'
    spec = importlib.util.spec_from_file_location('mellum_artifact', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_complete_directory_visible_only_after_builder_returns():
    api = publisher()
    with tempfile.TemporaryDirectory() as directory:
        container = Path(directory) / 'output'
        def build(stage):
            assert container.is_dir()
            assert not (container / 'artifact').exists()
            (stage / 'x').write_bytes(b'complete')
            assert not (container / 'artifact').exists()
        artifact = api.publish(container, build)
        assert artifact == container / 'artifact'
        assert (artifact / 'x').read_bytes() == b'complete'
        assert list(container.iterdir()) == [artifact]


def test_existing_outputs_untouched():
    api = publisher()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'sentinel').write_bytes(b'owner')
        (root / 'empty').mkdir()
        (root / 'link').symlink_to(root / 'empty', target_is_directory=True)
        for output in (root / 'sentinel', root / 'empty', root / 'link'):
            try:
                api.publish(output, lambda stage: (_ for _ in ()).throw(AssertionError('builder ran')))
            except FileExistsError:
                pass
            else:
                raise AssertionError('existing output accepted')
        assert (root / 'sentinel').read_bytes() == b'owner'
        assert not list((root / 'empty').iterdir())
        assert (root / 'link').is_symlink()


def test_competing_reservation_has_one_winner():
    api = publisher()
    with tempfile.TemporaryDirectory() as directory:
        container = Path(directory) / 'output'
        barrier = threading.Barrier(2)
        def attempt(value):
            barrier.wait()
            try:
                api.publish(container, lambda stage: (stage / 'sentinel').write_bytes(value))
            except FileExistsError:
                return False
            return value
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, (b'one', b'two')))
        winners = [value for value in results if value]
        assert len(winners) == 1
        assert (container / 'artifact/sentinel').read_bytes() == winners[0]


def test_builder_close_and_rename_failures_never_publish():
    api = publisher()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sibling = root / 'unrelated'
        sibling.write_bytes(b'keep')
        for failure in ('builder', 'close', 'rename'):
            container = root / failure
            marker = OSError(failure)
            def build(stage):
                with (stage / 'x').open('wb') as stream:
                    stream.write(b'partial')
                    if failure == 'builder':
                        raise marker
                if failure == 'close':
                    raise marker  # caller's checked-close failure, after writing
            try:
                if failure == 'rename':
                    with patch.object(Path, 'rename', side_effect=marker):
                        api.publish(container, build)
                else:
                    api.publish(container, build)
            except OSError as exc:
                assert exc is marker
            else:
                raise AssertionError('failed builder published')
            assert not container.exists()
            assert sibling.read_bytes() == b'keep'


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
