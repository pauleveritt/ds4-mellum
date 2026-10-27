"""Committed inputs stay exact, bounded and separate from generated evaluation."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = b'One phase only. Use single quotes.\n'


def module():
    name = 'mellum_workload_bundle'
    if name not in sys.modules:
        path = ROOT / 'tools' / f'{name}.py'
        assert path.is_file(), 'workload bundle is not implemented'
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
    return sys.modules[name]


def git(repo, *args):
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    return subprocess.run(['git', '-C', str(repo), *args], env=env, check=True, capture_output=True).stdout


@contextmanager
def repository():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        repo = root / 'repo'
        repo.mkdir()
        git(repo, 'init', '-q')
        (repo / 'AGENTS.md').write_bytes(ORIGINAL)
        (repo / 'run').write_bytes(b'#!/bin/sh\n')
        (repo / 'run').chmod(0o755)
        git(repo, 'add', '.')
        git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
        yield root, repo, git(repo, 'rev-parse', 'HEAD').decode().strip()


@contextmanager
def refused():
    try:
        yield
    except (ValueError, OSError, subprocess.CalledProcessError):
        pass
    else:
        raise AssertionError('unsafe source accepted')


def test_bundle_matches_git_bytes_and_content_id():
    api = module()
    with repository() as (root, repo, revision):
        first = api.bundle(repo, revision, root / 'first')
        second = api.bundle(repo, revision, root / 'second')
        manifest = json.loads((first / 'manifest.json').read_bytes())
        assert first == root / 'first/artifact'
        assert (first / 'source/AGENTS.md').read_bytes() == ORIGINAL
        assert (first / 'source/run').stat().st_mode & 0o777 == 0o755
        assert manifest['schema_version'] == 1
        assert manifest['kind'] == 'workload-source'
        assert manifest['repository_id'] == 'ac-demo'
        assert manifest['collection_complete'] is False
        assert manifest['split'] == 'evaluation'
        assert manifest['proposed_task']['phase'] == 1
        assert manifest['files']['AGENTS.md']['sha256'] == hashlib.sha256(ORIGINAL).hexdigest()
        assert manifest['files']['AGENTS.md']['git_blob'] == git(repo, 'rev-parse', f'{revision}:AGENTS.md').decode().strip()
        content = dict(revision=revision, tree=git(repo, 'rev-parse', f'{revision}^{{tree}}').decode().strip(),
                       files={name: entry['sha256'] for name, entry in manifest['files'].items()})
        expected = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert manifest['content_id'] == expected
        assert manifest['content_id'] == json.loads((second / 'manifest.json').read_bytes())['content_id']
        assert not git(repo, 'status', '--porcelain', '-uno')


def test_wrong_revision_and_tracked_changes_refused():
    api = module()
    with repository() as (root, repo, revision):
        with refused():
            api.bundle(repo, '0' * 40, root / 'wrong')
        for staged in (False, True):
            (repo / 'AGENTS.md').write_bytes(b'changed')
            if staged:
                git(repo, 'add', 'AGENTS.md')
            with refused():
                api.bundle(repo, revision, root / f'dirty-{staged}')
        with refused():
            api.bundle(repo, revision, repo / 'output')


def test_git_redirects_and_replacements_cannot_substitute_source():
    api = module()
    with repository() as (root, repo, revision), repository() as (_, other, _):
        original_blob = git(repo, 'rev-parse', 'HEAD:AGENTS.md').decode().strip()
        (repo / 'replacement').write_bytes(b'wrong bytes')
        replacement = git(repo, 'hash-object', '-w', 'replacement').decode().strip()
        git(repo, 'replace', original_blob, replacement)
        with patch.dict(os.environ, {'GIT_DIR': str(other / '.git'), 'GIT_WORK_TREE': str(other),
                                    'GIT_INDEX_FILE': str(other / 'wrong-index')}):
            artifact = api.bundle(repo, revision, root / 'output')
        assert (artifact / 'source/AGENTS.md').read_bytes() == ORIGINAL


def test_unsafe_modes_paths_and_blob_limits():
    api = module()
    # Directory entries are validated before any blob is requested.
    blob = b'a' * 40
    def entry(path=b'AGENTS.md', mode=b'100644', size=b'1'):
        return mode + b' blob ' + blob + b' ' + size + b'\t' + path + b'\0'
    for data in (entry(b'../outside'), entry(b'/absolute'), entry(b'bad\xff'), entry(b'.venv/secret'),
                 entry(mode=b'120000'), entry(mode=b'160000'), entry(size=b'16777217'),
                 b''.join(entry(str(n).encode()) for n in range(257)),
                 b''.join(entry(str(n).encode(), size=b'16777216') for n in range(5))):
        with refused():
            api.parse_tree(data)
    with repository() as (root, repo, revision):
        (repo / 'link').symlink_to('AGENTS.md')
        git(repo, 'add', 'link')
        git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'symlink')
        with refused():
            api.bundle(repo, git(repo, 'rev-parse', 'HEAD').decode().strip(), root / 'output')


def test_untracked_secrets_excluded():
    api = module()
    with repository() as (root, repo, revision):
        (repo / 'secret').write_bytes(b'credential')
        (repo / '.venv').mkdir()
        (repo / '.venv/private').write_bytes(b'never copy')
        artifact = api.bundle(repo, revision, root / 'output')
        manifest = json.loads((artifact / 'manifest.json').read_bytes())
        assert set(manifest['files']) == {'AGENTS.md', 'run'}
        assert sorted(str(p.relative_to(artifact / 'source')) for p in (artifact / 'source').rglob('*')) == ['AGENTS.md', 'run']
        assert (repo / '.venv/private').read_bytes() == b'never copy'


def test_mid_bundle_change_and_output_collision_fail():
    api = module()
    with repository() as (root, repo, revision):
        real_run = subprocess.run
        def moving_source(argv, **kwargs):
            result = real_run(argv, **kwargs)
            if 'cat-file' in argv:
                (repo / 'AGENTS.md').write_bytes(b'moved during bundle')
            return result
        with patch.object(subprocess, 'run', side_effect=moving_source), refused():
            api.bundle(repo, revision, root / 'moving')
        assert not (root / 'moving/artifact').exists()
        (repo / 'AGENTS.md').write_bytes(ORIGINAL)
        container = root / 'existing'
        container.mkdir()
        (container / 'sentinel').write_bytes(b'owner')
        with refused():
            api.bundle(repo, revision, container)
        assert (container / 'sentinel').read_bytes() == b'owner'


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
