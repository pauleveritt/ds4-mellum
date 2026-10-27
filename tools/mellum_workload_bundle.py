"""Freeze committed evaluation inputs; never generate or execute the workload."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

_spec = importlib.util.spec_from_file_location('mellum_capture_identity', Path(__file__).with_name('mellum_capture_identity.py'))
identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity)
publication = identity.sibling('mellum_artifact')
ROOT = Path(__file__).resolve().parents[1]


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def git(repo: Path, *args: str) -> bytes:
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env['GIT_NO_REPLACE_OBJECTS'] = '1'
    return subprocess.run(['git', '-C', str(repo), '-c', 'core.fsmonitor=false', *args],
                          env=env, check=True, capture_output=True).stdout


def check_source(repo: Path, revision: str) -> None:
    if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', revision):
        raise ValueError('full commit object ID required')
    if Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != repo:
        raise ValueError('repository root required')
    if git(repo, 'rev-parse', 'HEAD').decode().strip() != revision:
        raise ValueError('source HEAD differs from pinned revision')
    if git(repo, 'status', '--porcelain=v1', '--untracked-files=no'):
        raise ValueError('tracked source changes are unsupported')


def parse_tree(data: bytes) -> list[tuple[str, str, int, int]]:
    """Validate the entire directory and its bounds before requesting any blob."""
    entries = []
    total = 0
    seen = set()
    if data and not data.endswith(b'\0'):
        raise ValueError('unterminated Git tree')
    for raw in data.split(b'\0'):
        if not raw:
            continue
        header, name = raw.split(b'\t', 1)
        mode, kind, blob, size_text = header.split()
        path = name.decode('utf-8')
        parts = PurePosixPath(path).parts
        if (not path or path.startswith('/') or '\\' in path or
                any(part in ('', '.', '..', '.git', '.venv') for part in path.split('/')) or
                not parts or path in seen):
            raise ValueError(f'unsafe tracked path: {path!r}')
        if mode not in (b'100644', b'100755') or kind != b'blob':
            raise ValueError(f'unsupported tracked mode: {path}')
        if not re.fullmatch(rb'(?:[0-9a-f]{40}|[0-9a-f]{64})', blob):
            raise ValueError('invalid blob object ID')
        size = int(size_text)
        total += size
        if not 0 <= size <= 16 * 1024**2 or total > 64 * 1024**2 or len(entries) >= 256:
            raise ValueError('source blob/count budget exceeded')
        seen.add(path)
        entries.append((path, blob.decode(), size, int(mode, 8) & 0o777))
    return sorted(entries)


def bundle(repo: Path, revision: str, container: Path) -> Path:
    repo = repo.resolve(strict=True)
    if container.resolve().is_relative_to(repo):
        raise ValueError('output must be outside the source checkout')
    check_source(repo, revision)
    tree = git(repo, 'rev-parse', f'{revision}^{{tree}}').decode().strip()
    entries = parse_tree(git(repo, 'ls-tree', '-rz', '-l', revision))
    tool = identity.file_identity(Path(__file__))
    source = identity.source_identity(ROOT)

    def build(stage: Path) -> None:
        files = {}
        for name, blob, size, mode in entries:
            content = git(repo, 'cat-file', 'blob', blob)
            if len(content) != size:
                raise ValueError(f'blob size changed: {name}')
            target = stage / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                if stream.write(content) != size:
                    raise OSError(f'short source write: {name}')
                stream.flush()
            target.chmod(mode)
            files[name] = dict(bytes=size, sha256=hashlib.sha256(content).hexdigest(), git_blob=blob)
        check_source(repo, revision)
        content = dict(revision=revision, tree=tree, files={name: value['sha256'] for name, value in files.items()})
        manifest = dict(schema_version=1, kind='workload-source', repository_id='ac-demo', split='evaluation',
                        source=dict(path=str(repo), commit=revision, tree=tree), tool=tool, source_identity=source,
                        files=files, content_id=hashlib.sha256(canonical(content)).hexdigest(),
                        proposed_task=dict(phase=1, request='Implement Phase 1 only, including its home-page tests, without starting a server.'),
                        collection_complete=False)
        with (stage / 'manifest.json').open('xb') as stream:
            payload = canonical(manifest) + b'\n'
            if stream.write(payload) != len(payload):
                raise OSError('short manifest write')
            stream.flush()
        check_source(repo, revision)

    return publication.publish(container, build)


def main() -> None:
    if sys.version_info < (3, 14):
        raise SystemExit('Python 3.14+ required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(bundle(args.repo, args.revision, args.output))


if __name__ == '__main__':
    main()
