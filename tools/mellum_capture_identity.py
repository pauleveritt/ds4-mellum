"""Source consistency checks for diagnostic capture; never initializes a GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


def sibling(name: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f'{name}.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def file_stat(path: Path) -> dict:
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode):
        raise ValueError(f'symlink input is unsupported: {path}')
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f'regular file required: {path}')
    return {key: getattr(value, f'st_{key}') for key in ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}


def file_identity(path: Path) -> dict:
    before = file_stat(path)
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(65536):
            digest.update(block)
    if before != file_stat(path):
        raise ValueError(f'input changed while hashing: {path}')
    return dict(sha256=digest.hexdigest(), **before)


def source_identity(root: Path) -> dict:
    def git(*args: str) -> bytes:
        env = {name: value for name, value in os.environ.items() if not name.startswith('GIT_')}
        return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, env=env).stdout

    paths = {Path(name.decode()) for name in git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0') if name}
    paths.update(path.relative_to(root) for path in (root / 'metal').rglob('*.metal'))
    files = {}
    for relative in sorted(paths):
        if (relative.suffix not in ('.c', '.h', '.m', '.inc', '.metal', '.py') and
                not relative.name.startswith('Makefile') and relative.name not in ('GNUmakefile', 'makefile')):
            continue
        path = root / relative
        if any((root / parent).is_symlink() for parent in (relative, *relative.parents)):
            raise ValueError(f'symlink source input: {relative}')
        if not path.is_file():
            raise ValueError(f'missing source input: {relative}')
        files[str(relative)] = file_identity(path)
    if not files:
        raise ValueError('no source inputs')
    revision = git('rev-parse', 'HEAD').decode().strip()
    canonical = json.dumps({'revision': revision, 'files': {name: value['sha256'] for name, value in files.items()}},
                           sort_keys=True, separators=(',', ':')).encode()
    return dict(revision=revision, files=files, source_state_id=hashlib.sha256(canonical).hexdigest())


def checked_environment(env: dict[str, str]) -> dict[str, str]:
    for name in env:
        if ((name.startswith('DS4_') and name != 'DS4_TEST_MELLUM_MODEL') or name.startswith('MELLUM_TRACE_') or
                name.startswith('GIT_') or name in ('MAKEFLAGS', 'MFLAGS', 'MAKEFILES', 'MAKEOVERRIDES')):
            raise ValueError(f'inherited capture control is unsupported: {name}')
    return dict(env)
