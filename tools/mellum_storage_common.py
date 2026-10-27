"""Shared bounded JSON and opened-file identities for storage evidence."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
from typing import BinaryIO

_spec = importlib.util.spec_from_file_location('mellum_capture_identity', Path(__file__).with_name('mellum_capture_identity.py'))
identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity)
publication = identity.sibling('mellum_artifact')
ROOT = Path(__file__).resolve().parents[1]
STAT_KEYS = ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def load_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def number(raw):
        result = float(raw)
        if not math.isfinite(result):
            raise ValueError('nonfinite JSON number')
        return result
    with path.open('rb') as source:
        raw = source.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2:
        raise ValueError('JSON exceeds 8 MiB')
    value = json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                       parse_constant=lambda raw: (_ for _ in ()).throw(ValueError('nonfinite JSON constant')))
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def write_json(path: Path, value: dict) -> None:
    payload = canonical(value) + b'\n'
    with path.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short JSON write')
        stream.flush()


def stat_value(value: os.stat_result) -> dict:
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('regular file required')
    return {key: getattr(value, f'st_{key}') for key in STAT_KEYS}


def check_open(source: BinaryIO, path: Path, expected: dict) -> None:
    if stat_value(os.fstat(source.fileno())) != expected or identity.file_stat(path) != expected:
        raise ValueError(f'file changed or substituted: {path}')


@contextmanager
def opened(path: Path):
    before = identity.file_stat(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as source:
        check_open(source, path, before)
        yield source, before
        check_open(source, path, before)


def hash_stream(source: BinaryIO, length: int | None = None) -> str:
    digest = hashlib.sha256()
    source.seek(0)
    remaining = length
    while remaining is None or remaining:
        block = source.read(65536 if remaining is None else min(65536, remaining))
        if not block:
            if remaining:
                raise ValueError('short hashing read')
            break
        digest.update(block)
        if remaining is not None:
            remaining -= len(block)
    return digest.hexdigest()


def integer(value: object, low: int = 0, high: int = 2**63 - 1) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError('integer outside supported bounds')
    return value


def digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('SHA-256 required')
    return value
