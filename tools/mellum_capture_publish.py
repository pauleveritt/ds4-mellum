#!/usr/bin/env python3
"""Build, collect and publish source-checked teacher-forced Mellum route evidence."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile

_spec = importlib.util.spec_from_file_location('mellum_capture_identity', Path(__file__).with_name('mellum_capture_identity.py'))
if _spec.name not in sys.modules:
    sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules[_spec.name])
identity = sys.modules[_spec.name]
metadata_tools = identity.sibling('mellum_capture_metadata')
exporter = identity.sibling('mellum_routes_to_trace')


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open('wb') as output:
        subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)


def _check_routes(reference: Path, owned: Path, cap: int, words: list[str], first: int) -> None:
    with reference.open('rb') as left, owned.open('rb') as right:
        a, b = exporter.SpoolReader(left), exporter.SpoolReader(right)
        if a.cap_tokens != cap or b.cap_tokens != cap:
            raise ValueError('runtime receipt and spool caps differ')
        while True:
            x, y = a.read_forward(), b.read_forward()
            if x is None or y is None:
                if x is not None or y is not None:
                    raise ValueError('reference routes differ')
                expected = hashlib.sha256()
                for word in words:
                    expected.update(struct.pack('<I', int(word)))
                summary = b.summary()
                if (summary['first_position'] != 0 or summary['positions'] != len(words) or
                        summary['token_stream_sha256'] != expected.hexdigest()):
                    raise ValueError('captured tokens differ from input')
                return
            if replace(x, execution_id=0) != replace(y, execution_id=0):
                raise ValueError('reference routes differ')
            pos = y.first_position
            rows, phase, segment = ((min(cap, first-pos), 1, 0) if pos < first else
                                    (1, 1, 2) if pos == len(words)-1 else (1, 2, 1))
            if (y.rows, y.phase, y.segment_id) != (rows, phase, segment):
                raise ValueError('captured operation schedule differs from request')


def publish(root: Path, *, model: Path, tokens: Path, first_prefill: int, context: int,
            prefill_chunk: int, metadata: Path, output: Path, exact_prefill: bool = False) -> None:
    env = identity.checked_environment(dict(os.environ))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'output already exists: {output}')
    if type(context) is not int or not 65 <= context <= 2**31-1 or type(prefill_chunk) is not int or not 1 <= prefill_chunk <= 1024:
        raise ValueError('invalid context or prefill chunk')
    root, model, tokens = root.resolve(), model.resolve(strict=True), tokens.resolve(strict=True)
    data = metadata_tools.read_json(metadata, 2**20, exporter)
    if set(data) != {'header'} or not isinstance(data['header'], dict):
        raise ValueError('metadata must contain header only; profiles must be observed')
    raw_tokens = metadata_tools.bounded_read(tokens, 16 * 2**20)
    words = ' '.join(line.partition('#')[0] for line in raw_tokens.decode('ascii').splitlines()).split()
    if not words or any(not word.isascii() or not word.isdecimal() or len(word) > 10 or int(word) >= 98304 for word in words):
        raise ValueError('invalid token file')
    if type(first_prefill) is not int or not 1 <= first_prefill <= len(words)-2 or len(words)+64 > context:
        raise ValueError('initial prefill or token count exceeds context/protocol')
    source = identity.source_identity(root)
    token_file, model_stat = identity.file_identity(tokens), identity.file_stat(model)
    if token_file['sha256'] != hashlib.sha256(raw_tokens).hexdigest():
        raise ValueError('token file changed while parsing')
    with tempfile.TemporaryDirectory(prefix='.mellum-capture-', dir=output.parent) as directory:
        stage = Path(directory)
        build_log = stage / 'build.log'
        binary_path = root / 'tests/mellum_routes_gpu'
        old_binary = identity.file_stat(binary_path) if binary_path.exists() else None
        _run(['make', '-f', 'Makefile', '-B', '-j4', 'tests/mellum_routes_gpu'], cwd=root, env=env, log=build_log)
        if source != identity.source_identity(root):
            raise ValueError('source inputs changed during build')
        binary = identity.file_identity(binary_path)
        if old_binary is not None and old_binary == {key: binary[key] for key in old_binary}:
            raise ValueError('capture binary was not rebuilt')
        child_env = dict(env, MELLUM_TRACE_CONTEXT=str(context), DS4_QWEN4_PREFILL_CHUNK=str(prefill_chunk))
        if exact_prefill:
            child_env['DS4_MELLUM_EXACT_PREFILL'] = '1'
        _run([str(binary_path), str(model), str(tokens), str(first_prefill), str(stage)],
             cwd=root, env=child_env, log=stage / 'capture.log')
        if (source != identity.source_identity(root) or binary != identity.file_identity(binary_path) or
                token_file != identity.file_identity(tokens) or model_stat != identity.file_stat(model)):
            raise ValueError('capture input source/binary/token/model identity changed')
        receipt = metadata_tools.read_receipt(stage / 'owned.runtime.json', exporter)
        if receipt['context_tokens'] != context or receipt['cap_tokens'] > prefill_chunk:
            raise ValueError('runtime receipt conflicts with requested context/chunk')
        _check_routes(stage / 'reference.routes', stage / 'owned.routes', receipt['cap_tokens'], words, first_prefill)
        profiles = metadata_tools.read_profiles(stage / 'owned.profiles', root / 'ds4_mellum_kernels.inc')
        header = metadata_tools.complete_header(data['header'], source=source, binary=binary, receipt=receipt,
            token_file=token_file, model_stat=model_stat,
            build_log=metadata_tools.bounded_read(build_log, 256 * 1024).decode('utf-8'),
            first=first_prefill, count=len(words), exact=exact_prefill, chunk=prefill_chunk)
        prepared = stage / 'metadata.json'
        prepared.write_text(json.dumps(dict(header=header, segments={'0': 'initial-prefill',
            '1': 'teacher-forced-continuation', '2': 'final-append'}, execution_profiles=profiles),
            ensure_ascii=False, allow_nan=False), encoding='utf-8')
        exporter.export_trace(stage / 'owned.routes', prepared, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'tokens', 'metadata', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    for name in ('first-prefill', 'context', 'prefill-chunk'):
        parser.add_argument(f'--{name}', required=True, type=int)
    parser.add_argument('--exact-prefill', action='store_true')
    args = parser.parse_args()
    try:
        publish(Path(__file__).resolve().parents[1], **vars(args))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
