"""CPU-only publication tests; synthetic subprocesses never establish GPU behavior."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VALID_PROFILE = (
    'profile-v1;head=encoded;EMBED:kernel_get_rows_q8_0_f32();'
    'PROJECTION:kernel_mul_mv_q8_0_f32(nsg=4);ROUTER:kernel_qwen4_router_topk();'
    'MID:kernel_qwen4_moe_mid(specialized=0,type=8,shared_type=0,dim=2048,rows=2);'
    'DOWN:kernel_qwen4_moe_down(specialized=0,type=8,shared_type=0,dim=2048,rows=2);'
    'REDUCE:kernel_qwen4_moe_reduce();ATTENTION:kernel_mellum_attn_swa();'
    'HEAD:kernel_mul_mv_q8_0_f32(nsg=4)'
)


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
        root = Path(directory).resolve()
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
                 'MELLUM_TRACE_CONTEXT', 'DS4_QWEN4_PREFILL_CHUNK', 'MAKEFLAGS', 'MAKEFILES'):
        for value in ('', '0', 'other'):
            with raises(name):
                identity.checked_environment({'PATH': '/usr/bin', name: value})
    env = {'PATH': '/usr/bin', 'DS4_TEST_MELLUM_MODEL': '/model'}
    assert identity.checked_environment(env) == env
    assert identity.checked_environment(env) is not env



def exporter_fixture():
    name = 'publisher_export_fixture'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'tests/test_mellum_routes_to_trace.py')
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
    return sys.modules[name]


@contextmanager
def publication(change=None):
    publisher = module('mellum_capture_publish')
    fixture = exporter_fixture()
    with repository() as root:
        (root / 'tests').mkdir()
        (root / 'ds4_mellum_kernels.inc').write_bytes((ROOT / 'ds4_mellum_kernels.inc').read_bytes())
        binary = root / 'tests/mellum_routes_gpu'
        binary.write_bytes(b'old binary')
        model, tokens, metadata, output = (root / name for name in ('model.gguf', 'input.tokens', 'metadata.json', 'trace.jsonl'))
        model.write_bytes(b'synthetic model: never opened by inference')
        tokens.write_text('10 11 # first prompt\n12 13\n')
        header = fixture.metadata_fixture()['header']
        for key in ('schema_version', 'capture_kind', 'engine_revision', 'numerical_mode'):
            header.pop(key)
        for key in ('build', 'execution', 'generation'):
            header['capture_manifest'].pop(key)
        header['capture_manifest']['execution'] = {'device': 'caller-declared synthetic test host'}
        metadata.write_text(json.dumps({'header': header}))
        calls = []
        def run(command, *, cwd, env, log):
            calls.append(command)
            log.write_text('synthetic build/capture log\n')
            if command[0] == 'make':
                binary.write_bytes(b'newly compiled binary')
                if change:
                    change('build', root, None)
                return
            assert Path(command[0]) == binary
            assert env['DS4_QWEN4_PREFILL_CHUNK'] == '2'
            directory = Path(command[-1])
            frames = [fixture.frame(0, 0, rows=2, phase=1, segment=0, execution=0, tokens=[10, 11]),
                      fixture.frame(1, 2, phase=2, segment=1, execution=0, tokens=[12]),
                      fixture.frame(2, 3, phase=1, segment=2, execution=0, tokens=[13])]
            data = fixture.spool(*frames)
            (directory / 'owned.routes').write_bytes(data)
            (directory / 'reference.routes').write_bytes(data)
            (directory / 'owned.profiles').write_text('0 ' + VALID_PROFILE + '\n')
            (directory / 'owned.runtime.json').write_text(json.dumps({
                'receipt_version': 1, 'context_tokens': 128, 'cap_tokens': 2,
                'capabilities': {'tensor_tiles_available': True, 'pre_m5_apple': False, 'm5_apple': True}}))
            if change:
                change('capture', root, directory)
        with patch.object(publisher, '_run', run), patch.dict(os.environ, {'PATH': os.environ['PATH']}, clear=True):
            yield publisher, dict(root=root, model=model, tokens=tokens, metadata=metadata,
                                  output=output, first_prefill=2, context=128, prefill_chunk=2), calls
        assert not list(root.glob('.mellum-capture-*'))
        assert not list(root.glob('.mellum-routes-*'))


def test_publication_derives_identity_and_observed_profiles():
    with publication() as (publisher, arguments, calls):
        publisher.publish(**arguments)
        assert len(calls) == 2
        assert calls[0] == ['make', '-B', '-j4', 'tests/mellum_routes_gpu']
        with arguments['output'].open() as source:
            replay = module('mellum_routes_to_trace').replay_module()
            trace = replay.load_trace(source)
        with arguments['output'].open() as source:
            header = json.loads(source.readline())
        manifest = header['capture_manifest']
        assert header['capture_kind'] == 'measured'
        assert manifest['generation'] == {'mode': 'teacher-forced', 'settings': {}}
        assert manifest['build']['binary_sha256'] == hashlib.sha256(b'newly compiled binary').hexdigest()
        assert manifest['build']['source_inputs']['ds4.c'] == hashlib.sha256(b'initial source\n').hexdigest()
        assert manifest['execution']['capabilities']['tensor_tiles_available'] is True
        assert manifest['execution']['route_capture']['profiles'] == {'0': VALID_PROFILE}
        assert manifest['execution']['capture_schedule']['first_prefill'] == 2
        assert manifest['execution']['context_tokens'] == 128
        assert trace is not None


def test_publication_rejects_partial_changed_or_conflicting_evidence():
    def fail(stage, root, directory):
        raise ValueError('child failed')
    cases = [fail]
    for stage, name, data in (
        ('build', 'ds4.c', b'changed'), ('capture', 'ds4.c', b'changed'),
        ('capture', 'tests/mellum_routes_gpu', b'changed'), ('capture', 'input.tokens', b'changed'),
        ('capture', 'owned.runtime.json', b'{}'), ('capture', 'owned.runtime.json', b'{"receipt_version":1,"receipt_version":1}'),
        ('capture', 'owned.profiles', b'0 profile-v1;head=encoded\n0 duplicate\n'),
        ('capture', 'owned.profiles', b'1 profile-v1;head=encoded\n'),
        ('capture', 'owned.profiles', b''), ('capture', 'owned.routes', b'MLRT0001'),
        ('capture', 'reference.routes', b'MLRT0001'),
    ):
        def mutate(current, root, directory, *, stage=stage, name=name, data=data):
            if current == stage:
                path = directory / name if name.startswith(('owned.', 'reference.')) else root / name
                path.write_bytes(data)
        cases.append(mutate)
    for change in cases:
        with publication(change) as (publisher, arguments, calls):
            with raises(''):
                publisher.publish(**arguments)
            assert not arguments['output'].exists()
    for section, value in (
        ('execution_profiles', {'0': 'invented'}), ('build', {'binary_sha256': '0' * 64}),
        ('build', {'source_inputs': {'invented.c': '0' * 64}}),
        ('execution', {'capabilities': {'tensor_tiles_available': False}}),
        ('generation', {'mode': 'generated', 'settings': {}}),
    ):
        with publication() as (publisher, arguments, calls):
            data = json.loads(arguments['metadata'].read_text())
            if section == 'execution_profiles':
                data[section] = value
            else:
                data['header']['capture_manifest'][section] = value
            arguments['metadata'].write_text(json.dumps(data))
            with raises(''):
                publisher.publish(**arguments)
            assert not arguments['output'].exists()


def test_publication_preserves_existing_output_and_rejects_controls_early():
    for symlink in (False, True):
        with publication() as (publisher, arguments, calls):
            output = arguments['output']
            if symlink:
                output.symlink_to(arguments['model'])
            else:
                output.write_bytes(b'keep')
            with raises(''):
                publisher.publish(**arguments)
            assert not calls
            assert output.is_symlink() if symlink else output.read_bytes() == b'keep'
    with publication() as (publisher, arguments, calls):
        with patch.dict(os.environ, {'DS4_MOE_REPLAY_SELECTED_IDS': ''}):
            with raises('DS4_MOE_REPLAY_SELECTED_IDS'):
                publisher.publish(**arguments)
        assert not calls


def test_publisher_rejects_wrong_input_tokens_or_schedule_even_when_reference_agrees():
    accepted = []
    for offset, value in ((60, 99), (24 + len(exporter_fixture().frame(rows=2)) + 24, 1)):
        def mutate(stage, root, directory, *, offset=offset, value=value):
            if stage == 'capture':
                for name in ('owned.routes', 'reference.routes'):
                    path = directory / name
                    data = bytearray(path.read_bytes())
                    struct.pack_into('<I', data, offset, value)
                    path.write_bytes(data)
        with publication(mutate) as (publisher, arguments, calls):
            try:
                publisher.publish(**arguments)
            except ValueError:
                pass
            else:
                accepted.append(offset)
    assert not accepted, f'accepted wrong input token/schedule at offsets {accepted}'


def test_native_receipt_and_catalog_bounds():
    parser = module('mellum_capture_metadata')
    exporter = module('mellum_routes_to_trace')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'catalog'
        for text in ('', '0 profile-v1;head=encoded', '0 bad\n', '0 profile-v1;head=encoded' + 'x'*16384 + '\n',
                     '0 profile-v1;head=encoded\x00\n'):
            path.write_text(text)
            with raises('profile'):
                parser.read_profiles(path)
        path.write_text(''.join(f'{i} ' + VALID_PROFILE.replace('nsg=4', f'nsg={i+1}') + '\n' for i in range(128)))
        assert len(parser.read_profiles(path)) == 128
        with path.open('a') as out:
            out.write('128 profile-v1;head=encoded\n')
        with raises('profile'):
            parser.read_profiles(path)
        path.write_bytes(b' ' * 16385)
        with raises('exceeds'):
            parser.read_receipt(path, exporter)
        for value in (True, 0, 1025):
            path.write_text(json.dumps({'receipt_version': 1, 'context_tokens': 128, 'cap_tokens': value,
                                       'capabilities': {}}))
            with raises('receipt'):
                parser.read_receipt(path, exporter)


def test_publication_requires_a_rebuilt_binary():
    with publication() as (publisher, arguments, calls):
        original = publisher._run
        def no_build(command, *, cwd, env, log):
            if command[0] == 'make':
                log.write_text('compiler did not produce a binary\n')
            else:
                original(command, cwd=cwd, env=env, log=log)
        with patch.object(publisher, '_run', no_build):
            with raises('rebuilt'):
                publisher.publish(**arguments)
        assert not arguments['output'].exists()


def test_incomplete_or_noncanonical_profiles_never_publish():
    for text in ('profile-v1;head=encoded', 'profile-v1;head=encoded;garbage', VALID_PROFILE[:-1],
                 VALID_PROFILE.replace('head=encoded', 'head=not-encoded'),
                 VALID_PROFILE.replace('nsg=4', 'nsg=', 1),
                 VALID_PROFILE.replace('kernel_get_rows_q8_0_f32', 'unknown_kernel')):
        def mutate(stage, root, directory, *, text=text):
            if stage == 'capture':
                (directory / 'owned.profiles').write_text('0 ' + text + '\n')
        with publication(mutate) as (publisher, arguments, calls):
            with raises('profile'):
                publisher.publish(**arguments)
            assert not arguments['output'].exists()


def test_git_redirection_cannot_hide_tracked_source_mutation():
    def mutate(stage, root, directory):
        if stage == 'capture':
            (root / 'ignored.c').write_text('changed tracked source')
    with publication(mutate) as (publisher, arguments, calls):
        root = arguments['root']
        (root / '.gitignore').write_text('ignored.c\n')
        (root / 'ignored.c').write_text('tracked despite ignore')
        subprocess.run(['git', '-C', str(root), 'add', '-f', 'ignored.c', '.gitignore'], check=True)
        with patch.dict(os.environ, {'GIT_INDEX_FILE': str(root / 'alternate-index')}):
            with raises('GIT_INDEX_FILE'):
                publisher.publish(**arguments)
        assert not calls and not arguments['output'].exists()


def test_primary_makefile_selection_is_explicit_and_source_changes_rejected():
    def mutate(stage, root, directory):
        if stage == 'capture':
            (root / 'GNUmakefile').write_text('changed build input')
    with publication(mutate) as (publisher, arguments, calls):
        (arguments['root'] / 'GNUmakefile').write_text('alternate build input')
        with raises('source'):
            publisher.publish(**arguments)
        assert calls[0] == ['make', '-f', 'Makefile', '-B', '-j4', 'tests/mellum_routes_gpu']
        assert not arguments['output'].exists()

def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith('test_') and callable(value))


if __name__ == '__main__':
    unittest.main()
