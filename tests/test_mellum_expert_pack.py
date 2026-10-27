"""Exact original role bytes must survive packing and independent rereading."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = dict(layer=0, expert=0, ranges=[dict(role=role, offset=offset, bytes=2)
                                      for role, offset in zip(('gate','up','down'), (0,8,16))])


def module():
    name = 'mellum_expert_pack'
    if name not in sys.modules:
        path = ROOT / 'tools' / f'{name}.py'
        assert path.is_file(), 'expert pack is not implemented'
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
    return sys.modules[name]


@contextmanager
def rejected():
    try:
        yield
    except (ValueError, OSError):
        pass
    else:
        raise AssertionError('invalid copy/packing accepted')


def test_copy_noncontiguous_roles_preserves_exact_bytes():
    api = module()
    source, destination = io.BytesIO(b'abcdefghijklmnopqrstuvwxyz'), io.BytesIO()
    entries = api.copy_entries(source, destination, [ENTRY])
    assert destination.getvalue() == b'abijqr'
    assert entries[0]['ranges'] == [dict(role='packed', offset=0, bytes=6)]
    api.verify_copy(source, destination, [ENTRY], entries)
    destination.seek(3)
    destination.write(b'X')
    with rejected():
        api.verify_copy(source, destination, [ENTRY], entries)


def test_source_and_packed_hashes_match_per_expert():
    api = module()
    destination = io.BytesIO()
    entries = api.copy_entries(io.BytesIO(b'abcdefghijklmnopqrstuvwxyz'), destination, [ENTRY, ENTRY])
    expected = hashlib.sha256(b'abijqr').hexdigest()
    assert entries[0]['source_sha256'] == expected
    assert entries[0]['packed_sha256'] == expected
    assert entries[1]['ranges'][0]['offset'] == 6
    assert destination.getvalue() == b'abijqrabijqr'
    # Large generic parts still use bounded streaming requests.
    source = io.BytesIO(b'x' * 200000)
    requests = []
    real_read = source.read
    def bounded(n):
        requests.append(n)
        assert 0 < n <= 65536
        return real_read(n)
    with patch.object(source, 'read', side_effect=bounded):
        api.copy_entries(source, io.BytesIO(), [dict(layer=0,expert=0,ranges=[dict(offset=0,bytes=200000)])])
    assert len(requests) >= 4


@contextmanager
def tiny_public_inputs(api):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'source'
        source.write_bytes(b'abcdefghijklmnopqrstuvwxyz')
        index = dict(schema_version=1, geometry={'expert_bytes':6}, experts=[deepcopy(ENTRY)],
                     source=dict(path=str(source), **api.common.identity.file_identity(source)))
        path = root / 'index.json'
        path.write_text(json.dumps(index))
        # Only strict full-model prerequisites are injected; copy, verification,
        # opened-file identities, flush/close and publication remain real.
        with patch.object(api.index_tools, 'validate_index', side_effect=lambda value:value), \
             patch.object(api.index_tools, 'verify_index_source'), \
             patch.object(api, 'validate_packed', side_effect=lambda value:value):
            yield root, path, index


def test_short_read_write_close_or_verify_failure_never_publishes():
    api = module()
    with rejected():
        api.copy_entries(io.BytesIO(b'a'), io.BytesIO(), [ENTRY])
    output = io.BytesIO()
    with patch.object(output, 'write', return_value=0), rejected():
        api.copy_entries(io.BytesIO(b'abcdefghijklmnopqrstuvwxyz'), output, [ENTRY])
    for boundary in ('verify_copy', 'write_json'):
        with tiny_public_inputs(api) as (root, path, _):
            owner = api if boundary == 'verify_copy' else api.common
            with patch.object(owner, boundary, side_effect=OSError('checked verification/close failure')), rejected():
                api.pack(path, root / 'output')
            assert not (root / 'output/artifact').exists()


def test_source_index_mutation_and_insufficient_space_fail():
    api = module()
    with tiny_public_inputs(api) as (root, path, _):
        usage = api.shutil.disk_usage(root)
        with patch.object(api.shutil, 'disk_usage', return_value=type(usage)(usage.total, usage.used, 0)), rejected():
            api.pack(path, root / 'space')
        assert not (root / 'space/artifact').exists()
        real_copy = api.copy_entries
        def mutate(source, destination, entries):
            result = real_copy(source,destination,entries)
            path.write_bytes(b'changed index')
            return result
        with patch.object(api, 'copy_entries', side_effect=mutate), rejected():
            api.pack(path, root / 'mutated')
        assert not (root / 'mutated/artifact').exists()


def test_existing_payload_and_escaping_path_refused():
    api = module()
    with tiny_public_inputs(api) as (root, path, index):
        artifact = api.pack(path, root / 'success')
        assert (artifact / 'experts.bin').read_bytes() == b'abijqr'
        manifest = api.common.load_json(artifact / 'manifest.json')
        assert manifest['payload']['bytes'] == 6
        api.verify_packed(index, manifest, artifact)
        (artifact / 'experts.bin').write_bytes(b'abXjqr')
        with rejected():
            api.verify_packed(index, manifest, artifact)
        manifest['payload']['path'] = '../source'
        with rejected():
            api.verify_packed(index, manifest, artifact)
        with rejected():
            api.pack(path, root / 'success')
        assert (artifact / 'experts.bin').read_bytes() == b'abXjqr'


def test_replaced_payload_path_cannot_publish_detached_descriptor():
    api = module()
    for mutation in ('path', 'contents'):
        with tiny_public_inputs(api) as (root, path, _):
            real_verify = api.verify_copy
            def replace_output(source, destination, original, packed):
                real_verify(source, destination, original, packed)
                payload = Path(destination.name)
                if mutation == 'path':
                    payload.rename(payload.with_name('detached.bin'))
                payload.write_bytes(b'xxxxxx')
            with patch.object(api, 'verify_copy', side_effect=replace_output), rejected():
                api.pack(path, root / 'output')
            assert not (root / 'output/artifact').exists()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
