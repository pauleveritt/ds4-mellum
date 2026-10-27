"""GGUF tensor spans and expert indexes must bind to actual source bytes."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PART = 2193408


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
def rejected():
    try:
        yield
    except (ValueError, OSError):
        pass
    else:
        raise AssertionError('invalid index/directory accepted')


def string(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


def gguf(tensors=None, metadata=None, alignment=64):
    tensors = tensors if tensors is not None else [('a', (32,), 8, 0), ('b', (2,), 0, 64)]
    metadata = metadata if metadata is not None else [('general.architecture', 8, string('mellum')),
                                                      ('general.alignment', 4, struct.pack('<I', alignment))]
    data = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for name, kind, value in metadata:
        data += string(name) + struct.pack('<I', kind) + value
    for name, dims, kind, offset in tensors:
        data += string(name) + struct.pack('<I', len(dims)) + struct.pack('<' + 'Q' * len(dims), *dims)
        data += struct.pack('<IQ', kind, offset)
    start = (len(data) + alignment - 1) // alignment * alignment
    return data.ljust(start, b'\0') + b'x' * 128


def fabricated_directory():
    api = module('mellum_gguf')
    tensors = []
    offset = 1024
    for layer in range(28):
        for role in ('gate', 'up', 'down'):
            dims = (896, 2304, 64) if role == 'down' else (2304, 896, 64)
            tensors.append(api.Tensor(name=f'blk.{layer}.ffn_{role}_exps.weight', dims=dims,
                                      type=8, offset=offset, bytes=PART * 64))
            offset += PART * 64
    return api.Directory(data_start=1024, alignment=32, tensors=tuple(tensors), architecture='mellum')


def fabricated_index():
    api = module('mellum_expert_index')
    directory = fabricated_directory()
    return dict(schema_version=1, source=dict(path='/missing/model.gguf', dev=1, ino=1,
                size=1024 + 11791761408, mtime_ns=1, ctime_ns=1, sha256='a' * 64),
                directory=dict(data_start=1024, alignment=32, sha256='b' * 64),
                geometry=dict(layers=28, experts_per_layer=64, expert_bytes=6580224),
                tool={'sha256':'c' * 64}, source_identity={'source_state_id':'d' * 64},
                experts=api.expert_entries(directory))


def test_directory_padding_types_and_nonexpert_overlap():
    api = module('mellum_gguf')
    raw = gguf()
    directory = api.read_directory(io.BytesIO(raw), len(raw))
    assert directory.data_start == 192
    assert directory.alignment == 64
    assert directory.tensors[0].offset == 192
    assert directory.tensors[0].bytes == 34
    assert directory.tensors[1].offset == 256
    raw = gguf([('expert', (32,), 8, 0), ('fixed', (32,), 1, 0)])
    with rejected():
        api.read_directory(io.BytesIO(raw), len(raw))
    raw = gguf([('f16', (2,), 1, 0)])
    assert api.read_directory(io.BytesIO(raw), len(raw)).tensors[0].bytes == 4
    with rejected():
        api.read_directory(io.BytesIO(raw[:-128]), len(raw) - 128)


def test_duplicate_names_bad_alignment_shards_and_overflow():
    api = module('mellum_gguf')
    cases = [gguf([('a', (32,), 8, 0), ('a', (32,), 8, 64)]),
             gguf(alignment=3), gguf([('a', (2**63, 4), 0, 0)]),
             gguf([('a', (31,), 8, 0)]), gguf([('a', (32,), 99, 0)]),
             gguf([('a', (32,), 8, 1)]), gguf([('a', (), 0, 0)]),
             gguf(metadata=[('general.architecture',8,string('mellum')),
                            ('split.count',2,struct.pack('<H',2))]),
             gguf(metadata=[('general.architecture',8,string('mellum'))] * 2),
             b'GGUF' + struct.pack('<IQQ', 3, 4097, 0),
             b'GGUF' + struct.pack('<IQQ', 3, 0, 65537)]
    for raw in cases:
        with rejected():
            api.read_directory(io.BytesIO(raw), len(raw))


def test_expert_first_last_offsets_and_role_order():
    api = module('mellum_expert_index')
    directory = fabricated_directory()
    entries = api.expert_entries(directory)
    assert len(entries) == 1792
    assert (entries[0]['layer'], entries[0]['expert']) == (0, 0)
    assert entries[0]['ranges'] == [dict(role='gate', offset=1024, bytes=PART),
        dict(role='up', offset=140379136, bytes=PART), dict(role='down', offset=280757248, bytes=PART)]
    assert (entries[-1]['layer'], entries[-1]['expert']) == (27, 63)
    assert entries[-1]['ranges'][2]['offset'] == directory.tensors[-1].offset + 63 * PART


def test_wrong_shape_type_missing_or_duplicate_role():
    api = module('mellum_expert_index')
    d = fabricated_directory()
    for tensors in (d.tensors[:-1], d.tensors + (d.tensors[0],),
                    (replace(d.tensors[0], dims=(32, 896, 64)),) + d.tensors[1:],
                    (replace(d.tensors[0], type=1),) + d.tensors[1:]):
        with rejected():
            api.expert_entries(replace(d, tensors=tensors))


def test_index_reload_rejects_forged_geometry_or_spans():
    api = module('mellum_expert_index')
    value = fabricated_index()
    assert api.validate_index(value) == value
    cases = []
    for key, bad in (('layers', True), ('expert_bytes', 1)):
        altered = deepcopy(value)
        altered['geometry'][key] = bad
        cases.append(altered)
    for field, bad in (('offset', -1), ('bytes', 1), ('role','up')):
        altered = deepcopy(value)
        altered['experts'][0]['ranges'][0][field] = bad
        cases.append(altered)
    altered = deepcopy(value)
    altered['experts'][1] = altered['experts'][0]
    cases.append(altered)
    altered = deepcopy(value)
    altered['experts'][1]['ranges'][0]['offset'] = altered['experts'][0]['ranges'][0]['offset']
    cases.append(altered)
    for altered in cases:
        with rejected():
            api.validate_index(altered)


def test_correct_model_hash_cannot_authorize_forged_offsets():
    api = module('mellum_expert_index')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model'
        path.write_bytes(gguf())
        value = fabricated_index()
        value['source']['path'] = str(path)
        value['source']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Even a real tiny file's hash cannot authorize an invented full index.
        with rejected():
            api.verify_index_source(value)


def test_source_changes_during_inspection_fail():
    api = module('mellum_expert_index')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model'
        path.write_bytes(gguf())
        real = api.gguf.read_directory
        def changing(source, size):
            result = real(source, size)
            with path.open('ab') as stream:
                stream.write(b'changed')
            return result
        with patch.object(api.gguf, 'read_directory', side_effect=changing), rejected():
            api.build_index(path)
        output = Path(directory) / 'output'
        assert (ROOT / 'tools/mellum_storage.py').is_file(), 'storage CLI is not implemented'
        process = subprocess.run([sys.executable, '-I', str(ROOT / 'tools/mellum_storage.py'),
                                  'index', '--model', str(path), '--output', str(output)], capture_output=True)
        assert process.returncode != 0
        assert not (output / 'artifact').exists()


def test_bounded_json_refuses_duplicates_nonfinite_and_oversize():
    api = module('mellum_storage_common')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'value.json'
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b' ' * (8 * 1024**2 + 1)):
            path.write_bytes(raw)
            with rejected():
                api.load_json(path)
        path.write_bytes(b'{"a":1}')
        assert api.load_json(path) == {'a': 1}


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
