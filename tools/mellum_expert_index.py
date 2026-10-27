"""Exact Mellum Q8 expert ranges, structurally checked and source-bound."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('mellum_capture_identity', Path(__file__).with_name('mellum_capture_identity.py'))
identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity)
gguf = identity.sibling('mellum_gguf')
common = identity.sibling('mellum_storage_common')
PART = 2193408
GEOMETRY = dict(layers=28, experts_per_layer=64, expert_bytes=6580224)
ROLES = ('gate', 'up', 'down')


def expert_entries(directory: gguf.Directory) -> list[dict]:
    if directory.architecture != 'mellum':
        raise ValueError('Mellum directory required')
    tensors = {tensor.name: tensor for tensor in directory.tensors}
    if len(tensors) != len(directory.tensors):
        raise ValueError('duplicate tensor name')
    selected = set()
    entries = []
    for layer in range(28):
        parts = []
        for role in ROLES:
            name = f'blk.{layer}.ffn_{role}_exps.weight'
            tensor = tensors.get(name)
            dims = (896, 2304, 64) if role == 'down' else (2304, 896, 64)
            if tensor is None or tensor.dims != dims or tensor.type != 8 or tensor.bytes != PART * 64:
                raise ValueError(f'missing/unsupported Mellum expert tensor: {name}')
            selected.add(name)
            parts.append(tensor)
        for expert in range(64):
            entries.append(dict(layer=layer, expert=expert, ranges=[dict(role=role,
                offset=tensor.offset + expert * PART, bytes=PART) for role, tensor in zip(ROLES, parts, strict=True)]))
    if any('_exps.' in name and name not in selected for name in tensors):
        raise ValueError('unexpected expert tensor')
    return entries


def validate_index(value: dict) -> dict:
    try:
        if common.integer(value['schema_version']) != 1 or value['geometry'] != GEOMETRY:
            raise ValueError('unsupported expert index/geometry')
        for key, number in GEOMETRY.items():
            common.integer(value['geometry'][key], number, number)
        source, directory = value['source'], value['directory']
        if not isinstance(source['path'], str) or not Path(source['path']).is_absolute():
            raise ValueError('absolute model path required')
        for key in common.STAT_KEYS:
            common.integer(source[key])
        common.digest(source['sha256'])
        common.digest(directory['sha256'])
        start = common.integer(directory['data_start'], 24, 32 * 1024**2)
        alignment = common.integer(directory['alignment'], 1, 1024**2)
        if alignment & (alignment - 1) or start % alignment:
            raise ValueError('invalid index directory alignment')
        common.digest(value['tool']['sha256'])
        common.digest(value['source_identity']['source_state_id'])
        if len(value['experts']) != 1792:
            raise ValueError('incomplete expert index')
        spans = []
        for i, entry in enumerate(value['experts']):
            common.integer(entry['layer'], i // 64, i // 64)
            common.integer(entry['expert'], i % 64, i % 64)
            if len(entry['ranges']) != 3:
                raise ValueError('three expert roles required')
            for role, span in zip(ROLES, entry['ranges'], strict=True):
                if span['role'] != role:
                    raise ValueError('incorrect expert role order')
                offset = common.integer(span['offset'], start, source['size'])
                size = common.integer(span['bytes'], PART, PART)
                if offset % 32 or offset + size > source['size']:
                    raise ValueError('out-of-file/unaligned expert span')
                spans.append((offset, offset + size))
        end = start
        for offset, stop in sorted(spans):
            if offset < end:
                raise ValueError('overlapping expert ranges')
            end = stop
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('malformed expert index') from exc
    return value


def build_index(model: Path) -> dict:
    model = model.absolute()
    with common.opened(model) as (source, before):
        directory = gguf.read_directory(source, before['size'])
        common.check_open(source, model, before)
        entries = expert_entries(directory)
        prefix = common.hash_stream(source, directory.data_start)
        digest = common.hash_stream(source)
        common.check_open(source, model, before)
    result = dict(schema_version=1, geometry=dict(GEOMETRY),
                  source=dict(path=str(model), sha256=digest, **before),
                  directory=dict(data_start=directory.data_start, alignment=directory.alignment, sha256=prefix),
                  tool=common.file_identity(Path(__file__)), source_identity=common.source_identity(common.ROOT),
                  experts=entries)
    return validate_index(result)


def verify_index_source(index: dict) -> None:
    validate_index(index)
    actual = build_index(Path(index['source']['path']))
    for key in ('source', 'directory', 'geometry', 'experts'):
        if actual[key] != index[key]:
            raise ValueError(f'expert index differs from actual source: {key}')
