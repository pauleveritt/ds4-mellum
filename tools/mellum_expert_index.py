"""Exact Mellum Q8_0 or MXFP4 expert ranges, structurally checked and source-bound."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('mellum_capture_identity', Path(__file__).with_name('mellum_capture_identity.py'))
identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity)
gguf = identity.sibling('mellum_gguf')
common = identity.sibling('mellum_storage_common')
PART = 2193408   # one Q8_0 gate, up or down slice
GEOMETRY = dict(layers=28, experts_per_layer=64, expert_bytes=6580224)   # Q8_0, the default
PARTS = {8: PART, 39: 1096704}   # GGUF type -> slice bytes; MXFP4 (39) is S6's 8 GB artifact
GEOMETRIES = {kind: dict(layers=28, experts_per_layer=64, expert_bytes=3 * part) for kind, part in PARTS.items()}


def expert_kind(directory: gguf.Directory) -> int:
    """The routed experts' GGUF type, from blk.0.ffn_gate_exps; every expert tensor must share it."""
    first = next((t for t in directory.tensors if t.name == 'blk.0.ffn_gate_exps.weight'), None)
    if first is None or first.type not in PARTS:
        raise ValueError('missing/unsupported Mellum expert tensor: blk.0.ffn_gate_exps.weight')
    return first.type


def geometry_for(directory: gguf.Directory) -> dict:
    return dict(GEOMETRIES[expert_kind(directory)])


def part_for(geometry: dict) -> int:
    for kind, known in GEOMETRIES.items():
        if geometry == known:
            return PARTS[kind]
    raise ValueError('unsupported expert index/geometry')
ROLES = ('gate', 'up', 'down')


def expert_entries(directory: gguf.Directory) -> list[dict]:
    if directory.architecture != 'mellum':
        raise ValueError('Mellum directory required')
    tensors = {tensor.name: tensor for tensor in directory.tensors}
    if len(tensors) != len(directory.tensors):
        raise ValueError('duplicate tensor name')
    kind = expert_kind(directory)
    part = PARTS[kind]
    selected = set()
    entries = []
    for layer in range(28):
        parts = []
        for role in ROLES:
            name = f'blk.{layer}.ffn_{role}_exps.weight'
            tensor = tensors.get(name)
            dims = (896, 2304, 64) if role == 'down' else (2304, 896, 64)
            if tensor is None or tensor.dims != dims or tensor.type != kind or tensor.bytes != part * 64:
                raise ValueError(f'missing/unsupported Mellum expert tensor: {name}')
            selected.add(name)
            parts.append(tensor)
        for expert in range(64):
            entries.append(dict(layer=layer, expert=expert, ranges=[dict(role=role,
                offset=tensor.offset + expert * part, bytes=part) for role, tensor in zip(ROLES, parts, strict=True)]))
    if any('_exps.' in name and name not in selected for name in tensors):
        raise ValueError('unexpected expert tensor')
    return entries


def validate_index(value: dict) -> dict:
    try:
        if common.integer(value['schema_version']) != 1:
            raise ValueError('unsupported expert index/geometry')
        part = part_for(value['geometry'])
        for key, number in value['geometry'].items():
            common.integer(number, number, number)
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
                size = common.integer(span['bytes'], part, part)
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
    read_start = common.READ_BYTES
    model = model.absolute()
    with common.opened(model) as (source, before):
        directory = gguf.read_directory(source, before['size'])
        common.check_open(source, model, before)
        entries = expert_entries(directory)
        prefix = common.hash_stream(source, directory.data_start)
        digest = common.hash_stream(source)
        common.check_open(source, model, before)
    result = dict(schema_version=1, geometry=geometry_for(directory),
                  source=dict(path=str(model), sha256=digest, **before),
                  directory=dict(data_start=directory.data_start, alignment=directory.alignment, sha256=prefix),
                  tool=common.file_identity(Path(__file__)), source_identity=common.source_identity(common.ROOT),
                  experts=entries)
    result['untimed_python_read_bytes'] = common.READ_BYTES - read_start
    return validate_index(result)


def verify_index_source(index: dict) -> None:
    validate_index(index)
    actual = build_index(Path(index['source']['path']))
    for key in ('source', 'directory', 'geometry', 'experts'):
        if actual[key] != index[key]:
            raise ValueError(f'expert index differs from actual source: {key}')
