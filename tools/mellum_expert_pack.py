"""Exact-byte, bounded expert concatenation and independent copy verification."""
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
from typing import BinaryIO

_spec = importlib.util.spec_from_file_location('mellum_capture_identity', Path(__file__).with_name('mellum_capture_identity.py'))
identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity)
common = identity.sibling('mellum_storage_common')
index_tools = identity.sibling('mellum_expert_index')


def chunks(source: BinaryIO, spans: list[dict]):
    total = 0
    for span in spans:
        offset = common.integer(span['offset'])
        remaining = common.integer(span['bytes'], 1, 6580224)
        total += remaining
        if total > 6580224:
            raise ValueError('expert exceeds bounded buffer budget')
        source.seek(offset)
        while remaining:
            size = min(65536, remaining)
            block = common.read(source, size)
            if len(block) != size:
                raise ValueError('short expert read')
            yield block
            remaining -= size


def copy_entries(source: BinaryIO, destination: BinaryIO, entries: list[dict]) -> list[dict]:
    result = []
    offset = 0
    for entry in entries:
        digest = hashlib.sha256()
        size = 0
        for block in chunks(source, entry['ranges']):
            if destination.write(block) != len(block):
                raise OSError('short packed write')
            digest.update(block)
            size += len(block)
        result.append(dict(layer=entry['layer'], expert=entry['expert'],
                           ranges=[dict(role='packed', offset=offset, bytes=size)],
                           source_sha256=digest.hexdigest(), packed_sha256=digest.hexdigest()))
        offset += size
    return result


def verify_copy(source: BinaryIO, destination: BinaryIO, original_entries: list[dict], packed_entries: list[dict]) -> None:
    if len(original_entries) != len(packed_entries):
        raise ValueError('copy entry count differs')
    for original, packed in zip(original_entries, packed_entries, strict=True):
        digest = hashlib.sha256()
        destination.seek(packed['ranges'][0]['offset'])
        size = 0
        for block in chunks(source, original['ranges']):
            actual = common.read(destination, len(block))
            if actual != block:
                raise ValueError('packed bytes differ from source')
            digest.update(actual)
            size += len(actual)
        if (size != packed['ranges'][0]['bytes'] or digest.hexdigest() != packed['source_sha256'] or
                digest.hexdigest() != packed['packed_sha256']):
            raise ValueError('copy digest/length differs')


def validate_packed(value: dict) -> dict:
    try:
        if common.integer(value['schema_version']) != 1 or value['kind'] != 'packed-experts' or value['geometry'] != index_tools.GEOMETRY:
            raise ValueError('unsupported packed manifest')
        for key, number in index_tools.GEOMETRY.items():
            common.integer(value['geometry'][key], number, number)
        payload = value['payload']
        if payload['path'] != 'experts.bin' or len(value['experts']) != 1792:
            raise ValueError('invalid packed path/count')
        common.digest(payload['sha256'])
        common.digest(value['index']['sha256'])
        common.digest(value['model']['sha256'])
        offset = 0
        for i, entry in enumerate(value['experts']):
            common.integer(entry['layer'], i // 64, i // 64)
            common.integer(entry['expert'], i % 64, i % 64)
            if len(entry['ranges']) != 1 or entry['ranges'][0]['role'] != 'packed':
                raise ValueError('one contiguous range required')
            span = entry['ranges'][0]
            common.integer(span['offset'], offset, offset)
            offset += common.integer(span['bytes'], 6580224, 6580224)
            if common.digest(entry['source_sha256']) != common.digest(entry['packed_sha256']):
                raise ValueError('expert copy digests differ')
        common.integer(payload['bytes'], offset, offset)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('malformed packed manifest') from exc
    return value


def pack(index_path: Path, container: Path) -> Path:
    read_start = common.READ_BYTES
    index_file = common.file_identity(index_path)
    index = index_tools.validate_index(common.load_json(index_path))
    index_tools.verify_index_source(index)
    size = sum(sum(span['bytes'] for span in entry['ranges']) for entry in index['experts'])
    if shutil.disk_usage(container.parent).free < size + 2 * 1024**3:
        raise ValueError('insufficient space for payload plus 2 GiB margin')
    model = Path(index['source']['path'])
    expected = {key:index['source'][key] for key in common.STAT_KEYS}

    def build(stage: Path) -> None:
        payload = stage / 'experts.bin'
        with common.opened(model) as (source, before):
            if before != expected:
                raise ValueError('source changed since index verification')
            with payload.open('x+b') as destination:
                entries = copy_entries(source, destination, index['experts'])
                destination.flush()
                verify_copy(source, destination, index['experts'], entries)
                digest = common.hash_stream(destination)
                destination.seek(0, 2)
                if destination.tell() != size:
                    raise ValueError('packed size mismatch')
                destination.flush()
                packed_stat = common.stat_value(os.fstat(destination.fileno()))
                common.check_open(destination, payload, packed_stat)
            if identity.file_stat(payload) != packed_stat:
                raise ValueError('packed payload path changed during close')
        if common.file_identity(index_path) != index_file:
            raise ValueError('source index changed during packing')
        manifest = dict(schema_version=1, kind='packed-experts', geometry=index['geometry'],
                        model=index['source'], index=dict(path=str(index_path.absolute()), **index_file),
                        payload=dict(path='experts.bin', bytes=size, sha256=digest), experts=entries,
                        preparation_bytes=index['source']['size'] + size,
                        verification_bytes=3 * size,
                        tool=common.file_identity(Path(__file__)), source_identity=common.source_identity(common.ROOT))
        manifest['untimed_python_read_bytes'] = common.READ_BYTES - read_start
        common.write_json(stage / 'manifest.json', validate_packed(manifest))
    return common.publication.publish(container, build)


def verify_packed(index: dict, manifest: dict, packed_dir: Path) -> None:
    validate_packed(manifest)
    index_tools.verify_index_source(index)
    if manifest['model'] != index['source']:
        raise ValueError('packed model/index mismatch')
    path = Path(manifest['index']['path'])
    expected_index = {key:manifest['index'][key] for key in (*common.STAT_KEYS, 'sha256')}
    if common.file_identity(path) != expected_index or common.load_json(path) != index:
        raise ValueError('packed index linkage differs')
    if manifest['payload']['path'] != 'experts.bin':
        raise ValueError('packed payload must remain inside artifact')
    payload = packed_dir / 'experts.bin'
    expected = {key:index['source'][key] for key in common.STAT_KEYS}
    with common.opened(Path(index['source']['path'])) as (source, before), common.opened(payload) as (destination, packed_stat):
        if before != expected or packed_stat['size'] != manifest['payload']['bytes']:
            raise ValueError('source/packed identity differs')
        verify_copy(source, destination, index['experts'], manifest['experts'])
        if common.hash_stream(destination) != manifest['payload']['sha256']:
            raise ValueError('packed payload hash differs')
