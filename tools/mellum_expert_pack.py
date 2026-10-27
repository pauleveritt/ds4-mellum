"""Exact-byte, bounded expert concatenation and independent copy verification."""
import hashlib
import importlib.util
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
            block = source.read(size)
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
            actual = destination.read(len(block))
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
