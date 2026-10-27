"""Frozen diagnostic calibration identities and bounded native range plans."""
import importlib.util
from pathlib import Path
import random
import struct

_spec = importlib.util.spec_from_file_location('mellum_capture_identity', Path(__file__).with_name('mellum_capture_identity.py'))
identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity)
common = identity.sibling('mellum_storage_common')
index_tools = identity.sibling('mellum_expert_index')
pack_tools = identity.sibling('mellum_expert_pack')
replay = identity.sibling('mellum_ssd_replay')
TRACE_SHA256 = '75318d1374f8780beb3fb7c5dc880b70afb32786bd976c170d9b7aac0203b2de'
CAPACITY = 3 * 2**30


def uniform_ids() -> list[tuple[int, int]]:
    return [divmod(value, 64) for value in random.Random(20261027).sample(range(1792), 512)]


def lru_decode_misses(trace, capacity_bytes: int) -> list[tuple[int, int]]:
    slots = common.integer(capacity_bytes, trace.header.geometry.expert_bytes) // trace.header.geometry.expert_bytes
    cache = replay.LRU(slots)
    result = []
    for request in trace.requests:
        required = replay.group_ids(request)
        if len(required) > slots:
            raise ValueError('group exceeds atomic cache capacity')
        hits, loads, _, _ = cache.serve(required)
        missing = sorted(set(required) - hits)
        if len(missing) != loads:
            raise ValueError('inconsistent LRU miss accounting')
        if request.phase == 'decode':
            result.extend(missing)
    return result


def select_diagnostic(trace) -> list[tuple[int, int]]:
    h = trace.header
    if (h.schema_version != 2 or h.capture_kind != 'measured' or h.geometry != replay.MELLUM or
            not h.capture_manifest or h.capture_manifest.get('workload', {}).get('split') != 'tuning'):
        raise ValueError('v2 measured tuning Mellum trace required')
    misses = lru_decode_misses(trace, CAPACITY)
    if len(misses) < 512:
        raise ValueError('fewer than 512 diagnostic decode misses')
    return misses[:512]


def diagnostic_ids(trace_path: Path) -> list[tuple[int, int]]:
    before = identity.file_identity(trace_path)
    if before['sha256'] != TRACE_SHA256:
        raise ValueError('diagnostic trace differs from approved digest')
    with common.opened(trace_path) as (source, stat):
        if stat != {key:before[key] for key in common.STAT_KEYS}:
            raise ValueError('diagnostic trace changed')
        trace = replay.load_trace(line.decode('utf-8') for line in source)
    if identity.file_identity(trace_path) != before:
        raise ValueError('diagnostic trace changed')
    return select_diagnostic(trace)


def make_jobs(index: dict, ids: list[tuple[int, int]], layout: str) -> list[dict]:
    if layout == 'original':
        index_tools.validate_index(index)
    elif layout == 'packed':
        pack_tools.validate_packed(index)
    else:
        raise ValueError('unknown storage layout')
    jobs = []
    for job_id, (layer, expert) in enumerate(ids):
        common.integer(layer, 0, 27)
        common.integer(expert, 0, 63)
        entry = index['experts'][layer * 64 + expert]
        dest, ranges = 0, []
        for span in entry['ranges']:
            ranges.append(dict(offset=span['offset'], bytes=span['bytes'], dest=dest))
            dest += span['bytes']
        jobs.append(dict(job_id=job_id, expert_id=layer * 64 + expert, ranges=ranges))
    return jobs


def encode_plan(jobs: list[dict], file_bytes: int) -> bytes:
    common.integer(file_bytes, 1)
    common.integer(len(jobs), 1, 4096)
    records = []
    total, maximum = 0, 0
    try:
        for i, job in enumerate(jobs):
            common.integer(job['job_id'], i, i)
            expert = common.integer(job['expert_id'], 0, 1791)
            common.integer(len(job['ranges']), 1, 3)
            dest = 0
            for span in job['ranges']:
                offset = common.integer(span['offset'])
                length = common.integer(span['bytes'], 1, 6580224)
                common.integer(span['dest'], dest, dest)
                if offset > file_bytes or length > file_bytes - offset:
                    raise ValueError('read span exceeds file')
                records.append(struct.pack('<IIQQQ', i, expert, offset, length, dest))
                dest += length
            common.integer(dest, 1, 6580224)
            total += dest
            maximum = max(maximum, dest)
    except (KeyError, TypeError) as exc:
        raise ValueError('malformed logical read jobs') from exc
    header = struct.pack('<8sIIIIQQQQQ', b'MLREAD01', 1, 64, len(jobs), len(records),
                         file_bytes, total, maximum, 0, 0)
    return header + b''.join(records)
