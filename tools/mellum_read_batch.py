"""Freeze paired storage reads and validate complete native evidence."""
import importlib.util
import math
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import os

_spec = importlib.util.spec_from_file_location('mellum_capture_identity', Path(__file__).with_name('mellum_capture_identity.py'))
identity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity)
common = identity.sibling('mellum_storage_common')
index_tools = identity.sibling('mellum_expert_index')
pack_tools = identity.sibling('mellum_expert_pack')
plans = identity.sibling('mellum_read_plan')
BUDGET = 256 * 2**30
COMPILER_COMMAND = 'cc -std=c99 -O2 -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE -Wall -Wextra -Werror -pthread tools/mellum_read_bench.c tools/mellum_read_plan.c tools/mellum_read_io.c tools/mellum_read_run.c -o tools/mellum_read_bench'


def frozen_runs() -> list[dict]:
    return [dict(plan=plan,mode=mode,workers=workers,repetition=rep,layout=layout)
            for plan in ('uniform','diagnostic') for mode in ('buffered','avoid')
            for workers in (1,2,4) for rep in (0,1,2)
            for layout in (('original','packed') if rep % 2 == 0 else ('packed','original'))]


def binding(path: Path) -> dict:
    return dict(path=str(path.absolute()), **identity.file_identity(path))


def check_binding(value: dict, *, full: bool = True) -> None:
    path = Path(value['path'])
    actual = identity.file_identity(path) if full else identity.file_stat(path)
    if any(value.get(key) != item for key,item in actual.items()):
        raise ValueError(f'frozen input changed: {path}')


def prepare_batch(index_path: Path, packed_dir: Path, trace_path: Path, binary: Path, container: Path) -> Path:
    index = index_tools.validate_index(common.load_json(index_path))
    manifest = pack_tools.validate_packed(common.load_json(packed_dir / 'manifest.json'))
    index_tools.verify_index_source(index)
    pack_tools.verify_packed(index, manifest, packed_dir)
    ids = dict(uniform=plans.uniform_ids(), diagnostic=plans.diagnostic_ids(trace_path))
    inputs = dict(index=binding(index_path), packed_manifest=binding(packed_dir/'manifest.json'),
                  model=binding(Path(index['source']['path'])), payload=binding(packed_dir/'experts.bin'),
                  trace=binding(trace_path), binary=binding(binary))
    source = identity.source_identity(common.ROOT)
    def build(stage: Path) -> None:
        definitions = {}
        for name, logical_ids in ids.items():
            for layout, value in (('original',index),('packed',manifest)):
                key = f'{name}-{layout}'
                jobs = plans.make_jobs(value,logical_ids,layout)
                file_bytes = inputs['model' if layout == 'original' else 'payload']['size']
                encoded = plans.encode_plan(jobs,file_bytes)
                wire = stage / f'{key}.bin'
                with wire.open('xb') as stream:
                    if stream.write(encoded) != len(encoded):
                        raise OSError('short wire-plan write')
                    stream.flush()
                common.write_json(stage/f'{key}.json',dict(ids=logical_ids,jobs=jobs))
                definitions[key] = dict(wire=wire.name,sha256=identity.file_identity(wire)['sha256'],
                                        jobs=f'{key}.json', jobs_sha256=identity.file_identity(stage/f'{key}.json')['sha256'],
                                        logical_bytes=sum(sum(s['bytes'] for s in j['ranges']) for j in jobs))
        runs = frozen_runs()
        total = sum(definitions[f"{run['plan']}-{run['layout']}"]['logical_bytes'] for run in runs)
        common.integer(total,1,BUDGET-1)
        for value in inputs.values():
            check_binding(value,full=False)
        value = dict(schema_version=1,kind='storage-read-batch',split='tuning',inputs=inputs,
                     source_identity=source,compiler_command=COMPILER_COMMAND,
                     compiler_provenance='declared fixed Makefile recipe; binary and source independently hashed',
                     runs=runs,plans=definitions,timed_logical_bytes=total,physical_io_verified=False,
                     diagnostic=dict(approved_trace_sha256=plans.TRACE_SHA256,capacity_bytes=plans.CAPACITY,
                                     effective_slots=489,selection='full cold prefix LRU; first 512 sorted decode misses; repeats retained'),
                     cache_modes=dict(buffered='explicitly prewarmed jobs outside timing',avoid='F_NOCACHE=1 and F_RDAHEAD=0 required on macOS; physical I/O unverified'))
        common.write_json(stage/'manifest.json',value)
    return common.publication.publish(container,build)


def verify_batch(batch_dir: Path) -> tuple[dict, dict]:
    batch = common.load_json(batch_dir/'manifest.json')
    if (common.integer(batch['schema_version']) != 1 or batch['kind'] != 'storage-read-batch' or
            batch['split'] != 'tuning' or batch['runs'] != frozen_runs() or batch['physical_io_verified'] is not False):
        raise ValueError('unsupported frozen batch/protocol')
    if batch['source_identity']['source_state_id'] != identity.source_identity(common.ROOT)['source_state_id']:
        raise ValueError('measurement source changed since preparation')
    for value in batch['inputs'].values():
        check_binding(value)
    index = index_tools.validate_index(common.load_json(Path(batch['inputs']['index']['path'])))
    packed_path = Path(batch['inputs']['packed_manifest']['path'])
    packed = pack_tools.validate_packed(common.load_json(packed_path))
    index_tools.verify_index_source(index)
    pack_tools.verify_packed(index,packed,packed_path.parent)
    ids = dict(uniform=plans.uniform_ids(),diagnostic=plans.diagnostic_ids(Path(batch['inputs']['trace']['path'])))
    expected_keys = {f'{name}-{layout}' for name in ids for layout in ('original','packed')}
    if set(batch['plans']) != expected_keys:
        raise ValueError('incomplete plan matrix')
    jobs_by_key = {}
    total = 0
    for key, definition in batch['plans'].items():
        name,layout = key.split('-')
        if definition['wire'] != f'{key}.bin' or definition['jobs'] != f'{key}.json':
            raise ValueError('escaping or substituted plan path')
        for field,hash_key in (('wire','sha256'),('jobs','jobs_sha256')):
            if identity.file_identity(batch_dir/definition[field])['sha256'] != common.digest(definition[hash_key]):
                raise ValueError('frozen plan changed')
        logical = common.load_json(batch_dir/definition['jobs'])
        jobs = plans.make_jobs(index if layout == 'original' else packed,ids[name],layout)
        size = batch['inputs']['model' if layout == 'original' else 'payload']['size']
        if logical != dict(ids=[list(pair) for pair in ids[name]],jobs=jobs) or (batch_dir/definition['wire']).read_bytes() != plans.encode_plan(jobs,size):
            raise ValueError('frozen jobs differ from checked indexes/trace')
        count = sum(sum(span['bytes'] for span in job['ranges']) for job in jobs)
        common.integer(definition['logical_bytes'],count,count)
        jobs_by_key[key] = jobs
    for run in batch['runs']:
        total += batch['plans'][f"{run['plan']}-{run['layout']}"]['logical_bytes']
    common.integer(batch['timed_logical_bytes'],total,total)
    common.integer(total,1,BUDGET-1)
    return batch,jobs_by_key
