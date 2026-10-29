"""Source-derived expectations and a single post-timing scratch comparison."""
import hashlib
import importlib.util
from pathlib import Path
import time

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
common=identity.sibling('mellum_storage_common')


def checked_stat(binding: dict,actual: dict) -> None:
    if actual!={k:binding[k] for k in common.STAT_KEYS}:raise ValueError('verification input changed')


def blocks(source,offset: int,length: int):
    source.seek(offset)
    while length:
        block=common.read(source,min(65536,length))
        if not block:raise ValueError('short verification read')
        length-=len(block)
        yield block


def padding(length: int):
    while length:
        count=min(length,65536)
        yield bytes(count)
        length-=count


def expected(payload: dict,cell: dict,packed: dict) -> dict:
    start=time.monotonic_ns();before=common.READ_BYTES
    digest=hashlib.sha256();experts=[]
    with common.opened(Path(payload['path'])) as (source,stat):
        checked_stat(payload,stat)
        for job in cell['source_jobs']:
            expert=hashlib.sha256();length=0
            for span in job['ranges']:
                for block in blocks(source,span['offset'],span['bytes']):
                    digest.update(block);expert.update(block);length+=len(block)
            sha=expert.hexdigest()
            if sha!=packed['experts'][job['expert_id']]['packed_sha256']:raise ValueError('selected source digest differs')
            if length>cell['stride_bytes']:raise ValueError('expert exceeds slot')
            for block in padding(cell['stride_bytes']-length):digest.update(block)
            experts.append(dict(expert_id=job['expert_id'],sha256=sha,bytes=length))
    if len(experts)*cell['stride_bytes']!=cell['file_bytes']:raise ValueError('file geometry differs')
    return dict(sha256=digest.hexdigest(),experts=experts,source=payload,file_bytes=cell['file_bytes'],
        stride_bytes=cell['stride_bytes'],python_read_bytes=common.READ_BYTES-before,start_ns=start,end_ns=time.monotonic_ns())


def verify_scratch(payload: dict,scratch: dict,cell: dict,expected: dict) -> dict:
    start=time.monotonic_ns();before=common.READ_BYTES
    if expected['source']!=payload or expected['file_bytes']!=cell['file_bytes'] or expected['stride_bytes']!=cell['stride_bytes']:
        raise ValueError('expected provenance differs')
    digest=hashlib.sha256();experts=[]
    with common.opened(Path(payload['path'])) as (source,source_stat), common.opened(Path(scratch['path'])) as (target,target_stat):
        checked_stat(payload,source_stat);checked_stat(scratch,target_stat)
        if target_stat['size']!=cell['file_bytes']:raise ValueError('scratch size differs')
        for job in cell['source_jobs']:
            expert=hashlib.sha256();length=0
            for span in job['ranges']:
                for block in blocks(source,span['offset'],span['bytes']):
                    actual=common.read(target,len(block))
                    if actual!=block:raise ValueError('scratch expert differs from source')
                    digest.update(actual);expert.update(actual);length+=len(actual)
            for zeros in padding(cell['stride_bytes']-length):
                actual=common.read(target,len(zeros))
                if actual!=zeros:raise ValueError('scratch padding differs')
                digest.update(actual)
            experts.append(dict(expert_id=job['expert_id'],sha256=expert.hexdigest(),bytes=length))
        if target.tell()!=cell['file_bytes']:raise ValueError('scratch comparison length differs')
    if digest.hexdigest()!=expected['sha256'] or experts!=expected['experts']:raise ValueError('scratch digest differs')
    return dict(sha256=digest.hexdigest(),experts=experts,source=payload,scratch=scratch,
        python_read_bytes=common.READ_BYTES-before,start_ns=start,end_ns=time.monotonic_ns())
