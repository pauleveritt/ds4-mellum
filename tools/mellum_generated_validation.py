"""Artifact-only generated receipt gates before export and held-out analysis."""
from array import array
from dataclasses import asdict
import hashlib
import importlib.util
import io
import math
from pathlib import Path

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
inputs=identity.sibling('mellum_generated_inputs')
common=inputs.common
metadata=identity.sibling('mellum_capture_metadata')
schedule=identity.sibling('mellum_generated_schedule')
exporter=identity.sibling('mellum_routes_to_trace')
analysis=identity.sibling('mellum_generated_analysis')


def native_result(raw: Path,prepared: Path) -> tuple[dict,list[int],dict]:
    result=metadata.read_json(raw/'result.json',16384,exporter)
    fields={'complete','parity','counts','stop_classes','host_separator_tokens','response0_end','response1_start','tokens'}
    if set(result)!=fields or result['complete'] is not True or result['parity'] is not True:raise ValueError('native parity/completion required')
    for key in ('counts','stop_classes'):
        if not isinstance(result[key],list) or len(result[key])!=2:raise ValueError('invalid turn receipt')
    for count,reason,cap in zip(result['counts'],result['stop_classes'],(4096,1024),strict=True):
        common.integer(count,1,cap);common.integer(reason,0,2)
        if reason==0 and count!=cap:raise ValueError('short response without stop')
    tokens=schedule.read_tokens(raw/'tokens.txt');ops=schedule.read_operations(raw/'operations.txt')
    initial=schedule.read_tokens(prepared/'initial.tokens');append=schedule.read_tokens(prepared/'append.tokens')
    separator=schedule.read_tokens(prepared/('newline.tokens' if result['stop_classes'][0] else 'close.tokens'))
    end=len(initial)+result['counts'][0];start=end+len(separator)+len(append)
    for key,expected in dict(host_separator_tokens=len(separator),response0_end=end,response1_start=start,tokens=len(tokens)).items():
        if type(result[key]) is not int or result[key]!=expected:raise ValueError('native boundary differs')
    if len(tokens)!=start+result['counts'][1] or tokens[:len(initial)]!=initial or tokens[end:start]!=separator+append:
        raise ValueError('consumed native prefix differs')
    expected=[('S',0,len(initial))]+[('E',1,t) for t in tokens[len(initial):end]]+[('S',2,start)]+[('E',3,t) for t in tokens[start:]]
    if ops!=expected:raise ValueError('native operation ledger differs')
    for name in (*inputs.TOKEN_FILES,*inputs.RENDERED_FILES):
        if metadata.bounded_read(raw/'recreated'/name,2**20)!=metadata.bounded_read(prepared/name,2**20):raise ValueError('recreated native input differs')
    emitted=metadata.bounded_read(raw/'emitted.txt',512*1024).decode('ascii').splitlines()
    expected_emitted=[(segment,token) for kind,segment,token in ops if kind=='E']
    if len(emitted)!=len(expected_emitted):raise ValueError('emitted count differs')
    last=0
    for row,pair in zip(emitted,expected_emitted,strict=True):
        words=row.split()
        if len(words)!=3 or any(not s.isascii() or not s.isdecimal() for s in words):raise ValueError('invalid emitted series')
        segment,token,stamp=map(int,words)
        if (segment,token)!=pair or not last<=stamp<2**64:raise ValueError('emitted series differs')
        last=stamp
    if (raw/'gold.f32').stat().st_size!=len(ops)*393216:raise ValueError('parity tape length differs')
    with (raw/'gold.f32').open('rb') as stream:
        while block:=stream.read(393216):
            values=array('f');values.frombytes(block)
            if any(not math.isfinite(value) for value in values):raise ValueError('nonfinite parity tape')
    return result,tokens,dict(operations=len(ops),logit_rows=len(ops),floats_per_row=98304)


def header(frozen: dict,host: dict,receipt: dict,raw: Path) -> dict:
    source=frozen['source_identity'];model_id=inputs.MODEL_SHA
    return dict(schema_version=2,capture_kind='measured',model_id=model_id,engine_revision=source['revision'],numerical_mode='release',
        workload_id='ac-demo-phase1-generated-scripted-review',geometry=asdict(analysis.replay.MELLUM),capture_manifest=dict(
            manifest_version=1,capture_id='mellum-generated-phase1-2026-10-28',
            artifacts=dict(model_id=model_id,tokenizer_id=model_id,template_id=source['source_state_id'],
                model_file_stat=frozen['inputs']['model'],source_content_id=frozen['source_content_id']),
            build=dict(engine_revision=source['revision'],recorder_revision=source['revision'],source_state_id=source['source_state_id'],
                binary_sha256=frozen['build']['gpu']['sha256'],source_inputs={k:v['sha256'] for k,v in source['files'].items()},
                forced_rebuild=True,command_log=metadata.bounded_read(Path(frozen['build']['log']['path']),256*1024).decode()),
            execution=dict(numerical_mode='release',requested_kernel='default',device=f"{host['cpu']} / {host['model']} / {host['ram_bytes']} bytes RAM",
                os=host.get('platform','unknown'),context_tokens=receipt['context_tokens'],prefill_chunk_tokens=receipt['cap_tokens'],
                capabilities=receipt['capabilities'],capture_schedule='generated-two-turn-v1',host=inputs.host.serial(host)),
            generation=dict(mode='generated',settings=inputs.SETTINGS),
            workload=dict(protocol_id='generated-two-turn-v1',conversation_id='ac-demo-phase1-1',repository_id='ac-demo',split='evaluation',
                continuation_kind='scripted-user-review; generated code unexecuted',initial_policy_sha256=frozen['policy']['sha256'])))


def validate(prepared: Path,raw: Path,frozen: dict,progress: dict) -> dict:
    files=list(raw.rglob('*'))
    if any(p.is_symlink() for p in files) or sum(p.stat().st_size for p in files if p.is_file())>=4*2**30:
        raise ValueError('retained native outputs exceed bound or contain symlink')
    native,tokens,parity=native_result(raw,prepared)
    receipt=metadata.read_receipt(raw/'runtime.json',exporter)
    if receipt['context_tokens']!=8192 or receipt['cap_tokens']>1024:raise ValueError('runtime context/chunk differs')
    summary=schedule.check_schedule(raw/'owned.routes',raw/'operations.txt',tokens,receipt['cap_tokens'])
    profiles=metadata.read_profiles(raw/'owned.profiles')
    common.write_json(raw/'metadata.json',dict(header=header(frozen,progress['host'],receipt,raw),segments=analysis.SEGMENTS,execution_profiles=profiles))
    exporter.export_trace(raw/'owned.routes',raw/'metadata.json',raw/'trace.jsonl')
    with (raw/'trace.jsonl').open() as stream:trace=analysis.replay.load_trace(stream)
    report=analysis.analyze(trace,raw/'operations.txt')
    common.write_json(raw/'analysis.json',report)
    return dict(schema_version=1,kind='generated-phase1-report',complete=True,native=native,parity=parity,spool=summary,
        analysis=report,preflight=progress,source=frozen['source_identity'],policy=frozen['policy'],
        trace_sha256=common.file_identity(raw/'trace.jsonl')['sha256'],
        truncation=[reason==0 for reason in native['stop_classes']],
        timing='emitted monotonic timestamps include recorder and logit snapshot overhead; parity is separate; no SSD inference',
        files={str(p.relative_to(raw)):{k:v for k,v in common.file_identity(p).items() if k in ('size','sha256')} for p in raw.rglob('*') if p.is_file()})
