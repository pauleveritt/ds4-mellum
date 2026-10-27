"""One reserved CPU-only fresh-file storage-service attempt."""
import importlib.util
import json
import os
from pathlib import Path
import shutil

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
manifest=identity.sibling('mellum_service_plan')
verifier=identity.sibling('mellum_service_verify')
host=identity.sibling('mellum_observe_batch')
common=manifest.common
observer=host.observer
process=host.process
classifier=host.classifier


def geometry(topology: dict) -> dict:
    candidates=[r for r in topology['raw'] if r.get('argv')==['/usr/sbin/ioreg','-a','-l','-r','-c','IOBlockStorageDriver']]
    registry=observer.decoded(observer.one(candidates))
    driver=observer.one([d for d in registry if d.get('IORegistryEntryID')==topology['driver_id']])
    media=observer.one([m for m in driver.get('IORegistryEntryChildren',[]) if m.get('IORegistryEntryID')==topology['media_id']])
    block=common.integer(media.get('Preferred Block Size'),1)
    page=os.sysconf('SC_PAGE_SIZE')
    if page!=16384 or block!=4096:raise ValueError('unsupported storage-service geometry')
    return dict(page_bytes=page,block_bytes=block)


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def space(path: Path) -> None:
    if free_bytes(path)<32*2**30:raise ValueError('storage service requires 32 GiB free')


def stat_binding(path: Path) -> dict:
    return dict(path=str(path.absolute()),**identity.file_stat(path))


def close_metadata(batch: Path,frozen: dict,value: dict,scratches: list[dict]) -> None:
    actual,_=manifest.verify(batch)
    manifest.batches.check_binding(frozen)
    if actual!=value:raise ValueError('service manifest changed')
    for scratch in scratches:manifest.batches.check_binding(scratch,full=False)


def full_verify(value: dict) -> None:
    inputs=value['inputs']
    index=common.load_json(Path(inputs['index']['path']))
    packed=common.load_json(Path(inputs['packed_manifest']['path']))
    manifest.base.pack_tools.verify_packed(index,packed,Path(inputs['packed_manifest']['path']).parent)


def retain(directory: Path,name: str,value: dict) -> None:
    common.write_json(directory/name,host.serial(value))


def stage_cell(value: dict,cell: dict,batch: Path,scratch: Path,evidence: Path,progress: dict) -> dict:
    n=cell['cell'];inputs=value['inputs'];wire=value['plans'][f'{n}-source']
    argv=[inputs['stager']['path'],'--input',inputs['payload']['path'],'--plan',str(batch/wire['path']),'--output',str(scratch)]
    try:raw=observer.command(argv,timeout=120,limit=2**20)
    except observer.CommandFailure as exc:
        retain(evidence,f'{n}.staging.json',exc.evidence);raise
    retain(evidence,f'{n}.staging.json',raw)
    # command() has already bounded combined output; no payload is transported here.
    receipt=json.loads(raw['stdout'])
    retain(evidence,f'{n}.receipt.json',receipt)
    counts=receipt['counts']
    for source,target in (('source_bytes','native_source'),('write_bytes','write')):
        count=common.integer(counts[source],0,manifest.CAPS[target])
        progress['counts'][target]+=count
        common.integer(progress['counts'][target],0,manifest.CAPS[target])
    binding=stat_binding(scratch)
    progress['counts']['scratch']+=binding['size']
    common.integer(progress['counts']['scratch'],0,manifest.CAPS['scratch'])
    if (raw['returncode']!=0 or receipt['complete'] is not True or receipt['fsync_ok'] is not True
            or receipt['close_ok'] is not True or receipt['cache_controls']!={'nocache':True}
            or receipt['page_bytes']!=16384 or receipt['stride_bytes']!=cell['stride_bytes']
            or counts['source_bytes']!=cell['logical_bytes'] or counts['write_bytes']!=cell['file_bytes']
            or counts['slots_completed']!=len(cell['source_jobs']) or counts['buffer_bytes']!=cell['stride_bytes']):
        raise ValueError('incomplete or inconsistent staging receipt')
    for key,expected in (('source',inputs['payload']),('plan',wire),('scratch',binding)):
        if receipt[key]!={k:expected[k] for k in common.STAT_KEYS}:raise ValueError('staging consumed different input or scratch')
    common.integer(counts['source_syscalls'],len(cell['source_jobs']))
    common.integer(counts['write_syscalls'],len(cell['source_jobs']))
    common.integer(receipt['end_ns'],common.integer(receipt['start_ns'],1))
    return binding


def observe_cell(value: dict,cell: dict,batch: Path,scratch: dict,topology: dict,evidence: Path) -> dict:
    n=cell['cell'];wire=value['plans'][f'{n}-scratch']
    argv=[value['inputs']['reader']['path'],'--input',scratch['path'],'--plan',str(batch/wire['path']),'--workers','1','--mode','avoid']
    try:raw=process.observe(argv,topology)
    except process.ObservationFailure as exc:
        retain(evidence,f'{n}.failed.json',exc.evidence);raise
    retain(evidence,f'{n}.evidence.json',raw)
    for suffix in ('stdout','stderr'):
        with (evidence/f'{n}.{suffix}').open('xb') as stream:
            if stream.write(raw[suffix])!=len(raw[suffix]):raise OSError('short native evidence write')
    native=common.load_json(evidence/f'{n}.stdout')
    metrics=manifest.batches.validate_native(native,cell['scratch_jobs'],cell,
        {k:scratch[k] for k in common.STAT_KEYS},{k:wire[k] for k in common.STAT_KEYS})
    s=raw['snapshots']
    classification=classifier.classify(metrics['logical_bytes'],s['opening'],s['closing'],
        (s['idle_before_0'],s['idle_before_1']),(s['idle_after_0'],s['idle_after_1']),
        native['summary']['start_ns'],native['summary']['end_ns'])
    return dict(cell=n,metrics=metrics,classification=classification,evidence=f'{n}.evidence.json',
        stdout=f'{n}.stdout',stderr=f'{n}.stderr',staging=f'{n}.staging.json',receipt=f'{n}.receipt.json')
