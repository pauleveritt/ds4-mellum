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


def run(manifest_dir: Path,container: Path) -> Path:
    container.mkdir(mode=0o700)  # A spent root is never reused, including after interruption.
    scratch_dir=container/'scratch';scratch_dir.mkdir(mode=0o700)
    evidence=container/'evidence';evidence.mkdir(mode=0o700)
    start=common.READ_BYTES
    progress=dict(schema_version=1,kind='storage-service-report',batch_path=str(manifest_dir),complete=False,
        started_cells=[],runs=[],completed_logical_bytes=0,prewarm_bytes=0,physical_io_verified=False,
        closing_verification='incomplete',counts=dict(timed=0,native_source=0,write=0,scratch=0),
        unknown_reserved_maxima=True)
    scratches=[]
    def publish(success):
        progress.update(unstarted_cells=[i for i in range(4) if i not in progress['started_cells']],
            untimed_python_read_bytes=common.READ_BYTES-start,
            cumulative_python_read_bytes=common.READ_BYTES-start+progress.get('prepare_python_read_bytes',0),
            read_accounting='logical Python input bytes; excludes imports, Git and metadata subprocess internals, native reads',
            limits=['device-wide traffic cannot establish per-file, per-job or NAND attribution',
                'one worker, aligned packed subset; no inference tok/s or 8 GB feasibility conclusion'])
        def build(stage):
            for path in evidence.iterdir():os.link(path,stage/path.name)
            common.write_json(stage/('report.json' if success else 'failure.json'),host.serial(progress))
        return common.publication.publish(container/('result' if success else 'failure'),build)
    try:
        with common.read_budget(manifest.CAPS['python']):
            value=common.load_json(manifest_dir/'manifest.json')
            prep=common.integer(value['untimed_python_read_bytes'])
            progress.update(prepare_python_read_bytes=prep,reserved_maxima=manifest.forecast(value))
            remaining=manifest.forecast(value)['total_python_upper_bytes']-prep-(common.READ_BYTES-start)
            common.integer(remaining)
            with common.read_budget(remaining):
                if (manifest_dir.parent/'execution-reservation.json').exists():raise ValueError('service reservation already spent')
                progress['preflight']=host.preflight();progress['host']=host.host_metadata()
                space(container)
                value,definitions=manifest.verify(manifest_dir)
                frozen=manifest.batches.binding(manifest_dir/'manifest.json');progress['batch']=frozen
                files={k:value['inputs'][k] for k in ('model','payload')}
                topology=observer.discover(files)
                progress.update(topology_before=topology,geometry=geometry(topology),frozen=value)
                manifest.reserve(manifest_dir,frozen);progress['reserved']=True
                full_verify(value);close_metadata(manifest_dir,frozen,value,scratches)
                packed=common.load_json(Path(value['inputs']['packed_manifest']['path']))
                for cell in definitions:
                    n=cell['cell']
                    expected=verifier.expected(value['inputs']['payload'],cell,packed)
                    retain(evidence,f'{n}.expected.json',expected)
                    space(container);close_metadata(manifest_dir,frozen,value,scratches)
                    progress['started_cells'].append(n)
                    scratch=stage_cell(value,cell,manifest_dir,scratch_dir/f'{n}.bin',evidence,progress)
                    scratches.append(scratch)
                    close_metadata(manifest_dir,frozen,value,scratches)
                    mapped=observer.discover(dict(model=files['payload'],payload=scratch))
                    retain(evidence,f'{n}.topology.json',mapped)
                    if not host.same_topology(topology,mapped) or geometry(mapped)!=progress['geometry']:
                        raise ValueError('scratch/source device topology changed')
                    result=observe_cell(value,cell,manifest_dir,scratch,mapped,evidence)
                    progress['completed_logical_bytes']+=result['metrics']['logical_bytes']
                    progress['counts']['timed']+=result['metrics']['logical_bytes']
                    common.integer(progress['counts']['timed'],0,manifest.CAPS['timed'])
                    close_metadata(manifest_dir,frozen,value,scratches)
                    verification=verifier.verify_scratch(value['inputs']['payload'],scratch,cell,expected)
                    result.update(expected=expected,verification=verification,scratch=scratch)
                    progress['runs'].append(result)
                    retain(evidence,f'{n}.result.json',result)
                    del mapped,result,verification,expected
                    close_metadata(manifest_dir,frozen,value,scratches)
                    if progress['runs'][-1]['classification']['status']!='consistent-volume':break
                full_verify(value);close_metadata(manifest_dir,frozen,value,scratches)
                after=observer.discover(files);progress['topology_after']=after
                if not host.same_topology(topology,after) or geometry(after)!=progress['geometry']:
                    raise ValueError('closing device topology changed')
                progress['closing_verification']='complete'
                qualified=len(progress['runs'])==4 and all(c['classification']['status']=='consistent-volume' for c in progress['runs'])
                progress.update(complete=qualified,unknown_reserved_maxima=False)
                if not qualified:
                    progress.update(error_type='NonqualifyingCell',error='cell outside consistent-volume; remaining cells stopped')
                # Hash evidence only; scratch has already had its one comparison pass.
                progress['evidence_hashes']={p.name:common.file_identity(p)['sha256'] for p in evidence.iterdir()}
        output=publish(qualified)
    except BaseException as exc:
        if isinstance(exc,process.ObservationFailure):error_type=exc.evidence.get('error_type',type(exc).__name__)
        elif isinstance(exc,observer.CommandFailure) and exc.__cause__ is not None:error_type=type(exc.__cause__).__name__
        else:error_type=type(exc).__name__
        progress.update(complete=False,error_type=error_type,error=str(exc) or error_type,
            unknown_reserved_maxima=True)
        if isinstance(exc,observer.CommandFailure):progress['failed_command']=exc.evidence
        # Stat alone retains partial output size, even when the child produced no receipt.
        partial=[]
        for path in scratch_dir.iterdir():
            try:partial.append(stat_binding(path))
            except (OSError,ValueError):partial.append(dict(path=str(path),identity='unavailable'))
        progress['partial_scratch']=partial
        try:publish(False)
        except BaseException as publication_error:exc.add_note(f'failure publication failed: {publication_error}; persistent evidence retained at {evidence}')
        raise
    if not qualified:raise ValueError(f'storage-service cell did not qualify; evidence at {output}')
    return output
