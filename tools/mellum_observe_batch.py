"""One-shot checked publication of a device-observed CPU read diagnostic."""
import base64
import importlib.util
from pathlib import Path
import platform
import re
import shutil

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
manifest=identity.sibling('mellum_observe_manifest')
common=manifest.common
observer=identity.sibling('mellum_device_observer')
process=identity.sibling('mellum_observe_process')
classifier=identity.sibling('mellum_device_classify')


def compute_targets() -> set[str]:
    """Repository rules linking inference cores or GPU backends name compute users."""
    with common.opened(common.ROOT/'Makefile') as (source,before):
        common.integer(before['size'],1,2**20)
        lines=source.read(before['size']).decode().splitlines()
    variables={}
    for line in lines:
        match=re.match(r'^(\w+)\s*[:?+]?=\s*([\w./-]+)\s*$',line)
        if match:variables.setdefault(match[1],set()).add(match[2])
    targets=set()
    for line in lines:
        match=re.match(r'^(\S+):\s*(.*)$',line)
        if not match or not re.search(r'\$\((?:CORE_OBJS|CPU_CORE_OBJS)\)|ds4_(?:metal|cuda|rocm)\.o',match[2]):continue
        target=match[1];variable=re.fullmatch(r'\$\((\w+)\)',target)
        names=variables.get(variable[1],set()) if variable else {target}
        targets.update(Path(name).name for name in names if not name.endswith('.o'))
    return targets


def preflight() -> dict:
    result=observer.command(['/bin/ps','-axo','pid=,comm=,args='])
    if result['returncode']:raise observer.CommandFailure('process inspection failed',result)
    busy=[]
    targets=compute_targets()
    for line in result['stdout'].decode('utf-8',errors='replace').splitlines():
        fields=line.split(None,2)
        if len(fields)<2:continue
        name=Path(fields[1]).name
        if (name in targets or name in ('ds4','ds4-server','ds4-cli','ds4-agent') or
                name.startswith(('test_mellum_gpu','test_mellum_metal','test_mellum_resident','test_mellum_owned_live'))):
            busy.append(line)
    if busy:raise observer.CommandFailure('active inference process: '+'; '.join(busy),result)
    return dict(processes=busy,command=result)


def host_metadata() -> dict:
    raw=observer.command(['/usr/sbin/sysctl','-n','hw.memsize','hw.model','machdep.cpu.brand_string'])
    fields=raw['stdout'].decode('utf-8').splitlines()
    if raw['returncode'] or len(fields)!=3:raise observer.CommandFailure('host identity unavailable',raw)
    return dict(platform=platform.platform(),ram_bytes=common.integer(int(fields[0]),1),
                model=observer.text(fields[1]),cpu=observer.text(fields[2]),command=raw)


def serial(value: object):
    if isinstance(value,bytes):return dict(encoding='base64',bytes=len(value),data=base64.b64encode(value).decode())
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def same_topology(before: dict,after: dict) -> bool:
    return {k:v for k,v in before.items() if k!='raw'}=={k:v for k,v in after.items() if k!='raw'}


def run(manifest_dir: Path,container: Path) -> Path:
    start=common.READ_BYTES
    failure_path=container.with_name(container.name+'.failure')
    if failure_path.exists() or failure_path.is_symlink():raise FileExistsError(failure_path)
    progress=dict(batch_path=str(manifest_dir),started_cells=[],runs=[],completed_logical_bytes=0,prewarm_bytes=0)
    def build(stage):
        try:
            value=common.load_json(manifest_dir/'manifest.json')
            prep=common.integer(value['untimed_python_read_bytes'])
            progress.update(declared_batch=value,prepare_python_read_bytes=prep)
            forecast=manifest.forecast(value)
            remaining=forecast['total_python_upper_bytes']-prep-(common.READ_BYTES-start)
            with common.read_budget(remaining):
                if (manifest_dir.parent/'execution-reservation.json').exists():raise ValueError('observation reservation already spent')
                progress['preflight']=preflight()
                progress['host']=host_metadata()
                value,jobs=manifest.verify(manifest_dir,payloads=False)
                frozen=manifest.batches.binding(manifest_dir/'manifest.json')
                progress['batch']=frozen
                files={k:value['inputs'][k] for k in ('model','payload')}
                topology=observer.discover(files);progress['topology_before']=topology
                manifest.reserve(manifest_dir,frozen)
                progress['reserved']=True
                manifest.verify(manifest_dir,payloads=True)
                for number,cell in enumerate(value['runs']):
                    manifest.verify(manifest_dir,payloads=False)
                    manifest.batches.check_binding(frozen)
                    layout=cell['layout'];key='model' if layout=='original' else 'payload'
                    wire=value['plans'][layout]
                    argv=[value['inputs']['binary']['path'],'--input',files[key]['path'],'--plan',str(manifest_dir/wire['path']),
                          '--workers','1','--mode',cell['mode']]
                    progress['started_cells'].append(number)
                    try:evidence=process.observe(argv,topology)
                    except process.ObservationFailure as exc:
                        common.write_json(stage/f'{number}.failed.json',serial(exc.evidence));raise
                    common.write_json(stage/f'{number}.evidence.json',serial(evidence))
                    stdout=stage/f'{number}.stdout';stderr=stage/f'{number}.stderr'
                    for path,data in ((stdout,evidence['stdout']),(stderr,evidence['stderr'])):
                        with path.open('xb') as stream:
                            if stream.write(data)!=len(data):raise OSError('short native output write')
                            stream.flush()
                    native=common.load_json(stdout)
                    expected={k:files[key][k] for k in common.STAT_KEYS}
                    expected_plan={k:wire[k] for k in common.STAT_KEYS}
                    metrics=manifest.batches.validate_native(native,jobs[layout],cell,expected,expected_plan)
                    s=evidence['snapshots']
                    classification=classifier.classify(metrics['logical_bytes'],s['opening'],s['closing'],
                        (s['idle_before_0'],s['idle_before_1']),(s['idle_after_0'],s['idle_after_1']),
                        native['summary']['start_ns'],native['summary']['end_ns'])
                    progress['runs'].append(dict(cell=number,**cell,evidence=f'{number}.evidence.json',
                        evidence_sha256=common.file_identity(stage/f'{number}.evidence.json')['sha256'],
                        stdout=f'{number}.stdout',stdout_sha256=common.file_identity(stdout)['sha256'],
                        stderr=f'{number}.stderr',stderr_sha256=common.file_identity(stderr)['sha256'],
                        metrics=metrics,classification=classification))
                    progress['completed_logical_bytes']+=metrics['logical_bytes']
                    progress['prewarm_bytes']+=native['summary']['prewarm_bytes']
                    if classification['status']=='invalid':raise ValueError('invalid device observation')
                    del evidence,s,native
                    manifest.verify(manifest_dir,payloads=False)
                    manifest.batches.check_binding(frozen)
                manifest.verify(manifest_dir,payloads=True)
                manifest.batches.check_binding(frozen)
                after=observer.discover(files);progress['topology_after']=after
                if not same_topology(topology,after):raise ValueError('device topology changed during batch')
                if (progress['completed_logical_bytes']!=value['timed_logical_bytes'] or
                        progress['prewarm_bytes']!=value['prewarm_bytes']):raise ValueError('incomplete observation matrix')
                progress.update(schema_version=1,kind='device-observation-report',complete=True,batch=frozen,
                    frozen=value,physical_io_verified=False,
                    untimed_python_read_bytes=common.READ_BYTES-start,prepare_python_read_bytes=prep,
                    cumulative_python_read_bytes=common.READ_BYTES-start+prep,
                    read_accounting='logical Python input bytes; excludes Git internals, imports, metadata subprocess internals and native reads',
                    limits=['device-wide counters cannot prove file attribution or NAND service',
                            'hot-file diagnostic; no SSD bandwidth, layout comparison or inference speed claim'])
                common.write_json(stage/'report.json',serial(progress))
        except BaseException as exc:
            error_type=exc.evidence.get('error_type',type(exc).__name__) if isinstance(exc,process.ObservationFailure) else type(exc).__name__
            progress.update(schema_version=1,kind='failed-device-observation',complete=False,error=str(exc) or error_type,error_type=error_type,
                unstarted_cells=[i for i in range(4) if i not in progress['started_cells']],
                failed_cell_progress='unknown; reserved maxima retained',
                timed_reserved_max=value.get('timed_logical_bytes') if 'value' in locals() else None,
                prewarm_reserved_max=value.get('prewarm_bytes') if 'value' in locals() else None,
                untimed_python_read_bytes=common.READ_BYTES-start,
                cumulative_python_read_bytes=common.READ_BYTES-start+progress.get('prepare_python_read_bytes',0),
                physical_io_verified=False)
            if isinstance(exc,observer.CommandFailure):progress['failed_command']=exc.evidence
            def retain(failed):
                shutil.copytree(stage,failed,dirs_exist_ok=True)
                common.write_json(failed/'failure.json',serial(progress))
            try:common.publication.publish(failure_path,retain)
            except OSError as error:exc.add_note(f'failure evidence publication failed: {error}')
            raise
    with common.read_budget(manifest.CAPS['python']):
        return common.publication.publish(container,build)
