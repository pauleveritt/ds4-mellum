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
COMPILER_COMMAND = 'cc -std=c99 -O2 -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE -Wall -Wextra -Werror -pthread tools/mellum_read_bench.c tools/mellum_read_plan.c tools/mellum_read_io.c tools/mellum_read_run.c tools/mellum_read_gate.c -o tools/mellum_read_bench'


def frozen_runs() -> list[dict]:
    return [dict(plan=plan,mode=mode,workers=workers,repetition=rep,layout=layout)
            for plan in ('uniform','diagnostic') for mode in ('buffered','avoid')
            for workers in (1,2,4) for rep in (0,1,2)
            for layout in (('original','packed') if rep % 2 == 0 else ('packed','original'))]


def binding(path: Path) -> dict:
    return dict(path=str(path.absolute()), **common.file_identity(path))


def check_binding(value: dict, *, full: bool = True) -> None:
    path = Path(value['path'])
    actual = common.file_identity(path) if full else identity.file_stat(path)
    if any(value.get(key) != item for key,item in actual.items()):
        raise ValueError(f'frozen input changed: {path}')


def prepare_batch(index_path: Path, packed_dir: Path, trace_path: Path, binary: Path, container: Path) -> Path:
    read_start = common.READ_BYTES
    index = index_tools.validate_index(common.load_json(index_path))
    manifest = pack_tools.validate_packed(common.load_json(packed_dir / 'manifest.json'))
    index_tools.verify_index_source(index)
    pack_tools.verify_packed(index, manifest, packed_dir)
    ids = dict(uniform=plans.uniform_ids(), diagnostic=plans.diagnostic_ids(trace_path))
    inputs = dict(index=binding(index_path), packed_manifest=binding(packed_dir/'manifest.json'),
                  model=binding(Path(index['source']['path'])), payload=binding(packed_dir/'experts.bin'),
                  trace=binding(trace_path), binary=binding(binary))
    source = common.source_identity(common.ROOT)
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
                wire_identity = common.file_identity(wire)
                definitions[key] = dict(wire=wire.name,sha256=wire_identity['sha256'],
                                        identity={key:wire_identity[key] for key in common.STAT_KEYS},
                                        jobs=f'{key}.json', jobs_sha256=common.file_identity(stage/f'{key}.json')['sha256'],
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
        value['untimed_python_read_bytes'] = common.READ_BYTES - read_start
        common.write_json(stage/'manifest.json',value)
    return common.publication.publish(container,build)


def verify_batch(batch_dir: Path) -> tuple[dict, dict]:
    batch = common.load_json(batch_dir/'manifest.json')
    if (common.integer(batch['schema_version']) != 1 or batch['kind'] != 'storage-read-batch' or
            batch['split'] != 'tuning' or batch['runs'] != frozen_runs() or batch['physical_io_verified'] is not False):
        raise ValueError('unsupported frozen batch/protocol')
    if batch['source_identity']['source_state_id'] != common.source_identity(common.ROOT)['source_state_id']:
        raise ValueError('measurement source changed since preparation')
    for value in batch['inputs'].values():
        check_binding(value)
    index = index_tools.validate_index(common.load_json(Path(batch['inputs']['index']['path'])))
    packed_path = Path(batch['inputs']['packed_manifest']['path'])
    packed = pack_tools.validate_packed(common.load_json(packed_path))
    index_tools.verify_index_source(index)
    pack_tools.verify_packed(index,packed,packed_path.parent)
    if batch['inputs']['model'] != index['source']:
        raise ValueError('native model input differs from verified index')
    if (Path(batch['inputs']['payload']['path']) != packed_path.parent/'experts.bin' or
            batch['inputs']['payload']['sha256'] != packed['payload']['sha256'] or
            batch['inputs']['payload']['size'] != packed['payload']['bytes']):
        raise ValueError('native packed input differs from verified payload')
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
            if common.file_identity(batch_dir/definition[field])['sha256'] != common.digest(definition[hash_key]):
                raise ValueError('frozen plan changed')
        if identity.file_stat(batch_dir/definition['wire']) != definition['identity']:
            raise ValueError('wire plan identity changed')
        logical = common.load_json(batch_dir/definition['jobs'])
        jobs = plans.make_jobs(index if layout == 'original' else packed,ids[name],layout)
        size = batch['inputs']['model' if layout == 'original' else 'payload']['size']
        if logical != dict(ids=[list(pair) for pair in ids[name]],jobs=jobs) or read_wire(batch_dir/definition['wire']) != plans.encode_plan(jobs,size):
            raise ValueError('frozen jobs differ from checked indexes/trace')
        count = sum(sum(span['bytes'] for span in job['ranges']) for job in jobs)
        common.integer(definition['logical_bytes'],count,count)
        jobs_by_key[key] = jobs
    for run in batch['runs']:
        total += batch['plans'][f"{run['plan']}-{run['layout']}"]['logical_bytes']
    common.integer(batch['timed_logical_bytes'],total,total)
    common.integer(total,1,BUDGET-1)
    return batch,jobs_by_key


def distribution(values: list[float]) -> dict:
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('nonempty finite nonnegative samples required')
    ordered = sorted(values)
    return {name:ordered[math.ceil(q*len(ordered))-1] for name,q in (('p50',.5),('p95',.95),('p99',.99),('max',1))}


def validate_native(report: dict, jobs: list[dict], run: dict, expected_file: dict, expected_plan: dict | None = None) -> dict:
    try:
        total = sum(sum(span['bytes'] for span in job['ranges']) for job in jobs)
        maximum = max(sum(span['bytes'] for span in job['ranges']) for job in jobs)
        common.integer(report['schema_version'],1,1)
        common.integer(report['workers'],run['workers'],run['workers'])
        common.integer(report['jobs'],len(jobs),len(jobs))
        common.integer(report['logical_bytes'],total,total)
        if report['mode'] != run['mode'] or report['physical_io_verified'] is not False:
            raise ValueError('native mode/physical claim differs')
        avoid = run['mode'] == 'avoid'
        if any(report['cache_controls'][key] is not avoid for key in ('nocache','rdahead_disabled')):
            raise ValueError('native cache controls differ')
        if report['file'] != expected_file:
            raise ValueError('native file identity differs')
        for key in common.STAT_KEYS:
            common.integer(report['file'][key],expected_file[key],expected_file[key])
        for key in common.STAT_KEYS:
            common.integer(report['plan'][key])
        if expected_plan is not None and report['plan'] != expected_plan:
            raise ValueError('native consumed a different plan identity')
        summary = report['summary']
        start = common.integer(summary['start_ns'],1)
        end = common.integer(summary['end_ns'],start+1)
        common.integer(summary['buffer_bytes'],run['workers']*maximum,run['workers']*maximum)
        prewarm = 0 if avoid else total
        common.integer(summary['prewarm_bytes'],prewarm,prewarm)
        common.integer(summary['prewarm_syscalls'],0 if avoid else sum(len(j['ranges']) for j in jobs),
                       0 if avoid else 2**63-1)
        if len(report['samples']) != len(jobs):
            raise ValueError('incomplete native samples')
        queue, service, elapsed, details = [], [], [], []
        for i,(sample,job) in enumerate(zip(report['samples'],jobs,strict=True)):
            common.integer(sample['job_id'],i,i)
            common.integer(sample['expert_id'],job['expert_id'],job['expert_id'])
            common.integer(sample['enqueue_ns'],start,start)
            began = common.integer(sample['start_ns'],start,end)
            done = common.integer(sample['done_ns'],began,end)
            size = sum(span['bytes'] for span in job['ranges'])
            common.integer(sample['bytes'],size,size)
            common.integer(sample['syscalls'],len(job['ranges']))
            q,s,e = (began-start)/1e6, (done-began)/1e6, (done-start)/1e6
            queue.append(q); service.append(s); elapsed.append(e)
            details.append(dict(job_id=i,queue_ms=q,service_ms=s,end_to_end_ms=e))
        calls = sum(sample['syscalls'] for sample in report['samples'])
        return dict(logical_bytes=total,makespan_ms=(end-start)/1e6,bytes_per_second=total*1e9/(end-start),
                    queue_ms=distribution(queue),service_ms=distribution(service),end_to_end_ms=distribution(elapsed),
                    jobs=details,completion_order=sorted(range(len(jobs)),key=lambda i:(report['samples'][i]['done_ns'],i)),
                    syscalls=calls,syscalls_per_job=calls/len(jobs),returned_read_bytes_per_logical_byte=1.0,
                    physical_io_verified=False)
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('malformed native report') from exc


def host_metadata(inputs: dict) -> dict:
    result = dict(platform=platform.platform(),python=sys.version,cpu=platform.processor(),
                  ram_bytes=os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE') if sys.platform != 'darwin' else None,
                  storage={},device_observations='physical read traffic unavailable')
    if sys.platform == 'darwin':
        value = subprocess.run(['/usr/sbin/sysctl','-n','hw.memsize'],capture_output=True,check=False)
        if value.returncode == 0:
            result['ram_bytes'] = int(value.stdout)
    for name in ('model','payload'):
        path = Path(inputs[name]['path'])
        volume = os.statvfs(path)
        device = dict(st_dev=path.stat().st_dev,block_bytes=volume.f_bsize,fragment_bytes=volume.f_frsize,
                      free_bytes=volume.f_bavail*volume.f_frsize,diskutil='unavailable')
        if sys.platform == 'darwin':
            value = subprocess.run(['/usr/sbin/diskutil','info','-plist',str(path.parent)],capture_output=True,check=False)
            if value.returncode == 0:
                info = plistlib.loads(value.stdout)
                device['diskutil'] = {key:info[key] for key in ('DeviceIdentifier','DeviceNode','FilesystemType','VolumeName','BusProtocol','SolidState','MountPoint') if key in info}
        result['storage'][name] = device
    return result


def run_batch(batch_dir: Path, container: Path) -> Path:
    read_start = common.READ_BYTES
    failure_path = container.with_name(container.name+'.failure')
    if failure_path.exists() or failure_path.is_symlink():
        raise FileExistsError(failure_path)
    def build(stage: Path) -> None:
        try:
            batch,jobs_by_key = verify_batch(batch_dir)
            frozen_manifest = binding(batch_dir/'manifest.json')
            reservation = batch_dir.parent/'execution-reservation.json'
            common.write_json(reservation,dict(batch=frozen_manifest,timed_bytes_reserved=batch['timed_logical_bytes'],
                                              retry_permitted=False))
            raw = stage/'raw'; raw.mkdir()
            host = host_metadata(batch['inputs'])
            runs = []
            for number,run in enumerate(batch['runs']):
                key = f"{run['plan']}-{run['layout']}"
                definition = batch['plans'][key]
                input_key = 'model' if run['layout'] == 'original' else 'payload'
                for value in batch['inputs'].values():
                    check_binding(value,full=False)
                check_binding(batch['inputs']['binary'])
                check_binding(frozen_manifest)
                if common.file_identity(batch_dir/definition['wire'])['sha256'] != definition['sha256']:
                    raise ValueError('wire plan changed during batch')
                for name,value in batch['source_identity']['files'].items():
                    if identity.file_stat(common.ROOT/name) != {key:value[key] for key in common.STAT_KEYS}:
                        raise ValueError('measurement source changed during batch')
                argv = [batch['inputs']['binary']['path'],'--input',batch['inputs'][input_key]['path'],
                        '--plan',str(batch_dir/definition['wire']),'--workers',str(run['workers']),'--mode',run['mode']]
                stdout,stderr = raw/f'{number:02}.stdout',raw/f'{number:02}.stderr'
                with stdout.open('xb') as output,stderr.open('xb') as errors:
                    process = subprocess.run(argv,stdout=output,stderr=errors,check=False)
                    output.flush(); errors.flush()
                common.write_json(raw/f'{number:02}.exit.json',dict(argv=argv,returncode=process.returncode))
                if process.returncode:
                    raise ValueError(f'native run {number} failed with exit {process.returncode}')
                for value in batch['inputs'].values():
                    check_binding(value,full=False)
                native = common.load_json(stdout)
                expected = {key:batch['inputs'][input_key][key] for key in common.STAT_KEYS}
                metrics = validate_native(native,jobs_by_key[key],run,expected,definition['identity'])
                common.write_json(raw/f'{number:02}.metrics.json',metrics)
                # Raw samples and derived per-job metrics stay in separately bounded files.
                brief = {key:value for key,value in metrics.items() if key not in ('jobs','completion_order')}
                runs.append(dict(**run,raw_stdout=f'raw/{stdout.name}',stdout_sha256=common.file_identity(stdout)['sha256'],
                                 raw_stderr=f'raw/{stderr.name}',stderr_sha256=common.file_identity(stderr)['sha256'],
                                 metrics=brief,job_metrics=f'raw/{number:02}.metrics.json',
                                 job_metrics_sha256=common.file_identity(raw/f'{number:02}.metrics.json')['sha256']))
            verify_batch(batch_dir)
            check_binding(frozen_manifest)
            completed = sum(run['metrics']['logical_bytes'] for run in runs)
            if completed != batch['timed_logical_bytes']:
                raise ValueError('completed bytes differ from frozen batch')
            ranges = {key:[dict(job_id=j['job_id'],bytes=s['bytes'],offset=s['offset'],
                               filesystem_alignment=s['offset'] % host['storage']['model' if key.endswith('original') else 'payload']['block_bytes'],
                               alignment_16k=s['offset'] % 16384) for j in jobs for s in j['ranges']]
                      for key,jobs in jobs_by_key.items()}
            common.write_json(stage/'report.json',dict(schema_version=1,kind='storage-read-report',batch=frozen_manifest,
                frozen=batch,runs=runs,completed_logical_bytes=completed,physical_io_verified=False,host=host,ranges=ranges,
                prewarm_bytes=sum(batch['plans'][f"{r['plan']}-{r['layout']}"]['logical_bytes'] for r in runs if r['mode']=='buffered'),
                untimed_python_read_bytes=common.READ_BYTES-read_start,
                read_accounting='Python input bytes including metadata/source/report reads; excludes Git internals, runtime imports and native reads; logical, not physical I/O',
                memory=dict(max_worker_buffers=4*max(sum(s['bytes'] for s in j['ranges']) for jobs in jobs_by_key.values() for j in jobs),
                            excluded=['runtime','thread stacks','OS caches']),
                limits=['open-loop storage jobs; no inference tok/s conversion','cache avoidance requested; physical traffic unverified',
                        'all hashing, source/payload verification and prewarming excluded from timing']))
        except BaseException as exc:
            def retain(failed: Path) -> None:
                shutil.copytree(stage,failed,dirs_exist_ok=True)
                common.write_json(failed/'failure.json',dict(schema_version=1,kind='failed-storage-read-batch',
                                                            complete=False,error=str(exc),batch=str(batch_dir),untimed_python_read_bytes=common.READ_BYTES-read_start))
            try:
                common.publication.publish(failure_path,retain)
            except OSError as failure:
                exc.add_note(f'failure evidence publication failed: {failure}')
            raise
    return common.publication.publish(container,build)


def read_wire(path: Path) -> bytes:
    with common.opened(path) as (source, before):
        common.integer(before['size'],64,64+4096*3*32)
        return source.read(before['size'])
