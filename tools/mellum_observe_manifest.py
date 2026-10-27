"""Independent uniform-only observation manifests over unchanged S3a payloads."""
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
common=identity.sibling('mellum_storage_common')
batches=identity.sibling('mellum_read_batch')
index_tools=identity.sibling('mellum_expert_index')
pack_tools=identity.sibling('mellum_expert_pack')
plans=identity.sibling('mellum_read_plan')
CAPS=dict(timed=16*2**30,prewarm=8*2**30,python=128*2**30)


def runs() -> list[dict]:
    return [dict(layout=layout,mode=mode,workers=1) for layout in ('original','packed') for mode in ('buffered','avoid')]


def definitions(prior: Path) -> tuple[dict,dict,dict]:
    batch=common.load_json(prior/'manifest.json')
    if batch['kind']!='storage-read-batch' or batch['split']!='tuning' or batch['physical_io_verified'] is not False:
        raise ValueError('prior tuning batch required')
    index_path=Path(batch['inputs']['index']['path'])
    packed_path=Path(batch['inputs']['packed_manifest']['path'])
    for name in ('index','packed_manifest'):batches.check_binding(batch['inputs'][name])
    index=index_tools.validate_index(common.load_json(index_path))
    packed=pack_tools.validate_packed(common.load_json(packed_path))
    if batch['inputs']['model']!=index['source'] or packed['model']!=index['source']:
        raise ValueError('prior model linkage differs')
    payload=batch['inputs']['payload']
    if (Path(payload['path'])!=packed_path.parent/'experts.bin' or payload['size']!=packed['payload']['bytes']
            or payload['sha256']!=packed['payload']['sha256']):
        raise ValueError('prior packed linkage differs')
    for name in ('model','payload'):batches.check_binding(batch['inputs'][name],full=False)
    ids=plans.uniform_ids()
    jobs={}
    for layout,value in (('original',index),('packed',packed)):
        definition=batch['plans'][f'uniform-{layout}']
        if definition['wire']!=f'uniform-{layout}.bin' or definition['jobs']!=f'uniform-{layout}.json':
            raise ValueError('escaping prior plan')
        for key,sha in (('wire','sha256'),('jobs','jobs_sha256')):
            if common.file_identity(prior/definition[key])['sha256']!=definition[sha]:
                raise ValueError('prior uniform plan changed')
        jobs[layout]=plans.make_jobs(value,ids,layout)
        file_bytes=batch['inputs']['model' if layout=='original' else 'payload']['size']
        if (common.load_json(prior/definition['jobs'])!=dict(ids=[list(i) for i in ids],jobs=jobs[layout])
                or batches.read_wire(prior/definition['wire'])!=plans.encode_plan(jobs[layout],file_bytes)):
            raise ValueError('prior uniform ranges differ from indexes')
    inputs={name:batch['inputs'][name] for name in ('index','packed_manifest','model','payload')}
    inputs['prior']=batches.binding(prior/'manifest.json')
    return inputs,jobs,dict(index=index,packed=packed)


def prepare(prior_batch: Path,binary: Path,container: Path) -> Path:
    start=common.READ_BYTES
    inputs,jobs,_=definitions(prior_batch)
    inputs['binary']=batches.binding(binary)
    source=common.source_identity(common.ROOT)
    def build(stage):
        definitions_out={}
        for layout,items in jobs.items():
            key='model' if layout=='original' else 'payload'
            path=stage/f'{layout}.bin'
            with path.open('xb') as stream:
                encoded=plans.encode_plan(items,inputs[key]['size'])
                if stream.write(encoded)!=len(encoded):raise OSError('short observation wire write')
                stream.flush()
            definitions_out[layout]=dict(path=path.name,**common.file_identity(path))
        sizes=[sum(s['bytes'] for j in jobs[r['layout']] for s in j['ranges']) for r in runs()]
        value=dict(schema_version=1,kind='device-observation-batch',split='tuning',inputs=inputs,
            source_identity=source,compiler_command=batches.COMPILER_COMMAND,runs=runs(),plans=definitions_out,
            caps=CAPS,timed_logical_bytes=sum(sizes),prewarm_bytes=sizes[0]+sizes[2],
            physical_io_verified=False,payload_verification='pending execution; metadata-only preparation',
            untimed_python_read_bytes=common.READ_BYTES-start)
        if value['timed_logical_bytes']>CAPS['timed'] or value['prewarm_bytes']>CAPS['prewarm']:
            raise ValueError('observation read cap exceeded')
        common.write_json(stage/'manifest.json',value)
    return common.publication.publish(container,build)


def verify(manifest_dir: Path, *, payloads: bool) -> tuple[dict,dict]:
    value=common.load_json(manifest_dir/'manifest.json')
    if (value['schema_version']!=1 or value['kind']!='device-observation-batch' or value['split']!='tuning'
            or value['runs']!=runs() or value['caps']!=CAPS or value['physical_io_verified'] is not False):
        raise ValueError('unsupported observation protocol')
    inputs=value['inputs']
    for name in ('index','packed_manifest','prior','binary'):batches.check_binding(inputs[name])
    expected,jobs,artifacts=definitions(Path(inputs['prior']['path']).parent)
    if any(inputs[name]!=binding for name,binding in expected.items()):raise ValueError('observation input differs')
    if common.source_identity(common.ROOT)['source_state_id']!=value['source_identity']['source_state_id']:
        raise ValueError('observation source changed')
    if set(value['plans'])!={'original','packed'}:raise ValueError('observation plan matrix differs')
    total=prewarm=0
    for layout,items in jobs.items():
        binding=value['plans'][layout]
        if binding['path']!=f'{layout}.bin':raise ValueError('escaping observation wire')
        path=manifest_dir/binding['path']
        if common.file_identity(path)!={k:v for k,v in binding.items() if k!='path'}:
            raise ValueError('observation wire changed')
        size=inputs['model' if layout=='original' else 'payload']['size']
        if batches.read_wire(path)!=plans.encode_plan(items,size):raise ValueError('observation ranges differ')
        count=sum(s['bytes'] for j in items for s in j['ranges']);total+=2*count;prewarm+=count
    common.integer(value['timed_logical_bytes'],total,total);common.integer(value['prewarm_bytes'],prewarm,prewarm)
    if total>CAPS['timed'] or prewarm>CAPS['prewarm']:raise ValueError('read cap exceeded')
    if payloads:pack_tools.verify_packed(artifacts['index'],artifacts['packed'],Path(inputs['packed_manifest']['path']).parent)
    return value,jobs
