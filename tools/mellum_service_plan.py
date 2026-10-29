"""Frozen metadata-only plans for one fresh-file storage-service attempt."""
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
base=identity.sibling('mellum_observe_manifest')
common=base.common
batches=base.batches
plans=base.plans
CAPS=dict(timed=16*2**30,native_source=16*2**30,write=16*2**30,scratch=16*2**30,prewarm=0,python=128*2**30)
PAGE=16384


def cells(packed: dict) -> list[dict]:
    base.pack_tools.validate_packed(packed)
    uniform=plans.uniform_ids()
    result=[]
    for number,ids in enumerate([uniform[:64],uniform,uniform,uniform]):
        source=plans.make_jobs(packed,sorted(ids),'packed')
        sizes={j['expert_id']:sum(r['bytes'] for r in j['ranges']) for j in source}
        stride=((max(sizes.values())+PAGE-1)//PAGE)*PAGE
        slots={j['expert_id']:i*stride for i,j in enumerate(source)}
        ordered=[a*64+b for a,b in ids]
        scratch=[dict(job_id=i,expert_id=e,ranges=[dict(offset=slots[e],bytes=sizes[e],dest=0)]) for i,e in enumerate(ordered)]
        result.append(dict(cell=number,kind='calibration' if number==0 else 'measurement',
            repetition=None if number==0 else number-1,expert_ids=ordered,source_jobs=source,scratch_jobs=scratch,
            file_bytes=len(ids)*stride,logical_bytes=sum(sizes.values()),stride_bytes=stride,mode='avoid',workers=1))
    return result


def forecast(value: dict) -> dict:
    m=common.integer(value['inputs']['model']['size'],1)
    p=common.integer(value['inputs']['payload']['size'],1)
    selected=sum(c['logical_bytes'] for c in value['cells'])
    scratch=sum(c['file_bytes'] for c in value['cells'])
    maxima=dict(timed=selected,native_source=selected,write=scratch,scratch=scratch,prewarm=0)
    if any(v>CAPS[k] for k,v in maxima.items()):raise ValueError('native service cap exceeded')
    total=2*(m+3*p)+2*selected+scratch+134217728+8*2**30
    if total>CAPS['python']:raise ValueError('forecast exceeds Python cap')
    return dict(payload_base_bytes=2*(m+3*p),selected_source_bytes=2*selected,scratch_read_bytes=scratch,
        directory_upper_bytes=134217728,metadata_upper_bytes=8*2**30,total_python_upper_bytes=total,
        cap=CAPS['python'],reserved=maxima)


def prepare(prior_batch: Path,reader: Path,stager: Path,container: Path) -> Path:
    with common.read_budget(CAPS['python']):
        start=common.READ_BYTES
        inputs,_,artifacts=base.definitions(prior_batch)
        inputs.update(reader=batches.binding(reader),stager=batches.binding(stager))
        source=common.source_identity(common.ROOT)
        definitions=cells(artifacts['packed'])
        def build(stage):
            wires={}
            for c in definitions:
                for kind in ('source','scratch'):
                    name=f"{c['cell']}-{kind}"
                    path=stage/f'{name}.bin'
                    size=inputs['payload']['size'] if kind=='source' else c['file_bytes']
                    with path.open('xb') as stream:
                        encoded=plans.encode_plan(c[f'{kind}_jobs'],size)
                        if stream.write(encoded)!=len(encoded):raise OSError('short wire write')
                    wires[name]=dict(path=path.name,**common.file_identity(path))
            value=dict(schema_version=1,kind='storage-service-batch',split='tuning',inputs=inputs,
                source_identity=source,cells=definitions,plans=wires,caps=CAPS,prewarm_bytes=0,
                physical_io_verified=False,untimed_python_read_bytes=common.READ_BYTES-start)
            value['forecast']=forecast(value)
            common.write_json(stage/'manifest.json',value)
        return common.publication.publish(container,build)


def verify(manifest_dir: Path) -> tuple[dict,list[dict]]:
    with common.read_budget(CAPS['python']):
        value=common.load_json(manifest_dir/'manifest.json')
        if (value['schema_version']!=1 or value['kind']!='storage-service-batch' or value['split']!='tuning'
                or value['caps']!=CAPS or value['prewarm_bytes']!=0 or value['physical_io_verified'] is not False):
            raise ValueError('unsupported service protocol')
        inputs=value['inputs']
        for name in ('reader','stager','prior'):batches.check_binding(inputs[name])
        expected,_,artifacts=base.definitions(Path(inputs['prior']['path']).parent)
        if any(inputs[k]!=v for k,v in expected.items()):raise ValueError('service input differs')
        if common.source_identity(common.ROOT)['source_state_id']!=value['source_identity']['source_state_id']:
            raise ValueError('service source changed')
        definitions=cells(artifacts['packed'])
        if definitions!=value['cells'] or forecast(value)!=value['forecast']:raise ValueError('service definitions changed')
        names={f"{c['cell']}-{kind}" for c in definitions for kind in ('source','scratch')}
        if set(value['plans'])!=names:raise ValueError('service wire matrix changed')
        for c in definitions:
            for kind in ('source','scratch'):
                name=f"{c['cell']}-{kind}";binding=value['plans'][name]
                if binding['path']!=f'{name}.bin':raise ValueError('escaping service wire')
                path=manifest_dir/binding['path']
                if common.file_identity(path)!={k:v for k,v in binding.items() if k!='path'}:raise ValueError('service wire changed')
                size=inputs['payload']['size'] if kind=='source' else c['file_bytes']
                if batches.read_wire(path)!=plans.encode_plan(c[f'{kind}_jobs'],size):raise ValueError('service ranges changed')
        return value,definitions


def reserve(manifest_dir: Path,binding: dict) -> None:
    batches.check_binding(binding)
    value,_=verify(manifest_dir)
    common.write_json(manifest_dir.parent/'execution-reservation.json',dict(batch=binding,
        reserved=value['forecast']['reserved'],python_bytes_reserved=CAPS['python'],retry_permitted=False))
