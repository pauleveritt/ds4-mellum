"""Immutable exact-text/native-token inputs for one generated evaluation case."""
import hashlib
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
common=identity.sibling('mellum_storage_common')
bindings=identity.sibling('mellum_read_batch')
host=identity.sibling('mellum_observe_batch')
observer=host.observer
CONTENT_ID='e14b1b120a8563fd1dcb316ea6e6d815b593d2ba45605fb34b55c3b238834861'
REVISION='bfb6e1f6e2ef141a1ac88f5c25466d3a076caab3'
MODEL_BYTES=12925417760
MODEL_SHA='ea1e4ec737bb259ac2ffe1f20d412f458fd52dd230f42dce9e944f111093c29b'
FILES=('.gitignore','AGENTS.md','pyproject.toml','specs/mission.md','specs/roadmap.md','specs/tech-stack.md','templates/.gitkeep','tests/.gitkeep','uv.lock')
TOKEN_FILES=('initial.tokens','append.tokens','close.tokens','newline.tokens')
RENDERED_FILES=tuple(n.replace('.tokens','.rendered') for n in TOKEN_FILES)
SETTINGS=dict(context=8192,chunk=1024,thinking=True,temperature=0,top_k=1,top_p=1,min_p=0,caps=[4096,1024],gpu_timeout=900,cpu_timeout=120,deadline_utc='2026-10-28T12:00:00Z')


def messages(bundle: Path) -> dict[str,bytes]:
    value=common.load_json(bundle/'manifest.json')
    if (value['kind']!='workload-source' or value['split']!='evaluation' or value['repository_id']!='ac-demo'
            or value['source']['commit']!=REVISION or set(value['files'])!=set(FILES)):
        raise ValueError('frozen ac-demo bundle required')
    digests={};parts=[]
    for name in FILES:
        raw=(bundle/'source'/name).read_bytes()
        expected=value['files'][name]
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise ValueError('source bundle changed')
        digests[name]=expected['sha256']
        if name=='uv.lock':parts.append(f'uv.lock: {len(raw)} bytes; SHA-256 {digests[name]} (body omitted).\n')
        else:parts.append(f'--- {name} ---\n'+raw.decode('utf-8')+f'\n--- end {name} ---\n')
    content=dict(revision=REVISION,tree=value['source']['tree'],files=digests)
    if hashlib.sha256(common.canonical(content)).hexdigest()!=value['content_id'] or value['content_id']!=CONTENT_ID:
        raise ValueError('source content ID differs')
    user='Implement Phase 1 only of this frozen ac-demo roadmap, including its home-page tests. Return complete contents for each file to create or change. Do not start a server, execute commands, or implement later phases. Follow the preserved AGENTS.md instructions. No code or tests have been executed for you.\n\n'+''.join(parts)
    followup='Review your preceding Phase 1 answer against the preserved roadmap and AGENTS.md. Check single quotes, the Request import, home-page tests and the required test command. Correct concrete mistakes with complete corrected file contents. Stay within Phase 1. Do not start a server or claim commands/tests were executed.'
    return dict(system=b'You are a coding assistant. Follow the supplied project instructions and implement only the requested phase. Be precise about what has and has not been executed.',user=user.encode(),followup=followup.encode())


def linked_publish(container: Path,raw: Path,value: dict | None = None) -> Path:
    import os
    def build(stage):
        for path in raw.iterdir():os.link(path,stage/path.name)
        if value is not None:
            # Link creation changes ctime; bind each published inode afterward.
            value['files']={name:dict(bindings.binding(stage/name),path=name) for name in value['files']}
            common.write_json(stage/'manifest.json',value)
            os.link(stage/'manifest.json',raw/'manifest.json')
    return common.publication.publish(container/'result',build)


def prepare(bundle: Path,model: Path,binary: Path,container: Path) -> Path:
    container.mkdir(mode=0o700);raw=container/'raw';raw.mkdir(mode=0o700)
    try:
        source=common.source_identity(common.ROOT)
        model_stat=identity.file_stat(model)
        if model_stat['size']!=MODEL_BYTES:raise ValueError('unexpected model size')
        inputs=dict(bundle=bindings.binding(bundle/'manifest.json'),binary=bindings.binding(binary),
            model=dict(path=str(model.absolute()),sha256=MODEL_SHA,**model_stat))
        for name,data in messages(bundle).items():
            if len(data)>65536 or b'\0' in data:raise ValueError('invalid bounded prompt text')
            with (raw/f'{name}.txt').open('xb') as stream:
                if stream.write(data)!=len(data):raise OSError('short prompt write')
        argv=[str(binary),'prepare',str(model),str(raw),str(raw)]
        command=observer.command(argv,timeout=120,limit=2**20)
        common.write_json(raw/'prepare.command.json',host.serial(command))
        if command['returncode']:raise ValueError('native token preparation failed')
        files={name:bindings.binding(raw/name) for name in (*TOKEN_FILES,*RENDERED_FILES,'system.txt','user.txt','followup.txt')}
        # Bind paths relative to the published artifact; hardlinks retain stat identity.
        files={name:dict(binding,path=name) for name,binding in files.items()}
        if source!=common.source_identity(common.ROOT):raise ValueError('source changed during preparation')
        for name in ('bundle','binary'):bindings.check_binding(inputs[name])
        bindings.check_binding(inputs['model'],full=False)
        value=dict(schema_version=1,kind='generated-phase1-inputs',split='evaluation',settings=SETTINGS,
            source_identity=source,inputs=inputs,files=files,source_content_id=CONTENT_ID,
            policy=bindings.binding(common.ROOT/'tools/mellum_ssd_replay.py'))
        return linked_publish(container,raw,value)
    except BaseException as exc:
        failure=dict(error_type=type(exc).__name__,error=str(exc) or type(exc).__name__)
        if isinstance(exc,observer.CommandFailure):failure['command']=host.serial(exc.evidence)
        common.write_json(container/'failure.json',failure);raise


def verify(prepared: Path) -> dict:
    value=common.load_json(prepared/'manifest.json')
    if (value['kind']!='generated-phase1-inputs' or value['split']!='evaluation' or value['schema_version']!=1
            or value['settings']!=SETTINGS or value['source_content_id']!=CONTENT_ID):raise ValueError('input protocol differs')
    for name in ('bundle','binary'):bindings.check_binding(value['inputs'][name])
    bindings.check_binding(value['inputs']['model'],full=False);bindings.check_binding(value['policy'])
    if value['policy']['path']!=str(common.ROOT/'tools/mellum_ssd_replay.py'):raise ValueError('policy path differs')
    if value['source_identity']!=common.source_identity(common.ROOT):raise ValueError('input source changed')
    expected=set((*TOKEN_FILES,*RENDERED_FILES,'system.txt','user.txt','followup.txt'))
    if set(value['files'])!=expected:raise ValueError('input file matrix differs')
    for name,binding in value['files'].items():
        if binding['path']!=name:raise ValueError('escaping prepared input')
        bindings.check_binding(dict(binding,path=str(prepared/name)))
    texts=messages(Path(value['inputs']['bundle']['path']).parent)
    if any((prepared/f'{name}.txt').read_bytes()!=data for name,data in texts.items()):raise ValueError('prompt differs from source')
    return value
