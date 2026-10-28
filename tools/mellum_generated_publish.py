"""One reserved generated conversation; native failure keeps every partial file."""
from datetime import datetime, timedelta, timezone
import importlib.util
import os
from pathlib import Path
import shutil

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
inputs=identity.sibling('mellum_generated_inputs')
common=inputs.common
host=inputs.host
observer=inputs.observer
bindings=inputs.bindings
schedule=identity.sibling('mellum_generated_schedule')
check_schedule=schedule.check_schedule
DEADLINE=datetime(2026,10,28,12,tzinfo=timezone.utc)


def utc_now() -> datetime:return datetime.now(timezone.utc)


def preflight(parent: Path) -> dict:
    identity.checked_environment(dict(os.environ))
    if utc_now()+timedelta(seconds=905)>DEADLINE:raise ValueError('900second GPU timeout cannot fit before cutoff')
    if shutil.disk_usage(parent).free<32*2**30:raise ValueError('32GiB free space required')
    result=host.preflight()
    lock=observer.command(['/usr/sbin/lsof','-Fpn','/tmp/ds4.lock'])
    if lock['returncode'] not in (0,1) or lock['stdout']:raise ValueError('shared model lock is held or inspection failed')
    return dict(host=host.host_metadata(),processes=result,lock=lock)


def verify_inputs(prepared: Path,model: Path,binary: Path) -> dict:
    value=inputs.verify(prepared)
    if value['inputs']['model']['path']!=str(model.absolute()):raise ValueError('model path differs')
    build=value.get('build')
    if not isinstance(build,dict) or build.get('forced_rebuild') is not True:raise ValueError('forced build receipt required')
    if build['source_identity']!=value['source_identity']:raise ValueError('build source differs')
    if build['gpu']['path']!=str(binary.absolute()) or build['cpu']!=value['inputs']['binary']:raise ValueError('build binary differs')
    for name in ('gpu','cpu','log'):bindings.check_binding(build[name])
    return value


def run(prepared: Path,model: Path,binary: Path,container: Path) -> Path:
    if container.exists() or container.is_symlink():raise FileExistsError(container)
    progress=preflight(container.parent)
    frozen=verify_inputs(prepared,model,binary)
    container.mkdir(mode=0o700);raw=container/'raw';raw.mkdir(mode=0o700)
    try:
        common.write_json(container/'reservation.json',dict(conversations=1,parity_replays=1,gpu_seconds=900,
            caps=[4096,1024],deadline=DEADLINE.isoformat(),prepared=str(prepared),frozen=frozen))
        # No command retries after this durable reservation, including lock races.
        if utc_now()+timedelta(seconds=900)>DEADLINE:raise ValueError('deadline reached after reservation')
        command=observer.command([str(binary),'collect',str(model),str(prepared),str(raw)],timeout=900,limit=2**20)
        common.write_json(raw/'collect.command.json',host.serial(command))
        if command['returncode']:raise ValueError('native generated capture failed')
        validation=identity.sibling('mellum_generated_validation')
        report=validation.validate(prepared,raw,frozen,progress)
        # Final closure follows validation/export/replay so no mutation can hide there.
        if frozen!=verify_inputs(prepared,model,binary):raise ValueError('closing input identity changed')
        common.write_json(raw/'report.json',host.serial(report))
        return inputs.linked_publish(container,raw)
    except BaseException as exc:
        failure=dict(error_type=type(exc).__name__,error=str(exc) or type(exc).__name__,preflight=host.serial(progress))
        if isinstance(exc,observer.CommandFailure):
            failure['command']=host.serial(exc.evidence)
            failure['error_type']=exc.evidence.get('error_type',type(exc).__name__)
        common.write_json(container/'failure.json',failure);raise
