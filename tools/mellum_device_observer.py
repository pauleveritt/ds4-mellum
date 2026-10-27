"""Read-only macOS topology and timestamped block-driver observations."""
import hashlib
import os
from pathlib import Path
import plistlib
import re
import selectors
import stat
import subprocess
import sys
import time


def uint(value: object) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError('unsigned 64-bit counter/identity required')
    return value


def one(values: list):
    if len(values) != 1:
        raise ValueError('unique storage mapping required')
    return values[0]


def text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError('nonempty device identity required')
    return value


def parse_topology(files: dict[str,dict], volume_info: dict[str,dict], apfs: dict,
                   registry: list[dict], disk_info: dict) -> dict:
    try:
        volumes=[]
        for name in ('model','payload'):
            file,volume=files[name],volume_info[name]
            if (volume['FilesystemType'] != 'apfs' or uint(file['dev']) != uint(volume['observed_st_dev'])
                    or file['dev'] != uint(volume['device_rdev'])):
                raise ValueError('file device differs from APFS volume')
            container=one([c for c in apfs['Containers'] if c['ContainerReference']==volume['APFSContainerReference']])
            store=one(container['PhysicalStores'])
            member=one([v for v in container['Volumes'] if v['DeviceIdentifier']==volume['DeviceIdentifier']])
            if member['APFSVolumeUUID'] != volume['VolumeUUID']:
                raise ValueError('volume UUID differs')
            if volume['APFSPhysicalStores'] != [dict(APFSPhysicalStore=store['DeviceIdentifier'])]:
                raise ValueError('physical store differs')
            match=re.fullmatch(r'(disk[0-9]+)s[0-9]+',store['DeviceIdentifier'])
            if not match: raise ValueError('unsupported physical store')
            volumes.append(dict(volume=volume['DeviceIdentifier'],volume_uuid=text(volume['VolumeUUID']),
                container=container['ContainerReference'],container_uuid=text(container['APFSContainerUUID']),
                store=store['DeviceIdentifier'],store_uuid=text(store['DiskUUID']),disk=match[1]))
        first=volumes[0]
        if any(v != first for v in volumes):
            raise ValueError('payloads require the same APFS volume and disk')
        if disk_info['DeviceIdentifier'] != first['disk'] or disk_info['Whole'] is not True:
            raise ValueError('whole physical disk required')
        size=uint(disk_info['TotalSize'])
        candidates=[]
        for driver in registry:
            for media in driver.get('IORegistryEntryChildren',[]):
                if media.get('BSD Name')==first['disk']:
                    if media.get('Whole') is not True or uint(media['Size']) != size or media['IOObjectClass']!='IOMedia':
                        raise ValueError('whole media size/type differs')
                    candidates.append((uint(driver['IORegistryEntryID']),uint(media['IORegistryEntryID'])))
        driver_id,media_id=one(candidates)
        return dict(**first,size=size,model=text(disk_info['MediaName']),driver_id=driver_id,media_id=media_id)
    except (KeyError,TypeError,AttributeError,IndexError) as exc:
        raise ValueError('malformed or unsupported device topology') from exc


COUNTERS=('Bytes (Read)','Bytes (Write)','Operations (Read)','Operations (Write)',
          'Errors (Read)','Retries (Read)')


def now_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


class CommandFailure(ValueError):
    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence=evidence


def command(argv: list[str], *, timeout: float=5, limit: int=32*2**20) -> dict:
    result=dict(argv=argv,start_ns=now_ns(),stdout=b'',stderr=b'',returncode=None)
    child=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    streams={child.stdout:bytearray(),child.stderr:bytearray()}
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ)
            deadline=time.monotonic()+timeout
            total=0
            while selector.get_map():
                remaining=deadline-time.monotonic()
                if remaining<=0:raise ValueError('metadata deadline exceeded')
                for key,_ in selector.select(remaining):
                    block=os.read(key.fd,min(65536,limit-total+1))
                    if not block:selector.unregister(key.fileobj);continue
                    streams[key.fileobj].extend(block[:limit-total]);total+=len(block)
                    if total>limit:raise ValueError('metadata output exceeds bound')
            child.wait(timeout=max(.001,deadline-time.monotonic()))
    except (ValueError,subprocess.TimeoutExpired) as exc:
        if child.poll() is None:child.kill()
        child.wait()
        result.update(stdout=bytes(streams[child.stdout]),stderr=bytes(streams[child.stderr]),
                      returncode=child.returncode,end_ns=now_ns())
        raise CommandFailure(str(exc),result) from exc
    finally:
        child.stdout.close();child.stderr.close()
    result.update(stdout=bytes(streams[child.stdout]),stderr=bytes(streams[child.stderr]),
                  returncode=child.returncode,end_ns=now_ns())
    return result


def decoded(result: dict):
    try:
        if result['returncode']:raise ValueError('metadata command failed')
        return plistlib.loads(result['stdout'])
    except (ValueError,TypeError,plistlib.InvalidFileException) as exc:
        raise CommandFailure('invalid metadata plist/status',result) from exc


def snapshot(topology: dict) -> dict:
    result=command(['/usr/sbin/ioreg','-a','-r','-c','IOBlockStorageDriver'])
    try:
        driver=one([d for d in decoded(result) if d['IORegistryEntryID']==topology['driver_id']])
        one([m for m in driver.get('IORegistryEntryChildren',[]) if m['IORegistryEntryID']==topology['media_id']])
        counters={name:uint(driver['Statistics'][name]) for name in COUNTERS}
        return dict(command=result,sha256=hashlib.sha256(result['stdout']).hexdigest(),
                    driver_id=topology['driver_id'],media_id=topology['media_id'],counters=counters)
    except (ValueError,KeyError,TypeError) as exc:
        raise CommandFailure('driver/media/counter snapshot invalid',result) from exc


def discover(files: dict[str,dict]) -> dict:
    if sys.platform!='darwin':raise ValueError('macOS device observation required')
    raw=[]
    def query(argv):
        result=command(argv);raw.append(result);return decoded(result)
    volumes={}
    for name,file in files.items():
        path=Path(file['path'])
        observed=path.stat().st_dev
        candidates=[]
        for entry in Path('/dev').iterdir():
            if re.fullmatch(r'disk[0-9]+s[0-9]+',entry.name):
                device=entry.stat()
                if stat.S_ISBLK(device.st_mode) and device.st_rdev==observed:candidates.append(entry)
        node=one(candidates)
        volume=query(['/usr/sbin/diskutil','info','-plist',str(node)])
        if volume['DeviceNode']!=str(node):raise ValueError('device node differs')
        volumes[name]=dict(volume,observed_st_dev=observed,device_rdev=node.stat().st_rdev)
    apfs=query(['/usr/sbin/diskutil','apfs','list','-plist'])
    store=one(volumes['model']['APFSPhysicalStores'])['APFSPhysicalStore']
    match=re.fullmatch(r'(disk[0-9]+)s[0-9]+',store)
    if not match:raise ValueError('unsupported physical store')
    disk=query(['/usr/sbin/diskutil','info','-plist',match[1]])
    registry=query(['/usr/sbin/ioreg','-a','-l','-r','-c','IOBlockStorageDriver'])
    return dict(parse_topology(files,volumes,apfs,registry,disk),raw=raw)
