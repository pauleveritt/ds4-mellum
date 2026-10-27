"""Read-only macOS topology and timestamped block-driver observations."""
import re


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
