"""Map fabricated APFS topology without selecting a busy unrelated driver."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
def module():
    path=ROOT/'tools/mellum_device_observer.py'
    assert path.exists(), 'device observer not implemented'
    spec=importlib.util.spec_from_file_location('observer_test',path)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    return api

def fixture():
    files={k:dict(path='/tmp/'+k,dev=42) for k in ('model','payload')}
    volume=dict(DeviceIdentifier='disk3s5',VolumeUUID='vol',FilesystemType='apfs',APFSContainerReference='disk3',
                APFSPhysicalStores=[dict(APFSPhysicalStore='disk0s2')],observed_st_dev=42,device_rdev=42)
    volumes={k:deepcopy(volume) for k in files}
    apfs=dict(Containers=[dict(ContainerReference='disk3',APFSContainerUUID='container',
        PhysicalStores=[dict(DeviceIdentifier='disk0s2',DiskUUID='store',Size=900)],
        Volumes=[dict(DeviceIdentifier='disk3s5',APFSVolumeUUID='vol')])])
    media=dict(IOObjectClass='IOMedia',IORegistryEntryID=12,**{'BSD Name':'disk0','Whole':True,'Size':1000})
    registry=[dict(IORegistryEntryID=99,Statistics={'Bytes (Read)':999999}),
              dict(IORegistryEntryID=11,IORegistryEntryChildren=[media],Statistics={})]
    disk=dict(DeviceIdentifier='disk0',TotalSize=1000,MediaName='SSD',Whole=True)
    return files,volumes,apfs,registry,disk

def refused(call):
    try:call()
    except (ValueError,OSError):return
    raise AssertionError('invalid observation accepted')

def test_unique_apfs_store_maps_by_media_not_counter_size():
    api=module();d=api.parse_topology(*fixture())
    assert (d['disk'],d['driver_id'],d['media_id'])==('disk0',11,12)
    assert d['container_uuid']=='container' and d['store_uuid']=='store'

def test_multistore_duplicate_driver_wrong_size_and_st_dev_refused():
    api=module()
    for mode in range(5):
        args=fixture();f,v,a,r,d=args
        if mode==0:a['Containers'][0]['PhysicalStores']*=2
        if mode==1:r.append(deepcopy(r[-1]))
        if mode==2:d['TotalSize']=1001
        if mode==3:v['model']['device_rdev']=43
        if mode==4:v['payload']['VolumeUUID']='different'
        refused(lambda:api.parse_topology(*args))

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
