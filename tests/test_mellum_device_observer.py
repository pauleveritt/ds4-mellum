"""Map fabricated APFS topology without selecting a busy unrelated driver."""
from copy import deepcopy
import importlib.util
import os
import plistlib
import sys
import time
from unittest.mock import patch
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

def test_command_bounds_both_streams_and_reaps_timeout():
    api=module()
    for code,limit,timeout in (("import os;os.write(2,b'x'*10000)",100,.5),
                               ("import time;time.sleep(2)",100,.02)):
        try:api.command([sys.executable,'-c',code],limit=limit,timeout=timeout)
        except api.CommandFailure as exc:
            assert len(exc.evidence['stdout'])+len(exc.evidence['stderr']) <= limit
            assert exc.evidence['returncode'] is not None
        else:raise AssertionError('unbounded metadata command accepted')
    result=api.command([sys.executable,'-c',"print('ok')"])
    assert result['stdout']==b'ok\n' and result['returncode']==0
    assert result['start_ns'] <= result['end_ns'] <= time.clock_gettime_ns(time.CLOCK_MONOTONIC)

def test_snapshot_records_uncertainty_and_raw_bytes():
    api=module();top=api.parse_topology(*fixture())
    counters={name:0 for name in api.COUNTERS}
    root=dict(IORegistryEntryID=11,IORegistryEntryChildren=[dict(IORegistryEntryID=12)],Statistics=counters)
    raw=plistlib.dumps([root]);result=dict(argv=['ioreg'],start_ns=1,end_ns=2,returncode=0,stdout=raw,stderr=b'')
    with patch.object(api,'command',return_value=result):
        sample=api.snapshot(top)
    assert sample['command']['stdout']==raw and sample['command']['start_ns']==1
    assert sample['counters']['Bytes (Read)']==0 and sample['media_id']==12
    for mode in range(5):
        bad=deepcopy(root)
        if mode==0:bad['IORegistryEntryID']=13
        if mode==1:bad['IORegistryEntryChildren'][0]['IORegistryEntryID']=14
        if mode==2:bad['Statistics']['Bytes (Read)']=True
        if mode==3:bad['Statistics']['Bytes (Read)']=-1
        if mode==4:del bad['Statistics']['Bytes (Read)']
        with patch.object(api,'command',return_value=dict(result,stdout=plistlib.dumps([bad]))):
            refused(lambda:api.snapshot(top))
    with patch.object(api,'command',return_value=dict(result,stdout=b'not plist')):
        refused(lambda:api.snapshot(top))

def test_truncated_xml_and_structure_bombs_keep_raw_evidence():
    api=module()
    payloads=(b'<?xml version="1.0"?><plist><dict>',
              b'<?xml version="1.0"?><!DOCTYPE plist [<!ENTITY a "boom">]><plist><string>&a;</string></plist>',
              plistlib.dumps(['x']*20000))
    for raw in payloads:
        result=dict(stdout=raw,stderr=b'',returncode=0)
        try:api.decoded(result)
        except api.CommandFailure as exc:assert exc.evidence['stdout']==raw
        else:raise AssertionError('unbounded/malformed XML accepted')

def test_interrupted_metadata_command_is_killed_and_reaped():
    api=module();children=[];real=api.subprocess.Popen
    def spawn(*args,**kwargs):
        child=real(*args,**kwargs);children.append(child);return child
    try:
        with patch.object(api.subprocess,'Popen',side_effect=spawn),             patch.object(api.selectors.DefaultSelector,'select',side_effect=KeyboardInterrupt):
            try:api.command([sys.executable,'-c','import time;time.sleep(5)'])
            except BaseException:pass
        assert children and children[0].poll() is not None,'interrupted metadata helper left alive'
    finally:
        for child in children:
            if child.poll() is None:child.kill()
            child.wait()

def test_parser_caps_valid_xml_before_materializing_large_structure():
    api=module()
    for raw in (plistlib.dumps(['x']*20000),plistlib.dumps('x'*2**20)):
        try:api.decoded(dict(stdout=raw,stderr=b'',returncode=0))
        except api.CommandFailure:pass
        else:raise AssertionError('oversized decoded structure accepted')

def test_discovery_failure_keeps_all_collected_commands():
    api=module();files,volumes,apfs,registry,disk=fixture();disk['TotalSize']=999
    raw=[dict(stdout=plistlib.dumps(v),stderr=b'',returncode=0) for v in (apfs,disk,registry)]
    with patch.object(api,'volume_for_file',return_value=volumes['model'],create=True),         patch.object(api.sys,'platform','darwin'),patch.object(api,'command',side_effect=raw):
        try:api.discover(files)
        except api.CommandFailure as exc:assert len(exc.evidence['commands'])==3
        else:raise AssertionError('invalid discovery accepted')

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
