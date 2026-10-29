"""Bounded operation ledger and consumed-frame agreement for generated capture."""
import hashlib
import importlib.util
from pathlib import Path
import re
import struct

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
metadata=identity.sibling('mellum_capture_metadata')
exporter=identity.sibling('mellum_routes_to_trace')


def read_tokens(path: Path) -> list[int]:
    words=metadata.bounded_read(path,65536).decode('ascii').split()
    if not 1<=len(words)<=8192 or any(not re.fullmatch(r'[0-9]{1,5}',s) or int(s)>=98304 for s in words):
        raise ValueError('invalid generated token file')
    return list(map(int,words))


def read_operations(path: Path) -> list[tuple[str,int,int]]:
    lines=metadata.bounded_read(path,256*1024).decode('ascii').splitlines()
    if not 4<=len(lines)<=5122:raise ValueError('operation count outside bounds')
    result=[];stage=0;counts=[0,0]
    for line in lines:
        match=re.fullmatch(r'([SE]) ([0-3]) (0|[1-9][0-9]{0,4})',line)
        if not match:raise ValueError('invalid operation syntax')
        kind,segment,argument=match[1],int(match[2]),int(match[3])
        if kind=='S':
            if (stage,segment) not in ((0,0),(1,2)) or not 1<=argument<=8192:raise ValueError('sync order differs')
            if stage==1 and not counts[0]:raise ValueError('missing first response')
            stage=1 if segment==0 else 3
        else:
            if stage not in (1,3) or segment!=stage or argument>=98304:raise ValueError('decode order differs')
            counts[stage//2]+=1
            if counts[stage//2]>(4096 if stage==1 else 1024):raise ValueError('generation cap exceeded')
        result.append((kind,segment,argument))
    if stage!=3 or not all(counts):raise ValueError('incomplete conversation')
    return result


def check_schedule(spool: Path,operations: Path,tokens: list[int],cap: int) -> dict:
    if not 1<=len(tokens)<=8192 or any(type(t) is not int or not 0<=t<98304 for t in tokens):raise ValueError('invalid history')
    ops=read_operations(operations);cursor=0;forward=0
    with spool.open('rb') as stream:
        reader=exporter.SpoolReader(stream)
        if reader.cap_tokens!=cap:raise ValueError('spool cap differs')
        for kind,segment,argument in ops:
            end=argument if kind=='S' else cursor+1
            if not cursor<end<=len(tokens) or (kind=='E' and tokens[cursor]!=argument):raise ValueError('operation token/prefix differs')
            while cursor<end:
                rows=min(cap,end-cursor) if kind=='S' else 1
                item=reader.read_forward()
                if item is None or (item.forward,item.first_position,item.rows,item.phase,item.segment_id)!=(forward,cursor,rows,1 if kind=='S' else 2,segment):
                    raise ValueError('frame schedule differs')
                if item.tokens!=struct.pack(f'<{rows}I',*tokens[cursor:cursor+rows]):raise ValueError('frame tokens differ')
                cursor+=rows;forward+=1
        if cursor!=len(tokens) or reader.read_forward() is not None:raise ValueError('extra or incomplete history')
        summary=reader.summary()
    digest=hashlib.sha256(struct.pack(f'<{len(tokens)}I',*tokens)).hexdigest()
    if summary['positions']!=len(tokens) or summary['first_position']!=0 or summary['token_stream_sha256']!=digest:
        raise ValueError('footer history differs')
    return dict(**summary,operations=len(ops))
