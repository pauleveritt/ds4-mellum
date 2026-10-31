"""Checked GGUF v3 tensor directory, including non-expert spans."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import struct
from typing import BinaryIO

_spec = importlib.util.spec_from_file_location('mellum_gguf_reader', Path(__file__).with_name('mellum_gguf_reader.py'))
_reader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reader)
Reader = _reader.Reader


@dataclass(frozen=True, kw_only=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    type: int
    offset: int
    bytes: int


@dataclass(frozen=True, kw_only=True)
class Directory:
    data_start: int
    alignment: int
    tensors: tuple[Tensor, ...]
    architecture: str


def tensor_bytes(dims: tuple[int, ...], kind: int) -> int:
    if not 1 <= len(dims) <= 4 or any(not 0 < dim < 2**63 for dim in dims):
        raise ValueError('invalid tensor dimensions')
    if kind == 8:
        if dims[0] % 32:
            raise ValueError('Q8 row must contain complete blocks')
        size = dims[0] // 32 * 34
    elif kind == 39:   # MXFP4: one E8M0 byte and 16 nibble bytes per 32 values (S6)
        if dims[0] % 32:
            raise ValueError('MXFP4 row must contain complete blocks')
        size = dims[0] // 32 * 17
    elif kind in (0, 1):
        size = dims[0] * (4 if kind == 0 else 2)
    else:
        raise ValueError(f'unsupported GGUF tensor type: {kind}')
    for dim in dims[1:]:
        size *= dim
    if size >= 2**63:
        raise ValueError('tensor size exceeds signed offsets')
    return size


def read_directory(source: BinaryIO, file_size: int) -> Directory:
    r = Reader(source, file_size)
    if r.read(4) != b'GGUF' or r.u32() != 3:
        raise ValueError('monolithic GGUF v3 required')
    count, keys = r.u64(), r.u64()
    if count > 4096 or keys > 65536:
        raise ValueError('GGUF directory count exceeds limit')
    seen = set()
    architecture, alignment = '', 32
    for _ in range(keys):
        name, kind = r.string(), r.u32()
        if name in seen:
            raise ValueError('duplicate metadata key')
        seen.add(name)
        if name == 'general.architecture':
            if kind != 8:
                raise ValueError('architecture must be string')
            architecture = r.string()
        elif name == 'general.alignment':
            if kind != 4:
                raise ValueError('alignment must be uint32')
            alignment = r.u32()
        elif name == 'split.count':
            formats = {0:'<B', 2:'<H', 4:'<I', 10:'<Q'}
            if kind not in formats:
                raise ValueError('unsupported split metadata')
            fmt = formats[kind]
            if struct.unpack(fmt, r.read(struct.calcsize(fmt)))[0] != 1:
                raise ValueError('sharded GGUF unsupported')
        else:
            r.skip_value(kind)
    if architecture != 'mellum':
        raise ValueError('Mellum architecture required')
    if not 0 < alignment <= 1024**2 or alignment & (alignment - 1):
        raise ValueError('invalid GGUF alignment')
    tensors = []
    names = set()
    for _ in range(count):
        name, ndims = r.string(), r.u32()
        if name in names or not 1 <= ndims <= 4:
            raise ValueError('duplicate tensor or invalid dimension count')
        names.add(name)
        dims = tuple(r.u64() for _ in range(ndims))
        kind, offset = r.u32(), r.u64()
        try:
            size = tensor_bytes(dims, kind)
        except ValueError as exc:
            raise ValueError(f'{name}: {exc}') from exc
        if offset % alignment or offset >= 2**63:
            raise ValueError('invalid tensor offset/alignment')
        tensors.append(Tensor(name=name, dims=dims, type=kind, offset=offset, bytes=size))
    start = (r.position + alignment - 1) // alignment * alignment
    r.read(start - r.position)
    tensors = tuple(Tensor(name=t.name, dims=t.dims, type=t.type, offset=start+t.offset, bytes=t.bytes) for t in tensors)
    end = start
    for tensor in sorted(tensors, key=lambda t: t.offset):
        if tensor.offset < end or tensor.offset + tensor.bytes > file_size or tensor.offset + tensor.bytes >= 2**63:
            raise ValueError('overlapping/out-of-file tensor spans')
        end = tensor.offset + tensor.bytes
    return Directory(data_start=start, alignment=alignment, tensors=tensors, architecture=architecture)
