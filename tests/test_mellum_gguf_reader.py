"""Malformed metadata must be rejected before an unbounded I/O request."""
from contextlib import contextmanager
import importlib.util
import io
from pathlib import Path
import struct
import unittest


def reader_type():
    path = Path(__file__).resolve().parents[1] / 'tools/mellum_gguf_reader.py'
    assert path.is_file(), 'bounded GGUF reader is not implemented'
    spec = importlib.util.spec_from_file_location('mellum_gguf_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Reader


@contextmanager
def rejected():
    try:
        yield
    except ValueError:
        pass
    else:
        raise AssertionError('malformed metadata accepted')


def test_scalar_and_string_arrays_skip_exactly():
    Reader = reader_type()
    scalar_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    for kind, width in scalar_sizes.items():
        source = io.BytesIO(b'x' * (width * 3) + b'end')
        reader = Reader(source, len(source.getvalue()))
        reader.skip_value(kind)
        reader.skip_value(kind)
        reader.skip_value(kind)
        assert reader.read(3) == b'end'
    data = struct.pack('<Q', 3) + b'abc'
    reader = Reader(io.BytesIO(data), 11)
    assert reader.string() == 'abc'
    assert reader.position == 11
    data = struct.pack('<IQ', 8, 2) + struct.pack('<Q', 1) + b'a' + struct.pack('<Q', 2) + b'bc' + b'end'
    reader = Reader(io.BytesIO(data), len(data))
    reader.skip_value(9)
    assert reader.read(3) == b'end'
    data = struct.pack('<IQ', 4, 3) + b'x' * 12 + b'end'
    reader = Reader(io.BytesIO(data), len(data))
    reader.skip_value(9)
    assert reader.position == 24
    assert reader.read(3) == b'end'


def test_truncated_string_array_and_unknown_type():
    Reader = reader_type()
    for data, kind in ((struct.pack('<Q', 3) + b'ab', 8),
                       (struct.pack('<IQ', 10, 2) + b'x' * 8, 9),
                       (b'', 13), (struct.pack('<IQ', 13, 0), 9)):
        with rejected():
            Reader(io.BytesIO(data), len(data)).skip_value(kind)


def test_limits_apply_to_skips_before_read_or_allocation():
    Reader = reader_type()
    # A real stream with guarded read/seek boundaries catches unchecked jumps.
    data = struct.pack('<Q', 2**63)
    source = io.BytesIO(data)
    reader = Reader(source, len(data))
    with rejected():
        reader.string()
    assert source.tell() == 8
    source = io.BytesIO(b'')
    reader = Reader(source, 2**30)
    with rejected():
        reader.read(32 * 1024**2 + 1)
    assert source.tell() == 0
    for count in (1048577, 2**64 - 1):
        data = struct.pack('<IQ', 0, count)
        with rejected():
            Reader(io.BytesIO(data), len(data)).skip_value(9)
    with rejected():
        Reader(io.BytesIO(b''), 2**63)
    for n in (-1, True, 2**63):
        with rejected():
            Reader(io.BytesIO(b''), 0).read(n)


def test_nested_array_and_utf8_limits():
    Reader = reader_type()
    data = struct.pack('<IQ', 0, 1) + b'z'
    for _ in range(3):
        data = struct.pack('<IQ', 9, 1) + data
    reader = Reader(io.BytesIO(data), len(data))
    reader.skip_value(9)
    assert reader.position == len(data)
    data = struct.pack('<IQ', 9, 1) + data
    with rejected():
        Reader(io.BytesIO(data), len(data)).skip_value(9)
    for data in (struct.pack('<Q', 1) + b'\xff', struct.pack('<Q', 1048577)):
        with rejected():
            Reader(io.BytesIO(data), len(data)).string()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
