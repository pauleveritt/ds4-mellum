"""Bounded, streaming GGUF metadata access; no tensor payload allocation."""
import struct
from typing import BinaryIO

LIMIT = 32 * 1024**2
WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, source: BinaryIO, file_size: int):
        if type(file_size) is not int or not 0 <= file_size < 2**63:
            raise ValueError('invalid signed file size')
        self.source = source
        self.bound = min(file_size, LIMIT)
        self.position = source.tell()
        self._end(0)

    def _end(self, n: int) -> int:
        if type(n) is not int or n < 0 or not 0 <= self.position <= self.bound or n > self.bound - self.position:
            raise ValueError('GGUF metadata exceeds file/directory bounds')
        return self.position + n

    def read(self, n: int) -> bytes:
        end = self._end(n)
        result = self.source.read(n)
        if len(result) != n:
            raise ValueError('truncated GGUF metadata')
        self.position = end
        return result

    def _skip(self, n: int) -> None:
        end = self._end(n)
        if self.source.seek(end) != end:
            raise ValueError('GGUF seek failed')
        self.position = end

    def u32(self) -> int:
        return struct.unpack('<I', self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack('<Q', self.read(8))[0]

    def string(self) -> str:
        length = self.u64()
        if length > 1024**2:
            raise ValueError('GGUF string exceeds 1 MiB')
        return self.read(length).decode('utf-8')

    def skip_value(self, kind: int, depth: int = 0) -> None:
        if kind in WIDTHS:
            self._skip(WIDTHS[kind])
        elif kind == 8:
            self.string()
        elif kind == 9:
            if depth >= 4:
                raise ValueError('GGUF array nesting exceeds four')
            element, count = self.u32(), self.u64()
            if count > 1048576:
                raise ValueError('GGUF array count exceeds limit')
            if element not in (*WIDTHS, 8, 9):
                raise ValueError('unknown GGUF array element type')
            if element in WIDTHS:
                self._skip(count * WIDTHS[element])
            else:
                for _ in range(count):
                    self.skip_value(element, depth + 1)
        else:
            raise ValueError(f'unknown GGUF value type: {kind}')
