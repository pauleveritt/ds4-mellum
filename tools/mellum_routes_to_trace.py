#!/usr/bin/env python3
"""Convert complete native Mellum routing spools without using a model or GPU."""
from dataclasses import dataclass
import hashlib
import struct
from typing import BinaryIO

U64_MAX = 2**64 - 1


class RouteCaptureError(ValueError):
    """Invalid or incomplete native routing evidence."""


@dataclass(frozen=True, kw_only=True)
class RouteForward:
    forward: int
    first_position: int
    rows: int
    phase: int
    segment_id: int
    execution_id: int
    tokens: bytes
    selected: bytes


class SpoolReader:
    """One-pass, bounded reader; no successful summary before footer and EOF."""

    def __init__(self, source: BinaryIO):
        self.source = source
        self.offset = self.forwards = self.positions = 0
        self.first_position = self.next_position = None
        self.complete = self.failed = False
        self.spool_hash = hashlib.sha256()
        self.token_hash = hashlib.sha256()
        if self._read(8) != b"MLRT0001":
            self._error("unsupported magic or version")
        layers, experts, top_k, self.cap_tokens = struct.unpack("<4I", self._read(16))
        if (layers, experts, top_k) != (28, 64, 8):
            self._error("unsupported geometry")
        if not 1 <= self.cap_tokens <= 1024:
            self._error("cap_tokens must be in 1..1024")

    def _error(self, reason: str) -> None:
        raise RouteCaptureError(f"byte {self.offset}: {reason}")

    def _read(self, count: int) -> bytes:
        data = bytearray()
        while len(data) < count:
            chunk = self.source.read(count - len(data))
            if not chunk:
                self._error("truncated record")
            self.offset += len(chunk)
            self.spool_hash.update(chunk)
            data.extend(chunk)
        return bytes(data)

    def read_forward(self) -> RouteForward | None:
        if self.failed:
            self._error("reader failed; recovery is unsupported")
        if self.complete:
            return None
        try:
            return self._next()
        except (ValueError, OSError):
            self.failed = True
            raise

    def _next(self) -> RouteForward | None:
        tag = self._read(4)
        if tag == b"END1":
            counts = struct.unpack("<QQ", self._read(16))
            if not self.forwards:
                self._error("empty capture")
            if counts != (self.forwards, self.positions):
                self._error("footer counts differ from capture")
            if self.source.read(1):
                self._error("trailing bytes after footer")
            self.complete = True
            return None
        if tag != b"FWD1":
            self._error("unknown frame tag")
        number, position, rows, phase, segment, execution = struct.unpack("<QQIIII", self._read(32))
        if not 1 <= rows <= self.cap_tokens:
            self._error("rows exceed cap or are zero")
        if phase not in (1, 2):
            self._error("invalid phase")
        if phase == 2 and rows != 1:
            self._error("decode requires one row")
        if number != self.forwards:
            self._error("forward sequence must start at zero and continue")
        if number == U64_MAX or position > U64_MAX - rows or self.positions > U64_MAX - rows:
            self._error("position or counter overflow")
        if self.next_position is not None and position != self.next_position:
            self._error("position sequence must continue")
        tokens = self._read(rows * 4)
        selected = self._read(28 * rows * 8 * 4)
        for row in struct.iter_unpack("<8I", selected):
            if any(expert >= 64 for expert in row):
                self._error("expert ID out of range")
            if len(set(row)) != 8:
                self._error("duplicate expert in row")
        self.token_hash.update(tokens)
        if self.first_position is None:
            self.first_position = position
        self.next_position = position + rows
        self.forwards += 1
        self.positions += rows
        return RouteForward(forward=number, first_position=position, rows=rows, phase=phase,
                            segment_id=segment, execution_id=execution, tokens=tokens, selected=selected)

    def summary(self) -> dict:
        if not self.complete:
            self._error("successful footer and EOF required before summary")
        return dict(cap_tokens=self.cap_tokens, forwards=self.forwards, positions=self.positions,
                    first_position=self.first_position, last_position=self.next_position - 1,
                    spool_sha256=self.spool_hash.hexdigest(), token_stream_sha256=self.token_hash.hexdigest())
