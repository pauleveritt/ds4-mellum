#!/usr/bin/env python3
"""Convert complete native Mellum routing spools without using a model or GPU."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cache
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
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


@cache
def replay_module():
    # Explicit sibling loading keeps isolated (-I) CLI execution independent of cwd.
    spec = importlib.util.spec_from_file_location("mellum_capture_replay", Path(__file__).with_name("mellum_ssd_replay.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_metadata(path: Path) -> tuple[dict, dict[int, str], dict[int, str]]:
    try:
        with path.open("rb") as source:
            raw = source.read(2**20 + 1)
        if len(raw) > 2**20:
            raise RouteCaptureError("metadata exceeds 1 MiB")
        replay = replay_module()
        data = replay.parse_object(raw.decode("utf-8"), 1)
        header = replay.object_value(data.get("header"), 1, "header")
        catalogs = []
        for name in ("segments", "execution_profiles"):
            catalog = replay.object_value(data.get(name), 1, name)
            if not catalog:
                raise RouteCaptureError(f"{name} must not be empty")
            parsed = {}
            for key, value in catalog.items():
                if (not key.isascii() or not key.isdigit() or len(key) > 10 or
                    int(key) > 2**32 - 1 or str(int(key)) != key):
                    raise RouteCaptureError(f"{name} keys must be canonical decimal u32 IDs")
                if not isinstance(value, str) or not value.strip():
                    raise RouteCaptureError(f"{name}[{key}] must be a nonempty string")
                parsed[int(key)] = value
            catalogs.append(parsed)
        return header, catalogs[0], catalogs[1]
    except (ValueError, RecursionError) as exc:
        raise RouteCaptureError(f"invalid metadata: {exc}") from exc


def _computed(parent: dict, key: str, value: object) -> None:
    # JSON comparison distinguishes true from 1, including within extension objects.
    if key in parent and json.dumps(parent[key], sort_keys=True, allow_nan=False) != json.dumps(value, sort_keys=True, allow_nan=False):
        raise RouteCaptureError(f"computed {key} conflicts with native spool")
    parent[key] = value


def complete_metadata(header: dict, summary: dict, profiles: dict[int, str],
                      profile_counts: dict[int, int]) -> dict:
    try:
        replay = replay_module()
        result = deepcopy(header)
        if type(result.get("schema_version")) is not int or result["schema_version"] != 2:
            raise RouteCaptureError("schema_version must be 2")
        manifest = replay.object_value(result.get("capture_manifest"), 1, "capture_manifest")
        artifacts = replay.object_value(manifest.get("artifacts"), 1, "capture_manifest.artifacts")
        execution = replay.object_value(manifest.get("execution"), 1, "capture_manifest.execution")
        _computed(result, "geometry", asdict(replay.MELLUM))
        _computed(manifest, "complete", True)
        _computed(manifest, "extent", {key: summary[key] for key in ("forwards", "positions", "first_position", "last_position")})
        _computed(artifacts, "token_stream_sha256", summary["token_stream_sha256"])
        _computed(execution, "effective_kernel", "native-profile-catalog")
        _computed(execution, "route_capture", {
            "spool_sha256": summary["spool_sha256"], "cap_tokens": summary["cap_tokens"],
            "profiles": {str(key): profiles[key] for key in sorted(profile_counts)},
            "profile_forward_counts": {str(key): profile_counts[key] for key in sorted(profile_counts)},
        })
        replay.parse_header(result)
        return result
    except (ValueError, RecursionError) as exc:
        raise RouteCaptureError(f"invalid metadata: {exc}") from exc
