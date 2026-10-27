#!/usr/bin/env python3
"""Validate and replay Mellum router traces without loading weights or using a GPU."""

from collections.abc import Iterable
from dataclasses import dataclass
import json


class TraceError(ValueError):
    """A trace violates the versioned capture contract."""


@dataclass(frozen=True, kw_only=True)
class Geometry:
    layers: int
    experts_per_layer: int
    top_k: int
    expert_bytes: int


MELLUM = Geometry(layers=28, experts_per_layer=64, top_k=8, expert_bytes=6_580_224)


@dataclass(frozen=True, kw_only=True)
class Header:
    schema_version: int
    capture_kind: str
    model_id: str
    engine_revision: str
    numerical_mode: str
    workload_id: str
    geometry: Geometry


@dataclass(frozen=True, kw_only=True)
class Request:
    forward: int
    segment: str
    phase: str
    layer: int
    positions: tuple[int, ...]
    experts: tuple[tuple[int, ...], ...]
    line: int


@dataclass(frozen=True, kw_only=True)
class Trace:
    header: Header
    requests: tuple[Request, ...]


def fail(line: int, reason: str) -> None:
    raise TraceError(f"line {line}: {reason}")


def object_value(value: object, line: int, name: str) -> dict:
    if not isinstance(value, dict):
        fail(line, f"{name} must be an object")
    return value


def integer(value: object, line: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(line, f"{name} must be an integer >= {minimum}")
    return value


def string_field(data: dict, key: str, line: int) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(line, f"{key} must be a nonempty string")
    return value


def unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant {value}")


def parse_object(text: str, line: int) -> dict:
    try:
        raw = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except ValueError as exc:
        fail(line, f"invalid JSON: {exc}")
    return object_value(raw, line, "record")


def parse_header(raw: dict) -> Header:
    version = integer(raw.get("schema_version"), 1, "schema_version", 1)
    if version != 1:
        fail(1, "unsupported schema_version")
    geometry = object_value(raw.get("geometry"), 1, "geometry")
    shape = Geometry(**{
        key: integer(geometry.get(key), 1, key, 1)
        for key in ("layers", "experts_per_layer", "top_k", "expert_bytes")
    })
    if shape.top_k > shape.experts_per_layer:
        fail(1, "top_k exceeds experts_per_layer")
    fields = {
        key: string_field(raw, key, 1)
        for key in ("capture_kind", "model_id", "engine_revision", "numerical_mode", "workload_id")
    }
    if fields["capture_kind"] not in ("synthetic", "measured"):
        fail(1, "capture_kind must be synthetic or measured")
    if fields["numerical_mode"] not in ("release", "exact-prefill"):
        fail(1, "numerical_mode must be release or exact-prefill")
    if fields["capture_kind"] == "measured" and shape != MELLUM:
        fail(1, "measured traces require Mellum geometry (28/64/8, 6580224 bytes)")
    return Header(schema_version=version, geometry=shape, **fields)


def parse_request(raw: dict, line: int, geometry: Geometry) -> Request:
    number = integer(raw.get("forward"), line, "forward")
    layer = integer(raw.get("layer"), line, "layer")
    segment = string_field(raw, "segment", line)
    phase = raw.get("phase")
    if phase not in ("prefill", "decode"):
        fail(line, "phase must be prefill or decode")
    positions = raw.get("positions")
    if not isinstance(positions, list) or not positions:
        fail(line, "positions must be a nonempty array")
    for position in positions:
        integer(position, line, "positions entry")
    if any(b != a + 1 for a, b in zip(positions, positions[1:])):
        fail(line, "positions must increase consecutively")
    if phase == "decode" and len(positions) != 1:
        fail(line, "decode must have exactly one position")
    rows = raw.get("experts")
    if not isinstance(rows, list) or len(rows) != len(positions):
        fail(line, "expert row count must equal positions count")
    for row in rows:
        if not isinstance(row, list) or len(row) != geometry.top_k:
            fail(line, "each expert row must contain top_k IDs")
        for expert in row:
            integer(expert, line, "expert ID")
            if expert >= geometry.experts_per_layer:
                fail(line, "expert ID out of range")
        if len(set(row)) != len(row):
            fail(line, "duplicate expert ID in row")
    return Request(
        forward=number, segment=segment, phase=phase, layer=layer,
        positions=tuple(positions), experts=tuple(tuple(row) for row in rows), line=line,
    )


def load_trace(lines: Iterable[str]) -> Trace:
    """Validate the entire capture before returning any requests for replay."""
    source = iter(lines)
    try:
        header_text = next(source)
    except StopIteration:
        fail(1, "missing header")
    header = parse_header(parse_object(header_text, 1))
    requests = []
    anchor = None
    previous_end = None
    line = 1
    for line, text in enumerate(source, 2):
        request = parse_request(parse_object(text, line), line, header.geometry)
        expected_forward, expected_layer = divmod(len(requests), header.geometry.layers)
        if request.forward != expected_forward:
            fail(line, f"forward must be {expected_forward}")
        if request.layer != expected_layer:
            fail(line, f"layer must be {expected_layer}")
        if request.layer == 0:
            if previous_end is not None and request.positions[0] != previous_end + 1:
                fail(line, "positions must continue from previous forward")
            anchor = request
            previous_end = request.positions[-1]
        else:
            for name in ("positions", "segment", "phase"):
                if getattr(request, name) != getattr(anchor, name):
                    fail(line, f"{name} differs within forward")
        requests.append(request)
    if not requests:
        fail(line + 1, "no requests after header")
    if len(requests) % header.geometry.layers:
        fail(line + 1, "truncated final forward")
    return Trace(header=header, requests=tuple(requests))
