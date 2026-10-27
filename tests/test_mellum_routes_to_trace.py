"""Model-free spool/export tests: python3 -I tests/test_mellum_routes_to_trace.py."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/mellum_routes_to_trace.py"


def tool():
    assert SCRIPT.is_file(), "native spool reader is not implemented"
    name = "mellum_routes_to_trace"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def frame(number=0, position=42, rows=1, phase=2, segment=7, execution=11, tokens=None):
    tokens = list(range(1234, 1234 + rows)) if tokens is None else tokens
    selected = list(range(7, -1, -1)) * (28 * rows)
    return (b"FWD1" + struct.pack("<QQIIII", number, position, rows, phase, segment, execution)
            + struct.pack(f"<{rows}I", *tokens) + struct.pack(f"<{28 * rows * 8}I", *selected))


def spool(*frames, cap=2):
    frames = frames or (frame(),)
    count = sum(struct.unpack_from("<I", item, 20)[0] for item in frames)
    return (b"MLRT0001" + struct.pack("<4I", 28, 64, 8, cap)
            + b"".join(frames) + b"END1" + struct.pack("<QQ", len(frames), count))


@contextmanager
def invalid(reason):
    try:
        yield
    except tool().RouteCaptureError as exc:
        assert "byte " in str(exc) and reason in str(exc), str(exc)
    else:
        raise AssertionError("invalid spool accepted")


def consume(data):
    reader = tool().SpoolReader(io.BytesIO(data))
    frames = []
    while (item := reader.read_forward()) is not None:
        frames.append(item)
    return frames, reader.summary()


def test_reader_preserves_order_and_derives_identities():
    data = spool()
    assert len(data) == 980
    reader = tool().SpoolReader(io.BytesIO(data))
    assert reader.cap_tokens == 2
    with invalid("footer"):
        reader.summary()
    item = reader.read_forward()
    assert (item.forward, item.first_position, item.rows, item.phase, item.segment_id, item.execution_id) == (0, 42, 1, 2, 7, 11)
    assert item.tokens == b"\xd2\x04\0\0"
    assert struct.unpack("<224I", item.selected) == tuple(range(7, -1, -1)) * 28
    assert reader.read_forward() is None
    assert reader.read_forward() is None
    assert reader.summary() == {
        "cap_tokens": 2, "forwards": 1, "positions": 1, "first_position": 42, "last_position": 42,
        "spool_sha256": hashlib.sha256(data).hexdigest(),
        "token_stream_sha256": hashlib.sha256(b"\xd2\x04\0\0").hexdigest(),
    }
    items, summary = consume(spool(frame(phase=1), frame(1, 43, phase=2, tokens=[9])))
    assert [item.phase for item in items] == [1, 2]
    assert summary["positions"] == summary["forwards"] == 2
    assert summary["last_position"] == 43
    assert summary["token_stream_sha256"] == hashlib.sha256(b"\xd2\x04\0\0\x09\0\0\0").hexdigest()
    items, summary = consume(spool(frame(rows=2, phase=1)))
    assert items[0].rows == 2 and summary["positions"] == 2


def test_reader_rejects_all_truncated_prefixes_and_corruption():
    data = spool()
    for end in range(len(data)):
        with invalid("truncated"):
            consume(data[:end])
    for offset, fmt, value, reason in (
        (8, "I", 29, "geometry"), (20, "I", 1025, "cap"), (20, "I", 0, "cap"),
        (28, "Q", 1, "forward"), (36, "Q", 2**64 - 1, "overflow"),
        (44, "I", 0, "rows"), (44, "I", 1025, "rows"), (44, "I", 2**32-1, "rows"),
        (48, "I", 0, "phase"), (64, "I", 64, "expert"), (68, "I", 7, "duplicate"),
        (964, "Q", 2, "counts"), (972, "Q", 2, "counts"),
    ):
        broken = bytearray(data)
        struct.pack_into("<" + fmt, broken, offset, value)
        with invalid(reason):
            consume(broken)
    for broken, reason in ((b"BAD00001" + data[8:], "magic"), (data[:24] + b"BAD1" + data[28:], "tag"),
                           (data + b"x", "trailing"), (spool(frame(rows=2, phase=2)), "decode"),
                           (data[:24] + b"END1" + struct.pack("<QQ", 0, 0), "empty")):
        with invalid(reason):
            consume(broken)
    for second in (frame(2, 43), frame(1, 42), frame(1, 44)):
        with invalid("sequence"):
            consume(spool(frame(), second))


class BoundedStream(io.BytesIO):
    def read(self, size=-1):
        assert 0 <= size <= 921600, f"unbounded/oversized read {size}"
        return super().read(size)

    def seek(self, *args):
        raise AssertionError("spool must be read once without seeking")


def test_reader_bounds_lengths_before_payload_reads():
    broken = bytearray(spool())
    struct.pack_into("<I", broken, 44, 2**32 - 1)
    reader = tool().SpoolReader(BoundedStream(broken))
    with invalid("rows"):
        reader.read_forward()
    data = spool(*(frame(i, 42 + i) for i in range(200)))
    reader = tool().SpoolReader(BoundedStream(data))
    count = 0
    while reader.read_forward() is not None:
        count += 1
    assert count == reader.summary()["forwards"] == 200


def metadata_fixture():
    return {
        "header": {"schema_version": 2, "capture_kind": "synthetic", "model_id": "fixture-model",
                   "engine_revision": "fixture-engine", "numerical_mode": "release", "workload_id": "fixture-workload",
                   "capture_manifest": {
                       "manifest_version": 1, "capture_id": "fixture-capture",
                       "artifacts": {"model_id": "fixture-model", "tokenizer_id": "fixture-tokenizer", "template_id": "fixture-template"},
                       "build": {"engine_revision": "fixture-engine", "source_state_id": "fixture-source",
                                 "binary_sha256": "0" * 64, "recorder_revision": "fixture-recorder"},
                       "execution": {"context_tokens": 8192, "prefill_chunk_tokens": 1024, "numerical_mode": "release",
                                     "requested_kernel": "fixture-requested", "device": "fixture-device", "os": "fixture-os", "capabilities": {}},
                       "generation": {"mode": "teacher-forced", "settings": {"seed": None}},
                       "workload": {"protocol_id": "fixture-protocol", "conversation_id": "fixture-conversation", "repository_id": "fixture-repo",
                                    "split": "tuning", "continuation_kind": "fixture-continuation"},
                       "extension": {"text": "before\u2028after", "nested": [None, True, 1.5]},
                   }},
        "segments": {"7": "fixture-\u2029segment", "8": "append"},
        "execution_profiles": {"11": "fixture-exact", "12": "fixture-tiled"},
    }


@contextmanager
def bad_metadata(reason):
    try:
        yield
    except ValueError as exc:
        assert reason in str(exc), str(exc)
    else:
        raise AssertionError("invalid metadata accepted")


def read_metadata_bytes(raw):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.json"
        path.write_bytes(raw)
        return tool().read_metadata(path)


def test_metadata_limits_catalogs_and_unicode():
    data = metadata_fixture()
    raw = json.dumps(data, ensure_ascii=False).encode()
    header, segments, profiles = read_metadata_bytes(raw + b" " * (2**20 - len(raw)))
    assert header == data["header"]
    assert segments == {7: "fixture-\u2029segment", 8: "append"}
    assert profiles == {11: "fixture-exact", 12: "fixture-tiled"}
    with bad_metadata("1 MiB"):
        read_metadata_bytes(raw + b" " * (2**20 + 1 - len(raw)))
    with bad_metadata("decode"):
        read_metadata_bytes(b"\xff")
    with bad_metadata("duplicate"):
        read_metadata_bytes(b'{"x":1,"x":2}')
    with bad_metadata("finite"):
        read_metadata_bytes(b'{"extension":{"value":1e999}}')
    with bad_metadata("metadata"):
        read_metadata_bytes(b'{"extension":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}')
    for catalog in ("segments", "execution_profiles"):
        for value in (None, [], {}, {"01": "x"}, {"-1": "x"}, {"4294967296": "x"},
                      {"+1": "x"}, {"1": " "}, {"1": 1}, {"\u0661": "x"}):
            broken = deepcopy(data)
            broken[catalog] = value
            with bad_metadata(catalog):
                read_metadata_bytes(json.dumps(broken).encode())
        broken = deepcopy(data)
        del broken[catalog]
        with bad_metadata(catalog):
            read_metadata_bytes(json.dumps(broken).encode())


def completed_metadata():
    header = metadata_fixture()["header"]
    summary = consume(spool())[1]
    return tool().complete_metadata(header, summary, {11: "fixture-exact", 12: "fixture-tiled"}, {11: 1})


def test_completed_metadata_derives_and_preserves_evidence():
    data = metadata_fixture()["header"]
    saved = deepcopy(data)
    result = tool().complete_metadata(data, consume(spool())[1], {11: "fixture-exact", 12: "unused"}, {11: 1})
    assert data == saved and result["capture_kind"] == "synthetic"
    assert result["geometry"] == dict(layers=28, experts_per_layer=64, top_k=8, expert_bytes=6580224)
    manifest = result["capture_manifest"]
    assert manifest["complete"] is True
    assert manifest["extent"] == dict(forwards=1, positions=1, first_position=42, last_position=42)
    assert manifest["artifacts"]["token_stream_sha256"] == hashlib.sha256(b"\xd2\x04\0\0").hexdigest()
    assert manifest["execution"]["effective_kernel"] == "native-profile-catalog"
    assert manifest["execution"]["route_capture"] == {
        "spool_sha256": hashlib.sha256(spool()).hexdigest(), "cap_tokens": 2,
        "profiles": {"11": "fixture-exact"}, "profile_forward_counts": {"11": 1},
    }
    assert manifest["extension"] == saved["capture_manifest"]["extension"]
    data["capture_kind"] = "measured"
    assert tool().complete_metadata(data, consume(spool())[1], {11: "fixture-exact"}, {11: 1})["capture_kind"] == "measured"


def test_completed_metadata_rejects_computed_conflicts():
    complete = completed_metadata()
    paths = [("geometry",), ("capture_manifest", "complete"), ("capture_manifest", "extent"),
             ("capture_manifest", "artifacts", "token_stream_sha256"),
             ("capture_manifest", "execution", "effective_kernel"),
             ("capture_manifest", "execution", "route_capture")]
    assert tool().complete_metadata(complete, consume(spool())[1], {11: "fixture-exact"}, {11: 1}) == complete
    for path in paths:
        broken = deepcopy(complete)
        parent = broken
        for name in path[:-1]:
            parent = parent[name]
        parent[path[-1]] = "conflict"
        with bad_metadata(path[-1]):
            tool().complete_metadata(broken, consume(spool())[1], {11: "fixture-exact"}, {11: 1})
    broken = deepcopy(complete)
    broken["capture_manifest"]["execution"]["route_capture"]["profile_forward_counts"]["11"] = True
    with bad_metadata("route_capture"):
        tool().complete_metadata(broken, consume(spool())[1], {11: "fixture-exact"}, {11: 1})


def test_completed_metadata_requires_s1_contract():
    for path, value in ((("schema_version",), 1), (("model_id",), "other"),
                        (("capture_manifest", "build", "engine_revision"), "other"),
                        (("capture_manifest", "execution", "numerical_mode"), "exact-prefill"),
                        (("workload_id",), None), (("capture_manifest", "generation"), None)):
        broken = metadata_fixture()["header"]
        parent = broken
        for name in path[:-1]:
            parent = parent[name]
        parent[path[-1]] = value
        with bad_metadata(path[-1]):
            tool().complete_metadata(broken, consume(spool())[1], {11: "fixture-exact"}, {11: 1})


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
