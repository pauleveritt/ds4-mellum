"""CPU controller events -> native spool -> exporter -> replay; no GPU claims."""
import hashlib
import json
from pathlib import Path
import runpy
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPERS = runpy.run_path(str(ROOT / "tests/test_mellum_routes_to_trace.py"))
BINARY = ROOT / "tests/test_mellum_capture_state"
PROFILE = "profile-v1;head=encoded" + "".join(
    f";{role}:fixture_kernel(tile=32)"
    for role in ("EMBED", "PROJECTION", "ROUTER", "MID", "DOWN", "REDUCE", "ATTENTION", "HEAD")
)


def test_completed_controller_fixture_exports_with_exact_labels_and_profiles() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "capture.bin"
        subprocess.run([str(BINARY), "--emit-fixture", str(source)], check=True)
        data = source.read_bytes()
        assert len(data) == 4688
        frames, summary = HELPERS["consume"](data)
        assert [(f.first_position, f.rows, f.phase, f.segment_id, f.execution_id) for f in frames] == [
            (0, 2, 1, 0, 0), (2, 1, 1, 0, 0), (3, 1, 2, 1, 0), (4, 1, 1, 2, 0),
        ]
        assert b"".join(f.tokens for f in frames) == struct.pack("<5I", 10, 11, 12, 13, 14)
        assert summary["token_stream_sha256"] == hashlib.sha256(struct.pack("<5I", 10, 11, 12, 13, 14)).hexdigest()
        assert summary["forwards"] == 4 and summary["positions"] == 5
        assert Path(str(source) + ".profiles").read_text(encoding="ascii") == PROFILE + "\n"
        metadata = HELPERS["metadata_fixture"]()
        metadata["segments"] = {"0": "initial", "1": "decode", "2": "append"}
        metadata["execution_profiles"] = {"0": PROFILE}
        meta, output = Path(directory) / "meta.json", Path(directory) / "trace.jsonl"
        meta.write_text(json.dumps(metadata), encoding="utf-8")
        HELPERS["tool"]().export_trace(source, meta, output)
        trace, report = HELPERS["replay_output"](output)
        assert [(r.phase, r.segment, r.positions) for r in trace.requests[::28]] == [
            ("prefill", "initial", (0, 1)), ("prefill", "initial", (2,)),
            ("decode", "decode", (3,)), ("prefill", "append", (4,)),
        ]
        assert trace.requests[0].experts == (tuple(range(8)),) * 2
        assert report["evidence"]["provenance_status"] == "synthetic-declared"
        capture = report["trace"]["capture_manifest"]["execution"]["route_capture"]
        assert capture["profiles"] == {"0": PROFILE}
        assert capture["profile_forward_counts"] == {"0": 4}


def test_aborted_controller_fixture_cannot_be_published() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "capture.bin"
        subprocess.run([str(BINARY), "--emit-incomplete", str(source)], check=True)
        with HELPERS["invalid"]("truncated record"):
            HELPERS["consume"](source.read_bytes())
        metadata = HELPERS["metadata_fixture"]()
        metadata["segments"] = {"0": "initial", "1": "decode", "2": "append"}
        metadata["execution_profiles"] = {"0": PROFILE}
        meta, output = Path(directory) / "meta.json", Path(directory) / "trace.jsonl"
        meta.write_text(json.dumps(metadata), encoding="utf-8")
        with HELPERS["export_fails"]():
            HELPERS["tool"]().export_trace(source, meta, output)
        assert not output.exists()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
