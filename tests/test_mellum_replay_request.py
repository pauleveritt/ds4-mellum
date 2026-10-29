"""The request replay client (tools/mellum_replay_request.py) against a fake loopback server that streams a fixed
OpenAI chat-completions SSE answer: no model.

    python3 -I tests/test_mellum_replay_request.py    (part of `make test-mellum`)
"""

import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools' / 'mellum_replay_request.py'
SEEN = {}


def chunk(delta=None, usage=None):
    body = {'choices': [{'index': 0, 'delta': delta or {}}]} if delta is not None else {'choices': []}
    if usage:
        body['usage'] = usage
    return ('data: ' + json.dumps(body) + '\n\n').encode()


class Fake(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        SEEN['path'] = self.path
        SEEN['body'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        time.sleep(0.2)                                   # time to first token
        self.wfile.write(chunk({'reasoning_content': 'Think'}))
        self.wfile.flush()
        for _ in range(4):
            time.sleep(0.05)
            self.wfile.write(chunk({'content': 'x'}))
            self.wfile.flush()
        self.wfile.write(chunk(usage={'prompt_tokens': 123, 'completion_tokens': 5}))
        self.wfile.write(b'data: [DONE]\n\n')

    def log_message(self, *args):
        pass


def serve():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run(url, payload, *extra):
    with tempfile.TemporaryDirectory() as d:
        req = Path(d) / 'req.json'
        req.write_text(json.dumps(payload))
        out = Path(d) / 'out.json'
        r = subprocess.run([sys.executable, '-I', str(SCRIPT), url, str(req), str(out), *extra],
                           capture_output=True, text=True, timeout=60)
        return r, (json.loads(out.read_text()) if out.exists() else None)


def test_replay_times_a_streamed_answer_and_fixes_sampling():
    server = serve()
    try:
        payload = {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}], 'tools': [], 'temperature': 1,
                   'max_tokens': 16384, 'stream': False}
        r, report = run('http://127.0.0.1:%d' % server.server_port, payload, '--max-tokens', '64')
    finally:
        server.shutdown()
    assert r.returncode == 0, r.stderr
    assert SEEN['path'] == '/v1/chat/completions'
    body = SEEN['body']
    assert body['stream'] is True and body['temperature'] == 0 and body['max_tokens'] == 64
    assert body['stream_options'] == {'include_usage': True} and body['messages'] == payload['messages']
    assert report['prompt_tokens'] == 123 and report['completion_tokens'] == 5 and report['chunks'] == 5
    assert 0.15 <= report['ttft_s'] < 2.0, report
    assert 0.1 <= report['decode_s'] < 2.0 and report['decode_tps'] == 4 / report['decode_s'], report
    assert len(report['text_sha256']) == 64


def test_replay_refuses_an_answer_without_usage():
    class NoUsage(Fake):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(chunk({'content': 'x'}))
            self.wfile.write(b'data: [DONE]\n\n')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), NoUsage)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        r, report = run('http://127.0.0.1:%d' % server.server_port, {'messages': []})
    finally:
        server.shutdown()
    assert r.returncode == 1 and 'usage' in r.stderr and report is None, (r.returncode, r.stderr)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items())
        if name.startswith('test_') and callable(value)
    )


if __name__ == '__main__':
    unittest.main()
