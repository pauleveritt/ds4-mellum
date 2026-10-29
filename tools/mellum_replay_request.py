"""Replay one frozen OpenAI chat-completions request against a running ds4-server and time it (S6 phase C's
Pi-shaped workload: a mellum-worker request with its system prompt, tools and file reads).

    python3 -I tools/mellum_replay_request.py URL REQUEST.json OUT.json [--max-tokens N]

The request's messages, tools and chat_template_kwargs are sent as frozen.  For a timing that compares across
machines, sampling is fixed: greedy (temperature 0) and max_tokens N (default 512), streamed, with usage.
Writes OUT.json with time to first token (the first streamed delta, reasoning or content), decode seconds and
t/s over the deltas after it, the server's usage counts and a SHA-256 of the streamed text.  Exit 0, 1 when the
answer is not a complete stream with usage, 2 on bad usage.
"""

import hashlib
import json
import sys
import time
import urllib.request


def main(argv):
    args = argv[1:]
    max_tokens = 512
    if '--max-tokens' in args:
        i = args.index('--max-tokens')
        max_tokens = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 3:
        sys.stderr.write(__doc__)
        return 2
    url, request_path, out_path = args
    body = json.loads(open(request_path).read())
    body.update(stream=True, stream_options={'include_usage': True}, temperature=0, max_tokens=max_tokens)
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.monotonic()
    first = last = None
    chunks, usage, done = 0, None, False
    text = hashlib.sha256()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # a local server: no proxy lookup
    with opener.open(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices') or []:
                delta = choice.get('delta') or {}
                piece = (delta.get('reasoning_content') or '') + (delta.get('content') or '')
                for call in delta.get('tool_calls') or []:
                    piece += json.dumps(call, sort_keys=True)
                if piece:
                    now = time.monotonic()
                    first = now if first is None else first
                    last = now
                    chunks += 1
                    text.update(piece.encode())
    if not done or usage is None or first is None:
        sys.stderr.write('mellum_replay_request: no complete stream with usage (done %s, usage %s, deltas %d)\n'
                         % (done, usage is not None, chunks))
        return 1
    decode_s = last - first
    report = dict(url=url, request=request_path, max_tokens=max_tokens, ttft_s=first - t0, decode_s=decode_s,
                  decode_tps=(chunks - 1) / decode_s if decode_s > 0 else None, chunks=chunks,
                  prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                  text_sha256=text.hexdigest())
    with open(out_path, 'x') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
