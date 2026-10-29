"""Flatten the frozen workload request into one prompt for tools/mellum_s4_measure (phase C amendment: ds4-server does
not serve Mellum yet).  System prompt, then the tool schemas as JSON, then the transcript in order with each tool call
as JSON, then an instruction to continue.  Usage: python3 -I flatten.py workload-request.json workload-prompt.txt"""
import json
import sys

req = json.load(open(sys.argv[1]))
parts = []
for m in req['messages']:
    if m['role'] == 'system':
        parts.append(m['content'])
parts.append('Tools you can call (JSON schemas):\n' + json.dumps(req['tools'], indent=1, ensure_ascii=False))
parts.append('The conversation so far:')
for m in req['messages']:
    if m['role'] == 'system':
        continue
    text = m.get('content') or ''
    if isinstance(text, list):
        text = '\n'.join(p.get('text', '') for p in text if isinstance(p, dict))
    for call in m.get('tool_calls') or []:
        text += '\n[tool call] ' + json.dumps(call.get('function', call), ensure_ascii=False)
    parts.append('[%s]\n%s' % (m['role'], text.strip()))
parts.append('Continue as the assistant: decide the next step and make it.')
open(sys.argv[2], 'w').write('\n\n'.join(parts) + '\n')
