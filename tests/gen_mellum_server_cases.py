# /// script
# requires-python = ">=3.10"
# dependencies = ["jinja2", "tokenizers"]
# ///
"""Write tests/mellum_server_cases.inc: OpenAI-shaped chat requests and the text and token ids Mellum 2.1's
own chat template and tokenizer give for them, which ds4-server's Hermes renderer is pinned against
(tests/ds4_test.c).  A request is its messages (role, content, reasoning_content, tool calls with their
arguments as the JSON strings OpenAI clients send), its tools array as JSON text, and whether thinking is
on.  The template is rendered as transformers renders it (gen_mellum_chat_ids.py has the details).

Usage (from the repository root; SNAP holds the model's chat_template.jinja and tokenizer.json)::

    uv run --no-project --with jinja2==3.1.6 --with tokenizers==0.23.3 python -I \\
        tests/gen_mellum_server_cases.py $SNAP > tests/mellum_server_cases.inc
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gen_mellum_chat_ids", HERE / "gen_mellum_chat_ids.py")
_chat_ids = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chat_ids)
TOOLS = _chat_ids.TOOLS


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


SYS = {"role": "system", "content": "You are a coding agent. Current working directory: /testbed"}
CASES = [
    ("system+user, think on", True, None,
     [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello, world!"}]),
    ("system+user, think off", False, None,
     [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello, world!"}]),
    # untrimmed text, and an earlier assistant turn whose reasoning the template drops
    ("history: earlier reasoning dropped", True, None,
     [{"role": "user", "content": "  What is 2 + 2?\n"},
      {"role": "assistant", "content": "\n4", "reasoning_content": "A simple sum."},
      {"role": "user", "content": "And times 3?"}]),
    ("a later system message renders in place", True, None,
     [{"role": "system", "content": "First."}, {"role": "user", "content": "Hi"},
      {"role": "system", "content": "Second."}, {"role": "user", "content": "Again"}]),
    ("tools: system+user", True, TOOLS,
     [SYS, {"role": "user", "content": "Read hello.py and tell me what it prints."}]),
    ("tools: no system message", True, TOOLS,
     [{"role": "user", "content": "List the files."}]),
    # within the tool cycle the reasoning is kept; a call's arguments are the client's JSON string
    ("tools: a call and its result", True, TOOLS,
     [SYS, {"role": "user", "content": "Read hello.py and tell me what it prints."},
      {"role": "assistant", "content": "", "reasoning_content": "I should read the file first.",
       "tool_calls": [call("c1", "read", {"path": "hello.py"})]},
      {"role": "tool", "tool_call_id": "c1", "content": "print(\"hello, world\")\n"}]),
    ("tools: content, two calls, two results", True, TOOLS,
     [SYS, {"role": "user", "content": "Show both files."},
      {"role": "assistant", "content": "Reading both.", "reasoning_content": "Two reads.",
       "tool_calls": [call("c1", "read", {"path": "a.txt"}), call("c2", "read", {"path": "b.txt", "limit": 5})]},
      {"role": "tool", "tool_call_id": "c1", "content": "A"},
      {"role": "tool", "tool_call_id": "c2", "content": "B"}]),
    # a finished tool cycle, then a new question: the cycle's reasoning is dropped
    ("tools: a finished cycle, then a new question", True, TOOLS,
     [SYS, {"role": "user", "content": "Run ls."},
      {"role": "assistant", "content": "", "reasoning_content": "Use bash.",
       "tool_calls": [call("c1", "bash", {"command": "ls"})]},
      {"role": "tool", "tool_call_id": "c1", "content": "hello.py"},
      {"role": "assistant", "content": "One file: hello.py.", "reasoning_content": "Report it."},
      {"role": "user", "content": "Thanks. Now print it."}]),
    # no reasoning_content: within the cycle the template takes the reasoning from the content's think block
    ("tools: reasoning inside the content", True, TOOLS,
     [SYS, {"role": "user", "content": "Run ls."},
      {"role": "assistant", "content": "<think>\nUse bash.\n</think>\n\nRunning it.",
       "tool_calls": [call("c1", "bash", {"command": "ls"})]},
      {"role": "tool", "tool_call_id": "c1", "content": "hello.py"}]),
    # the template appends the generation prompt whenever add_generation_prompt is set, after an assistant turn too
    ("history ending in an assistant turn", True, None,
     [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", "reasoning_content": "r"}]),
    ("tools: a call, think off", False, TOOLS,
     [SYS, {"role": "user", "content": "Run ls."},
      {"role": "assistant", "content": "", "tool_calls": [call("c1", "bash", {"command": "ls -la"})]},
      {"role": "tool", "tool_call_id": "c1", "content": "total 0"}]),
]


def cstr(s):
    return json.dumps(s, ensure_ascii=True)


def main(snap):
    tpl_text = open(snap + "/chat_template.jinja", encoding="utf-8").read()
    tok_path = snap + "/tokenizer.json"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    tpl = env.from_string(tpl_text)
    tok = Tokenizer.from_file(tok_path)
    out = []
    p = out.append
    p("/* Generated by tests/gen_mellum_server_cases.py; do not edit.")
    p(" * chat_template.jinja sha256 %s" % hashlib.sha256(tpl_text.encode()).hexdigest())
    p(" * tokenizer.json sha256 %s */" % hashlib.sha256(open(tok_path, "rb").read()).hexdigest())
    for i, (name, think, tools, msgs) in enumerate(CASES):
        kw = {} if think else {"enable_thinking": False}
        text = tpl.render(messages=msgs, tools=tools, add_generation_prompt=True, **kw)
        for j, m in enumerate(msgs):
            calls = m.get("tool_calls", [])
            if calls:
                p("static const msc_call msc_calls_%d_%d[] = {%s};" % (i, j, ", ".join(
                    "{%s, %s}" % (cstr(c["function"]["name"]), cstr(c["function"]["arguments"])) for c in calls)))
        p("static const msc_msg msc_msgs_%d[] = {" % i)
        for j, m in enumerate(msgs):
            calls = m.get("tool_calls", [])
            p("    {%s, %s, %s, %d, %s}," % (cstr(m["role"]), cstr(m.get("content", "")),
                                            cstr(m["reasoning_content"]) if "reasoning_content" in m else "NULL",
                                            len(calls), "msc_calls_%d_%d" % (i, j) if calls else "NULL"))
        p("};")
        p("static const char msc_text_%d[] = %s;" % (i, cstr(text)))
        p("static const int msc_ids_%d[] = {%s};" % (i, ", ".join(map(str, tok.encode(text, add_special_tokens=False).ids))))
    p("static const char msc_tools_json[] = %s;" % cstr(json.dumps(TOOLS)))
    p("static const msc_case msc_cases[] = {")
    for i, (name, think, tools, msgs) in enumerate(CASES):
        p("    {%s, %s, %s, %d, msc_msgs_%d, msc_text_%d, (int)(sizeof(msc_ids_%d) / sizeof(int)), msc_ids_%d},"
          % (cstr(name), "true" if think else "false", "true" if tools else "false", len(msgs), i, i, i, i))
    p("};")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
