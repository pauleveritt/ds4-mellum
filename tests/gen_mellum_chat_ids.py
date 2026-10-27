# /// script
# requires-python = ">=3.10"
# dependencies = ["jinja2", "tokenizers"]
# ///
"""Write tests/mellum_chat_ids.inc: the token ids Mellum 2.1's own chat template
and tokenizer give for the conversations the engine's chat renderer is tested
with.  The template and tokenizer.json are the model's; the template is rendered
as transformers renders it (an ImmutableSandboxedEnvironment with trim_blocks
and lstrip_blocks, the loopcontrols extension, tojson without ASCII escaping);
the encoder is `tokenizers` with the file's own added tokens, so <|im_start|>,
<think> and the rest encode as their ids.

Usage (from the repository root; SNAP holds the model's chat_template.jinja and
tokenizer.json, JetBrains/Mellum2.1-12B-A2.5B-Thinking)::

    uv run --no-project --with jinja2==3.1.6 --with tokenizers==0.23.3 python -I \\
        tests/gen_mellum_chat_ids.py $SNAP > tests/mellum_chat_ids.inc

The tool-bearing shapes give the template Pi's four tools, read, bash, edit and
write (descriptions and parameter schemas from Pi's MIT source,
@earendil-works/pi-coding-agent, dist/core/tools/{read,bash,edit,write}.js),
as {"type", "function": {"name", "description", "parameters": {"type",
"required", "properties"}, "strict": false}} in that key order.  `tojson` is
transformers' (default separators, insertion order, no key sorting, no HTML
escaping), so the key order inside a property ("type" before "description")
is this file's choice and shows in the text.
"""

import hashlib
import json
import sys
from importlib.metadata import version

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

CASES = [
    # (name, think, messages); think False renders with enable_thinking false
    ("system+user, think on", True,
     [("system", "You are a helpful assistant."), ("user", "Hello, world!")]),
    ("system+user, think off", False,
     [("system", "You are a helpful assistant."), ("user", "Hello, world!")]),
    ("user only, think on", True, [("user", "Write a haiku about autumn.")]),
    # a tool message is a user turn holding one <tool_response> block
    ("system, user, assistant, tool, think on", True,
     [("system", "You are a coding agent."), ("user", "How many files are here?"),
      ("assistant", "Let me check."), ("tool", "42 files\nin 3 directories")]),
    ("tool only, think on", True, [("tool", "42 files")]),
    # consecutive tool messages share one user turn, a <tool_response> block each
    ("user, assistant, two tool messages, think on", True,
     [("user", "Count files and directories."), ("assistant", "Let me check."),
      ("tool", "42 files"), ("tool", "7 directories")]),
]


def _tool(name, description, required, properties):
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "required": required,
                                        "properties": properties},
                         "strict": False}}


def _str(description):
    return {"type": "string", "description": description}


def _num(description):
    return {"type": "number", "description": description}


# Pi's four default tools, in Pi's order
TOOLS = [
    _tool("read",
          "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp)."
          " Images are sent as attachments. For text files, output is truncated to 2000 lines or"
          " 50KB (whichever is hit first). Use offset/limit for large files. When you need the full"
          " file, continue with offset until complete.",
          ["path"],
          {"path": _str("Path to the file to read (relative or absolute)"),
           "offset": _num("Line number to start reading from (1-indexed)"),
           "limit": _num("Maximum number of lines to read")}),
    _tool("bash",
          "Execute a bash command in the current working directory. Returns stdout and stderr."
          " Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated,"
          " full output is saved to a temp file. Optionally provide a timeout in seconds.",
          ["command"],
          {"command": _str("Shell command to execute"),
           "timeout": _num("Timeout in seconds (optional, no default timeout)")}),
    _tool("edit",
          "Edit a single file using exact text replacement. Every edits[].oldText must match a"
          " unique, non-overlapping region of the original file. If two changes affect the same"
          " block or nearby lines, merge them into one edit instead of emitting overlapping edits."
          " Do not include large unchanged regions just to connect distant changes.",
          ["path", "edits"],
          {"path": _str("Path to the file to edit (relative or absolute)"),
           "edits": {"type": "array",
                     "items": {"type": "object", "required": ["oldText", "newText"],
                               "properties": {
                                   "oldText": _str(
                                       "Exact text for one targeted replacement. It must be unique"
                                       " in the original file and must not overlap with any other"
                                       " edits[].oldText in the same call."),
                                   "newText": _str("Replacement text for this targeted edit.")}},
                     "description":
                         "One or more targeted replacements. Each edit is matched against the"
                         " original file, not incrementally. Do not include overlapping or nested"
                         " edits. If two changes touch the same block or nearby lines, merge them"
                         " into one edit instead."}}),
    _tool("write",
          "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
          " Automatically creates parent directories.",
          ["path", "content"],
          {"path": _str("Path to the file to write (relative or absolute)"),
           "content": _str("Content to write to the file")}),
]

# The tool-bearing shapes, thinking on: (2) a system turn with the tools and a user turn; (3) and
# an assistant turn with reasoning and one tool call; (4) and a tool message; (5) and a second
# tool message, which shares the first one's user turn.
TOOL_SYSTEM = "You are a coding agent. Current working directory: /testbed"
TOOL_USER = "Read hello.py and tell me what it prints."
TOOL_REASONING = "The user wants to know what hello.py prints. I should read the file first."
TOOL_CALL_NAME = "read"
TOOL_CALL_ARGS = {"path": "hello.py"}
TOOL_RESULT = 'print("hello, world")\n'
TOOL_RESULT_2 = "second result: 7 files\n"
TOOL_CASES = [
    ("tools: system+user, think on", 2),
    ("tools: +assistant reasoning and one tool call, think on", 3),
    ("tools: +tool message, think on", 4),
    ("tools: +two tool messages, one user turn, think on", 5),
]

# Imported histories: the turns before the last come from a prompt-prefix file (USER: and
# ASSISTANT: turns, ds4_prompt_prefix_append), the last user turn is the live prompt.
HISTORIES = [
    ("history: user, assistant, user, think on", True,
     [("user", "What is 2 + 2?"), ("assistant", "4"), ("user", "And times 3?")]),
    ("history: system, two exchanges, user, think on", True,
     [("system", "You are terse."), ("user", "Name a prime."), ("assistant", "7"),
      ("user", "Another?"), ("assistant", "11"), ("user", "Their sum?")]),
    # reasoning written into an earlier assistant turn: the template drops it (every prefix turn comes
    # before the live prompt, the latest user message)
    ("history: assistant with reasoning, think on", True,
     [("user", "What is 2 + 2?"), ("assistant", "<think>\nA simple sum.\n</think>\n\n4"),
      ("user", "And times 3?")]),
]

# the generation prompt alone, as the template's last block writes it
PREFIXES = [
    ("assistant prefix, think on", True, "<|im_start|>assistant\n"),
    ("assistant prefix, think off", False, "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
]


def cstr(s):
    return json.dumps(s, ensure_ascii=True)


def main(snap):
    tpl_path = snap + "/chat_template.jinja"
    tok_path = snap + "/tokenizer.json"
    tpl_text = open(tpl_path, encoding="utf-8").read()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    tpl = env.from_string(tpl_text)
    tok = Tokenizer.from_file(tok_path)

    def ids(text):
        return tok.encode(text, add_special_tokens=False).ids

    out = []
    p = out.append
    p("/* Generated by tests/gen_mellum_chat_ids.py; do not edit.")
    p(" * chat_template.jinja sha256 %s" % hashlib.sha256(tpl_text.encode()).hexdigest())
    p(" * tokenizer.json sha256 %s" % hashlib.sha256(open(tok_path, "rb").read()).hexdigest())
    p(" * jinja2 %s, tokenizers %s */" % (version("jinja2"), version("tokenizers")))
    for i, (name, think, msgs) in enumerate(CASES):
        messages = [{"role": r, "content": c} for r, c in msgs]
        kw = {} if think else {"enable_thinking": False}
        text = tpl.render(messages=messages, add_generation_prompt=True, **kw)
        p("static const fx_msg fx_msgs_%d[] = {" % i)
        for r, c in msgs:
            p("    {%s, %s}," % (cstr(r), cstr(c)))
        p("};")
        p("static const int fx_ids_%d[] = {%s};" % (i, ", ".join(map(str, ids(text)))))
    for j, (name, think, text) in enumerate(PREFIXES):
        # the prefix must be what the template itself appends for the generation prompt
        kw = {} if think else {"enable_thinking": False}
        tail = tpl.render(messages=[{"role": "user", "content": "x"}], add_generation_prompt=True, **kw)
        assert tail.endswith(text), name
        p("static const int fx_prefix_ids_%d[] = {%s};" % (j, ", ".join(map(str, ids(text)))))
    p("static const fx_case fx_cases[] = {")
    for i, (name, think, msgs) in enumerate(CASES):
        p("    {%s, %s, %d, fx_msgs_%d, (int)(sizeof(fx_ids_%d) / sizeof(int)), fx_ids_%d},"
          % (cstr(name), "true" if think else "false", len(msgs), i, i, i))
    p("};")
    for k, (name, think, msgs) in enumerate(HISTORIES):
        messages = [{"role": r, "content": c} for r, c in msgs]
        kw = {} if think else {"enable_thinking": False}
        text = tpl.render(messages=messages, add_generation_prompt=True, **kw)
        p("static const fx_msg fx_history_msgs_%d[] = {" % k)
        for r, c in msgs:
            p("    {%s, %s}," % (cstr(r), cstr(c)))
        p("};")
        p("static const int fx_history_ids_%d[] = {%s};" % (k, ", ".join(map(str, ids(text)))))
    p("static const fx_case fx_histories[] = {")
    for k, (name, think, msgs) in enumerate(HISTORIES):
        p("    {%s, %s, %d, fx_history_msgs_%d, (int)(sizeof(fx_history_ids_%d) / sizeof(int)), fx_history_ids_%d},"
          % (cstr(name), "true" if think else "false", len(msgs), k, k, k))
    p("};")
    p("static const fx_prefix fx_prefixes[] = {")
    for j, (name, think, text) in enumerate(PREFIXES):
        p("    {%s, %s, (int)(sizeof(fx_prefix_ids_%d) / sizeof(int)), fx_prefix_ids_%d},"
          % (cstr(name), "true" if think else "false", j, j))
    p("};")

    # the tool-bearing shapes: inputs, the template's exact text, and the tokenizer's ids for it
    def lines(text):
        parts = text.split("\n")
        pieces = [x + "\n" for x in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
        return "\n".join("    " + cstr(x) for x in pieces) or '    ""'

    p("static const char fx_tool_system[] = %s;" % cstr(TOOL_SYSTEM))
    p("static const char fx_tool_user[] = %s;" % cstr(TOOL_USER))
    p("static const char fx_tool_reasoning[] = %s;" % cstr(TOOL_REASONING))
    p("static const char fx_tool_call_name[] = %s;" % cstr(TOOL_CALL_NAME))
    p("static const char fx_tool_call_args[] = %s;" % cstr(json.dumps(TOOL_CALL_ARGS, ensure_ascii=False)))
    p("static const char fx_tool_result[] = %s;" % cstr(TOOL_RESULT))
    p("static const char fx_tool_result_2[] = %s;" % cstr(TOOL_RESULT_2))
    for i, (name, nmsg) in enumerate(TOOL_CASES):
        messages = [{"role": "system", "content": TOOL_SYSTEM},
                    {"role": "user", "content": TOOL_USER}]
        if nmsg >= 3:
            messages.append({"role": "assistant", "content": "", "reasoning_content": TOOL_REASONING,
                             "tool_calls": [{"type": "function", "function": {
                                 "name": TOOL_CALL_NAME, "arguments": TOOL_CALL_ARGS}}]})
        if nmsg >= 4:
            messages.append({"role": "tool", "content": TOOL_RESULT})
        if nmsg >= 5:
            messages.append({"role": "tool", "content": TOOL_RESULT_2})
        text = tpl.render(messages=messages, tools=TOOLS, add_generation_prompt=True)
        p("static const char fx_tool_text_%d[] =\n%s;" % (i, lines(text)))
        p("static const int fx_tool_ids_%d[] = {%s};" % (i, ", ".join(map(str, ids(text)))))
    p("static const fx_tool_case fx_tool_cases[] = {")
    for i, (name, nmsg) in enumerate(TOOL_CASES):
        p("    {%s, %d, fx_tool_text_%d, (int)(sizeof(fx_tool_ids_%d) / sizeof(int)), fx_tool_ids_%d},"
          % (cstr(name), nmsg, i, i, i))
    p("};")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
