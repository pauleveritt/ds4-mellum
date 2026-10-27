# /// script
# requires-python = ">=3.10"
# dependencies = ["jinja2", "tokenizers"]
# ///
"""Write the held-out token files the oracle's per-position caps were set from (MELLUM.md, "Oracle
caps"): four chat-templated prompts unlike the oracle's own three, rendered by Mellum 2.1's chat
template and encoded by its tokenizer.  Each holds an earlier assistant turn, so traced positions
fall in text of the kind the model writes as well as in prompts.

  tests/mellum_heldout_code.tokens     Python, a reply with code, a follow-up question
  tests/mellum_heldout_tool.tokens     Pi's tools, a Hermes tool call, a tool response
  tests/mellum_heldout_unicode.tokens  Italian, Greek, Japanese, Cyrillic, symbols and emoji
  tests/mellum_heldout_long.tokens     exactly 6,000 ids: the opening of ds4_server.c, cut to fit

Usage (from the repository root; SNAP holds the model's chat_template.jinja and tokenizer.json)::

    uv run --no-project --with jinja2==3.1.6 --with tokenizers==0.23.3 python -I \\
        tests/gen_mellum_heldout_tokens.py $SNAP
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

CODE = [
    {"role": "system", "content": "You are a careful Python reviewer."},
    {"role": "user", "content": "Write a function that returns the n-th Fibonacci number without recursion."},
    {"role": "assistant", "reasoning_content": "Iterate with two accumulators; handle n < 0.",
     "content": "```python\ndef fib(n: int) -> int:\n    if n < 0:\n        raise ValueError(\"n must be >= 0\")\n"
                "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\n\n"
                "`fib(10)` is `55`; the loop runs `n` times and keeps two integers."},
    {"role": "user", "content": "Now make it a generator that yields the first n numbers, with a doctest."},
]

TOOL = [
    {"role": "system", "content": "You are a coding agent. Current working directory: /work"},
    {"role": "user", "content": "Find the TODO in src/app.py and fix it."},
    {"role": "assistant", "content": "", "reasoning_content": "I should read the file before editing it.",
     "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": {"path": "src/app.py"}}}]},
    {"role": "tool", "content": "import sys\n\n\ndef main(argv):\n    # TODO: exit 2 when no argument is given\n"
                                "    print(argv[1])\n\n\nif __name__ == \"__main__\":\n    main(sys.argv)\n"},
]

UNICODE = [
    {"role": "system", "content": "Rispondi nella lingua della domanda."},
    {"role": "user", "content": "Perché il cielo è blu? Spiega in due frasi, con la formula ∝ 1/λ⁴."},
    {"role": "assistant", "reasoning_content": "Rayleigh: σ ∝ 1/λ⁴, quindi il blu è diffuso più del rosso.",
     "content": "La luce del Sole è diffusa dalle molecole dell'aria con intensità ∝ 1/λ⁴ (diffusione di "
                "Rayleigh). Il blu (λ ≈ 450 nm) è diffuso circa 4,4 volte più del rosso (λ ≈ 650 nm) 🌤️."},
    {"role": "user", "content": "Ora in giapponese, greco e russo, una frase ciascuno: 空はなぜ青い？ "
                                "Γιατί ο ουρανός είναι μπλε; Почему небо голубое? ✓ ½ € — «grazie»"},
]

LONG_TARGET = 6000
LONG_QUESTION = "Here is the start of a C file. Which function would you test first, and why?\n\n"


def main(snap):
    tpl_text = open(snap + "/chat_template.jinja", encoding="utf-8").read()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    tpl = env.from_string(tpl_text)
    tok = Tokenizer.from_file(snap + "/tokenizer.json")
    header = open("ds4_server.c", encoding="utf-8").read()

    def render(messages, tools=None):
        text = tpl.render(messages=messages, tools=tools, add_generation_prompt=True)
        return tok.encode(text, add_special_tokens=False).ids

    def long_prompt(n_chars):
        return render([{"role": "user", "content": LONG_QUESTION + header[:n_chars]}])

    def exact_long():
        # the longest prefix of the file, in characters, whose rendered prompt fits, then one
        # character at a time until the length is exactly the target
        lo, hi = 0, len(header)
        assert len(long_prompt(hi)) > LONG_TARGET, "ds4_server.c is too short for the long file"
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(long_prompt(mid)) <= LONG_TARGET:
                lo = mid
            else:
                hi = mid - 1
        ids = long_prompt(lo)
        while len(ids) < LONG_TARGET:
            lo += 1
            ids = long_prompt(lo)
        assert len(ids) == LONG_TARGET, len(ids)
        return ids

    tpl_sha = hashlib.sha256(tpl_text.encode()).hexdigest()
    files = [
        ("tests/mellum_heldout_code.tokens", render(CODE)),
        ("tests/mellum_heldout_tool.tokens", render(TOOL, TOOLS)),
        ("tests/mellum_heldout_unicode.tokens", render(UNICODE)),
        ("tests/mellum_heldout_long.tokens", exact_long()),
    ]
    for path, ids in files:
        with open(path, "w") as f:
            f.write("# Generated by tests/gen_mellum_heldout_tokens.py; do not edit.\n")
            f.write("# %d ids; chat_template.jinja sha256 %s\n" % (len(ids), tpl_sha))
            for i in range(0, len(ids), 16):
                f.write(" ".join(map(str, ids[i:i + 16])) + "\n")
        print("wrote %s: %d ids" % (path, len(ids)))


if __name__ == "__main__":
    main(sys.argv[1])
