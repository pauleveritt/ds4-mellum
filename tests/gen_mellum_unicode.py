# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers"]
# ///
"""Derive the Mellum content segmenter's Unicode classification from the
reference tokenizer itself, and emit `ds4_mellum_unicode.inc`.

The three predicates below are *probes*, run against pre-tokenizer stages
deserialized out of the snapshot's own `tokenizer.json`, not re-implemented,
not read from `unicodedata`, not read from a Unicode data file:

  number(cp)  a Digits(individual_digits=true) stage **alone** cuts `{cp}{cp}`
              into two pieces. Digits splits every numeric character into its
              own piece and leaves a non-numeric run whole, so the doubled
              string is the discriminating probe (`1{cp}1` is not: a
              non-digit `{cp}` also lands between the two `1`s).
  letter(cp)  a ByteLevel-only pre-tokenizer keeps `a{cp}` as **one** piece,
              i.e. `{cp}` continues ` ?\\p{L}+`.
  space(cp)   `{cp}` is neither of the above and `.{cp}` splits in **two**,
              i.e. `{cp}` is outside `[^\\s\\p{L}\\p{N}]` -- the regex's own
              `\\s`, whatever that build's Unicode tables say it is.

Python's own Unicode tables would not do: the `regex` module's classes (which
the Qwen pre-tokenizer tables follow) call 4,644 codepoints letters that
tokenizers 0.23.3 does not.

The tables are derived from the reference by construction, so the evidence
that they are right is the engine's pieces matching the reference's
(tests/test_mellum_tokenizer), not this generator agreeing with itself.

The emitted file records the `tokenizers` version, the Unicode version that
build reports, the snapshot directory it was derived from, this generator's
`sha256`, the date and each table's row count. **It is never edited by hand.**

Usage (from the repository root; tokenizers 0.23.3, the version the table
was made with)::

    uv run --no-project --with tokenizers==0.23.3 python -I tests/gen_mellum_unicode.py \\
        --snapshot $SNAP --out ds4_mellum_unicode.inc

where $SNAP is a directory holding Mellum 2.1's `tokenizer.json`
(JetBrains/Mellum2.1-12B-A2.5B-Thinking).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date as date_type
from pathlib import Path

MAX_CODEPOINT = 0x10FFFF
SURROGATES = range(0xD800, 0xE000)

TABLES = ("letter", "number", "space")


def is_swept(cp: int) -> bool:
    """U+0000 cannot cross a C string interface; surrogates are not UTF-8."""
    return cp != 0 and cp not in SURROGATES


# --------------------------------------------------------------------------
# The reference's own stages


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    spec["added_tokens"] = []
    return spec


def _stage_only(spec: dict, stage: dict):
    """A pre-tokenizer built from one stage of the reference's own sequence.

    The stage is handed back to `tokenizers` as JSON exactly as the snapshot
    serialized it, so its parameters are the reference's, not this file's.
    """
    from tokenizers import Tokenizer

    one = dict(spec)
    one["pre_tokenizer"] = stage
    return Tokenizer.from_str(json.dumps(one)).pre_tokenizer


def stage_pre_tokenizers(spec: dict):
    """(Digits-only, ByteLevel-only), from the reference's own sequence."""
    pre = spec.get("pre_tokenizer") or {}
    if pre.get("type") != "Sequence":
        raise ValueError(f"pre_tokenizer is {pre.get('type')!r}, expected a Sequence")
    stages = {stage.get("type"): stage for stage in pre.get("pretokenizers", [])}
    missing = [name for name in ("Digits", "ByteLevel") if name not in stages]
    if missing:
        raise ValueError(
            f"pre_tokenizer sequence has no {' and no '.join(missing)} stage"
        )
    return _stage_only(spec, stages["Digits"]), _stage_only(spec, stages["ByteLevel"])


def pieces(pre, text: str) -> list[str]:
    return [text[a:b] for _, (a, b) in pre.pre_tokenize_str(text)]


# --------------------------------------------------------------------------
# The three predicates


def is_number(digits_pre, cp: int) -> bool:
    c = chr(cp)
    return len(pieces(digits_pre, c + c)) > 1


def is_letter(bytelevel_pre, cp: int) -> bool:
    return len(pieces(bytelevel_pre, "a" + chr(cp))) == 1


def is_space(bytelevel_pre, cp: int) -> bool:
    """Outside `[^\\s\\p{L}\\p{N}]`, and not a letter or a number.

    `.{cp}` staying whole means `{cp}` joined the punctuation run; splitting
    means it did not, and for a codepoint that is neither a letter nor a
    number the only remaining alternative is `\\s`.
    """
    c = chr(cp)
    if is_letter(bytelevel_pre, cp):
        return False
    if len(pieces(bytelevel_pre, "1" + c)) == 1:
        # A numeric continuation of ` ?\p{N}+`; not whitespace.
        return False
    return len(pieces(bytelevel_pre, "." + c)) > 1


def classify(digits_pre, bytelevel_pre, cp: int) -> str | None:
    if is_number(digits_pre, cp):
        return "number"
    if is_letter(bytelevel_pre, cp):
        return "letter"
    if is_space(bytelevel_pre, cp):
        return "space"
    return None


# --------------------------------------------------------------------------
# Ranges


def coalesce(codepoints: list[int]) -> list[tuple[int, int]]:
    """Minimal sorted ranges: adjacent codepoints merge, never neighbour."""
    ranges: list[tuple[int, int]] = []
    for cp in sorted(codepoints):
        if ranges and cp == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], cp)
        elif not ranges or cp > ranges[-1][1]:
            ranges.append((cp, cp))
    return ranges


def derive(
    digits_pre, bytelevel_pre, lo: int, hi: int
) -> dict[str, list[tuple[int, int]]]:
    found: dict[str, list[int]] = {name: [] for name in TABLES}
    for cp in range(lo, hi + 1):
        if not is_swept(cp):
            continue
        name = classify(digits_pre, bytelevel_pre, cp)
        if name is not None:
            found[name].append(cp)
    return {name: coalesce(found[name]) for name in TABLES}


# --------------------------------------------------------------------------
# Emission


def generator_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def tokenizers_version() -> str:
    import tokenizers

    return tokenizers.__version__


def unicode_version() -> str:
    """What the `tokenizers` build reports, or `unknown` with the reason."""
    import tokenizers

    for attr in ("__unicode_version__", "unicode_version"):
        value = getattr(tokenizers, attr, None)
        if value:
            return str(value)
    return (
        f"unknown (tokenizers {tokenizers_version()} exposes no Unicode version; "
        "the tables ARE this build's behaviour and their row counts and this "
        "file's sha256 are the fingerprint)"
    )


def render_table(name: str, ranges: list[tuple[int, int]]) -> str:
    rows = [f"    {{0x{lo:04X}, 0x{hi:04X}}}," for lo, hi in ranges]
    if not rows:
        rows = ["    {0x0001, 0x0000}, /* empty: lo > hi never matches */"]
    body = "\n".join(rows)
    return (
        f"static const mellum_unicode_range mellum_{name}_ranges[] = {{\n{body}\n}};\n"
    )


def render(
    tables: dict[str, list[tuple[int, int]]],
    *,
    snapshot: str,
    generated: str,
    codepoints: str,
) -> str:
    counts = "\n".join(f" * {name}-ranges: {len(tables[name])}" for name in TABLES)
    out = f"""/* ds4_mellum_unicode.inc -- GENERATED FILE, NEVER EDITED BY HAND.
 *
 * The Mellum *content* segmenter's letter / number / whitespace
 * classification, derived from the reference tokenizer's own behaviour by
 * tests/gen_mellum_unicode.py.  Regenerate it; do not patch it.
 *
 * The three predicates, each a probe of the snapshot's own pre-tokenizer
 * stages -- never unicodedata, never a Unicode data file:
 *   number  a Digits(individual_digits=true) stage alone cuts "<cp><cp>" in two
 *   letter  a ByteLevel-only pre-tokenizer keeps "a<cp>" as one piece
 *   space   neither of those, and "."<cp> splits in two
 *
 * These tables are used ONLY by the Mellum pre-tokenizer
 * (mellum_char_at in ds4.c).  Every other family keeps its own classifiers
 * byte for byte.
 *
 * tokenizers: {tokenizers_version()}
 * unicode-version: {unicode_version()}
 * snapshot: {snapshot}
 * codepoints: {codepoints}
 * generator-sha256: {generator_sha256()}
 * generated: {generated}
{counts}
 */

typedef struct {{
    uint32_t lo, hi;
}} mellum_unicode_range;

static bool mellum_unicode_in(const mellum_unicode_range *ranges, size_t n,
                              uint32_t cp) {{
    size_t lo = 0, hi = n;
    while (lo < hi) {{
        const size_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid].lo)      hi = mid;
        else if (cp > ranges[mid].hi) lo = mid + 1;
        else                          return true;
    }}
    return false;
}}

"""
    for name in TABLES:
        out += render_table(name, tables[name])
        out += "\n"
    for name in TABLES:
        out += (
            f"static bool mellum_unicode_{name}(uint32_t cp) {{\n"
            f"    return mellum_unicode_in(mellum_{name}_ranges,\n"
            f"                             sizeof(mellum_{name}_ranges) /\n"
            f"                                 sizeof(mellum_{name}_ranges[0]), cp);\n"
            f"}}\n\n"
        )
    return out.rstrip("\n") + "\n"  # one newline at the end, no blank line


# --------------------------------------------------------------------------


def parse_codepoint_range(text: str) -> tuple[int, int]:
    lo_s, _, hi_s = text.partition("-")
    if not hi_s:
        raise ValueError(f"--codepoints wants FROM-TO, got {text!r}")
    lo, hi = int(lo_s, 0), int(hi_s, 0)
    if lo < 0 or hi > MAX_CODEPOINT or lo > hi:
        raise ValueError(f"--codepoints {text!r} is not inside 0x0-0x10FFFF")
    return lo, hi


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", type=Path, help="the model snapshot directory")
    ap.add_argument(
        "--tokenizer", type=Path, help="a tokenizer.json (overrides --snapshot)"
    )
    ap.add_argument("--out", required=True, type=Path, help="the .inc to write")
    ap.add_argument(
        "--codepoints",
        default="0x0000-0x10FFFF",
        help="FROM-TO, default the whole space",
    )
    ap.add_argument("--date", help="the header's date (default: today)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not args.snapshot and not args.tokenizer:
        print("gen_mellum_unicode: need --snapshot or --tokenizer", file=sys.stderr)
        return 2
    tokenizer_path = args.tokenizer or (args.snapshot / "tokenizer.json")
    try:
        lo, hi = parse_codepoint_range(args.codepoints)
        spec = load_spec(tokenizer_path)
        digits_pre, bytelevel_pre = stage_pre_tokenizers(spec)
        tables = derive(digits_pre, bytelevel_pre, lo, hi)
        text = render(
            tables,
            # The directory's name, not the file's target: tokenizer.json is a
            # symlink into the hub's blob store, and resolving it would record
            # "blobs" instead of the snapshot.
            snapshot=Path(tokenizer_path).parent.resolve().name,
            generated=args.date or date_type.today().isoformat(),
            codepoints=args.codepoints,
        )
        args.out.write_text(text, encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        print(f"gen_mellum_unicode: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        for name in TABLES:
            print(f"{name}-ranges: {len(tables[name])}")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
