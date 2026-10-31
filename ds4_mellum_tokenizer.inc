/* Mellum 2.1, included by ds4.c where this code sits: not a separate translation unit. */

/* Mellum 2.1 pre-tokenization (tokenizer.ggml.pre = "mellum2"), as the
 * snapshot's tokenizer.json `pre_tokenizer` has it: Digits(individual_digits)
 * then ByteLevel(use_regex) with GPT-2's expression
 *   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 * (contractions are case sensitive; a combining mark is neither \p{L} nor
 * \s, so it falls under [^\s\p{L}\p{N}]).  Digits cuts every \p{N} character
 * out as its own piece first, so the expression only ever sees the stretches
 * between them, and `(?!\S)` at a stretch's end holds even when a digit
 * follows.  The classes are the generated ds4_mellum_unicode.inc, derived from
 * the snapshot's own tokenizer: the Qwen tables (the Python `regex` module's
 * Unicode) class 4,644 codepoints as letters that tokenizers 0.23.3 does not. */
#include "ds4_mellum_unicode.inc"

typedef void (*mellum_piece_fn)(void *ud, uint64_t start, uint64_t end);

/* qwen4_char_info with the Mellum classes; `mark` stays false, because a
 * combining mark is only [^\s\p{L}\p{N}] to this expression. */
static qwen4_char_info mellum_char_at(const char *s, uint64_t len, uint64_t pos) {
    qwen4_char_info c;
    memset(&c, 0, sizeof(c));
    if (pos >= len) return c;
    c.valid = true;
    c.cp = utf8_peek_one(s, len, pos, &c.next);
    c.letter = mellum_unicode_letter(c.cp);
    c.number = mellum_unicode_number(c.cp);
    c.space = mellum_unicode_space(c.cp);
    return c;
}

/* The piece at `pos` within the stretch ending at `end` (no \p{N} inside). */
static uint64_t mellum_piece_end(const char *t, uint64_t end, uint64_t pos) {
    const qwen4_char_info cur = mellum_char_at(t, end, pos);
    if (cur.cp == '\'') {
        const uint64_t room = end - pos;
        if (room >= 2) {
            const char c1 = t[pos + 1];
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return pos + 2;
        }
        if (room >= 3) {
            const char c1 = t[pos + 1], c2 = t[pos + 2];
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return pos + 3;
        }
    }

    /* ` ?` takes only a literal space; then a run of one class */
    qwen4_char_info lead = cur;
    if (cur.cp == ' ') lead = mellum_char_at(t, end, cur.next);
    if (lead.valid && lead.letter) {
        uint64_t p = lead.next;
        while (p < end) {
            const qwen4_char_info c = mellum_char_at(t, end, p);
            if (!c.letter) break;
            p = c.next;
        }
        return p;
    }
    if (lead.valid && !lead.space && !lead.letter && !lead.number) {
        uint64_t p = lead.next;
        while (p < end) {
            const qwen4_char_info c = mellum_char_at(t, end, p);
            if (c.space || c.letter || c.number) break;
            p = c.next;
        }
        return p;
    }

    if (cur.space) {
        uint64_t p = pos, last = pos;
        int n = 0;
        while (p < end) {
            const qwen4_char_info c = mellum_char_at(t, end, p);
            if (!c.space) break;
            last = p;
            p = c.next;
            n++;
        }
        /* \s+(?!\S) gives back the last space when a non-space follows */
        return (p < end && n > 1) ? last : p;
    }

    return cur.next > pos ? cur.next : pos + 1;
}

/* Calls fn for each piece, in order; the pieces tile text[0, len). */
static void mellum_segment_pieces(const char *text, uint64_t len, mellum_piece_fn fn, void *ud) {
    uint64_t pos = 0, stretch_end = 0;
    while (pos < len) {
        const qwen4_char_info c = mellum_char_at(text, len, pos);
        if (c.number) {
            fn(ud, pos, c.next);
            pos = c.next;
            continue;
        }
        if (stretch_end <= pos) {
            uint64_t p = pos;
            while (p < len) {
                const qwen4_char_info s = mellum_char_at(text, len, p);
                if (s.number) break;
                p = s.next;
            }
            stretch_end = p;
        }
        const uint64_t e = mellum_piece_end(text, stretch_end, pos);
        fn(ud, pos, e);
        pos = e;
    }
}

typedef struct {
    const ds4_vocab *vocab;
    const char *text;
    token_vec *out;
} mellum_emit;

static void mellum_emit_piece(void *ud, uint64_t start, uint64_t end) {
    mellum_emit *m = ud;
    bpe_emit_piece(m->vocab, (ds4_str){ m->text + start, end - start }, m->out);
}

static void bpe_tokenize_text_mellum(const ds4_vocab *vocab, const char *text, token_vec *out) {
    mellum_emit m = { vocab, text, out };
    mellum_segment_pieces(text, strlen(text), mellum_emit_piece, &m);
}
