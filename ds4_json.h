/* Small JSON readers for ds4_server.c, to be shared with ds4_agent.c, in the
 * manner of ds4_tool_text.h: static inline, no allocation policy of their own.
 * The including file defines xmalloc and xrealloc (abort on failure) before the
 * #include.  They parse a NUL-terminated buffer through a cursor and give no way
 * to tell "incomplete" from "malformed"; a streaming caller must find the end of
 * a value itself. */
#ifndef DS4_JSON_H
#define DS4_JSON_H

#include <ctype.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} json_sbuf;

static inline void json_sbuf_putc(json_sbuf *b, char c) {
    if (b->len + 2 > b->cap) {
        b->cap = b->cap ? b->cap * 2 : 64;
        b->ptr = xrealloc(b->ptr, b->cap);
    }
    b->ptr[b->len++] = c;
    b->ptr[b->len] = '\0';
}

static inline void json_sbuf_free(json_sbuf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static inline char *json_sbuf_take(json_sbuf *b) {
    char *p = b->ptr;
    if (!p) {
        p = xmalloc(1);
        p[0] = '\0';
    }
    memset(b, 0, sizeof(*b));
    return p;
}

static inline void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static inline bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static inline int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static inline void json_utf8_put(json_sbuf *b, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp <= 0x7f) {
        json_sbuf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        json_sbuf_putc(b, (char)(0xc0 | (cp >> 6)));
        json_sbuf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        json_sbuf_putc(b, (char)(0xe0 | (cp >> 12)));
        json_sbuf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        json_sbuf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        json_sbuf_putc(b, (char)(0xf0 | (cp >> 18)));
        json_sbuf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        json_sbuf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        json_sbuf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static inline bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static inline bool json_string(const char **p, char **out) {
    /* Always define *out. Every failure path below returns false without
     * producing a string, and several callers reparse in place with
     * `free(x); json_string(&p, &x)` (e.g. duplicate JSON keys, the "model"
     * field). Without this, a non-string or malformed value leaves *out
     * holding the just-freed pointer, which a later cleanup frees again --
     * a double-free. Nulling on entry closes that whole class at the root. */
    *out = NULL;
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    json_sbuf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') {
            json_sbuf_putc(&b, (char)c);
            continue;
        }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': json_sbuf_putc(&b, '"'); break;
        case '\\': json_sbuf_putc(&b, '\\'); break;
        case '/': json_sbuf_putc(&b, '/'); break;
        case 'b': json_sbuf_putc(&b, '\b'); break;
        case 'f': json_sbuf_putc(&b, '\f'); break;
        case 'n': json_sbuf_putc(&b, '\n'); break;
        case 'r': json_sbuf_putc(&b, '\r'); break;
        case 't': json_sbuf_putc(&b, '\t'); break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                const char *low_start = *p;
                if (json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                } else {
                    *p = low_start;
                    cp = 0xfffd;
                }
            }
            json_utf8_put(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = json_sbuf_take(&b);
    return true;
fail:
    json_sbuf_free(&b);
    return false;
}

#endif
