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

static inline bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}

static inline bool json_int(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    /* json_number() uses strtod(), which accepts "NaN"/"Infinity". NaN fails
     * every comparison, so a plain `v < 0` clamp would let it through to the
     * (int) cast, which is undefined for NaN; `!(v >= 0)` folds NaN (and
     * negatives) to 0. +/-Infinity are handled by the two clamps. */
    if (!(v >= 0)) v = 0;
    if (v > INT_MAX) v = INT_MAX;
    *out = (int)v;
    return true;
}

static inline bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (json_lit(p, "true")) {
        *out = true;
        return true;
    }
    if (json_lit(p, "false")) {
        *out = false;
        return true;
    }
    return false;
}

/* The request parser only understands the API fields we use and skips the
 * rest.  Skipping is recursive because JSON values nest, so keep an explicit
 * ceiling: without it, a useless ignored field like {"x":[[[...]]]} can spend
 * the whole C stack before the request is rejected. */
#define JSON_MAX_NESTING 256

static inline bool json_skip_value_depth(const char **p, int depth);

static inline bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') {
        (*p)++;
        return true;
    }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static inline bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') {
        (*p)++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static inline bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

static inline bool json_skip_value(const char **p) {
    return json_skip_value_depth(p, 0);
}

static inline bool json_raw_value(const char **p, char **out) {
    json_ws(p);
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    size_t n = (size_t)(*p - start);
    char *s = xmalloc(n + 1);
    memcpy(s, start, n);
    s[n] = '\0';
    *out = s;
    return true;
}

#endif
