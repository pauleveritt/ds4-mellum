#ifndef MELLUM_TRACE_CONTEXT_H
#define MELLUM_TRACE_CONTEXT_H
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

/* -1 is a usage error. NULL preserves the driver's historical default. */
static int mellum_trace_context(const char *value, int tokens) {
    if (tokens < 0 || tokens > INT_MAX - 64) return -1;
    const int minimum = tokens + 64;
    if (!value) return minimum > 4096 ? minimum : 4096;
    if (!value[0]) return -1;
    int context = 0;
    for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
        if (*p < '0' || *p > '9') return -1;
        const int digit = *p - '0';
        if (context > (INT_MAX - digit) / 10) return -1;
        context = context * 10 + digit;
    }
    return context >= minimum ? context : -1;
}
/* MELLUM_TRACE_STREAM_BYTES (S6): NULL is resident (*bytes = 0); a decimal byte count of at least 1 runs
 * streamed with that expert cache (the engine refuses a cache under its 64-slot minimum); anything else,
 * 0 included, is a usage error (-1). */
static int mellum_trace_stream_bytes(const char *value, uint64_t *bytes) {
    *bytes = 0;
    if (!value) return 0;
    if (!value[0]) return -1;
    uint64_t v = 0;
    for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
        if (*p < '0' || *p > '9') return -1;
        const uint64_t digit = (uint64_t)(*p - '0');
        if (v > (UINT64_MAX - digit) / 10u) return -1;
        v = v * 10u + digit;
    }
    if (v == 0) return -1;
    *bytes = v;
    return 0;
}
#endif
