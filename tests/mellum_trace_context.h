#ifndef MELLUM_TRACE_CONTEXT_H
#define MELLUM_TRACE_CONTEXT_H
#include <limits.h>
#include <stddef.h>

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
#endif
