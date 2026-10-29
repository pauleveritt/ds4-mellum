/* CPU-only test of the trace driver's streaming setting (tests/mellum_trace_context.h): MELLUM_TRACE_STREAM_BYTES
 * unset runs resident; a decimal byte count runs streamed with that expert cache.  Build and run:
 * make test-mellum-trace-stream. */
#include <stdint.h>
#include <stdio.h>

#include "mellum_trace_context.h"

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do { \
    g_checks++; \
    if (!(cond)) { \
        g_failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

int main(void) {
    uint64_t b = 7;
    CHECK(mellum_trace_stream_bytes(NULL, &b) == 0 && b == 0, "unset is resident (0 bytes), got %llu",
          (unsigned long long)b);
    CHECK(mellum_trace_stream_bytes("2684354560", &b) == 0 && b == 2684354560ull, "2.5 GiB is taken as it is");
    CHECK(mellum_trace_stream_bytes("1", &b) == 0 && b == 1ull, "1 byte is a value (the engine refuses it later)");
    b = 7;
    CHECK(mellum_trace_stream_bytes("", &b) == -1 && mellum_trace_stream_bytes("0", &b) == -1 &&
          mellum_trace_stream_bytes("2.5", &b) == -1 && mellum_trace_stream_bytes("-1", &b) == -1 &&
          mellum_trace_stream_bytes("99999999999999999999", &b) == -1, "empty, 0, 2.5, -1 and an overflow are refused");
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
