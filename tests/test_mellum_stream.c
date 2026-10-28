/* CPU-only tests for S4 Mellum expert streaming's host-pure pieces (ds4_mellum_stream.c): no GPU, no
 * model.  Build and run: make test-mellum-stream. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ds4_mellum_stream.h"

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

static void test_slots(void) {
    CHECK(ds4_mellum_stream_slots(6ull << 30) == 979u, "6 GiB must hold 979 experts (got %u)",
          ds4_mellum_stream_slots(6ull << 30));
    CHECK(ds4_mellum_stream_slots(DS4_MELLUM_EXPERT_BYTES - 1u) == 0u, "less than one expert holds none");
    CHECK(ds4_mellum_stream_slots(0) == 0u, "no budget holds none");
}

static void test_config_refuses_zero_cache(void) {
    char err[256] = "";
    uint32_t slots = 7u;
    const int rc = ds4_mellum_stream_config(true, 0, &slots, err, sizeof(err));
    CHECK(rc == -1, "streaming with no cache must be refused (rc %d)", rc);
    CHECK(strstr(err, "--ssd-streaming-cache-experts") != NULL, "the refusal must name the flag (err \"%s\")", err);
}

static void test_config_off_has_no_slots(void) {
    char err[256] = "";
    uint32_t slots = 7u;
    const int rc = ds4_mellum_stream_config(false, 6ull << 30, &slots, err, sizeof(err));
    CHECK(rc == 0 && slots == 0u, "streaming off must configure no slots (rc %d, slots %u)", rc, slots);
}

static void test_config_6gb(void) {
    char err[256] = "";
    uint32_t slots = 0u;
    const int rc = ds4_mellum_stream_config(true, 6ull << 30, &slots, err, sizeof(err));
    CHECK(rc == 0 && slots == 979u, "6GB must configure 979 slots (rc %d, slots %u, err \"%s\")", rc, slots, err);
}

int main(void) {
    test_slots();
    test_config_refuses_zero_cache();
    test_config_off_has_no_slots();
    test_config_6gb();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
