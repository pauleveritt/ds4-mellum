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

/* ---- decode stop/resume: fake ops replay a script of stops and record every call ---- */

typedef struct {
    ds4_mellum_stream_stop stops[300];
    uint32_t n_stops, next;
    bool load_fails, cancel_after_load, cancelled;
    char log[16384];
} fake_t;

static void fake_log(fake_t *f, const char *s) {
    strncat(f->log, s, sizeof(f->log) - strlen(f->log) - 1u);
}

static bool fake_encode(void *ctx, uint32_t from_layer, bool resume_at_experts) {
    fake_t *f = ctx;
    char b[32];
    snprintf(b, sizeof(b), "E%u%s ", from_layer, resume_at_experts ? "x" : "");
    fake_log(f, b);
    return true;
}

static bool fake_finish(void *ctx, bool *stopped, ds4_mellum_stream_stop *stop) {
    fake_t *f = ctx;
    fake_log(f, "F ");
    *stopped = f->next < f->n_stops;
    if (*stopped) *stop = f->stops[f->next++];
    return true;
}

static bool fake_load(void *ctx, uint32_t layer, const int32_t *ids, uint32_t n) {
    fake_t *f = ctx;
    char b[64];
    snprintf(b, sizeof(b), "L%u:", layer);
    fake_log(f, b);
    for (uint32_t i = 0; i < n; i++) {
        snprintf(b, sizeof(b), i ? ",%d" : "%d", ids[i]);
        fake_log(f, b);
    }
    fake_log(f, " ");
    if (f->cancel_after_load) f->cancelled = true;
    return !f->load_fails;
}

static bool fake_cancelled(void *ctx) {
    return ((fake_t *)ctx)->cancelled;
}

static void fake_stop(fake_t *f, uint32_t layer, uint32_t n, const int32_t *ids) {
    ds4_mellum_stream_stop *s = &f->stops[f->n_stops++];
    memset(s, 0, sizeof(*s));
    s->layer = layer;
    s->n_missing = n;
    for (uint32_t i = 0; i < n; i++) s->missing[i] = ids[i];
}

static int fake_run(fake_t *f, uint32_t *stops) {
    const ds4_mellum_stream_ops ops = { f, fake_encode, fake_finish, fake_load, fake_cancelled };
    *stops = 12345u;
    return ds4_mellum_stream_decode(&ops, 28u, stops);
}

static void test_all_hit_one_finish(void) {
    fake_t f = { 0 };
    uint32_t stops;
    const int rc = fake_run(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_OK && stops == 0u, "all-hit: ok with no stops (rc %d, stops %u)", rc, stops);
    CHECK(!strcmp(f.log, "E0 F "), "all-hit: one encode and one finish (log \"%s\")", f.log);
}

static void test_stop_resume_at_experts(void) {
    fake_t f = { 0 };
    fake_stop(&f, 9u, 2u, (const int32_t[]){ 3, 41 });
    uint32_t stops;
    const int rc = fake_run(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_OK && stops == 1u, "one stop: ok with one stop (rc %d, stops %u)", rc, stops);
    CHECK(!strcmp(f.log, "E0 F L9:3,41 E9x F "), "one stop: load the missing ids, resume at layer 9's experts (log \"%s\")", f.log);
}

static void test_stop_layer0_and_layer27(void) {
    fake_t f = { 0 };
    fake_stop(&f, 0u, 1u, (const int32_t[]){ 5 });
    fake_stop(&f, 27u, 1u, (const int32_t[]){ 63 });
    uint32_t stops;
    const int rc = fake_run(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_OK && stops == 2u, "both ends: ok with two stops (rc %d, stops %u)", rc, stops);
    CHECK(!strcmp(f.log, "E0 F L0:5 E0x F L27:63 E27x F "), "both ends: resume at layer 0 and 27 experts (log \"%s\")", f.log);
}

static void test_stop_regression_is_protocol(void) {
    fake_t f = { 0 };
    fake_stop(&f, 9u, 1u, (const int32_t[]){ 3 });
    fake_stop(&f, 4u, 1u, (const int32_t[]){ 7 });
    uint32_t stops;
    const int rc = fake_run(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_PROTOCOL, "a stop before the previous one is a protocol error (rc %d)", rc);
    CHECK(strstr(f.log, "L4") == NULL, "a regressed stop must not be loaded (log \"%s\")", f.log);
}

static void test_same_layer_after_reload_is_protocol(void) {
    fake_t f = { 0 };
    fake_stop(&f, 9u, 2u, (const int32_t[]){ 3, 41 });
    fake_stop(&f, 9u, 2u, (const int32_t[]){ 3, 41 });
    uint32_t stops;
    const int rc = fake_run(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_PROTOCOL, "the same layer missing an expert it just loaded is a protocol error (rc %d)", rc);
    /* a disjoint second miss at the same layer is fine: another expert was evicted meanwhile */
    fake_t g = { 0 };
    fake_stop(&g, 9u, 1u, (const int32_t[]){ 3 });
    fake_stop(&g, 9u, 1u, (const int32_t[]){ 4 });
    const int rc2 = fake_run(&g, &stops);
    CHECK(rc2 == DS4_MELLUM_STREAM_OK && stops == 2u, "a disjoint second miss at the same layer resumes (rc %d)", rc2);
}

static void test_load_failure(void) {
    fake_t f = { 0 };
    f.load_fails = true;
    fake_stop(&f, 9u, 1u, (const int32_t[]){ 3 });
    uint32_t stops;
    const int rc = fake_run(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_LOAD, "a failed load returns LOAD (rc %d)", rc);
    CHECK(!strcmp(f.log, "E0 F L9:3 "), "a failed load encodes nothing more (log \"%s\")", f.log);
}

static void test_cancel_between_stops(void) {
    fake_t f = { 0 };
    f.cancel_after_load = true;
    fake_stop(&f, 9u, 1u, (const int32_t[]){ 3 });
    uint32_t stops;
    const int rc = fake_run(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_CANCELLED, "cancellation after a load returns CANCELLED (rc %d)", rc);
    CHECK(!strcmp(f.log, "E0 F L9:3 "), "cancellation encodes nothing more (log \"%s\")", f.log);
}

static void test_stop_cap(void) {
    fake_t f = { 0 };
    /* 28 layers x 8 disjoint single-expert stops is the most a token can take; one more is a protocol error */
    for (uint32_t i = 0; i < 225u; i++) fake_stop(&f, i < 224u ? i / 8u : 27u, 1u, (const int32_t[]){ (int32_t)(i % 8u) + (i < 224u ? 0 : 8) });
    uint32_t stops;
    const int rc = fake_run(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_PROTOCOL && f.next == 225u, "the 225th stop is a protocol error (rc %d, finishes %u)", rc, f.next);
}

int main(void) {
    test_slots();
    test_config_refuses_zero_cache();
    test_config_off_has_no_slots();
    test_config_6gb();
    test_all_hit_one_finish();
    test_stop_resume_at_experts();
    test_stop_layer0_and_layer27();
    test_stop_regression_is_protocol();
    test_same_layer_after_reload_is_protocol();
    test_load_failure();
    test_cancel_between_stops();
    test_stop_cap();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
