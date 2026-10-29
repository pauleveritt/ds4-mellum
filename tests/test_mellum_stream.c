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
    CHECK(ds4_mellum_stream_slots(6ull << 30, DS4_MELLUM_Q8_EXPERT_BYTES) == 979u, "6 GiB must hold 979 experts (got %u)",
          ds4_mellum_stream_slots(6ull << 30, DS4_MELLUM_Q8_EXPERT_BYTES));
    CHECK(ds4_mellum_stream_slots(DS4_MELLUM_Q8_EXPERT_BYTES - 1u, DS4_MELLUM_Q8_EXPERT_BYTES) == 0u, "less than one expert holds none");
    CHECK(ds4_mellum_stream_slots(0, DS4_MELLUM_Q8_EXPERT_BYTES) == 0u, "no budget holds none");
}

static void test_config_refuses_zero_cache(void) {
    char err[256] = "";
    uint32_t slots = 7u;
    const int rc = ds4_mellum_stream_config(true, 0, DS4_MELLUM_Q8_EXPERT_BYTES, &slots, err, sizeof(err));
    CHECK(rc == -1, "streaming with no cache must be refused (rc %d)", rc);
    CHECK(strstr(err, "--ssd-streaming-cache-experts") != NULL, "the refusal must name the flag (err \"%s\")", err);
}

static void test_config_off_has_no_slots(void) {
    char err[256] = "";
    uint32_t slots = 7u;
    const int rc = ds4_mellum_stream_config(false, 6ull << 30, DS4_MELLUM_Q8_EXPERT_BYTES, &slots, err, sizeof(err));
    CHECK(rc == 0 && slots == 0u, "streaming off must configure no slots (rc %d, slots %u)", rc, slots);
}

/* a prefill chunk can need all 64 of a layer's experts at once, and every one must sit in a slab slot: a
 * smaller cache is refused rather than spilling into buffers the kernels never make resident */
static void test_config_refuses_fewer_than_a_layer(void) {
    char err[256] = "";
    uint32_t slots = 7u;
    CHECK(ds4_mellum_stream_config(true, 63u * DS4_MELLUM_Q8_EXPERT_BYTES, DS4_MELLUM_Q8_EXPERT_BYTES, &slots, err, sizeof(err)) == -1 &&
          strstr(err, "64") != NULL, "63 slots must be refused, naming 64 (err \"%s\")", err);
    CHECK(ds4_mellum_stream_config(true, 64u * DS4_MELLUM_Q8_EXPERT_BYTES, DS4_MELLUM_Q8_EXPERT_BYTES, &slots, err, sizeof(err)) == 0 && slots == 64u,
          "64 slots must be accepted (slots %u)", slots);
}

static void test_config_6gb(void) {
    char err[256] = "";
    uint32_t slots = 0u;
    const int rc = ds4_mellum_stream_config(true, 6ull << 30, DS4_MELLUM_Q8_EXPERT_BYTES, &slots, err, sizeof(err));
    CHECK(rc == 0 && slots == 979u, "6GB must configure 979 slots (rc %d, slots %u, err \"%s\")", rc, slots, err);
}

/* S6: the slot count follows the file's expert type; MXFP4 experts are half Q8_0's size */
static void test_expert_bytes_by_type(void) {
    CHECK(ds4_mellum_expert_bytes(8) == 6580224ull && ds4_mellum_expert_bytes(39) == 3290112ull &&
          ds4_mellum_expert_bytes(12) == 0ull, "expert bytes by GGUF type");
    CHECK(DS4_MELLUM_Q8_EXPERT_BYTES == 6580224ull && DS4_MELLUM_MXFP4_EXPERT_BYTES == 3290112ull, "the two sizes");
}

static void test_mxfp4_slots(void) {
    CHECK(ds4_mellum_stream_slots(6ull << 30, DS4_MELLUM_MXFP4_EXPERT_BYTES) == 1958u, "6 GiB MXFP4 (got %u)",
          ds4_mellum_stream_slots(6ull << 30, DS4_MELLUM_MXFP4_EXPERT_BYTES));
    CHECK(ds4_mellum_stream_slots(2684354560ull, DS4_MELLUM_MXFP4_EXPERT_BYTES) == 815u, "2.5 GiB MXFP4 (got %u)",
          ds4_mellum_stream_slots(2684354560ull, DS4_MELLUM_MXFP4_EXPERT_BYTES));
    CHECK(ds4_mellum_stream_slots(1ull << 30, 0) == 0u, "unknown type holds none");
}

static void test_config_mxfp4_minimum(void) {
    char err[256] = "";
    uint32_t slots = 7u;
    CHECK(ds4_mellum_stream_config(true, 63u * DS4_MELLUM_MXFP4_EXPERT_BYTES, DS4_MELLUM_MXFP4_EXPERT_BYTES, &slots,
                                   err, sizeof(err)) == -1 && strstr(err, "210567168") != NULL,
          "63 MXFP4 slots must be refused, naming 64 * 3,290,112 = 210567168 bytes (err \"%s\")", err);
    CHECK(ds4_mellum_stream_config(true, 64u * DS4_MELLUM_MXFP4_EXPERT_BYTES, DS4_MELLUM_MXFP4_EXPERT_BYTES, &slots,
                                   err, sizeof(err)) == 0 && slots == 64u, "64 MXFP4 slots must be accepted (slots %u)", slots);
}

static void test_config_refuses_unknown_expert_type(void) {
    char err[256] = "";
    uint32_t slots = 7u;
    const int rc = ds4_mellum_stream_config(true, 6ull << 30, 0, &slots, err, sizeof(err));
    CHECK(rc == -1 && slots == 0u && strstr(err, "unsupported Mellum expert type") != NULL,
          "expert_bytes 0 must be refused (rc %d, slots %u, err \"%s\")", rc, slots, err);
}

/* ---- decode stop/resume: fake ops replay a script of stops and record every call ---- */

typedef struct {
    ds4_mellum_stream_stop stops[300];
    uint32_t n_stops, next;
    bool load_fails, cancel_after_load, cancelled, begin_fails;
    uint32_t fail_encode_at;   /* 1-based: that encode fails (0: none) */
    uint32_t encodes;
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
    return ++f->encodes != f->fail_encode_at;
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
    const ds4_mellum_stream_ops ops = { f, fake_encode, fake_finish, fake_load, fake_cancelled, NULL, NULL };
    *stops = 12345u;
    return ds4_mellum_stream_decode(&ops, 28u, stops);
}

/* the overlapped protocol: load_begin starts a stop's reads, the resumed pass is encoded while they run, load
 * finishes them, and discard drops an encoded pass that must not run */
static bool fake_load_begin(void *ctx, uint32_t layer, const int32_t *ids, uint32_t n) {
    fake_t *f = ctx;
    char b[64];
    snprintf(b, sizeof(b), "B%u:", layer);
    fake_log(f, b);
    for (uint32_t i = 0; i < n; i++) {
        snprintf(b, sizeof(b), i ? ",%d" : "%d", ids[i]);
        fake_log(f, b);
    }
    fake_log(f, " ");
    return !f->begin_fails;
}

static void fake_discard(void *ctx) {
    fake_log(ctx, "D ");
}

static int fake_run_overlap(fake_t *f, uint32_t *stops) {
    const ds4_mellum_stream_ops ops = { f, fake_encode, fake_finish, fake_load, fake_cancelled, fake_load_begin,
                                        fake_discard };
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

static void test_overlap_encodes_while_loading(void) {
    fake_t f = { 0 };
    fake_stop(&f, 9u, 2u, (const int32_t[]){ 3, 41 });
    fake_stop(&f, 20u, 1u, (const int32_t[]){ 7 });
    uint32_t stops;
    const int rc = fake_run_overlap(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_OK && stops == 2u, "overlap: ok with two stops (rc %d, stops %u)", rc, stops);
    CHECK(!strcmp(f.log, "E0 F B9:3,41 E9x L9:3,41 F B20:7 E20x L20:7 F "),
          "overlap: begin the reads, encode the resumed pass, finish the reads, then commit (log \"%s\")", f.log);
}

static void test_overlap_all_hit_is_unchanged(void) {
    fake_t f = { 0 };
    uint32_t stops;
    const int rc = fake_run_overlap(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_OK && stops == 0u && !strcmp(f.log, "E0 F "),
          "overlap, all-hit: one encode and one finish (rc %d, log \"%s\")", rc, f.log);
}

static void test_overlap_load_failure_discards(void) {
    fake_t f = { 0 };
    f.load_fails = true;
    fake_stop(&f, 9u, 1u, (const int32_t[]){ 3 });
    uint32_t stops;
    const int rc = fake_run_overlap(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_LOAD, "overlap: a failed load returns LOAD (rc %d)", rc);
    CHECK(!strcmp(f.log, "E0 F B9:3 E9x L9:3 D "), "overlap: a failed load discards the encoded pass (log \"%s\")", f.log);
}

static void test_overlap_cancel_discards(void) {
    fake_t f = { 0 };
    f.cancel_after_load = true;
    fake_stop(&f, 9u, 1u, (const int32_t[]){ 3 });
    uint32_t stops;
    const int rc = fake_run_overlap(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_CANCELLED, "overlap: cancellation returns CANCELLED (rc %d)", rc);
    CHECK(!strcmp(f.log, "E0 F B9:3 E9x L9:3 D "), "overlap: cancellation discards the encoded pass (log \"%s\")", f.log);
}

static void test_overlap_begin_failure_encodes_nothing(void) {
    fake_t f = { 0 };
    f.begin_fails = true;
    fake_stop(&f, 9u, 1u, (const int32_t[]){ 3 });
    uint32_t stops;
    const int rc = fake_run_overlap(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_LOAD && !strcmp(f.log, "E0 F B9:3 "),
          "overlap: a failed begin returns LOAD and encodes nothing (rc %d, log \"%s\")", rc, f.log);
}

static void test_overlap_encode_failure_discards(void) {
    /* a failed encode leaves a partly encoded pass: with discard set it is dropped, never committed */
    fake_t f = { 0 };
    f.fail_encode_at = 1u;
    uint32_t stops;
    int rc = fake_run_overlap(&f, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_GPU && !strcmp(f.log, "E0 D "),
          "overlap: a failed first encode is discarded (rc %d, log \"%s\")", rc, f.log);
    fake_t g = { 0 };
    g.fail_encode_at = 2u;
    fake_stop(&g, 9u, 1u, (const int32_t[]){ 3 });
    rc = fake_run_overlap(&g, &stops);
    CHECK(rc == DS4_MELLUM_STREAM_GPU && !strcmp(g.log, "E0 F B9:3 E9x D "),
          "overlap: a failed resumed encode is discarded (rc %d, log \"%s\")", rc, g.log);
}

static void test_overlap_needs_discard(void) {
    fake_t f = { 0 };
    const ds4_mellum_stream_ops ops = { &f, fake_encode, fake_finish, fake_load, fake_cancelled, fake_load_begin, NULL };
    uint32_t stops = 0;
    CHECK(ds4_mellum_stream_decode(&ops, 28u, &stops) == DS4_MELLUM_STREAM_PROTOCOL && f.log[0] == 0,
          "load_begin without discard is a protocol error before any encode (log \"%s\")", f.log);
}

/* ---- streamed prefill: one layer's chunk needs the unique experts its rows chose ---- */

static void test_unique_sorted(void) {
    const int32_t rows[3][8] = { { 5, 1, 9, 63, 2, 3, 4, 6 }, { 1, 5, 9, 0, 7, 8, 10, 11 }, { 63, 63, 0, 1, 2, 3, 4, 5 } };
    int32_t out[64];
    const uint32_t n = ds4_mellum_stream_unique(&rows[0][0], 3u, 8u, out);
    static const int32_t want[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 63 };
    CHECK(n == sizeof(want) / sizeof(want[0]), "unique: 13 distinct experts over three rows (got %u)", n);
    CHECK(n == 13u && memcmp(out, want, sizeof(want)) == 0, "unique: the set comes back sorted");
}

static void test_unique_rejects_out_of_range(void) {
    const int32_t rows[8] = { 0, 1, 2, 3, 4, 5, 6, 64 };
    int32_t out[64];
    CHECK(ds4_mellum_stream_unique(rows, 1u, 8u, out) == UINT32_MAX, "unique: an id of 64 is refused");
    const int32_t neg[8] = { 0, 1, 2, 3, 4, 5, 6, -1 };
    CHECK(ds4_mellum_stream_unique(neg, 1u, 8u, out) == UINT32_MAX, "unique: a negative id is refused");
}

/* S5 grouped layer-major prefill: K from DS4_MELLUM_STREAM_PREFILL_GROUP, 1-64, default 8, else 0 (refuse) */
static void test_prefill_group_setting(void) {
    CHECK(ds4_mellum_stream_prefill_group(NULL) == DS4_MELLUM_STREAM_GROUP_DEFAULT &&
          DS4_MELLUM_STREAM_GROUP_DEFAULT == 8u, "unset must give the default 8");
    CHECK(ds4_mellum_stream_prefill_group("2") == 2u && ds4_mellum_stream_prefill_group("64") == 64u &&
          ds4_mellum_stream_prefill_group("1") == 1u, "1, 2 and 64 must be taken as they are");
    CHECK(ds4_mellum_stream_prefill_group("0") == 0u && ds4_mellum_stream_prefill_group("65") == 0u &&
          ds4_mellum_stream_prefill_group("") == 0u && ds4_mellum_stream_prefill_group("8x") == 0u &&
          ds4_mellum_stream_prefill_group("-1") == 0u, "0, 65, empty, 8x and -1 must refuse (0)");
}

/* rows for the next group: up to k chunks, ending before a one-row last chunk */
static void test_group_rows(void) {
    CHECK(ds4_mellum_stream_group_rows(30620, 1024, 8) == 8192u, "30,620 rows at K 8 start with 8 chunks");
    CHECK(ds4_mellum_stream_group_rows(2200, 1024, 2) == 2048u && ds4_mellum_stream_group_rows(152, 1024, 2) == 152u,
          "2,200 rows at K 2: 2,048 then 152");
    CHECK(ds4_mellum_stream_group_rows(1281, 256, 2) == 512u && ds4_mellum_stream_group_rows(257, 256, 2) == 256u &&
          ds4_mellum_stream_group_rows(1, 256, 2) == 1u, "a one-row last chunk is left out of its group (got %u)",
          ds4_mellum_stream_group_rows(257, 256, 2));
    CHECK(ds4_mellum_stream_group_rows(1025, 1024, 8) == 1024u, "1,025 rows: 1,024, then the one-row tail (got %u)",
          ds4_mellum_stream_group_rows(1025, 1024, 8));
    CHECK(ds4_mellum_stream_group_rows(2200, 1024, 1) == 1024u, "K 1 is one chunk");
    CHECK(ds4_mellum_stream_group_rows(0, 1024, 8) == 0u && ds4_mellum_stream_group_rows(5, 0, 8) == 0u &&
          ds4_mellum_stream_group_rows(5, 1024, 0) == 0u, "no rows, no chunk or no K give 0");
}

int main(void) {
    test_slots();
    test_config_refuses_zero_cache();
    test_config_off_has_no_slots();
    test_config_6gb();
    test_expert_bytes_by_type();
    test_mxfp4_slots();
    test_config_mxfp4_minimum();
    test_config_refuses_unknown_expert_type();
    test_config_refuses_fewer_than_a_layer();
    test_all_hit_one_finish();
    test_stop_resume_at_experts();
    test_stop_layer0_and_layer27();
    test_stop_regression_is_protocol();
    test_same_layer_after_reload_is_protocol();
    test_load_failure();
    test_cancel_between_stops();
    test_stop_cap();
    test_overlap_encodes_while_loading();
    test_overlap_all_hit_is_unchanged();
    test_overlap_load_failure_discards();
    test_overlap_cancel_discards();
    test_overlap_begin_failure_encodes_nothing();
    test_overlap_needs_discard();
    test_overlap_encode_failure_discards();
    test_unique_sorted();
    test_unique_rejects_out_of_range();
    test_prefill_group_setting();
    test_group_rows();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
