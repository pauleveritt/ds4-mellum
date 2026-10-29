/* CPU-only tests for the Mellum double reference's MXFP4 experts (tests/mellum_moe_ref.inc), checked by hand
 * and against the engine's own CPU MXFP4 dot (ds4_vec_dot_mxfp4_f32 in ds4.c): no GPU, no model.  Build and
 * run: make test-mellum-moe-ref. */
#include "../ds4.c"
#include "mellum_moe_ref.inc"

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

static uint32_t g_lcg = 1u;
static uint32_t lcg(void) { g_lcg = g_lcg * 1664525u + 1013904223u; return g_lcg >> 8; }
static double lcg_unit(void) { return (double)lcg() / 16777216.0 * 2.0 - 1.0; }  /* [-1, 1) */

/* one block, e, every nibble 0 but qs[0] */
static void one_block(uint8_t b[17], uint8_t e, uint8_t qs0) {
    memset(b, 0, 17);
    b[0] = e;
    b[1] = qs0;
}

static void test_mxfp4_hand_block(void) {
    uint8_t b[17];
    double x[32] = { 0 };
    x[0] = x[16] = 1.0;
    one_block(b, 127, 0x71);   /* element 0 is 0.5, element 16 is 6 */
    CHECK(mref_mxfp4_dot(b, x, 32) == 6.5, "e 127, 0x71: got %g", mref_mxfp4_dot(b, x, 32));
    one_block(b, 128, 0x71);
    CHECK(mref_mxfp4_dot(b, x, 32) == 13.0, "e 128, 0x71: got %g", mref_mxfp4_dot(b, x, 32));
    one_block(b, 127, 0xF8);   /* element 0 is -0, element 16 is -6 */
    double lo[32] = { 0 }, hi[32] = { 0 };
    lo[0] = 1.0;
    hi[16] = 1.0;
    CHECK(mref_mxfp4_dot(b, lo, 32) == 0.0, "nibble 8 is -0: got %g", mref_mxfp4_dot(b, lo, 32));
    CHECK(mref_mxfp4_dot(b, hi, 32) == -6.0, "nibble 15 is -6: got %g", mref_mxfp4_dot(b, hi, 32));
}

static void test_mxfp4_edge_scales(void) {
    const uint8_t es[] = { 0, 254 };
    for (unsigned i = 0; i < sizeof(es); i++) {
        block_mxfp4 blk;
        memset(&blk, 0, sizeof(blk));
        blk.e = es[i];
        blk.qs[3] = 0x01;          /* element 3 is 0.5: 0.5 * 2^127 fits a float, 6 * 2^127 does not */
        double x[32] = { 0 };
        float xf[32] = { 0 };
        x[3] = 1.0;
        xf[3] = 1.0f;
        const double ref = mref_mxfp4_dot((const uint8_t *)&blk, x, 32);
        const double eng = (double)ds4_vec_dot_mxfp4_f32(32, &blk, xf);
        CHECK(ref == eng && ref != 0.0, "e %u: ref %a, engine %a", es[i], ref, eng);
    }
}

static void fill_mxfp4_row(uint8_t *row, uint32_t n, uint32_t e_lo, uint32_t e_span) {
    for (uint32_t ib = 0; ib < n / 32u; ib++) {
        uint8_t *b = row + (uint64_t)ib * MREF_MXFP4_BLOCK_BYTES;
        b[0] = (uint8_t)(e_lo + lcg() % e_span);
        for (uint32_t j = 0; j < 16u; j++) b[1u + j] = (uint8_t)lcg();
    }
}

/* A 2,304-wide row with scales 2^-3..2^3 and x on a 1/8 grid in [-1, 1): every product is a multiple of 2^-7
 * below 48 and every partial sum below 2^17, so the engine's float sum is exact and must equal the double one. */
static void test_mxfp4_row_matches_engine_dot(void) {
    g_lcg = 1u;
    uint8_t row[(MREF_E / 32u) * MREF_MXFP4_BLOCK_BYTES];
    double x[MREF_E];
    float xf[MREF_E];
    fill_mxfp4_row(row, MREF_E, 124u, 7u);
    for (uint32_t i = 0; i < MREF_E; i++) { xf[i] = (float)((int)(lcg() % 16u) - 8) / 8.0f; x[i] = xf[i]; }
    const double ref = mref_mxfp4_dot(row, x, MREF_E);
    const double eng = (double)ds4_vec_dot_mxfp4_f32((int)MREF_E, (const block_mxfp4 *)row, xf);
    CHECK(ref == eng && ref != 0.0, "2,304-wide row: ref %.17g, engine %.17g", ref, eng);
}

/* A full-size MXFP4 layer from the LCG: mref_moe_row at type 39 gives finite outputs, and each output is the
 * renormalized sum, over the selected experts, of the down rows at MXFP4's strides (1,224 and 476 bytes)
 * applied to SwiGLU of the gate and up rows. */
static void test_moe_row_mxfp4_runs(void) {
    g_lcg = 1u;
    const uint64_t gate_row = (MREF_E / 32u) * MREF_MXFP4_BLOCK_BYTES, down_row = (MREF_FF / 32u) * MREF_MXFP4_BLOCK_BYTES;
    CHECK(gate_row == 1224u && down_row == 476u, "MXFP4 row strides");
    const uint64_t gate_bytes = (uint64_t)MREF_NEXP * MREF_FF * gate_row, down_bytes = (uint64_t)MREF_NEXP * MREF_E * down_row;
    const uint64_t router = 0, gate = MREF_ROUTER_BYTES, up = gate + gate_bytes, down = up + gate_bytes;
    uint8_t *base = malloc(down + down_bytes);
    CHECK(base != NULL, "allocate the layer");
    if (!base) return;
    float *rw = (float *)base;
    for (uint64_t i = 0; i < (uint64_t)MREF_NEXP * MREF_E; i++) rw[i] = (float)(lcg_unit() * 0.05);
    for (uint64_t r = 0; r < (uint64_t)MREF_NEXP * MREF_FF; r++) fill_mxfp4_row(base + gate + r * gate_row, MREF_E, 120u, 15u);
    for (uint64_t r = 0; r < (uint64_t)MREF_NEXP * MREF_FF; r++) fill_mxfp4_row(base + up + r * gate_row, MREF_E, 120u, 15u);
    for (uint64_t r = 0; r < (uint64_t)MREF_NEXP * MREF_E; r++) fill_mxfp4_row(base + down + r * down_row, MREF_FF, 120u, 15u);
    float x[MREF_E];
    double xd[MREF_E];
    for (uint32_t i = 0; i < MREF_E; i++) { x[i] = (float)(lcg_unit() * 0.1); xd[i] = x[i]; }

    const mref_weights w = { .base = base, .router = router, .gate = gate, .up = up, .down = down, .expert_type = 39u };
    static mref_t r;
    mref_moe_row(&w, x, &r);
    bool finite = true;
    for (uint32_t o = 0; o < MREF_E; o++) finite = finite && isfinite(r.out[o]);
    CHECK(finite, "MXFP4 MoE output is finite");

    static double want[MREF_E], mid[MREF_FF];
    for (uint32_t o = 0; o < MREF_E; o++) want[o] = 0.0;
    for (uint32_t s = 0; s < MREF_USED; s++) {
        const uint64_t e = (uint64_t)r.sel[s];
        for (uint32_t f = 0; f < MREF_FF; f++) {
            const double g = mref_mxfp4_dot(base + gate + (e * MREF_FF + f) * gate_row, xd, MREF_E);
            const double u = mref_mxfp4_dot(base + up + (e * MREF_FF + f) * gate_row, xd, MREF_E);
            mid[f] = g / (1.0 + exp(-g)) * u;
        }
        for (uint32_t o = 0; o < MREF_E; o++)
            want[o] += r.w[s] * mref_mxfp4_dot(base + down + (e * MREF_E + o) * down_row, mid, MREF_FF);
    }
    uint32_t same = 0;
    for (uint32_t o = 0; o < MREF_E; o++) same += r.out[o] == want[o];
    CHECK(same == MREF_E, "MXFP4 MoE output at MXFP4 strides: %u of %u equal", same, MREF_E);
    free(base);
}

int main(void) {
    test_mxfp4_hand_block();
    test_mxfp4_edge_scales();
    test_mxfp4_row_matches_engine_dot();
    test_moe_row_mxfp4_runs();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
