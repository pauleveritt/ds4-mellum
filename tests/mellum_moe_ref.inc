/* The double reference for Mellum 2.1's FFN, shared by the model-free kernel tests (tests/test_mellum_kernels.c) and
 * the gated real-weights test in tests/ds4_test.c: the model's router and MoE (as its Hugging Face modeling code
 * computes them) in double: softmax over all experts, top-k by probability (a tie goes to the lower index), renormalize the k, SwiGLU
 * experts, weighted sum.  F32 router, Q8_0 experts, read from the same bytes the kernels read.  Needs <math.h>,
 * <stdbool.h>, <stdint.h>, <string.h> and an arm64 compiler for __fp16 (the file is Darwin-only like its users). */

#define MREF_E 2304u
#define MREF_NEXP 64u
#define MREF_USED 8u
#define MREF_FF 896u
#define MREF_Q8_BLOCK_BYTES 34u
#define MREF_ROW_GATE ((MREF_E / 32u) * MREF_Q8_BLOCK_BYTES)  /* 2,448: a gate or up row */
#define MREF_ROW_DOWN ((MREF_FF / 32u) * MREF_Q8_BLOCK_BYTES) /* 952: a down row */
#define MREF_GATE_BYTES ((uint64_t)MREF_NEXP * MREF_FF * MREF_ROW_GATE)
#define MREF_DOWN_BYTES ((uint64_t)MREF_NEXP * MREF_E * MREF_ROW_DOWN)
#define MREF_ROUTER_BYTES ((uint64_t)MREF_NEXP * MREF_E * sizeof(float))

typedef struct {
    int sel[MREF_USED];
    double w[MREF_USED];       /* renormalized */
    double top_mass;           /* the eight softmax probabilities summed before renormalizing */
    double out[MREF_E];
} mref_t;

/* where the weights are: byte offsets into base */
typedef struct {
    const uint8_t *base;
    uint64_t router, gate, up, down;
} mref_weights;

/* softmax over all logits, then top-k by probability (a tie goes to the lower index), then renormalize */
static inline void mref_route(const double *logit, uint32_t n, uint32_t k, int *sel, double *w, double *top_mass) {
    double mx = logit[0], sum = 0.0, p[512];
    for (uint32_t e = 1; e < n; e++) if (logit[e] > mx) mx = logit[e];
    for (uint32_t e = 0; e < n; e++) { p[e] = exp(logit[e] - mx); sum += p[e]; }
    for (uint32_t e = 0; e < n; e++) p[e] /= sum;
    bool used[512] = { false };
    double top = 0.0;
    for (uint32_t r = 0; r < k; r++) {
        int best = -1;
        for (uint32_t e = 0; e < n; e++) if (!used[e] && (best < 0 || p[e] > p[best])) best = (int)e;
        used[best] = true;
        sel[r] = best;
        w[r] = p[best];
        top += p[best];
    }
    for (uint32_t r = 0; r < k; r++) w[r] /= top;
    if (top_mass) *top_mass = top;
}

static inline double mref_q8_dot(const uint8_t *row, const double *x, uint32_t n) {
    double acc = 0.0;
    for (uint32_t ib = 0; ib < n / 32u; ib++) {
        const uint8_t *b = row + (uint64_t)ib * MREF_Q8_BLOCK_BYTES;
        __fp16 h;
        memcpy(&h, b, 2);
        double s = 0.0;
        for (uint32_t j = 0; j < 32u; j++) s += (double)(int8_t)b[2u + j] * x[ib * 32u + j];
        acc += (double)(float)h * s;
    }
    return acc;
}

static inline void mref_moe_row(const mref_weights *a, const float *x, mref_t *r) {
    double xd[MREF_E], logit[MREF_NEXP], mid[MREF_FF];
    const float *rw = (const float *)(a->base + a->router);
    for (uint32_t i = 0; i < MREF_E; i++) xd[i] = x[i];
    for (uint32_t e = 0; e < MREF_NEXP; e++) {
        double s = 0.0;
        for (uint32_t i = 0; i < MREF_E; i++) s += (double)rw[(uint64_t)e * MREF_E + i] * xd[i];
        logit[e] = s;
    }
    mref_route(logit, MREF_NEXP, MREF_USED, r->sel, r->w, &r->top_mass);
    for (uint32_t i = 0; i < MREF_E; i++) r->out[i] = 0.0;
    for (uint32_t s = 0; s < MREF_USED; s++) {
        const uint64_t e = (uint64_t)r->sel[s];
        for (uint32_t f = 0; f < MREF_FF; f++) {
            const double g = mref_q8_dot(a->base + a->gate + (e * MREF_FF + f) * MREF_ROW_GATE, xd, MREF_E);
            const double u = mref_q8_dot(a->base + a->up + (e * MREF_FF + f) * MREF_ROW_GATE, xd, MREF_E);
            mid[f] = g / (1.0 + exp(-g)) * u;
        }
        for (uint32_t o = 0; o < MREF_E; o++)
            r->out[o] += r->w[s] * mref_q8_dot(a->base + a->down + (e * MREF_E + o) * MREF_ROW_DOWN, mid, MREF_FF);
    }
}
