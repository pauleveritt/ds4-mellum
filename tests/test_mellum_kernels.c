/* Model-free GPU kernel tests for Mellum 2.1: each Mellum kernel against a
 * double reference, with the control that proves the test can fail.
 * Build and run: make test-mellum-kernels (macOS; no model, no session).
 *
 * q/k norm + RoPE: per-head RMS norm and NeoX RoPE, plain on sliding layers
 * and YaRN on full layers.  The reference is llama.cpp's YaRN formulas
 * (rope_yarn_corr_dims and the ramp blend) in double.  Then attention: the
 * sliding window over a ring, the ring commit, and the full layers' ungated
 * attention on upstream's qwen4 decode kernels.  Then the MoE: top-8 of 64
 * routed experts, no shared expert.  Last, every op the forward pass runs over
 * a chunk's rows against the same op one row at a time, in raw bits. */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#define MELLUM_HEAD_DIM 128u
#define MELLUM_N_HEAD 32u
#define MELLUM_N_HEAD_KV 4u
#define MELLUM_MSCALE 1.2772589f /* 1 + 0.1 ln 16; the GGUF's yarn_attn_factor, already the mscale */

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

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static uint32_t g_rng = 0x9e3779b9u;

static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xffffffu) / 8388608.0f) - 1.0f;
}

static float *rand_vec(uint64_t n, float scale) {
    float *v = malloc(n * sizeof(float));
    require_ok(v != NULL, "malloc");
    for (uint64_t i = 0; i < n; i++) v[i] = scale * frand();
    return v;
}

static ds4_gpu_tensor *upload(const float *data, uint64_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    require_ok(t != NULL, "tensor alloc");
    require_ok(ds4_gpu_tensor_write(t, 0, data, n * sizeof(float)), "tensor write");
    return t;
}

static float *download(const ds4_gpu_tensor *t, uint64_t n) {
    float *out = malloc(n * sizeof(float));
    require_ok(out != NULL, "malloc");
    require_ok(ds4_gpu_tensor_read(t, 0, out, n * sizeof(float)), "tensor read");
    return out;
}

/* ---- the weight arena: an anonymous mmap standing in for the model map ---- */

typedef struct {
    uint8_t *base;
    uint64_t size;
    uint64_t used;
} arena_t;

static uint64_t arena_f32(arena_t *a, uint64_t n, double **shadow, float lo, float hi) {
    const uint64_t off = (a->used + 63u) & ~63ull;
    require_ok(off + n * sizeof(float) <= a->size, "arena");
    a->used = off + n * sizeof(float);
    float *w = (float *)(a->base + off);
    *shadow = malloc(n * sizeof(double));
    require_ok(*shadow != NULL, "malloc");
    for (uint64_t i = 0; i < n; i++) {
        w[i] = lo + (hi - lo) * (0.5f * frand() + 0.5f);
        (*shadow)[i] = w[i];
    }
    return off;
}

/* ---- the reference ---- */

/* The model's rope parameters, as a layer kind selects them: full layers
 * freq_scale 1/16, ext_factor 1, attn_factor 1.0, beta 32/1, n_ctx_orig 8192,
 * base 5e5, NeoX; sliding layers plain rope over base 5e5. */
static ds4_gpu_mellum_rope model_rope(float attn_factor) {
    ds4_gpu_mellum_rope p;
    memset(&p, 0, sizeof(p));
    p.n_head_q = MELLUM_N_HEAD;
    p.n_head_k = MELLUM_N_HEAD_KV;
    p.head_dim = MELLUM_HEAD_DIM;
    p.eps = 1e-6f;
    p.freq_base_sliding = 500000.0f;
    p.freq_base_full = 500000.0f;
    p.freq_scale = 1.0f / 16.0f;
    p.ext_factor = 1.0f;
    p.attn_factor = attn_factor;
    p.beta_fast = 32.0f;
    p.beta_slow = 1.0f;
    p.n_ctx_orig = 8192u;
    return p;
}

/* llama.cpp rope_yarn_corr_dim */
static double corr_dim(double head_dim, double n_ctx_orig, double n_rot, double base) {
    return head_dim * log(n_ctx_orig / (n_rot * 2.0 * M_PI)) / (2.0 * log(base));
}

/* The frequency the angle multiplies, theta(pos) = pos * freq[j]: plain
 * base^(-2j/D), or YaRN's blend of the interpolated and extrapolated angle
 * (linear in pos, so one number per pair).  `mscale` is the cos/sin scale
 * YaRN applies (1 + 0.1 ln 16 on a full layer). */
static void ref_freqs(bool full, double base, double *freq, double *mscale) {
    const double D = MELLUM_HEAD_DIM;
    const double low = fmax(0.0, floor(corr_dim(D, 8192.0, 32.0, base)));
    const double high = fmin(D - 1.0, ceil(corr_dim(D, 8192.0, 1.0, base)));
    for (uint32_t j = 0; j < MELLUM_HEAD_DIM / 2u; j++) {
        const double extrap = pow(base, -2.0 * j / D);
        if (!full) {
            freq[j] = extrap;
            continue;
        }
        const double interp = extrap / 16.0;
        const double y = (j - low) / fmax(0.001, high - low);
        const double ramp_mix = 1.0 - fmin(1.0, fmax(0.0, y)); /* ext_factor == 1 */
        freq[j] = interp * (1.0 - ramp_mix) + extrap * ramp_mix;
    }
    *mscale = full ? (double)MELLUM_MSCALE : 1.0;
}

/* rms norm per head times w (eps 1e-6), then NeoX rope over all 128 dims at
 * position pos0 + row.  gpu_angle: the angle is float32(pos) * float32(freq),
 * as the kernel forms it; otherwise the double product. */
static void ref_norm_rope_base(const float *x, const double *w, double *out, uint32_t T, uint32_t n_head,
                               uint32_t pos0, bool full, bool gpu_angle, double base) {
    double freq[MELLUM_HEAD_DIM / 2u], mscale;
    ref_freqs(full, base, freq, &mscale);
    const uint32_t D = MELLUM_HEAD_DIM, half = D / 2u;
    double y[MELLUM_HEAD_DIM];
    for (uint32_t r = 0; r < T; r++) {
        for (uint32_t h = 0; h < n_head; h++) {
            const float *src = x + ((uint64_t)r * n_head + h) * D;
            double *dst = out + ((uint64_t)r * n_head + h) * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
            const double s = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) y[i] = src[i] * s * w[i];
            for (uint32_t i = 0; i < half; i++) {
                const double pos = (double)(pos0 + r);
                const double th = gpu_angle ? (double)((float)pos * (float)freq[i]) : pos * freq[i];
                const double c = cos(th) * mscale, sn = sin(th) * mscale;
                dst[i] = y[i] * c - y[i + half] * sn;
                dst[i + half] = y[i] * sn + y[i + half] * c;
            }
        }
    }
}

static void ref_norm_rope(const float *x, const double *w, double *out, uint32_t T, uint32_t n_head,
                          uint32_t pos0, bool full, bool gpu_angle) {
    ref_norm_rope_base(x, w, out, T, n_head, pos0, full, gpu_angle, 500000.0);
}

/* max|got - ref| / max|ref|, the scale-relative error the other kernel tests use */
static double max_rel(const float *got, const double *ref, uint64_t n) {
    double worst = 0.0, scale = 1e-30;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) return INFINITY;
        const double d = fabs((double)got[i] - ref[i]);
        if (d > worst) worst = d;
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    return worst / scale;
}

static double l2(const float *v, uint64_t n) {
    double s = 0.0;
    for (uint64_t i = 0; i < n; i++) s += (double)v[i] * v[i];
    return sqrt(s);
}

static double l2d(const double *v, uint64_t n) {
    double s = 0.0;
    for (uint64_t i = 0; i < n; i++) s += v[i] * v[i];
    return sqrt(s);
}

typedef struct {
    arena_t arena;
    double *wq, *wk;
    uint64_t q_off, k_off;
} fixture_t;

static int kind_of(int full) {
    return full ? DS4_MELLUM_LAYER_FULL : DS4_MELLUM_LAYER_SLIDING;
}

/* Run the host on fresh q and k of T rows at pos0; returns both outputs. */
static int run_host(fixture_t *f, const float *xq, const float *xk, uint32_t T, uint32_t pos0, int full,
                    const ds4_gpu_mellum_rope *p, float **q_out, float **k_out) {
    const uint64_t nq = (uint64_t)T * MELLUM_N_HEAD * MELLUM_HEAD_DIM;
    const uint64_t nk = (uint64_t)T * MELLUM_N_HEAD_KV * MELLUM_HEAD_DIM;
    ds4_gpu_tensor *tq = upload(xq, nq), *tk = upload(xk, nk);
    const int ok = ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, f->arena.base, f->arena.size, f->q_off, f->k_off, T,
                                                      pos0, kind_of(full), p);
    if (ok) {
        *q_out = download(tq, nq);
        *k_out = download(tk, nk);
    }
    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tk);
    return ok;
}

/* One call (T rows from pos0), one layer kind, q and k, against two references.
 *
 * gpu-angle: the angle is float32(pos) * float32(freq), as the kernel forms it.
 * It pins the kernel's arithmetic: every other step (norm, cos/sin, mscale,
 * the rotation) is f32 against double, so the error is a few f32 ulps and 1e-6
 * is generous.
 *
 * double: the angle in double.  An f32 angle cannot hold what a double
 * does: the frequency rounds once and the product rounds once (2^-24 each), so
 * the angle is off by at most pos * freq * 1.2e-7, and freq <= 1.  A pair (a, b)
 * rotated by the wrong angle is off by at most |d angle| * sqrt(a^2 + b^2) per
 * element (mscale scales the reference and the kernel alike, so it cancels), and
 * the largest element of the reference is at least sqrt(a^2 + b^2) / sqrt(2) for
 * the largest pair, so the scale-relative error is at most sqrt(2) * 1.2e-7 *
 * pos = 1.7e-7 * pos.  The tolerance is that bound plus 1e-5 for the rest: 1e-5 at
 * pos 0 and 1, 1.4e-3 at 8,191 (the test prints the measured errors); a flat
 * 1e-5 there cannot be met by any f32 angle. */
static void test_position_scaled(fixture_t *f, uint32_t T, uint32_t pos0, int full, float in_scale) {
    const uint64_t nq = (uint64_t)T * MELLUM_N_HEAD * MELLUM_HEAD_DIM;
    const uint64_t nk = (uint64_t)T * MELLUM_N_HEAD_KV * MELLUM_HEAD_DIM;
    float *xq = rand_vec(nq, in_scale), *xk = rand_vec(nk, in_scale);
    double *rq = malloc(nq * sizeof(double)), *rk = malloc(nk * sizeof(double));
    double *gq_ref = malloc(nq * sizeof(double)), *gk_ref = malloc(nk * sizeof(double));
    ref_norm_rope(xq, f->wq, rq, T, MELLUM_N_HEAD, pos0, full, false);
    ref_norm_rope(xk, f->wk, rk, T, MELLUM_N_HEAD_KV, pos0, full, false);
    ref_norm_rope(xq, f->wq, gq_ref, T, MELLUM_N_HEAD, pos0, full, true);
    ref_norm_rope(xk, f->wk, gk_ref, T, MELLUM_N_HEAD_KV, pos0, full, true);
    const ds4_gpu_mellum_rope p = model_rope(1.0f);
    float *gq = NULL, *gk = NULL;
    require_ok(run_host(f, xq, xk, T, pos0, full, &p, &gq, &gk), "mellum qk norm rope");
    const double dq = max_rel(gq, rq, nq), dk = max_rel(gk, rk, nk);
    const double aq = max_rel(gq, gq_ref, nq), ak = max_rel(gk, gk_ref, nk);
    const double dbl_tol = 1e-5 + 1.7e-7 * (double)(pos0 + T - 1u);
    CHECK(aq <= 1e-6 && ak <= 1e-6, "%s T=%u pos0=%u vs the f32-angle reference: q %.3e k %.3e (tol 1e-6)",
          full ? "full" : "sliding", T, pos0, aq, ak);
    CHECK(dq <= dbl_tol && dk <= dbl_tol, "%s T=%u pos0=%u vs the double-angle reference: q %.3e k %.3e (tol %.2e)",
          full ? "full" : "sliding", T, pos0, dq, dk, dbl_tol);
    printf("  %-7s T=%-2u pos0=%-5u in %.0e  vs f32-angle ref: q %.2e k %.2e (tol 1e-6)   vs double ref: q %.2e k %.2e (tol %.1e)\n",
           full ? "full" : "sliding", T, pos0, (double)in_scale, aq, ak, dq, dk, dbl_tol);
    free(xq); free(xk); free(rq); free(rk); free(gq_ref); free(gk_ref); free(gq); free(gk);
}

static void test_position(fixture_t *f, uint32_t T, uint32_t pos0, int full) {
    test_position_scaled(f, T, pos0, full, 2.0f);
}

/* The double-scaled control: the model's 1.2772589 passed as attn_factor on top
 * of the rope's own mscale scales every output by exactly
 * 1.2772589, so it differs from the reference, by that factor in norm.
 * On a full layer the host refuses it, so the bug cannot be built from a caller
 * that maps the shape's rope_yarn_attn_factor onto attn_factor; a sliding layer
 * (no YaRN, attn_factor is only a cos/sin scale) runs it, and that run is the
 * control that shows this test can see the factor, in q and in k. */
static void test_double_scale_control(fixture_t *f, int full, uint32_t pos0) {
    const uint32_t T = 3;
    const uint64_t nq = (uint64_t)T * MELLUM_N_HEAD * MELLUM_HEAD_DIM;
    const uint64_t nk = (uint64_t)T * MELLUM_N_HEAD_KV * MELLUM_HEAD_DIM;
    float *xq = rand_vec(nq, 2.0f), *xk = rand_vec(nk, 2.0f);
    double *rq = malloc(nq * sizeof(double)), *rk = malloc(nk * sizeof(double));
    ref_norm_rope(xq, f->wq, rq, T, MELLUM_N_HEAD, pos0, full, true);
    ref_norm_rope(xk, f->wk, rk, T, MELLUM_N_HEAD_KV, pos0, full, true);
    const ds4_gpu_mellum_rope ok_p = model_rope(1.0f), ctl_p = model_rope(MELLUM_MSCALE);
    float *gq = NULL, *gk = NULL, *cq = NULL, *ck = NULL;
    require_ok(run_host(f, xq, xk, T, pos0, full, &ok_p, &gq, &gk), "neutral run");
    const double e_ok = fmax(max_rel(gq, rq, nq), max_rel(gk, rk, nk));
    CHECK(e_ok <= 1e-6, "neutral attn_factor must match the reference: %.3e", e_ok);
    const int ctl_ok = run_host(f, xq, xk, T, pos0, full, &ctl_p, &cq, &ck);
    if (full) {
        CHECK(!ctl_ok, "a full layer must refuse attn_factor 1.2772589 (it would apply the mscale twice)");
        printf("  control  full    pos0=%-5u neutral err %.2e; attn_factor %.7f %s\n", pos0, e_ok,
               (double)MELLUM_MSCALE, ctl_ok ? "was NOT refused" : "refused");
    } else {
        require_ok(ctl_ok, "sliding control run");
        const double e_ctl = fmin(max_rel(cq, rq, nq), max_rel(ck, rk, nk));
        const double ratio_q = l2(cq, nq) / l2d(rq, nq), ratio_k = l2(ck, nk) / l2d(rk, nk);
        CHECK(e_ctl > 1e-2, "control must differ from the reference in q and in k, got %.3e", e_ctl);
        CHECK(fabs(ratio_q - (double)MELLUM_MSCALE) <= 1e-5 && fabs(ratio_k - (double)MELLUM_MSCALE) <= 1e-5,
              "control norm ratio q %.7f k %.7f, want %.7f", ratio_q, ratio_k, (double)MELLUM_MSCALE);
        printf("  control  sliding pos0=%-5u neutral err %.2e; double-scaled err %.2e (smaller of q, k), norm ratio q %.7f k %.7f (mscale %.7f)\n",
               pos0, e_ok, e_ctl, ratio_q, ratio_k, (double)MELLUM_MSCALE);
    }
    free(xq); free(xk); free(rq); free(rk); free(gq); free(gk); free(cq); free(ck);
}

/* The sliding table is not the YaRN table: the same input at one position
 * differs by far more than the tolerance between the two layer kinds. */
static void test_kinds_differ(fixture_t *f, uint32_t pos0) {
    const uint32_t T = 1;
    const uint64_t nq = (uint64_t)MELLUM_N_HEAD * MELLUM_HEAD_DIM, nk = (uint64_t)MELLUM_N_HEAD_KV * MELLUM_HEAD_DIM;
    float *xq = rand_vec(nq, 2.0f), *xk = rand_vec(nk, 2.0f);
    const ds4_gpu_mellum_rope p = model_rope(1.0f);
    float *sq = NULL, *sk = NULL, *fq = NULL, *fk = NULL;
    require_ok(run_host(f, xq, xk, T, pos0, 0, &p, &sq, &sk), "sliding");
    require_ok(run_host(f, xq, xk, T, pos0, 1, &p, &fq, &fk), "full");
    double *as_d = malloc(nq * sizeof(double));
    for (uint64_t i = 0; i < nq; i++) as_d[i] = sq[i];
    const double d = max_rel(fq, as_d, nq);
    CHECK(d > 1e-2, "full and sliding layers must not share a rope table (rel diff %.3e at pos %u)", d, pos0);
    printf("  kinds    sliding vs full at pos0=%u differ by %.2e (scale-relative)\n", pos0, d);
    free(xq); free(xk); free(sq); free(sk); free(fq); free(fk); free(as_d);
}

/* The layer kind picks its own base.  Mellum's two bases are both 5e5, so a
 * call that confused them would pass every test above; with 1e4 and 5e5 it
 * cannot.  The reference with the bases swapped must not match (the sibling). */
static void test_bases(fixture_t *f) {
    const uint32_t T = 2, pos0 = 1024;
    const uint64_t nq = (uint64_t)T * MELLUM_N_HEAD * MELLUM_HEAD_DIM;
    const uint64_t nk = (uint64_t)T * MELLUM_N_HEAD_KV * MELLUM_HEAD_DIM;
    float *xq = rand_vec(nq, 2.0f), *xk = rand_vec(nk, 2.0f);
    ds4_gpu_mellum_rope p = model_rope(1.0f);
    p.freq_base_sliding = 10000.0f;
    p.freq_base_full = 500000.0f;
    double *rq = malloc(nq * sizeof(double));
    for (int full = 0; full <= 1; full++) {
        float *gq = NULL, *gk = NULL;
        require_ok(run_host(f, xq, xk, T, pos0, full, &p, &gq, &gk), "bases run");
        ref_norm_rope_base(xq, f->wq, rq, T, MELLUM_N_HEAD, pos0, full, true, full ? 500000.0 : 10000.0);
        const double own = max_rel(gq, rq, nq);
        ref_norm_rope_base(xq, f->wq, rq, T, MELLUM_N_HEAD, pos0, full, true, full ? 10000.0 : 500000.0);
        const double other = max_rel(gq, rq, nq);
        CHECK(own <= 1e-6, "%s layer with its own base: %.3e", full ? "full" : "sliding", own);
        CHECK(other > 1e-2, "%s layer must not match the other kind's base (%.3e)", full ? "full" : "sliding", other);
        printf("  bases    %-7s own base err %.2e, the other kind's base err %.2e\n", full ? "full" : "sliding", own, other);
        free(gq); free(gk);
    }
    free(xq); free(xk); free(rq);
}

/* In place and out of place agree is not a question here: the host is in
 * place by contract.  Refusals, each with a sibling success. */
static void test_refusals(fixture_t *f) {
    const uint32_t T = 2;
    const uint64_t nq = (uint64_t)T * MELLUM_N_HEAD * MELLUM_HEAD_DIM;
    const uint64_t nk = (uint64_t)T * MELLUM_N_HEAD_KV * MELLUM_HEAD_DIM;
    float *xq = rand_vec(nq, 1.0f), *xk = rand_vec(nk, 1.0f);
    ds4_gpu_tensor *tq = upload(xq, nq), *tk = upload(xk, nk);
    ds4_gpu_tensor *tq_small = upload(xq, nq - 1u);
    const ds4_gpu_mellum_rope good = model_rope(1.0f);
    const uint8_t *map = f->arena.base;
    const uint64_t size = f->arena.size;

    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_SLIDING, &good) != 0,
          "sliding call with valid arguments succeeds");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_FULL, &good) != 0,
          "full call with valid arguments succeeds");

    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, 2, &good) == 0,
          "layer kind 2 is refused");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, -1, &good) == 0,
          "layer kind -1 is refused");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, 0, 5, DS4_MELLUM_LAYER_SLIDING, &good) == 0,
          "zero rows are refused");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_SLIDING, NULL) == 0,
          "a missing parameter block is refused");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq_small, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_SLIDING, &good) == 0,
          "an undersized q is refused");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, size, f->k_off, T, 5, DS4_MELLUM_LAYER_SLIDING, &good) == 0,
          "a q norm weight past the map is refused");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 0xffffffffu - 1u, DS4_MELLUM_LAYER_SLIDING, &good) == 0,
          "a position past 2^24 is refused (float32(pos) stops being exact)");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, (1u << 24) - T, DS4_MELLUM_LAYER_SLIDING, &good) != 0,
          "the last position below 2^24 is accepted");
    /* a full layer is handed the GGUF's yarn_attn_factor: refused; handed the neutral 1.0: accepted */
    ds4_gpu_mellum_rope twice = good;
    twice.attn_factor = MELLUM_MSCALE;
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_FULL, &twice) == 0,
          "a full layer with attn_factor 1.2772589 is refused");
    twice.attn_factor = 1.0000001f;
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_FULL, &twice) == 0,
          "a full layer with attn_factor != 1.0f at all is refused");
    ds4_gpu_mellum_rope bad = good;
    bad.head_dim = 96;
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_SLIDING, &bad) == 0,
          "head_dim 96 (the host is Mellum's 128) is refused");
    bad = good;
    bad.freq_scale = 0.0f;
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_FULL, &bad) == 0,
          "a zero freq_scale on a full layer is refused");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_SLIDING, &bad) != 0,
          "the same zero freq_scale is unused on a sliding layer, which has no YaRN");

    /* A refused call leaves q as it was: k's size is checked before q is rewritten. */
    ds4_gpu_tensor *tq2 = upload(xq, nq), *tk_small = upload(xk, nk - 1u);
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq2, tk_small, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_SLIDING, &good) == 0,
          "an undersized k is refused");
    float *q_after = download(tq2, nq);
    CHECK(memcmp(q_after, xq, nq * sizeof(float)) == 0, "a call refused for k's size left q rewritten");
    CHECK(ds4_gpu_mellum_qk_norm_rope_tensor(tq2, tk, map, size, f->q_off, f->k_off, T, 5, DS4_MELLUM_LAYER_SLIDING, &good) != 0,
          "the same q with a full-size k succeeds");
    free(q_after);
    ds4_gpu_tensor_free(tq2); ds4_gpu_tensor_free(tk_small);

    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tk); ds4_gpu_tensor_free(tq_small);
    free(xq); free(xk);
}


/* ======================================================================
 * Attention.  GQA 32/4, head 128.  Sliding layers: a 1,024-slot f16 ring plus
 * the chunk's f16 staging copy (kernel_mellum_attn_swa), the chunk committed
 * to the ring afterwards (kernel_mellum_ring_commit).  The reference is causal
 * GQA attention in double (a window keeps keys with q - k < window), over the
 * keys as the kernel reads them, rounded to f16.
 * ====================================================================== */

#define KV_W (MELLUM_N_HEAD_KV * MELLUM_HEAD_DIM)  /* 512 */
#define Q_W (MELLUM_N_HEAD * MELLUM_HEAD_DIM)      /* 4096 */
#define RING 1024u
/* f32 accumulation over at most 1,025 keys against a double reference, on keys both read as f16 */
#define ATTN_TOL 1e-5

static float attn_scale(void) {
    return 1.0f / sqrtf((float)MELLUM_HEAD_DIM);
}

static float round_f16(float x) {
    return (float)(__fp16)x;
}

typedef struct {
    uint32_t n;
    float *q;      /* [n][Q_W] */
    float *k, *v;  /* [n][KV_W], f32 as the projections leave them */
    float *kr, *vr; /* the same rounded to f16 */
} seq_t;

static seq_t make_seq(uint32_t n) {
    seq_t s;
    s.n = n;
    /* scores of standard deviation about 2: the softmax is not flat, so a dropped or reordered key shows */
    s.q = rand_vec((uint64_t)n * Q_W, 3.0f);
    s.k = rand_vec((uint64_t)n * KV_W, 2.0f);
    s.v = rand_vec((uint64_t)n * KV_W, 1.5f);
    s.kr = malloc((uint64_t)n * KV_W * sizeof(float));
    s.vr = malloc((uint64_t)n * KV_W * sizeof(float));
    require_ok(s.kr != NULL && s.vr != NULL, "malloc");
    for (uint64_t i = 0; i < (uint64_t)n * KV_W; i++) {
        s.kr[i] = round_f16(s.k[i]);
        s.vr[i] = round_f16(s.v[i]);
    }
    return s;
}

static void free_seq(seq_t *s) {
    free(s->q); free(s->k); free(s->v); free(s->kr); free(s->vr);
}

/* One query row of attention() in double.  window 0: no window. */
static void ref_attn_row(const float *q, const float *K, const float *V, uint32_t qpos, uint32_t window, double *out) {
    const uint32_t first = (window && qpos + 1u > window) ? qpos + 1u - window : 0u;
    const uint32_t n = qpos + 1u - first;
    const uint32_t rep = MELLUM_N_HEAD / MELLUM_N_HEAD_KV;
    double *p = malloc((uint64_t)n * sizeof(double));
    require_ok(p != NULL, "malloc");
    for (uint32_t h = 0; h < MELLUM_N_HEAD; h++) {
        const uint32_t kvh = h / rep;
        const float *qh = q + (uint64_t)h * MELLUM_HEAD_DIM;
        double mx = -INFINITY;
        for (uint32_t i = 0; i < n; i++) {
            const float *kr = K + (uint64_t)(first + i) * KV_W + (uint64_t)kvh * MELLUM_HEAD_DIM;
            double dot = 0.0;
            for (uint32_t d = 0; d < MELLUM_HEAD_DIM; d++) dot += (double)qh[d] * kr[d];
            p[i] = dot * (double)attn_scale();
            if (p[i] > mx) mx = p[i];
        }
        double sum = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            p[i] = exp(p[i] - mx);
            sum += p[i];
        }
        double *oh = out + (uint64_t)h * MELLUM_HEAD_DIM;
        for (uint32_t d = 0; d < MELLUM_HEAD_DIM; d++) oh[d] = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            const float *vr = V + (uint64_t)(first + i) * KV_W + (uint64_t)kvh * MELLUM_HEAD_DIM;
            for (uint32_t d = 0; d < MELLUM_HEAD_DIM; d++) oh[d] += p[i] * vr[d];
        }
        for (uint32_t d = 0; d < MELLUM_HEAD_DIM; d++) oh[d] /= sum;
    }
    free(p);
}

/* Worst scale-relative error of rows [r0, r1) of out against the reference (window 0: full attention);
 * rounded: the reference reads the keys as f16, as the kernel does. */
static double rows_vs_ref(const seq_t *s, const float *out, uint32_t r0, uint32_t r1, uint32_t window, bool rounded) {
    double worst = 0.0;
    double *ref = malloc((uint64_t)Q_W * sizeof(double));
    require_ok(ref != NULL, "malloc");
    for (uint32_t r = r0; r < r1; r++) {
        ref_attn_row(s->q + (uint64_t)r * Q_W, rounded ? s->kr : s->k, rounded ? s->vr : s->v, r, window, ref);
        const double e = max_rel(out + (uint64_t)r * Q_W, ref, Q_W);
        if (e > worst) worst = e;
    }
    free(ref);
    return worst;
}

/* the number of float words that differ in bits, and the index of the first */
static uint64_t bit_diffs(const float *a, const float *b, uint64_t n, uint64_t *first) {
    uint64_t d = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (d == 0 && first) *first = i;
            d++;
        }
    }
    return d;
}

static ds4_gpu_tensor *zeroed(uint64_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    require_ok(t != NULL, "tensor alloc");
    uint8_t *z = calloc(1, bytes);
    require_ok(z != NULL, "calloc");
    require_ok(ds4_gpu_tensor_write(t, 0, z, bytes), "tensor zero");
    free(z);
    return t;
}

/* Round the chunk's f32 rows to its f16 staging copy: the one place a sliding layer's keys are rounded. */
static void stage_f16(ds4_gpu_tensor *dst16, const ds4_gpu_tensor *src32, uint64_t count) {
    require_ok(ds4_gpu_tensor_copy_f32_to_f16(dst16, 0, src32, 0, count), "stage f32 to f16");
}

/* A chunking: the first chunk has `lead` rows (0: `size`), the rest `size`.  {0,1} is decode. */
typedef struct {
    uint32_t lead, size;
} plan_t;

/* The forward pass encodes a layer's ops into one command buffer that runs at end_commands, so a table the CPU
 * rewrites between two ops is seen by the GPU only in its last state.  Outside a batch every op runs to completion
 * before the call returns.  The sessions run both ways; this selects which. */
static bool g_batched;

/* Run a whole sequence through the sliding kernel as a session would: stage, attend, then commit to the ring.
 * commit_first runs the wrong order (the control for the read-then-commit contract).  out is [n][Q_W]. */
static void swa_session(const seq_t *s, plan_t plan, bool commit_first, float *out) {
    const uint32_t max_rows = plan.size > plan.lead ? plan.size : plan.lead;
    ds4_gpu_tensor *ring_k = zeroed((uint64_t)RING * KV_W * 2u), *ring_v = zeroed((uint64_t)RING * KV_W * 2u);
    ds4_gpu_tensor *tags = ds4_gpu_tensor_alloc(RING * sizeof(int32_t));
    int32_t none[RING];
    for (uint32_t i = 0; i < RING; i++) none[i] = -1;
    require_ok(tags && ds4_gpu_tensor_write(tags, 0, none, sizeof(none)), "tags");
    ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc((uint64_t)max_rows * KV_W * 4u), *vt = ds4_gpu_tensor_alloc((uint64_t)max_rows * KV_W * 4u);
    ds4_gpu_tensor *k16 = ds4_gpu_tensor_alloc((uint64_t)max_rows * KV_W * 2u), *v16 = ds4_gpu_tensor_alloc((uint64_t)max_rows * KV_W * 2u);
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc((uint64_t)max_rows * Q_W * 4u), *ot = ds4_gpu_tensor_alloc((uint64_t)max_rows * Q_W * 4u);
    require_ok(kt && vt && k16 && v16 && qt && ot, "session tensors");
    for (uint32_t a = 0; a < s->n;) {
        const uint32_t rows = (a == 0 && plan.lead) ? plan.lead : plan.size;
        const uint32_t n = (s->n - a < rows) ? s->n - a : rows;
        require_ok(ds4_gpu_tensor_write(kt, 0, s->k + (uint64_t)a * KV_W, (uint64_t)n * KV_W * 4u), "k upload");
        require_ok(ds4_gpu_tensor_write(vt, 0, s->v + (uint64_t)a * KV_W, (uint64_t)n * KV_W * 4u), "v upload");
        require_ok(ds4_gpu_tensor_write(qt, 0, s->q + (uint64_t)a * Q_W, (uint64_t)n * Q_W * 4u), "q upload");
        if (g_batched) require_ok(ds4_gpu_begin_commands(), "begin commands");
        stage_f16(k16, kt, (uint64_t)n * KV_W);
        stage_f16(v16, vt, (uint64_t)n * KV_W);
        if (commit_first) {
            require_ok(ds4_gpu_mellum_ring_commit_tensor(ring_k, ring_v, tags, kt, vt, n, KV_W, a, RING), "ring commit");
        }
        require_ok(ds4_gpu_mellum_attn_swa_tensor(ot, qt, ring_k, ring_v, tags, k16, v16, n, a, RING, MELLUM_N_HEAD,
                                                  MELLUM_N_HEAD_KV, MELLUM_HEAD_DIM, attn_scale()), "sliding attention");
        if (!commit_first) {
            require_ok(ds4_gpu_mellum_ring_commit_tensor(ring_k, ring_v, tags, kt, vt, n, KV_W, a, RING), "ring commit");
        }
        if (g_batched) require_ok(ds4_gpu_end_commands(), "end commands");
        require_ok(ds4_gpu_tensor_read(ot, 0, out + (uint64_t)a * Q_W, (uint64_t)n * Q_W * 4u), "out download");
        a += n;
    }
    ds4_gpu_tensor_free(ring_k); ds4_gpu_tensor_free(ring_v); ds4_gpu_tensor_free(tags);
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(vt); ds4_gpu_tensor_free(k16); ds4_gpu_tensor_free(v16);
    ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(ot);
}

static bool all_finite(const float *v, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) if (!isfinite(v[i])) return false;
    return true;
}

/* Sliding attention at n keys: decode (one row at a time) and prefill (a chunk of 64 rows straddling 1,024,
 * and a 1,024-row chunk), each against the reference with window 1,024; the identity of every row of each
 * prefill with the decode steps; the off-by-one control at 1,025 keys. */
static void test_swa_session(uint32_t n) {
    static const plan_t decode = { 0, 1 }, pre_b = { 32, 64 }, pre_c = { 33, 1024 };
    seq_t s = make_seq(n);
    float *dec = malloc((uint64_t)n * Q_W * 4u), *pb = malloc((uint64_t)n * Q_W * 4u), *pc = malloc((uint64_t)n * Q_W * 4u);
    require_ok(dec && pb && pc, "malloc");
    swa_session(&s, decode, false, dec);
    swa_session(&s, pre_b, false, pb);
    swa_session(&s, pre_c, false, pc);
    uint64_t fb = 0, fc = 0;
    const uint64_t db = bit_diffs(dec, pb, (uint64_t)n * Q_W, &fb), dc = bit_diffs(dec, pc, (uint64_t)n * Q_W, &fc);
    CHECK(all_finite(dec, (uint64_t)n * Q_W), "sliding decode at %u keys produced a non-finite value", n);
    CHECK(db == 0, "sliding prefill (32 then 64-row chunks) at %u keys differs from decode in %llu words, first at row %llu",
          n, (unsigned long long)db, (unsigned long long)(fb / Q_W));
    CHECK(dc == 0, "sliding prefill (33 then 1,024-row chunks) at %u keys differs from decode in %llu words, first at row %llu",
          n, (unsigned long long)dc, (unsigned long long)(fc / Q_W));
    const uint32_t tail = n < 8u ? n : 8u;
    const double e_dec = rows_vs_ref(&s, dec, n - tail, n, RING, true);
    const double e_pre = rows_vs_ref(&s, pb, n - tail, n, RING, true);
    CHECK(e_dec <= ATTN_TOL && e_pre <= ATTN_TOL, "sliding at %u keys vs the window-1024 reference: decode %.3e prefill %.3e (tol %.0e)",
          n, e_dec, e_pre, ATTN_TOL);
    /* the keys are f16: against the reference that reads them as f32 the kernel must be visibly off */
    const double e_f32 = rows_vs_ref(&s, dec, n - tail, n, RING, false);
    CHECK(e_f32 > 3.0 * ATTN_TOL, "sliding at %u keys must read f16 keys, but is only %.3e from the f32-key reference", n, e_f32);
    printf("  sliding  %4u keys: last %u rows vs reference decode %.2e prefill %.2e (tol %.0e); vs f32-key ref %.2e; identity: 64-row chunks %llu / 1,024-row chunks %llu words differ of %llu\n",
           n, tail, e_dec, e_pre, ATTN_TOL, e_f32, (unsigned long long)db, (unsigned long long)dc, (unsigned long long)n * Q_W);
    if (n == 1057) {
        /* the 64-row chunk [992, 1056) straddles 1,024: every row against the reference, prefill and decode */
        const double s_pre = rows_vs_ref(&s, pb, 992, 1056, RING, true), s_dec = rows_vs_ref(&s, dec, 992, 1056, RING, true);
        CHECK(s_pre <= ATTN_TOL && s_dec <= ATTN_TOL, "the chunk straddling 1,024 vs the reference: prefill %.3e decode %.3e (tol %.0e)",
              s_pre, s_dec, ATTN_TOL);
        printf("  sliding  chunk of 64 rows [992,1056) straddling 1,024: worst row vs reference prefill %.2e decode %.2e (tol %.0e)\n",
               s_pre, s_dec, ATTN_TOL);
        /* the same sessions with each chunk's ops (stage, attend, commit) encoded into one command buffer, as the
         * forward pass runs them: the order the ops were encoded in is the order they run */
        float *bd = malloc((uint64_t)n * Q_W * 4u), *bp = malloc((uint64_t)n * Q_W * 4u);
        require_ok(bd && bp, "malloc");
        g_batched = true;
        swa_session(&s, decode, false, bd);
        swa_session(&s, pre_b, false, bp);
        g_batched = false;
        const uint64_t d_bd = bit_diffs(dec, bd, (uint64_t)n * Q_W, NULL), d_bp = bit_diffs(dec, bp, (uint64_t)n * Q_W, NULL);
        CHECK(d_bd == 0 && d_bp == 0, "batched sliding sessions at %u keys differ from the unbatched decode: decode %llu words, prefill %llu words",
              n, (unsigned long long)d_bd, (unsigned long long)d_bp);
        printf("  sliding  %u keys with each chunk's ops in one command buffer: decode %llu, prefill %llu words differ from the unbatched decode\n",
               n, (unsigned long long)d_bd, (unsigned long long)d_bp);
        free(bd); free(bp);
        /* the read-then-commit contract: the chunk committed before its attention makes row 0 read a slot that now holds
         * position 1,024 + s; the tag check turns every row that still needs a lost position into NaN.
         * Rows 0 .. 62 need one (row r's first position is r - 31 when r > 31, and slot (first) % 1,024 < 32
         * was overwritten by position 1,024 + first); row 63's first position is 32, kept. */
        float *wrong = malloc((uint64_t)n * Q_W * 4u);
        require_ok(wrong != NULL, "malloc");
        swa_session(&s, pre_b, true, wrong);
        uint32_t nan_rows = 0, bad_after = 0;
        for (uint32_t r = 992; r < 1056; r++) {
            const bool fin = all_finite(wrong + (uint64_t)r * Q_W, Q_W);
            if (!fin) nan_rows++;
            if (r - 992u < 63u && fin) bad_after++;
        }
        const bool last_ok = all_finite(wrong + (uint64_t)1055 * Q_W, Q_W) &&
                             bit_diffs(wrong + (uint64_t)1055 * Q_W, pb + (uint64_t)1055 * Q_W, Q_W, NULL) == 0;
        CHECK(nan_rows == 63 && bad_after == 0, "committing the chunk before its attention must lose the ring's history of rows 0..62 of the chunk: %u rows non-finite, %u of rows 0..62 finite",
              nan_rows, bad_after);
        CHECK(last_ok, "row 63 of the chunk needs nothing the early commit overwrote and must still match");
        printf("  control  commit before attention: %u of the 64 rows [992,1056) are NaN (the stale-tag check), row 1055 unchanged\n", nan_rows);
        free(wrong);
    }
    if (n == 1025) {
        /* the off-by-one control: a window of 1,025 keeps position 0 for the query at 1,024 */
        double *r24 = malloc((uint64_t)Q_W * 8u), *r25 = malloc((uint64_t)Q_W * 8u);
        require_ok(r24 && r25, "malloc");
        ref_attn_row(s.q + (uint64_t)1024 * Q_W, s.kr, s.vr, 1024, 1024, r24);
        ref_attn_row(s.q + (uint64_t)1024 * Q_W, s.kr, s.vr, 1024, 1025, r25);
        const double e24 = max_rel(dec + (uint64_t)1024 * Q_W, r24, Q_W), e25 = max_rel(dec + (uint64_t)1024 * Q_W, r25, Q_W);
        double apart = 0.0, sc = 1e-30;
        for (uint32_t i = 0; i < Q_W; i++) {
            if (fabs(r24[i] - r25[i]) > apart) apart = fabs(r24[i] - r25[i]);
            if (fabs(r24[i]) > sc) sc = fabs(r24[i]);
        }
        apart /= sc;
        CHECK(e24 <= ATTN_TOL, "row 1024 vs the window-1024 reference: %.3e", e24);
        CHECK(apart > 100.0 * ATTN_TOL && e25 > 100.0 * ATTN_TOL, "off-by-one control: the references differ by %.3e and the kernel is %.3e from window 1,025 (it must NOT match)", apart, e25);
        printf("  control  off-by-one at 1,025 keys: window-1024 ref %.2e, window-1025 ref %.2e (the two references differ by %.2e)\n", e24, e25, apart);
        free(r24); free(r25);
    }
    free(dec); free(pb); free(pc);
    free_seq(&s);
}

/* The tag check's sibling: a ring with the right tags is finite and right; wrong tags give NaN, not a wrong number. */
static void test_swa_tags(void) {
    const uint32_t n = 1100, qpos = 1099;
    seq_t s = make_seq(n);
    ds4_gpu_tensor *ring_k = zeroed((uint64_t)RING * KV_W * 2u), *ring_v = zeroed((uint64_t)RING * KV_W * 2u);
    ds4_gpu_tensor *tags = ds4_gpu_tensor_alloc(RING * sizeof(int32_t));
    ds4_gpu_tensor *kt = upload(s.k, (uint64_t)n * KV_W), *vt = upload(s.v, (uint64_t)n * KV_W);
    ds4_gpu_tensor *k16 = ds4_gpu_tensor_alloc((uint64_t)n * KV_W * 2u), *v16 = ds4_gpu_tensor_alloc((uint64_t)n * KV_W * 2u);
    ds4_gpu_tensor *qt = upload(s.q + (uint64_t)qpos * Q_W, Q_W), *ot = ds4_gpu_tensor_alloc(Q_W * 4u);
    require_ok(tags && k16 && v16 && ot, "tag tensors");
    stage_f16(k16, kt, (uint64_t)n * KV_W);
    stage_f16(v16, vt, (uint64_t)n * KV_W);
    /* positions 0 .. 1098 in the ring (slot = pos % 1,024, the later position wins), then the row at 1,099 */
    ds4_gpu_tensor *k_lo = ds4_gpu_tensor_view(kt, 0, 1024ull * KV_W * 4u), *v_lo = ds4_gpu_tensor_view(vt, 0, 1024ull * KV_W * 4u);
    ds4_gpu_tensor *k_hi = ds4_gpu_tensor_view(kt, 1024ull * KV_W * 4u, 75ull * KV_W * 4u);
    ds4_gpu_tensor *v_hi = ds4_gpu_tensor_view(vt, 1024ull * KV_W * 4u, 75ull * KV_W * 4u);
    ds4_gpu_tensor *k_row = ds4_gpu_tensor_view(k16, (uint64_t)qpos * KV_W * 2u, KV_W * 2u);
    ds4_gpu_tensor *v_row = ds4_gpu_tensor_view(v16, (uint64_t)qpos * KV_W * 2u, KV_W * 2u);
    require_ok(k_lo && v_lo && k_hi && v_hi && k_row && v_row, "views");
    require_ok(ds4_gpu_mellum_ring_commit_tensor(ring_k, ring_v, tags, k_lo, v_lo, 1024, KV_W, 0, RING), "commit 0..1023");
    require_ok(ds4_gpu_mellum_ring_commit_tensor(ring_k, ring_v, tags, k_hi, v_hi, 75, KV_W, 1024, RING), "commit 1024..1098");
    float *out = malloc(Q_W * 4u);
    double *ref = malloc(Q_W * 8u);
    require_ok(out && ref, "malloc");
    CHECK(ds4_gpu_mellum_attn_swa_tensor(ot, qt, ring_k, ring_v, tags, k_row, v_row, 1, qpos, RING, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                         MELLUM_HEAD_DIM, attn_scale()) != 0, "sliding attention with a live ring runs");
    require_ok(ds4_gpu_tensor_read(ot, 0, out, Q_W * 4u), "read");
    ref_attn_row(s.q + (uint64_t)qpos * Q_W, s.kr, s.vr, qpos, RING, ref);
    const double e_ok = max_rel(out, ref, Q_W);
    CHECK(e_ok <= ATTN_TOL, "row 1099 from a live ring vs the reference: %.3e", e_ok);
    /* The row sees positions 76 .. 1099.  Slot 100 holds position 100, which it needs: a wrong tag there gives NaN. */
    int32_t t[RING];
    require_ok(ds4_gpu_tensor_read(tags, 0, t, sizeof(t)), "read tags");
    CHECK(t[100] == 100 && t[75] == 75 && t[50] == 1074, "the ring holds position 100 in slot 100, 75 in slot 75, 1074 in slot 50");
    t[100] = 101;
    require_ok(ds4_gpu_tensor_write(tags, 0, t, sizeof(t)), "write tags");
    require_ok(ds4_gpu_mellum_attn_swa_tensor(ot, qt, ring_k, ring_v, tags, k_row, v_row, 1, qpos, RING, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                              MELLUM_HEAD_DIM, attn_scale()), "sliding attention with a stale tag");
    require_ok(ds4_gpu_tensor_read(ot, 0, out, Q_W * 4u), "read");
    CHECK(!all_finite(out, Q_W), "a wrong tag on a slot the row needs must not give a finite value");
    /* Slot 75 holds position 75, outside the window: a wrong tag there is not read. */
    t[100] = 100;
    t[75] = 9999;
    require_ok(ds4_gpu_tensor_write(tags, 0, t, sizeof(t)), "write tags");
    require_ok(ds4_gpu_mellum_attn_swa_tensor(ot, qt, ring_k, ring_v, tags, k_row, v_row, 1, qpos, RING, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                              MELLUM_HEAD_DIM, attn_scale()), "sliding attention, tag outside the window");
    require_ok(ds4_gpu_tensor_read(ot, 0, out, Q_W * 4u), "read");
    const double e_out = max_rel(out, ref, Q_W);
    CHECK(e_out <= ATTN_TOL, "a wrong tag on a slot the row does not need must not matter: %.3e from the reference", e_out);
    printf("  tags     live ring %.2e from the reference; a wrong tag on a needed slot gives NaN; one outside the window changes nothing (%.2e)\n",
           e_ok, e_out);
    free(out); free(ref);
    ds4_gpu_tensor_free(k_lo); ds4_gpu_tensor_free(v_lo); ds4_gpu_tensor_free(k_hi); ds4_gpu_tensor_free(v_hi);
    ds4_gpu_tensor_free(k_row); ds4_gpu_tensor_free(v_row);
    ds4_gpu_tensor_free(ring_k); ds4_gpu_tensor_free(ring_v); ds4_gpu_tensor_free(tags);
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(vt); ds4_gpu_tensor_free(k16); ds4_gpu_tensor_free(v16);
    ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(ot);
    free_seq(&s);
}

/* A rejected draft row on a ring of exactly the window: positions 0 .. 1098 are committed, then a draft row at
 * 1,099 is attended and committed (overwriting slot 75, which held position 75); the draft is dropped and the true
 * row at 1,099 is attended again.  It reads positions 76 .. 1,098 from the ring and itself from staging, so
 * nothing it needs was lost, and after its commit the row at 1,100 reads the true row.  Both must equal the
 * reference over the true sequence, bit for bit equal to a session that never saw the draft. */
static void test_swa_rewind_one(void) {
    const uint32_t n = 1101, draft_pos = 1099;
    seq_t s = make_seq(n);
    seq_t d = make_seq(1);  /* the draft's q, k, v */
    float *straight = malloc((uint64_t)n * Q_W * 4u);
    require_ok(straight != NULL, "malloc");
    swa_session(&s, (plan_t){ 0, 1 }, false, straight);

    ds4_gpu_tensor *ring_k = zeroed((uint64_t)RING * KV_W * 2u), *ring_v = zeroed((uint64_t)RING * KV_W * 2u);
    ds4_gpu_tensor *tags = ds4_gpu_tensor_alloc(RING * sizeof(int32_t));
    int32_t none[RING];
    for (uint32_t i = 0; i < RING; i++) none[i] = -1;
    require_ok(tags && ds4_gpu_tensor_write(tags, 0, none, sizeof(none)), "tags");
    ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc(KV_W * 4u), *vt = ds4_gpu_tensor_alloc(KV_W * 4u);
    ds4_gpu_tensor *k16 = ds4_gpu_tensor_alloc(KV_W * 2u), *v16 = ds4_gpu_tensor_alloc(KV_W * 2u);
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(Q_W * 4u), *ot = ds4_gpu_tensor_alloc(Q_W * 4u);
    require_ok(kt && vt && k16 && v16 && qt && ot, "rewind tensors");
    float *row = malloc(Q_W * 4u);
    require_ok(row != NULL, "malloc");
    uint64_t bad = 0;
    /* one decode step at pos from (q, k, v); commit after attending */
    #define STEP(q_, k_, v_, pos_) do { \
        require_ok(ds4_gpu_tensor_write(kt, 0, k_, KV_W * 4u) && ds4_gpu_tensor_write(vt, 0, v_, KV_W * 4u) && \
                   ds4_gpu_tensor_write(qt, 0, q_, Q_W * 4u), "upload"); \
        stage_f16(k16, kt, KV_W); \
        stage_f16(v16, vt, KV_W); \
        require_ok(ds4_gpu_mellum_attn_swa_tensor(ot, qt, ring_k, ring_v, tags, k16, v16, 1, pos_, RING, MELLUM_N_HEAD, \
                                                  MELLUM_N_HEAD_KV, MELLUM_HEAD_DIM, attn_scale()), "attend"); \
        require_ok(ds4_gpu_mellum_ring_commit_tensor(ring_k, ring_v, tags, kt, vt, 1, KV_W, pos_, RING), "commit"); \
        require_ok(ds4_gpu_tensor_read(ot, 0, row, Q_W * 4u), "read"); \
    } while (0)
    for (uint32_t p = 0; p < draft_pos; p++) {
        STEP(s.q + (uint64_t)p * Q_W, s.k + (uint64_t)p * KV_W, s.v + (uint64_t)p * KV_W, p);
    }
    STEP(d.q, d.k, d.v, draft_pos);                         /* the draft, then dropped */
    for (uint32_t p = draft_pos; p < n; p++) {
        STEP(s.q + (uint64_t)p * Q_W, s.k + (uint64_t)p * KV_W, s.v + (uint64_t)p * KV_W, p);
        bad += bit_diffs(row, straight + (uint64_t)p * Q_W, Q_W, NULL);
        CHECK(all_finite(row, Q_W), "row %u after the dropped draft is not finite", p);
    }
    #undef STEP
    const double e = rows_vs_ref(&s, straight, draft_pos, n, RING, true);
    CHECK(bad == 0, "after a dropped draft at %u, rows %u and %u differ from the straight session in %llu words",
          draft_pos, draft_pos, draft_pos + 1u, (unsigned long long)bad);
    CHECK(e <= ATTN_TOL, "the straight session's rows %u .. %u vs the reference: %.3e", draft_pos, n - 1u, e);
    printf("  rewind   a draft at %u committed then dropped on a %u-slot ring: rows %u and %u equal the straight session "
           "(%llu words differ), %.2e from the reference\n", draft_pos, RING, draft_pos, draft_pos + 1u,
           (unsigned long long)bad, e);
    free(row); free(straight);
    ds4_gpu_tensor_free(ring_k); ds4_gpu_tensor_free(ring_v); ds4_gpu_tensor_free(tags);
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(vt); ds4_gpu_tensor_free(k16); ds4_gpu_tensor_free(v16);
    ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(ot);
    free_seq(&d);
    free_seq(&s);
}

static void test_swa_refusals(void) {
    ds4_gpu_tensor *ring = zeroed((uint64_t)RING * KV_W * 2u), *tags = zeroed(RING * 4u);
    ds4_gpu_tensor *k16 = zeroed(2ull * KV_W * 2u), *v16 = zeroed(2ull * KV_W * 2u);
    ds4_gpu_tensor *q = zeroed(2ull * Q_W * 4u), *o = zeroed(2ull * Q_W * 4u);
    ds4_gpu_tensor *small = zeroed(64);
    const float sc = attn_scale();
#define SWA(out_, q_, rk_, rv_, tg_, k_, v_, n_, p_, w_, nh_, nkv_, hd_) \
    ds4_gpu_mellum_attn_swa_tensor(out_, q_, rk_, rv_, tg_, k_, v_, n_, p_, w_, nh_, nkv_, hd_, sc)
    CHECK(SWA(o, q, ring, ring, tags, k16, v16, 2, 5, RING, 32, 4, 128) != 0, "a valid sliding call succeeds");
    CHECK(SWA(o, q, ring, ring, tags, k16, v16, 0, 5, RING, 32, 4, 128) == 0, "zero rows are refused");
    CHECK(SWA(o, q, ring, ring, tags, k16, v16, 2, 5, 0, 32, 4, 128) == 0, "a zero window is refused");
    CHECK(SWA(o, q, ring, ring, tags, k16, v16, 2, 5, RING, 32, 4, 96) == 0, "head_dim 96 is refused (Mellum's is 128)");
    CHECK(SWA(o, q, ring, ring, tags, k16, v16, 2, 5, RING, 30, 4, 128) == 0, "30 heads over 4 KV heads is refused");
    CHECK(SWA(o, q, ring, ring, tags, k16, v16, 2, 5, RING, 16, 2, 128) == 0, "16 query and 2 KV heads is refused (Mellum's is 32 and 4)");
    CHECK(SWA(o, small, ring, ring, tags, k16, v16, 2, 5, RING, 32, 4, 128) == 0, "an undersized q is refused");
    CHECK(SWA(small, q, ring, ring, tags, k16, v16, 2, 5, RING, 32, 4, 128) == 0, "an undersized out is refused");
    CHECK(SWA(o, q, small, ring, tags, k16, v16, 2, 5, RING, 32, 4, 128) == 0, "an undersized ring k is refused");
    CHECK(SWA(o, q, ring, small, tags, k16, v16, 2, 5, RING, 32, 4, 128) == 0, "an undersized ring v is refused");
    CHECK(SWA(o, q, ring, ring, small, k16, v16, 2, 5, RING, 32, 4, 128) == 0, "an undersized tag array is refused");
    CHECK(SWA(o, q, ring, ring, tags, small, v16, 2, 5, RING, 32, 4, 128) == 0, "an undersized staged k is refused");
    CHECK(SWA(o, q, ring, ring, tags, k16, small, 2, 5, RING, 32, 4, 128) == 0, "an undersized staged v is refused");
    CHECK(SWA(o, q, ring, ring, tags, k16, v16, 2, 0x7fffffffu, RING, 32, 4, 128) == 0, "a position past INT32_MAX is refused (tags are int32)");
    CHECK(SWA(o, q, ring, ring, tags, k16, v16, 2, 0x7ffffff0u, RING, 32, 4, 128) != 0, "a position below INT32_MAX with room for the rows is accepted");
#undef SWA
    ds4_gpu_tensor_free(ring); ds4_gpu_tensor_free(tags); ds4_gpu_tensor_free(k16); ds4_gpu_tensor_free(v16);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(o); ds4_gpu_tensor_free(small);
}

/* ---- full attention ---- */

#define CTX 1100u

/* A full layer's cache: rows [0, n) of K and V as f16 at their position rows, written with the one op the forward
 * pass will use (ds4_gpu_tensor_copy_f32_to_f16 at dst_offset = pos0 * KV_W * 2). */
static void full_cache_write(ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, const seq_t *s, uint32_t pos0, uint32_t n) {
    ds4_gpu_tensor *kt = upload(s->k + (uint64_t)pos0 * KV_W, (uint64_t)n * KV_W);
    ds4_gpu_tensor *vt = upload(s->v + (uint64_t)pos0 * KV_W, (uint64_t)n * KV_W);
    require_ok(ds4_gpu_tensor_copy_f32_to_f16(kc, (uint64_t)pos0 * KV_W * 2u, kt, 0, (uint64_t)n * KV_W), "full k write");
    require_ok(ds4_gpu_tensor_copy_f32_to_f16(vc, (uint64_t)pos0 * KV_W * 2u, vt, 0, (uint64_t)n * KV_W), "full v write");
    ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(vt);
}

/* Every position p in [0, n) through the single-row dispatch: out is [n][Q_W]. */
static void full_single_rows(const seq_t *s, ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, uint32_t n, float *out) {
    ds4_gpu_tensor *q1 = ds4_gpu_tensor_alloc(Q_W * 4u), *o1 = ds4_gpu_tensor_alloc(Q_W * 4u);
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(ds4_gpu_qwen4_attn_part_floats(1, MELLUM_N_HEAD, MELLUM_HEAD_DIM) * 4u);
    require_ok(q1 && o1 && part, "single-row tensors");
    for (uint32_t p = 0; p < n; p++) {
        require_ok(ds4_gpu_tensor_write(q1, 0, s->q + (uint64_t)p * Q_W, Q_W * 4u), "q upload");
        require_ok(ds4_gpu_qwen4_attn_decode_ungated_tensor(o1, q1, kc, vc, part, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                                            MELLUM_HEAD_DIM, p, attn_scale()), "ungated single-row");
        require_ok(ds4_gpu_tensor_read(o1, 0, out + (uint64_t)p * Q_W, Q_W * 4u), "out download");
    }
    ds4_gpu_tensor_free(q1); ds4_gpu_tensor_free(o1); ds4_gpu_tensor_free(part);
}

/* rows [pos0, pos0 + rows) through the rows path: out is [rows][Q_W]. */
static void full_rows(const seq_t *s, ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, uint32_t pos0, uint32_t rows, float *out) {
    const uint32_t slice = rows < 64u ? rows : 64u;
    ds4_gpu_tensor *qt = upload(s->q + (uint64_t)pos0 * Q_W, (uint64_t)rows * Q_W), *ot = ds4_gpu_tensor_alloc((uint64_t)rows * Q_W * 4u);
    ds4_gpu_tensor *table = ds4_gpu_tensor_alloc((uint64_t)rows * DS4_GPU_QWEN4_ATTN_ROW_BYTES);
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(ds4_gpu_qwen4_attn_part_floats(slice, MELLUM_N_HEAD, MELLUM_HEAD_DIM) * 4u);
    require_ok(ot && table && part, "rows tensors");
    if (g_batched) require_ok(ds4_gpu_begin_commands(), "begin commands");
    require_ok(ds4_gpu_mellum_attn_full_rows_tensor(ot, qt, kc, vc, table, 0, part, rows, pos0, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                                    MELLUM_HEAD_DIM, attn_scale()), "ungated rows");
    if (g_batched) require_ok(ds4_gpu_end_commands(), "end commands");
    require_ok(ds4_gpu_tensor_read(ot, 0, out, (uint64_t)rows * Q_W * 4u), "out download");
    ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(ot); ds4_gpu_tensor_free(table); ds4_gpu_tensor_free(part);
}

/* Full attention, ungated, at n keys: the last rows of the single-row dispatch and of the rows path against the
 * reference (window none); then every position's row from the rows path bit-identical to its single-row step. */
static void test_full_session(uint32_t n) {
    seq_t s = make_seq(n);
    const uint64_t cache_rows = n > CTX ? n : CTX;   /* 2,100 and 4,100 keys outgrow the 1,100-row cache of the other tests */
    ds4_gpu_tensor *kc = zeroed(cache_rows * KV_W * 2u), *vc = zeroed(cache_rows * KV_W * 2u);
    full_cache_write(kc, vc, &s, 0, n);
    /* the cache holds f16 K at position rows */
    uint16_t *raw = malloc((uint64_t)n * KV_W * 2u);
    require_ok(raw && ds4_gpu_tensor_read(kc, 0, raw, (uint64_t)n * KV_W * 2u), "cache read");
    uint64_t wrong = 0;
    for (uint64_t i = 0; i < (uint64_t)n * KV_W; i++) {
        __fp16 h;
        memcpy(&h, &raw[i], 2);
        if ((float)h != s.kr[i]) wrong++;
    }
    CHECK(wrong == 0, "the full cache must hold K rounded to f16 at position rows: %llu of %llu words differ",
          (unsigned long long)wrong, (unsigned long long)n * KV_W);
    free(raw);
    float *one = malloc((uint64_t)n * Q_W * 4u), *many = malloc((uint64_t)n * Q_W * 4u);
    require_ok(one && many, "malloc");
    full_single_rows(&s, kc, vc, n, one);
    full_rows(&s, kc, vc, 0, n, many);
    uint64_t first = 0;
    const uint64_t d = bit_diffs(one, many, (uint64_t)n * Q_W, &first);
    const uint32_t tail = n < 8u ? n : 8u;
    const double e1 = rows_vs_ref(&s, one, n - tail, n, 0, true), e2 = rows_vs_ref(&s, many, n - tail, n, 0, true);
    CHECK(all_finite(one, (uint64_t)n * Q_W), "full attention at %u keys produced a non-finite value", n);
    CHECK(e1 <= ATTN_TOL && e2 <= ATTN_TOL, "full ungated at %u keys vs the reference: single-row %.3e rows %.3e (tol %.0e)", n, e1, e2, ATTN_TOL);
    CHECK(d == 0, "full rows path at %u keys differs from the single-row dispatch in %llu words, first at row %llu", n,
          (unsigned long long)d, (unsigned long long)(first / Q_W));
    printf("  full     %4u keys: last %u rows vs reference single-row %.2e rows %.2e (tol %.0e); identity rows vs single-row: %llu of %llu words differ\n",
           n, tail, e1, e2, ATTN_TOL, (unsigned long long)d, (unsigned long long)n * Q_W);
    {
        /* the rows path inside one command buffer: the CPU writes each slice's table entries while the earlier slices
         * have not run yet, so a slice that reused entries would read the last slice's rows */
        float *bat = malloc((uint64_t)n * Q_W * 4u);
        require_ok(bat != NULL, "malloc");
        g_batched = true;
        full_rows(&s, kc, vc, 0, n, bat);
        g_batched = false;
        const uint64_t db = bit_diffs(one, bat, (uint64_t)n * Q_W, &first);
        CHECK(db == 0, "full rows path in one command buffer at %u keys differs from the single-row dispatch in %llu words, first at row %llu",
              n, (unsigned long long)db, (unsigned long long)(first / Q_W));
        printf("  full     %4u keys: rows path in one command buffer: %llu of %llu words differ from the single-row steps\n",
               n, (unsigned long long)db, (unsigned long long)n * Q_W);
        free(bat);
    }
    {
        /* the 32-lane merge, which the single-row dispatch takes off an M5 (the rows path always runs the wide one):
         * the identity must hold there too, and the merge kernel must not change the single-row output */
        float *one0 = malloc((uint64_t)n * Q_W * 4u), *many0 = malloc((uint64_t)n * Q_W * 4u);
        require_ok(one0 && many0, "malloc");
        setenv("DS4_QWEN4_ATTN_MERGE_WIDE", "0", 1);
        full_single_rows(&s, kc, vc, n, one0);
        full_rows(&s, kc, vc, 0, n, many0);
        unsetenv("DS4_QWEN4_ATTN_MERGE_WIDE");
        const uint64_t d_id = bit_diffs(one0, many0, (uint64_t)n * Q_W, &first), d_x = bit_diffs(one0, one, (uint64_t)n * Q_W, NULL);
        const double e0 = rows_vs_ref(&s, one0, n - tail, n, 0, true);
        CHECK(d_id == 0, "with the 32-lane merge, the rows path at %u keys differs from the single-row dispatch in %llu words, first at row %llu",
              n, (unsigned long long)d_id, (unsigned long long)(first / Q_W));
        CHECK(d_x == 0, "the single-row output at %u keys changed with the merge kernel: %llu words", n, (unsigned long long)d_x);
        CHECK(e0 <= ATTN_TOL, "32-lane merge, single-row at %u keys vs the reference: %.3e (tol %.0e)", n, e0, ATTN_TOL);
        printf("  full     %4u keys: 32-lane merge: rows vs single-row %llu, single-row vs the wide merge's %llu of %llu words differ; vs reference %.2e\n",
               n, (unsigned long long)d_id, (unsigned long long)d_x, (unsigned long long)n * Q_W, e0);
        free(one0); free(many0);
    }
    if (n == 1025) {
        /* a chunk that does not start at 0: rows [961, 1025) of the same cache */
        float *chunk = malloc(64ull * Q_W * 4u);
        require_ok(chunk != NULL, "malloc");
        full_rows(&s, kc, vc, 961, 64, chunk);
        const uint64_t dc = bit_diffs(chunk, one + 961ull * Q_W, 64ull * Q_W, NULL);
        CHECK(dc == 0, "full rows [961,1025) differ from the single-row dispatch in %llu words", (unsigned long long)dc);
        printf("  full     rows [961,1025) from pos0=961: %llu of %llu words differ from the single-row steps\n",
               (unsigned long long)dc, 64ull * Q_W);
        free(chunk);
    }
    free(one); free(many);
    ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc);
    free_seq(&s);
}

/* Two full layers in one command buffer, as the forward pass runs them: layers with different caches, one rows
 * table, each call with its own entries.  The CPU writes the table when a call encodes and the GPU reads it when
 * the buffer runs, so calls that shared entries would all see the last call's cache addresses (control). */
static void test_full_two_layers(void) {
    const uint32_t n = 100;
    seq_t a = make_seq(n), b = make_seq(n);
    ds4_gpu_tensor *kca = zeroed((uint64_t)CTX * KV_W * 2u), *vca = zeroed((uint64_t)CTX * KV_W * 2u);
    ds4_gpu_tensor *kcb = zeroed((uint64_t)CTX * KV_W * 2u), *vcb = zeroed((uint64_t)CTX * KV_W * 2u);
    full_cache_write(kca, vca, &a, 0, n);
    full_cache_write(kcb, vcb, &b, 0, n);
    float *ref_a = malloc((uint64_t)n * Q_W * 4u), *ref_b = malloc((uint64_t)n * Q_W * 4u);
    float *got_a = malloc((uint64_t)n * Q_W * 4u), *got_b = malloc((uint64_t)n * Q_W * 4u);
    require_ok(ref_a && ref_b && got_a && got_b, "malloc");
    full_single_rows(&a, kca, vca, n, ref_a);
    full_single_rows(&b, kcb, vcb, n, ref_b);
    ds4_gpu_tensor *qa = upload(a.q, (uint64_t)n * Q_W), *qb = upload(b.q, (uint64_t)n * Q_W);
    ds4_gpu_tensor *oa = ds4_gpu_tensor_alloc((uint64_t)n * Q_W * 4u), *ob = ds4_gpu_tensor_alloc((uint64_t)n * Q_W * 4u);
    ds4_gpu_tensor *table = zeroed(2ull * n * DS4_GPU_QWEN4_ATTN_ROW_BYTES);
    ds4_gpu_tensor *part = zeroed(ds4_gpu_qwen4_attn_part_floats(64, MELLUM_N_HEAD, MELLUM_HEAD_DIM) * 4u);
    require_ok(qa && qb && oa && ob, "layer tensors");
    uint64_t d_a = 0, d_b = 0, d_ctl = 0;
    for (int shared = 0; shared <= 1; shared++) {
        require_ok(ds4_gpu_begin_commands(), "begin commands");
        require_ok(ds4_gpu_mellum_attn_full_rows_tensor(oa, qa, kca, vca, table, 0, part, n, 0, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                                        MELLUM_HEAD_DIM, attn_scale()), "layer A");
        require_ok(ds4_gpu_mellum_attn_full_rows_tensor(ob, qb, kcb, vcb, table, shared ? 0 : n, part, n, 0, MELLUM_N_HEAD,
                                                        MELLUM_N_HEAD_KV, MELLUM_HEAD_DIM, attn_scale()), "layer B");
        require_ok(ds4_gpu_end_commands(), "end commands");
        require_ok(ds4_gpu_tensor_read(oa, 0, got_a, (uint64_t)n * Q_W * 4u), "read A");
        require_ok(ds4_gpu_tensor_read(ob, 0, got_b, (uint64_t)n * Q_W * 4u), "read B");
        if (!shared) {
            d_a = bit_diffs(ref_a, got_a, (uint64_t)n * Q_W, NULL);
            d_b = bit_diffs(ref_b, got_b, (uint64_t)n * Q_W, NULL);
        } else {
            d_ctl = bit_diffs(ref_a, got_a, (uint64_t)n * Q_W, NULL);
        }
    }
    CHECK(d_a == 0 && d_b == 0, "two layers with separate table regions in one command buffer differ from their single-row steps: A %llu, B %llu words",
          (unsigned long long)d_a, (unsigned long long)d_b);
    CHECK(d_ctl > 0, "control: two layers sharing one table region in one command buffer must corrupt layer A, but it matched");
    printf("  full     two layers (two caches, one table) in one command buffer: entry0 0 and %u: A %llu, B %llu of %llu words differ; sharing entries: A %llu differ (the control)\n",
           n, (unsigned long long)d_a, (unsigned long long)d_b, (unsigned long long)n * Q_W, (unsigned long long)d_ctl);
    free(ref_a); free(ref_b); free(got_a); free(got_b);
    ds4_gpu_tensor_free(kca); ds4_gpu_tensor_free(vca); ds4_gpu_tensor_free(kcb); ds4_gpu_tensor_free(vcb);
    ds4_gpu_tensor_free(qa); ds4_gpu_tensor_free(qb); ds4_gpu_tensor_free(oa); ds4_gpu_tensor_free(ob);
    ds4_gpu_tensor_free(table); ds4_gpu_tensor_free(part);
    free_seq(&a); free_seq(&b);
}

static void fill_f32(ds4_gpu_tensor *t, uint64_t n, float v) {
    float *h = malloc(n * 4u);
    require_ok(h != NULL, "malloc");
    for (uint64_t i = 0; i < n; i++) h[i] = v;
    require_ok(ds4_gpu_tensor_write(t, 0, h, n * 4u), "fill");
    free(h);
}

static double max_abs(const float *v, uint64_t n) {
    double m = 0.0;
    for (uint64_t i = 0; i < n; i++) if (fabs((double)v[i]) > m) m = fabs((double)v[i]);
    return m;
}

/* The qwen sibling.  The gated path keeps its signature and its arithmetic (the evidence directory hashes it
 * before and after the change, bit for bit); here the flag's two sides meet: a gate of +20 has
 * sigmoid(20) == 1.0f exactly in f32, so the gated kernels must give the ungated output bit for bit (the tile
 * written directly at 26 keys, the merge at 1,025 keys, the wide and the 32-lane merge, and the rows kernel),
 * and a gate of -20 must still gate (output about 2e-9 of the ungated), and a real gate must match the
 * reference times sigmoid(gate).  A NULL gate is refused by the gated entry and accepted by the ungated one. */
static void test_qwen_sibling(void) {
    static const uint32_t counts[] = { 26, 1025 };
    for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
        const uint32_t n = counts[c];
        seq_t s = make_seq(n);
        ds4_gpu_tensor *kc = zeroed((uint64_t)CTX * KV_W * 2u), *vc = zeroed((uint64_t)CTX * KV_W * 2u);
        full_cache_write(kc, vc, &s, 0, n);
        ds4_gpu_tensor *q = upload(s.q + (uint64_t)(n - 1) * Q_W, Q_W), *o = ds4_gpu_tensor_alloc(Q_W * 4u);
        ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(Q_W * 4u);
        ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(ds4_gpu_qwen4_attn_part_floats(1, MELLUM_N_HEAD, MELLUM_HEAD_DIM) * 4u);
        require_ok(o && gate && part, "sibling tensors");
        float *ung = malloc(Q_W * 4u), *g20 = malloc(Q_W * 4u), *gm20 = malloc(Q_W * 4u), *gr = malloc(Q_W * 4u);
        require_ok(ung && g20 && gm20 && gr, "malloc");
        require_ok(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, q, kc, vc, part, MELLUM_N_HEAD, MELLUM_N_HEAD_KV, MELLUM_HEAD_DIM,
                                                            n - 1, attn_scale()), "ungated");
        require_ok(ds4_gpu_tensor_read(o, 0, ung, Q_W * 4u), "read");
        fill_f32(gate, Q_W, 20.0f);
        for (int wide = 0; wide <= 1; wide++) {
            setenv("DS4_QWEN4_ATTN_MERGE_WIDE", wide ? "1" : "0", 1);
            require_ok(ds4_gpu_qwen4_attn_decode_tensor(o, q, gate, kc, vc, NULL, NULL, part, 1, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                                        MELLUM_HEAD_DIM, n - 1, false, 0, attn_scale()), "gated +20");
            require_ok(ds4_gpu_tensor_read(o, 0, g20, Q_W * 4u), "read");
            const uint64_t d = bit_diffs(ung, g20, Q_W, NULL);
            CHECK(d == 0, "gate +20 (sigmoid exactly 1) at %u keys, %s merge: the gated path differs from the ungated in %llu words", n,
                  wide ? "wide" : "32-lane", (unsigned long long)d);
            /* the ungated output itself does not depend on which merge kernel runs */
            require_ok(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, q, kc, vc, part, MELLUM_N_HEAD, MELLUM_N_HEAD_KV, MELLUM_HEAD_DIM,
                                                                n - 1, attn_scale()), "ungated");
            require_ok(ds4_gpu_tensor_read(o, 0, gr, Q_W * 4u), "read");
            CHECK(bit_diffs(ung, gr, Q_W, NULL) == 0, "the ungated output at %u keys changed with the merge kernel (%s)", n, wide ? "wide" : "32-lane");
        }
        unsetenv("DS4_QWEN4_ATTN_MERGE_WIDE");
        /* the rows kernel with a gate of +20 */
        {
            ds4_gpu_tensor *table = ds4_gpu_tensor_alloc(DS4_GPU_QWEN4_ATTN_ROW_BYTES);
            ds4_gpu_qwen4_attn_row row;
            row.k_cache = kc; row.v_cache = vc; row.ik_cache = kc; row.block_key = kc; row.pos3 = kc; row.pos = n - 1; row.use_sel = 0;
            require_ok(table && ds4_gpu_qwen4_attn_rows_stage(table, 0, &row, 1, 1u), "stage");
            require_ok(ds4_gpu_qwen4_attn_decode_rows_tensor(o, q, gate, q, q, part, table, 0, &row, 1, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                                            MELLUM_HEAD_DIM, 0, attn_scale()), "gated rows");
            require_ok(ds4_gpu_tensor_read(o, 0, g20, Q_W * 4u), "read");
            const uint64_t d = bit_diffs(ung, g20, Q_W, NULL);
            CHECK(d == 0, "gate +20 through the gated rows kernel at %u keys differs from the ungated in %llu words", n, (unsigned long long)d);
            ds4_gpu_tensor_free(table);
        }
        fill_f32(gate, Q_W, -20.0f);
        require_ok(ds4_gpu_qwen4_attn_decode_tensor(o, q, gate, kc, vc, NULL, NULL, part, 1, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                                    MELLUM_HEAD_DIM, n - 1, false, 0, attn_scale()), "gated -20");
        require_ok(ds4_gpu_tensor_read(o, 0, gm20, Q_W * 4u), "read");
        const double ratio = max_abs(gm20, Q_W) / max_abs(ung, Q_W);
        CHECK(ratio < 1e-8 && ratio > 0.0, "gate -20 at %u keys must still gate: output / ungated = %.3e (want about 2e-9)", n, ratio);
        /* a real gate: reference times sigmoid(gate) */
        float *gh = rand_vec(Q_W, 2.0f);
        require_ok(ds4_gpu_tensor_write(gate, 0, gh, Q_W * 4u), "gate upload");
        require_ok(ds4_gpu_qwen4_attn_decode_tensor(o, q, gate, kc, vc, NULL, NULL, part, 1, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                                    MELLUM_HEAD_DIM, n - 1, false, 0, attn_scale()), "gated real");
        require_ok(ds4_gpu_tensor_read(o, 0, gr, Q_W * 4u), "read");
        double *ref = malloc(Q_W * 8u);
        require_ok(ref != NULL, "malloc");
        ref_attn_row(s.q + (uint64_t)(n - 1) * Q_W, s.kr, s.vr, n - 1, 0, ref);
        for (uint32_t i = 0; i < Q_W; i++) ref[i] *= 1.0 / (1.0 + exp(-(double)gh[i]));
        const double e_gate = max_rel(gr, ref, Q_W);
        CHECK(e_gate <= ATTN_TOL, "the gated path at %u keys vs reference * sigmoid(gate): %.3e (tol %.0e)", n, e_gate, ATTN_TOL);
        /* NULL gate: the gated entry refuses, the ungated has none to refuse */
        CHECK(ds4_gpu_qwen4_attn_decode_tensor(o, q, NULL, kc, vc, NULL, NULL, part, 1, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                               MELLUM_HEAD_DIM, n - 1, false, 0, attn_scale()) == 0, "the gated entry refuses a NULL gate");
        CHECK(ds4_gpu_qwen4_attn_decode_tensor(o, q, gate, kc, vc, NULL, NULL, part, 1, MELLUM_N_HEAD, MELLUM_N_HEAD_KV,
                                               MELLUM_HEAD_DIM, n - 1, false, 0, attn_scale()) != 0, "the gated entry accepts a real gate");
        printf("  qwen     %4u keys: gate +20 == ungated (wide and 32-lane merge, and rows): 0 words differ; gate -20 gives %.1e of ungated; real gate vs ref*sigmoid %.2e\n",
               n, ratio, e_gate);
        free(gh); free(ref); free(ung); free(g20); free(gm20); free(gr);
        ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(o);
        ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(part);
        free_seq(&s);
    }
}

static void test_full_refusals(void) {
    ds4_gpu_tensor *kc = zeroed((uint64_t)CTX * KV_W * 2u), *vc = zeroed((uint64_t)CTX * KV_W * 2u);
    ds4_gpu_tensor *q = zeroed(Q_W * 4u * 2u), *o = zeroed(Q_W * 4u * 2u);
    ds4_gpu_tensor *part = zeroed(ds4_gpu_qwen4_attn_part_floats(2, MELLUM_N_HEAD, MELLUM_HEAD_DIM) * 4u);
    ds4_gpu_tensor *table = zeroed(2ull * DS4_GPU_QWEN4_ATTN_ROW_BYTES), *small = zeroed(64);
    const float sc = attn_scale();
    CHECK(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, q, kc, vc, part, 32, 4, 128, 5, sc) != 0, "a valid ungated single-row call succeeds");
    CHECK(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, q, kc, vc, part, 32, 4, 128, CTX, sc) == 0, "a position past the cache is refused");
    CHECK(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, q, kc, vc, part, 32, 4, 128, CTX - 1u, sc) != 0, "the last cache row is accepted");
    CHECK(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, q, kc, vc, part, 30, 4, 128, 5, sc) == 0, "30 heads over 4 is refused");
    CHECK(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, q, kc, vc, part, 16, 2, 256, 5, sc) == 0,
          "head_dim 256 is refused when ungated (the gated 256 path has a matrix-tile kernel that is not told about the flag)");
    CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, q, kc, vc, table, 0, part, 2, 5, 16, 2, 256, sc) == 0, "head_dim 256 rows are refused");
    CHECK(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, small, kc, vc, part, 32, 4, 128, 5, sc) == 0, "an undersized q is refused");
    CHECK(ds4_gpu_qwen4_attn_decode_ungated_tensor(small, q, kc, vc, part, 32, 4, 128, 5, sc) == 0, "an undersized out is refused");
    CHECK(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, q, small, vc, part, 32, 4, 128, 5, sc) == 0, "an undersized k cache is refused");
    CHECK(ds4_gpu_qwen4_attn_decode_ungated_tensor(o, q, kc, vc, small, 32, 4, 128, 1000, sc) == 0, "undersized partials are refused when the row splits");
    CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, q, kc, vc, table, 0, part, 2, 5, 32, 4, 128, sc) != 0, "a valid rows call succeeds");
    CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, q, kc, vc, table, 0, part, 0, 5, 32, 4, 128, sc) == 0, "zero rows are refused");
    CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, q, kc, vc, table, 0, part, 2, CTX - 1u, 32, 4, 128, sc) == 0, "rows past the cache are refused");
    CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, q, kc, vc, small, 0, part, 2, 5, 32, 4, 128, sc) == 0, "an undersized rows table is refused");
    {
        /* the entries are entry0 .. entry0 + n_rows - 1: a table of two does not hold two rows from entry 1 (sibling: three does);
         * a huge entry0 is refused (its byte offset would wrap) */
        ds4_gpu_tensor *table3 = zeroed(3ull * DS4_GPU_QWEN4_ATTN_ROW_BYTES);
        CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, q, kc, vc, table, 1, part, 2, 5, 32, 4, 128, sc) == 0,
              "two rows from entry 1 do not fit a table of two entries");
        CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, q, kc, vc, table3, 1, part, 2, 5, 32, 4, 128, sc) != 0,
              "two rows from entry 1 fit a table of three entries");
        CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, q, kc, vc, table3, 1ull << 58, part, 2, 5, 32, 4, 128, sc) == 0,
              "an entry0 whose byte offset wraps 64-bit arithmetic (2^58: the size check would see 128 bytes) is refused");
        ds4_gpu_tensor_free(table3);
    }
    {
        /* 100 rows are two slices of the rows kernels: a table that holds only the first slice must be refused
         * before the first dispatch, so out is left as it was (sibling: a table for 100 rows is accepted) */
        ds4_gpu_tensor *q100 = zeroed(100ull * Q_W * 4u), *o100 = zeroed(100ull * Q_W * 4u);
        ds4_gpu_tensor *part64 = zeroed(ds4_gpu_qwen4_attn_part_floats(64, MELLUM_N_HEAD, MELLUM_HEAD_DIM) * 4u);
        ds4_gpu_tensor *table64 = zeroed(64ull * DS4_GPU_QWEN4_ATTN_ROW_BYTES), *table100 = zeroed(100ull * DS4_GPU_QWEN4_ATTN_ROW_BYTES);
        float *mark = malloc(100ull * Q_W * 4u), *back = malloc(100ull * Q_W * 4u);
        require_ok(q100 && o100 && part64 && table64 && table100 && mark && back, "slice tensors");
        for (uint64_t i = 0; i < 100ull * Q_W; i++) mark[i] = 7.0f;
        require_ok(ds4_gpu_tensor_write(o100, 0, mark, 100ull * Q_W * 4u), "mark out");
        CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o100, q100, kc, vc, table64, 0, part64, 100, 5, 32, 4, 128, sc) == 0,
              "a rows table for 64 rows is refused for 100 rows");
        require_ok(ds4_gpu_tensor_read(o100, 0, back, 100ull * Q_W * 4u), "read out");
        CHECK(memcmp(mark, back, 100ull * Q_W * 4u) == 0, "a refused rows call must not have run its first slice");
        CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o100, q100, kc, vc, table100, 0, part64, 100, 5, 32, 4, 128, sc) != 0,
              "a rows table for 100 rows is accepted for 100 rows");
        /* the same from entry 10: 110 entries are needed, a table of 100 holds the first slice only (sibling: 110 is accepted) */
        ds4_gpu_tensor *table110 = zeroed(110ull * DS4_GPU_QWEN4_ATTN_ROW_BYTES);
        require_ok(table110 != NULL, "table110");
        require_ok(ds4_gpu_tensor_write(o100, 0, mark, 100ull * Q_W * 4u), "mark out");
        CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o100, q100, kc, vc, table100, 10, part64, 100, 5, 32, 4, 128, sc) == 0,
              "100 rows from entry 10 do not fit a table of 100 entries");
        require_ok(ds4_gpu_tensor_read(o100, 0, back, 100ull * Q_W * 4u), "read out");
        CHECK(memcmp(mark, back, 100ull * Q_W * 4u) == 0, "a call refused for its entry0 must not have run its first slice");
        CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o100, q100, kc, vc, table110, 10, part64, 100, 5, 32, 4, 128, sc) != 0,
              "100 rows from entry 10 fit a table of 110 entries");
        ds4_gpu_tensor_free(table110);
        free(mark); free(back);
        ds4_gpu_tensor_free(q100); ds4_gpu_tensor_free(o100); ds4_gpu_tensor_free(part64);
        ds4_gpu_tensor_free(table64); ds4_gpu_tensor_free(table100);
    }
    CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, small, kc, vc, table, 0, part, 2, 5, 32, 4, 128, sc) == 0, "an undersized q is refused");
    CHECK(ds4_gpu_mellum_attn_full_rows_tensor(o, q, kc, vc, table, 0, small, 2, 5, 32, 4, 128, sc) == 0, "undersized partials are refused");
    ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(o);
    ds4_gpu_tensor_free(part); ds4_gpu_tensor_free(table); ds4_gpu_tensor_free(small);
}


/* ======================================================================
 * The MoE driver (ds4_gpu_mellum_moe_tensor).
 *
 * Mellum's FFN at its real shape: F32 router 2304 -> 64, softmax over all 64,
 * top 8 (ties to the lower index), renormalize the eight, SwiGLU Q8_0 experts
 * of width 896, weighted sum, and no shared expert.  The reference is the
 * model's router and MoE in double (tests/mellum_moe_ref.inc), reading the same
 * Q8_0 bytes the kernels read.  The weight arena is poisoned:
 * every byte no declared tensor owns is 0xFF, which is NaN as an f32 and as an
 * f16 scale, so a read of a shared-expert tensor or a stray offset makes the
 * output NaN instead of quietly wrong.
 * ====================================================================== */

#include "mellum_moe_ref.inc"

/* f32 accumulation over 2,304 and 896 terms against a double reference on the same bytes.  Ten times the worst
 * measured (output 7.4e-07 at T=64; weights 2.2e-07), not a derived bound. */
#define MOE_TOL 1e-5
#define MOE_W_TOL 2.5e-6

typedef struct {
    uint8_t *base;
    uint64_t size, used;
    uint64_t router;      /* random f32 [64][2304] */
    uint64_t router_tie;  /* rows built from one x: logits are chosen, 10 and 40 tie at the 8th/9th place */
    uint64_t router_near; /* the same, with expert 40 above expert 10 by 1e-3 */
    uint64_t gate, up, down;
    /* declared and poisoned: a shared-expert input gate and gate/up/down rows; the driver must never read them */
    uint64_t sh_gate_inp, sh_gate, sh_up, sh_down;
    float x0[MREF_E];      /* the input the tie routers are built for */
    uint32_t expert_type;  /* MREF_TYPE_Q8_0 or MREF_TYPE_MXFP4 (S6): the routed experts' type */
} moe_arena_t;

/* one routed-expert row of in values in the arena's expert type */
static uint64_t moe_row_bytes(const moe_arena_t *a, uint32_t in) {
    return (uint64_t)(in / 32u) * (a->expert_type == MREF_TYPE_MXFP4 ? MREF_MXFP4_BLOCK_BYTES : MREF_Q8_BLOCK_BYTES);
}

static uint64_t g_rng64 = 0x2545F4914F6CDD1Dull;

static uint64_t rng64(void) {
    g_rng64 ^= g_rng64 << 13;
    g_rng64 ^= g_rng64 >> 7;
    g_rng64 ^= g_rng64 << 17;
    return g_rng64;
}

static uint64_t moe_take(moe_arena_t *a, uint64_t bytes) {
    const uint64_t off = (a->used + 255u) & ~255ull;
    require_ok(off + bytes <= a->size, "moe arena");
    a->used = off + bytes;
    return off;
}

/* random Q8_0 blocks: scale in 2e-4 .. 8e-4 (f16), quants in -127..127, so a weight is about 0.03 and a
 * 2,304-term dot of unit-variance inputs is about 1.5 */
static void fill_q8(uint8_t *dst, uint64_t n_blocks) {
    for (uint64_t i = 0; i < n_blocks; i++) {
        uint8_t *b = dst + i * MREF_Q8_BLOCK_BYTES;
        const __fp16 d = (__fp16)(0.0002f + 0.00004f * (float)(rng64() & 15u));
        memcpy(b, &d, 2);
        for (uint32_t j = 0; j < 32u; j += 8u) {
            const uint64_t v = rng64();
            for (uint32_t k = 0; k < 8u; k++) {
                int q = (int)((v >> (8u * k)) & 0xffu) - 128;
                if (q < -127) q = -127;
                b[2u + j + k] = (uint8_t)(int8_t)q;
            }
        }
    }
}

/* random MXFP4 blocks (S6): scale 2^-8 .. 2^-6 (E8M0 119..121), random E2M1 nibbles, so a weight is at most
 * 6 * 2^-6 = 0.09 and about 0.02 typically, near fill_q8's */
static void fill_mxfp4(uint8_t *dst, uint64_t n_blocks) {
    for (uint64_t i = 0; i < n_blocks; i++) {
        uint8_t *b = dst + i * MREF_MXFP4_BLOCK_BYTES;
        b[0] = (uint8_t)(119u + rng64() % 3u);
        const uint64_t v0 = rng64(), v1 = rng64();
        for (uint32_t j = 0; j < 8u; j++) { b[1u + j] = (uint8_t)(v0 >> (8u * j)); b[9u + j] = (uint8_t)(v1 >> (8u * j)); }
    }
}

static void fill_router_random(float *w, float amp) {
    for (uint64_t i = 0; i < (uint64_t)MREF_NEXP * MREF_E; i++) w[i] = amp * frand();
}

/* A router whose logit for expert e is alpha[e] on input x0: row e = alpha[e] * x0 / |x0|^2 */
static void fill_router_for(float *w, const float *x0, const float *alpha) {
    double n2 = 0.0;
    for (uint32_t i = 0; i < MREF_E; i++) n2 += (double)x0[i] * x0[i];
    for (uint32_t e = 0; e < MREF_NEXP; e++)
        for (uint32_t i = 0; i < MREF_E; i++) w[(uint64_t)e * MREF_E + i] = (float)(alpha[e] * x0[i] / n2);
}

static ds4_gpu_mellum_moe moe_params(const moe_arena_t *a, uint64_t router_off) {
    ds4_gpu_mellum_moe p;
    memset(&p, 0, sizeof(p));
    p.n_embd = MREF_E;
    p.n_expert = MREF_NEXP;
    p.n_used = MREF_USED;
    p.ff_dim = MREF_FF;
    p.expert_type = a->expert_type;
    p.router_offset = router_off;
    p.gate_offset = a->gate;
    p.up_offset = a->up;
    p.down_offset = a->down;
    return p;
}

/* ---- the double reference ---- */

static void ref_moe_row(const moe_arena_t *a, uint64_t router_off, const float *x, mref_t *r) {
    const mref_weights w = { a->base, router_off, a->gate, a->up, a->down, a->expert_type };
    mref_moe_row(&w, x, r);
}

/* ---- running the driver ---- */

typedef struct {
    uint32_t T;
    ds4_gpu_tensor *x, *logits, *sel, *w, *mid, *part, *out;
} moe_run_t;

static moe_run_t moe_run_alloc(uint32_t T) {
    moe_run_t r;
    r.T = T;
    r.x = zeroed((uint64_t)T * MREF_E * 4u);
    r.logits = zeroed((uint64_t)T * MREF_NEXP * 4u);
    r.sel = zeroed((uint64_t)T * MREF_USED * 4u);
    r.w = zeroed((uint64_t)T * MREF_USED * 4u);
    r.mid = zeroed((uint64_t)T * MREF_USED * MREF_FF * 4u);
    r.part = zeroed((uint64_t)T * MREF_USED * MREF_E * 4u);
    r.out = zeroed((uint64_t)T * MREF_E * 4u);
    return r;
}

static void moe_run_free(moe_run_t *r) {
    ds4_gpu_tensor_free(r->x); ds4_gpu_tensor_free(r->logits); ds4_gpu_tensor_free(r->sel); ds4_gpu_tensor_free(r->w);
    ds4_gpu_tensor_free(r->mid); ds4_gpu_tensor_free(r->part); ds4_gpu_tensor_free(r->out);
}

static int moe_call(const moe_arena_t *a, const ds4_gpu_mellum_moe *p, moe_run_t *r) {
    return ds4_gpu_mellum_moe_tensor(r->out, r->x, r->logits, r->sel, r->w, r->mid, r->part, a->base, a->size, p, r->T);
}

typedef struct {
    uint32_t T;
    int32_t *sel;  /* [T][8] */
    float *w;      /* [T][8] */
    float *out;    /* [T][2304] */
} moe_out_t;

static moe_out_t moe_read(const moe_run_t *r) {
    moe_out_t o;
    o.T = r->T;
    o.sel = malloc((uint64_t)r->T * MREF_USED * 4u);
    o.w = malloc((uint64_t)r->T * MREF_USED * 4u);
    o.out = malloc((uint64_t)r->T * MREF_E * 4u);
    require_ok(o.sel && o.w && o.out, "malloc");
    require_ok(ds4_gpu_tensor_read(r->sel, 0, o.sel, (uint64_t)r->T * MREF_USED * 4u), "sel download");
    require_ok(ds4_gpu_tensor_read(r->w, 0, o.w, (uint64_t)r->T * MREF_USED * 4u), "w download");
    require_ok(ds4_gpu_tensor_read(r->out, 0, o.out, (uint64_t)r->T * MREF_E * 4u), "out download");
    return o;
}

static void moe_out_free(moe_out_t *o) { free(o->sel); free(o->w); free(o->out); }

static moe_out_t moe_run_rows(const moe_arena_t *a, uint64_t router_off, const float *x, uint32_t T) {
    const ds4_gpu_mellum_moe p = moe_params(a, router_off);
    moe_run_t r = moe_run_alloc(T);
    require_ok(ds4_gpu_tensor_write(r.x, 0, x, (uint64_t)T * MREF_E * 4u), "x upload");
    require_ok(moe_call(a, &p, &r), "mellum moe");
    moe_out_t o = moe_read(&r);
    moe_run_free(&r);
    return o;
}

/* Rows [0, n) of a run against the reference rows: the worst output error, the worst weight error, the worst
 * |sum(w) - 1|, and how many selected experts and rows are wrong. */
typedef struct { double out, w, wsum; uint64_t sel_bad, nonfinite; } moe_err_t;

static moe_err_t moe_vs_ref(const moe_out_t *o, const mref_t *ref, uint32_t n) {
    moe_err_t e = { 0, 0, 0, 0, 0 };
    for (uint32_t t = 0; t < n; t++) {
        double wsum = 0.0;
        for (uint32_t r = 0; r < MREF_USED; r++) {
            if (o->sel[(uint64_t)t * MREF_USED + r] != ref[t].sel[r]) e.sel_bad++;
            const double d = fabs((double)o->w[(uint64_t)t * MREF_USED + r] - ref[t].w[r]);
            if (!(d <= e.w)) e.w = d;
            wsum += o->w[(uint64_t)t * MREF_USED + r];
        }
        if (!(fabs(wsum - 1.0) <= e.wsum)) e.wsum = fabs(wsum - 1.0);
        if (!all_finite(o->out + (uint64_t)t * MREF_E, MREF_E)) e.nonfinite++;
        const double d = max_rel(o->out + (uint64_t)t * MREF_E, ref[t].out, MREF_E);
        if (!(d <= e.out)) e.out = d;
    }
    return e;
}

#define MOE_ROWS_MAX 64u

/* One row, then 7 and 64 rows, against the double reference (softmax over all, top-8, renormalize, SwiGLU
 * experts, weighted sum); the rows of the 64-row run are bit-identical to the 7-row and 1-row runs (rows do not
 * see each other: the decode arithmetic over rows); a 1,024-row chunk keeps rows 0..63 and a far row correct. */
static void test_moe_reference(const moe_arena_t *a) {
    float *x = rand_vec((uint64_t)1024 * MREF_E, 1.7f);
    mref_t *ref = calloc(MOE_ROWS_MAX, sizeof(mref_t));
    require_ok(ref != NULL, "calloc");
    for (uint32_t t = 0; t < MOE_ROWS_MAX; t++) ref_moe_row(a, a->router, x + (uint64_t)t * MREF_E, &ref[t]);
    /* the control: renormalization must be visible, so the eight softmax probabilities must not already sum to 1 */
    double min_gap = 1.0;
    for (uint32_t t = 0; t < MOE_ROWS_MAX; t++) min_gap = fmin(min_gap, 1.0 - ref[t].top_mass);
    CHECK(min_gap > 0.01, "the eight selected probabilities already sum to %.4f in a row: renormalization would be invisible",
          1.0 - min_gap);

    static const uint32_t counts[] = { 1, 7, 64 };
    moe_out_t runs[3];
    for (int i = 0; i < 3; i++) {
        runs[i] = moe_run_rows(a, a->router, x, counts[i]);
        const moe_err_t e = moe_vs_ref(&runs[i], ref, counts[i]);
        CHECK(e.sel_bad == 0, "T=%u: %llu selected experts differ from the reference", counts[i], (unsigned long long)e.sel_bad);
        CHECK(e.nonfinite == 0, "T=%u: %llu rows have a non-finite output", counts[i], (unsigned long long)e.nonfinite);
        CHECK(e.w <= MOE_W_TOL, "T=%u: renormalized weights are %.3e from the reference (tol %.1e)", counts[i], e.w, MOE_W_TOL);
        CHECK(e.wsum <= 1e-6, "T=%u: the weights sum to 1 within only %.3e", counts[i], e.wsum);
        CHECK(e.out <= MOE_TOL, "T=%u: output is %.3e from the reference (max|d|/max|ref|, tol %.0e)", counts[i], e.out, MOE_TOL);
        printf("  moe    T=%-4u vs double reference: selected differ %llu, weights %.2e, |sum(w)-1| %.2e, output %.2e (tol %.0e)\n",
               counts[i], (unsigned long long)e.sel_bad, e.w, e.wsum, e.out, MOE_TOL);
    }
    /* the rows of a bigger call equal the same rows of a smaller one, bit for bit */
    for (int i = 0; i < 2; i++) {
        const uint32_t n = counts[i];
        uint64_t first = 0;
        const uint64_t d = bit_diffs(runs[i].out, runs[2].out, (uint64_t)n * MREF_E, &first);
        const bool sel_same = memcmp(runs[i].sel, runs[2].sel, (uint64_t)n * MREF_USED * 4u) == 0;
        const uint64_t dw = bit_diffs(runs[i].w, runs[2].w, (uint64_t)n * MREF_USED, NULL);
        CHECK(d == 0 && sel_same && dw == 0,
              "T=%u rows differ from the same rows of the T=64 run: %llu output words (first at row %llu), selected same %d, weights %llu",
              n, (unsigned long long)d, (unsigned long long)(first / MREF_E), (int)sel_same, (unsigned long long)dw);
    }
    printf("  moe    rows of T=64 vs T=1 and T=7: 0 words differ (output, weights, selected)\n");
    /* a prefill chunk: 1,024 rows. rows 0..63 keep their bits; a far row is right against the reference. */
    moe_out_t big = moe_run_rows(a, a->router, x, 1024);
    CHECK(all_finite(big.out, (uint64_t)1024 * MREF_E), "T=1024 produced a non-finite value");
    uint64_t first = 0;
    const uint64_t d = bit_diffs(big.out, runs[2].out, (uint64_t)MOE_ROWS_MAX * MREF_E, &first);
    CHECK(d == 0, "T=1024 rows 0..63 differ from the T=64 run in %llu words, first at row %llu", (unsigned long long)d,
          (unsigned long long)(first / MREF_E));
    mref_t far;
    ref_moe_row(a, a->router, x + (uint64_t)1023 * MREF_E, &far);
    moe_out_t row1023 = { 1, big.sel + (uint64_t)1023 * MREF_USED, big.w + (uint64_t)1023 * MREF_USED, big.out + (uint64_t)1023 * MREF_E };
    const moe_err_t e = moe_vs_ref(&row1023, &far, 1);
    CHECK(e.sel_bad == 0 && e.w <= MOE_W_TOL && e.out <= MOE_TOL, "T=1024 row 1023: selected differ %llu, weights %.3e, output %.3e",
          (unsigned long long)e.sel_bad, e.w, e.out);
    printf("  moe    T=1024: rows 0..63 vs T=64 %llu of %u words differ; row 1023 vs reference: selected differ %llu, weights %.2e, output %.2e\n",
           (unsigned long long)d, MOE_ROWS_MAX * MREF_E, (unsigned long long)e.sel_bad, e.w, e.out);
    for (int i = 0; i < 3; i++) moe_out_free(&runs[i]);
    moe_out_free(&big);
    free(ref);
    free(x);
}

/* ---- ties ---- */

static void tie_row(float *l, const int *anchors, int n_anchors, float anchor_hi, const int *grp, int n_grp, float gval) {
    for (uint32_t e = 0; e < MREF_NEXP; e++) l[e] = -3.0f - 0.03f * (float)((e * 37u) % 64u); /* distinct, below -3 */
    for (int i = 0; i < n_anchors; i++) l[anchors[i]] = anchor_hi - 0.5f * (float)i;
    for (int i = 0; i < n_grp; i++) l[grp[i]] = gval;
}

/* The router op on hand-made logits: a tie at the boundary between the 8th and 9th place goes to the lower index,
 * and a difference of 1e-3 does not (the sibling: the comparison is a comparison, not "lowest index wins"). */
static void test_moe_ties_direct(void) {
    enum { R = 7 };
    static const int anchors7[] = { 60, 55, 45, 30, 25, 12, 8 };
    static const int anchors5[] = { 60, 55, 45, 30, 25 };
    static const int anchors6[] = { 60, 55, 45, 30, 25, 12 };
    float logits[R][MREF_NEXP];
    int expect[R][MREF_USED];
    const char *what[R] = {
        "tie 10/40 at places 8/9 (two simdgroups)", "tie 3/20 at places 8/9 (one simdgroup)",
        "four-way tie 1/2/33/63 for three places", "all 64 equal",
        "10 and 40 differ by 1e-3, 40 above", "tie 7/50 for the top place", "tie 31/32 at places 8/9 (lane 31, lane 0)",
    };
    static const int g0[] = { 10, 40 }, g1[] = { 3, 20 }, g2[] = { 63, 33, 2, 1 }, g5[] = { 50, 7 }, g6[] = { 31, 32 };
    tie_row(logits[0], anchors7, 7, 6.0f, g0, 2, 0.0f);
    tie_row(logits[1], anchors7, 7, 6.0f, g1, 2, 0.0f);
    tie_row(logits[2], anchors5, 5, 6.0f, g2, 4, 0.0f);
    for (uint32_t e = 0; e < MREF_NEXP; e++) logits[3][e] = 0.0f;
    tie_row(logits[4], anchors7, 7, 6.0f, g0, 2, 0.0f);
    logits[4][40] = 0.001f;
    tie_row(logits[5], anchors6, 6, 6.0f, g5, 2, 8.0f);
    tie_row(logits[6], anchors7, 7, 6.0f, g6, 2, 0.0f);
    static const int e0[] = { 60, 55, 45, 30, 25, 12, 8, 10 }, e1[] = { 60, 55, 45, 30, 25, 12, 8, 3 };
    static const int e2[] = { 60, 55, 45, 30, 25, 1, 2, 33 }, e3[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    static const int e4[] = { 60, 55, 45, 30, 25, 12, 8, 40 }, e5[] = { 7, 50, 60, 55, 45, 30, 25, 12 };
    static const int e6[] = { 60, 55, 45, 30, 25, 12, 8, 31 };
    memcpy(expect[0], e0, sizeof(e0)); memcpy(expect[1], e1, sizeof(e1)); memcpy(expect[2], e2, sizeof(e2));
    memcpy(expect[3], e3, sizeof(e3)); memcpy(expect[4], e4, sizeof(e4)); memcpy(expect[5], e5, sizeof(e5));
    memcpy(expect[6], e6, sizeof(e6));

    ds4_gpu_tensor *lt = upload(&logits[0][0], (uint64_t)R * MREF_NEXP);
    ds4_gpu_tensor *st = zeroed((uint64_t)R * MREF_USED * 4u), *wt = zeroed((uint64_t)R * MREF_USED * 4u);
    require_ok(ds4_gpu_qwen4_router_topk_tensor(st, wt, lt, NULL, NULL, 0, 0, 0, 0, NULL, R, MREF_NEXP, MREF_USED), "router top-k");
    int32_t sel[R][MREF_USED];
    float w[R][MREF_USED];
    require_ok(ds4_gpu_tensor_read(st, 0, sel, sizeof(sel)) && ds4_gpu_tensor_read(wt, 0, w, sizeof(w)), "router download");
    for (int r = 0; r < R; r++) {
        double ld[MREF_NEXP], rw[MREF_USED], top;
        int rs[MREF_USED];
        for (uint32_t e = 0; e < MREF_NEXP; e++) ld[e] = logits[r][e];
        mref_route(ld, MREF_NEXP, MREF_USED, rs, rw, &top);
        double worst_w = 0.0;
        int bad = 0, bad_ref = 0;
        for (uint32_t k = 0; k < MREF_USED; k++) {
            bad += sel[r][k] != expect[r][k];
            bad_ref += sel[r][k] != rs[k];
            const double d = fabs((double)w[r][k] - rw[k]);
            if (!(d <= worst_w)) worst_w = d;
        }
        CHECK(bad == 0, "%s: selected %d %d %d %d %d %d %d %d, expected %d %d %d %d %d %d %d %d", what[r],
              sel[r][0], sel[r][1], sel[r][2], sel[r][3], sel[r][4], sel[r][5], sel[r][6], sel[r][7],
              expect[r][0], expect[r][1], expect[r][2], expect[r][3], expect[r][4], expect[r][5], expect[r][6], expect[r][7]);
        CHECK(bad_ref == 0, "%s: the double reference disagrees with the selection in %d places", what[r], bad_ref);
        CHECK(worst_w <= MOE_W_TOL, "%s: weights are %.3e from the reference", what[r], worst_w);
        printf("  router %-44s place 8 -> expert %-2d, weights %.2e from double\n", what[r], sel[r][7], worst_w);
    }
    ds4_gpu_tensor_free(lt); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt);
}

/* The same through the driver: routers built so one input gives chosen logits.  Experts 10 and 40 share one
 * router row in the first, so their logits are the same f32 bits, and they sit at places 8 and 9; in the second
 * 40 is above 10 by 1e-3.  Seven rows of the same input must agree with each other bit for bit (a row's position
 * in the call changes nothing). */
static void test_moe_ties_driver(const moe_arena_t *a) {
    float x7[7 * MREF_E];
    for (int r = 0; r < 7; r++) memcpy(x7 + (uint64_t)r * MREF_E, a->x0, MREF_E * sizeof(float));
    for (int pass = 0; pass < 2; pass++) {
        const uint64_t router = pass == 0 ? a->router_tie : a->router_near;
        mref_t ref;
        ref_moe_row(a, router, a->x0, &ref);
        const int want = pass == 0 ? 10 : 40, other = pass == 0 ? 40 : 10;
        CHECK(ref.sel[7] == want, "the reference itself must pick expert %d at place 8, picked %d", want, ref.sel[7]);
        moe_out_t o = moe_run_rows(a, router, x7, 7);
        bool same = true;
        int wrong = 0;
        for (int r = 0; r < 7; r++) {
            same = same && memcmp(o.sel + (uint64_t)r * MREF_USED, o.sel, MREF_USED * 4u) == 0 &&
                   memcmp(o.w + (uint64_t)r * MREF_USED, o.w, MREF_USED * 4u) == 0 &&
                   memcmp(o.out + (uint64_t)r * MREF_E, o.out, MREF_E * 4u) == 0;
            wrong += o.sel[(uint64_t)r * MREF_USED + 7] != want;
            for (int k = 0; k < 8; k++) wrong += o.sel[(uint64_t)r * MREF_USED + k] == other;
        }
        CHECK(wrong == 0, "%s: a row did not pick expert %d at place 8 (or picked expert %d)",
              pass == 0 ? "exact tie 10/40" : "40 above 10 by 1e-3", want, other);
        CHECK(same, "seven rows of one input must agree bit for bit");
        const moe_err_t e = moe_vs_ref(&o, &ref, 1);
        CHECK(e.sel_bad == 0 && e.out <= MOE_TOL && e.w <= MOE_W_TOL, "driver on the %s router: selected differ %llu, weights %.3e, output %.3e",
              pass == 0 ? "tie" : "near-tie", (unsigned long long)e.sel_bad, e.w, e.out);
        printf("  moe    %-36s -> expert %d at place 8 in 7 of 7 rows; rows agree bit for bit; output %.2e from double\n",
               pass == 0 ? "exact tie 10/40 (router rows equal)" : "40 above 10 by 1e-3", want, e.out);
        moe_out_free(&o);
    }
}

/* ---- no shared expert ---- */

/* The arena's shared-expert tensors are NaN.  The driver reads no shared tensor and takes no offset for one: its
 * outputs on this arena are finite and equal the reference.  The control: the same bytes read by the qwen path's
 * shared-expert slot are NaN, so the poison would have been seen. */
static void test_moe_no_shared(const moe_arena_t *a) {
    bool intact = true;
    const uint64_t spans[4][2] = { { a->sh_gate_inp, MREF_E * 4u }, { a->sh_gate, (uint64_t)MREF_FF * MREF_ROW_GATE },
                                   { a->sh_up, (uint64_t)MREF_FF * MREF_ROW_GATE }, { a->sh_down, (uint64_t)MREF_E * MREF_ROW_DOWN } };
    for (int s = 0; s < 4; s++)
        for (uint64_t i = 0; i < spans[s][1]; i++) intact = intact && a->base[spans[s][0] + i] == 0xFFu;
    CHECK(intact, "the shared-expert poison was overwritten");

    float *x = rand_vec(MREF_E, 1.7f);
    mref_t ref;
    ref_moe_row(a, a->router, x, &ref);
    moe_out_t o = moe_run_rows(a, a->router, x, 1);
    CHECK(all_finite(o.out, MREF_E), "the driver's output is not finite on an arena whose shared tensors are NaN");
    const moe_err_t e = moe_vs_ref(&o, &ref, 1);
    CHECK(e.out <= MOE_TOL && e.sel_bad == 0, "driver output on the poisoned arena is %.3e from the reference", e.out);

    /* control: the shared slot and the shared gate of the qwen kernels, over the poison, are NaN */
    float zeros[MREF_NEXP] = { 0 };
    ds4_gpu_tensor *xt = upload(x, MREF_E), *sel = zeroed(MREF_USED * 4u), *w = zeroed(MREF_USED * 4u);
    ds4_gpu_tensor *lg = upload(zeros, MREF_NEXP), *sg = zeroed(4u), *mid9 = zeroed((MREF_USED + 1u) * MREF_FF * 4u);
    require_ok(ds4_gpu_qwen4_router_topk_tensor(sel, w, lg, xt, a->base, a->size, a->sh_gate_inp, 0u, MREF_E, sg, 1, MREF_NEXP, MREF_USED),
               "router with a shared gate");
    float gl = 0.0f;
    require_ok(ds4_gpu_tensor_read(sg, 0, &gl, 4), "shared gate readback");
    CHECK(isnan(gl), "control: the shared gate logit read over the poison must be NaN, got %g", (double)gl);
    require_ok(ds4_gpu_qwen4_moe_mid_tensor(mid9, xt, sel, a->base, a->size, a->gate, a->up, a->expert_type, MREF_NEXP, 1, MREF_USED, MREF_E, MREF_FF,
                                            a->sh_gate, a->sh_up, 8u), "mid with a shared slot");
    float shared_mid[MREF_FF];
    require_ok(ds4_gpu_tensor_read(mid9, (uint64_t)MREF_USED * MREF_FF * 4u, shared_mid, sizeof(shared_mid)), "shared slot readback");
    CHECK(!all_finite(shared_mid, MREF_FF), "control: the shared slot read over the poison must be NaN");
    printf("  moe    no shared tensor: outputs finite and %.2e from double on an arena whose shared tensors are NaN; "
           "control: the shared gate logit and the shared slot over the same bytes are NaN\n", e.out);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(sel); ds4_gpu_tensor_free(w);
    ds4_gpu_tensor_free(lg); ds4_gpu_tensor_free(sg); ds4_gpu_tensor_free(mid9);
    moe_out_free(&o);
    free(x);
}

/* ---- refusals: each refused call changes nothing; the sibling is the valid call that follows ---- */

static void test_moe_refusals(const moe_arena_t *a) {
    float *x = rand_vec(2u * MREF_E, 1.7f);
    moe_run_t r = moe_run_alloc(2);
    require_ok(ds4_gpu_tensor_write(r.x, 0, x, 2u * MREF_E * 4u), "x upload");
    const ds4_gpu_mellum_moe good = moe_params(a, a->router);
    ds4_gpu_tensor *outs[] = { r.logits, r.sel, r.w, r.mid, r.part, r.out };
    const uint64_t out_bytes[] = { 2u * MREF_NEXP * 4u, 2u * MREF_USED * 4u, 2u * MREF_USED * 4u, 2u * MREF_USED * MREF_FF * 4u,
                                   2u * MREF_USED * MREF_E * 4u, 2u * MREF_E * 4u };
    float *back = malloc((uint64_t)2 * MREF_USED * MREF_E * 4u), *mk = malloc((uint64_t)2 * MREF_USED * MREF_E * 4u);
    require_ok(back != NULL && mk != NULL, "malloc");
    for (uint64_t i = 0; i < 2ull * MREF_USED * MREF_E; i++) mk[i] = 7.0f;
    /* mark every output, so a refused call that ran its first op shows */
#define MOE_REMARK() do { for (int i_ = 0; i_ < 6; i_++) require_ok(ds4_gpu_tensor_write(outs[i_], 0, mk, out_bytes[i_]), "mark"); } while (0)
#define MOE_UNTOUCHED(what) do { bool same_ = true; for (int i_ = 0; i_ < 6; i_++) { \
        require_ok(ds4_gpu_tensor_read(outs[i_], 0, back, out_bytes[i_]), "read"); same_ = same_ && memcmp(back, mk, out_bytes[i_]) == 0; } \
        CHECK(same_, "a call refused for %s must leave every output as it was", what); } while (0)
    ds4_gpu_mellum_moe p;

    MOE_REMARK();
    /* Q4_0 has valid row sizes at both widths, so only the type check refuses it; Q4_K is also refused by its
     * 256-wide blocks (896 is not a multiple of 256) */
    p = good; p.expert_type = 2u;
    CHECK(moe_call(a, &p, &r) == 0, "expert type Q4_0 is refused (Mellum's experts are Q8_0 or MXFP4)");
    MOE_UNTOUCHED("an expert type");
    p = good; p.expert_type = 12u;
    CHECK(moe_call(a, &p, &r) == 0, "expert type Q4_K is refused (Mellum's experts are Q8_0 or MXFP4)");
    MOE_UNTOUCHED("an expert type (Q4_K)");
    p = good; p.n_expert = 513u;
    CHECK(moe_call(a, &p, &r) == 0, "513 experts are refused");
    p = good; p.n_used = 0u;
    CHECK(moe_call(a, &p, &r) == 0, "top-0 is refused");
    p = good; p.n_used = 65u;
    CHECK(moe_call(a, &p, &r) == 0, "top-65 of 64 is refused");
    {
        /* top-17 with tensors big enough for 17 slots, so only the cap refuses it, and before the logits are
         * written (sibling: top-16 runs) */
        const uint64_t k17 = 17;
        ds4_gpu_tensor *s17 = zeroed(2 * k17 * 4u), *w17 = zeroed(2 * k17 * 4u), *m17 = zeroed(2 * k17 * MREF_FF * 4u),
                       *p17 = zeroed(2 * k17 * MREF_E * 4u);
        MOE_REMARK();
        p = good; p.n_used = 17u;
        CHECK(ds4_gpu_mellum_moe_tensor(r.out, r.x, r.logits, s17, w17, m17, p17, a->base, a->size, &p, 2) == 0, "top-17 is refused");
        MOE_UNTOUCHED("top-17");
        p = good; p.n_used = 16u;
        CHECK(ds4_gpu_mellum_moe_tensor(r.out, r.x, r.logits, s17, w17, m17, p17, a->base, a->size, &p, 2) != 0, "top-16 is accepted");
        ds4_gpu_tensor_free(s17); ds4_gpu_tensor_free(w17); ds4_gpu_tensor_free(m17); ds4_gpu_tensor_free(p17);
        MOE_REMARK();
    }
    p = good; p.n_embd = 2300u;
    CHECK(moe_call(a, &p, &r) == 0, "a width that is not a multiple of 128 is refused");
    p = good; p.ff_dim = 900u;
    CHECK(moe_call(a, &p, &r) == 0, "an expert width that is not a multiple of 32 is refused");
    MOE_UNTOUCHED("a shape");
    p = good; p.router_offset = a->size - 1024u;
    CHECK(moe_call(a, &p, &r) == 0, "a router past the end of the map is refused");
    MOE_UNTOUCHED("a router offset");
    p = good; p.down_offset = a->size - 1024u;
    CHECK(moe_call(a, &p, &r) == 0, "down experts past the end of the map are refused before the first dispatch");
    MOE_UNTOUCHED("a down offset");
    p = good; p.gate_offset = UINT64_MAX - 100u;
    CHECK(moe_call(a, &p, &r) == 0, "a gate offset that wraps 64-bit arithmetic is refused");
    MOE_UNTOUCHED("a wrapping offset");
    CHECK(ds4_gpu_mellum_moe_tensor(r.out, r.x, r.logits, r.sel, r.w, r.mid, r.part, a->base, a->size, &good, 0) == 0, "zero rows are refused");
    CHECK(ds4_gpu_mellum_moe_tensor(r.out, r.x, r.logits, r.sel, r.w, r.mid, r.part, a->base, a->size, NULL, 2) == 0, "no parameters are refused");
    CHECK(ds4_gpu_mellum_moe_tensor(r.out, r.x, r.logits, r.sel, r.w, r.mid, r.part, NULL, 0, &good, 2) == 0, "no model map is refused");
    MOE_UNTOUCHED("rows, parameters or the map");
    /* an undersized tensor, one at a time: refused up front, so nothing ran (32 bytes: selected and weights of two
     * rows need exactly 64) */
    ds4_gpu_tensor *small = zeroed(32);
    ds4_gpu_tensor *args[7] = { r.out, r.x, r.logits, r.sel, r.w, r.mid, r.part };
    const char *names[7] = { "out", "x", "router logits", "selected", "weights", "mid", "part" };
    for (int i = 0; i < 7; i++) {
        ds4_gpu_tensor *t[7];
        memcpy(t, args, sizeof(t));
        t[i] = small;
        MOE_REMARK();
        CHECK(ds4_gpu_mellum_moe_tensor(t[0], t[1], t[2], t[3], t[4], t[5], t[6], a->base, a->size, &good, 2) == 0,
              "an undersized %s is refused", names[i]);
        MOE_UNTOUCHED(names[i]);
    }
    /* the sibling: the valid call runs, on the same tensors the refusals left alone */
    CHECK(moe_call(a, &good, &r) != 0, "the valid call succeeds after the refusals");
    moe_out_t o = moe_read(&r);
    CHECK(all_finite(o.out, 2u * MREF_E), "the valid call's output is finite");
    moe_out_free(&o);
    printf("  moe    refusals: Q4_0 and Q4_K experts, 513 experts, top-0/17/65 (top-16 runs), width 2300, ff 900, router/down offset past the map, "
           "a wrapping offset, 0 rows, NULL params/map, 7 undersized tensors -> all refused, outputs untouched; valid call runs\n");
#undef MOE_REMARK
#undef MOE_UNTOUCHED
    ds4_gpu_tensor_free(small);
    moe_run_free(&r);
    free(back); free(mk); free(x);
}

/* ---- the driver's rows equal one-row calls, stage by stage ---- */

#define T4_BEGIN() do { if (g_batched) require_ok(ds4_gpu_begin_commands(), "begin commands"); } while (0)
#define T4_END() do { if (g_batched) require_ok(ds4_gpu_end_commands(), "end commands"); } while (0)

enum { T4_STAGES = 6 };
static const char *const T4_STAGE_NAME[T4_STAGES] = { "router logits", "selected experts", "weights", "mid", "part", "output" };

/* The six buffers of a run, read back as 32-bit words: logits [T][64], selected [T][8], weights [T][8],
 * mid [T][8][896], part [T][8][2304], out [T][2304]. */
typedef struct {
    uint32_t T;
    uint32_t *stage[T4_STAGES];
    uint64_t words[T4_STAGES]; /* words per row */
} moe_stages_t;

static const uint64_t T4_MOE_ROW_WORDS[T4_STAGES] = { MREF_NEXP, MREF_USED, MREF_USED, (uint64_t)MREF_USED * MREF_FF,
                                                       (uint64_t)MREF_USED * MREF_E, MREF_E };

static moe_stages_t moe_stages_read(const moe_run_t *r) {
    moe_stages_t s;
    s.T = r->T;
    ds4_gpu_tensor *const t[T4_STAGES] = { r->logits, r->sel, r->w, r->mid, r->part, r->out };
    for (int i = 0; i < T4_STAGES; i++) {
        s.words[i] = T4_MOE_ROW_WORDS[i];
        s.stage[i] = malloc((uint64_t)r->T * s.words[i] * 4u);
        require_ok(s.stage[i] != NULL, "malloc");
        require_ok(ds4_gpu_tensor_read(t[i], 0, s.stage[i], (uint64_t)r->T * s.words[i] * 4u), "stage download");
    }
    return s;
}

static void moe_stages_free(moe_stages_t *s) {
    for (int i = 0; i < T4_STAGES; i++) free(s->stage[i]);
}

/* T rows in one call, against T calls of one row each, on the same weights and the same inputs: every stage of
 * every row must match in raw bits.  The stages are compared one by one, so a failure names the first op that is
 * not independent of the batch.  The 130-row case has a tail that fills no whole tile of 8, 32 or 64 rows. */
static void test_moe_rows_equal_singles(const moe_arena_t *a) {
    static const uint32_t counts[] = { 7, 33, 64, 130 };
    const ds4_gpu_mellum_moe p = moe_params(a, a->router);
    for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
        const uint32_t T = counts[c];
        float *x = rand_vec((uint64_t)T * MREF_E, 1.7f);
        moe_run_t big = moe_run_alloc(T);
        require_ok(ds4_gpu_tensor_write(big.x, 0, x, (uint64_t)T * MREF_E * 4u), "x upload");
        T4_BEGIN();
        require_ok(moe_call(a, &p, &big), "mellum moe over rows");
        T4_END();
        moe_stages_t rows = moe_stages_read(&big);
        uint64_t diff[T4_STAGES] = { 0 }, first_row[T4_STAGES] = { 0 };
        moe_run_t one = moe_run_alloc(1);
        for (uint32_t t = 0; t < T; t++) {
            require_ok(ds4_gpu_tensor_write(one.x, 0, x + (uint64_t)t * MREF_E, MREF_E * 4u), "x row upload");
            require_ok(moe_call(a, &p, &one), "mellum moe one row");
            moe_stages_t single = moe_stages_read(&one);
            for (int s = 0; s < T4_STAGES; s++) {
                const uint64_t d = bit_diffs((const float *)(rows.stage[s] + (uint64_t)t * rows.words[s]),
                                             (const float *)single.stage[s], rows.words[s], NULL);
                if (d && !diff[s]) first_row[s] = t;
                diff[s] += d;
            }
            moe_stages_free(&single);
        }
        for (int s = 0; s < T4_STAGES; s++) {
            CHECK(diff[s] == 0, "MoE %s: T=%u rows differ from T one-row calls in %llu words (first row %llu)%s", T4_STAGE_NAME[s], T,
                  (unsigned long long)diff[s], (unsigned long long)first_row[s], g_batched ? " [batched]" : "");
        }
        CHECK(all_finite((const float *)rows.stage[5], (uint64_t)T * MREF_E), "MoE T=%u output is not finite", T);
        printf("  moe    T=%-3u rows vs %u one-row calls%s: words differing per stage (logits, selected, weights, mid, part, output) "
               "%llu %llu %llu %llu %llu %llu\n", T, T, g_batched ? " [batched]" : "", (unsigned long long)diff[0],
               (unsigned long long)diff[1], (unsigned long long)diff[2], (unsigned long long)diff[3], (unsigned long long)diff[4],
               (unsigned long long)diff[5]);
        moe_stages_free(&rows);
        moe_run_free(&big);
        moe_run_free(&one);
        free(x);
    }
}

/* Experiment E1's boundary.  The tensor tiles take Q8_0 experts only at the level Mellum sets around its own tiled
 * calls (ds4_gpu_qwen4_moe_q8_tiles); Flash Next's qwen4_graph_moe calls the same wrappers and never sets it, so its
 * Q8_0 experts keep the simdgroup tiles, under its own DS4_QWEN4_MOE_MM_NAX too.  64 random rows, each routed to
 * eight distinct experts, through build_lists, moe_mm_mid and moe_mm_down as such a caller runs them. */
static void moe_q8_tiled(const moe_arena_t *a, const float *x, const int32_t *sel, float *part) {
    enum { T = 64, K = 8 };
    ds4_gpu_tensor *xt = upload(x, (uint64_t)T * MREF_E), *st = zeroed((uint64_t)T * K * 4u);
    ds4_gpu_tensor *lists = zeroed((uint64_t)MREF_NEXP * T * 4u), *counts = zeroed(MREF_NEXP * 4u);
    ds4_gpu_tensor *mid = zeroed((uint64_t)T * K * MREF_FF * 4u), *pt = zeroed((uint64_t)T * K * MREF_E * 4u);
    require_ok(ds4_gpu_tensor_write(st, 0, sel, (uint64_t)T * K * 4u), "selected upload");
    require_ok(ds4_gpu_qwen4_moe_build_lists_tensor(lists, counts, st, T, K, MREF_NEXP, T), "build lists");
    require_ok(ds4_gpu_qwen4_moe_mm_mid_tensor(mid, xt, lists, counts, a->base, a->size, a->gate, a->up, 8u, MREF_NEXP, T,
                                               K, K, MREF_E, MREF_FF, T), "moe_mm mid, Q8_0");
    require_ok(ds4_gpu_qwen4_moe_mm_down_tensor(pt, mid, lists, counts, a->base, a->size, a->down, 8u, MREF_NEXP, T, K, K,
                                                MREF_FF, MREF_E, T), "moe_mm down, Q8_0");
    require_ok(ds4_gpu_tensor_read(pt, 0, part, (uint64_t)T * K * MREF_E * 4u), "part download");
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(lists); ds4_gpu_tensor_free(counts);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(pt);
}

static void test_moe_q8_tiles_mellum_only(const moe_arena_t *a) {
    enum { T = 64, K = 8 };
    const uint64_t n = (uint64_t)T * K * MREF_E;
    float *x = malloc((uint64_t)T * MREF_E * 4u), *plain = malloc(n * 4u), *flash = malloc(n * 4u), *tiles = malloc(n * 4u);
    int32_t *sel = malloc((uint64_t)T * K * 4u);
    require_ok(x && plain && flash && tiles && sel, "malloc");
    for (uint64_t i = 0; i < (uint64_t)T * MREF_E; i++) x[i] = 1.7f * frand();
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t k = 0; k < K; k++) sel[t * K + k] = (int32_t)((t * 7u + k * 9u) % MREF_NEXP);   /* distinct per row */
    moe_q8_tiled(a, x, sel, plain);
    setenv("DS4_QWEN4_MOE_MM_NAX", "2", 1);
    moe_q8_tiled(a, x, sel, flash);
    unsetenv("DS4_QWEN4_MOE_MM_NAX");
    ds4_gpu_qwen4_moe_q8_tiles(2);
    moe_q8_tiled(a, x, sel, tiles);
    ds4_gpu_qwen4_moe_q8_tiles(0);
    uint64_t d_flash = 0, d_tiles = 0, bad = 0;
    for (uint64_t i = 0; i < n; i++) {
        d_flash += memcmp(&plain[i], &flash[i], 4) != 0;
        d_tiles += memcmp(&plain[i], &tiles[i], 4) != 0;
        bad += !isfinite(plain[i]) || !isfinite(tiles[i]);
    }
    const bool avail = ds4_gpu_tensor_tiles_available() != 0;
    printf("  Q8_0 tiled MoE without Mellum's level: %llu of %llu words differ under DS4_QWEN4_MOE_MM_NAX=2; at Mellum's "
           "level 2 (%s): %llu differ\n", (unsigned long long)d_flash, (unsigned long long)n,
           avail ? "tensor tiles" : "no tensor API", (unsigned long long)d_tiles);
    CHECK(bad == 0, "non-finite output words: %llu", (unsigned long long)bad);
    CHECK(d_flash == 0, "a caller without Mellum's level took another path under DS4_QWEN4_MOE_MM_NAX=2 (%llu words)",
          (unsigned long long)d_flash);
    CHECK(avail ? d_tiles > 0 : d_tiles == 0, "Mellum's level %s the tensor tiles (%llu words differ)",
          avail ? "did not reach" : "changed the output without", (unsigned long long)d_tiles);
    free(x); free(plain); free(flash); free(tiles); free(sel);
}

/* ---- S4 address-table variants: each expert in its own slot buffer, reached through a uint64 GPU-address
 * table, must give the resident kernels' bits; a zero address reads nothing and writes nothing ---- */

typedef struct { ds4_gpu_tensor *gate[MREF_NEXP], *up[MREF_NEXP], *down[MREF_NEXP], *ga, *ua, *da; } addr_slots_t;

static void addr_slots_make(const moe_arena_t *a, addr_slots_t *s) {
    const uint64_t gb = (uint64_t)MREF_FF * moe_row_bytes(a, MREF_E), db = (uint64_t)MREF_E * moe_row_bytes(a, MREF_FF);
    uint64_t ga[MREF_NEXP], ua[MREF_NEXP], da[MREF_NEXP];
    for (uint32_t e = 0; e < MREF_NEXP; e++) {
        s->gate[e] = ds4_gpu_tensor_alloc(gb); s->up[e] = ds4_gpu_tensor_alloc(gb); s->down[e] = ds4_gpu_tensor_alloc(db);
        require_ok(s->gate[e] && s->up[e] && s->down[e] &&
                   ds4_gpu_tensor_write(s->gate[e], 0, a->base + a->gate + e * gb, gb) &&
                   ds4_gpu_tensor_write(s->up[e], 0, a->base + a->up + e * gb, gb) &&
                   ds4_gpu_tensor_write(s->down[e], 0, a->base + a->down + e * db, db), "slot copy");
        ga[e] = ds4_gpu_tensor_gpu_address(s->gate[e]); ua[e] = ds4_gpu_tensor_gpu_address(s->up[e]);
        da[e] = ds4_gpu_tensor_gpu_address(s->down[e]);
        require_ok(ga[e] && ua[e] && da[e], "slot address");
    }
    s->ga = ds4_gpu_tensor_alloc(sizeof(ga)); s->ua = ds4_gpu_tensor_alloc(sizeof(ua)); s->da = ds4_gpu_tensor_alloc(sizeof(da));
    require_ok(s->ga && s->ua && s->da && ds4_gpu_tensor_write(s->ga, 0, ga, sizeof(ga)) &&
               ds4_gpu_tensor_write(s->ua, 0, ua, sizeof(ua)) && ds4_gpu_tensor_write(s->da, 0, da, sizeof(da)), "address tables");
}

static void addr_slots_free(addr_slots_t *s) {
    for (uint32_t e = 0; e < MREF_NEXP; e++) {
        ds4_gpu_tensor_free(s->gate[e]); ds4_gpu_tensor_free(s->up[e]); ds4_gpu_tensor_free(s->down[e]);
    }
    ds4_gpu_tensor_free(s->ga); ds4_gpu_tensor_free(s->ua); ds4_gpu_tensor_free(s->da);
}

/* eight distinct experts, in a random order */
static void addr_pick(int32_t sel[MREF_USED]) {
    for (uint32_t k = 0; k < MREF_USED; k++) {
        int32_t e;
        bool dup;
        do {
            e = (int32_t)((uint32_t)((frand() + 1.0f) * 32.0f) % MREF_NEXP);
            dup = false;
            for (uint32_t j = 0; j < k; j++) dup = dup || sel[j] == e;
        } while (dup);
        sel[k] = e;
    }
}

static void test_moe_addr_decode(const moe_arena_t *a) {
    addr_slots_t s;
    addr_slots_make(a, &s);
    const uint64_t mid_bytes = (uint64_t)MREF_USED * MREF_FF * 4u, part_bytes = (uint64_t)MREF_USED * MREF_E * 4u;
    bool mid_eq = true, down_eq = true, mid_untouched = true, down_untouched = true, others_eq = true;
    for (int trial = 0; trial < 4; trial++) {
        int32_t sel_h[MREF_USED];
        addr_pick(sel_h);
        float *x = rand_vec(MREF_E, 1.7f);
        ds4_gpu_tensor *xt = upload(x, MREF_E), *sel = ds4_gpu_tensor_alloc(sizeof(sel_h));
        require_ok(sel && ds4_gpu_tensor_write(sel, 0, sel_h, sizeof(sel_h)), "selection");
        ds4_gpu_tensor *res[2u * MREF_USED], *dres[MREF_USED];
        for (uint32_t k = 0; k < MREF_USED; k++) {
            res[2u * k] = s.gate[sel_h[k]]; res[2u * k + 1u] = s.up[sel_h[k]]; dres[k] = s.down[sel_h[k]];
        }
        ds4_gpu_tensor *mid_r = zeroed(mid_bytes), *mid_a = zeroed(mid_bytes), *part_r = zeroed(part_bytes), *part_a = zeroed(part_bytes);
        require_ok(ds4_gpu_qwen4_moe_mid_tensor(mid_r, xt, sel, a->base, a->size, a->gate, a->up, 8u, MREF_NEXP, 1, MREF_USED,
                                                MREF_E, MREF_FF, 0u, 0u, UINT32_MAX), "resident mid");
        require_ok(ds4_gpu_qwen4_moe_mid_addr_tensor(mid_a, xt, sel, s.ga, s.ua, res, 2u * MREF_USED, 8u, MREF_NEXP, 1, MREF_USED,
                                                     MREF_E, MREF_FF), "addr mid");
        require_ok(ds4_gpu_qwen4_moe_down_tensor(part_r, mid_r, sel, a->base, a->size, a->down, 8u, MREF_NEXP, 1, MREF_USED,
                                                 MREF_FF, MREF_E, 0u, UINT32_MAX), "resident down");
        require_ok(ds4_gpu_qwen4_moe_down_addr_tensor(part_a, mid_r, sel, s.da, dres, MREF_USED, 8u, MREF_NEXP, 1, MREF_USED,
                                                      MREF_FF, MREF_E), "addr down");
        float *mr = download(mid_r, mid_bytes / 4u), *ma = download(mid_a, mid_bytes / 4u);
        float *pr = download(part_r, part_bytes / 4u), *pa = download(part_a, part_bytes / 4u);
        mid_eq = mid_eq && memcmp(mr, ma, mid_bytes) == 0;
        down_eq = down_eq && memcmp(pr, pa, part_bytes) == 0;
        /* zero the first selected expert's addresses: its rows keep a sentinel, the other seven still match */
        const uint64_t zero = 0u, e0 = (uint64_t)sel_h[0] * 8u;
        const uint64_t restore_g = ds4_gpu_tensor_gpu_address(s.gate[sel_h[0]]), restore_u = ds4_gpu_tensor_gpu_address(s.up[sel_h[0]]);
        const uint64_t restore_d = ds4_gpu_tensor_gpu_address(s.down[sel_h[0]]);
        require_ok(ds4_gpu_tensor_write(s.ga, e0, &zero, 8) && ds4_gpu_tensor_write(s.ua, e0, &zero, 8) &&
                   ds4_gpu_tensor_write(s.da, e0, &zero, 8), "zero address");
        float *sent = malloc(part_bytes);
        for (uint64_t i = 0; i < part_bytes / 4u; i++) sent[i] = 12345.0f;
        require_ok(ds4_gpu_tensor_write(mid_a, 0, sent, mid_bytes) && ds4_gpu_tensor_write(part_a, 0, sent, part_bytes), "sentinel");
        require_ok(ds4_gpu_qwen4_moe_mid_addr_tensor(mid_a, xt, sel, s.ga, s.ua, res, 2u * MREF_USED, 8u, MREF_NEXP, 1, MREF_USED,
                                                     MREF_E, MREF_FF), "addr mid, zero address");
        require_ok(ds4_gpu_qwen4_moe_down_addr_tensor(part_a, mid_r, sel, s.da, dres, MREF_USED, 8u, MREF_NEXP, 1, MREF_USED,
                                                      MREF_FF, MREF_E), "addr down, zero address");
        float *mz = download(mid_a, mid_bytes / 4u), *pz = download(part_a, part_bytes / 4u);
        mid_untouched = mid_untouched && memcmp(mz, sent, (uint64_t)MREF_FF * 4u) == 0;
        down_untouched = down_untouched && memcmp(pz, sent, (uint64_t)MREF_E * 4u) == 0;
        others_eq = others_eq && memcmp(mz + MREF_FF, mr + MREF_FF, mid_bytes - (uint64_t)MREF_FF * 4u) == 0 &&
                    memcmp(pz + MREF_E, pr + MREF_E, part_bytes - (uint64_t)MREF_E * 4u) == 0;
        require_ok(ds4_gpu_tensor_write(s.ga, e0, &restore_g, 8) && ds4_gpu_tensor_write(s.ua, e0, &restore_u, 8) &&
                   ds4_gpu_tensor_write(s.da, e0, &restore_d, 8), "restore address");
        free(sent); free(mr); free(ma); free(pr); free(pa); free(mz); free(pz); free(x);
        ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(sel); ds4_gpu_tensor_free(mid_r); ds4_gpu_tensor_free(mid_a);
        ds4_gpu_tensor_free(part_r); ds4_gpu_tensor_free(part_a);
    }
    CHECK(mid_eq, "addr: decode mid through slot addresses must equal the resident kernel's bits");
    CHECK(down_eq, "addr: decode down through slot addresses must equal the resident kernel's bits");
    CHECK(mid_untouched && down_untouched, "addr: a zero address must leave its expert's rows untouched");
    CHECK(others_eq, "addr: with one zero address the other seven experts must still equal the resident kernel");
    printf("  moe    addr decode: mid %s, down %s; zero address untouched %s\n", mid_eq ? "equal" : "DIFFER",
           down_eq ? "equal" : "DIFFER", mid_untouched && down_untouched ? "yes" : "NO");
    addr_slots_free(&s);
}

/* the tiled prefill path at Q8 tile level `level` (0: simdgroup tiles; 2: Mellum's E1 tensor tiles):
 * resident and address-table mid/down over the same lists give the same bits; a zero address leaves the
 * (token, slot) rows routed to that expert at a sentinel */
static void test_moe_addr_tiled(const moe_arena_t *a, uint32_t level) {
    addr_slots_t s;
    addr_slots_make(a, &s);
    ds4_gpu_tensor *gres[2u * MREF_NEXP], *dres[MREF_NEXP];
    for (uint32_t e = 0; e < MREF_NEXP; e++) { gres[2u * e] = s.gate[e]; gres[2u * e + 1u] = s.up[e]; dres[e] = s.down[e]; }
    static const uint32_t Ts[] = { 31, 32, 64, 100 };
    bool mid_eq = true, down_eq = true, untouched = true, others_eq = true;
    ds4_gpu_qwen4_moe_q8_tiles(level);
    for (size_t ti = 0; ti < sizeof(Ts) / sizeof(Ts[0]); ti++) {
        const uint32_t T = Ts[ti], K = MREF_USED;
        int32_t *sel_h = malloc((uint64_t)T * K * 4u);
        for (uint32_t t = 0; t < T; t++) addr_pick(sel_h + (uint64_t)t * K);
        float *x = rand_vec((uint64_t)T * MREF_E, 1.7f);
        const uint64_t mid_n = (uint64_t)T * K * MREF_FF, part_n = (uint64_t)T * K * MREF_E;
        ds4_gpu_tensor *xt = upload(x, (uint64_t)T * MREF_E), *st = zeroed((uint64_t)T * K * 4u);
        ds4_gpu_tensor *lists = zeroed((uint64_t)MREF_NEXP * T * 4u), *counts = zeroed(MREF_NEXP * 4u);
        ds4_gpu_tensor *mid_r = zeroed(mid_n * 4u), *mid_a = zeroed(mid_n * 4u), *part_r = zeroed(part_n * 4u), *part_a = zeroed(part_n * 4u);
        require_ok(ds4_gpu_tensor_write(st, 0, sel_h, (uint64_t)T * K * 4u) &&
                   ds4_gpu_qwen4_moe_build_lists_tensor(lists, counts, st, T, K, MREF_NEXP, T), "lists");
        require_ok(ds4_gpu_qwen4_moe_mm_mid_tensor(mid_r, xt, lists, counts, a->base, a->size, a->gate, a->up, 8u, MREF_NEXP, T,
                                                   K, K, MREF_E, MREF_FF, T), "resident mm mid");
        require_ok(ds4_gpu_qwen4_moe_mm_mid_addr_tensor(mid_a, xt, lists, counts, s.ga, s.ua, gres, 2u * MREF_NEXP, 8u, MREF_NEXP,
                                                        T, K, K, MREF_E, MREF_FF, T), "addr mm mid");
        require_ok(ds4_gpu_qwen4_moe_mm_down_tensor(part_r, mid_r, lists, counts, a->base, a->size, a->down, 8u, MREF_NEXP, T, K, K,
                                                    MREF_FF, MREF_E, T), "resident mm down");
        require_ok(ds4_gpu_qwen4_moe_mm_down_addr_tensor(part_a, mid_r, lists, counts, s.da, dres, MREF_NEXP, 8u, MREF_NEXP, T, K, K,
                                                         MREF_FF, MREF_E, T), "addr mm down");
        float *mr = download(mid_r, mid_n), *ma = download(mid_a, mid_n), *pr = download(part_r, part_n), *pa = download(part_a, part_n);
        mid_eq = mid_eq && memcmp(mr, ma, mid_n * 4u) == 0;
        down_eq = down_eq && memcmp(pr, pa, part_n * 4u) == 0;
        /* zero row 0 slot 0's expert */
        const int32_t e0 = sel_h[0];
        const uint64_t zero = 0u, restore[3] = { ds4_gpu_tensor_gpu_address(s.gate[e0]), ds4_gpu_tensor_gpu_address(s.up[e0]),
                                                 ds4_gpu_tensor_gpu_address(s.down[e0]) };
        require_ok(ds4_gpu_tensor_write(s.ga, (uint64_t)e0 * 8u, &zero, 8) && ds4_gpu_tensor_write(s.ua, (uint64_t)e0 * 8u, &zero, 8) &&
                   ds4_gpu_tensor_write(s.da, (uint64_t)e0 * 8u, &zero, 8), "zero address");
        float *sent = malloc(part_n * 4u);
        for (uint64_t i = 0; i < part_n; i++) sent[i] = 12345.0f;
        require_ok(ds4_gpu_tensor_write(mid_a, 0, sent, mid_n * 4u) && ds4_gpu_tensor_write(part_a, 0, sent, part_n * 4u), "sentinel");
        require_ok(ds4_gpu_qwen4_moe_mm_mid_addr_tensor(mid_a, xt, lists, counts, s.ga, s.ua, gres, 2u * MREF_NEXP, 8u, MREF_NEXP,
                                                        T, K, K, MREF_E, MREF_FF, T), "addr mm mid, zero address");
        require_ok(ds4_gpu_qwen4_moe_mm_down_addr_tensor(part_a, mid_r, lists, counts, s.da, dres, MREF_NEXP, 8u, MREF_NEXP, T, K, K,
                                                         MREF_FF, MREF_E, T), "addr mm down, zero address");
        float *mz = download(mid_a, mid_n), *pz = download(part_a, part_n);
        for (uint64_t pair = 0; pair < (uint64_t)T * K; pair++) {
            const bool hit = sel_h[pair] == e0;
            const float *m_got = mz + pair * MREF_FF, *p_got = pz + pair * MREF_E;
            if (hit) untouched = untouched && memcmp(m_got, sent, MREF_FF * 4u) == 0 && memcmp(p_got, sent, MREF_E * 4u) == 0;
            else others_eq = others_eq && memcmp(m_got, mr + pair * MREF_FF, MREF_FF * 4u) == 0 &&
                             memcmp(p_got, pr + pair * MREF_E, MREF_E * 4u) == 0;
        }
        require_ok(ds4_gpu_tensor_write(s.ga, (uint64_t)e0 * 8u, &restore[0], 8) && ds4_gpu_tensor_write(s.ua, (uint64_t)e0 * 8u, &restore[1], 8) &&
                   ds4_gpu_tensor_write(s.da, (uint64_t)e0 * 8u, &restore[2], 8), "restore address");
        free(sel_h); free(x); free(sent); free(mr); free(ma); free(pr); free(pa); free(mz); free(pz);
        ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(lists); ds4_gpu_tensor_free(counts);
        ds4_gpu_tensor_free(mid_r); ds4_gpu_tensor_free(mid_a); ds4_gpu_tensor_free(part_r); ds4_gpu_tensor_free(part_a);
    }
    ds4_gpu_qwen4_moe_q8_tiles(0);
    CHECK(mid_eq, "addr: tiled mid (level %u) through slot addresses must equal the resident kernel's bits", level);
    CHECK(down_eq, "addr: tiled down (level %u) through slot addresses must equal the resident kernel's bits", level);
    CHECK(untouched, "addr: tiled (level %u) rows routed to a zero address must stay untouched", level);
    CHECK(others_eq, "addr: tiled (level %u) rows of the other experts must still equal the resident kernel", level);
    printf("  moe    addr tiled level %u (T 31/32/64/100): mid %s, down %s; zero address untouched %s\n", level,
           mid_eq ? "equal" : "DIFFER", down_eq ? "equal" : "DIFFER", untouched ? "yes" : "NO");
    addr_slots_free(&s);
}

/* An arena of random weights whose routed experts are of type expert_type; the shared-expert poison stays
 * Q8_0-sized (it is never read as experts).  Registers the arena as the model map. */
static moe_arena_t *moe_arena_make(uint32_t expert_type) {
    moe_arena_t *a = calloc(1, sizeof(*a));
    require_ok(a != NULL, "calloc");
    a->expert_type = expert_type;
    const uint64_t gate_bytes = (uint64_t)MREF_NEXP * MREF_FF * moe_row_bytes(a, MREF_E);
    const uint64_t down_bytes = (uint64_t)MREF_NEXP * MREF_E * moe_row_bytes(a, MREF_FF);
    a->size = (uint64_t)448 << 20;
    a->base = mmap(NULL, a->size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (a->base == MAP_FAILED) { perror("mmap"); exit(1); }
    memset(a->base, 0xFF, a->size);
    a->router = moe_take(a, MREF_ROUTER_BYTES);
    a->router_tie = moe_take(a, MREF_ROUTER_BYTES);
    a->router_near = moe_take(a, MREF_ROUTER_BYTES);
    a->sh_gate_inp = moe_take(a, MREF_E * 4u);
    a->gate = moe_take(a, gate_bytes);
    a->up = moe_take(a, gate_bytes);
    a->down = moe_take(a, down_bytes);
    a->sh_gate = moe_take(a, (uint64_t)MREF_FF * MREF_ROW_GATE);
    a->sh_up = moe_take(a, (uint64_t)MREF_FF * MREF_ROW_GATE);
    a->sh_down = moe_take(a, (uint64_t)MREF_E * MREF_ROW_DOWN);
    /* logits about 2 in standard deviation: a softmax that is not flat, with the top eight holding most of its mass */
    fill_router_random((float *)(a->base + a->router), 0.072f);
    void (*fill)(uint8_t *, uint64_t) = expert_type == MREF_TYPE_MXFP4 ? fill_mxfp4 : fill_q8;
    fill(a->base + a->gate, (uint64_t)MREF_NEXP * MREF_FF * (MREF_E / 32u));
    fill(a->base + a->up, (uint64_t)MREF_NEXP * MREF_FF * (MREF_E / 32u));
    fill(a->base + a->down, (uint64_t)MREF_NEXP * MREF_E * (MREF_FF / 32u));
    /* the tie routers: expert e's logit on x0 is alpha[e]; experts 10 and 40 share a row (or 40 is 1e-3 above) */
    for (uint32_t i = 0; i < MREF_E; i++) a->x0[i] = 1.7f * frand();
    {
        float alpha[MREF_NEXP];
        static const int anchors[] = { 60, 55, 45, 30, 25, 12, 8 };
        for (uint32_t e = 0; e < MREF_NEXP; e++) alpha[e] = -1.0f - 0.01f * (float)((e * 37u) % 64u);
        for (int i = 0; i < 7; i++) alpha[anchors[i]] = 6.0f - 0.5f * (float)i;
        alpha[10] = 2.0f;
        alpha[40] = 2.0f;
        fill_router_for((float *)(a->base + a->router_tie), a->x0, alpha);
        alpha[40] = 2.001f;
        fill_router_for((float *)(a->base + a->router_near), a->x0, alpha);
    }
    require_ok(ds4_gpu_set_model_map(a->base, a->size), "model map registration");
    return a;
}

static void moe_arena_free(moe_arena_t *a) {
    munmap(a->base, a->size);
    free(a);
}

static void test_moe(void) {
    printf("Mellum MoE driver (2304 -> 64 F32 router, softmax over all, top-8, renormalized, 896-wide Q8_0 experts, no shared expert)\n");
    moe_arena_t *a = moe_arena_make(MREF_TYPE_Q8_0);
    test_moe_reference(a);
    test_moe_ties_direct();
    test_moe_ties_driver(a);
    test_moe_no_shared(a);
    test_moe_refusals(a);
    test_moe_q8_tiles_mellum_only(a);
    test_moe_addr_decode(a);
    test_moe_addr_tiled(a, 0u);
    /* E1 tensor tiles: level 2 is Mellum's default (64-wide half); 1, 3-6 cover the 32-wide, float and
     * compensated variants the same encoder can pick */
    for (uint32_t level = 1; level <= 6; level++) test_moe_addr_tiled(a, level);
    for (int batched = 0; batched <= 1; batched++) {
        g_batched = batched != 0;
        test_moe_rows_equal_singles(a);
    }
    g_batched = false;
    moe_arena_free(a);

    /* S6: the same driver on MXFP4 experts (GGUF type 39), against the double reference at type 39 */
    printf("Mellum MoE driver, MXFP4 experts (E8M0 scales, E2M1 values, 32-wide blocks of 17 bytes)\n");
    a = moe_arena_make(MREF_TYPE_MXFP4);
    test_moe_reference(a);
    test_moe_no_shared(a);
    test_moe_refusals(a);
    for (int batched = 0; batched <= 1; batched++) {
        g_batched = batched != 0;
        test_moe_rows_equal_singles(a);
    }
    g_batched = false;
    moe_arena_free(a);
}

/* ======================================================================
 * Prefill equals decode, op by op.
 *
 * The forward pass runs every op over a chunk's rows where decode runs it on one row.  For a chunk to give the same
 * bits as its decode steps, each op over T rows must equal T one-row calls, in raw bits (memcmp), on the same weights and inputs.  The
 * attention ops are tested above (test_swa_session, test_full_session: chunks against successive single-row steps)
 * and the MoE driver's stages in test_moe_rows_equal_singles; this section covers the rest.  Weights are random Q8_0
 * and f32 in an anonymous arena at the model's real shapes.  Each op is also checked against a double reference on
 * row 0, so a pair of equal but wrong outputs does not pass.  Controls are the batch-dependent ops upstream also has:
 * they must differ, or the tests would not see a reroute.
 * ====================================================================== */

#define T4_E 2304u /* Mellum's hidden width */

/* `MELLUM_T4_BREAK=<op>` (q8, router, rms, rope, eltwise) breaks the op under test in this section, so each
 * identity test is shown to fail: q8 and router run the rows through the batched upstream path instead of the exact
 * one, rms changes epsilon, rope shifts pos0 by one, eltwise adds a row to itself.  Unset, nothing changes. */
static bool t4_break(const char *op) {
    const char *b = getenv("MELLUM_T4_BREAK");
    return b != NULL && strcmp(b, op) == 0;
}

/* Prefill's batched Q8_0 projection and F32 router batch are the controls: they pick a different kernel by row count. */
static uint64_t g_t4_q8_control_diff;
static uint64_t g_t4_f32_control_diff;

typedef struct {
    arena_t arena;
    uint64_t q, k, v, o, router, norm;
    double *norm_shadow;
} t4_arena_t;

static uint64_t t4_q8_bytes(uint64_t in, uint64_t out) { return out * (in / 32u) * 34u; }

static uint64_t t4_take(t4_arena_t *a, uint64_t bytes) {
    const uint64_t off = (a->arena.used + 255u) & ~255ull;
    require_ok(off + bytes <= a->arena.size, "t4 arena");
    a->arena.used = off + bytes;
    return off;
}

/* max|d| / max|ref| of one Q8_0 projection row-0 against the double dot of the same bytes */
static double t4_q8_vs_ref(const t4_arena_t *a, uint64_t off, const float *x, const float *got, uint64_t in, uint64_t out) {
    double *xd = malloc(in * sizeof(double)), *ref = malloc(out * sizeof(double));
    require_ok(xd && ref, "malloc");
    for (uint64_t i = 0; i < in; i++) xd[i] = x[i];
    for (uint64_t r = 0; r < out; r++) ref[r] = mref_q8_dot(a->arena.base + off + r * (in / 32u) * 34u, xd, (uint32_t)in);
    const double d = max_rel(got, ref, out);
    free(xd);
    free(ref);
    return d;
}

/* the one-row reference of every Q8_0 projection: ds4_gpu_matmul_q8_0_tensor with one token, as decode runs it */
static void t4_q8_single_rows(const t4_arena_t *a, uint64_t off, uint64_t in, uint64_t out, const float *x, uint32_t T, float *dst) {
    ds4_gpu_tensor *x1 = zeroed(in * 4u), *o1 = zeroed(out * 4u);
    for (uint32_t t = 0; t < T; t++) {
        require_ok(ds4_gpu_tensor_write(x1, 0, x + (uint64_t)t * in, in * 4u), "x row upload");
        require_ok(ds4_gpu_matmul_q8_0_tensor(o1, a->arena.base, a->arena.size, off, in, out, x1, 1), "one-row Q8_0 projection");
        require_ok(ds4_gpu_tensor_read(o1, 0, dst + (uint64_t)t * out, out * 4u), "row download");
    }
    ds4_gpu_tensor_free(x1);
    ds4_gpu_tensor_free(o1);
}

#define T4_TOL_Q8 1e-5 /* f32 accumulation over up to 4,096 terms against double on the same bytes */

static void test_q8_projection(const t4_arena_t *a, const char *name, uint64_t off, uint64_t in, uint64_t out, uint32_t T) {
    float *x = rand_vec((uint64_t)T * in, 1.7f);
    ds4_gpu_tensor *xt = upload(x, (uint64_t)T * in), *ot = zeroed((uint64_t)T * out * 4u);
    T4_BEGIN();
    if (t4_break("q8")) {
        require_ok(ds4_gpu_matmul_q8_0_tensor(ot, a->arena.base, a->arena.size, off, in, out, xt, T), "batched projection (broken)");
    } else {
        require_ok(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ot, a->arena.base, a->arena.size, off, in, out, xt, T), "exact rows projection");
    }
    T4_END();
    float *rows = download(ot, (uint64_t)T * out), *one = malloc((uint64_t)T * out * 4u);
    require_ok(one != NULL, "malloc");
    t4_q8_single_rows(a, off, in, out, x, T, one);
    uint64_t first = 0;
    const uint64_t d = bit_diffs(rows, one, (uint64_t)T * out, &first);
    CHECK(d == 0, "Q8_0 %s (%llu -> %llu): %u rows through the exact rows kernel differ from %u one-row calls in %llu words, first at row %llu%s",
          name, (unsigned long long)in, (unsigned long long)out, T, T, (unsigned long long)d,
          (unsigned long long)(first / out), g_batched ? " [batched]" : "");
    const double e = t4_q8_vs_ref(a, off, x, one, in, out);
    CHECK(all_finite(rows, (uint64_t)T * out) && e <= T4_TOL_Q8, "Q8_0 %s one-row output is %.3e from the double reference", name, e);
    uint64_t cd = 0;
    {
        /* the control: the batched path of the ordinary projection, over the same rows */
        ds4_gpu_tensor *ct = zeroed((uint64_t)T * out * 4u);
        require_ok(ds4_gpu_matmul_q8_0_tensor(ct, a->arena.base, a->arena.size, off, in, out, xt, T), "batched Q8_0 projection");
        float *ctl = download(ct, (uint64_t)T * out);
        cd = bit_diffs(ctl, one, (uint64_t)T * out, NULL);
        g_t4_q8_control_diff += cd;
        free(ctl);
        ds4_gpu_tensor_free(ct);
    }
    printf("  q8_0   %-6s %4llu -> %-4llu T=%-3u exact rows vs %u one-row calls%s: %llu words differ; one-row vs double %.2e; "
           "control (batched ds4_gpu_matmul_q8_0_tensor): %llu words differ\n", name, (unsigned long long)in,
           (unsigned long long)out, T, T, g_batched ? " [batched]" : "", (unsigned long long)d, e, (unsigned long long)cd);
    free(x); free(rows); free(one);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(ot);
}

/* The router's logits as the driver computes them: ds4_gpu_qwen4_multi_gemv_tensor, one F32 output of 64 rows.  n_tok
 * rows in, n_tok rows out. */
static void t4_router_gemv(const t4_arena_t *a, const float *x, uint32_t n_tok, float *dst) {
    ds4_gpu_tensor *xt = upload(x, (uint64_t)n_tok * T4_E), *lt = zeroed((uint64_t)n_tok * MREF_NEXP * 4u);
    ds4_gpu_tensor *outs[1] = { lt };
    const uint64_t offs[1] = { a->router };
    const uint32_t types[1] = { 0u }, rows[1] = { MREF_NEXP };
    T4_BEGIN();
    require_ok(ds4_gpu_qwen4_multi_gemv_tensor(xt, n_tok, T4_E, 1, outs, a->arena.base, a->arena.size, offs, types, rows), "router gemv");
    T4_END();
    require_ok(ds4_gpu_tensor_read(lt, 0, dst, (uint64_t)n_tok * MREF_NEXP * 4u), "router download");
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(lt);
}

/* the batched F32 matmul: ds4_gpu_matmul_f32_tensor, which picks its kernel by row count */
static void t4_router_matmul_f32(const t4_arena_t *a, const float *x, uint32_t n_tok, float *dst) {
    ds4_gpu_tensor *xt = upload(x, (uint64_t)n_tok * T4_E), *lt = zeroed((uint64_t)n_tok * MREF_NEXP * 4u);
    require_ok(ds4_gpu_matmul_f32_tensor(lt, a->arena.base, a->arena.size, a->router, T4_E, MREF_NEXP, xt, n_tok), "F32 matmul");
    require_ok(ds4_gpu_tensor_read(lt, 0, dst, (uint64_t)n_tok * MREF_NEXP * 4u), "router download");
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(lt);
}

static void test_router_rows(const t4_arena_t *a, uint32_t T) {
    float *x = rand_vec((uint64_t)T * T4_E, 1.7f);
    float *rows = malloc((uint64_t)T * MREF_NEXP * 4u), *one = malloc((uint64_t)T * MREF_NEXP * 4u);
    require_ok(rows && one, "malloc");
    if (t4_break("router")) t4_router_matmul_f32(a, x, T, rows);
    else t4_router_gemv(a, x, T, rows);
    const bool was_batched = g_batched;
    g_batched = false; /* the one-row calls are the decode reference: each runs to completion */
    for (uint32_t t = 0; t < T; t++) t4_router_gemv(a, x + (uint64_t)t * T4_E, 1, one + (uint64_t)t * MREF_NEXP);
    g_batched = was_batched;
    uint64_t first = 0;
    const uint64_t d = bit_diffs(rows, one, (uint64_t)T * MREF_NEXP, &first);
    CHECK(d == 0, "F32 router (multi-output GEMV): %u rows differ from %u one-row calls in %llu words, first at row %llu%s", T, T,
          (unsigned long long)d, (unsigned long long)(first / MREF_NEXP), g_batched ? " [batched]" : "");
    /* sanity against the double dot of the same bytes */
    double ref[MREF_NEXP], xd[T4_E];
    const float *rw = (const float *)(a->arena.base + a->router);
    for (uint32_t i = 0; i < T4_E; i++) xd[i] = x[i];
    for (uint32_t e = 0; e < MREF_NEXP; e++) {
        double s = 0.0;
        for (uint32_t i = 0; i < T4_E; i++) s += (double)rw[(uint64_t)e * T4_E + i] * xd[i];
        ref[e] = s;
    }
    const double e0 = max_rel(one, ref, MREF_NEXP);
    CHECK(all_finite(rows, (uint64_t)T * MREF_NEXP) && e0 <= 1e-5, "router row 0 is %.3e from the double reference", e0);
    /* the control: the batched ds4_gpu_matmul_f32_tensor, over the same rows, against the same one-row bits */
    uint64_t cd = 0;
    {
        float *ctl = malloc((uint64_t)T * MREF_NEXP * 4u);
        require_ok(ctl != NULL, "malloc");
        t4_router_matmul_f32(a, x, T, ctl);
        cd = bit_diffs(ctl, one, (uint64_t)T * MREF_NEXP, NULL);
        g_t4_f32_control_diff += cd;
        free(ctl);
    }
    printf("  router T=%-3u multi-output GEMV rows vs %u one-row calls%s: %llu words differ; row 0 vs double %.2e; "
           "control (batched ds4_gpu_matmul_f32_tensor vs the one-row bits): %llu words differ\n", T, T, g_batched ? " [batched]" : "",
           (unsigned long long)d, e0, (unsigned long long)cd);
    free(x); free(rows); free(one);
}

/* RMSNorm with a weight: ds4_gpu_rms_norm_weight_rows_tensor over T rows against ds4_gpu_rms_norm_weight_tensor per row */
static void test_rms_norm_rows(const t4_arena_t *a, uint32_t T) {
    const float eps = 1e-6f;
    float *x = rand_vec((uint64_t)T * T4_E, 1.7f);
    ds4_gpu_tensor *xt = upload(x, (uint64_t)T * T4_E), *ot = zeroed((uint64_t)T * T4_E * 4u);
    T4_BEGIN();
    require_ok(ds4_gpu_rms_norm_weight_rows_tensor(ot, xt, a->arena.base, a->arena.size, a->norm, T4_E, T, t4_break("rms") ? 2.0f * eps : eps),
               "rms norm rows");
    T4_END();
    float *rows = download(ot, (uint64_t)T * T4_E), *one = malloc((uint64_t)T * T4_E * 4u);
    require_ok(one != NULL, "malloc");
    ds4_gpu_tensor *x1 = zeroed(T4_E * 4u), *o1 = zeroed(T4_E * 4u);
    for (uint32_t t = 0; t < T; t++) {
        require_ok(ds4_gpu_tensor_write(x1, 0, x + (uint64_t)t * T4_E, T4_E * 4u), "x row upload");
        require_ok(ds4_gpu_rms_norm_weight_tensor(o1, x1, a->arena.base, a->arena.size, a->norm, T4_E, eps), "rms norm one row");
        require_ok(ds4_gpu_tensor_read(o1, 0, one + (uint64_t)t * T4_E, T4_E * 4u), "row download");
    }
    uint64_t first = 0;
    const uint64_t d = bit_diffs(rows, one, (uint64_t)T * T4_E, &first);
    CHECK(d == 0, "RMSNorm: %u rows differ from %u one-row calls in %llu words, first at row %llu%s", T, T, (unsigned long long)d,
          (unsigned long long)(first / T4_E), g_batched ? " [batched]" : "");
    double ss = 0.0, ref[T4_E];
    for (uint32_t i = 0; i < T4_E; i++) ss += (double)x[i] * x[i];
    const double inv = 1.0 / sqrt(ss / T4_E + (double)eps);
    for (uint32_t i = 0; i < T4_E; i++) ref[i] = (double)x[i] * inv * a->norm_shadow[i];
    const double e0 = max_rel(rows, ref, T4_E);
    CHECK(all_finite(rows, (uint64_t)T * T4_E) && e0 <= 1e-5, "RMSNorm row 0 is %.3e from the double reference", e0);
    printf("  norm   T=%-3u weighted RMSNorm rows vs %u one-row calls%s: %llu words differ; row 0 vs double %.2e\n", T, T,
           g_batched ? " [batched]" : "", (unsigned long long)d, e0);
    free(x); free(rows); free(one);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(ot); ds4_gpu_tensor_free(x1); ds4_gpu_tensor_free(o1);
}

/* the residual add and the f16 staging copy: elementwise, so independent of rows by construction; tested like the rest */
static void test_elementwise_rows(uint32_t T) {
    float *x = rand_vec((uint64_t)T * T4_E, 3.0f), *y = rand_vec((uint64_t)T * T4_E, 3.0f);
    ds4_gpu_tensor *xt = upload(x, (uint64_t)T * T4_E), *yt = upload(y, (uint64_t)T * T4_E), *ot = zeroed((uint64_t)T * T4_E * 4u);
    ds4_gpu_tensor *h = zeroed((uint64_t)T * T4_E * 2u);
    T4_BEGIN();
    require_ok(ds4_gpu_add_tensor(ot, xt, t4_break("eltwise") ? xt : yt, T * T4_E), "residual add");
    require_ok(ds4_gpu_tensor_copy_f32_to_f16(h, 0, xt, 0, (uint64_t)T * T4_E), "f16 staging");
    T4_END();
    float *rows = download(ot, (uint64_t)T * T4_E);
    uint16_t *hr = malloc((uint64_t)T * T4_E * 2u), *h1 = malloc((uint64_t)T * T4_E * 2u);
    require_ok(hr && h1 && ds4_gpu_tensor_read(h, 0, hr, (uint64_t)T * T4_E * 2u), "f16 download");
    float *one = malloc((uint64_t)T * T4_E * 4u);
    require_ok(one != NULL, "malloc");
    ds4_gpu_tensor *x1 = zeroed(T4_E * 4u), *y1 = zeroed(T4_E * 4u), *o1 = zeroed(T4_E * 4u), *hh = zeroed(T4_E * 2u);
    for (uint32_t t = 0; t < T; t++) {
        require_ok(ds4_gpu_tensor_write(x1, 0, x + (uint64_t)t * T4_E, T4_E * 4u) && ds4_gpu_tensor_write(y1, 0, y + (uint64_t)t * T4_E, T4_E * 4u), "row upload");
        require_ok(ds4_gpu_add_tensor(o1, x1, y1, T4_E), "one-row add");
        require_ok(ds4_gpu_tensor_read(o1, 0, one + (uint64_t)t * T4_E, T4_E * 4u), "row download");
        require_ok(ds4_gpu_tensor_copy_f32_to_f16(hh, 0, x1, 0, T4_E) && ds4_gpu_tensor_read(hh, 0, h1 + (uint64_t)t * T4_E, T4_E * 2u), "one-row f16");
    }
    const uint64_t d = bit_diffs(rows, one, (uint64_t)T * T4_E, NULL);
    uint64_t dh = 0;
    for (uint64_t i = 0; i < (uint64_t)T * T4_E; i++) dh += hr[i] != h1[i];
    CHECK(d == 0 && dh == 0, "residual add / f16 staging: %u rows differ from one-row calls in %llu and %llu words%s", T,
          (unsigned long long)d, (unsigned long long)dh, g_batched ? " [batched]" : "");
    printf("  eltwise T=%-3u residual add: %llu words differ; f32 -> f16 staging: %llu words differ%s\n", T, (unsigned long long)d,
           (unsigned long long)dh, g_batched ? " [batched]" : "");
    free(x); free(y); free(rows); free(one); free(hr); free(h1);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(yt); ds4_gpu_tensor_free(ot); ds4_gpu_tensor_free(h);
    ds4_gpu_tensor_free(x1); ds4_gpu_tensor_free(y1); ds4_gpu_tensor_free(o1); ds4_gpu_tensor_free(hh);
}

/* q/k norm + RoPE over T rows from pos0 against T one-row calls at pos0 + r, both layer kinds */
static void rope_run(fixture_t *f, const float *xq, const float *xk, uint32_t T, uint32_t pos0, int full, float *q_out, float *k_out) {
    const ds4_gpu_mellum_rope p = model_rope(1.0f);
    const uint64_t nq = (uint64_t)T * MELLUM_N_HEAD * MELLUM_HEAD_DIM, nk = (uint64_t)T * MELLUM_N_HEAD_KV * MELLUM_HEAD_DIM;
    ds4_gpu_tensor *tq = upload(xq, nq), *tk = upload(xk, nk);
    T4_BEGIN();
    require_ok(ds4_gpu_mellum_qk_norm_rope_tensor(tq, tk, f->arena.base, f->arena.size, f->q_off, f->k_off, T, pos0 + (T > 1u && t4_break("rope") ? 1u : 0u), kind_of(full), &p),
               "mellum q/k norm and rope");
    T4_END();
    require_ok(ds4_gpu_tensor_read(tq, 0, q_out, nq * 4u) && ds4_gpu_tensor_read(tk, 0, k_out, nk * 4u), "rope download");
    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tk);
}

static void test_rope_rows(fixture_t *f, uint32_t T, uint32_t pos0, int full) {
    const uint64_t qw = (uint64_t)MELLUM_N_HEAD * MELLUM_HEAD_DIM, kw = (uint64_t)MELLUM_N_HEAD_KV * MELLUM_HEAD_DIM;
    float *xq = rand_vec(T * qw, 1.7f), *xk = rand_vec(T * kw, 1.7f);
    float *rq = malloc(T * qw * 4u), *rk = malloc(T * kw * 4u), *sq = malloc(T * qw * 4u), *sk = malloc(T * kw * 4u);
    require_ok(rq && rk && sq && sk, "malloc");
    rope_run(f, xq, xk, T, pos0, full, rq, rk);
    const bool was = g_batched;
    g_batched = false; /* the one-row calls are the decode reference: each runs to completion */
    for (uint32_t t = 0; t < T; t++) rope_run(f, xq + t * qw, xk + t * kw, 1, pos0 + t, full, sq + t * qw, sk + t * kw);
    g_batched = was;
    uint64_t fq = 0, fk = 0;
    const uint64_t dq = bit_diffs(rq, sq, T * qw, &fq), dk = bit_diffs(rk, sk, T * kw, &fk);
    CHECK(dq == 0 && dk == 0, "q/k norm+RoPE (%s layer, pos0 %u): %u rows differ from one-row calls in %llu (q) and %llu (k) words%s",
          full ? "full" : "sliding", pos0, T, (unsigned long long)dq, (unsigned long long)dk, g_batched ? " [batched]" : "");
    CHECK(all_finite(rq, T * qw) && all_finite(rk, T * kw), "q/k norm+RoPE produced a non-finite value");
    printf("  rope   %-7s T=%-3u pos0=%-5u rows vs one-row calls%s: %llu words differ in q, %llu in k\n", full ? "full" : "sliding", T,
           pos0, g_batched ? " [batched]" : "", (unsigned long long)dq, (unsigned long long)dk);
    free(xq); free(xk); free(rq); free(rk); free(sq); free(sk);
}

static void test_rope_rows_all(fixture_t *f) {
    printf("Mellum prefill equals decode (rows against one-row calls, raw bits)\n");
    for (int batched = 0; batched <= 1; batched++) {
        g_batched = batched != 0;
        for (int full = 0; full <= 1; full++) {
            test_rope_rows(f, 7, 0, full);
            test_rope_rows(f, 64, 1021, full);   /* straddles the window edge */
            test_rope_rows(f, 33, 8190, full);   /* straddles YaRN's original context */
        }
    }
    g_batched = false;
}

static void test_prefill_equals_decode(void) {
    t4_arena_t *a = calloc(1, sizeof(*a));
    require_ok(a != NULL, "calloc");
    a->arena.size = (uint64_t)48 << 20;
    a->arena.base = mmap(NULL, a->arena.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (a->arena.base == MAP_FAILED) { perror("mmap"); exit(1); }
    memset(a->arena.base, 0xFF, a->arena.size);
    a->q = t4_take(a, t4_q8_bytes(T4_E, 4096u));
    a->k = t4_take(a, t4_q8_bytes(T4_E, 512u));
    a->v = t4_take(a, t4_q8_bytes(T4_E, 512u));
    a->o = t4_take(a, t4_q8_bytes(4096u, T4_E));
    a->router = t4_take(a, MREF_ROUTER_BYTES);
    a->norm = t4_take(a, T4_E * 4u);
    fill_q8(a->arena.base + a->q, (uint64_t)4096u * (T4_E / 32u));
    fill_q8(a->arena.base + a->k, (uint64_t)512u * (T4_E / 32u));
    fill_q8(a->arena.base + a->v, (uint64_t)512u * (T4_E / 32u));
    fill_q8(a->arena.base + a->o, (uint64_t)T4_E * (4096u / 32u));
    fill_router_random((float *)(a->arena.base + a->router), 0.072f);
    a->norm_shadow = malloc(T4_E * sizeof(double));
    require_ok(a->norm_shadow != NULL, "malloc");
    {
        float *w = (float *)(a->arena.base + a->norm);
        for (uint32_t i = 0; i < T4_E; i++) { w[i] = 0.5f + 0.5f * (frand() + 1.0f); a->norm_shadow[i] = w[i]; }
    }
    require_ok(ds4_gpu_set_model_map(a->arena.base, a->arena.size), "model map registration");
    static const uint32_t counts[] = { 7, 33, 64, 130 };
    for (int batched = 0; batched <= 1; batched++) {
        g_batched = batched != 0;
        for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
            const uint32_t T = counts[c];
            test_q8_projection(a, "Q", a->q, T4_E, 4096u, T);
            test_q8_projection(a, "K", a->k, T4_E, 512u, T);
            test_q8_projection(a, "V", a->v, T4_E, 512u, T);
            test_q8_projection(a, "O", a->o, 4096u, T4_E, T);
            test_router_rows(a, T);
            test_rms_norm_rows(a, T);
            test_elementwise_rows(T);
        }
    }
    g_batched = false;
    CHECK(g_t4_q8_control_diff > 0, "control: the batched ds4_gpu_matmul_q8_0_tensor must differ from one-row calls somewhere (it matched everywhere)");
    CHECK(g_t4_f32_control_diff > 0, "control: the batched ds4_gpu_matmul_f32_tensor must differ from one-row calls somewhere (it matched everywhere)");
    printf("  controls: batched Q8_0 projection differs from one-row calls in %llu words, batched F32 router in %llu\n",
           (unsigned long long)g_t4_q8_control_diff, (unsigned long long)g_t4_f32_control_diff);
    munmap(a->arena.base, a->arena.size);
    free(a->norm_shadow);
    free(a);
}


/* S4 streamed decode: one command buffer per token that stops at its first missed layer.  A chain of
 * 28 adds stands in for the layers; a trip after add l leaves l + 1 adds applied. */
#define GATE_N 1024u
#define GATE_LAYERS 28u

static ds4_gpu_tensor *gate_filled(float v) {
    float *h = malloc(GATE_N * sizeof(float));
    for (uint32_t i = 0; i < GATE_N; i++) h[i] = v;
    ds4_gpu_tensor *t = upload(h, GATE_N);
    free(h);
    return t;
}

static bool gate_all(const ds4_gpu_tensor *t, float v) {
    float *h = download(t, GATE_N);
    bool ok = true;
    for (uint32_t i = 0; i < GATE_N; i++) ok = ok && h[i] == v;
    free(h);
    return ok;
}

static void gate_chain(ds4_gpu_tensor *acc, const ds4_gpu_tensor *one, ds4_gpu_tensor *gate,
                       const ds4_gpu_tensor *probe, int trip_layer) {
    require_ok(ds4_gpu_begin_commands(), "gate chain begin");
    for (uint32_t l = 0; l < GATE_LAYERS; l++) {
        require_ok(ds4_gpu_add_tensor(acc, acc, one, GATE_N), "gate chain add");
        if ((int)l == trip_layer) require_ok(ds4_gpu_stream_gate_trip_if_zero_tensor(gate, probe, l), "gate trip");
    }
    require_ok(ds4_gpu_end_commands(), "gate chain end");
}

static void check_gate_suppresses_after_trip(void) {
    ds4_gpu_tensor *acc = gate_filled(0.0f), *one = gate_filled(1.0f), *zero = gate_filled(0.0f);
    ds4_gpu_tensor *gate = ds4_gpu_stream_gate_create();
    require_ok(gate != NULL && ds4_gpu_stream_gate_reset(gate), "gate create");
    require_ok(ds4_gpu_stream_gate_bind(gate), "gate bind");
    gate_chain(acc, one, gate, zero, 9);
    require_ok(ds4_gpu_stream_gate_bind(NULL), "gate unbind");
    uint32_t layer = 0;
    bool stopped = false;
    require_ok(ds4_gpu_stream_gate_read(gate, &layer, &stopped), "gate read");
    CHECK(stopped && layer == 9, "gate: a zero probe at layer 9 must stop at layer 9 (stopped %d, layer %u)", stopped, layer);
    CHECK(gate_all(acc, 10.0f), "gate: layers after the trip must not run (want 10 adds everywhere)");
    /* reset clears the stop: the same gate then runs the whole chain */
    require_ok(ds4_gpu_stream_gate_reset(gate) && ds4_gpu_stream_gate_bind(gate), "gate reset");
    gate_chain(acc, one, gate, zero, -1);
    require_ok(ds4_gpu_stream_gate_bind(NULL), "gate unbind");
    CHECK(gate_all(acc, 38.0f), "gate: a reset gate must run all 28 layers (want 10 + 28)");
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(acc); ds4_gpu_tensor_free(one); ds4_gpu_tensor_free(zero);
}

static void check_gate_all_hit_one_wait(void) {
    ds4_gpu_tensor *acc = gate_filled(0.0f), *one = gate_filled(1.0f);
    ds4_gpu_tensor *gate = ds4_gpu_stream_gate_create();
    require_ok(gate != NULL && ds4_gpu_stream_gate_reset(gate) && ds4_gpu_stream_gate_bind(gate), "gate create");
    const uint64_t w0 = ds4_gpu_wait_count();
    gate_chain(acc, one, gate, one, 9);  /* a nonzero probe never trips */
    const uint64_t waits = ds4_gpu_wait_count() - w0;
    require_ok(ds4_gpu_stream_gate_bind(NULL), "gate unbind");
    uint32_t layer = 0;
    bool stopped = true;
    require_ok(ds4_gpu_stream_gate_read(gate, &layer, &stopped), "gate read");
    CHECK(!stopped, "gate: a nonzero probe must not stop");
    CHECK(gate_all(acc, 28.0f), "gate: an all-hit chain must run all 28 layers");
    CHECK(waits == 1, "gate: an all-hit chain must wait once on the host (waited %llu)", (unsigned long long)waits);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(acc); ds4_gpu_tensor_free(one);
}

/* past its record cap the gate can no longer suppress, so a token that overflows it must read back as an
 * error rather than as a clean all-hit or a stop */
static void check_gate_overflow_is_error(void) {
    ds4_gpu_tensor *acc = gate_filled(0.0f), *one = gate_filled(1.0f);
    ds4_gpu_tensor *gate = ds4_gpu_stream_gate_create();
    require_ok(gate != NULL && ds4_gpu_stream_gate_reset(gate) && ds4_gpu_stream_gate_bind(gate), "gate create");
    require_ok(ds4_gpu_begin_commands(), "overflow begin");
    for (uint32_t i = 0; i < 4100u; i++) require_ok(ds4_gpu_add_tensor(acc, acc, one, GATE_N), "overflow add");
    require_ok(ds4_gpu_end_commands(), "overflow end");
    require_ok(ds4_gpu_stream_gate_bind(NULL), "gate unbind");
    uint32_t layer = 0;
    bool stopped = false;
    CHECK(!ds4_gpu_stream_gate_read(gate, &layer, &stopped), "gate: more dispatches than records must read back as an error");
    /* a reset clears it: the next token is judged on its own */
    require_ok(ds4_gpu_stream_gate_reset(gate) && ds4_gpu_stream_gate_bind(gate), "gate reset");
    gate_chain(acc, one, gate, one, -1);
    require_ok(ds4_gpu_stream_gate_bind(NULL), "gate unbind");
    CHECK(ds4_gpu_stream_gate_read(gate, &layer, &stopped) && !stopped, "gate: a reset gate must read cleanly again");
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(acc); ds4_gpu_tensor_free(one);
}

/* the Q8_0 projection chain and the f32 -> f16 copy are on the streamed decode path too: after a trip they
 * must write nothing; unbound they run as before */
static void check_gate_suppresses_projection_and_copy(void) {
    const uint64_t in = 256, out = 64, wbytes = out * (in / 32u) * 34u;
    uint8_t *arena = mmap(NULL, 1u << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    require_ok(arena != MAP_FAILED, "projection arena");
    fill_q8(arena, out * (in / 32u));
    require_ok(ds4_gpu_set_model_map(arena, 1u << 20), "projection arena map");
    float *xh = rand_vec(in, 1.0f), sent[64];
    for (uint32_t i = 0; i < out; i++) sent[i] = 12345.0f;
    uint16_t sent16[256];
    for (uint32_t i = 0; i < in; i++) sent16[i] = 0xABCDu;
    for (int bound = 1; bound >= 0; bound--) {
        for (uint32_t rows = 1; rows <= 8; rows += 7) {
            ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(rows * in * 4u), *o = ds4_gpu_tensor_alloc(rows * out * 4u);
            ds4_gpu_tensor *d16 = ds4_gpu_tensor_alloc(in * 2u), *zero = gate_filled(0.0f);
            ds4_gpu_tensor *gate = ds4_gpu_stream_gate_create();
            require_ok(x && o && d16 && gate && ds4_gpu_stream_gate_reset(gate), "projection tensors");
            for (uint32_t r = 0; r < rows; r++) require_ok(ds4_gpu_tensor_write(x, r * in * 4u, xh, in * 4u), "x");
            for (uint32_t r = 0; r < rows; r++) require_ok(ds4_gpu_tensor_write(o, r * out * 4u, sent, out * 4u), "o");
            require_ok(ds4_gpu_tensor_write(d16, 0, sent16, in * 2u), "d16");
            require_ok(ds4_gpu_stream_gate_bind(bound ? gate : NULL) && ds4_gpu_begin_commands(), "begin");
            require_ok(ds4_gpu_stream_gate_trip_if_zero_tensor(gate, zero, 0), "trip");
            require_ok(ds4_gpu_matmul_q8_0_tensor(o, arena, 1u << 20, 0, in, out, x, rows), "projection");
            require_ok(ds4_gpu_tensor_copy_f32_to_f16(d16, 0, x, 0, in), "copy");
            require_ok(ds4_gpu_end_commands() && ds4_gpu_stream_gate_bind(NULL), "end");
            float *oh = download(o, rows * out);
            uint16_t got16[256];
            require_ok(ds4_gpu_tensor_read(d16, 0, got16, in * 2u), "d16 read");
            bool o_kept = true;
            for (uint32_t r = 0; r < rows; r++) o_kept = o_kept && memcmp(oh + r * out, sent, out * 4u) == 0;
            const bool d_kept = memcmp(got16, sent16, in * 2u) == 0;
            if (bound) {
                CHECK(o_kept, "gate: a Q8_0 projection of %u rows after a trip must write nothing", rows);
                CHECK(d_kept, "gate: an f32-to-f16 copy after a trip must write nothing");
            } else {
                CHECK(!o_kept && !d_kept, "gate: unbound, the projection (%u rows) and copy must run", rows);
            }
            free(oh);
            ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(o); ds4_gpu_tensor_free(d16); ds4_gpu_tensor_free(zero);
            ds4_gpu_tensor_free(gate);
        }
    }
    (void)wbytes;
    free(xh);
    munmap(arena, 1u << 20);
}

/* the per-layer check kernel: a selected expert with no address trips the gate and suppresses every later
 * dispatch (even with far more gate records than the layer has hidden units), and saves the layer's resume
 * state; with every address present it trips nothing */
static void check_stream_check_kernel(void) {
    enum { CE = 128 };
    for (int missing = 1; missing >= 0; missing--) {
        int32_t sel[8] = { 3, 9, 17, 22, 40, 41, 50, 63 };
        float wts[8] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
        uint64_t addr[64];
        for (uint32_t e = 0; e < 64; e++) addr[e] = 0x1000u + e;
        if (missing) addr[22] = 0;
        float *ar = rand_vec(CE, 1.0f), *fn = rand_vec(CE, 1.0f);
        ds4_gpu_tensor *st = ds4_gpu_tensor_alloc(sizeof(sel)), *wt = ds4_gpu_tensor_alloc(sizeof(wts));
        ds4_gpu_tensor *at = upload(ar, CE), *ft = upload(fn, CE), *save = zeroed(2u * (2u * CE + 16u) * 4u);
        ds4_gpu_tensor *ga = ds4_gpu_tensor_alloc(sizeof(addr)), *ua = ds4_gpu_tensor_alloc(sizeof(addr)), *da = ds4_gpu_tensor_alloc(sizeof(addr));
        ds4_gpu_tensor *acc = gate_filled(0.0f), *one = gate_filled(1.0f), *gate = ds4_gpu_stream_gate_create();
        require_ok(st && wt && ga && ua && da && gate && ds4_gpu_tensor_write(st, 0, sel, sizeof(sel)) &&
                   ds4_gpu_tensor_write(wt, 0, wts, sizeof(wts)) && ds4_gpu_tensor_write(ga, 0, addr, sizeof(addr)) &&
                   ds4_gpu_tensor_write(ua, 0, addr, sizeof(addr)) && ds4_gpu_tensor_write(da, 0, addr, sizeof(addr)) &&
                   ds4_gpu_stream_gate_reset(gate) && ds4_gpu_stream_gate_bind(gate) && ds4_gpu_begin_commands(), "check setup");
        require_ok(ds4_gpu_mellum_stream_check_tensor(gate, st, wt, at, ft, save, ga, ua, da, 1, CE), "check");
        /* more dispatches after the trip than any grid sized to the layer: records past it must be zeroed too */
        for (uint32_t i = 0; i < 600; i++) require_ok(ds4_gpu_add_tensor(acc, acc, one, GATE_N), "check chain add");
        require_ok(ds4_gpu_end_commands() && ds4_gpu_stream_gate_bind(NULL), "check end");
        uint32_t layer = 0;
        bool stopped = false;
        require_ok(ds4_gpu_stream_gate_read(gate, &layer, &stopped), "check read");
        float *sv = download(save, 2u * (2u * CE + 16u));
        const float *slot = sv + (2u * CE + 16u);
        bool saved = memcmp(slot, ar, CE * 4u) == 0 && memcmp(slot + CE, fn, CE * 4u) == 0 &&
                     memcmp(slot + 2u * CE, sel, sizeof(sel)) == 0 && memcmp(slot + 2u * CE + 8u, wts, sizeof(wts)) == 0;
        if (missing) {
            CHECK(stopped && layer == 1, "check: a missing address must stop at its layer (stopped %d, layer %u)", stopped, layer);
            CHECK(gate_all(acc, 0.0f), "check: every dispatch after a trip must do nothing, past the layer's width");
        } else {
            CHECK(!stopped && gate_all(acc, 600.0f), "check: with every address present nothing is suppressed");
        }
        CHECK(saved, "check: the layer's resume state must be saved in its slot (missing %d)", missing);
        free(ar); free(fn); free(sv);
        ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt); ds4_gpu_tensor_free(at); ds4_gpu_tensor_free(ft);
        ds4_gpu_tensor_free(save); ds4_gpu_tensor_free(ga); ds4_gpu_tensor_free(ua); ds4_gpu_tensor_free(da);
        ds4_gpu_tensor_free(acc); ds4_gpu_tensor_free(one); ds4_gpu_tensor_free(gate);
    }
}

static void check_unbound_gate_inert(void) {
    ds4_gpu_tensor *acc = gate_filled(0.0f), *one = gate_filled(1.0f), *zero = gate_filled(0.0f);
    ds4_gpu_tensor *gate = ds4_gpu_stream_gate_create();
    require_ok(gate != NULL && ds4_gpu_stream_gate_reset(gate) && ds4_gpu_stream_gate_bind(NULL), "gate create");
    gate_chain(acc, one, gate, zero, 9);
    CHECK(gate_all(acc, 28.0f), "gate: an unbound gate must not suppress anything, even when tripped");
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(acc); ds4_gpu_tensor_free(one); ds4_gpu_tensor_free(zero);
}

int main(void) {
    fixture_t f;
    memset(&f, 0, sizeof(f));
    f.arena.size = (uint64_t)1 << 20;
    f.arena.base = mmap(NULL, f.arena.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (f.arena.base == MAP_FAILED) { perror("mmap"); return 1; }
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(f.arena.base, f.arena.size), "model map registration");
    f.q_off = arena_f32(&f.arena, MELLUM_HEAD_DIM, &f.wq, 0.5f, 1.5f);
    f.k_off = arena_f32(&f.arena, MELLUM_HEAD_DIM, &f.wk, 0.5f, 1.5f);

    printf("Mellum q/k norm + RoPE (32 Q heads, 4 K heads, head 128, eps 1e-6, base 5e5)\n");
    /* rows at the window edge, then far past YaRN's context */
    static const uint32_t positions[] = { 0, 1, 1023, 1024, 1025, 8191, 50000 };
    for (int full = 0; full <= 1; full++) {
        for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); i++) {
            test_position(&f, 1, positions[i], full);
        }
        test_position(&f, 7, 1021, full);  /* row r sits at pos0 + r: 1021..1027 */
        test_position(&f, 3, 49999, full);
        /* inputs near eps: x ~ 1e-3 puts mean(x^2) at the same size as the norm's 1e-6, so eps is seen */
        test_position_scaled(&f, 2, 1025, full, 1e-3f);
    }
    test_kinds_differ(&f, 1025);
    test_bases(&f);
    test_double_scale_control(&f, 1, 1025);
    test_double_scale_control(&f, 1, 8191);
    test_double_scale_control(&f, 0, 1025);
    test_refusals(&f);
    test_rope_rows_all(&f);

    printf("Mellum attention (GQA 32/4, head 128, sliding window 1,024, f16 keys; max|d| / max|ref|)\n");
    static const uint32_t swa_counts[] = { 26, 1000, 1023, 1024, 1025, 1057, 2100 };
    for (size_t i = 0; i < sizeof(swa_counts) / sizeof(swa_counts[0]); i++) test_swa_session(swa_counts[i]);
    test_swa_tags();
    test_swa_refusals();
    test_swa_rewind_one();
    static const uint32_t full_counts[] = { 26, 1000, 1024, 1025 };
    for (size_t i = 0; i < sizeof(full_counts) / sizeof(full_counts[0]); i++) test_full_session(full_counts[i]);
    test_full_two_layers();
    test_qwen_sibling();
    test_full_refusals();

    test_moe();
    test_prefill_equals_decode();

    /* 2,100 and 4,100 keys are past 64 splits of 32 keys: the splits stop growing at their cap
     * (QWEN4_ATTN_MAX_SPLITS), so a key count past 2,048 runs more keys per split than any count below it */
    static const uint32_t long_full_counts[] = { 2100, 4100 };
    for (size_t i = 0; i < sizeof(long_full_counts) / sizeof(long_full_counts[0]); i++) test_full_session(long_full_counts[i]);

    printf("S4 streamed-decode gate (28-layer add chain)\n");
    check_gate_suppresses_after_trip();
    check_gate_all_hit_one_wait();
    check_unbound_gate_inert();
    check_stream_check_kernel();
    check_gate_suppresses_projection_and_copy();
    check_gate_overflow_is_error();

    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
