/* Model-free GPU kernel tests for Mellum 2.1: each Mellum kernel against a
 * double reference, with the control that proves the test can fail.
 * Build and run: make test-mellum-kernels (macOS; no model, no session).
 *
 * q/k norm + RoPE: per-head RMS norm and NeoX RoPE, plain on sliding layers
 * and YaRN on full layers.  The reference is llama.cpp's YaRN formulas
 * (rope_yarn_corr_dims and the ramp blend) in double. */

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

    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
