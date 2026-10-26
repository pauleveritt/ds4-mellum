/* Model-free GPU kernel tests for Mellum 2.1: each Mellum kernel against a
 * double reference, with the control that proves the test can fail.
 * Build and run: make test-mellum-kernels (macOS; no model, no session).
 *
 * q/k norm + RoPE: per-head RMS norm and NeoX RoPE, plain on sliding layers
 * and YaRN on full layers.  The reference is llama.cpp's YaRN formulas
 * (rope_yarn_corr_dims and the ramp blend) in double.  Then attention: the
 * sliding window over a ring, and the ring commit. */

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

    printf("Mellum attention (GQA 32/4, head 128, sliding window 1,024, f16 keys; max|d| / max|ref|)\n");
    static const uint32_t swa_counts[] = { 26, 1000, 1023, 1024, 1025, 1057, 2100 };
    for (size_t i = 0; i < sizeof(swa_counts) / sizeof(swa_counts[0]); i++) test_swa_session(swa_counts[i]);
    test_swa_tags();
    test_swa_refusals();
    test_swa_rewind_one();

    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
