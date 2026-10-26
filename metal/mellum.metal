/* Mellum 2.1 kernels.  Appended after qwen4_vision.metal in the runtime source
 * list (ds4_metal.m, ds4_gpu_full_source). */

/* --- q/k head norm with rope ---------------------------------------------- */

struct ds4_metal_args_mellum_qk_norm_rope {
    uint32_t n_rows;
    uint32_t n_head;
    uint32_t head_dim;     /* multiple of 32, <= 256 */
    uint32_t rope_dims;    /* even, <= head_dim; pairs (i, i + rope_dims/2) */
    uint32_t pos0;         /* position of row 0; row r sits at pos0 + r */
    float    eps;
    float    mscale;       /* scales cos and sin: YaRN's magnitude on full layers, 1.0f on sliding ones */
    uint32_t pad1;
    float    inv_freq[128];/* the angle's frequency per pair, rounded once from double on the host:
                              base^(-2i/rope_dims), or the YaRN blend of it */
};

/* y = rope(x * rsqrt(mean(x^2) + eps) * w): llama.cpp's build_norm (RMS) on
 * each head followed by NeoX rope.  The angle is (float)pos * inv_freq[i],
 * as llama.cpp's Metal rope forms it; cos and sin are the precise ones, so
 * the only error past the float angle is its own rounding.  (Re-homed from the
 * ds4-engine fork's kernel_dflash_head_norm_rope, the arithmetic unchanged.) */
kernel void kernel_mellum_qk_norm_rope(
        constant ds4_metal_args_mellum_qk_norm_rope & args,
        device const float *x,         /* [n_rows][n_head][head_dim] */
        device const float *w,         /* [head_dim] */
        device float       *y,         /* [n_rows][n_head][head_dim], may alias x */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x, r = tgpig.y;
    if (h >= args.n_head || r >= args.n_rows) return;
    const uint D = args.head_dim, npt = D / 32u;
    device const float *src = x + ((uint64_t)r * args.n_head + h) * D;
    device float *dst = y + ((uint64_t)r * args.n_head + h) * D;
    threadgroup float row[256];
    float ss = 0.0f;
    for (uint k = 0; k < npt; k++) {
        const float v = src[tiisg + 32u * k];
        row[tiisg + 32u * k] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    const float scale = 1.0f / sqrt(ss / (float)D + args.eps);
    for (uint k = 0; k < npt; k++) {
        const uint i = tiisg + 32u * k;
        row[i] = row[i] * scale * w[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_dims = args.rope_dims / 2u;
    const float pos = (float)(args.pos0 + r);
    for (uint k = 0; k < npt; k++) {
        const uint i = tiisg + 32u * k;
        float out = row[i];
        if (i < args.rope_dims) {
            const uint pair = i < half_dims ? i : i - half_dims;
            const float theta = pos * args.inv_freq[pair];
            const float c = precise::cos(theta) * args.mscale, s = precise::sin(theta) * args.mscale;
            out = i < half_dims ? row[i] * c - row[i + half_dims] * s
                                : row[i - half_dims] * s + row[i] * c;
        }
        dst[i] = out;
    }
}
