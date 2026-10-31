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
static inline void mellum_qk_norm_rope_head(constant ds4_metal_args_mellum_qk_norm_rope & args, device const float *x,
                                            device const float *w, device float *y, uint n_head, uint h, uint r,
                                            ushort tiisg, threadgroup float *row) {
    if (h >= n_head || r >= args.n_rows) return;
    const uint D = args.head_dim, npt = D / 32u;
    device const float *src = x + ((uint64_t)r * n_head + h) * D;
    device float *dst = y + ((uint64_t)r * n_head + h) * D;
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

kernel void kernel_mellum_qk_norm_rope(
        constant ds4_metal_args_mellum_qk_norm_rope & args,
        device const float *x,         /* [n_rows][n_head][head_dim] */
        device const float *w,         /* [head_dim] */
        device float       *y,         /* [n_rows][n_head][head_dim], may alias x */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float row[256];
    mellum_qk_norm_rope_head(args, x, w, y, args.n_head, tgpig.x, tgpig.y, tiisg, row);
}

/* Q's heads (args.n_head of them) and then K's (args.pad1) in one dispatch, each in place, each head as
 * kernel_mellum_qk_norm_rope computes it (ds4_gpu_mellum_qk_norm_rope_fused_tensor). */
kernel void kernel_mellum_qk_norm_rope2(
        constant ds4_metal_args_mellum_qk_norm_rope & args,
        device float       *q,         /* [n_rows][n_head][head_dim] */
        device const float *wq,
        device float       *k,         /* [n_rows][pad1][head_dim] */
        device const float *wk,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float row[256];
    if (tgpig.x < args.n_head) {
        mellum_qk_norm_rope_head(args, q, wq, q, args.n_head, tgpig.x, tgpig.y, tiisg, row);
    } else {
        mellum_qk_norm_rope_head(args, k, wk, k, args.pad1, tgpig.x - args.n_head, tgpig.y, tiisg, row);
    }
}

/* Two contiguous f32 -> f16 conversions of n values in one dispatch, each as kernel_cpy_contig_f32_f16_4 does it:
 * four values per thread, a ragged tail of 1-3 one at a time (ds4_gpu_mellum_copy2_f32_to_f16). */
kernel void kernel_mellum_cpy2_f32_f16(
        constant uint & n,
        device const float * src_a,
        device       half  * dst_a,
        device const float * src_b,
        device       half  * dst_b,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) return;
    const uint remaining = n - i;
    if (remaining >= 4u) {
        ((device packed_half4 *)dst_a)[gid] = packed_half4(half4(float4(((device const packed_float4 *)src_a)[gid])));
        ((device packed_half4 *)dst_b)[gid] = packed_half4(half4(float4(((device const packed_float4 *)src_b)[gid])));
        return;
    }
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_a[i + lane] = half(src_a[i + lane]);
        dst_b[i + lane] = half(src_b[i + lane]);
    }
}

/* Mellum's MoE reduce and the residual add after it, in one dispatch (ds4_gpu_mellum_moe_reduce_add_tensor):
 * kernel_qwen4_moe_reduce's weighted sum over the slots (no shared expert, no hyper-connections), in its order,
 * then out = res + sum, as kernel_add2_f32 adds the two. */
struct ds4_metal_args_mellum_moe_reduce_add {
    uint32_t n_tokens;
    uint32_t n_slots;
    uint32_t dim;
    uint32_t pad0;
};

kernel void kernel_mellum_moe_reduce_add(
        constant ds4_metal_args_mellum_moe_reduce_add & args,
        device const float *part,        /* [T][n_slots][dim] */
        device const float *weights,     /* [T][n_slots] */
        device const float *res,         /* [T][dim] */
        device float       *out,         /* [T][dim] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint d = tgpig.x * ntg.x + tid;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens || d >= args.dim) return;
    float acc = 0.0f;
    for (uint s = 0; s < args.n_slots; s++) {
        acc += weights[(uint64_t)tok * args.n_slots + s] *
               part[((uint64_t)tok * args.n_slots + s) * args.dim + d];
    }
    out[(uint64_t)tok * args.dim + d] = res[(uint64_t)tok * args.dim + d] + acc;
}

/* --- the sliding layers' K/V ring ------------------------------------------ */

struct ds4_metal_args_mellum_ring_commit {
    uint32_t n_rows;
    uint32_t pos0;         /* position of row 0; row r goes to slot (pos0 + r) % cache_cap */
    uint32_t cache_cap;
    uint32_t width;        /* n_head_kv * head_dim */
};

/* Commit a chunk to a sliding layer's ring: K and V rows as f16 at slot
 * position % cache_cap, plus the slot's position tag, which the attention
 * reads to refuse a stale slot.  (antirez's kernel_dflash_commit_kv_f16 with
 * the tag, as the ds4-engine fork's kernel_dflash_ring_write had it.) */
kernel void kernel_mellum_ring_commit(
        constant ds4_metal_args_mellum_ring_commit &args,
        device const float *k,
        device const float *v,
        device        half *key_cache,
        device        half *value_cache,
        device     int32_t *tags,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t count = (uint64_t)args.n_rows * args.width;
    if ((uint64_t)gid >= count || args.cache_cap == 0u) return;

    const uint row = gid / args.width;
    const uint col = gid - row * args.width;
    const uint cache_row = (args.pos0 + row) % args.cache_cap;
    const uint64_t src = (uint64_t)row * args.width + col;
    const uint64_t dst = (uint64_t)cache_row * args.width + col;
    key_cache[dst] = (half)k[src];
    value_cache[dst] = (half)v[src];
    if (col == 0u) tags[cache_row] = (int32_t)(args.pos0 + row);
}

/* --- sliding-window attention ---------------------------------------------- */

struct ds4_metal_args_mellum_attn_swa {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t pos0;      /* position of row 0; row r is the query at pos0 + r */
    uint32_t window;    /* keys per query, and the ring's slots */
    float    scale;
    uint32_t pad0;
    uint32_t pad1;
};

#define MELLUM_SWA_NSG 8u   /* simdgroups per (head, row): keys split by their index in the logical sequence */
#define MELLUM_SWA_PF 2u    /* keys each simdgroup loads before folding them in (kernel_mellum_attn_swa) */
#define MELLUM_SWA_D 128u   /* head_dim; a lane owns four dims */

/* Sliding-window attention of one (head, row), prefill and decode alike.
 *  1. K and V are f16: the chunk's own rows come from the staging copy
 *     stage_k/stage_v (the f32 rows rounded to f16), the older ones from the
 *     ring, so a key has the same value in prefill and in decode.
 *  2. A query at qpos visits positions qpos - n + 1 .. qpos, n = min(qpos + 1,
 *     window), in that order.  A position below pos0 is ring slot pos % window
 *     and must carry its position as tag, else the row is NaN; any other is
 *     staged row pos - pos0.  Simdgroup s takes the keys whose index in that
 *     sequence is s, s + 8, ...; one simdgroup merges the eight partials in
 *     order.  The arithmetic depends on qpos and the keys, never on pos0 or
 *     the number of rows, which is what makes a prefill chunk equal its decode
 *     steps bit for bit.
 *  3. The ring is only read here; the host commits the chunk to it afterwards
 *     (kernel_mellum_ring_commit). */
kernel void kernel_mellum_attn_swa(
        constant ds4_metal_args_mellum_attn_swa & args,
        device const float   *q,          /* [T][H*128] */
        device const half    *ring_k,     /* [window][Hkv*128] */
        device const half    *ring_v,
        device const int32_t *ring_tag,   /* [window] */
        device const half    *stage_k,    /* [T][Hkv*128] */
        device const half    *stage_v,
        device float         *out,        /* [T][H*128] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float part_m[MELLUM_SWA_NSG], part_l[MELLUM_SWA_NSG];
    threadgroup float part_acc[MELLUM_SWA_NSG][MELLUM_SWA_D];
    threadgroup uint part_bad[MELLUM_SWA_NSG];
    const uint h = tgpig.x, row = tgpig.y;
    if (h >= args.n_head || row >= args.n_tokens) return;   /* uniform over the threadgroup */
    const uint kvh = h / (args.n_head / args.n_head_kv);
    const uint width = args.n_head_kv * MELLUM_SWA_D;
    const uint qpos = args.pos0 + row;
    const uint n = min(qpos + 1u, args.window);
    const uint first = qpos + 1u - n;
    const float4 qv = *reinterpret_cast<device const float4 *>(q + ((uint64_t)row * args.n_head + h) * MELLUM_SWA_D + tiisg * 4u) * args.scale;

    float4 acc = float4(0.0f);
    float m = -3.0e38f, l = 0.0f;
    uint bad = 0u;
    /* Each simdgroup loads its next MELLUM_SWA_PF keys before folding any of them in, so that many loads are in
     * flight; a slot past the end reloads the last key (the query's own, staged, never tag-checked) and is skipped.
     * The keys, the order they are folded in and every operation are kernel_mellum_attn_swa_ref's (phase 8c, S1),
     * and the kernel tests hold the two to the same words. */
    for (uint i0 = sgitg; i0 < n; i0 += MELLUM_SWA_PF * MELLUM_SWA_NSG) {
        float4 kf[MELLUM_SWA_PF], vf[MELLUM_SWA_PF];
        _Pragma("unroll") for (uint j = 0u; j < MELLUM_SWA_PF; j++) {
            const uint pos = first + min(i0 + j * MELLUM_SWA_NSG, n - 1u);
            if (pos < args.pos0) {
                const uint slot = pos % args.window;
                bad |= (uint)(ring_tag[slot] != (int32_t)pos);
                const uint64_t at = (uint64_t)slot * width + kvh * MELLUM_SWA_D + tiisg * 4u;
                kf[j] = float4(*reinterpret_cast<device const half4 *>(ring_k + at));
                vf[j] = float4(*reinterpret_cast<device const half4 *>(ring_v + at));
            } else {
                const uint64_t at = (uint64_t)(pos - args.pos0) * width + kvh * MELLUM_SWA_D + tiisg * 4u;
                kf[j] = float4(*reinterpret_cast<device const half4 *>(stage_k + at));
                vf[j] = float4(*reinterpret_cast<device const half4 *>(stage_v + at));
            }
        }
        _Pragma("unroll") for (uint j = 0u; j < MELLUM_SWA_PF; j++) {
            if (i0 + j * MELLUM_SWA_NSG >= n) continue;
            const float s = simd_sum(dot(qv, kf[j]));
            const float m_new = max(m, s);
            const float corr = exp(m - m_new);
            const float w = exp(s - m_new);
            l = l * corr + w;
            acc = acc * corr + w * vf[j];
            m = m_new;
        }
    }
    if (tiisg == 0u) {
        part_m[sgitg] = m;
        part_l[sgitg] = l;
        part_bad[sgitg] = bad;
    }
    *reinterpret_cast<threadgroup float4 *>(&part_acc[sgitg][tiisg * 4u]) = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0u) return;

    float gm = part_m[0];
    for (uint s = 1u; s < MELLUM_SWA_NSG; s++) gm = max(gm, part_m[s]);
    float ll = 0.0f;
    uint any_bad = 0u;
    float4 o = float4(0.0f);
    for (uint s = 0u; s < MELLUM_SWA_NSG; s++) {
        const float c = part_l[s] > 0.0f ? exp(part_m[s] - gm) : 0.0f;
        ll += part_l[s] * c;
        o += *reinterpret_cast<threadgroup float4 *>(&part_acc[s][tiisg * 4u]) * c;
        any_bad |= part_bad[s];
    }
    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    const float4 result = any_bad != 0u ? float4(as_type<float>(0x7fc00000u)) : o * inv;
    *reinterpret_cast<device float4 *>(out + ((uint64_t)row * args.n_head + h) * MELLUM_SWA_D + tiisg * 4u) = result;
}

/* The sliding kernel's reference arithmetic: kernel_mellum_attn_swa as it was before its loads were reordered
 * for speed (phase 8c, S1), kept so the kernel tests can hold the kernel to it word for word
 * (ds4_gpu_mellum_attn_swa_ref_tensor).  Nothing else dispatches it. */
kernel void kernel_mellum_attn_swa_ref(
        constant ds4_metal_args_mellum_attn_swa & args,
        device const float   *q,          /* [T][H*128] */
        device const half    *ring_k,     /* [window][Hkv*128] */
        device const half    *ring_v,
        device const int32_t *ring_tag,   /* [window] */
        device const half    *stage_k,    /* [T][Hkv*128] */
        device const half    *stage_v,
        device float         *out,        /* [T][H*128] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float part_m[MELLUM_SWA_NSG], part_l[MELLUM_SWA_NSG];
    threadgroup float part_acc[MELLUM_SWA_NSG][MELLUM_SWA_D];
    threadgroup uint part_bad[MELLUM_SWA_NSG];
    const uint h = tgpig.x, row = tgpig.y;
    if (h >= args.n_head || row >= args.n_tokens) return;   /* uniform over the threadgroup */
    const uint kvh = h / (args.n_head / args.n_head_kv);
    const uint width = args.n_head_kv * MELLUM_SWA_D;
    const uint qpos = args.pos0 + row;
    const uint n = min(qpos + 1u, args.window);
    const uint first = qpos + 1u - n;
    const float4 qv = *reinterpret_cast<device const float4 *>(q + ((uint64_t)row * args.n_head + h) * MELLUM_SWA_D + tiisg * 4u) * args.scale;

    float4 acc = float4(0.0f);
    float m = -3.0e38f, l = 0.0f;
    uint bad = 0u;
    for (uint i = sgitg; i < n; i += MELLUM_SWA_NSG) {
        const uint pos = first + i;
        float4 kf, vf;
        if (pos < args.pos0) {
            const uint slot = pos % args.window;
            bad |= (uint)(ring_tag[slot] != (int32_t)pos);
            const uint64_t at = (uint64_t)slot * width + kvh * MELLUM_SWA_D + tiisg * 4u;
            kf = float4(*reinterpret_cast<device const half4 *>(ring_k + at));
            vf = float4(*reinterpret_cast<device const half4 *>(ring_v + at));
        } else {
            const uint64_t at = (uint64_t)(pos - args.pos0) * width + kvh * MELLUM_SWA_D + tiisg * 4u;
            kf = float4(*reinterpret_cast<device const half4 *>(stage_k + at));
            vf = float4(*reinterpret_cast<device const half4 *>(stage_v + at));
        }
        const float s = simd_sum(dot(qv, kf));
        const float m_new = max(m, s);
        const float corr = exp(m - m_new);
        const float w = exp(s - m_new);
        l = l * corr + w;
        acc = acc * corr + w * vf;
        m = m_new;
    }
    if (tiisg == 0u) {
        part_m[sgitg] = m;
        part_l[sgitg] = l;
        part_bad[sgitg] = bad;
    }
    *reinterpret_cast<threadgroup float4 *>(&part_acc[sgitg][tiisg * 4u]) = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0u) return;

    float gm = part_m[0];
    for (uint s = 1u; s < MELLUM_SWA_NSG; s++) gm = max(gm, part_m[s]);
    float ll = 0.0f;
    uint any_bad = 0u;
    float4 o = float4(0.0f);
    for (uint s = 0u; s < MELLUM_SWA_NSG; s++) {
        const float c = part_l[s] > 0.0f ? exp(part_m[s] - gm) : 0.0f;
        ll += part_l[s] * c;
        o += *reinterpret_cast<threadgroup float4 *>(&part_acc[s][tiisg * 4u]) * c;
        any_bad |= part_bad[s];
    }
    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    const float4 result = any_bad != 0u ? float4(as_type<float>(0x7fc00000u)) : o * inv;
    *reinterpret_cast<device float4 *>(out + ((uint64_t)row * args.n_head + h) * MELLUM_SWA_D + tiisg * 4u) = result;
}

/* --- tiled prefill attention --------------------------------------------- */

struct ds4_metal_args_mellum_attn_tiled {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t pos0;      /* position of row 0; row r is the query at pos0 + r */
    uint32_t window;    /* keys per query (sliding), or 0: every position <= qpos (full) */
    uint32_t stage0;    /* position of staged row 0; a position below it is ring slot pos % window */
    float    scale;
    uint32_t pad0;
};

#define MELLUM_AT_TB 8u     /* tokens per threadgroup: simdgroup s takes token t0 + s */
#define MELLUM_AT_D 128u

#define MELLUM_AT_KT_HALF 32u   /* keys per K/V tile */

/* Prefill attention of a sliding or full layer on simdgroup matrices (ds4_gpu.h:
 * ds4_gpu_mellum_attn_tiled_half_tensor has the contract).  One threadgroup per (KV head, 8 consecutive tokens);
 * simdgroup s holds token t0 + s's eight query heads as one 8-row tile, so a tile has one query position and one
 * mask, and the eight simdgroups share each tile of 32 keys, staged once from the ring (positions below stage0,
 * tag-checked as in kernel_mellum_attn_swa: a key with the wrong tag makes the rows that read it NaN) or from the
 * staged rows / the full cache.  The scaled queries are rounded to half once, K and V staged as the f16 they are,
 * each tile's probabilities rounded to half for the value product; scores, the softmax and the output accumulate
 * in f32.  The softmax runs on the score fragments in registers: a lane of an 8x8 simdgroup matrix holds two
 * elements of one row, row (lane & 16) / 4 + (lane / 2) % 4, columns fn = (lane & 8) / 2 + (lane & 1) * 2 and
 * fn + 1, so the lanes of a row differ in bits 0 and 3, and a row's max and sum are two shuffles. */
kernel void kernel_mellum_attn_tiled_half(
        constant ds4_metal_args_mellum_attn_tiled & args,
        device const float   *q,
        device const half    *ring_k,
        device const half    *ring_v,
        device const int32_t *ring_tag,
        device const half    *stage_k,
        device const half    *stage_v,
        device float         *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint D = MELLUM_AT_D, KT = MELLUM_AT_KT_HALF, TB = MELLUM_AT_TB, NF = MELLUM_AT_KT_HALF / 8u;
    threadgroup half Ks[KT * D], Vs[KT * D];
    threadgroup int   kpos[KT];
    threadgroup uint  kbad[KT];
    const uint kvh = tgpig.x, t0 = tgpig.y * TB;
    if (kvh >= args.n_head_kv || t0 >= args.n_tokens) return;
    const uint H = args.n_head, group = H / args.n_head_kv, width = args.n_head_kv * D;
    const uint last_tok = min(t0 + TB, args.n_tokens) - 1u;
    const bool live = t0 + sgitg <= last_tok;
    const uint tok = min(t0 + sgitg, last_tok);
    const uint qpos = args.pos0 + tok;
    const uint qlo = args.pos0 + t0, qhi = args.pos0 + last_tok;
    const uint first = (args.window != 0u && qlo + 1u > args.window) ? qlo + 1u - args.window : 0u;
    const uint fn = (tiisg & 8u) / 2u + (tiisg & 1u) * 2u;

    simdgroup_half8x8 Qm[16];
    simdgroup_float8x8 O[16];
    device const float *qrow = q + ((uint64_t)tok * H + kvh * group) * D;
#pragma unroll
    for (uint j = 0; j < 16; j++) {
        simdgroup_float8x8 t;
        simdgroup_load(t, qrow + j * 8u, D, 0, false);
        Qm[j].thread_elements()[0] = (half)(t.thread_elements()[0] * args.scale);
        Qm[j].thread_elements()[1] = (half)(t.thread_elements()[1] * args.scale);
        O[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m_row = -3.0e38f, l_row = 0.0f;
    uint bad = 0u;

    for (uint k0 = first; k0 <= qhi; k0 += KT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < KT) {
            const uint p = k0 + tid;
            int kp = -1;
            uint kb = 0u;
            if (p <= qhi) {
                kp = (int)p;
                if (p < args.stage0) kb = (uint)(ring_tag[p % args.window] != (int32_t)p);
            }
            kpos[tid] = kp;
            kbad[tid] = kb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < KT * 16u; i += 256u) {   /* (key, segment of 8 dims) */
            const uint key = i >> 4, seg = i & 15u;
            const int p = kpos[key];
            uint4 kw = uint4(0u), vw = uint4(0u);
            if (p >= 0) {
                const bool ring = (uint)p < args.stage0;
                const uint64_t at = (ring ? (uint64_t)((uint)p % args.window) : (uint64_t)((uint)p - args.stage0)) * width +
                                    kvh * D + seg * 8u;
                kw = *reinterpret_cast<device const uint4 *>((ring ? ring_k : stage_k) + at);
                vw = *reinterpret_cast<device const uint4 *>((ring ? ring_v : stage_v) + at);
            }
            *reinterpret_cast<threadgroup uint4 *>(Ks + key * D + seg * 8u) = kw;
            *reinterpret_cast<threadgroup uint4 *>(Vs + key * D + seg * 8u) = vw;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[NF];
#pragma unroll
        for (uint f = 0; f < NF; f++) S[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma unroll
        for (uint j = 0; j < 16; j++) {
#pragma unroll
            for (uint f = 0; f < NF; f++) {
                simdgroup_half8x8 Kf;
                simdgroup_load(Kf, Ks + f * 8u * D + j * 8u, D, 0, true);
                simdgroup_multiply_accumulate(S[f], Qm[j], Kf, S[f]);
            }
        }
        float sv[2 * NF];
        bool valid[2 * NF];
        float mx = -3.0e38f;
#pragma unroll
        for (uint c = 0; c < 2 * NF; c++) {
            const uint key = (c >> 1) * 8u + fn + (c & 1u);
            const int p = kpos[key];
            valid[c] = p >= 0 && (uint)p <= qpos && (args.window == 0u || qpos - (uint)p < args.window);
            bad |= (uint)(valid[c] && kbad[key] != 0u);
            sv[c] = valid[c] ? S[c >> 1].thread_elements()[c & 1u] : -3.0e38f;
            mx = max(mx, sv[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        const float m_new = max(m_row, mx);
        const float corr = exp(m_row - m_new);
        float rs = 0.0f;
        simdgroup_half8x8 P[NF];
#pragma unroll
        for (uint c = 0; c < 2 * NF; c++) {
            const float pv = valid[c] ? exp(sv[c] - m_new) : 0.0f;
            rs += pv;
            P[c >> 1].thread_elements()[c & 1u] = (half)pv;
        }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 8);
        l_row = l_row * corr + rs;
        m_row = m_new;
        if (simd_any(corr != 1.0f)) {
#pragma unroll
            for (uint j = 0; j < 16; j++) {
                O[j].thread_elements()[0] *= corr;
                O[j].thread_elements()[1] *= corr;
            }
        }
#pragma unroll
        for (uint j = 0; j < 16; j++) {
#pragma unroll
            for (uint f = 0; f < NF; f++) {
                simdgroup_half8x8 Vf;
                simdgroup_load(Vf, Vs + f * 8u * D + j * 8u, D, 0, false);
                simdgroup_multiply_accumulate(O[j], P[f], Vf, O[j]);
            }
        }
    }

    const bool any_bad = simd_any(bad != 0u);
    if (!live) return;
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    device float *orow = out + ((uint64_t)tok * H + kvh * group) * D;
#pragma unroll
    for (uint j = 0; j < 16; j++) {
        O[j].thread_elements()[0] *= inv;
        O[j].thread_elements()[1] *= inv;
        simdgroup_store(O[j], orow + j * 8u, D, 0, false);
    }
    if (any_bad) {
        simdgroup_barrier(mem_flags::mem_device);
        for (uint i = tiisg; i < 8u * D; i += 32u) orow[i] = as_type<float>(0x7fc00000u);
    }
}
