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
#define MELLUM_AT_KT 16u    /* keys per K/V tile */
#define MELLUM_AT_D 128u

/* Prefill attention of a sliding or full layer on simdgroup matrices (ds4_gpu.h has the contract).  One
 * threadgroup per (KV head, 8 consecutive tokens); simdgroup s holds token t0 + s's eight query heads as one 8-row
 * tile, so a tile has one query position and one mask, and the eight simdgroups share each tile of 16 keys, staged
 * once from the ring (positions below stage0, tag-checked as in kernel_mellum_attn_swa: a key with the wrong tag makes
 * the rows that read it NaN) or from the staged rows / the full cache.  Every operand is f32 (K and V widened from f16
 * exactly), so only the summation order differs from the decode kernels.  The softmax runs on the score fragments in
 * registers: a lane of an 8x8 simdgroup matrix holds two elements of one row, row (lane & 16) / 4 + (lane / 2) % 4,
 * columns fn = (lane & 8) / 2 + (lane & 1) * 2 and fn + 1, so the lanes of a row differ in bits 0 and 3, and a row's
 * max and sum are two shuffles. */
kernel void kernel_mellum_attn_tiled(
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
    constexpr uint D = MELLUM_AT_D, KT = MELLUM_AT_KT, TB = MELLUM_AT_TB;
    threadgroup float Ks[KT * D], Vs[KT * D];
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
    const uint fn = (tiisg & 8u) / 2u + (tiisg & 1u) * 2u;   /* this lane's first column in a fragment */

    simdgroup_float8x8 Qm[16], O[16];
    device const float *qrow = q + ((uint64_t)tok * H + kvh * group) * D;
#pragma unroll
    for (uint j = 0; j < 16; j++) {
        simdgroup_load(Qm[j], qrow + j * 8u, D, 0, false);
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
        {
            const uint key = tid >> 4, seg = tid & 15u;
            const int p = kpos[key];
            float4 ka = float4(0.0f), kb = float4(0.0f), va = float4(0.0f), vb = float4(0.0f);
            if (p >= 0) {
                const bool ring = (uint)p < args.stage0;
                const uint64_t at = (ring ? (uint64_t)((uint)p % args.window) : (uint64_t)((uint)p - args.stage0)) * width +
                                    kvh * D + seg * 8u;
                device const half *ks = ring ? ring_k : stage_k;
                device const half *vs = ring ? ring_v : stage_v;
                ka = float4(*reinterpret_cast<device const half4 *>(ks + at));
                kb = float4(*reinterpret_cast<device const half4 *>(ks + at + 4u));
                va = float4(*reinterpret_cast<device const half4 *>(vs + at));
                vb = float4(*reinterpret_cast<device const half4 *>(vs + at + 4u));
            }
            *reinterpret_cast<threadgroup float4 *>(Ks + key * D + seg * 8u) = ka;
            *reinterpret_cast<threadgroup float4 *>(Ks + key * D + seg * 8u + 4u) = kb;
            *reinterpret_cast<threadgroup float4 *>(Vs + key * D + seg * 8u) = va;
            *reinterpret_cast<threadgroup float4 *>(Vs + key * D + seg * 8u + 4u) = vb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 S1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma unroll
        for (uint j = 0; j < 16; j++) {
            simdgroup_float8x8 K0, K1;
            simdgroup_load(K0, Ks + j * 8u, D, 0, true);
            simdgroup_load(K1, Ks + 8u * D + j * 8u, D, 0, true);
            simdgroup_multiply_accumulate(S0, Qm[j], K0, S0);
            simdgroup_multiply_accumulate(S1, Qm[j], K1, S1);
        }
        /* this lane's four scores: keys fn, fn + 1 (S0) and 8 + fn, 9 + fn (S1), all of row fm */
        float sv[4] = { S0.thread_elements()[0], S0.thread_elements()[1], S1.thread_elements()[0], S1.thread_elements()[1] };
        const uint keys[4] = { fn, fn + 1u, 8u + fn, 9u + fn };
        bool valid[4];
        float mx = -3.0e38f;
#pragma unroll
        for (uint c = 0; c < 4; c++) {
            const int p = kpos[keys[c]];
            valid[c] = p >= 0 && (uint)p <= qpos && (args.window == 0u || qpos - (uint)p < args.window);
            bad |= (uint)(valid[c] && kbad[keys[c]] != 0u);
            sv[c] = valid[c] ? sv[c] * args.scale : -3.0e38f;
            mx = max(mx, sv[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        const float m_new = max(m_row, mx);
        const float corr = exp(m_row - m_new);
        float pv[4], rs = 0.0f;
#pragma unroll
        for (uint c = 0; c < 4; c++) {
            pv[c] = valid[c] ? exp(sv[c] - m_new) : 0.0f;
            rs += pv[c];
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
        simdgroup_float8x8 P0, P1;
        P0.thread_elements()[0] = pv[0]; P0.thread_elements()[1] = pv[1];
        P1.thread_elements()[0] = pv[2]; P1.thread_elements()[1] = pv[3];
#pragma unroll
        for (uint j = 0; j < 16; j++) {
            simdgroup_float8x8 V0, V1;
            simdgroup_load(V0, Vs + j * 8u, D, 0, false);
            simdgroup_load(V1, Vs + 8u * D + j * 8u, D, 0, false);
            simdgroup_multiply_accumulate(O[j], P0, V0, O[j]);
            simdgroup_multiply_accumulate(O[j], P1, V1, O[j]);
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
