/* Mellum 2.1, included by ds4.c where this code sits: not a separate translation unit. */

/* From this many rows a prefill chunk runs its experts as tiled GEMMs over each expert's token list, and its
 * Q8_0 projections through the batched ds4_gpu_matmul_q8_0_tensor, instead of one row's arithmetic at a time.
 * Measured on the M5 Max (tests/mellum_trace, prefill, steady runs; MELLUM.md has the table): tiled experts lose
 * at 16 rows and win from 32; batched projections win from 16 and by 2.2x at 1,024; with both, a 2,200-token
 * prompt prefills at about 2,060 t/s against 350 with neither.  The price: these kernels stage activations as
 * half, so a chunk of 32 rows or more no longer equals its decode steps bit for bit.  Over 64 positions each of
 * a 1,100- and a 2,200-token prompt the experts alone move the logits by at most 0.65 and 0.87, and with the
 * projections by at most 2.0 and 8.5 (mean KL 2e-3), with the same top-1 token at all 128.  Chunks under 32 rows
 * and decode keep the exact arithmetic. */
#define MELLUM_TILED_MIN_ROWS 32u

/* The row count from which this graph's chunks take the batched and tiled paths: never under the
 * exact-prefill diagnostic (mellum_graph_alloc). */
static uint32_t mellum_tiled_min_rows(const ds4_qwen4_gpu_graph *g) {
    return g->mellum_exact_prefill ? UINT32_MAX : MELLUM_TILED_MIN_ROWS;
}

/* Mellum's FFN for the T rows of the FFN_NORM scratch, into MOE_OUT (the caller adds the residual): F32 router,
 * softmax over all 64, top 8 renormalized, Q8_0 SwiGLU experts, weighted sum.  It is not qwen4_graph_moe, which
 * reads ffn_gate_inp_shexp and the shared expert unconditionally and folds the hyper-connection combine into its
 * reduce.  Mellum's layer has none of those tensors; this binds only the router and the three expert tensors, and
 * the buffers are the MoE entries of mellum_requests, so the plan counts what the driver uses.  Below
 * MELLUM_TILED_MIN_ROWS the ops are ds4_gpu_mellum_moe_tensor's (contract in ds4_gpu.h): none of them tiles
 * rows together, so a row's bits do not depend on T.  From there the experts are tiled, and they do.
 * add_residual (resident, below MELLUM_TILED_MIN_ROWS only) writes HIDDEN = ATTN_RES + the sum instead of
 * MOE_OUT, the words of the separate add, through ds4_gpu_mellum_moe_add_tensor; the streamed paths refuse it. */
#if defined(__APPLE__)
/* S4: make `ids` of layer il resident in the expert cache and publish their addresses. */
static bool mellum_stream_load_ids(const ds4_model *m, const ds4_layer_weights *l, uint32_t il, const int32_t *ids,
                                   uint32_t n, uint64_t *loads) {
    if (g_mellum_stream_fail_load_at && ++g_mellum_stream_load_count == g_mellum_stream_fail_load_at) {
        fprintf(stderr, "ds4: Mellum streamed load of layer %u failed (test fault)\n", il);
        return false;
    }
    uint64_t gate_bytes = 0, down_bytes = 0;
    if (!streaming_layer_gate_down_expert_bytes(l, &gate_bytes, &down_bytes)) return false;
    const ds4_gpu_stream_expert_table table = graph_stream_expert_table_make(m, l, il, gate_bytes, down_bytes);
    uint32_t loaded = 0;
    const bool ok = ds4_gpu_mellum_stream_load(&table, ids, n, &loaded) != 0;
    if (loads) *loads += loaded;
    return ok;
}
#endif

static bool mellum_graph_moe(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il, uint32_t T,
                             bool add_residual) {
#if !defined(__APPLE__)
    (void)g; (void)m; (void)l; (void)il; (void)T; (void)add_residual;
    return false;
#else
    if (!g || !m || !l || T == 0 || T > g->cap_tokens || !l->ffn_gate_inp || !l->ffn_gate_exps || !l->ffn_up_exps ||
        !l->ffn_down_exps || l->ffn_gate_inp->type != DS4_TENSOR_F32 ||
        (l->ffn_gate_exps->type != DS4_TENSOR_Q8_0 && l->ffn_gate_exps->type != DS4_TENSOR_MXFP4) ||
        l->ffn_up_exps->type != l->ffn_gate_exps->type || l->ffn_down_exps->type != l->ffn_gate_exps->type) {
        return false;   /* S6: Q8_0 or MXFP4 experts, one type for gate, up and down */
    }
    ds4_gpu_mellum_moe p;
    memset(&p, 0, sizeof(p));
    p.n_embd = DS4_N_EMBD;
    p.n_expert = DS4_N_EXPERT;
    p.n_used = DS4_N_EXPERT_USED;
    p.ff_dim = DS4_N_FF_EXP;
    p.expert_type = l->ffn_gate_exps->type;
    p.router_offset = l->ffn_gate_inp->abs_offset;
    p.gate_offset = l->ffn_gate_exps->abs_offset;
    p.up_offset = l->ffn_up_exps->abs_offset;
    p.down_offset = l->ffn_down_exps->abs_offset;
    ds4_gpu_tensor *const *s = g->mellum_scratch;
    if (g->mellum_stream && add_residual) return false;   /* the streamed paths write MOE_OUT */
    if (g->mellum_stream && T == 1u) {
        /* S4 streamed decode: the resident path's router, top-8, experts and reduce, with the experts read
         * through the cache's address tables.  A fresh layer checks (and saves its resume state) after the
         * top-8; a resumed layer restores that state instead.  A miss stops the gate; see mellum_stream_decode. */
        ds4_gpu_tensor *const *addr = g->mellum_stream_addr[il];
        ds4_gpu_tensor *router_out[1] = { s[MELLUM_S_ROUTER_LOGITS] };
        const uint64_t router_off[1] = { p.router_offset };
        const uint32_t router_type[1] = { 0u }; /* F32 */
        const uint32_t router_rows[1] = { p.n_expert };
        const bool resume = g->mellum_stream_resume == il;
        bool ok = resume ?
            ds4_gpu_mellum_stream_restore_tensor(g->mellum_stream_save, s[MELLUM_S_SELECTED], s[MELLUM_S_WEIGHTS],
                                                 s[MELLUM_S_ATTN_RES], s[MELLUM_S_FFN_NORM], il, p.n_embd) != 0 :
            ds4_gpu_qwen4_multi_gemv_tensor(s[MELLUM_S_FFN_NORM], 1, p.n_embd, 1, router_out, m->map, m->size,
                                            router_off, router_type, router_rows) != 0 &&
            ds4_gpu_qwen4_router_topk_tensor(s[MELLUM_S_SELECTED], s[MELLUM_S_WEIGHTS], s[MELLUM_S_ROUTER_LOGITS], NULL,
                                             m->map, m->size, 0u, 0u, 0u, NULL, 1, p.n_expert, p.n_used) != 0 &&
            ds4_gpu_mellum_stream_check_tensor(g->mellum_stream_gate, s[MELLUM_S_SELECTED], s[MELLUM_S_WEIGHTS],
                                               s[MELLUM_S_ATTN_RES], s[MELLUM_S_FFN_NORM], g->mellum_stream_save,
                                               addr[0], addr[1], addr[2], il, p.n_embd) != 0;
        ok = ok &&
             ds4_gpu_qwen4_moe_mid_addr_tensor(s[MELLUM_S_MID], s[MELLUM_S_FFN_NORM], s[MELLUM_S_SELECTED], addr[0], addr[1],
                                               NULL, DS4_GPU_MELLUM_STREAM_SLABS, p.expert_type, p.n_expert, 1, p.n_used,
                                               p.n_embd, p.ff_dim) != 0 &&
             ds4_gpu_qwen4_moe_down_addr_tensor(s[MELLUM_S_PARTIAL], s[MELLUM_S_MID], s[MELLUM_S_SELECTED], addr[2],
                                                NULL, DS4_GPU_MELLUM_STREAM_SLABS, p.expert_type, p.n_expert, 1, p.n_used,
                                                p.ff_dim, p.n_embd) != 0 &&
             ds4_gpu_qwen4_moe_reduce_tensor(s[MELLUM_S_MOE_OUT], s[MELLUM_S_PARTIAL], s[MELLUM_S_WEIGHTS], NULL, NULL, NULL,
                                             NULL, 1, p.n_used, p.n_used, p.n_embd, 0u) != 0;
        if (!ok) fprintf(stderr, "ds4: Mellum streamed MoE failed (layer %u)\n", il);
        return ok;
    }
    if (g->mellum_stream && T > 1u) {
        /* S4 streamed prefill: the resident path's router and top-8, then a host wait to load every expert the
         * chunk chose for this layer, then the resident path's experts (tiled/E1 at mellum_tiled_min_rows rows
         * or more, the exact kernels below) through the address tables, and its reduce. */
        ds4_gpu_tensor *const *addr = g->mellum_stream_addr[il];
        const bool tiled = T >= mellum_tiled_min_rows(g);
        ds4_gpu_tensor *router_out[1] = { s[MELLUM_S_ROUTER_LOGITS] };
        const uint64_t router_off[1] = { p.router_offset };
        const uint32_t router_type[1] = { 0u }, router_rows[1] = { p.n_expert };
        int32_t *sel = malloc((size_t)T * p.n_used * sizeof(int32_t));
        int32_t uniq[64];
        if (tiled) ds4_gpu_qwen4_moe_q8_tiles(g->mellum_moe_nax);
        bool ok = sel &&
            ds4_gpu_qwen4_multi_gemv_tensor(s[MELLUM_S_FFN_NORM], T, p.n_embd, 1, router_out, m->map, m->size,
                                            router_off, router_type, router_rows) &&
            ds4_gpu_qwen4_router_topk_tensor(s[MELLUM_S_SELECTED], s[MELLUM_S_WEIGHTS], s[MELLUM_S_ROUTER_LOGITS], NULL,
                                             m->map, m->size, 0u, 0u, 0u, NULL, T, p.n_expert, p.n_used) &&
            ds4_gpu_end_commands() != 0 &&
            ds4_gpu_tensor_read(s[MELLUM_S_SELECTED], 0, sel, (uint64_t)T * p.n_used * sizeof(int32_t)) != 0;
        const uint32_t n_unique = ok ? ds4_mellum_stream_unique(sel, T, p.n_used, uniq) : UINT32_MAX;
        ok = ok && n_unique != UINT32_MAX && mellum_stream_load_ids(m, l, il, uniq, n_unique, &g->mellum_stream_loads) &&
             glm_graph_begin_commands_if_needed();
        if (tiled) {
            ok = ok &&
                 ds4_gpu_qwen4_moe_build_lists_tensor(s[MELLUM_S_MOE_LISTS], s[MELLUM_S_MOE_COUNTS], s[MELLUM_S_SELECTED], T,
                                                      p.n_used, p.n_expert, g->cap_tokens) &&
                 ds4_gpu_qwen4_moe_mm_mid_addr_tensor(s[MELLUM_S_MID], s[MELLUM_S_FFN_NORM], s[MELLUM_S_MOE_LISTS],
                                                      s[MELLUM_S_MOE_COUNTS], addr[0], addr[1], NULL,
                                                      DS4_GPU_MELLUM_STREAM_SLABS, p.expert_type, p.n_expert, T, p.n_used,
                                                      p.n_used, p.n_embd, p.ff_dim, g->cap_tokens) &&
                 ds4_gpu_qwen4_moe_mm_down_addr_tensor(s[MELLUM_S_PARTIAL], s[MELLUM_S_MID], s[MELLUM_S_MOE_LISTS],
                                                       s[MELLUM_S_MOE_COUNTS], addr[2], NULL, DS4_GPU_MELLUM_STREAM_SLABS,
                                                       p.expert_type, p.n_expert, T, p.n_used, p.n_used, p.ff_dim,
                                                       p.n_embd, g->cap_tokens);
        } else {
            ok = ok &&
                 ds4_gpu_qwen4_moe_mid_addr_tensor(s[MELLUM_S_MID], s[MELLUM_S_FFN_NORM], s[MELLUM_S_SELECTED], addr[0],
                                                   addr[1], NULL, DS4_GPU_MELLUM_STREAM_SLABS, p.expert_type, p.n_expert, T,
                                                   p.n_used, p.n_embd, p.ff_dim) &&
                 ds4_gpu_qwen4_moe_down_addr_tensor(s[MELLUM_S_PARTIAL], s[MELLUM_S_MID], s[MELLUM_S_SELECTED], addr[2],
                                                    NULL, DS4_GPU_MELLUM_STREAM_SLABS, p.expert_type, p.n_expert, T,
                                                    p.n_used, p.ff_dim, p.n_embd);
        }
        ok = ok && ds4_gpu_qwen4_moe_reduce_tensor(s[MELLUM_S_MOE_OUT], s[MELLUM_S_PARTIAL], s[MELLUM_S_WEIGHTS], NULL,
                                                   NULL, NULL, NULL, T, p.n_used, p.n_used, p.n_embd, 0u) != 0;
        if (tiled) ds4_gpu_qwen4_moe_q8_tiles(0);
        free(sel);
        if (!ok) fprintf(stderr, "ds4: Mellum streamed prefill MoE failed (layer %u, T=%u)\n", il, T);
        return ok;
    }
    if (T >= mellum_tiled_min_rows(g)) {
        if (add_residual) return false;
        /* The router, top-8 and reduce as below; the experts as tiled GEMMs over each expert's
         * token list (upstream's prefill MoE), which stage the activations as half, on the
         * tensor tiles unless DS4_MELLUM_MOE_NAX=0 (the level is set for these calls only). */
        ds4_gpu_qwen4_moe_q8_tiles(g->mellum_moe_nax);
        ds4_gpu_tensor *router_out[1] = { s[MELLUM_S_ROUTER_LOGITS] };
        const uint64_t router_off[1] = { p.router_offset };
        const uint32_t router_type[1] = { 0u }, router_rows[1] = { p.n_expert };
        const bool ok =
            ds4_gpu_qwen4_multi_gemv_tensor(s[MELLUM_S_FFN_NORM], T, p.n_embd, 1, router_out, m->map, m->size,
                                            router_off, router_type, router_rows) &&
            ds4_gpu_qwen4_router_topk_tensor(s[MELLUM_S_SELECTED], s[MELLUM_S_WEIGHTS], s[MELLUM_S_ROUTER_LOGITS], NULL,
                                             m->map, m->size, 0u, 0u, 0u, NULL, T, p.n_expert, p.n_used) &&
            ds4_gpu_qwen4_moe_build_lists_tensor(s[MELLUM_S_MOE_LISTS], s[MELLUM_S_MOE_COUNTS], s[MELLUM_S_SELECTED], T,
                                                 p.n_used, p.n_expert, g->cap_tokens) &&
            ds4_gpu_qwen4_moe_mm_mid_tensor(s[MELLUM_S_MID], s[MELLUM_S_FFN_NORM], s[MELLUM_S_MOE_LISTS],
                                            s[MELLUM_S_MOE_COUNTS], m->map, m->size, p.gate_offset, p.up_offset,
                                            p.expert_type, p.n_expert, T, p.n_used, p.n_used, p.n_embd, p.ff_dim,
                                            g->cap_tokens) &&
            ds4_gpu_qwen4_moe_mm_down_tensor(s[MELLUM_S_PARTIAL], s[MELLUM_S_MID], s[MELLUM_S_MOE_LISTS],
                                             s[MELLUM_S_MOE_COUNTS], m->map, m->size, p.down_offset, p.expert_type,
                                             p.n_expert, T, p.n_used, p.n_used, p.ff_dim, p.n_embd, g->cap_tokens) &&
            ds4_gpu_qwen4_moe_reduce_tensor(s[MELLUM_S_MOE_OUT], s[MELLUM_S_PARTIAL], s[MELLUM_S_WEIGHTS], NULL, NULL,
                                            NULL, NULL, T, p.n_used, p.n_used, p.n_embd, 0u) != 0;
        ds4_gpu_qwen4_moe_q8_tiles(0);
        if (!ok) fprintf(stderr, "ds4: Mellum MoE (tiled experts) failed (T=%u)\n", T);
        return ok;
    }
    const bool ok = (add_residual
        ? ds4_gpu_mellum_moe_add_tensor(s[MELLUM_S_HIDDEN], s[MELLUM_S_ATTN_RES], s[MELLUM_S_FFN_NORM],
                                        s[MELLUM_S_ROUTER_LOGITS], s[MELLUM_S_SELECTED], s[MELLUM_S_WEIGHTS],
                                        s[MELLUM_S_MID], s[MELLUM_S_PARTIAL], m->map, m->size, &p, T)
        : ds4_gpu_mellum_moe_tensor(s[MELLUM_S_MOE_OUT], s[MELLUM_S_FFN_NORM], s[MELLUM_S_ROUTER_LOGITS],
                                    s[MELLUM_S_SELECTED], s[MELLUM_S_WEIGHTS], s[MELLUM_S_MID],
                                    s[MELLUM_S_PARTIAL], m->map, m->size, &p, T)) != 0;
    if (!ok) fprintf(stderr, "ds4: Mellum MoE failed (T=%u)\n", T);
    return ok;
#endif
}

/* ---------------------------------------------------------------------
 * Mellum 2.1 forward pass, one command buffer per chunk:
 *
 *   X = embed(tokens)
 *   per layer:  a = rmsnorm(X) * attn_norm
 *               q, k, v = Q8_0 projections of a;  q, k per-head RMS-normed, NeoX RoPE
 *               sliding layer: K/V rounded to f16, attention over the 1,024-row ring and the
 *                              chunk, THEN the chunk's rows committed to the ring
 *               full layer:    K/V rows written to the f16 cache at their positions, ungated
 *                              attention (decode: the single-row dispatch)
 *               r = X + O(heads);  X = r + moe(rmsnorm(r) * ffn_norm)
 *   logits = output(rmsnorm(X_last) * output_norm)
 *
 * Below MELLUM_TILED_MIN_ROWS every op is bit-identical over T rows and over T one-row calls
 * (the projections through the exact rows kernel, the router and experts one row per
 * threadgroup, both attentions row by row), so such a chunk and its decode steps compute the
 * same bytes.  From that row count the projections and experts run as batched and tiled GEMMs
 * on half-staged activations, about 6x faster at prefill and no longer bit-identical to decode
 * (the measured difference is at MELLUM_TILED_MIN_ROWS).  The attentions stay row by row.
 *
 * After a forward, HIDDEN holds the chunk's rows as the last layer left them, before the output
 * norm (the head normalizes into another scratch), so a later graph can read the last row's
 * hidden state there: an MTP head takes it.  The trace's hidden file ends with that row.
 *
 * Trace (DS4_MELLUM_TRACE=<prefix>, DS4_MELLUM_TRACE_AT=<pos,...>): for every listed absolute
 * position a forward covers, four files:
 *   <prefix>.p<pos>.hidden.f32   n_layer x n_embd: each layer's output row
 *   <prefix>.p<pos>.qk.f32       n_layer x (q_dim + kv_dim): post-norm, post-RoPE Q then K
 *   <prefix>.p<pos>.ffn_in.f32   n_layer x n_embd: each layer's MoE input (the FFN-normed row)
 *   <prefix>.p<pos>.logits.f32   one row of n_vocab
 * Tracing waits for the GPU after every layer to read the rows, so it is slow, and it is on only
 * when the variable is set.  Setting one of the two variables without the other is an error.
 * --------------------------------------------------------------------- */
#if defined(__APPLE__)
#define MELLUM_TRACE_MAX_AT 64u

typedef struct {
    const char *prefix;                 /* NULL: no trace */
    uint32_t n;
    uint32_t pos[MELLUM_TRACE_MAX_AT];
} mellum_trace_cfg;

static bool mellum_trace_parse(mellum_trace_cfg *c) {
    memset(c, 0, sizeof(*c));
    const char *prefix = getenv("DS4_MELLUM_TRACE");
    const char *at = getenv("DS4_MELLUM_TRACE_AT");
    if (!prefix || !prefix[0]) {
        if (at && at[0]) {
            fprintf(stderr, "ds4: DS4_MELLUM_TRACE_AT=%s needs DS4_MELLUM_TRACE=<prefix>\n", at);
            return false;
        }
        return true;
    }
    if (!at || !at[0]) {
        fprintf(stderr, "ds4: DS4_MELLUM_TRACE=%s needs DS4_MELLUM_TRACE_AT=<position,...>\n", prefix);
        return false;
    }
    const char *p = at;
    while (*p) {
        char *end = NULL;
        const unsigned long v = (*p >= '0' && *p <= '9') ? strtoul(p, &end, 10) : 0ul;
        if (!end || v > INT32_MAX || c->n >= MELLUM_TRACE_MAX_AT || (*end != ',' && *end != '\0')) {
            fprintf(stderr, "ds4: DS4_MELLUM_TRACE_AT=%s is not a list of at most %u positions\n", at,
                    MELLUM_TRACE_MAX_AT);
            return false;
        }
        c->pos[c->n++] = (uint32_t)v;
        p = *end == ',' ? end + 1 : end;
    }
    c->prefix = prefix;
    return true;
}

static bool mellum_trace_write(const char *prefix, uint32_t pos, const char *what, const float *data, uint64_t count) {
    char path[4096];
    snprintf(path, sizeof(path), "%s.p%u.%s.f32", prefix, pos, what);
    FILE *f = fopen(path, "wb");
    bool ok = f && fwrite(data, sizeof(float), (size_t)count, f) == (size_t)count;
    if (f && fclose(f) != 0) ok = false;
    if (!ok) fprintf(stderr, "ds4: cannot write %s\n", path);
    return ok;
}

/* The model's rope numbers for ds4_gpu_mellum_qk_norm_rope_tensor.  attn_factor is the neutral
 * external factor, 1.0, for BOTH layer kinds: a full layer's rope applies 1 + 0.1 ln(1/freq_scale)
 * (1.2772589) itself, so g_ds4_shape.rope_yarn_attn_factor (that same number, validated at open
 * only) must never reach this struct, and the host refuses it on a full layer. */
static ds4_gpu_mellum_rope mellum_rope_params(void) {
    ds4_gpu_mellum_rope p;
    memset(&p, 0, sizeof(p));
    p.n_head_q = DS4_N_HEAD;
    p.n_head_k = DS4_N_HEAD_KV;
    p.head_dim = DS4_N_HEAD_DIM;
    p.n_ctx_orig = (uint32_t)DS4_ROPE_ORIG_CTX;
    p.eps = DS4_RMS_EPS;
    p.freq_base_sliding = DS4_ROPE_FREQ_BASE_SWA;
    p.freq_base_full = DS4_ROPE_FREQ_BASE;
    p.freq_scale = 1.0f / DS4_ROPE_SCALE_FACTOR;
    p.ext_factor = 1.0f;
    p.attn_factor = 1.0f;
    p.beta_fast = DS4_ROPE_YARN_BETA_FAST;
    p.beta_slow = DS4_ROPE_YARN_BETA_SLOW;
    return p;
}

/* A Q8_0 projection of T rows: one row is the decode kernel; from MELLUM_TILED_MIN_ROWS rows the batched
 * ds4_gpu_matmul_q8_0_tensor (its T-row bits differ from the one-row ones: see the constant); between, the
 * exact rows kernel, which is bit-identical to T one-row calls. */
static bool mellum_proj(const ds4_qwen4_gpu_graph *g, ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w,
                        const ds4_gpu_tensor *x, uint32_t T) {
    if (T == 1u || T >= mellum_tiled_min_rows(g)) {
        return ds4_gpu_matmul_q8_0_tensor(out, m->map, m->size, w->abs_offset, w->dim[0], w->dim[1], x, T) != 0;
    }
    return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out, m->map, m->size, w->abs_offset, w->dim[0], w->dim[1],
                                                        x, T) != 0;
}

/* One layer over the T rows of HIDDEN (positions pos0 ..), in place; encodes into the open command
 * buffer.  full_slot is the layer's index among the full layers: its rows-table region. */
static bool mellum_graph_layer(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il,
                               uint32_t full_slot, uint32_t pos0, uint32_t T, const ds4_gpu_mellum_rope *rope) {
    ds4_gpu_tensor *const *s = g->mellum_scratch;
    const uint32_t E = DS4_N_EMBD;
    const uint32_t kv_w = DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const bool sliding = ds4_mellum_layer_is_sliding(il);
    const float scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    ds4_gpu_tensor *kc = g->layer_k_cache[il], *vc = g->layer_v_cache[il];
    if (g->mellum_stream && g->mellum_stream_resume == il) {
        /* S4 resume at this layer's experts: its attention, KV write, residual and FFN input ran in the
         * stopped encode, and mellum_graph_moe restores them from the saved slot */
        return mellum_graph_moe(g, m, l, il, T, false) &&
               ds4_gpu_add_tensor(s[MELLUM_S_HIDDEN], s[MELLUM_S_ATTN_RES], s[MELLUM_S_MOE_OUT], T * E) != 0;
    }
    bool ok = ds4_gpu_rms_norm_weight_rows_tensor(s[MELLUM_S_ATTN_NORM], s[MELLUM_S_HIDDEN], m->map, m->size,
                                                  l->attn_norm->abs_offset, E, T, DS4_RMS_EPS) != 0 &&
              mellum_proj(g, s[MELLUM_S_Q], m, l->attn_q, s[MELLUM_S_ATTN_NORM], T) &&
              mellum_proj(g, s[MELLUM_S_K], m, l->attn_k, s[MELLUM_S_ATTN_NORM], T) &&
              mellum_proj(g, s[MELLUM_S_V], m, l->attn_v, s[MELLUM_S_ATTN_NORM], T) &&
              /* decode: q's and k's heads in one dispatch, the same words as the two */
              (T == 1u ? ds4_gpu_mellum_qk_norm_rope_fused_tensor : ds4_gpu_mellum_qk_norm_rope_tensor)(
                  s[MELLUM_S_Q], s[MELLUM_S_K], m->map, m->size, l->attn_q_norm->abs_offset,
                  l->attn_k_norm->abs_offset, T, pos0, sliding ? DS4_MELLUM_LAYER_SLIDING : DS4_MELLUM_LAYER_FULL,
                  rope) != 0;
    const bool tiled = g->mellum_attn_tiled && T >= mellum_tiled_min_rows(g);
    if (ok && sliding) {
        /* the chunk's own keys are read as f16 (staged) so a key has one value in prefill and in decode;
         * the ring is written only after every row has read it.  Decode stages K and V in one dispatch. */
        ok = (T == 1u ? ds4_gpu_mellum_copy2_f32_to_f16(s[MELLUM_S_K16], 0, s[MELLUM_S_K], s[MELLUM_S_V16], 0,
                                                        s[MELLUM_S_V], kv_w) != 0
                      : ds4_gpu_tensor_copy_f32_to_f16(s[MELLUM_S_K16], 0, s[MELLUM_S_K], 0, (uint64_t)T * kv_w) != 0 &&
                        ds4_gpu_tensor_copy_f32_to_f16(s[MELLUM_S_V16], 0, s[MELLUM_S_V], 0, (uint64_t)T * kv_w) != 0) &&
             (tiled ? ds4_gpu_mellum_attn_tiled_half_tensor : ds4_gpu_mellum_attn_swa_tensor)(
                 s[MELLUM_S_HEADS], s[MELLUM_S_Q], kc, vc, g->mellum_tag[il], s[MELLUM_S_K16], s[MELLUM_S_V16], T,
                 pos0, DS4_N_SWA, DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, scale) != 0 &&
             ds4_gpu_mellum_ring_commit_tensor(kc, vc, g->mellum_tag[il], s[MELLUM_S_K], s[MELLUM_S_V], T, kv_w, pos0,
                                               DS4_N_SWA) != 0;
    } else if (ok) {
        const uint64_t row_off = (uint64_t)pos0 * kv_w * sizeof(uint16_t);
        ok = T == 1u ? ds4_gpu_mellum_copy2_f32_to_f16(kc, row_off, s[MELLUM_S_K], vc, row_off, s[MELLUM_S_V], kv_w) != 0
                     : ds4_gpu_tensor_copy_f32_to_f16(kc, row_off, s[MELLUM_S_K], 0, (uint64_t)T * kv_w) != 0 &&
                       ds4_gpu_tensor_copy_f32_to_f16(vc, row_off, s[MELLUM_S_V], 0, (uint64_t)T * kv_w) != 0;
        if (ok && T == 1u) {
            ok = ds4_gpu_qwen4_attn_decode_ungated_tensor(s[MELLUM_S_HEADS], s[MELLUM_S_Q], kc, vc,
                                                          s[MELLUM_S_ATTN_PART], DS4_N_HEAD, DS4_N_HEAD_KV,
                                                          DS4_N_HEAD_DIM, pos0, scale) != 0;
        } else if (ok && tiled) {
            /* window 0: the cache is the staged rows, from position 0 */
            ok = ds4_gpu_mellum_attn_tiled_half_tensor(s[MELLUM_S_HEADS], s[MELLUM_S_Q], kc, vc, kc, kc, vc, T, pos0, 0u, DS4_N_HEAD, DS4_N_HEAD_KV,
                            DS4_N_HEAD_DIM, scale) != 0;
        } else if (ok) {
            ok = ds4_gpu_mellum_attn_full_rows_tensor(s[MELLUM_S_HEADS], s[MELLUM_S_Q], kc, vc, s[MELLUM_S_ATTN_TABLE],
                                                      (uint64_t)full_slot * g->cap_tokens, s[MELLUM_S_ATTN_PART], T,
                                                      pos0, DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, scale) != 0;
        }
    }
    return ok &&
           mellum_proj(g, s[MELLUM_S_ATTN_PROJ], m, l->attn_output, s[MELLUM_S_HEADS], T) &&
           /* decode: the attention residual and the FFN norm in one dispatch (upstream's fused kernel), streamed or not,
            * so streamed decode keeps resident's words */
           (T == 1u ? ds4_gpu_add_rms_norm_weight_tensor(s[MELLUM_S_FFN_NORM], s[MELLUM_S_ATTN_RES],
                                                         s[MELLUM_S_HIDDEN], s[MELLUM_S_ATTN_PROJ], m->map, m->size,
                                                         l->ffn_norm->abs_offset, E, DS4_RMS_EPS) != 0
                    : ds4_gpu_add_tensor(s[MELLUM_S_ATTN_RES], s[MELLUM_S_HIDDEN], s[MELLUM_S_ATTN_PROJ], T * E) != 0 &&
                      ds4_gpu_rms_norm_weight_rows_tensor(s[MELLUM_S_FFN_NORM], s[MELLUM_S_ATTN_RES], m->map,
                                                          m->size, l->ffn_norm->abs_offset, E, T, DS4_RMS_EPS) != 0) &&
           /* resident decode: the MoE writes HIDDEN = ATTN_RES + its sum itself, one dispatch fewer; the streamed
            * MoE reads its experts through the cache's tables and adds the residual separately (the same words) */
           (T == 1u && !g->mellum_stream ? mellum_graph_moe(g, m, l, il, T, true)
                                         : mellum_graph_moe(g, m, l, il, T, false) &&
                                           ds4_gpu_add_tensor(s[MELLUM_S_HIDDEN], s[MELLUM_S_ATTN_RES], s[MELLUM_S_MOE_OUT],
                                                              T * E) != 0);
}

/* The logits of one row of HIDDEN into g->logits: output norm, then the one-row LM head.  Prefill
 * and decode make this same call, and a listed position that is not a chunk's last row uses it too. */
static bool mellum_graph_head(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w, uint32_t row) {
    ds4_gpu_tensor *const *s = g->mellum_scratch;
    const uint64_t row_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *x = ds4_gpu_tensor_view(s[MELLUM_S_HIDDEN], (uint64_t)row * row_bytes, row_bytes);
    ds4_gpu_tensor *normed = ds4_gpu_tensor_view(s[MELLUM_S_ATTN_NORM], 0, row_bytes);
    bool ok = x && normed &&
        ds4_gpu_rms_norm_weight_rows_tensor(normed, x, m->map, m->size, w->output_norm->abs_offset, DS4_N_EMBD, 1u,
                                            DS4_RMS_EPS) != 0;
    ok = ok && ds4_gpu_matmul_q8_0_tensor(g->logits, m->map, m->size, w->output->abs_offset, w->output->dim[0],
                                   w->output->dim[1], normed, 1u) != 0;
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(normed);
    return ok;
}
#endif

/* Forward T tokens at g->pos..; logits_out (optional) receives the last token's row.  T <= cap_tokens
 * (the chunk is at most the ring's 1,024 rows).  The command buffer ends, and waits, before this
 * returns, so the next chunk may reuse every scratch, the rows-table regions included. */
#if defined(__APPLE__)
/* S4 streamed decode of one token (ds4_mellum_stream_decode): encode the token from layer 0 with the gate
 * bound; at a stop, load that layer's experts and encode again from its experts.  The finished encode's
 * head leaves the logits in g->logits. */
typedef struct {
    ds4_qwen4_gpu_graph *g;
    const ds4_model *m;
    const ds4_weights *w;
    int32_t token;
} mellum_stream_ctx;

static bool mellum_stream_encode(void *ctx, uint32_t from_layer, bool resume) {
    mellum_stream_ctx *c = ctx;
    ds4_qwen4_gpu_graph *g = c->g;
    ds4_gpu_tensor *const *s = g->mellum_scratch;
    const ds4_gpu_mellum_rope rope = mellum_rope_params();
    g->mellum_stream_resume = resume ? from_layer : UINT32_MAX;
    bool ok = ds4_gpu_stream_gate_reset(g->mellum_stream_gate) && ds4_gpu_stream_gate_bind(g->mellum_stream_gate) &&
              glm_graph_begin_commands_if_needed();
    if (ok && !resume) {
        ok = ds4_gpu_tensor_write(s[MELLUM_S_TOKENS], 0, &c->token, sizeof(int32_t)) != 0 &&
             ds4_gpu_embed_tokens_q8_0_tensor(s[MELLUM_S_HIDDEN], s[MELLUM_S_TOKENS], c->m->map, c->m->size,
                                              c->w->token_embd->abs_offset, (uint32_t)c->w->token_embd->dim[1], 1,
                                              DS4_N_EMBD) != 0;
    }
    uint32_t full_slot = 0;
    for (uint32_t il = 0; il < from_layer; il++) if (!ds4_mellum_layer_is_sliding(il)) full_slot++;
    for (uint32_t il = from_layer; il < DS4_N_LAYER && ok; il++) {
        ok = mellum_graph_layer(g, c->m, &c->w->layer[il], il, full_slot, g->pos, 1, &rope);
        if (!ds4_mellum_layer_is_sliding(il)) full_slot++;
    }
    return ok && mellum_graph_head(g, c->m, c->w, 0);
}

/* Tests (tests/ds4_test.c --mellum-stream-stop-scratch): when set, every stop compares scratch ATTN_RES and
 * FFN_NORM with the stopped layer's save slot; a difference means a dispatch ran past the gate's trip. */
static bool g_mellum_stream_check_scratch;
static uint64_t g_mellum_stream_scratch_intact, g_mellum_stream_scratch_broken;

static void mellum_stream_check_scratch(ds4_qwen4_gpu_graph *g, uint32_t layer) {
    const uint64_t E = DS4_N_EMBD, row = E * sizeof(float);
    float *a = malloc(2u * row), *b = malloc(2u * row);
    bool same = a && b &&
        ds4_gpu_tensor_read(g->mellum_scratch[MELLUM_S_ATTN_RES], 0, a, row) &&
        ds4_gpu_tensor_read(g->mellum_scratch[MELLUM_S_FFN_NORM], 0, a + E, row) &&
        ds4_gpu_tensor_read(g->mellum_stream_save, (uint64_t)layer * (2u * E + 16u) * sizeof(float), b, 2u * row);
    same = same && memcmp(a, b, 2u * row) == 0;
    if (same) g_mellum_stream_scratch_intact++; else g_mellum_stream_scratch_broken++;
    free(a);
    free(b);
}

static bool mellum_stream_finish(void *ctx, bool *stopped, ds4_mellum_stream_stop *stop) {
    mellum_stream_ctx *c = ctx;
    ds4_qwen4_gpu_graph *g = c->g;
    uint32_t layer = 0;
    const bool ok = ds4_gpu_end_commands() != 0;
    (void)ds4_gpu_stream_gate_bind(NULL);
    g->mellum_stream_resume = UINT32_MAX;
    if (!ok || !ds4_gpu_stream_gate_read(g->mellum_stream_gate, &layer, stopped)) return false;
    if (!*stopped) return true;
    if (g_mellum_stream_check_scratch) mellum_stream_check_scratch(g, layer);
    /* the stopped layer's top-8 is still in SELECTED (its saved slot holds the same): load all eight, the
     * cache keeps the resident ones */
    memset(stop, 0, sizeof(*stop));
    stop->layer = layer;
    stop->n_missing = DS4_N_EXPERT_USED;
    return ds4_gpu_tensor_read(g->mellum_scratch[MELLUM_S_SELECTED], 0, stop->missing,
                               DS4_N_EXPERT_USED * sizeof(int32_t)) != 0;
}

static bool mellum_stream_load(void *ctx, uint32_t layer, const int32_t *ids, uint32_t n) {
    mellum_stream_ctx *c = ctx;
    return mellum_stream_load_ids(c->m, &c->w->layer[layer], layer, ids, n, &c->g->mellum_stream_loads);
}

/* Starts the stop's reads (the load that follows finishes them); see ds4_gpu_mellum_stream_load_begin. */
static bool mellum_stream_load_begin(void *ctx, uint32_t layer, const int32_t *ids, uint32_t n) {
    mellum_stream_ctx *c = ctx;
    const ds4_layer_weights *l = &c->w->layer[layer];
    uint64_t gate_bytes = 0, down_bytes = 0;
    if (!streaming_layer_gate_down_expert_bytes(l, &gate_bytes, &down_bytes)) return false;
    const ds4_gpu_stream_expert_table table = graph_stream_expert_table_make(c->m, l, layer, gate_bytes, down_bytes);
    uint32_t loaded = 0;
    const bool ok = ds4_gpu_mellum_stream_load_begin(&table, ids, n, &loaded) != 0;
    c->g->mellum_stream_loads += loaded;
    return ok;
}

/* Drops an encoded resumed pass whose experts did not load, or whose token was cancelled: it must not run. */
static void mellum_stream_discard(void *ctx) {
    (void)ctx;
    ds4_gpu_discard_commands();
    (void)ds4_gpu_stream_gate_bind(NULL);
}

static bool mellum_stream_cancelled(void *ctx) {
    const ds4_qwen4_gpu_graph *g = ((mellum_stream_ctx *)ctx)->g;
    return g->mellum_stream_cancel && g->mellum_stream_cancel(g->mellum_stream_cancel_ud);
}

static bool mellum_stream_decode(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w, int32_t token,
                                 float *logits_out) {
    mellum_stream_ctx c = { g, m, w, token };
    const ds4_mellum_stream_ops ops = { &c, mellum_stream_encode, mellum_stream_finish, mellum_stream_load,
                                        mellum_stream_cancelled, mellum_stream_load_begin, mellum_stream_discard };
    uint32_t stops = 0;
    const int rc = ds4_mellum_stream_decode(&ops, DS4_N_LAYER, &stops);
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    (void)ds4_gpu_stream_gate_bind(NULL);
    g->mellum_stream_resume = UINT32_MAX;
    g->mellum_stream_stops += stops;
    if (rc != DS4_MELLUM_STREAM_OK) {
        fprintf(stderr, "ds4: Mellum streamed decode failed at position %u (%d after %u stops)\n", g->pos, rc, stops);
        return false;
    }
    if (logits_out && !ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)DS4_N_VOCAB * sizeof(float))) return false;
    g->pos += 1;
    return true;
}
#endif

static bool mellum_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                        const int *tokens, uint32_t T, float *logits_out, bool all_rows) {
#if !defined(__APPLE__)
    (void)g; (void)m; (void)w; (void)tokens; (void)T; (void)logits_out; (void)all_rows;
    return false;
#else
    if (!g || !tokens || T == 0 || T > g->cap_tokens || g->pos + T > g->ctx_cap) return false;
    if (all_rows && T > 1u) {
        fprintf(stderr, "ds4: the Mellum forward returns the last row's logits only\n");
        return false;
    }
    mellum_trace_cfg tc;
    if (!mellum_trace_parse(&tc)) return false;
#if defined(__APPLE__)
    if (g->mellum_stream) {
        if (tc.prefix) {
            fprintf(stderr, "ds4: Mellum SSD streaming does not support tracing\n");
            return false;
        }
        /* one token: stop/resume decode; a chunk: the ordinary forward, whose MoE syncs per layer */
        if (T == 1u) return mellum_stream_decode(g, m, w, tokens[0], logits_out);
    }
#endif
    const uint32_t pos0 = g->pos, n_layer = DS4_N_LAYER, E = DS4_N_EMBD, V = DS4_N_VOCAB;
    const uint32_t q_dim = DS4_N_HEAD * DS4_N_HEAD_DIM, kv_dim = DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint32_t qk_dim = q_dim + kv_dim;
    uint32_t tr_row[MELLUM_TRACE_MAX_AT], tr_n = 0;
    for (uint32_t r = 0; tc.prefix && r < T; r++) {
        for (uint32_t i = 0; i < tc.n; i++) {
            if (tc.pos[i] == pos0 + r) { tr_row[tr_n++] = r; break; }
        }
    }
    float *hid = tr_n ? malloc((size_t)tr_n * n_layer * E * sizeof(float)) : NULL;
    float *ffn = tr_n ? malloc((size_t)tr_n * n_layer * E * sizeof(float)) : NULL;
    float *qk = tr_n ? malloc((size_t)tr_n * n_layer * qk_dim * sizeof(float)) : NULL;
    float *row_logits = tr_n ? malloc((size_t)V * sizeof(float)) : NULL;
    if (tr_n && (!hid || !ffn || !qk || !row_logits)) {
        free(hid); free(ffn); free(qk); free(row_logits);
        return false;
    }
    ds4_gpu_tensor *const *s = g->mellum_scratch;
    const ds4_gpu_mellum_rope rope = mellum_rope_params();
    bool ok = ds4_gpu_tensor_write(s[MELLUM_S_TOKENS], 0, tokens, (uint64_t)T * sizeof(int32_t)) != 0 &&
              glm_graph_begin_commands_if_needed() &&
              ds4_gpu_embed_tokens_q8_0_tensor(s[MELLUM_S_HIDDEN], s[MELLUM_S_TOKENS], m->map, m->size,
                                               w->token_embd->abs_offset, (uint32_t)w->token_embd->dim[1], T, E) != 0;
    uint32_t full_slot = 0;
    for (uint32_t il = 0; il < n_layer && ok; il++) {
        ok = mellum_graph_layer(g, m, &w->layer[il], il, full_slot, pos0, T, &rope);
        if (!ds4_mellum_layer_is_sliding(il)) full_slot++;
        if (ok && tr_n) {
            /* the rows are on the GPU: wait, then read this layer's output row, its MoE input row and its
             * post-RoPE Q and K row */
            ok = ds4_gpu_end_commands() != 0;
            for (uint32_t k = 0; ok && k < tr_n; k++) {
                const uint64_t r = tr_row[k];
                float *h = hid + ((uint64_t)k * n_layer + il) * E;
                float *x = qk + ((uint64_t)k * n_layer + il) * qk_dim;
                ok = ds4_gpu_tensor_read(s[MELLUM_S_HIDDEN], r * E * sizeof(float), h, (uint64_t)E * sizeof(float)) != 0 &&
                     ds4_gpu_tensor_read(s[MELLUM_S_FFN_NORM], r * E * sizeof(float),
                                         ffn + ((uint64_t)k * n_layer + il) * E, (uint64_t)E * sizeof(float)) != 0 &&
                     ds4_gpu_tensor_read(s[MELLUM_S_Q], r * q_dim * sizeof(float), x, (uint64_t)q_dim * sizeof(float)) != 0 &&
                     ds4_gpu_tensor_read(s[MELLUM_S_K], r * kv_dim * sizeof(float), x + q_dim,
                                         (uint64_t)kv_dim * sizeof(float)) != 0;
            }
            if (ok && il + 1u < n_layer) ok = glm_graph_begin_commands_if_needed();
        }
    }
    /* the head: each traced row's logits (a wait and a read each), then the last row's for the caller */
    bool last_done = false;
    for (uint32_t k = 0; ok && k < tr_n; k++) {
        ok = glm_graph_begin_commands_if_needed() && mellum_graph_head(g, m, w, tr_row[k]) &&
             ds4_gpu_end_commands() != 0 &&
             ds4_gpu_tensor_read(g->logits, 0, row_logits, (uint64_t)V * sizeof(float)) != 0;
        if (ok) {
            const uint32_t p = pos0 + tr_row[k];
            ok = mellum_trace_write(tc.prefix, p, "hidden", hid + (uint64_t)k * n_layer * E, (uint64_t)n_layer * E) &&
                 mellum_trace_write(tc.prefix, p, "ffn_in", ffn + (uint64_t)k * n_layer * E, (uint64_t)n_layer * E) &&
                 mellum_trace_write(tc.prefix, p, "qk", qk + (uint64_t)k * n_layer * qk_dim, (uint64_t)n_layer * qk_dim) &&
                 mellum_trace_write(tc.prefix, p, "logits", row_logits, V);
        }
        if (ok && logits_out && tr_row[k] == T - 1u) {
            memcpy(logits_out, row_logits, (size_t)V * sizeof(float));
            last_done = true;
        }
    }
    if (ok && logits_out && !last_done) {
        ok = glm_graph_begin_commands_if_needed() && mellum_graph_head(g, m, w, T - 1u) &&
             ds4_gpu_end_commands() != 0 &&
             ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)V * sizeof(float)) != 0;
    }
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = false;
    free(hid); free(ffn); free(qk); free(row_logits);
    if (ok) g->pos += T;
    return ok;
#endif
}

/* S5 grouped layer-major prefill (streamed Mellum, K > 1).  rows (two or more, at most K x cap_tokens, never
 * ending in a one-row chunk) at g->pos.. run as consecutive chunks of cap_tokens, every layer over all of the
 * group's chunks, in order, before the next layer, so a layer's experts load about once per group rather than
 * once per chunk.  Each (layer, chunk) is the ordinary mellum_graph_layer on that chunk's rows, copied from the
 * group buffer into HIDDEN and back, at the chunk's own position: its KV, ring and hidden inputs equal the
 * per-chunk forward's.  The streamed MoE waits on the GPU after every (layer, chunk) router, which keeps the
 * CPU-written full-attention rows table safe to reuse across a layer's chunks.  logits_out receives the last
 * row's logits; display, when set, hears a prefill_display event after each layer pass (not a checkpoint). */
static bool mellum_graph_prefill_group(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                       const int *tokens, uint32_t rows, float *logits_out,
                                       ds4_session_progress_fn display, void *display_ud,
                                       int display_base, int display_total) {
#if !defined(__APPLE__)
    (void)g; (void)m; (void)w; (void)tokens; (void)rows; (void)logits_out;
    (void)display; (void)display_ud; (void)display_base; (void)display_total;
    return false;
#else
    const uint32_t cap = g ? g->cap_tokens : 0u, E = DS4_N_EMBD;
    if (!g || !tokens || !g->mellum_stream || !g->mellum_group_hidden || cap == 0 || rows < 2u ||
        rows > g->mellum_group_k * cap || g->pos + rows > g->ctx_cap) return false;
    const uint32_t n_chunk = (rows + cap - 1u) / cap, last = rows - (n_chunk - 1u) * cap;
    if (last == 1u) return false;   /* a one-row chunk runs on the decode path */
    mellum_trace_cfg tc;
    if (!mellum_trace_parse(&tc)) return false;
    if (tc.prefix) {
        fprintf(stderr, "ds4: Mellum SSD streaming does not support tracing\n");
        return false;
    }
    ds4_gpu_tensor *const *s = g->mellum_scratch;
    ds4_gpu_tensor *hid = g->mellum_group_hidden;
    const ds4_gpu_mellum_rope rope = mellum_rope_params();
    const uint32_t pos0 = g->pos;
    const uint64_t row_bytes = (uint64_t)E * sizeof(float);
    bool ok = true;
    for (uint32_t c = 0; c < n_chunk && ok; c++) {
        const uint32_t r0 = c * cap, T = c + 1u < n_chunk ? cap : last;
        /* TOKENS is one chunk wide: the previous chunk's embed must have run before it is rewritten */
        if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = false;
        ok = ok && ds4_gpu_tensor_write(s[MELLUM_S_TOKENS], 0, tokens + r0, (uint64_t)T * sizeof(int32_t)) != 0 &&
             glm_graph_begin_commands_if_needed() &&
             ds4_gpu_embed_tokens_q8_0_tensor(s[MELLUM_S_HIDDEN], s[MELLUM_S_TOKENS], m->map, m->size,
                                              w->token_embd->abs_offset, (uint32_t)w->token_embd->dim[1], T, E) != 0 &&
             ds4_gpu_tensor_copy(hid, r0 * row_bytes, s[MELLUM_S_HIDDEN], 0, T * row_bytes) != 0;
    }
    uint32_t full_slot = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        for (uint32_t c = 0; c < n_chunk && ok; c++) {
            const uint32_t r0 = c * cap, T = c + 1u < n_chunk ? cap : last;
            ok = glm_graph_begin_commands_if_needed() &&
                 ds4_gpu_tensor_copy(s[MELLUM_S_HIDDEN], 0, hid, r0 * row_bytes, T * row_bytes) != 0 &&
                 mellum_graph_layer(g, m, &w->layer[il], il, full_slot, pos0 + r0, T, &rope) &&
                 ds4_gpu_tensor_copy(hid, r0 * row_bytes, s[MELLUM_S_HIDDEN], 0, T * row_bytes) != 0;
        }
        if (!ds4_mellum_layer_is_sliding(il)) full_slot++;
        if (ok && display) {
            display(display_ud, "prefill_display", display_base + (int)((uint64_t)rows * (il + 1u) / DS4_N_LAYER),
                    display_total);
        }
    }
    /* the last chunk's rows are still in HIDDEN: its last row's logits */
    if (ok && logits_out) {
        ok = glm_graph_begin_commands_if_needed() && mellum_graph_head(g, m, w, last - 1u) &&
             ds4_gpu_end_commands() != 0 &&
             ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = false;
    if (ok) g->pos += rows;
    return ok;
#endif
}
