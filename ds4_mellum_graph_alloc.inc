/* Mellum 2.1, included by ds4.c where this code sits: not a separate translation unit. */

static void qwen4_graph_free(ds4_qwen4_gpu_graph *g);

/* Mellum's tensors were validated at load (weights_validate_mellum_layout:
 * Q8_0 projections and experts, F32 norms and router); what the session
 * needs is that every layer and the output head are bound, and Metal. */
static bool mellum_graph_weights_supported(const ds4_weights *w) {
#if !defined(__APPLE__)
    (void)w;
    fprintf(stderr, "ds4: Mellum runs on the Metal graph only\n");
    return false;
#else
    if (!w || !w->token_embd || !w->output || !w->output_norm) {
        fprintf(stderr, "ds4: Mellum GPU graph: the embedding or output head is not bound\n");
        return false;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_mellum_layer_has_required(&w->layer[il], il)) {
            fprintf(stderr, "ds4: Mellum GPU graph: layer %u is missing a tensor\n", il);
            return false;
        }
    }
    return true;
#endif
}

/* A ring slot with tag -1 holds no row.  Runs at allocation and at reset: a
 * session that rewinds or restarts must not read the previous run's slots. */
static bool mellum_graph_forget_ring(ds4_qwen4_gpu_graph *g) {
    const uint32_t window = DS4_N_SWA;
    int32_t empty[1024];
    if (window > sizeof(empty) / sizeof(empty[0])) return false;
    for (uint32_t i = 0; i < window; i++) empty[i] = -1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (g->mellum_tag[il] &&
            !ds4_gpu_tensor_write(g->mellum_tag[il], 0, empty, (uint64_t)window * sizeof(empty[0]))) return false;
    }
    return true;
}

/* Every GPU tensor a Mellum session holds, from the one table the plan sums
 * (mellum_requests).  Allocated at create for cap_tokens rows, never at the
 * first sync.  Prints once: allocated (the sum of the bytes requested from the
 * device for these tensors, ds4_gpu_tensor_bytes; not the pages Metal rounds
 * them up to) against planned (the estimate the startup line reports). */
static bool mellum_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens) {
    /* A diagnostic, not a semantic variant (AGENT.md): DS4_MELLUM_EXACT_PREFILL=1 sends every prefill
     * chunk through the exact paths (MELLUM_TILED_MIN_ROWS), so a long prefill equals its decode steps
     * byte for byte and the oracle can check it against the old engine.  Read once, here, per session. */
    const char *exact = getenv("DS4_MELLUM_EXACT_PREFILL");
    g->mellum_exact_prefill = exact && exact[0] && strcmp(exact, "0") != 0;
    /* The tiled experts run on the Metal 4 tensor tiles where the device has them (experiment E1: the
     * same half operands as the simdgroup tiles, summed in another order).  DS4_MELLUM_MOE_NAX=0, a
     * diagnostic like the one above, keeps the simdgroup tiles to compare against. */
    const char *nax = getenv("DS4_MELLUM_MOE_NAX");
    g->mellum_moe_nax = nax && !strcmp(nax, "0") ? 0u : 2u;
    /* Chunks from MELLUM_TILED_MIN_ROWS rows run their attention in tiles (phase 8a, T2), eight tokens sharing each
     * tile of keys, with half operands (the scaled queries and the probabilities).  A diagnostic, like the one
     * above: DS4_MELLUM_ATTN_TILED=0 keeps the kernels the decode steps use; any other value is the default. */
    const char *attn = getenv("DS4_MELLUM_ATTN_TILED");
    g->mellum_attn_tiled = !(attn && !strcmp(attn, "0"));
#if defined(__APPLE__)
    g->mellum_stream = g_mellum_stream_engine;
    g->mellum_group_k = mellum_group_k(cap_tokens);
    g->mellum_stream_resume = UINT32_MAX;
    if (g->mellum_stream) {
        g->mellum_stream_gate = ds4_gpu_stream_gate_create();
        g->mellum_stream_save = ds4_gpu_tensor_alloc((uint64_t)DS4_N_LAYER * (2u * DS4_N_EMBD + 16u) * sizeof(float));
        if (!g->mellum_stream_gate || !g->mellum_stream_save) return false;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (!ds4_gpu_mellum_stream_layer_tables(il, &g->mellum_stream_addr[il][0], &g->mellum_stream_addr[il][1],
                                                    &g->mellum_stream_addr[il][2])) {
                fprintf(stderr, "ds4: Mellum SSD streaming: layer %u's expert address tables are unavailable\n", il);
                return false;
            }
        }
    }
#endif
    g->ctx_cap = ctx_cap;
    g->cap_tokens = cap_tokens;
    g->n_logit_rows = 1u;
    g->owns_scratch = true;
    mellum_req reqs[MELLUM_MAX_REQS];
    const uint32_t n = mellum_requests(ctx_cap, cap_tokens, mellum_group_rows(cap_tokens), reqs);
    bool ok = true;
    uint64_t allocated = 0;
    for (uint32_t i = 0; i < n && ok; i++) {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(reqs[i].bytes);
        if (!t) { ok = false; break; }
        allocated += ds4_gpu_tensor_bytes(t);
        const uint32_t x = reqs[i].index;
        switch (reqs[i].kind) {
        case MELLUM_REQ_K: g->layer_k_cache[x] = t; break;
        case MELLUM_REQ_V: g->layer_v_cache[x] = t; break;
        case MELLUM_REQ_TAG: g->mellum_tag[x] = t; break;
        case MELLUM_REQ_SCRATCH: g->mellum_scratch[x] = t; break;
        case MELLUM_REQ_LOGITS: g->logits = t; break;
        case MELLUM_REQ_FRONTIER_K: g->mellum_frontier_k[x] = t; break;
        case MELLUM_REQ_FRONTIER_V: g->mellum_frontier_v[x] = t; break;
        case MELLUM_REQ_FRONTIER_TAG: g->mellum_frontier_tag[x] = t; break;
        case MELLUM_REQ_GROUP_HIDDEN: g->mellum_group_hidden = t; break;
        }
    }
    g->mellum_frontier_logits = ok ? malloc((size_t)DS4_N_VOCAB * sizeof(float)) : NULL;
    ok = ok && g->mellum_frontier_logits;
    ok = ok && mellum_graph_forget_ring(g);
#if defined(__APPLE__)
    /* the table's partials are MELLUM_ATTN_MAX_SPLITS wide; the kernels' own count decides what they need */
    if (ok) {
        const uint32_t slice = cap_tokens < MELLUM_ATTN_SLICE_ROWS ? cap_tokens : MELLUM_ATTN_SLICE_ROWS;
        const uint64_t need = ds4_gpu_qwen4_attn_part_floats(slice, DS4_N_HEAD, DS4_N_HEAD_DIM) * sizeof(float);
        const ds4_gpu_tensor *part = g->mellum_scratch[MELLUM_S_ATTN_PART];
        if (MELLUM_ATTN_ROW_BYTES != DS4_GPU_QWEN4_ATTN_ROW_BYTES) {
            fprintf(stderr, "ds4: Mellum rows-table entries are %u bytes in the plan, the kernels use %u\n",
                    MELLUM_ATTN_ROW_BYTES, (unsigned)DS4_GPU_QWEN4_ATTN_ROW_BYTES);
            ok = false;
        } else if (!part || ds4_gpu_tensor_bytes(part) != need) {
            fprintf(stderr, "ds4: Mellum attention partials are %" PRIu64 " bytes in the plan, the kernels need %" PRIu64 "\n",
                    part ? ds4_gpu_tensor_bytes(part) : 0u, need);
            ok = false;
        }
    }
#endif
    g->host_logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    ok=ok && g->host_logits!=NULL;
    g->mellum_allocated_bytes = allocated;
    if (!ok) {
        fprintf(stderr, "ds4: Mellum graph allocation failed (ctx %u, chunk %u, %" PRIu64 " bytes taken)\n",
                ctx_cap, cap_tokens, allocated);
        qwen4_graph_free(g);
        return false;
    }
    const ds4_context_memory plan =
        ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, (int)ctx_cap, cap_tokens, false);
    fprintf(stderr, "mellum: session allocated %" PRIu64 " bytes, planned %" PRIu64 " bytes\n",
            allocated, plan.total_bytes);
    return true;
}
