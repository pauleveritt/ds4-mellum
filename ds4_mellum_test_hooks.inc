/* Mellum 2.1, included by ds4.c where this code sits: not a separate translation unit. */

/* For tests/ds4_test.c --mellum-moe: where one real layer's FFN weights are, found by their GGUF names and not
 * through the layer struct mellum_graph_moe reads (so a test's reference does not share the driver's binding),
 * and false if the layer has a shared-expert tensor. */
bool ds4_test_mellum_moe_weights(ds4_engine *engine, uint32_t layer, const void **map,
                                 uint64_t *router, uint64_t *gate, uint64_t *up, uint64_t *down,
                                 uint32_t *expert_type) {
    if (!engine || !ds4_model_is_mellum() || layer >= DS4_N_LAYER) return false;
    const ds4_model *m = &engine->model;
    const ds4_tensor *r = tensor_by_namef(m, "blk.%u.ffn_gate_inp.weight", layer);
    const ds4_tensor *g = tensor_by_namef(m, "blk.%u.ffn_gate_exps.weight", layer);
    const ds4_tensor *u = tensor_by_namef(m, "blk.%u.ffn_up_exps.weight", layer);
    const ds4_tensor *d = tensor_by_namef(m, "blk.%u.ffn_down_exps.weight", layer);
    if (!r || !g || !u || !d || tensor_by_namef(m, "blk.%u.ffn_gate_inp_shexp.weight", layer) ||
        tensor_by_namef(m, "blk.%u.ffn_gate_shexp.weight", layer)) {
        return false;
    }
    *map = m->map;
    *router = r->abs_offset;
    *gate = g->abs_offset;
    *up = u->abs_offset;
    *down = d->abs_offset;
    *expert_type = g->type;   /* S6: Q8_0 or MXFP4; the loader made gate, up and down agree */
    return true;
}

/* mellum_graph_moe on T host rows through one real layer: out [T][n_embd], selected [T][n_used], weights
 * [T][n_used]; then row 0 alone (T = 1) into out1 [n_embd].  A graph for this call only. */
bool ds4_test_mellum_moe(ds4_engine *engine, uint32_t layer, const float *x, uint32_t T,
                         float *out, float *out1, int32_t *selected, float *weights) {
#if defined(__APPLE__)
    if (!engine || !ds4_model_is_mellum() || layer >= DS4_N_LAYER || T == 0 || T > DS4_N_SWA) return false;
    ds4_qwen4_gpu_graph g;
    memset(&g, 0, sizeof(g));   /* qwen4_graph_alloc zeroes the graph before it allocates */
    if (!mellum_graph_alloc(&g, DS4_N_SWA, T)) return false;
    ds4_gpu_tensor *const *s = g.mellum_scratch;
    const ds4_layer_weights *l = &engine->weights.layer[layer];
    const uint64_t E = DS4_N_EMBD, K = DS4_N_EXPERT_USED;
    const bool ok =
        ds4_gpu_tensor_write(s[MELLUM_S_FFN_NORM], 0, x, (uint64_t)T * E * sizeof(float)) &&
        mellum_graph_moe(&g, &engine->model, l, layer, T, false) &&
        ds4_gpu_tensor_read(s[MELLUM_S_MOE_OUT], 0, out, (uint64_t)T * E * sizeof(float)) &&
        ds4_gpu_tensor_read(s[MELLUM_S_SELECTED], 0, selected, (uint64_t)T * K * sizeof(int32_t)) &&
        ds4_gpu_tensor_read(s[MELLUM_S_WEIGHTS], 0, weights, (uint64_t)T * K * sizeof(float)) &&
        ds4_gpu_tensor_write(s[MELLUM_S_FFN_NORM], 0, x, E * sizeof(float)) &&
        mellum_graph_moe(&g, &engine->model, l, layer, 1, false) &&
        ds4_gpu_tensor_read(s[MELLUM_S_MOE_OUT], 0, out1, E * sizeof(float));
    qwen4_graph_free(&g);
    return ok;
#else
    (void)engine; (void)layer; (void)x; (void)T; (void)out; (void)out1; (void)selected; (void)weights;
    return false;
#endif
}
/* For tests/ds4_test.c --mellum-stream-faults: fail the nth streamed load from now (0: never). */
bool ds4_test_mellum_stream_fail_load(ds4_session *s, uint64_t nth) {
    if (!s || !ds4_session_is_qwen4(s) || !s->qwen4_graph.mellum_stream) return false;
    g_mellum_stream_fail_load_at = nth;
    g_mellum_stream_load_count = 0;
    return true;
}

/* For tests/ds4_test.c --mellum-stream-stop-scratch: switch the per-stop scratch check on or off, and read its
 * counts since it was last switched on. */
bool ds4_test_mellum_stream_stop_scratch(ds4_session *s, bool on, uint64_t *intact, uint64_t *broken) {
    if (!s || !ds4_session_is_qwen4(s) || !s->qwen4_graph.mellum_stream) return false;
    if (on && !g_mellum_stream_check_scratch) g_mellum_stream_scratch_intact = g_mellum_stream_scratch_broken = 0;
    g_mellum_stream_check_scratch = on;
    if (intact) *intact = g_mellum_stream_scratch_intact;
    if (broken) *broken = g_mellum_stream_scratch_broken;
    return true;
}

/* For tests/ds4_test.c --mellum-stream-read-fault: fail the nth expert read from now (0: never), and count the
 * expert cache's slab slots (free, held, handed out). */
bool ds4_test_mellum_stream_fail_read(ds4_session *s, uint64_t nth) {
    if (!s || !ds4_session_is_qwen4(s) || !s->qwen4_graph.mellum_stream) return false;
#if defined(__APPLE__)
    ds4_gpu_test_stream_fail_read(nth);
    return true;
#else
    (void)nth;
    return false;
#endif
}

bool ds4_test_mellum_stream_slots(ds4_session *s, uint32_t *free_slots, uint32_t *held_slots, uint32_t *handed_out) {
    if (!s || !ds4_session_is_qwen4(s) || !s->qwen4_graph.mellum_stream) return false;
#if defined(__APPLE__)
    ds4_gpu_test_stream_slots(free_slots, held_slots, handed_out);
    return true;
#else
    (void)free_slots; (void)held_slots; (void)handed_out;
    return false;
#endif
}

/* For tests/ds4_test.c --mellum-stream-identity and tools/mellum_s4_measure: experts read into slots. */
bool ds4_test_mellum_stream_loads(ds4_session *s, uint64_t *loads) {
    if (!s || !loads || !ds4_session_is_qwen4(s) || !s->qwen4_graph.mellum_stream) return false;
    *loads = s->qwen4_graph.mellum_stream_loads;
    return true;
}

/* For tests/ds4_test.c --mellum-stream-identity: the stops a streamed Mellum session has taken. */
bool ds4_test_mellum_stream_stops(ds4_session *s, uint64_t *stops) {
    if (!s || !stops || !ds4_session_is_qwen4(s) || !s->qwen4_graph.mellum_stream) return false;
    *stops = s->qwen4_graph.mellum_stream_stops;
    return true;
}
/* For tests/ds4_test.c --mellum-rewind: where a qwen4-family session's graph stands (its next position). */
bool ds4_test_session_graph_pos(ds4_session *s, uint32_t *pos) {
    if (!s || !pos || !ds4_session_is_qwen4(s)) return false;
    *pos = s->qwen4_graph.pos;
    return true;
}

/* For tests/ds4_test.c --mellum-bk: at the rewind frontier, k one-row forwards (chunk false) or one k-row forward
 * (chunk true); logits_last gets the last row's logits and ms the wall time from the first encode to that read.
 * Then the frontier is restored, outside the timing. */
bool ds4_test_mellum_rows_arm(ds4_session *s, const int *tokens, uint32_t k, bool chunk, float *logits_last,
                              double *ms) {
    (void)s; (void)tokens; (void)k; (void)chunk; (void)logits_last; (void)ms;
    return false;
}

/* For tests/ds4_test.c --mellum-b0 (phase 8b, D2), at a Mellum session's position, which must be where its last
 * sync ended (the rewind frontier): verify false runs two one-row forwards (two decode steps), verify true one
 * two-row forward and then the LM head on its first row, in its own command buffer (what a two-row verify
 * computes).  logits0 and logits1 get the two rows' logits; ms is the wall time from the first encode to the last
 * logits read, head_ms the first row's head alone (verify only).  Then the frontier is restored, outside the
 * timing: the second row's ring write evicted a row the first position reads. */
bool ds4_test_mellum_b0_arm(ds4_session *s, const int tokens[2], bool verify, float *logits0, float *logits1,
                            double *ms, double *head_ms) {
#if defined(__APPLE__)
    if (!s || !tokens || !logits0 || !logits1 || !ms || !head_ms || !ds4_session_is_qwen4(s) ||
        !ds4_model_is_mellum() || !s->qwen4_graph_ready) {
        return false;
    }
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const ds4_engine *e = s->engine;
    if (!g->mellum_frontier_valid || g->mellum_frontier_pos != g->pos || g->pos + 2u > g->ctx_cap) return false;
    *head_ms = 0.0;
    const double t0 = now_sec();
    bool ok;
    if (!verify) {
        ok = mellum_graph_forward_tokens(g, &e->model, &e->weights, &tokens[0], 1, logits0, false) &&
             mellum_graph_forward_tokens(g, &e->model, &e->weights, &tokens[1], 1, logits1, false);
    } else {
        ok = mellum_graph_forward_tokens(g, &e->model, &e->weights, tokens, 2, logits1, false);
        const double t1 = now_sec();
        ok = ok && glm_graph_begin_commands_if_needed() && mellum_graph_head(g, &e->model, &e->weights, 0) &&
             ds4_gpu_end_commands() != 0 &&
             ds4_gpu_tensor_read(g->logits, 0, logits0, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        *head_ms = (now_sec() - t1) * 1e3;
    }
    *ms = (now_sec() - t0) * 1e3;
    return mellum_graph_frontier_copy(g, false) && ok;
#else
    (void)s; (void)tokens; (void)verify; (void)logits0; (void)logits1; (void)ms; (void)head_ms;
    return false;
#endif
}
