/* Mellum 2.1, included by ds4.c where this code sits: not a separate translation unit. */

/* ---------------------------------------------------------------------
 * Mellum 2.1 memory plan and session allocation.
 *
 * One table is the single source of truth for every GPU tensor a Mellum
 * session holds: mellum_requests lists them with their sizes, the plan
 * (mellum_context_memory, behind ds4_context_memory_estimate) sums the table
 * by class, and mellum_graph_alloc (GPU build) walks the same table to
 * allocate.  A tensor the session allocates is in the plan because there is
 * no second list.
 *
 * Layout.  Full-attention layers (il % 4 == 3) hold K and V for every
 * position, [ctx][n_head_kv * head_dim] f16.  Sliding layers hold a ring of
 * DS4_N_SWA slots, the same row shape: position p lives in slot p % window,
 * and each ring carries an int32 position tag per slot (one array per ring: a
 * row's K and V share it) so a stale slot is never read as a live row.
 * Attention reads the ring before the chunk is committed to it, so a ring of
 * exactly the window survives a one-token rewind: after a forward ending at
 * P - 1 the ring holds P - W .. P - 1, and a query at P - 1 needs only
 * P - W .. P - 2 of it.  The prefill and MoE transients are sized for
 * cap_tokens rows and allocated at session create, never at the first sync.
 * The full-attention layers' two scratch buffers (the rows table and the
 * 64-row partials) are in the table too.
 *
 * The chunk.  Mellum's prefill chunk is min(requested, DS4_N_SWA): the ring
 * commit refuses more rows than the ring holds, and DS4_QWEN4_PREFILL_CHUNK
 * accepts up to 65,536, so an unclamped value would fail a forward with no
 * message naming the cause.
 * --------------------------------------------------------------------- */
typedef enum {
    MELLUM_CLASS_KV,        /* K/V caches, rings and ring tags: the session's KV state */
    MELLUM_CLASS_PREFILL,   /* cap_tokens rows of activations */
    MELLUM_CLASS_MOE,       /* cap_tokens rows of routing and expert transients */
    MELLUM_CLASS_LOGITS,    /* the row a caller reads back */
} mellum_class;

typedef enum {
    MELLUM_REQ_K,           /* index = layer: full cache [ctx] or ring [window], f16 */
    MELLUM_REQ_V,
    MELLUM_REQ_TAG,         /* index = sliding layer: int32 position tag per ring slot (K and V rows share it) */
    MELLUM_REQ_SCRATCH,     /* index = mellum_scratch_id */
    MELLUM_REQ_LOGITS,
    MELLUM_REQ_FRONTIER_K,  /* index = sliding layer: the ring's K, V and tags as the last sync left them */
    MELLUM_REQ_FRONTIER_V,
    MELLUM_REQ_FRONTIER_TAG,
    MELLUM_REQ_GROUP_HIDDEN, /* S5: a streamed prefill group's hidden rows, K x cap_tokens, f32 */
} mellum_req_kind;

typedef enum {
    /* prefill */
    MELLUM_S_TOKENS,
    MELLUM_S_HIDDEN, MELLUM_S_ATTN_NORM, MELLUM_S_ATTN_PROJ,
    MELLUM_S_ATTN_RES, MELLUM_S_FFN_NORM, MELLUM_S_MOE_OUT,
    MELLUM_S_Q, MELLUM_S_HEADS, MELLUM_S_K, MELLUM_S_V, MELLUM_S_K16, MELLUM_S_V16,
    /* full attention: one rows-table region per full layer, and the partials of one 64-row slice (shared) */
    MELLUM_S_ATTN_TABLE, MELLUM_S_ATTN_PART,
    /* MoE: the router logits, the selection and its weights, the experts' mid and partial outputs,
     * and for the tiled experts each expert's token list and count */
    MELLUM_S_ROUTER_LOGITS, MELLUM_S_SELECTED, MELLUM_S_WEIGHTS,
    MELLUM_S_MID, MELLUM_S_PARTIAL, MELLUM_S_MOE_LISTS, MELLUM_S_MOE_COUNTS,
    MELLUM_SCRATCH_COUNT
} mellum_scratch_id;

typedef struct {
    mellum_req_kind kind;
    mellum_class cls;
    uint32_t index;
    uint64_t bytes;
} mellum_req;

#define MELLUM_MAX_REQS (6u * DS4_MAX_LAYER + MELLUM_SCRATCH_COUNT + 2u)   /* + logits and the S5 group buffer */

/* Full attention's rows kernels work in slices of 64 rows and split each row's keys
 * into at most 64 parts (QWEN4_ATTN_MAX_SPLITS, ds4_metal.m; mellum_graph_alloc checks
 * the product against ds4_gpu_qwen4_attn_part_floats, which this model-free table cannot call). */
#define MELLUM_ATTN_SLICE_ROWS 64u
#define MELLUM_ATTN_MAX_SPLITS 64u
#define MELLUM_ATTN_ROW_BYTES 64u   /* one rows-table entry (DS4_GPU_QWEN4_ATTN_ROW_BYTES, checked at allocation) */

/* Mellum's chunk, one place: min(requested, window).  Says once per engine
 * that it clamped (ds4_engine_open re-arms the flag, so a /model reopen says it
 * again), so a DS4_QWEN4_PREFILL_CHUNK=4096 or a host --prefill-chunk=4096
 * is visible and not a silent change. */
static bool g_mellum_clamp_said;

static uint32_t mellum_clamp_chunk(uint32_t requested) {
    const uint32_t window = DS4_N_SWA;
    if (requested <= window) return requested;
    if (!g_mellum_clamp_said) {
        g_mellum_clamp_said = true;
        fprintf(stderr,
                "ds4: Mellum prefill chunk %u clamped to %u (the sliding-window ring holds %u rows)\n",
                requested, window, window);
    }
    return window;
}

/* The rows a Mellum session allocates its transients for, and prefills in:
 * the host's chunk, else DS4_QWEN4_PREFILL_CHUNK, else the window; never more
 * than the window or the context. */
static uint32_t mellum_prefill_cap(uint32_t ctx, uint32_t requested) {
    uint32_t chunk = requested;
    if (chunk == 0) {
        const char *env = getenv("DS4_QWEN4_PREFILL_CHUNK");
        const unsigned long v = env && env[0] ? strtoul(env, NULL, 10) : 0ul;
        chunk = v == 0 || v > 65536ul ? DS4_N_SWA : (uint32_t)v;
    }
    chunk = mellum_clamp_chunk(chunk);
    if (chunk > ctx) chunk = ctx;
    return chunk ? chunk : 1u;
}

/* Every GPU tensor of a Mellum session, in allocation order.  Returns the
 * count; out holds MELLUM_MAX_REQS entries. */
/* S5: K for a streamed Mellum graph of cap_tokens rows: 1 (per chunk) unless streaming with K > 1 and
 * chunks of two rows or more; one-row chunks all run on the decode path, so they never group. */
static uint32_t mellum_group_k(uint32_t cap_tokens) {
#if defined(__APPLE__)
    return g_mellum_stream_engine && cap_tokens > 1u ? g_mellum_stream_group_k : 1u;
#else
    (void)cap_tokens;
    return 1u;
#endif
}

/* S5: the rows of the streamed prefill group buffer a Mellum graph of cap_tokens rows holds (0: none). */
static uint32_t mellum_group_rows(uint32_t cap_tokens) {
#if defined(__APPLE__)
    return mellum_group_k(cap_tokens) > 1u ? mellum_group_k(cap_tokens) * cap_tokens : 0u;
#else
    (void)cap_tokens;
    return 0u;
#endif
}

static uint32_t mellum_requests(uint32_t ctx, uint32_t cap_tokens, uint32_t group_rows, mellum_req *out) {
    const uint64_t E = DS4_N_EMBD, T = cap_tokens, window = DS4_N_SWA;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t n_exp = DS4_N_EXPERT, n_used = DS4_N_EXPERT_USED, ff = DS4_N_FF_EXP;
    uint32_t n = 0;
#define MELLUM_ADD(kind_, cls_, index_, bytes_) \
    do { out[n].kind = (kind_); out[n].cls = (cls_); out[n].index = (uint32_t)(index_); \
         out[n].bytes = (bytes_); n++; } while (0)
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const bool ring = ds4_mellum_layer_is_sliding(il);
        const uint64_t rows = ring ? window : ctx;
        MELLUM_ADD(MELLUM_REQ_K, MELLUM_CLASS_KV, il, rows * kv_dim * sizeof(uint16_t));
        MELLUM_ADD(MELLUM_REQ_V, MELLUM_CLASS_KV, il, rows * kv_dim * sizeof(uint16_t));
        if (ring) MELLUM_ADD(MELLUM_REQ_TAG, MELLUM_CLASS_KV, il, window * sizeof(int32_t));
    }
    uint64_t n_full = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) n_full += !ds4_mellum_layer_is_sliding(il);
    const uint64_t slice = T < MELLUM_ATTN_SLICE_ROWS ? T : MELLUM_ATTN_SLICE_ROWS;
#define MELLUM_SCRATCH(id_, cls_, bytes_) MELLUM_ADD(MELLUM_REQ_SCRATCH, cls_, id_, bytes_)
    MELLUM_SCRATCH(MELLUM_S_TOKENS, MELLUM_CLASS_PREFILL, T * sizeof(int32_t));
    MELLUM_SCRATCH(MELLUM_S_HIDDEN, MELLUM_CLASS_PREFILL, T * E * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_ATTN_NORM, MELLUM_CLASS_PREFILL, T * E * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_ATTN_PROJ, MELLUM_CLASS_PREFILL, T * E * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_ATTN_RES, MELLUM_CLASS_PREFILL, T * E * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_FFN_NORM, MELLUM_CLASS_PREFILL, T * E * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_MOE_OUT, MELLUM_CLASS_PREFILL, T * E * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_Q, MELLUM_CLASS_PREFILL, T * q_dim * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_HEADS, MELLUM_CLASS_PREFILL, T * q_dim * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_K, MELLUM_CLASS_PREFILL, T * kv_dim * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_V, MELLUM_CLASS_PREFILL, T * kv_dim * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_K16, MELLUM_CLASS_PREFILL, T * kv_dim * sizeof(uint16_t));
    MELLUM_SCRATCH(MELLUM_S_V16, MELLUM_CLASS_PREFILL, T * kv_dim * sizeof(uint16_t));
    /* region j of the table (j = the j-th full layer) is entries j * T .. j * T + T - 1: each layer of one
     * command buffer needs its own, because the CPU writes the table when the layer is encoded and the GPU
     * reads it when the command buffer runs */
    MELLUM_SCRATCH(MELLUM_S_ATTN_TABLE, MELLUM_CLASS_PREFILL, n_full * T * MELLUM_ATTN_ROW_BYTES);
    MELLUM_SCRATCH(MELLUM_S_ATTN_PART, MELLUM_CLASS_PREFILL,
                   slice * DS4_N_HEAD * MELLUM_ATTN_MAX_SPLITS * (2u + DS4_N_HEAD_DIM) * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_ROUTER_LOGITS, MELLUM_CLASS_MOE, T * n_exp * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_SELECTED, MELLUM_CLASS_MOE, T * n_used * sizeof(int32_t));
    MELLUM_SCRATCH(MELLUM_S_WEIGHTS, MELLUM_CLASS_MOE, T * n_used * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_MID, MELLUM_CLASS_MOE, T * n_used * ff * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_PARTIAL, MELLUM_CLASS_MOE, T * n_used * E * sizeof(float));
    MELLUM_SCRATCH(MELLUM_S_MOE_LISTS, MELLUM_CLASS_MOE, n_exp * T * sizeof(int32_t));
    MELLUM_SCRATCH(MELLUM_S_MOE_COUNTS, MELLUM_CLASS_MOE, n_exp * sizeof(int32_t));
#undef MELLUM_SCRATCH
    MELLUM_ADD(MELLUM_REQ_LOGITS, MELLUM_CLASS_LOGITS, 0, (uint64_t)DS4_N_VOCAB * sizeof(float));
    /* the rewind frontier (mellum_graph_frontier_copy): a copy of every ring, KV state like the rings */
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!ds4_mellum_layer_is_sliding(il)) continue;
        MELLUM_ADD(MELLUM_REQ_FRONTIER_K, MELLUM_CLASS_KV, il, window * kv_dim * sizeof(uint16_t));
        MELLUM_ADD(MELLUM_REQ_FRONTIER_V, MELLUM_CLASS_KV, il, window * kv_dim * sizeof(uint16_t));
        MELLUM_ADD(MELLUM_REQ_FRONTIER_TAG, MELLUM_CLASS_KV, il, window * sizeof(int32_t));
    }
    if (group_rows > cap_tokens) MELLUM_ADD(MELLUM_REQ_GROUP_HIDDEN, MELLUM_CLASS_PREFILL, 0, (uint64_t)group_rows * E * sizeof(float));
#undef MELLUM_ADD
    return n;
}

/* The plan: the table summed by class.  KV state is raw_bytes (the startup
 * line's "KV"), everything else scratch_bytes (its "buffers"); the weights are
 * the caller's.  ctx_size <= 0 plans a one-token context. */
static ds4_context_memory mellum_context_memory(int ctx_size, uint32_t requested_chunk) {
    ds4_context_memory m = {0};
    const uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;
    const uint32_t cap = mellum_prefill_cap(ctx, requested_chunk);
    mellum_req reqs[MELLUM_MAX_REQS];
    const uint32_t n = mellum_requests(ctx, cap, mellum_group_rows(cap), reqs);
    for (uint32_t i = 0; i < n; i++) {
        if (reqs[i].cls == MELLUM_CLASS_KV) m.raw_bytes += reqs[i].bytes;
        else m.scratch_bytes += reqs[i].bytes;
    }
    m.prefill_cap = cap;
    m.raw_cap = ctx;
    m.total_bytes = m.raw_bytes + m.scratch_bytes;
    return m;
}
