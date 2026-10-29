/* Mellum 2.1 recognition, validation and binding, model-free.  Each case writes
 * the sparse GGUF of tests/mellum_fixture.h and opens it through model_open /
 * config_validate_model / weights_bind, or ds4_engine_open.  The validators
 * exit(1) with a message, so a refusal is checked in a forked child whose
 * stderr is read back and must name what it refuses. */
#include "../ds4.c"
#include "mellum_fixture.h"
#include <sys/wait.h>

typedef struct { const char *role, *content; } fx_msg;
typedef struct { const char *name; bool think; int nmsg; const fx_msg *msgs; int nids; const int *ids; } fx_case;
typedef struct { const char *name; bool think; int nids; const int *ids; } fx_prefix;
typedef struct { const char *name; int nmsg; const char *text; int nids; const int *ids; } fx_tool_case;
#include "mellum_chat_ids.inc"
#include "mellum_tool_text.h"
#include "../ds4_prompt_prefix.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

/* Run fn in a child; return its exit status and what it wrote to stderr. */
typedef void (*child_fn)(const char *path);

static int run_child(child_fn fn, const char *path, char *err, size_t cap) {
    int pipefd[2];
    CHECK(pipe(pipefd) == 0);
    fflush(stdout);   /* a child's exit(1) would otherwise flush a copy of our pending output */
    fflush(stderr);
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], 2);
        close(pipefd[1]);
        fn(path);
        _exit(0);
    }
    close(pipefd[1]);
    size_t n = 0;
    ssize_t r;
    while (n + 1 < cap && (r = read(pipefd[0], err + n, cap - 1 - n)) > 0) n += (size_t)r;
    err[n] = '\0';
    close(pipefd[0]);
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

/* general.architecture = mellum names the variant, inside the QWEN4_EXP family. */
static void check_variant(void) {
    char path[512];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    CHECK(ds4_model_is_mellum());
    CHECK(ds4_model_is_qwen4());
    CHECK(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN4_EXP);
    CHECK(DS4_MODEL_VARIANT == DS4_VARIANT_MELLUM);
    model_close(&m);
    unlink(path);

    /* sibling: the Qwen3.8 shapes are not Mellum */
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    CHECK(!ds4_model_is_mellum() && ds4_model_is_qwen4());
    g_ds4_shape = DS4_SHAPE_QWEN4_MINI;
    CHECK(!ds4_model_is_mellum() && ds4_model_is_qwen4());
    printf("ok   variant: mellum is its own variant in the QWEN4_EXP family\n");
}

/* What an open does with the file before the forward pass: validate the
 * keys, then bind and check every tensor of the full model. */
static void child_validate(const char *path) {
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    ds4_weights w;
    weights_bind(&w, &m, false, 0, UINT32_MAX, true, false);
    model_close(&m);
}

/* Refused: exits 1 and the message names `needle`. */
static void expect_refused(mf_mutation mu, const char *needle, const char *label) {
    char path[512], err[4096];
    mellum_fixture_write(mu, path, sizeof(path));
    const int rc = run_child(child_validate, path, err, sizeof(err));
    unlink(path);
    if (rc != 1 || !strstr(err, needle)) {
        fprintf(stderr, "FAIL %s: exit %d, stderr [%s], wanted exit 1 naming '%s'\n", label, rc, err, needle);
        exit(1);
    }
    printf("ok   refused: %-34s names %s\n", label, needle);
}

static void expect_accepted(mf_mutation mu, const char *label) {
    char path[512], err[4096];
    mellum_fixture_write(mu, path, sizeof(path));
    const int rc = run_child(child_validate, path, err, sizeof(err));
    unlink(path);
    if (rc != 0) {
        fprintf(stderr, "FAIL %s: exit %d, stderr [%s]\n", label, rc, err);
        exit(1);
    }
    printf("ok   accepted: %s\n", label);
}

/* The real key set gives the shape the kernels are written for, the 3:1
 * sliding pattern, and resets what a model opened earlier left behind. */
static void check_good_set_in_process(void) {
    char path[512];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    g_ds4_qwen4_ple.n_heads = 7;                     /* stale PLE state from another model */
    g_ds4_compress_ratios[3] = 4;
    g_qwen4_native_ctx = 262144;                     /* stale rope state from a Qwen model */
    g_qwen4_rope_yarn = true;
    g_qwen4_rope_freq[0] = 7.0f;
    g_qwen4_rope_mscale = 2.0f;
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);

    CHECK(DS4_N_LAYER == 28 && DS4_N_EMBD == 2304 && DS4_N_VOCAB == 98304);
    CHECK(DS4_N_HEAD == 32 && DS4_N_HEAD_KV == 4 && DS4_N_HEAD_DIM == 128 && DS4_N_VALUE_DIM == 128);
    CHECK(DS4_N_ROT == 128 && DS4_N_SWA == 1024);
    CHECK(DS4_N_EXPERT == 64 && DS4_N_EXPERT_USED == 8 && DS4_N_FF_EXP == 896);
    CHECK(DS4_N_FF_DENSE == 0);                      /* feed_forward_length (7168) is not a dense FFN */
    CHECK(DS4_N_EXPERT_SHARED == 0 && DS4_N_NEXTN_PREDICT == 0 && DS4_N_HC == 1);
    CHECK(DS4_ROPE_FREQ_BASE == 500000.0f && DS4_ROPE_FREQ_BASE_SWA == 500000.0f);
    CHECK(DS4_ROPE_SCALE_FACTOR == 16.0f && DS4_ROPE_ORIG_CTX == 8192);
    CHECK(DS4_ROPE_YARN_BETA_FAST == 32.0f && DS4_ROPE_YARN_BETA_SLOW == 1.0f);
    CHECK(fabsf(DS4_ROPE_YARN_ATTN_FACTOR - 1.2772589f) < 1.0e-6f);
    CHECK(g_ds4_qwen4_ple.n_heads == 0 && g_ds4_compress_ratios[3] == 0);
    CHECK(g_qwen4_native_ctx == 0 && !g_qwen4_rope_yarn && g_qwen4_rope_freq[0] == 0.0f &&
          g_qwen4_rope_mscale == 1.0f);

    /* layers 0, 1, 2 slide; 3 and 27 are full; 21 of 28 slide */
    CHECK(ds4_mellum_layer_is_sliding(0) && ds4_mellum_layer_is_sliding(1) && ds4_mellum_layer_is_sliding(2));
    CHECK(!ds4_mellum_layer_is_sliding(3) && !ds4_mellum_layer_is_sliding(27));
    uint32_t n_sliding = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) n_sliding += ds4_mellum_layer_is_sliding(il);
    CHECK(n_sliding == 21);
    /* without its guard the GDN predicate is silently true here (interval 0, and
     * arm64 udiv makes x % 0 == x) */
    CHECK(DS4_N_FULL_ATTN_INTERVAL == 0);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        CHECK(!ds4_qwen4_layer_is_linear(il));
        CHECK(!ds4_qwen4_layer_is_ple(il));
        CHECK(!ds4_qwen4_layer_is_nextn(il));
    }
    /* the guards hold even if a shape field the Qwen rules read were set */
    {
        const ds4_shape saved = g_ds4_shape;
        g_ds4_shape.n_full_attn_interval = 4;
        g_ds4_shape.n_ple_layer = 0;
        g_ds4_shape.n_nextn_predict = 1;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            CHECK(!ds4_qwen4_layer_is_linear(il));
            CHECK(!ds4_qwen4_layer_is_ple(il));
            CHECK(!ds4_qwen4_layer_is_nextn(il));
        }
        g_ds4_shape = saved;
    }

    ds4_weights w;
    weights_bind(&w, &m, false, 0, UINT32_MAX, true, false);
    CHECK(w.token_embd && w.output && w.output_norm);
    CHECK(w.output != w.token_embd);                 /* untied head */
    CHECK(!w.output_hc_down && !w.output_hc_up && !w.output_hc_norm && !w.ple_embd);
    CHECK(weights_have_output_head(&w));
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w.layer[il];
        CHECK(l->attn_norm && l->ffn_norm && l->attn_q && l->attn_k && l->attn_v && l->attn_output);
        CHECK(l->attn_q_norm && l->attn_k_norm && l->ffn_gate_inp);
        CHECK(l->ffn_gate_exps && l->ffn_up_exps && l->ffn_down_exps);
        CHECK(!l->ffn_gate_inp_shexp && !l->ffn_gate_shexp && !l->ffn_up_shexp && !l->ffn_down_shexp);
        CHECK(!l->hc_attn_norm && !l->lin_qkv && !l->indexer_q_proj && !l->ple_key);
        CHECK(weights_layer_has_required(l, il));
    }
    model_close(&m);
    unlink(path);
    printf("ok   good key set: shape, sliding pattern, predicates, stale state reset, binding\n");
}

/* The qwen predicates keep their meaning for the models that already use them. */
static void check_qwen_predicates_unchanged(void) {
    g_ds4_shape = DS4_SHAPE_QWEN4_MINI;
    CHECK(!ds4_model_is_mellum());
    CHECK(ds4_qwen4_layer_is_linear(0) && ds4_qwen4_layer_is_linear(1) && ds4_qwen4_layer_is_linear(2));
    CHECK(!ds4_qwen4_layer_is_linear(3) && ds4_qwen4_layer_is_linear(4));
    CHECK(!ds4_qwen4_layer_is_linear(8));            /* the MTP block is attention */
    CHECK(ds4_qwen4_layer_is_nextn(8) && !ds4_qwen4_layer_is_nextn(7));
    CHECK(ds4_qwen4_layer_is_ple(DS4_N_PLE_LAYER));
    CHECK(!ds4_mellum_layer_is_sliding(0) && !ds4_mellum_layer_is_sliding(1));   /* not a Mellum model */
    printf("ok   qwen4exp predicates unchanged\n");
}

/* The open gate: CPU and CUDA are refused for Mellum by name, inspect_only is not. */
static ds4_backend g_open_backend;
static bool g_open_inspect;
static const char *g_open_steer_file;
static float g_open_steer_attn;

static void child_engine_open(const char *path) {
    ds4_engine *e = NULL;
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = path;
    opt.backend = g_open_backend;
    opt.context_size = 4096;
    opt.inspect_only = g_open_inspect;
    opt.directional_steering_file = (char *)g_open_steer_file;
    opt.directional_steering_attn = g_open_steer_attn;
    const int rc = ds4_engine_open(&e, &opt);
    _exit(rc == 0 && e && ds4_engine_is_mellum(e) ? 0 : 7);
}

static void expect_open(ds4_backend be, bool inspect, bool refused, const char *label) {
    char path[512], err[4096];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    g_open_backend = be;
    g_open_inspect = inspect;
    const int rc = run_child(child_engine_open, path, err, sizeof(err));
    unlink(path);
    if (refused ? rc != 7 || !strstr(err, "Mellum 2.1") || !strstr(err, "Metal graph only") : rc != 0) {
        fprintf(stderr, "FAIL %s: exit %d, stderr [%s]\n", label, rc, err);
        exit(1);
    }
    printf("ok   open: %s\n", label);
}

/* Steering allocates a tensor outside mellum_requests, so a Mellum open that asks for it is
 * refused by name.  The CPU backend would also be refused, by a different message, so the
 * message text tells the two gates apart; the siblings are the same open without steering
 * (refused by the backend gate, not the steering one) and an inspect-only open with it. */
static void check_mellum_steering_refused(void) {
    char path[512], err[4096];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    g_open_backend = DS4_BACKEND_CPU;
    g_open_inspect = false;
    g_open_steer_file = "/nonexistent/steer.f32";
    g_open_steer_attn = 1.0f;
    int rc = run_child(child_engine_open, path, err, sizeof(err));
    CHECK(rc == 7 && strstr(err, "directional steering is not supported for Mellum 2.1"));
    g_open_steer_file = NULL;
    g_open_steer_attn = 0.0f;
    rc = run_child(child_engine_open, path, err, sizeof(err));
    CHECK(rc == 7 && strstr(err, "Metal graph only") && !strstr(err, "directional steering"));
    g_open_inspect = true;
    g_open_steer_file = "/nonexistent/steer.f32";
    g_open_steer_attn = 1.0f;
    rc = run_child(child_engine_open, path, err, sizeof(err));
    CHECK(rc == 0);
    g_open_inspect = false;
    g_open_steer_file = NULL;
    g_open_steer_attn = 0.0f;
    unlink(path);
    printf("ok   open: steering refused for Mellum by name; the same open without it reaches the backend gate\n");
}

/* The startup line: Mellum prints the same "GiB planned" line the other models do. */
static void child_print_memory(const char *path) {
    (void)path;
    static ds4_engine e;
    memset(&e, 0, sizeof(e));
    e.backend = DS4_BACKEND_METAL;
    ds4_engine_print_startup_memory(&e, 4096);
}

static void check_memory_plan(void) {
    char path[512], err[4096];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    const ds4_context_memory est = ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, 4096, 0, false);
    CHECK(est.total_bytes > 0 && est.raw_bytes > 0 && est.scratch_bytes > 0);
    CHECK(run_child(child_print_memory, "", err, sizeof(err)) == 0);
    CHECK(strstr(err, "GiB planned") != NULL);
    CHECK(strstr(err, "prefill_cap=1024 raw_kv_rows=4096") != NULL);
    model_close(&m);
    unlink(path);

    /* sibling: a model that always had a plan still prints one */
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    const ds4_context_memory q = ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, 4096, 0, false);
    CHECK(q.total_bytes > 0);
    CHECK(run_child(child_print_memory, "", err, sizeof(err)) == 0);
    CHECK(strstr(err, "GiB planned") != NULL);
    printf("ok   memory plan: Mellum and qwen4exp both print a plan line\n");
}


/* ---- Mellum memory plan ---------------------------------------------------------------
 * Expectations are written out here from the model's dims, not read from the shape the
 * plan reads: E 2304, 32 q heads x 128, 4 kv heads x 128, 64 experts / 8 used, expert
 * width 896, a 1024-slot ring on 21 of 28 layers (full attention on the 7 others), one int32 tag
 * array per ring (K and V rows share it), and for the full layers' attention a rows table of 64 B per
 * row per full layer plus the partials of one slice of at most 64 rows (64 splits of 2 + 128 floats
 * per head: the kernels' own constants, ds4_gpu_qwen4_attn_part_floats, which the live group checks). */
enum { T_E = 2304, T_QDIM = 4096, T_KVDIM = 512, T_EXPERTS = 64, T_USED = 8, T_FF = 896, T_RING = 1024 };

static uint64_t expect_kv(uint64_t ctx) {
    return 7ull * ctx * (2ull * T_KVDIM * 2u) +
           21ull * T_RING * (2ull * T_KVDIM * 2u + sizeof(int32_t));
}

/* bytes per prefill row (activations and MoE transients) */
static uint64_t expect_per_row(void) {
    return 4ull                                        /* the token id */
         + 6ull * T_E * 4u                             /* the six residual-width activations */
         + 2ull * T_QDIM * 4u                          /* q, heads */
         + 2ull * T_KVDIM * 4u + 2ull * T_KVDIM * 2u   /* k, v f32; staged k, v f16 */
         + (uint64_t)T_EXPERTS * 4u                    /* router logits */
         + 2ull * T_USED * 4u                          /* selected, weights */
         + (uint64_t)T_USED * T_FF * 4u                /* mid */
         + (uint64_t)T_USED * T_E * 4u                 /* per-expert partial outputs */
         + (uint64_t)T_EXPERTS * 4u                    /* the tiled experts' token lists: a slot per expert per row */
         + 7ull * 64u;                                 /* full attention's rows table: 64 B a row, 7 full layers */
}

/* the partials of one slice of min(cap, 64) rows of 32 heads: 64 splits, 2 + 128 floats each */
static uint64_t expect_attn_part(uint64_t cap) {
    return (cap < 64u ? cap : 64u) * 32ull * 64u * (2u + 128u) * 4u;
}

static uint64_t expect_fixed(uint64_t cap) {
    /* the logits row; the shared attention partials; the tiled experts' token counts */
    return 98304ull * 4u + expect_attn_part(cap) + (uint64_t)T_EXPERTS * 4u;
}

static uint64_t sum_requests(uint32_t ctx, uint32_t cap, uint32_t *n_out) {
    mellum_req reqs[MELLUM_MAX_REQS];
    const uint32_t n = mellum_requests(ctx, cap, 0u, reqs);
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; i++) total += reqs[i].bytes;
    if (n_out) *n_out = n;
    return total;
}

static void check_mellum_plan_sums(void) {
    static const uint32_t ctxs[] = {4096, 20000, 65536, 131072};
    for (size_t c = 0; c < sizeof(ctxs) / sizeof(ctxs[0]); c++) {
        for (int chunk_one = 0; chunk_one < 2; chunk_one++) {
            const uint32_t ctx = ctxs[c], cap = chunk_one ? 1u : 1024u;
            const ds4_context_memory m =
                ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, (int)ctx, chunk_one ? 1u : 0u, false);
            /* the plan is exactly what the allocator will request */
            CHECK(m.total_bytes == sum_requests(ctx, cap, NULL));
            CHECK(m.prefill_cap == cap && m.raw_cap == ctx && m.compressed_bytes == 0);
            /* and exactly what the dims say */
            CHECK(m.raw_bytes == expect_kv(ctx));
            CHECK(m.scratch_bytes == cap * expect_per_row() + expect_fixed(cap));
            CHECK(m.total_bytes == m.raw_bytes + m.scratch_bytes);
        }
    }
    /* KV at 20,000: 7 full layers x 20,000 rows x 2,048 B, plus 21 rings of 1,024 rows x 2,048 B and a 4 B tag */
    const ds4_context_memory m20 = ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, 20000, 0, false);
    CHECK(m20.raw_bytes == 7ull * 20000 * 2048 + 21ull * 1024 * (2048 + 4));
    CHECK(m20.raw_bytes == 330846208ull);
    printf("ok   plan == sum of the allocator's requests == the dims, at 4 contexts x chunk {1024, 1}\n");
}

/* The prefill scratch scales with cap_tokens and is part of the create-time table: the
 * allocator walks the same table when the session is created; nothing is allocated at a sync. */
static void check_mellum_plan_scales_with_chunk(void) {
    const ds4_context_memory a = ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, 20000, 1, false);
    const ds4_context_memory b = ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, 20000, 512, false);
    const ds4_context_memory c = ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, 20000, 1024, false);
    CHECK(a.raw_bytes == b.raw_bytes && b.raw_bytes == c.raw_bytes);      /* KV does not depend on the chunk */
    CHECK(b.scratch_bytes - a.scratch_bytes == 511ull * expect_per_row() + expect_attn_part(512) - expect_attn_part(1));
    CHECK(c.scratch_bytes - b.scratch_bytes == 512ull * expect_per_row());   /* both past 64 rows: the same partials */
    /* a context shorter than the chunk plans for the context */
    const ds4_context_memory s = ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, 300, 0, false);
    CHECK(s.prefill_cap == 300 && s.scratch_bytes == 300 * expect_per_row() + expect_fixed(300));

    mellum_req reqs[MELLUM_MAX_REQS];
    const uint32_t n = mellum_requests(20000, 1024, 0u, reqs);
    uint32_t k = 0, v = 0, tags = 0, scratch = 0, logits = 0;
    uint64_t moe = 0;
    for (uint32_t i = 0; i < n; i++) {
        switch (reqs[i].kind) {
        case MELLUM_REQ_K: k++; break;
        case MELLUM_REQ_V: v++; break;
        case MELLUM_REQ_TAG: tags++; CHECK(reqs[i].bytes == T_RING * 4u); break;
        case MELLUM_REQ_SCRATCH: scratch++; break;
        case MELLUM_REQ_LOGITS: logits++; break;
        }
        if (reqs[i].cls == MELLUM_CLASS_MOE) moe += reqs[i].bytes;
    }
    CHECK(k == 28 && v == 28 && tags == 21 && scratch == MELLUM_SCRATCH_COUNT && logits == 1);
    /* the per-expert partial outputs are in the table, and counted */
    CHECK(moe > 72ull * 1024 * 1024);
    for (uint32_t i = 0; i < n; i++) {
        if (reqs[i].kind == MELLUM_REQ_K && ds4_mellum_layer_is_sliding(reqs[i].index)) {
            CHECK(reqs[i].bytes == (uint64_t)T_RING * T_KVDIM * 2u);       /* a ring */
        } else if (reqs[i].kind == MELLUM_REQ_K) {
            CHECK(reqs[i].index % 4 == 3 && reqs[i].bytes == 20000ull * T_KVDIM * 2u);   /* full */
        }
    }
    printf("ok   plan scales with cap_tokens; 28 K + 28 V + 21 tags + %u scratch + logits\n", (unsigned)MELLUM_SCRATCH_COUNT);
}

/* The chunk clamp.  DS4_QWEN4_PREFILL_CHUNK reaches the plan through qwen4_prefill_chunk_tokens
 * in a GPU build and through mellum_prefill_cap here; the host's chunk reaches the estimate as
 * its argument and the engine as e->prefill_chunk. */
static const char *g_chunk_env;
static uint32_t g_chunk_arg;

static int count_str(const char *hay, const char *needle) {
    int n = 0;
    for (const char *p = hay; (p = strstr(p, needle)) != NULL; p += strlen(needle)) n++;
    return n;
}

static void child_plan_chunk(const char *path) {
    (void)path;
    if (g_chunk_env) setenv("DS4_QWEN4_PREFILL_CHUNK", g_chunk_env, 1);
    const ds4_context_memory m = ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, 20000, g_chunk_arg, false);
    /* a second and a third reader: the clamp message must still be one line */
    (void)ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL, 20000, g_chunk_arg, false);
    (void)mellum_prefill_cap(20000, g_chunk_arg);
    _exit(m.prefill_cap == 1024 ? 40 : (m.prefill_cap == 512 ? 41 : 42));
}

static int run_plan_chunk(const char *env, uint32_t arg, char *err, size_t cap) {
    g_chunk_env = env;
    g_chunk_arg = arg;
    return run_child(child_plan_chunk, "", err, cap);
}

static void child_open_chunk(const char *path) {
    ds4_engine *e = NULL;
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = path;
    opt.backend = DS4_BACKEND_CPU;
    opt.context_size = 4096;
    opt.inspect_only = true;
    opt.prefill_chunk = g_chunk_arg;
    if (ds4_engine_open(&e, &opt) != 0) _exit(7);
    _exit(ds4_engine_prefill_chunk(e) == 1024 ? 40 : (ds4_engine_prefill_chunk(e) == 512 ? 41 : 42));
}

static void check_mellum_chunk_clamp(void) {
    char err[4096];
    const char *say = "Mellum prefill chunk 4096 clamped to 1024";

    /* the env value: 4096 is used as 1024, and one line says so */
    CHECK(run_plan_chunk("4096", 0, err, sizeof(err)) == 40);
    CHECK(count_str(err, say) == 1);
    /* the host-passed value, which the estimate and session create read directly */
    CHECK(run_plan_chunk(NULL, 4096, err, sizeof(err)) == 40);
    CHECK(count_str(err, say) == 1);
    /* the largest value the env accepts */
    CHECK(run_plan_chunk("65536", 0, err, sizeof(err)) == 40);
    CHECK(count_str(err, "clamped to 1024") == 1);
    /* sibling success: a chunk at or under the window is used as asked and says nothing */
    CHECK(run_plan_chunk("512", 0, err, sizeof(err)) == 41);
    CHECK(strstr(err, "clamped") == NULL);
    CHECK(run_plan_chunk(NULL, 512, err, sizeof(err)) == 41);
    CHECK(strstr(err, "clamped") == NULL);
    CHECK(run_plan_chunk("1024", 0, err, sizeof(err)) == 40);
    CHECK(strstr(err, "clamped") == NULL);
    CHECK(run_plan_chunk(NULL, 0, err, sizeof(err)) == 40);        /* the default */
    CHECK(strstr(err, "clamped") == NULL);

    /* engine open: the chunk the host passes is clamped where it is stored */
    char path[512];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    g_chunk_arg = 4096;
    CHECK(run_child(child_open_chunk, path, err, sizeof(err)) == 40);
    CHECK(count_str(err, say) == 1);
    g_chunk_arg = 512;
    CHECK(run_child(child_open_chunk, path, err, sizeof(err)) == 41);
    CHECK(strstr(err, "clamped") == NULL);
    unlink(path);
    printf("ok   chunk 4096 (env, host argument, engine open) used as 1024 with one line; 512 and 1024 untouched\n");
}

/* The clamp line is per engine: a /model reopen in the same process says it again. */
static void child_open_twice(const char *path) {
    for (int i = 0; i < 2; i++) {
        ds4_engine *e = NULL;
        ds4_engine_options opt;
        memset(&opt, 0, sizeof(opt));
        opt.model_path = path;
        opt.backend = DS4_BACKEND_CPU;
        opt.context_size = 4096;
        opt.inspect_only = true;
        opt.prefill_chunk = 4096;
        if (ds4_engine_open(&e, &opt) != 0) _exit(7);
        ds4_engine_close(e);
    }
    _exit(40);
}

static void check_mellum_clamp_per_engine(void) {
    char path[512], err[8192];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    CHECK(run_child(child_open_twice, path, err, sizeof(err)) == 40);
    CHECK(count_str(err, "Mellum prefill chunk 4096 clamped to 1024") == 2);
    unlink(path);
    printf("ok   the clamp line is per engine (two opens, two lines)\n");
}

/* Full attention's two scratch buffers, from the dims and not from the table's own formula.
 * The rows table has one region per full layer, because the CPU writes
 * the table when a layer is encoded and the GPU reads it when the command buffer runs, so two full layers of
 * one command buffer cannot share entries; the partials are shared (each layer's kernels finish with them). */
static void check_mellum_attention_scratch(void) {
    static const uint32_t caps[] = {1, 63, 64, 100, 1024};
    uint32_t n_full = 0;
    for (uint32_t il = 0; il < 28; il++) n_full += il % 4 == 3;
    CHECK(n_full == 7);
    for (size_t c = 0; c < sizeof(caps) / sizeof(caps[0]); c++) {
        mellum_req reqs[MELLUM_MAX_REQS];
        const uint32_t n = mellum_requests(20000, caps[c], 0u, reqs);
        int tables = 0, parts = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (reqs[i].kind != MELLUM_REQ_SCRATCH) continue;
            if (reqs[i].index == MELLUM_S_ATTN_TABLE) {
                tables++;
                CHECK(reqs[i].bytes == (uint64_t)n_full * caps[c] * 64u && reqs[i].cls == MELLUM_CLASS_PREFILL);
            }
            if (reqs[i].index == MELLUM_S_ATTN_PART) {
                parts++;
                CHECK(reqs[i].bytes == expect_attn_part(caps[c]) && reqs[i].cls == MELLUM_CLASS_PREFILL);
            }
        }
        CHECK(tables == 1 && parts == 1);
    }
    /* at the default chunk: 7 x 1,024 x 64 B = 448 KiB, and 65.0 MiB of partials */
    mellum_req reqs[MELLUM_MAX_REQS];
    const uint32_t n = mellum_requests(20000, 1024, 0u, reqs);
    for (uint32_t i = 0; i < n; i++) {
        if (reqs[i].kind == MELLUM_REQ_SCRATCH && reqs[i].index == MELLUM_S_ATTN_TABLE) CHECK(reqs[i].bytes == 458752ull);
        if (reqs[i].kind == MELLUM_REQ_SCRATCH && reqs[i].index == MELLUM_S_ATTN_PART) CHECK(reqs[i].bytes == 68157440ull);
    }
    printf("ok   full attention scratch: rows table 7 x cap x 64 B, partials min(cap, 64) rows (448 KiB and 65.0 MiB at 1,024)\n");
}

static void check_mellum_plan(void) {
    char path[512];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    CHECK(ds4_model_is_mellum() && DS4_N_VOCAB == 98304 && DS4_N_SWA == T_RING);
    check_mellum_plan_sums();
    check_mellum_plan_scales_with_chunk();
    model_close(&m);
    unlink(path);
    check_mellum_chunk_clamp();
    check_mellum_clamp_per_engine();
    check_mellum_attention_scratch();
}

/* ---- stop classes ---------------------------------------------------------------------
 * 2.1's special ids, as its tokenizer_config.json spells and types them: <|im_end|> (28) ends
 * a turn (EOS), <|endoftext|> (0) ends a document (EOS-alt), the other CONTROL tokens are
 * protocol except the corpus markers 3-20, which are content in a reply; <think>, </think>
 * and the tool tags (USER_DEFINED) are none.  Unnamed CONTROL tokens fail closed. */
static const int EOS_IDS[] = {28};
static const int EOS_ALT_IDS[] = {0};
static const int PROTOCOL_IDS[] = {1, 2, 21, 22, 25, 26, 27, 33, 34};   /* 33, 34: reserved, fail-closed */
static const int MARKER_IDS[] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
static const int NONE_IDS[] = {23, 24, 29, 30, 31, 32, 35};
/* ds4_token_is_stop is true exactly here */
static const int STOP_IDS[] = {0, 1, 2, 21, 22, 25, 26, 27, 28, 33, 34};

#define N_OF(a) (sizeof(a) / sizeof((a)[0]))

static bool in_list(const int *l, size_t n, int id) {
    for (size_t i = 0; i < n; i++) if (l[i] == id) return true;
    return false;
}

static void child_vocab_load(const char *path) {
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    ds4_vocab v;
    vocab_load(&v, &m);
}

static void expect_vocab_refused(mf_mutation mu, const char *needle, const char *label) {
    char path[512], err[4096];
    mellum_fixture_write(mu, path, sizeof(path));
    const int rc = run_child(child_vocab_load, path, err, sizeof(err));
    unlink(path);
    if (rc != 1 || !strstr(err, needle)) {
        fprintf(stderr, "FAIL %s: exit %d, stderr [%s], wanted exit 1 naming '%s'\n", label, rc, err, needle);
        exit(1);
    }
    printf("ok   vocab refused: %-30s names %s\n", label, needle);
}

/* The classes of the engine whose vocabulary is loaded (the shape is Mellum's): the literal
 * sets, every id, and both think modes. */
static void check_mellum_stop_classes(ds4_engine *e, const char *what) {
    CHECK(ds4_model_is_mellum());
    for (size_t i = 0; i < N_OF(EOS_IDS); i++) CHECK(ds4_token_stop_class(e, EOS_IDS[i]) == DS4_STOP_EOS);
    for (size_t i = 0; i < N_OF(EOS_ALT_IDS); i++) CHECK(ds4_token_stop_class(e, EOS_ALT_IDS[i]) == DS4_STOP_EOS_ALT);
    for (size_t i = 0; i < N_OF(PROTOCOL_IDS); i++) CHECK(ds4_token_stop_class(e, PROTOCOL_IDS[i]) == DS4_STOP_PROTOCOL);
    for (size_t i = 0; i < N_OF(MARKER_IDS); i++) CHECK(ds4_token_stop_class(e, MARKER_IDS[i]) == DS4_STOP_MARKER);
    for (size_t i = 0; i < N_OF(NONE_IDS); i++) CHECK(ds4_token_stop_class(e, NONE_IDS[i]) == DS4_STOP_NONE);
    for (int id = 0; id < e->vocab.n_vocab; id++) {
        const bool want = in_list(STOP_IDS, N_OF(STOP_IDS), id);
        if (ds4_token_is_stop(e, id) != want) {
            fprintf(stderr, "FAIL %s: id %d stop=%d, the literal set says %d\n", what, id,
                    ds4_token_is_stop(e, id), want);
            exit(1);
        }
        /* think mode sits on top of the class lookup: ids 23 and 24 (<think>, </think>) are stops
         * under DS4_THINK_NONE only; every other id follows the class in both modes */
        const bool think_id = id == 23 || id == 24;
        if (ds4_token_is_stop_for_think_mode(e, id, DS4_THINK_HIGH) != want ||
            ds4_token_is_stop_for_think_mode(e, id, DS4_THINK_NONE) != (want || think_id)) {
            fprintf(stderr, "FAIL %s: id %d think-mode stops high=%d none=%d, wanted %d and %d\n", what, id,
                    ds4_token_is_stop_for_think_mode(e, id, DS4_THINK_HIGH),
                    ds4_token_is_stop_for_think_mode(e, id, DS4_THINK_NONE), want, want || think_id);
            exit(1);
        }
    }
    CHECK(ds4_token_stop_class(e, -1) == DS4_STOP_NONE && ds4_token_stop_class(e, e->vocab.n_vocab) == DS4_STOP_NONE);
    CHECK(!ds4_token_is_stop(e, -1) && !ds4_token_is_stop(e, e->vocab.n_vocab));
    printf("ok   %s: stop classes and stops (literal sets, every id, both think modes)\n", what);
}

static void check_mellum_stops_tiny(void) {
    char path[512];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    static ds4_engine e;
    memset(&e, 0, sizeof(e));
    vocab_load(&e.vocab, &m);
    check_mellum_stop_classes(&e, "Mellum (tiny vocabulary)");
    vocab_free(&e.vocab);
    model_close(&m);
    unlink(path);
}

static void check_mellum_unnamed_control_is_protocol(void) {
    char path[512];
    mellum_fixture_write(MF_EXTRA_CONTROL, path, sizeof(path));
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    static ds4_engine e;
    memset(&e, 0, sizeof(e));
    vocab_load(&e.vocab, &m);
    CHECK(e.vocab.n_vocab == 292);
    CHECK(ds4_token_stop_class(&e, 291) == DS4_STOP_PROTOCOL && ds4_token_is_stop(&e, 291));
    /* sibling: the named markers beside it stay content */
    CHECK(ds4_token_stop_class(&e, 7) == DS4_STOP_MARKER && !ds4_token_is_stop(&e, 7));
    CHECK(ds4_token_stop_class(&e, 290) == DS4_STOP_NONE && !ds4_token_is_stop(&e, 290));
    vocab_free(&e.vocab);
    model_close(&m);
    unlink(path);
    printf("ok   Mellum: an unnamed CONTROL token (<zzz>) is protocol, the named markers are not\n");
}

/* Flash Next's classes, model-free on the tiny vocabulary read as a Qwen model: eos and
 * <|endoftext|> stop, nothing else does (the truth table before the classes), <think> and
 * </think> are class none. */
static void check_qwen_stop_classes_tiny(void) {
    char path[512];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    CHECK(!ds4_model_is_mellum() && ds4_model_is_qwen4());
    static ds4_engine e;
    memset(&e, 0, sizeof(e));
    vocab_load(&e.vocab, &m);
    const int eos = vocab_lookup(&e.vocab, "<|im_end|>"), alt = vocab_lookup(&e.vocab, "<|endoftext|>");
    const int ts = vocab_lookup(&e.vocab, "<think>"), te = vocab_lookup(&e.vocab, "</think>");
    CHECK(eos == 28 && alt == 0 && ts == 23 && te == 24);
    CHECK(ds4_token_stop_class(&e, eos) == DS4_STOP_EOS);
    CHECK(ds4_token_stop_class(&e, alt) == DS4_STOP_EOS_ALT);
    CHECK(ds4_token_stop_class(&e, ts) == DS4_STOP_NONE && ds4_token_stop_class(&e, te) == DS4_STOP_NONE);
    for (int id = 0; id < e.vocab.n_vocab; id++) {
        const bool before = id == eos || id == alt;
        CHECK(ds4_token_is_stop(&e, id) == before);
        CHECK(ds4_token_is_stop_for_think_mode(&e, id, DS4_THINK_HIGH) == before);
        CHECK(ds4_token_is_stop_for_think_mode(&e, id, DS4_THINK_NONE) == (before || id == ts || id == te));
        if (id != eos && id != alt) CHECK(ds4_token_stop_class(&e, id) == DS4_STOP_NONE);
    }
    vocab_free(&e.vocab);
    model_close(&m);
    unlink(path);
    printf("ok   qwen4exp (tiny vocabulary): eos EOS, <|endoftext|> EOS-alt, <think> none, is_stop unchanged\n");
}

/* ---- chat prefix ----------------------------------------------------------------------- */
static void expect_ids(const char *label, const ds4_tokens *got, const int *want, int n) {
    bool ok = got->len == n;
    for (int i = 0; ok && i < n; i++) ok = got->v[i] == want[i];
    if (!ok) {
        fprintf(stderr, "FAIL %s: got %d ids [", label, got->len);
        for (int i = 0; i < got->len; i++) fprintf(stderr, "%s%d", i ? " " : "", got->v[i]);
        fprintf(stderr, "], want %d [", n);
        for (int i = 0; i < n; i++) fprintf(stderr, "%s%d", i ? " " : "", want[i]);
        fprintf(stderr, "]\n");
        exit(1);
    }
}

static int count_id(const ds4_tokens *t, int id) {
    int n = 0;
    for (int i = 0; i < t->len; i++) n += t->v[i] == id;
    return n;
}

/* Push the byte-level ids of an ASCII string: in the tiny vocabulary byte c is id 35 + c. */
static void push_bytes(ds4_tokens *t, const char *s) {
    for (; *s; s++) ds4_tokens_push(t, 35 + (unsigned char)*s);
}

/* Model-free: 2.1's template opens the assistant turn with nothing after the tag when
 * thinking is on (the model writes <think> itself) and with an empty think block when off;
 * the system turn has no reasoning-effort instruction. */
static void check_mellum_chat_tiny(void) {
    char path[512];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    static ds4_engine e;
    memset(&e, 0, sizeof(e));
    vocab_load(&e.vocab, &m);

    ds4_tokens want = {0}, got = {0};
    ds4_tokens_push(&want, 27);
    push_bytes(&want, "assistant\n");
    ds4_chat_append_assistant_prefix(&e, &got, DS4_THINK_HIGH);
    expect_ids("tiny: think-on prefix", &got, want.v, want.len);
    CHECK(count_id(&got, 23) == 0);
    want.len = got.len = 0;
    ds4_tokens_push(&want, 27);
    push_bytes(&want, "assistant\n");
    ds4_tokens_push(&want, 23); push_bytes(&want, "\n\n"); ds4_tokens_push(&want, 24); push_bytes(&want, "\n\n");
    ds4_chat_append_assistant_prefix(&e, &got, DS4_THINK_NONE);
    expect_ids("tiny: think-off prefix", &got, want.v, want.len);
    /* the CLI -p path: no system turn when none is given, whatever the think mode
     * (Qwen's adds a reasoning-effort system turn here) */
    for (int high = 0; high < 2; high++) {
        want.len = got.len = 0;
        ds4_encode_chat_prompt(&e, NULL, "hi", high ? DS4_THINK_HIGH : DS4_THINK_LOW, &got);
        ds4_tokens_push(&want, 27); push_bytes(&want, "user\nhi"); ds4_tokens_push(&want, 28); push_bytes(&want, "\n");
        ds4_tokens_push(&want, 27); push_bytes(&want, "assistant\n");
        expect_ids("tiny: encode_chat_prompt think-on", &got, want.v, want.len);
    }
    /* a tool message is a user turn whose <tool_response> wrapper is tokens 31 and 32, as the
     * template's text encodes (tests/mellum_chat_ids.inc pins it on the real vocabulary) */
    want.len = got.len = 0;
    ds4_chat_append_message(&e, &got, "tool", "42 files");
    ds4_tokens_push(&want, 27); push_bytes(&want, "user\n"); ds4_tokens_push(&want, 31);
    push_bytes(&want, "\n42 files\n"); ds4_tokens_push(&want, 32); ds4_tokens_push(&want, 28);
    push_bytes(&want, "\n");
    expect_ids("tiny: tool message", &got, want.v, want.len);
    {   /* the agent's text-only observation path renders the same turn */
        const char *parts[] = {"42 files"};
        char err[128] = "";
        got.len = 0;
        CHECK(ds4_chat_append_multimodal_message(&e, &got, "tool", parts, NULL, 0, NULL, err, sizeof(err)) == 1);
        expect_ids("tiny: tool message, multimodal path", &got, want.v, want.len);
    }
    /* the system turn's text is the template's: the tools block's <tool_call> is token 29 */
    got.len = 0;
    ds4_chat_append_message(&e, &got, "system", "call <tool_call> here");
    CHECK(count_id(&got, 29) == 1);
    {   /* the CLI's -sys path renders the system turn the same way */
        ds4_tokens each = {0};
        ds4_chat_append_message(&e, &each, "system", "call <tool_call> here");
        ds4_chat_append_message(&e, &each, "user", "hi");
        ds4_chat_append_assistant_prefix(&e, &each, DS4_THINK_HIGH);
        got.len = 0;
        ds4_encode_chat_prompt(&e, "call <tool_call> here", "hi", DS4_THINK_HIGH, &got);
        expect_ids("tiny: encode_chat_prompt with a system prompt", &got, each.v, each.len);
        ds4_tokens_free(&each);
    }
    {   /* ds4_chat_append_tool_results: one result is that same turn; no results, or a NULL
         * one, is refused and appends nothing */
        const char *one[] = {"42 files"}, *holed[] = {"42 files", NULL};
        got.len = 0;
        CHECK(ds4_chat_append_tool_results(&e, &got, one, 1) == 1);
        expect_ids("tiny: one tool result", &got, want.v, want.len);
        got.len = 0;
        CHECK(ds4_chat_append_tool_results(&e, &got, one, 0) == 0 && got.len == 0);
        CHECK(ds4_chat_append_tool_results(&e, &got, holed, 2) == 0 && got.len == 0);
        CHECK(ds4_chat_append_tool_results(&e, &got, NULL, 1) == 0 && got.len == 0);
    }
    {   /* two results are one user turn, a <tool_response> block each */
        const char *two[] = {"42 files", "7 dirs"};
        want.len = got.len = 0;
        ds4_tokens_push(&want, 27); push_bytes(&want, "user\n"); ds4_tokens_push(&want, 31);
        push_bytes(&want, "\n42 files\n"); ds4_tokens_push(&want, 32); push_bytes(&want, "\n");
        ds4_tokens_push(&want, 31); push_bytes(&want, "\n7 dirs\n"); ds4_tokens_push(&want, 32);
        ds4_tokens_push(&want, 28); push_bytes(&want, "\n");
        CHECK(ds4_chat_append_tool_results(&e, &got, two, 2) == 1);
        expect_ids("tiny: two tool results", &got, want.v, want.len);
    }
    /* an imported history (a prompt-prefix file): ds4_chat_append_message already closes each
     * turn, so the assistant turn ends with one <|im_end|>, as the template writes it */
    ds4_prompt_prefix_turn turns[] = {{DS4_PROMPT_PREFIX_USER, "hi"}, {DS4_PROMPT_PREFIX_ASSISTANT, "ok"}};
    const ds4_prompt_prefix history = {turns, 2};
    want.len = got.len = 0;
    ds4_tokens_push(&want, 27); push_bytes(&want, "user\nhi"); ds4_tokens_push(&want, 28); push_bytes(&want, "\n");
    ds4_tokens_push(&want, 27); push_bytes(&want, "assistant\nok"); ds4_tokens_push(&want, 28); push_bytes(&want, "\n");
    ds4_prompt_prefix_append(&e, &got, &history);
    expect_ids("tiny: imported history", &got, want.v, want.len);
    /* its reasoning is dropped, as the template drops it from every assistant turn before the latest
     * user message, which a prefix file's turns all are */
    turns[1].content = "<think>\nwhy\n</think>\n\nok";
    got.len = 0;
    ds4_prompt_prefix_append(&e, &got, &history);
    expect_ids("tiny: imported history, reasoning dropped", &got, want.v, want.len);
    turns[1].content = "ok";
    /* DeepSeek's max-effort preamble is not Mellum's: its template has no effort text, so max thinking
     * prefixes nothing */
    got.len = 0;
    ds4_chat_append_think_prefix(&e, &got, DS4_THINK_MAX);
    ds4_chat_append_max_effort_prefix(&e, &got);
    CHECK(got.len == 0);
    /* sibling: Flash Next keeps its own prefix (<think> and a newline when thinking) */
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    want.len = got.len = 0;
    ds4_tokens_push(&want, 27); push_bytes(&want, "assistant\n"); ds4_tokens_push(&want, 23); push_bytes(&want, "\n");
    ds4_chat_append_assistant_prefix(&e, &got, DS4_THINK_HIGH);
    expect_ids("tiny: Flash Next think-on prefix", &got, want.v, want.len);
    /* sibling: Flash Next's system text stays plain */
    got.len = 0;
    ds4_chat_append_message(&e, &got, "system", "call <tool_call> here");
    CHECK(count_id(&got, 29) == 0);
    /* sibling: Flash Next's tool message keeps its wrapper as plain text */
    got.len = 0;
    ds4_chat_append_message(&e, &got, "tool", "42 files");
    CHECK(count_id(&got, 31) == 0 && count_id(&got, 32) == 0);
    {   /* and two results are two tool messages, as its template renders them */
        const char *two[] = {"42 files", "7 dirs"};
        ds4_tokens each = {0};
        ds4_chat_append_message(&e, &each, "tool", two[0]);
        ds4_chat_append_message(&e, &each, "tool", two[1]);
        got.len = 0;
        CHECK(ds4_chat_append_tool_results(&e, &got, two, 2) == 1);
        expect_ids("tiny: Flash Next, two tool results", &got, each.v, each.len);
        ds4_tokens_free(&each);
    }
    /* sibling: Flash Next's imported assistant turn is unchanged, its EOS still pushed after the
     * closed turn (MELLUM.md records the double close as a possible upstream issue) */
    got.len = 0;
    ds4_prompt_prefix_append(&e, &got, &history);
    CHECK(count_id(&got, 28) == 3 && got.v[got.len - 1] == 28);
    /* sibling: Flash Next's max thinking still pushes the preamble it pushed before */
    got.len = 0;
    ds4_chat_append_think_prefix(&e, &got, DS4_THINK_MAX);
    CHECK(got.len > 0);
    ds4_tokens_free(&want); ds4_tokens_free(&got);
    vocab_free(&e.vocab);
    model_close(&m);
    unlink(path);
    printf("ok   chat (tiny vocabulary): prefix arms, no effort turn, tool turn, imported history; Flash Next keeps "
           "its prefix, tool text and imported history\n");
}

static void text_cat(char **buf, size_t *len, const char *s) {
    size_t k = strlen(s);
    *buf = realloc(*buf, *len + k + 1);
    CHECK(*buf != NULL);
    memcpy(*buf + *len, s, k + 1);
    *len += k;
}

/* The tool-bearing shapes: Pi's four tools in the system turn, an assistant turn with reasoning
 * and one call, tool results.  Two checks per shape: the text mellum_tool_text.h builds equals
 * what jinja2 rendered, and the engine's renderer gives the ids the tokenizer gives that text. */
static void check_mellum_tool_renders(ds4_engine *e) {
    for (size_t i = 0; i < N_OF(fx_tool_cases); i++) {
        const fx_tool_case *tc = &fx_tool_cases[i];
        char *sys = mtt_system_text(fx_tool_system);
        char *asst = tc->nmsg >= 3 ? mtt_assistant_text(fx_tool_reasoning, fx_tool_call_name, fx_tool_call_args) : NULL;
        char *text = NULL;
        size_t tlen = 0;
        text_cat(&text, &tlen, "<|im_start|>system\n");
        text_cat(&text, &tlen, sys);
        text_cat(&text, &tlen, "<|im_end|>\n<|im_start|>user\n");
        text_cat(&text, &tlen, fx_tool_user);
        text_cat(&text, &tlen, "<|im_end|>\n");
        if (asst) {
            text_cat(&text, &tlen, "<|im_start|>assistant\n");
            text_cat(&text, &tlen, asst);
            text_cat(&text, &tlen, "<|im_end|>\n");
        }
        if (tc->nmsg >= 4) {
            text_cat(&text, &tlen, "<|im_start|>user\n<tool_response>\n");
            text_cat(&text, &tlen, fx_tool_result);
            text_cat(&text, &tlen, "\n</tool_response>");
            if (tc->nmsg >= 5) {
                text_cat(&text, &tlen, "\n<tool_response>\n");
                text_cat(&text, &tlen, fx_tool_result_2);
                text_cat(&text, &tlen, "\n</tool_response>");
            }
            text_cat(&text, &tlen, "<|im_end|>\n");
        }
        text_cat(&text, &tlen, "<|im_start|>assistant\n");
        if (strcmp(text, tc->text) != 0) {
            size_t k = 0;
            while (text[k] && text[k] == tc->text[k]) k++;
            fprintf(stderr, "FAIL %s: mellum_tool_text.h's text differs from jinja2's at byte %zu\n", tc->name, k);
            exit(1);
        }
        ds4_tokens got = {0};
        ds4_chat_append_message(e, &got, "system", sys);
        ds4_chat_append_message(e, &got, "user", fx_tool_user);
        if (asst) ds4_chat_append_message(e, &got, "assistant", asst);
        const char *results[] = {fx_tool_result, fx_tool_result_2};
        if (tc->nmsg >= 4) CHECK(ds4_chat_append_tool_results(e, &got, results, (size_t)tc->nmsg - 3) == 1);
        ds4_chat_append_assistant_prefix(e, &got, DS4_THINK_HIGH);
        expect_ids(tc->name, &got, tc->ids, tc->nids);
        ds4_tokens_free(&got);
        free(text); free(asst); free(sys);
    }
}

/* The real GGUF: the renderer against ids from 2.1's own template (tests/gen_mellum_chat_ids.py). */
static void check_mellum_chat_model(const char *model_path) {
    ds4_model m;
    model_open(&m, model_path, false, false);
    config_validate_model(&m);
    static ds4_engine e;
    memset(&e, 0, sizeof(e));
    vocab_load(&e.vocab, &m);
    CHECK(e.vocab.im_start_id == 27 && e.vocab.im_end_id == 28);
    for (size_t i = 0; i < N_OF(fx_prefixes); i++) {
        ds4_tokens got = {0};
        ds4_chat_append_assistant_prefix(&e, &got, fx_prefixes[i].think ? DS4_THINK_HIGH : DS4_THINK_NONE);
        expect_ids(fx_prefixes[i].name, &got, fx_prefixes[i].ids, fx_prefixes[i].nids);
        ds4_tokens_free(&got);
    }
    for (size_t i = 0; i < N_OF(fx_cases); i++) {
        const fx_case *fc = &fx_cases[i];
        ds4_tokens got = {0};
        for (int j = 0; j < fc->nmsg; j++) {
            if (strcmp(fc->msgs[j].role, "tool")) {
                ds4_chat_append_message(&e, &got, fc->msgs[j].role, fc->msgs[j].content);
                continue;
            }
            /* a run of tool messages is one turn's results */
            const char *parts[8];
            size_t n = 0;
            while (j < fc->nmsg && !strcmp(fc->msgs[j].role, "tool") && n < N_OF(parts))
                parts[n++] = fc->msgs[j++].content;
            j--;
            CHECK(ds4_chat_append_tool_results(&e, &got, parts, n) == 1);
        }
        ds4_chat_append_assistant_prefix(&e, &got, fc->think ? DS4_THINK_HIGH : DS4_THINK_NONE);
        expect_ids(fc->name, &got, fc->ids, fc->nids);
        ds4_tokens_free(&got);
    }
    check_mellum_tool_renders(&e);
    for (size_t i = 0; i < N_OF(fx_histories); i++) {
        /* the CLI's order: a system turn, the prefix file's turns, the live prompt, the generation prompt */
        const fx_case *fc = &fx_histories[i];
        ds4_prompt_prefix_turn turns[8];
        ds4_prompt_prefix history = {turns, 0};
        ds4_tokens got = {0};
        int j = 0;
        if (!strcmp(fc->msgs[0].role, "system")) ds4_chat_append_message(&e, &got, "system", fc->msgs[j++].content);
        for (; j < fc->nmsg - 1 && history.count < N_OF(turns); j++) {
            const bool assistant = !strcmp(fc->msgs[j].role, "assistant");
            turns[history.count++] = (ds4_prompt_prefix_turn){
                assistant ? DS4_PROMPT_PREFIX_ASSISTANT : DS4_PROMPT_PREFIX_USER, (char *)fc->msgs[j].content};
        }
        ds4_prompt_prefix_append(&e, &got, &history);
        ds4_chat_append_message(&e, &got, "user", fc->msgs[fc->nmsg - 1].content);
        ds4_chat_append_assistant_prefix(&e, &got, fc->think ? DS4_THINK_HIGH : DS4_THINK_NONE);
        expect_ids(fc->name, &got, fc->ids, fc->nids);
        ds4_tokens_free(&got);
    }
    {   /* the CLI -p path: system and prompt through encode_chat_prompt */
        ds4_tokens got = {0};
        ds4_encode_chat_prompt(&e, "You are a helpful assistant.", "Hello, world!", DS4_THINK_HIGH, &got);
        expect_ids("encode_chat_prompt think on", &got, fx_cases[0].ids, fx_cases[0].nids);
        got.len = 0;
        ds4_encode_chat_prompt(&e, "You are a helpful assistant.", "Hello, world!", DS4_THINK_NONE, &got);
        expect_ids("encode_chat_prompt think off", &got, fx_cases[1].ids, fx_cases[1].nids);
        ds4_tokens_free(&got);
    }
    vocab_free(&e.vocab);
    model_close(&m);
    printf("ok   Mellum (real GGUF): %zu renders, %zu with tools, %zu imported histories and 2 prefixes equal 2.1's "
           "template ids\n", N_OF(fx_cases), N_OF(fx_tool_cases), N_OF(fx_histories));
}

/* The engine's one sampling call: Mellum's tuple (2.1's generation_config.json carries no
 * sampler keys; temperature 1.0, top_p 1.0, min_p 0, no top_k), the generic DS4_DEFAULT_*
 * for the others. */
static void check_sampling_defaults(void) {
    static ds4_engine e;
    float t = -1, p = -1, m = -1;
    int k = -1;
    g_ds4_shape = DS4_SHAPE_MELLUM;
    ds4_engine_sampling_defaults(&e, &t, &k, &p, &m);
    CHECK(t == 1.0f && k == 0 && p == 1.0f && m == 0.0f);
    static const ds4_shape *const others[] = { &DS4_SHAPE_QWEN4_EXP, &DS4_SHAPE_GLM52 };
    for (size_t i = 0; i < N_OF(others); i++) {
        g_ds4_shape = *others[i];
        t = p = m = -1; k = -1;
        ds4_engine_sampling_defaults(&e, &t, &k, &p, &m);
        CHECK(t == DS4_DEFAULT_TEMPERATURE && k == 0 && p == DS4_DEFAULT_TOP_P && m == DS4_DEFAULT_MIN_P);
    }
    printf("ok   sampling: Mellum (1.0, top_k 0, 1.0, 0.0); Flash Next and GLM the generic defaults\n");
}

/* Session save and load refuse Mellum by name at every entry point (the payload layout of its
 * rings is not written yet); a Flash Next session reaches its own checks instead. */
static void check_payload_refused(void) {
    static ds4_engine e;
    static ds4_session s;
    memset(&s, 0, sizeof(s));
    s.engine = &e;
    char err[256];
    FILE *fp = tmpfile();
    CHECK(fp != NULL);
    const char *msg = "Mellum 2.1 session save and load are not supported yet";
    ds4_session_payload_file pf;
    ds4_session_snapshot snap;
    uint8_t byte = 0;
    const int toks[1] = {0};

    g_ds4_shape = DS4_SHAPE_MELLUM;
    CHECK(ds4_session_payload_bytes(&s) == 0);
    CHECK(ds4_session_layer_payload_bytes(&s, 0, 1) == 0);
    err[0] = '\0'; CHECK(ds4_session_save_payload(&s, fp, err, sizeof(err)) == 1 && !strcmp(err, msg));
    err[0] = '\0'; CHECK(ds4_session_load_payload(&s, fp, 64, err, sizeof(err)) == 1 && !strcmp(err, msg));
    err[0] = '\0'; CHECK(ds4_session_stage_payload(&s, &pf, err, sizeof(err)) == 1 && !strcmp(err, msg));
    memset(&snap, 0, sizeof(snap));
    err[0] = '\0'; CHECK(ds4_session_save_snapshot(&s, &snap, err, sizeof(err)) == 1 && !strcmp(err, msg));
    snap.ptr = &byte; snap.len = 1;
    err[0] = '\0'; CHECK(ds4_session_load_snapshot(&s, &snap, err, sizeof(err)) == 1 && !strcmp(err, msg));
    err[0] = '\0'; CHECK(ds4_session_save_layer_payload(&s, fp, 0, 1, err, sizeof(err)) == 1 && !strcmp(err, msg));
    err[0] = '\0';
    CHECK(ds4_session_load_layer_payload(&s, fp, 64, toks, 1, 0, 1, err, sizeof(err)) == 1 && !strcmp(err, msg));

    /* siblings: the same calls on a Flash Next session fail on their own checks, not on Mellum's */
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    err[0] = '\0'; CHECK(ds4_session_save_payload(&s, fp, err, sizeof(err)) == 1 && strcmp(err, msg) != 0);
    err[0] = '\0'; CHECK(ds4_session_load_payload(&s, fp, 64, err, sizeof(err)) == 1 && strcmp(err, msg) != 0);
    err[0] = '\0'; CHECK(ds4_session_stage_payload(&s, &pf, err, sizeof(err)) == 1 && strcmp(err, msg) != 0);
    snap.ptr = NULL; snap.len = 0;
    err[0] = '\0'; CHECK(ds4_session_save_snapshot(&s, &snap, err, sizeof(err)) == 1 && strcmp(err, msg) != 0);
    err[0] = '\0'; CHECK(ds4_session_load_snapshot(&s, &snap, err, sizeof(err)) == 1 && strcmp(err, msg) != 0);
    err[0] = '\0'; CHECK(ds4_session_save_layer_payload(&s, fp, 0, 1, err, sizeof(err)) == 1 && strcmp(err, msg) != 0);
    err[0] = '\0';
    CHECK(ds4_session_load_layer_payload(&s, fp, 64, toks, 1, 0, 1, err, sizeof(err)) == 1 && strcmp(err, msg) != 0);
    fclose(fp);
    printf("ok   session payload and snapshot: Mellum refused by name at every entry; Flash Next unchanged\n");
}

/* The real vocabulary: the same classes over every one of its 98,304 ids. */
static void check_mellum_stops_model(const char *path) {
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    static ds4_engine e;
    memset(&e, 0, sizeof(e));
    vocab_load(&e.vocab, &m);
    CHECK(e.vocab.n_vocab == 98304);
    check_mellum_stop_classes(&e, "Mellum (real GGUF)");
    vocab_free(&e.vocab);
    model_close(&m);
}

/* With DS4_TEST_MELLUM_MODEL, the real GGUF passes the same validation and
 * binding (mapped, not read). */
static void check_real_model(void) {
    const char *path = getenv("DS4_TEST_MELLUM_MODEL");
    if (!path || !path[0]) {
        printf("skip real model: DS4_TEST_MELLUM_MODEL unset\n");
        return;
    }
    char err[4096];
    const int rc = run_child(child_validate, path, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "FAIL real model: exit %d, stderr [%s]\n", rc, err);
        exit(1);
    }
    printf("ok   real model: keys validated, every tensor bound and checked\n");
    check_mellum_stops_model(path);
    check_mellum_chat_model(path);
}

int main(void) {
    check_variant();
    check_good_set_in_process();
    check_qwen_predicates_unchanged();
    expect_accepted(MF_NONE, "the real key set");
    expect_refused(MF_PATTERN_FLIP, "mellum.attention.sliding_window_pattern", "pattern with one entry flipped");
    expect_refused(MF_PATTERN_LEN27, "mellum.attention.sliding_window_pattern", "pattern of length 27");
    expect_refused(MF_PATTERN_LEN29, "mellum.attention.sliding_window_pattern", "pattern of length 29");
    expect_refused(MF_PATTERN_ABSENT, "mellum.attention.sliding_window_pattern", "no pattern");
    expect_refused(MF_ROPE_NOT_YARN, "mellum.rope.scaling.type", "rope.scaling.type not yarn");
    expect_refused(MF_NO_FREQ_BASE_SWA, "mellum.rope.freq_base_swa", "no rope.freq_base_swa");
    expect_refused(MF_EXPERTS_128, "mellum.expert_count", "expert_count 128");
    expect_refused(MF_USED_10, "mellum.expert_used_count", "expert_used_count 10");
    expect_refused(MF_KEY_LENGTH_256, "mellum.attention.key_length", "key_length 256");
    /* shapes the sliding attention (32 and 4 heads) and the MoE (a width in 128s) would refuse later */
    expect_refused(MF_HEADS_16, "mellum.attention.head_count=32", "head_count 16");
    expect_refused(MF_HEADS_KV_8, "mellum.attention.head_count_kv=4", "head_count_kv 8");
    expect_refused(MF_EMBD_2336, "mellum.embedding_length: must be a multiple of 128", "embedding_length 2336");
    /* the '=' keeps the needle from matching ...sliding_window_pattern */
    expect_refused(MF_WINDOW_512, "mellum.attention.sliding_window=1024", "sliding_window 512");
    expect_refused(MF_YARN_FACTOR_8, "mellum.rope.scaling.factor", "yarn factor 8");
    expect_refused(MF_ATTN_FACTOR_1, "mellum.rope.scaling.yarn_attn_factor", "yarn_attn_factor 1.0");
    expect_refused(MF_ORIG_CTX_4096, "mellum.rope.scaling.original_context_length", "original context 4096");
    expect_refused(MF_SHEXP_TENSOR, "ffn_gate_inp_shexp", "a shared-expert tensor present");
    expect_refused(MF_DOWN_Q4K, "ffn_down_exps", "ffn_down_exps not Q8_0");
    expect_refused(MF_UP_Q4K, "ffn_up_exps", "ffn_up_exps not Q8_0");
    expect_refused(MF_NO_OUTPUT_NORM, "output_norm.weight", "no output_norm");
    expect_refused(MF_NO_ATTN_Q_NORM, "blk.5.attn_q_norm.weight", "no attn_q_norm in layer 5");
    expect_refused(MF_BAD_ATTN_Q_DIM, "blk.5.attn_q.weight", "attn_q of the wrong width");
    expect_open(DS4_BACKEND_CPU, false, true, "CPU backend refused, naming Mellum");
    expect_open(DS4_BACKEND_CUDA, false, true, "CUDA backend refused, naming Mellum");
    expect_open(DS4_BACKEND_CPU, true, false, "inspect-only open succeeds");
    check_mellum_steering_refused();
    check_mellum_plan();
    check_memory_plan();
    expect_vocab_refused(MF_NO_TOKEN_TYPE, "tokenizer.ggml.token_type", "no token_type array");
    expect_vocab_refused(MF_TOKEN_TYPE_SHORT, "tokenizer.ggml.token_type", "token_type one entry short");
    expect_vocab_refused(MF_TOKEN_TYPE_U32, "tokenizer.ggml.token_type", "token_type stored as uint32");
    expect_vocab_refused(MF_NO_IM_START, "<|im_start|>", "<|im_start|> absent");
    expect_vocab_refused(MF_NO_IM_END, "<|im_end|>", "<|im_end|> absent");
    check_mellum_stops_tiny();
    check_mellum_unnamed_control_is_protocol();
    check_qwen_stop_classes_tiny();
    check_mellum_chat_tiny();
    check_sampling_defaults();
    check_payload_refused();
    check_real_model();
    printf("test-mellum-config: ok\n");
    return 0;
}
