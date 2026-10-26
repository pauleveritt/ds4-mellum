/* Mellum 2.1 recognition, validation and binding, model-free.  Each case writes
 * the sparse GGUF of tests/mellum_fixture.h and opens it through model_open /
 * config_validate_model / weights_bind, or ds4_engine_open.  The validators
 * exit(1) with a message, so a refusal is checked in a forked child whose
 * stderr is read back and must name what it refuses. */
#include "../ds4.c"
#include "mellum_fixture.h"
#include <sys/wait.h>

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

static bool g_open_inspect;

static void child_engine_open(const char *path) {
    ds4_engine *e = NULL;
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = path;
    opt.backend = DS4_BACKEND_CPU;
    opt.context_size = 4096;
    opt.inspect_only = g_open_inspect;
    const int rc = ds4_engine_open(&e, &opt);
    _exit(rc == 0 && (!e || ds4_engine_is_mellum(e)) ? 0 : 7);
}

/* Until the forward pass is ported, an open is refused, by name; an
 * inspect-only open (metadata only) succeeds. */
static void check_open_refused(void) {
    char path[512], err[4096];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    g_open_inspect = false;
    int rc = run_child(child_engine_open, path, err, sizeof(err));
    if (rc != 7 || !strstr(err, "Mellum 2.1") || !strstr(err, "not ported")) {
        fprintf(stderr, "FAIL open: exit %d, stderr [%s]\n", rc, err);
        exit(1);
    }
    g_open_inspect = true;
    rc = run_child(child_engine_open, path, err, sizeof(err));
    g_open_inspect = false;
    unlink(path);
    if (rc != 0) {
        fprintf(stderr, "FAIL inspect-only open: exit %d, stderr [%s]\n", rc, err);
        exit(1);
    }
    printf("ok   open: refused by name until the forward pass is ported; inspect-only open succeeds\n");
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
    check_open_refused();
    check_real_model();
    printf("test-mellum-config: ok\n");
    return 0;
}
