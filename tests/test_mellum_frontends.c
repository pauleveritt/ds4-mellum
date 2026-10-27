/* How the front ends treat a Mellum engine, model-free: the engine predicates
 * read the loaded shape, so the test sets the shape directly.  Built once per
 * front end (each has its own statics): -DMELLUM_FRONTEND_AGENT includes
 * ds4_agent.c, -DMELLUM_FRONTEND_SERVER ds4_server.c, -DMELLUM_FRONTEND_CLI
 * ds4_cli.c. */
#include "../ds4.c"
#if defined(MELLUM_FRONTEND_CLI)
#define main ds4_cli_main
#include "../ds4_cli.c"
#undef main
#elif defined(MELLUM_FRONTEND_AGENT)
#define xmalloc agent_xmalloc
#define xrealloc agent_xrealloc
#define now_sec agent_now_sec
#define DS4_AGENT_TEST
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"
#elif defined(MELLUM_FRONTEND_SERVER)
#define xmalloc server_xmalloc
#define xrealloc server_xrealloc
#define now_sec server_now_sec
#define utf8_put server_utf8_put
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#else
#error "build with -DMELLUM_FRONTEND_CLI, -DMELLUM_FRONTEND_AGENT or -DMELLUM_FRONTEND_SERVER"
#endif

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

/* Sampling defaults: a front end fills the knobs its user left unset from the
 * engine's one call.  Mellum's are temperature 1.0, top_p 1.0, min_p 0, no
 * top_k; every other model keeps what it had. */
typedef struct { float t, p, m; } expect_sampling;

static const expect_sampling MELLUM_SAMPLING = {1.0f, 1.0f, 0.0f};
static const expect_sampling GENERIC_SAMPLING = {DS4_DEFAULT_TEMPERATURE, DS4_DEFAULT_TOP_P, DS4_DEFAULT_MIN_P};

#if defined(MELLUM_FRONTEND_CLI) || defined(MELLUM_FRONTEND_AGENT)
/* the CLI and agent keep upstream's GLM defaults in the front end */
static const expect_sampling GLM_SAMPLING = {1.0f, 0.95f, 0.0f};

#if defined(MELLUM_FRONTEND_CLI)
typedef cli_generation_options frontend_gen;
#define APPLY_DEFAULTS cli_apply_model_sampling_defaults
#else
typedef agent_generation_options frontend_gen;
#define APPLY_DEFAULTS agent_apply_model_sampling_defaults
#endif

static void expect_gen(const ds4_shape *shape, expect_sampling want, bool thinking) {
    static ds4_engine e;
    static frontend_gen g;
    g_ds4_shape = *shape;
    /* the values the front end starts from, as its option parser sets them */
    memset(&g, 0, sizeof(g));
    g.temperature = DS4_DEFAULT_TEMPERATURE;
    g.top_p = DS4_DEFAULT_TOP_P;
    g.min_p = DS4_DEFAULT_MIN_P;
    g.think_mode = thinking ? DS4_THINK_HIGH : DS4_THINK_NONE;
    APPLY_DEFAULTS(&e, &g);
    CHECK(g.temperature == want.t && g.top_p == want.p && g.min_p == want.m);
    /* a flag given on the command line wins over the model's default */
    memset(&g, 0, sizeof(g));
    g.temperature = 0.25f; g.temperature_set = true;
    g.top_p = 0.5f; g.top_p_set = true;
    g.min_p = 0.125f; g.min_p_set = true;
    APPLY_DEFAULTS(&e, &g);
    CHECK(g.temperature == 0.25f && g.top_p == 0.5f && g.min_p == 0.125f);
}

static void check_sampling_defaults(void) {
    for (int thinking = 0; thinking < 2; thinking++) {
        expect_gen(&DS4_SHAPE_MELLUM, MELLUM_SAMPLING, thinking);
        expect_gen(&DS4_SHAPE_QWEN4_EXP, GENERIC_SAMPLING, thinking);
        expect_gen(&DS4_SHAPE_GLM52, GLM_SAMPLING, thinking);
    }
    puts("ok   sampling defaults: Mellum (1.0, 1.0, min_p 0) from the engine; Flash Next and GLM unchanged");
}
#else
/* The server: a knob the request leaves out takes the engine's default, in
 * both think modes; a request value wins. */
static void expect_req(const ds4_shape *shape, expect_sampling want, bool thinking) {
    static ds4_engine e;
    g_ds4_shape = *shape;
    request r;
    request_init(&r, REQ_CHAT, 16);
    r.think_mode = thinking ? DS4_THINK_HIGH : DS4_THINK_NONE;
    float t, p, m;
    int k;
    server_sampling_for_request(&e, &r, &t, &k, &p, &m);
    CHECK(t == want.t && p == want.p && m == want.m && k == 0);
    r.temperature = 0.25f; r.temperature_set = true;
    r.top_p = 0.5f; r.top_p_set = true;
    r.min_p = 0.125f; r.min_p_set = true;
    r.top_k = 7; r.top_k_set = true;
    server_sampling_for_request(&e, &r, &t, &k, &p, &m);
    CHECK(t == 0.25f && p == 0.5f && m == 0.125f && k == 7);
    request_free(&r);
}

static void check_sampling_defaults(void) {
    for (int thinking = 0; thinking < 2; thinking++) {
        expect_req(&DS4_SHAPE_MELLUM, MELLUM_SAMPLING, thinking);
        expect_req(&DS4_SHAPE_QWEN4_EXP, GENERIC_SAMPLING, thinking);
        expect_req(&DS4_SHAPE_GLM52, GENERIC_SAMPLING, thinking);   /* the server has no GLM defaults */
    }
    puts("ok   server sampling: Mellum's defaults from the engine in both think modes; others unchanged");
}
#endif

#if defined(MELLUM_FRONTEND_AGENT)
/* A byte-level vocabulary with 2.1's special ids at their real places and no merges: byte c is
 * id 35 + c, as in tests/mellum_fixture.h. */
static void tiny_vocab(ds4_engine *e) {
    static char fill[35][24], bytes[256][8];
    static ds4_str toks[35 + 256];
    memset(e, 0, sizeof(*e));
    ds4_vocab *v = &e->vocab;
    v->n_vocab = 35 + 256;
    v->token = toks;
    table_init(&v->token_to_id, 291);
    table_init(&v->merge_rank, 0);
    const char *special[35];
    for (int i = 0; i < 35; i++) { snprintf(fill[i], sizeof(fill[i]), "<fill_%d>", i); special[i] = fill[i]; }
    special[0] = "<|endoftext|>"; special[23] = "<think>"; special[24] = "</think>";
    special[27] = "<|im_start|>"; special[28] = "<|im_end|>";
    special[29] = "<tool_call>"; special[30] = "</tool_call>";
    special[31] = "<tool_response>"; special[32] = "</tool_response>";
    int extra = 0;
    for (int i = 0; i < 291; i++) {
        const char *t = NULL;
        if (i < 35) t = special[i];
        else {
            const int c = i - 35;
            const bool direct = (c >= 33 && c <= 126) || (c >= 161 && c <= 172) || (c >= 174);
            const int cp = direct ? c : 256 + extra++;
            if (cp < 0x80) { bytes[c][0] = (char)cp; bytes[c][1] = 0; }
            else { bytes[c][0] = (char)(0xC0 | (cp >> 6)); bytes[c][1] = (char)(0x80 | (cp & 0x3F)); bytes[c][2] = 0; }
            t = bytes[c];
        }
        toks[i].ptr = t; toks[i].len = strlen(t);
        table_put(&v->token_to_id, toks[i], i);
    }
    v->im_start_id = 27; v->im_end_id = 28; v->endoftext_id = 0; v->eos_id = 28;
    v->think_start_id = 23; v->think_end_id = 24;
    v->bos_id = v->system_id = v->user_id = v->assistant_id = -1;
}

static void push_bytes(ds4_tokens *t, const char *s) {
    for (; *s; s++) ds4_tokens_push(t, 35 + (unsigned char)*s);
}

/* Compaction resumes inside an open reasoning block: the prefix must carry <think> and its newline
 * whichever model: Qwen's own thinking-on prefix has them, Mellum's does not (the model writes
 * <think> itself). */
static void check_resume_prefix(void) {
    static ds4_engine e;
    tiny_vocab(&e);
    ds4_tokens want = {0}, got = {0};
    ds4_tokens_push(&want, 27);
    push_bytes(&want, "assistant\n");
    ds4_tokens_push(&want, 23);
    push_bytes(&want, "\n");

    g_ds4_shape = DS4_SHAPE_MELLUM;
    agent_append_open_think_prefix(&e, &got);
    CHECK(got.len == want.len && !memcmp(got.v, want.v, (size_t)want.len * sizeof(int)));

    /* sibling: Qwen's prefix already ends <think> and a newline, so nothing is added */
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    got.len = 0;
    agent_append_open_think_prefix(&e, &got);
    CHECK(got.len == want.len && !memcmp(got.v, want.v, (size_t)want.len * sizeof(int)));
    ds4_tokens plain = {0};
    ds4_chat_append_assistant_prefix(&e, &plain, DS4_THINK_HIGH);
    CHECK(plain.len == got.len && !memcmp(plain.v, got.v, (size_t)got.len * sizeof(int)));
    ds4_tokens_free(&want); ds4_tokens_free(&got); ds4_tokens_free(&plain);
    puts("ok   agent: compaction resume reopens <think> for Mellum; Flash Next's prefix unchanged");
}
#else
static void check_resume_prefix(void) {}
#endif

#if defined(MELLUM_FRONTEND_CLI)
static void check_engine_selection(void) {}
#elif defined(MELLUM_FRONTEND_AGENT)
/* Mellum is also a qwen4 engine, so a check of ds4_engine_is_qwen4 alone would
 * speak it Flash Next's Qwen3-coder XML: Mellum gets the Hermes syntax, ahead of
 * that check.  Until the agent runs Pi's tools it still refuses the engine by
 * name; the other qwen4 models keep their syntax. */
static void check_engine_selection(void) {
    static ds4_engine e;
    g_ds4_shape = DS4_SHAPE_MELLUM;
    CHECK(ds4_engine_is_mellum(&e) && ds4_engine_is_qwen4(&e));
    CHECK(agent_tool_syntax_for_engine(&e) == AGENT_TOOL_SYNTAX_HERMES);
    const char *why = agent_engine_refusal(&e);
    CHECK(why && strstr(why, "Mellum 2.1"));

    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    CHECK(!ds4_engine_is_mellum(&e));
    CHECK(agent_engine_refusal(&e) == NULL);
    CHECK(agent_tool_syntax_for_engine(&e) == AGENT_TOOL_SYNTAX_QWEN);
    puts("ok   agent: Mellum gets the Hermes syntax and is still refused by name; Flash Next keeps its syntax");
}
#else
/* The server names a Mellum engine as Mellum and refuses to serve it until it
 * has the Hermes syntax; Flash Next keeps its id and syntax. */
static void check_engine_selection(void) {
    static ds4_engine e;
    g_ds4_shape = DS4_SHAPE_MELLUM;
    CHECK(ds4_engine_is_mellum(&e) && ds4_engine_is_qwen4(&e));
    CHECK(!strcmp(server_model_id_from_engine(&e), "mellum-2.1"));
    const char *why = server_engine_refusal(&e);
    CHECK(why && strstr(why, "Mellum 2.1"));

    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    CHECK(!ds4_engine_is_mellum(&e));
    CHECK(!strcmp(server_model_id_from_engine(&e), "qwen3.8-flash-next"));
    CHECK(server_engine_refusal(&e) == NULL);
    CHECK(server_model_syntax_for_engine(&e) == SERVER_MODEL_SYNTAX_QWEN);
    puts("ok   server: Mellum named and refused by name, Flash Next keeps its id and syntax");
}
#endif

int main(void) {
    check_engine_selection();
    check_sampling_defaults();
    check_resume_prefix();
    return 0;
}
