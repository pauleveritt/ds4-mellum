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

#if defined(MELLUM_FRONTEND_CLI)
static void check_engine_selection(void) {}
#elif defined(MELLUM_FRONTEND_AGENT)
/* Mellum is also a qwen4 engine, so a check of ds4_engine_is_qwen4 alone would
 * speak it Flash Next's Qwen3-coder XML.  Until the agent has Mellum's Hermes
 * syntax it refuses the engine by name; the other qwen4 models keep theirs. */
static void check_engine_selection(void) {
    static ds4_engine e;
    g_ds4_shape = DS4_SHAPE_MELLUM;
    CHECK(ds4_engine_is_mellum(&e) && ds4_engine_is_qwen4(&e));
    const char *why = agent_engine_refusal(&e);
    CHECK(why && strstr(why, "Mellum 2.1"));

    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    CHECK(!ds4_engine_is_mellum(&e));
    CHECK(agent_engine_refusal(&e) == NULL);
    CHECK(agent_tool_syntax_for_engine(&e) == AGENT_TOOL_SYNTAX_QWEN);
    puts("ok   agent: Mellum refused by name, Flash Next keeps its syntax");
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
    return 0;
}
