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

#include <glob.h>

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

/* One round's tool results for Mellum: each call's result is its own part, the text as the tool returned it (no
 * "Tool result N" header), and the round is one user turn with a <tool_response> block per result, as 2.1's
 * template renders consecutive tool messages (ds4_chat_append_tool_results). */
static void check_tool_observation(void) {
    static ds4_engine e;
    tiny_vocab(&e);
    g_ds4_shape = DS4_SHAPE_MELLUM;
    agent_worker w;
    memset(&w, 0, sizeof(w));
    w.engine = &e;
    pthread_mutex_init(&w.mu, NULL);
    agent_tool_calls calls = {0};
    agent_tool_call a = {.name = xstrdup("alpha")}, b = {.name = xstrdup("beta")};
    agent_tool_calls_push(&calls, &a);
    agent_tool_calls_push(&calls, &b);
    char *r0 = agent_execute_tool_call(&w, &calls.v[0]), *r1 = agent_execute_tool_call(&w, &calls.v[1]);
    agent_tool_observation obs = agent_execute_tool_observation(&w, &calls);
    ds4_tokens got = {0}, want = {0};
    ds4_vision_span *spans = NULL;
    char err[128] = "";
    CHECK(agent_tool_observation_build(&w, &obs, &got, &spans, err, sizeof(err)));
    const char *parts[] = {r0, r1};
    CHECK(ds4_chat_append_tool_results(&e, &want, parts, 2) == 1);
    CHECK(got.len == want.len && !memcmp(got.v, want.v, (size_t)want.len * sizeof(int)));
    agent_tool_observation_free(&obs);
    agent_tool_calls_free(&calls);
    free(r0); free(r1); free(w.out);
    ds4_tokens_free(&got); ds4_tokens_free(&want);
    puts("ok   agent: a Mellum round's results are one user turn, a raw <tool_response> block per call");
}

/* ---- Pi's four tools, for Mellum: Pi's argument names and result texts (Pi 1.1.0, dist/core/tools) ---- */

static agent_worker *pi_worker(void) {
    static ds4_engine e;
    static agent_worker w;
    static bool ready;
    if (!ready) {
        tiny_vocab(&e);
        memset(&w, 0, sizeof(w));
        w.engine = &e;
        pthread_mutex_init(&w.mu, NULL);
        ready = true;
    }
    g_ds4_shape = DS4_SHAPE_MELLUM;
    return &w;
}

/* run one call: name, then name/value pairs (a value starting with '#' is a raw JSON value, '#' dropped) */
static char *pi_run(const char *name, ...) {
    agent_tool_call c = {.name = xstrdup(name)};
    va_list ap;
    va_start(ap, name);
    for (const char *k; (k = va_arg(ap, const char *)) != NULL;) {
        const char *v = va_arg(ap, const char *);
        agent_tool_call_add_arg_owned(&c, k, xstrdup(v[0] == '#' ? v + 1 : v), v[0] != '#');
    }
    va_end(ap);
    char *r = agent_execute_tool_call(pi_worker(), &c);
    agent_tool_call_free(&c);
    return r;
}

static char *pi_fixture(const char *dir, const char *name, const char *data) {
    char *path = xmalloc(strlen(dir) + strlen(name) + 2);
    sprintf(path, "%s/%s", dir, name);
    FILE *fp = fopen(path, "wb");
    CHECK(fp != NULL);
    fwrite(data, 1, strlen(data), fp);
    fclose(fp);
    return path;
}

#define PI_EXPECT(got, want) do { char *g_ = (got); CHECK(!strcmp(g_, (want))); free(g_); } while (0)
#define PI_EXPECT_PREFIX(got, want) do { char *g_ = (got); CHECK(strstr(g_, (want)) == g_); free(g_); } while (0)

static void check_pi_read(const char *dir) {
    char *five = pi_fixture(dir, "five.txt", "l1\nl2\nl3\nl4\nl5\n");
    /* the file's text as it is: no line numbers, no header */
    PI_EXPECT(pi_run("read", "path", five, NULL), "l1\nl2\nl3\nl4\nl5\n");
    PI_EXPECT(pi_run("read", "path", five, "offset", "#3", NULL), "l3\nl4\nl5\n");
    /* Pi counts lines with split("\n"): the empty line after the final newline is one */
    PI_EXPECT(pi_run("read", "path", five, "limit", "#2", NULL), "l1\nl2\n\n[4 more lines in file. Use offset=3 to continue.]");
    PI_EXPECT(pi_run("read", "path", five, "offset", "#2", "limit", "#2", NULL),
              "l2\nl3\n\n[3 more lines in file. Use offset=4 to continue.]");
    PI_EXPECT(pi_run("read", "path", five, "offset", "#99", NULL), "Offset 99 is beyond end of file (6 lines total)");
    /* absurd numbers are clamped to the file before any conversion: Pi's Math.min, without C's undefined cast */
    PI_EXPECT(pi_run("read", "path", five, "limit", "#1e30", NULL), "l1\nl2\nl3\nl4\nl5\n");
    PI_EXPECT(pi_run("read", "path", five, "offset", "#2", "limit", "#1e30", NULL), "l2\nl3\nl4\nl5\n");
    PI_EXPECT(pi_run("read", "path", five, "offset", "#1e30", NULL), "Offset 1e30 is beyond end of file (6 lines total)");
    PI_EXPECT(pi_run("read", "path", five, "offset", "#18446744073709551617", NULL),
              "Offset 18446744073709551617 is beyond end of file (6 lines total)");
    /* a negative limit reads nothing and says how much is left (Pi's own arithmetic would count past the start) */
    PI_EXPECT(pi_run("read", "path", five, "limit", "#-3", NULL), "\n\n[6 more lines in file. Use offset=1 to continue.]");
    PI_EXPECT(pi_run("read", "path", five, "offset", "#-4", "limit", "#1", NULL),
              "l1\n\n[5 more lines in file. Use offset=2 to continue.]");
    char *missing = xmalloc(strlen(dir) + 16);
    sprintf(missing, "%s/nope.txt", dir);
    PI_EXPECT_PREFIX(pi_run("read", "path", missing, NULL), "ENOENT: no such file or directory, access '");
    PI_EXPECT(pi_run("read", NULL), "read requires a path string.");
    /* 2,500 lines: the head is cut at 2,000 with the continuation footer */
    agent_buf big = {0};
    for (int i = 1; i <= 2500; i++) { char l[32]; snprintf(l, sizeof(l), "line %d\n", i); agent_buf_puts(&big, l); }
    char *bigpath = pi_fixture(dir, "big.txt", big.ptr);
    char *r = pi_run("read", "path", bigpath, NULL);
    CHECK(strstr(r, "line 2000\n\n[Showing lines 1-2000 of 2501. Use offset=2001 to continue.]") && !strstr(r, "line 2001"));
    free(r);
    r = pi_run("read", "path", bigpath, "offset", "#2001", NULL);
    CHECK(strstr(r, "line 2001\n") == r && strstr(r, "line 2500\n") && !strstr(r, "[Showing") && !strstr(r, "more lines"));
    free(r);
    /* 100 lines of 600 bytes: cut by the 50KB limit */
    agent_buf wide = {0};
    for (int i = 0; i < 100; i++) { char l[700]; memset(l, 'a', 600); strcpy(l + 600, "\n"); agent_buf_puts(&wide, l); }
    char *widepath = pi_fixture(dir, "wide.txt", wide.ptr);
    r = pi_run("read", "path", widepath, NULL);
    CHECK(strstr(r, "\n\n[Showing lines 1-85 of 101 (50.0KB limit). Use offset=86 to continue.]") != NULL);
    free(r);
    /* a first line over the limit: Pi points the model at bash */
    char *huge = xmalloc(60000);
    memset(huge, 'b', 59999);
    huge[59999] = '\0';
    char *hugepath = pi_fixture(dir, "huge.txt", huge);
    PI_EXPECT_PREFIX(pi_run("read", "path", hugepath, NULL), "[Line 1 is 58.6KB, exceeds 50.0KB limit. Use bash: sed -n '1p' ");
    free(five); free(missing); free(bigpath); free(widepath); free(hugepath); free(huge); free(big.ptr); free(wide.ptr);
    puts("ok   agent: Pi's read for Mellum (raw text, offset and limit, Pi's footers and errors)");
}

static char *pi_slurp(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return xstrdup("<unreadable>");
    agent_buf b = {0};
    char tmp[4096];
    size_t n;
    while ((n = fread(tmp, 1, sizeof(tmp), fp)) > 0) agent_buf_append(&b, tmp, n);
    fclose(fp);
    return b.ptr ? agent_buf_take(&b) : xstrdup("");
}

/* write: the training-era text (Pi 0.84.4, as 0.80.10 here; 1.1.0 dropped the count), whose count is the
 * JavaScript string length, UTF-16 units: "é" is one, an emoji two */
static void check_pi_write(const char *dir) {
    char *nested = xmalloc(strlen(dir) + 32), want[PATH_MAX + 64];
    sprintf(nested, "%s/a/b/new.py", dir);
    snprintf(want, sizeof(want), "Successfully wrote 9 bytes to %s", nested);
    PI_EXPECT(pi_run("write", "path", nested, "content", "print(1)\n", NULL), want);
    char *back = pi_slurp(nested);
    CHECK(!strcmp(back, "print(1)\n"));
    free(back);
    snprintf(want, sizeof(want), "Successfully wrote 4 bytes to %s", nested);
    PI_EXPECT(pi_run("write", "path", nested, "content", "\xc3\xa9\xf0\x9f\x98\x80!", NULL), want);
    back = pi_slurp(nested);
    CHECK(!strcmp(back, "\xc3\xa9\xf0\x9f\x98\x80!"));
    free(back);
    /* a missing content is refused, and the file is left as it was */
    PI_EXPECT(pi_run("write", "path", nested, NULL), "write requires a content string.");
    back = pi_slurp(nested);
    CHECK(!strcmp(back, "\xc3\xa9\xf0\x9f\x98\x80!"));
    free(back);
    PI_EXPECT(pi_run("write", "content", "x", NULL), "write requires a path string.");
    free(nested);
    puts("ok   agent: Pi's write for Mellum (parents created, the training-era text with Pi's count)");
}

/* edit, exact matching (Pi's fuzzy pass comes with step 35): every edit is matched against the original */
static void check_pi_edit(const char *dir) {
    char *f = pi_fixture(dir, "m.py", "def f():\n    return 1\n\ndef g():\n    return 2\n");
    char want[PATH_MAX + 160];
    snprintf(want, sizeof(want), "Successfully replaced 2 block(s) in %s.", f);
    PI_EXPECT(pi_run("edit", "path", f, "edits",
                     "#[{\"oldText\": \"return 1\", \"newText\": \"return 10\"}, {\"oldText\": \"return 2\", \"newText\": \"return 20\"}]",
                     NULL), want);
    char *back = pi_slurp(f);
    CHECK(!strcmp(back, "def f():\n    return 10\n\ndef g():\n    return 20\n"));
    free(back);
    /* the refusals; the edit above is their success sibling, and none of them writes */
    snprintf(want, sizeof(want), "Could not find the exact text in %s. The old text must match exactly including all "
             "whitespace and newlines.", f);
    PI_EXPECT(pi_run("edit", "path", f, "edits", "#[{\"oldText\": \"return 99\", \"newText\": \"x\"}]", NULL), want);
    snprintf(want, sizeof(want), "Found 2 occurrences of the text in %s. The text must be unique. Please provide more "
             "context to make it unique.", f);
    PI_EXPECT(pi_run("edit", "path", f, "edits", "#[{\"oldText\": \"return\", \"newText\": \"x\"}]", NULL), want);
    snprintf(want, sizeof(want), "edits[0] and edits[1] overlap in %s. Merge them into one edit or target disjoint regions.", f);
    PI_EXPECT(pi_run("edit", "path", f, "edits",
                     "#[{\"oldText\": \"return 10\\n\\ndef\", \"newText\": \"a\"}, {\"oldText\": \"\\n\\ndef g\", \"newText\": \"b\"}]",
                     NULL), want);
    snprintf(want, sizeof(want), "oldText must not be empty in %s.", f);
    PI_EXPECT(pi_run("edit", "path", f, "edits", "#[{\"oldText\": \"\", \"newText\": \"b\"}]", NULL), want);
    PI_EXPECT_PREFIX(pi_run("edit", "path", f, "edits", "#[{\"oldText\": \"return 10\", \"newText\": \"return 10\"}]", NULL),
                     "No changes made to ");
    PI_EXPECT(pi_run("edit", "path", f, "edits", "#[]", NULL),
              "Edit tool input is invalid. edits must contain at least one replacement.");
    back = pi_slurp(f);
    CHECK(!strcmp(back, "def f():\n    return 10\n\ndef g():\n    return 20\n"));
    free(back);
    /* as Pi takes them: edits sent as a JSON string, or as one object */
    snprintf(want, sizeof(want), "Successfully replaced 1 block(s) in %s.", f);
    PI_EXPECT(pi_run("edit", "path", f, "edits", "[{\"oldText\": \"return 20\", \"newText\": \"return 2\"}]", NULL), want);
    PI_EXPECT(pi_run("edit", "path", f, "edits", "#{\"oldText\": \"return 10\", \"newText\": \"return 1\"}", NULL), want);
    back = pi_slurp(f);
    CHECK(!strcmp(back, "def f():\n    return 1\n\ndef g():\n    return 2\n"));
    free(back);
    /* a BOM and CRLF line ends survive */
    char *crlf = pi_fixture(dir, "w.txt", "\xef\xbb\xbf" "a\r\nb\r\nc\r\n");
    PI_EXPECT_PREFIX(pi_run("edit", "path", crlf, "edits", "#[{\"oldText\": \"b\", \"newText\": \"B\"}]", NULL),
                     "Successfully replaced 1 block(s)");
    back = pi_slurp(crlf);
    CHECK(!strcmp(back, "\xef\xbb\xbf" "a\r\nB\r\nc\r\n"));
    free(back);
    char *gone = xmalloc(strlen(dir) + 16);
    sprintf(gone, "%s/gone.txt", dir);
    snprintf(want, sizeof(want), "Could not edit file: %s. Error code: ENOENT.", gone);
    PI_EXPECT(pi_run("edit", "path", gone, "edits", "#[{\"oldText\": \"a\", \"newText\": \"b\"}]", NULL), want);
    free(f); free(crlf); free(gone);
    puts("ok   agent: Pi's edit for Mellum, exact (edits[] against the original, Pi's refusals, BOM and CRLF kept)");
}

/* bash: stdout and stderr merged, Pi's texts for no output, a failing exit, a timeout, and a tail cut */
static void check_pi_bash(void) {
    glob_t before = {0}, after = {0};
    glob("/tmp/ds4_agent_output_*", 0, NULL, &before);
    agent_worker *w = pi_worker();
    w->wake_fd[0] = w->wake_fd[1] = -1;
    PI_EXPECT(pi_run("bash", "command", "printf hi", NULL), "hi");
    PI_EXPECT(pi_run("bash", "command", "true", NULL), "(no output)");
    PI_EXPECT(pi_run("bash", "command", "exit 3", NULL), "(no output)\n\nCommand exited with code 3");
    char *r = pi_run("bash", "command", "echo out; echo err 1>&2; exit 2", NULL);
    CHECK(strstr(r, "out\n") && strstr(r, "err\n") && strstr(r, "\n\nCommand exited with code 2"));
    free(r);
    PI_EXPECT(pi_run("bash", "command", "sleep 5", "timeout", "#1", NULL), "Command timed out after 1 seconds");
    /* the tail is kept: 3,000 lines become the last 2,000, with the footer naming the full output */
    r = pi_run("bash", "command", "seq 1 3000", NULL);
    CHECK(strstr(r, "1001\n1002\n") == r);
    CHECK(strstr(r, "2999\n3000\n\n[Showing lines 1001-3000 of 3000. Full output: /tmp/ds4_agent_output_") != NULL);
    free(r);
    PI_EXPECT(pi_run("bash", NULL), "bash requires a command string.");
    glob("/tmp/ds4_agent_output_*", 0, NULL, &after);
    for (size_t i = 0; i < after.gl_pathc; i++) {
        bool old = false;
        for (size_t j = 0; j < before.gl_pathc && !old; j++) old = !strcmp(after.gl_pathv[i], before.gl_pathv[j]);
        if (!old) unlink(after.gl_pathv[i]);
    }
    globfree(&before);
    globfree(&after);
    puts("ok   agent: Pi's bash for Mellum (merged output, Pi's status lines, timeout, the tail cut)");
}

/* the four are all a Mellum engine is offered: any other name, the agent's own tools included, gets the answer Pi's
 * agent loop gives (pi-agent-core, agent-loop.js) */
static void check_pi_unknown(void) {
    PI_EXPECT(pi_run("grep", "pattern", "x", NULL), "Tool grep not found");
    PI_EXPECT(pi_run("list", "path", ".", NULL), "Tool list not found");
    PI_EXPECT(pi_run("view_image", "path", "a.png", NULL), "Tool view_image not found");
    puts("ok   agent: Mellum is offered Pi's four tools only; another name is Pi's \"Tool X not found\"");
}

/* edit's fuzzy pass (Pi's normalizeForFuzzyMatch without NFKC): trailing whitespace per line, smart quotes, Unicode
 * dashes and spaces; the lines nobody edited keep their bytes */
static void check_pi_edit_fuzzy(const char *dir) {
    char want[PATH_MAX + 160];
    /* the model types straight quotes and drops trailing spaces; the file keeps its smart quotes elsewhere */
    char *q = pi_fixture(dir, "q.txt", "say \xe2\x80\x9chi\xe2\x80\x9d   \nkeep \xe2\x80\x98this\xe2\x80\x99 line\nend\n");
    snprintf(want, sizeof(want), "Successfully replaced 1 block(s) in %s.", q);
    PI_EXPECT(pi_run("edit", "path", q, "edits", "#[{\"oldText\": \"say \\\"hi\\\"\\n\", \"newText\": \"say \\\"bye\\\"\\n\"}]",
                     NULL), want);
    char *back = pi_slurp(q);
    CHECK(!strcmp(back, "say \"bye\"\nkeep \xe2\x80\x98this\xe2\x80\x99 line\nend\n"));
    free(back);
    /* sibling: text that is nowhere, normalized or not, is still not found */
    PI_EXPECT_PREFIX(pi_run("edit", "path", q, "edits", "#[{\"oldText\": \"say zzz\", \"newText\": \"x\"}]", NULL),
                     "Could not find the exact text in ");
    /* an indented blank line is an empty one to the fuzzy pass */
    char *blank = pi_fixture(dir, "blank.py", "def f():\n    x = 1\n    \n    return x\n");
    snprintf(want, sizeof(want), "Successfully replaced 1 block(s) in %s.", blank);
    PI_EXPECT(pi_run("edit", "path", blank, "edits",
                     "#[{\"oldText\": \"    x = 1\\n\\n    return x\", \"newText\": \"    x = 2\\n\\n    return x\"}]", NULL), want);
    back = pi_slurp(blank);
    CHECK(!strcmp(back, "def f():\n    x = 2\n\n    return x\n"));
    free(back);
    /* an oldText of only whitespace normalizes to nothing, which Pi finds everywhere: not unique, and no endless loop */
    PI_EXPECT_PREFIX(pi_run("edit", "path", blank, "edits", "#[{\"oldText\": \"    \", \"newText\": \"\"}]", NULL), "Found ");
    /* Pi counts occurrences on normalized text: unique exactly, twice once trailing spaces go, so not unique */
    char *twice = pi_fixture(dir, "twice.py", "x = 1\nx = 1  \n");
    snprintf(want, sizeof(want), "Found 2 occurrences of the text in %s. The text must be unique. Please provide more "
             "context to make it unique.", twice);
    PI_EXPECT(pi_run("edit", "path", twice, "edits", "#[{\"oldText\": \"x = 1\\n\", \"newText\": \"x = 2\\n\"}]", NULL), want);
    /* a 4-byte character on an edited line keeps its four bytes (U+1F620 is f0 9f 98 a0, and a0 alone is NBSP) */
    char *emoji = pi_fixture(dir, "emoji.py", "s = \xe2\x80\x9chi\xe2\x80\x9d  # \xf0\x9f\x98\xa0\nend\n");
    PI_EXPECT_PREFIX(pi_run("edit", "path", emoji, "edits", "#[{\"oldText\": \"s = \\\"hi\\\"\", \"newText\": \"s = \\\"bye\\\"\"}]",
                            NULL), "Successfully replaced 1 block(s)");
    back = pi_slurp(emoji);
    CHECK(!strcmp(back, "s = \"bye\"  # \xf0\x9f\x98\xa0\nend\n"));
    free(back);
    free(q); free(blank); free(twice); free(emoji);
    puts("ok   agent: Pi's edit for Mellum, fuzzy (quotes, dashes, spaces, trailing whitespace; other lines kept)");
}

static void check_pi_tools(void) {
    char dir[] = "/tmp/ds4-mellum-pi-XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    check_pi_read(dir);
    check_pi_write(dir);
    check_pi_edit(dir);
    check_pi_edit_fuzzy(dir);
    check_pi_bash();
    check_pi_unknown();
    char cmd[PATH_MAX + 16];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", dir);
    CHECK(system(cmd) == 0);
}
#else
static void check_resume_prefix(void) {}
static void check_tool_observation(void) {}
static void check_pi_tools(void) {}
#endif

#if defined(MELLUM_FRONTEND_CLI)
static void check_engine_selection(void) {}
#elif defined(MELLUM_FRONTEND_AGENT)
/* Mellum is also a qwen4 engine, so a check of ds4_engine_is_qwen4 alone would
 * speak it Flash Next's Qwen3-coder XML: Mellum gets the Hermes syntax, ahead of
 * that check, and with Pi's tools the agent runs it; the other qwen4 models keep
 * their syntax. */
static void check_engine_selection(void) {
    static ds4_engine e;
    g_ds4_shape = DS4_SHAPE_MELLUM;
    CHECK(ds4_engine_is_mellum(&e) && ds4_engine_is_qwen4(&e));
    CHECK(agent_tool_syntax_for_engine(&e) == AGENT_TOOL_SYNTAX_HERMES);
    CHECK(agent_engine_refusal(&e) == NULL);

    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    CHECK(!ds4_engine_is_mellum(&e));
    CHECK(agent_engine_refusal(&e) == NULL);
    CHECK(agent_tool_syntax_for_engine(&e) == AGENT_TOOL_SYNTAX_QWEN);
    puts("ok   agent: Mellum gets the Hermes syntax and is not refused; Flash Next keeps its syntax");
}
#else
/* The server names a Mellum engine as Mellum and speaks it the Hermes syntax, ahead of the qwen4
 * check (it refuses to serve it until that syntax is complete); Flash Next keeps its id and syntax. */
static void check_engine_selection(void) {
    static ds4_engine e;
    g_ds4_shape = DS4_SHAPE_MELLUM;
    CHECK(ds4_engine_is_mellum(&e) && ds4_engine_is_qwen4(&e));
    CHECK(!strcmp(server_model_id_from_engine(&e), "mellum-2.1"));
    CHECK(server_model_syntax_for_engine(&e) == SERVER_MODEL_SYNTAX_HERMES);
    const char *why = server_engine_refusal(&e);
    CHECK(why && strstr(why, "Mellum 2.1"));

    /* step 43: one session, no disk KV (the engine refuses session save and load for Mellum, and Mellum does
     * not batch with other sessions); each refused by name, and the defaults accepted */
    const char *why2 = server_session_option_refusal(&e, 2, NULL);
    CHECK(why2 && strstr(why2, "Mellum 2.1") && strstr(why2, "--batched-session"));
    why2 = server_session_option_refusal(&e, 0, "/tmp/kv");
    CHECK(why2 && strstr(why2, "Mellum 2.1") && strstr(why2, "--kv-disk-dir"));
    CHECK(server_session_option_refusal(&e, 1, NULL) == NULL && server_session_option_refusal(&e, 0, NULL) == NULL);

    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    CHECK(server_session_option_refusal(&e, 4, "/tmp/kv") == NULL);
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
    check_tool_observation();
    check_pi_tools();
    return 0;
}
