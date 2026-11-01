/* Mellum 2.1, included by ds4_test.c where this code sits: not a separate translation unit. */

/* Mellum's session allocation, on the real 2.1 GGUF (DS4_TEST_MELLUM_MODEL;
 * skipped without it).  The allocator prints "mellum: session allocated N
 * bytes, planned M bytes" at create; N is the sum of the bytes it requested,
 * M the estimate the startup line reports.  They must be equal at every
 * context, with the chunk the host passes clamped to the window, Metal's own
 * tally of live tensors must move by N and come back, and the session must
 * run: sync, eval and eval_argmax succeed and leave finite logits. */
typedef struct { FILE *tmp; int saved; } test_stderr_capture;

static test_stderr_capture test_stderr_begin(void) {
    test_stderr_capture c = {0};
    fflush(stderr);
    c.tmp = tmpfile();
    TEST_ASSERT(c.tmp != NULL);
    c.saved = dup(2);
    TEST_ASSERT(c.saved >= 0);
    TEST_ASSERT(dup2(fileno(c.tmp), 2) >= 0);
    return c;
}

/* Restores stderr and returns what was written (NUL-terminated, caller frees). */
static char *test_stderr_end(test_stderr_capture c) {
    fflush(stderr);
    dup2(c.saved, 2);
    close(c.saved);
    const long n = ftell(c.tmp);
    char *text = xmalloc((size_t)(n > 0 ? n : 0) + 1);
    rewind(c.tmp);
    const size_t got = n > 0 ? fread(text, 1, (size_t)n, c.tmp) : 0;
    text[got] = '\0';
    fclose(c.tmp);
    return text;
}

/* Metal's own count of live tensor bytes (its allocator's tally, not the
 * session's), through the memory report, at the report's 0.01 MiB resolution. */
static double test_live_tensor_mib(void) {
    test_stderr_capture c = test_stderr_begin();
    ds4_gpu_print_memory_report("mellum-session");
    char *text = test_stderr_end(c);
    double mib = -1.0;
    const char *p = strstr(text, "runtime tensors live ");
    if (p) sscanf(p, "runtime tensors live %lf MiB", &mib);
    free(text);
    TEST_ASSERT(mib >= 0.0);
    return mib;
}

int ds4_session_eval_argmax(ds4_session *s, int token, char *err, size_t errlen);   /* ds4.c, not in ds4.h */

static bool test_capture_session_line(ds4_engine *engine, uint32_t ctx, uint64_t *allocated, uint64_t *planned) {
    const double live0 = test_live_tensor_mib();
    test_stderr_capture c = test_stderr_begin();
    ds4_session *s = NULL;
    const int rc = ds4_session_create(&s, engine, (int)ctx);
    char *text = test_stderr_end(c);
    TEST_ASSERT(rc == 0 && s != NULL);
    unsigned long long a = 0, p = 0;
    const char *line = strstr(text, "mellum: session allocated ");
    const bool found = line && sscanf(line, "mellum: session allocated %llu bytes, planned %llu bytes", &a, &p) == 2;
    free(text);
    const double live1 = test_live_tensor_mib();
    TEST_ASSERT(fabs((live1 - live0) - (double)a / 1048576.0) < 0.015);
    /* the session runs: the first tokens of a chat prompt (<|im_start|>, "user", "\n") */
    {
        ds4_tokens prompt = {0};
        static const int ids[] = {27, 1397, 233};
        for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); i++) ds4_tokens_push(&prompt, ids[i]);
        char err[256] = {0};
        float *logits = malloc(98304u * sizeof(float));
        TEST_ASSERT(logits != NULL);
        TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
        TEST_ASSERT(logits && ds4_session_copy_logits(s, logits, 98304) == 98304);
        for (uint32_t i = 0; logits && i < 98304u; i++) TEST_ASSERT(isfinite(logits[i]));
        const int top = ds4_session_argmax(s);
        TEST_ASSERT(top >= 0 && top < 98304);
        TEST_ASSERT(ds4_session_eval(s, top, err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_session_pos(s) == 4);
        TEST_ASSERT(logits && ds4_session_copy_logits(s, logits, 98304) == 98304);
        for (uint32_t i = 0; logits && i < 98304u; i++) TEST_ASSERT(isfinite(logits[i]));
        const int next = ds4_session_argmax(s);
        const int via_argmax = ds4_session_eval_argmax(s, next, err, sizeof(err));
        TEST_ASSERT(via_argmax >= 0 && via_argmax < 98304 && via_argmax == ds4_session_argmax(s));
        TEST_ASSERT(ds4_session_pos(s) == 5);
        free(logits);
        ds4_tokens_free(&prompt);
    }
    ds4_session_free(s);
    const double live2 = test_live_tensor_mib();
    TEST_ASSERT(fabs(live2 - live0) < 0.015);
    *allocated = a;
    *planned = p;
    return found;
}

static void test_mellum_session_allocation(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-session: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    static const uint32_t ctxs[] = {20000, 65536, 131072};
    /* the chunk arrives three ways: the host's --prefill-chunk, DS4_QWEN4_PREFILL_CHUNK, neither */
    static const struct { const char *name; uint32_t host; const char *env; uint32_t used; } arms[] = {
        {"default", 0, NULL, 1024}, {"host 4096", 4096, NULL, 1024}, {"env 4096", 0, "4096", 1024},
        {"host 512", 512, NULL, 512},
    };
    for (size_t a = 0; a < sizeof(arms) / sizeof(arms[0]); a++) {
        for (size_t i = 0; i < sizeof(ctxs) / sizeof(ctxs[0]); i++) {
            if (arms[a].env) setenv("DS4_QWEN4_PREFILL_CHUNK", arms[a].env, 1);
            ds4_engine *engine = NULL;
            ds4_engine_options opt = {
                .model_path = model,
                .backend = DS4_BACKEND_METAL,
                .context_size = (int)ctxs[i],
                .prefill_chunk = arms[a].host,
            };
            TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0);
            if (!engine) { unsetenv("DS4_QWEN4_PREFILL_CHUNK"); return; }
            TEST_ASSERT(ds4_engine_prefill_chunk(engine) == (arms[a].host ? arms[a].used : 0u));
            uint64_t allocated = 0, planned = 0;
            TEST_ASSERT(test_capture_session_line(engine, ctxs[i], &allocated, &planned));
            const ds4_context_memory plan = ds4_context_memory_estimate_with_prefill_mode(
                DS4_BACKEND_METAL, (int)ctxs[i], ds4_engine_prefill_chunk(engine), false);
            TEST_ASSERT(planned == plan.total_bytes);
            TEST_ASSERT(allocated == planned);
            TEST_ASSERT(plan.prefill_cap == arms[a].used);
            printf("mellum-session: ctx %u chunk %s allocated %llu planned %llu\n", ctxs[i],
                   arms[a].name, (unsigned long long)allocated, (unsigned long long)planned);
            ds4_engine_close(engine);
            unsetenv("DS4_QWEN4_PREFILL_CHUNK");
        }
    }
}

/* S5 grouped prefill (DS4_TEST_MELLUM_MODEL): a streamed session's group buffer, K x 1,024 rows x 2,304 floats, is
 * in its plan (allocated equals planned) only when K > 1; a K outside 1-64 refuses the streamed engine. */
static void test_mellum_stream_group_memory(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-group-memory: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    char *saved = test_save_env("DS4_MELLUM_STREAM_PREFILL_GROUP");
    static const struct { const char *k; bool opens; } arms[] = { {"1", true}, {"8", true}, {"0", false}, {"2", true} };
    uint64_t allocated[2] = {0};
    for (size_t a = 0; a < sizeof(arms) / sizeof(arms[0]); a++) {
        setenv("DS4_MELLUM_STREAM_PREFILL_GROUP", arms[a].k, 1);
        ds4_engine *engine = NULL;
        ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 8192,
                                   .prefill_chunk = 1024, .ssd_streaming = true, .ssd_streaming_cache_bytes = 6ull << 30 };
        const int rc = ds4_engine_open(&engine, &opt);
        TEST_ASSERT((rc == 0 && engine) == arms[a].opens);
        if (!engine) continue;
        if (a < 2) {
            /* the session line alone: S4's gate and save buffers (about 0.54 MiB) sit outside the plan, so the
             * shared helper's live-tensor check does not apply to streamed sessions */
            test_stderr_capture c = test_stderr_begin();
            ds4_session *s = NULL;
            const int src = ds4_session_create(&s, engine, 8192);
            char *text = test_stderr_end(c);
            TEST_ASSERT(src == 0 && s != NULL);
            unsigned long long al = 0, pl = 0;
            const char *line = text ? strstr(text, "mellum: session allocated ") : NULL;
            TEST_ASSERT(line && sscanf(line, "mellum: session allocated %llu bytes, planned %llu bytes", &al, &pl) == 2);
            free(text);
            if (s) ds4_session_free(s);
            allocated[a] = al;
            const uint64_t planned = pl;
            TEST_ASSERT(allocated[a] == planned);
            printf("mellum-stream-group-memory: K %s allocated %llu planned %llu\n", arms[a].k,
                   (unsigned long long)allocated[a], (unsigned long long)planned);
        }
        ds4_engine_close(engine);
    }
    TEST_ASSERT(allocated[1] - allocated[0] == 75497472ull);
    printf("mellum-stream-group-memory: K 8 adds %lld bytes\n", (long long)(allocated[1] - allocated[0]));
    test_restore_env("DS4_MELLUM_STREAM_PREFILL_GROUP", saved);
}

/* Session rewind (DS4_TEST_MELLUM_MODEL): after a prompt of `len` tokens, decode k tokens recording each step's
 * logits, rewind k, decode the same k again; every step's logits must equal the first pass's, word for word.  One
 * token back keeps both caches (the sliding ring is read before a chunk is committed, so the slot the dropped row
 * overwrote held a position no later query reads).  Further back the ring has lost what the next query needs, but
 * a rewind to where the last sync ended (here `len`) restores the rings saved there: the graph stays at the rewound
 * position either way.  A deeper rewind anywhere else resets (test_mellum_rewind_fresh_case). */
bool ds4_test_session_graph_pos(ds4_session *s, uint32_t *pos);   /* ds4.c test hook */

static int test_mellum_token(int i) {
    return 40 + (int)(((uint32_t)i * 7919u) % 90000u);   /* ordinary byte-level and merge ids */
}

/* S4 streamed decode (DS4_TEST_MELLUM_MODEL): a session run from a bounded expert-slot cache must give the
 * resident session's logits word for word at every step.  The prompt is fed one token at a time (decode only;
 * streamed prefill is its own test), then 64 greedy tokens.  At a 64-slot cache nearly every token stops. */
bool ds4_test_mellum_stream_stops(ds4_session *s, uint64_t *stops);   /* ds4.c test hook */
bool ds4_test_mellum_stream_loads(ds4_session *s, uint64_t *loads);   /* ds4.c test hook: experts read into slots */

enum { MELLUM_STREAM_PROMPT = 24, MELLUM_STREAM_GEN = 64, MELLUM_STREAM_STEPS = MELLUM_STREAM_PROMPT + MELLUM_STREAM_GEN };

static int mellum_stream_argmax(const float *l) {
    int best = 0;
    for (int i = 1; i < 98304; i++) if (l[i] > l[best]) best = i;
    return best;
}

/* toks[0..steps) are fed; when record, toks[PROMPT..) are chosen greedily and logits[step] kept; otherwise
 * each step's logits are compared with logits[step].  Returns the number of differing words, or -1. */
static int64_t mellum_stream_run(const char *model, bool streaming, uint32_t cache_experts, uint64_t cache_bytes,
                                 int *toks, float *logits, bool record, uint64_t *stops, uint32_t *stopped_tokens,
                                 uint64_t *loads) {
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 8192,
        .ssd_streaming = streaming, .ssd_streaming_cache_experts = cache_experts, .ssd_streaming_cache_bytes = cache_bytes,
    };
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) return -1;
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, 8192) != 0 || !s) { ds4_engine_close(engine); return -1; }
    char err[256] = {0};
    float *got = malloc(98304u * sizeof(float));
    int64_t diff = 0;
    uint64_t last = 0;
    if (stopped_tokens) *stopped_tokens = 0;
    for (int i = 0; i < MELLUM_STREAM_STEPS && diff >= 0; i++) {
        int rc;
        if (i == 0) {
            ds4_tokens one = {0};
            ds4_tokens_push(&one, toks[0]);
            rc = ds4_session_sync(s, &one, err, sizeof(err));
            ds4_tokens_free(&one);
        } else {
            rc = ds4_session_eval(s, toks[i], err, sizeof(err));
        }
        if (rc != 0 || ds4_session_copy_logits(s, got, 98304) != 98304) { fprintf(stderr, "mellum-stream: step %d: %s\n", i, err); diff = -1; break; }
        float *want = logits + (size_t)i * 98304u;
        if (record) memcpy(want, got, 98304u * sizeof(float));
        else for (uint32_t w = 0; w < 98304u; w++) diff += memcmp(&got[w], &want[w], sizeof(float)) != 0;
        if (record && i + 1 >= MELLUM_STREAM_PROMPT && i + 1 < MELLUM_STREAM_STEPS) toks[i + 1] = mellum_stream_argmax(got);
        if (streaming && stops) {
            uint64_t now = 0;
            if (!ds4_test_mellum_stream_stops(s, &now)) { diff = -1; break; }
            if (stopped_tokens && now > last) (*stopped_tokens)++;
            last = now;
        }
    }
    if (stops) *stops = last;
    if (loads && streaming && !ds4_test_mellum_stream_loads(s, loads)) diff = -1;
    free(got);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return diff;
}

/* S4 streamed prefill (DS4_TEST_MELLUM_MODEL): a prompt synced in one call (chunks of up to 1,024 rows, the
 * tiled/E1 experts at 32 rows or more, the exact path below) gives the resident session's last-row logits
 * word for word, at 6 GiB and at a 64-slot cache that holds no more than one layer's experts. */
static int64_t mellum_stream_prefill(const char *model, bool streaming, uint32_t cache_experts, uint64_t cache_bytes,
                                     int n, float *logits, bool record) {
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 8192,
        .ssd_streaming = streaming, .ssd_streaming_cache_experts = cache_experts, .ssd_streaming_cache_bytes = cache_bytes,
    };
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) return -1;
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, 8192) != 0 || !s) { ds4_engine_close(engine); return -1; }
    ds4_tokens prompt = {0};
    for (int i = 0; i < n; i++) ds4_tokens_push(&prompt, test_mellum_token(i));
    char err[256] = {0};
    float *got = malloc(98304u * sizeof(float));
    int64_t diff = -1;
    if (ds4_session_sync(s, &prompt, err, sizeof(err)) == 0 && ds4_session_copy_logits(s, got, 98304) == 98304) {
        diff = 0;
        if (record) memcpy(logits, got, 98304u * sizeof(float));
        else for (uint32_t w = 0; w < 98304u; w++) diff += memcmp(&got[w], &logits[w], sizeof(float)) != 0;
    } else {
        fprintf(stderr, "mellum-stream-prefill: %d tokens: %s\n", n, err);
    }
    free(got);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return diff;
}

static void test_mellum_stream_prefill_identity(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-prefill: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    float *ref = malloc(98304u * sizeof(float));
    TEST_ASSERT(ref != NULL);
    if (!ref) return;
    static const int lens[] = { 2200, 31 };
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
        TEST_ASSERT(mellum_stream_prefill(model, false, 0, 0, lens[i], ref, true) == 0);
        static const struct { const char *name; uint32_t experts; uint64_t bytes; } arms[] = {
            { "6 GiB", 0, 6ull << 30 }, { "64 slots", 64, 0 },
        };
        for (size_t a = 0; a < sizeof(arms) / sizeof(arms[0]); a++) {
            const int64_t diff = mellum_stream_prefill(model, true, arms[a].experts, arms[a].bytes, lens[i], ref, false);
            TEST_ASSERT(diff == 0);
            printf("mellum-stream-prefill: %d tokens, %s: %lld of 98304 last-row logit words differ\n", lens[i],
                   arms[a].name, (long long)diff);
        }
    }
    free(ref);
}

/* S4 failure paths (DS4_TEST_MELLUM_MODEL), at 64 slots so every token stops: a streamed load that fails, and a
 * cancellation between stops, each make that eval fail promptly; re-syncing the prefix then recovers, and the
 * rest of the run gives the resident logits word for word. */
bool ds4_test_mellum_stream_fail_load(ds4_session *s, uint64_t nth);   /* ds4.c test hook: fail the nth load from now */

static bool mellum_stream_always_cancel(void *ud) {
    (void)ud;
    return true;
}

bool ds4_test_mellum_stream_stop_scratch(ds4_session *s, bool on, uint64_t *intact, uint64_t *broken);

/* Nothing runs past a streamed stop: at every stop, scratch ATTN_RES and FFN_NORM still hold what the check kernel
 * saved for the stopped layer (a later layer's ungated dispatch would overwrite them).  The decode equals resident. */
static void test_mellum_stream_stop_scratch(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-stop-scratch: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    enum { N = 8 };
    float *ref = malloc((size_t)N * 98304u * sizeof(float)), *got = malloc(98304u * sizeof(float));
    TEST_ASSERT(ref && got);
    if (!ref || !got) return;
    char err[256] = {0};
    for (int pass = 0; pass < 2; pass++) {
        const bool streaming = pass == 1;
        ds4_engine *engine = NULL;
        ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 8192,
                                   .ssd_streaming = streaming, .ssd_streaming_cache_experts = streaming ? 64u : 0u };
        TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine);
        if (!engine) break;
        ds4_session *s = NULL;
        TEST_ASSERT(ds4_session_create(&s, engine, 8192) == 0 && s);
        ds4_tokens prefix = {0};
        uint64_t diff = 0, intact = 0, broken = 0;
        ds4_tokens_push(&prefix, test_mellum_token(0));
        TEST_ASSERT(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
        if (streaming) TEST_ASSERT(ds4_test_mellum_stream_stop_scratch(s, true, NULL, NULL));
        for (int i = 1; i < N; i++) {
            TEST_ASSERT(ds4_session_eval(s, test_mellum_token(i), err, sizeof(err)) == 0);
            TEST_ASSERT(ds4_session_copy_logits(s, got, 98304) == 98304);
            if (!streaming) memcpy(ref + (size_t)i * 98304u, got, 98304u * sizeof(float));
            else for (uint32_t w = 0; w < 98304u; w++) diff += memcmp(&got[w], &ref[(size_t)i * 98304u + w], sizeof(float)) != 0;
        }
        if (streaming) {
            TEST_ASSERT(ds4_test_mellum_stream_stop_scratch(s, false, &intact, &broken));
            TEST_ASSERT(intact > 0);
            TEST_ASSERT(broken == 0);
            TEST_ASSERT(diff == 0);
            printf("mellum-stream-stop-scratch: %llu stops intact, %llu overwritten after the trip; %llu logit words differ\n",
                   (unsigned long long)intact, (unsigned long long)broken, (unsigned long long)diff);
        }
        ds4_tokens_free(&prefix);
        ds4_session_free(s);
        ds4_engine_close(engine);
    }
    free(ref);
    free(got);
}

bool ds4_test_mellum_stream_fail_read(ds4_session *s, uint64_t nth);   /* ds4.c test hook: fail the nth expert read */
bool ds4_test_mellum_stream_slots(ds4_session *s, uint32_t *free_slots, uint32_t *held_slots, uint32_t *handed_out);

/* A real expert read fails (the pread itself, not the host hook above): every slab slot the cache handed out is
 * afterwards on the free list or held by a resident expert, exactly once, and a re-sync of the prefix gives resident
 * logits.  Passes: resident; streamed without a fault (the sibling); a fault in a decode load on the early-load path;
 * the same with early load off (the synchronous path); and a fault in a prefill (a 6-token sync, loads of more than
 * one group). */
static void test_mellum_stream_read_fault(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-read-fault: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    enum { N = 10 };
    int toks[N];
    for (int i = 0; i < N; i++) toks[i] = test_mellum_token(i);
    float *ref = malloc((size_t)N * 98304u * sizeof(float)), *got = malloc(98304u * sizeof(float));
    TEST_ASSERT(ref && got);
    if (!ref || !got) return;
    char err[256] = {0};
    static const char *names[] = { "resident", "no fault", "a failed decode read (early load)",
                                   "a failed decode read (synchronous)", "a failed prefill read" };
    for (int pass = 0; pass < 5; pass++) {
        const bool streaming = pass > 0, fault = pass >= 2, prefill = pass == 4;
        if (pass == 3) setenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD", "1", 1);
        ds4_engine *engine = NULL;
        ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 8192,
                                   .ssd_streaming = streaming, .ssd_streaming_cache_experts = streaming ? 64u : 0u };
        TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine);
        if (!engine) break;
        ds4_session *s = NULL;
        TEST_ASSERT(ds4_session_create(&s, engine, 8192) == 0 && s);
        ds4_tokens prefix = {0};
        uint64_t diff = 0;
        uint32_t lost = 0, doubled = 0;
        for (int i = 0; i < N; i++) {
            ds4_tokens_push(&prefix, toks[i]);
            if (prefill && i < 5) continue;   /* the first sync is the 6-token prefix, a prefill */
            if (fault && i == 5) {
                TEST_ASSERT(ds4_test_mellum_stream_fail_read(s, 2));
                TEST_ASSERT(ds4_session_sync(s, &prefix, err, sizeof(err)) != 0);
                TEST_ASSERT(ds4_test_mellum_stream_fail_read(s, 0));
            }
            TEST_ASSERT(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
            TEST_ASSERT(ds4_session_copy_logits(s, got, 98304) == 98304);
            if (!streaming) memcpy(ref + (size_t)i * 98304u, got, 98304u * sizeof(float));
            else for (uint32_t w = 0; w < 98304u; w++) diff += memcmp(&got[w], &ref[(size_t)i * 98304u + w], sizeof(float)) != 0;
            if (streaming) {
                uint32_t f = 0, h = 0, handed = 0;
                TEST_ASSERT(ds4_test_mellum_stream_slots(s, &f, &h, &handed));
                if (f + h < handed && handed - f - h > lost) lost = handed - f - h;
                if (f + h > handed && f + h - handed > doubled) doubled = f + h - handed;
            }
        }
        if (streaming) {
            TEST_ASSERT(diff == 0);
            TEST_ASSERT(lost == 0);
            TEST_ASSERT(doubled == 0);
            printf("mellum-stream-read-fault: %s, %llu logit words differ, %u slab slots lost, %u counted twice\n",
                   names[pass], (unsigned long long)diff, lost, doubled);
        }
        if (pass == 3) unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD");
        ds4_tokens_free(&prefix);
        ds4_session_free(s);
        ds4_engine_close(engine);
    }
    free(ref);
    free(got);
}

static void test_mellum_stream_faults(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-faults: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    enum { N = 12 };
    int toks[N];
    for (int i = 0; i < N; i++) toks[i] = test_mellum_token(i);
    float *ref = malloc((size_t)N * 98304u * sizeof(float)), *got = malloc(98304u * sizeof(float));
    TEST_ASSERT(ref && got);
    if (!ref || !got) return;
    char err[256] = {0};
    {
        /* every expert must sit in a slab slot the kernels make resident: with slabs off, a streamed sync
         * refuses cleanly instead of reading buffers it never made resident */
        setenv("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS", "1", 1);
        ds4_engine *engine = NULL;
        ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 8192,
                                   .ssd_streaming = true, .ssd_streaming_cache_experts = 64u };
        TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine);
        if (engine) {
            ds4_session *s = NULL;
            TEST_ASSERT(ds4_session_create(&s, engine, 8192) == 0 && s);
            ds4_tokens one = {0};
            ds4_tokens_push(&one, toks[0]);
            if (s) TEST_ASSERT(ds4_session_sync(s, &one, err, sizeof(err)) != 0);
            ds4_tokens_free(&one);
            if (s) ds4_session_free(s);
            ds4_engine_close(engine);
        }
        unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS");
    }
    for (int pass = 0; pass < 2; pass++) {
        const bool streaming = pass == 1;
        ds4_engine *engine = NULL;
        ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 8192,
                                   .ssd_streaming = streaming, .ssd_streaming_cache_experts = streaming ? 64u : 0u };
        TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine);
        if (!engine) break;
        ds4_session *s = NULL;
        TEST_ASSERT(ds4_session_create(&s, engine, 8192) == 0 && s);
        ds4_tokens prefix = {0};
        uint64_t diff = 0;
        for (int i = 0; i < N; i++) {
            ds4_tokens_push(&prefix, toks[i]);
            if (streaming && i == 4) {
                /* a load fails mid-token: the eval reports it, and a re-sync of the prefix recovers */
                TEST_ASSERT(ds4_test_mellum_stream_fail_load(s, 3));
                TEST_ASSERT(ds4_session_sync(s, &prefix, err, sizeof(err)) != 0);
                TEST_ASSERT(ds4_test_mellum_stream_fail_load(s, 0));
            }
            if (streaming && i == 8) {
                /* a cancellation between stops ends the eval (the session's sync checks cancellation itself
                 * before any work, so the eval is what reaches the stop loop); clearing it and re-syncing
                 * recovers */
                ds4_session_set_cancel(s, mellum_stream_always_cancel, NULL);
                TEST_ASSERT(ds4_session_eval(s, toks[i], err, sizeof(err)) != 0);
                ds4_session_set_cancel(s, NULL, NULL);
            }
            TEST_ASSERT(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
            TEST_ASSERT(ds4_session_copy_logits(s, got, 98304) == 98304);
            if (!streaming) memcpy(ref + (size_t)i * 98304u, got, 98304u * sizeof(float));
            else for (uint32_t w = 0; w < 98304u; w++) diff += memcmp(&got[w], &ref[(size_t)i * 98304u + w], sizeof(float)) != 0;
        }
        if (streaming) {
            TEST_ASSERT(diff == 0);
            printf("mellum-stream-faults: after a failed load and a cancellation, %llu of %d logit words differ\n",
                   (unsigned long long)diff, N * 98304);
        }
        ds4_tokens_free(&prefix);
        ds4_session_free(s);
        ds4_engine_close(engine);
    }
    free(ref);
    free(got);
}

/* S4 review minor (DS4_TEST_MELLUM_MODEL): ds4_engine_close(NULL) is a no-op, so it must leave an open
 * streamed engine's uncached model descriptor alone; that engine's next loads still read. */
static void test_mellum_stream_close_null(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-close: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 8192,
                               .ssd_streaming = true, .ssd_streaming_cache_experts = 64u };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine);
    if (!engine) return;
    ds4_session *s = NULL;
    TEST_ASSERT(ds4_session_create(&s, engine, 8192) == 0 && s);
    ds4_engine_close(NULL);
    char err[256] = {0};
    ds4_tokens prefix = {0};
    for (int i = 0; i < 4; i++) ds4_tokens_push(&prefix, test_mellum_token(i));
    uint64_t loads = 0;
    if (s) {
        TEST_ASSERT(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_test_mellum_stream_loads(s, &loads) && loads > 0);
        printf("mellum-stream-close: after ds4_engine_close(NULL), the sync %s with %llu loads%s%s\n",
               err[0] ? "failed" : "succeeded", (unsigned long long)loads, err[0] ? ": " : "", err);
        ds4_session_free(s);
    }
    ds4_tokens_free(&prefix);
    ds4_engine_close(engine);
}

/* S5 grouped prefill (DS4_TEST_MELLUM_MODEL): one run of `n` test tokens with prefill chunk `chunk`, context `ctx`
 * and DS4_MELLUM_STREAM_PREFILL_GROUP `k` (NULL: unset).  n_first > 0 syncs that many tokens first (a warm
 * append); two_sessions syncs the prompt in one session, frees it, and measures a second; rewind_to > 0 rewinds
 * the synced session there and syncs the prompt again.  Records the last
 * row's logits (record) or returns how many words differ from them; *loads gets the measured session's loads. */
static int64_t mellum_group_run(const char *model, const char *k, uint32_t experts, uint64_t bytes, uint32_t chunk,
                                int ctx, int n_first, int n, bool two_sessions, int rewind_to, float *logits, bool record,
                                uint64_t *loads) {
    char *saved = test_save_env("DS4_MELLUM_STREAM_PREFILL_GROUP");
    if (k) setenv("DS4_MELLUM_STREAM_PREFILL_GROUP", k, 1);
    else unsetenv("DS4_MELLUM_STREAM_PREFILL_GROUP");
    const bool streaming = experts || bytes;
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = ctx, .prefill_chunk = chunk,
                               .ssd_streaming = streaming, .ssd_streaming_cache_experts = experts,
                               .ssd_streaming_cache_bytes = bytes };
    int64_t diff = -1;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) { test_restore_env("DS4_MELLUM_STREAM_PREFILL_GROUP", saved); return -1; }
    ds4_tokens first = {0}, prompt = {0};
    for (int i = 0; i < n; i++) {
        ds4_tokens_push(&prompt, test_mellum_token(i));
        if (i < n_first) ds4_tokens_push(&first, test_mellum_token(i));
    }
    char err[256] = {0};
    float *got = malloc(98304u * sizeof(float));
    bool ok = got != NULL;
    for (int pass = 0; ok && pass < (two_sessions ? 2 : 1); pass++) {
        ds4_session *s = NULL;
        ok = ds4_session_create(&s, engine, ctx) == 0 && s;
        if (ok && n_first > 0) ok = ds4_session_sync(s, &first, err, sizeof(err)) == 0;
        ok = ok && ds4_session_sync(s, &prompt, err, sizeof(err)) == 0;
        if (ok && rewind_to > 0) {
            ds4_session_rewind(s, rewind_to);
            ok = ds4_session_sync(s, &prompt, err, sizeof(err)) == 0;
        }
        ok = ok && ds4_session_copy_logits(s, got, 98304) == 98304;
        if (ok && loads) ok = !streaming ? (*loads = 0, true) : ds4_test_mellum_stream_loads(s, loads);
        if (s) ds4_session_free(s);
    }
    if (ok) {
        diff = 0;
        if (record) memcpy(logits, got, 98304u * sizeof(float));
        else for (uint32_t w = 0; w < 98304u; w++) diff += memcmp(&got[w], &logits[w], sizeof(float)) != 0;
    } else {
        fprintf(stderr, "mellum-stream-group: %d tokens: %s\n", n, err);
    }
    free(got);
    ds4_tokens_free(&first);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    test_restore_env("DS4_MELLUM_STREAM_PREFILL_GROUP", saved);
    return diff;
}

/* S5 grouped layer-major prefill: streamed syncs in groups of K chunks give the resident logits word for word
 * (several groups, a one-row tail, one partial group at the default K, the context limit, a warm append, a
 * second session), and K = 2 loads fewer experts than K = 1. */
static void test_mellum_stream_group(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-group: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    float *ref = malloc(98304u * sizeof(float));
    TEST_ASSERT(ref != NULL);
    if (!ref) return;
    static const struct { const char *name; const char *k; uint32_t chunk; int ctx, n_first, n; bool two, six_gib; int rewind; } cases[] = {
        { "1,281 tokens, chunk 256, K 2 (one-row tail)", "2", 256, 8192, 0, 1281, false, true, 0 },
        { "1,281 tokens, chunk 256, K 2 (one-row tail)", "2", 256, 8192, 0, 1281, false, false, 0 },
        { "1,100 tokens, chunk 256, K 2", "2", 256, 8192, 0, 1100, false, true, 0 },
        { "1,100 tokens, chunk 256, K 2", "2", 256, 8192, 0, 1100, false, false, 0 },
        { "2,200 tokens, chunk 1,024, default K", NULL, 1024, 8192, 0, 2200, false, true, 0 },
        { "2,047 tokens in context 2,048, chunk 256, K 2", "2", 256, 2048, 0, 2047, false, false, 0 },
        { "700 then 1,281 tokens, chunk 256, K 2", "2", 256, 8192, 700, 1281, false, false, 0 },
        { "a second session, 1,100 tokens, chunk 256, K 2", "2", 256, 8192, 0, 1100, true, false, 0 },
        { "1,100 tokens, rewind to 1,000, sync again, chunk 256, K 2", "2", 256, 8192, 0, 1100, false, false, 1000 },
        { "6 tokens, chunk 1, K 2 (every chunk one row)", "2", 1, 8192, 0, 6, false, false, 0 },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        TEST_ASSERT(mellum_group_run(model, NULL, 0, 0, cases[i].chunk, cases[i].ctx, cases[i].n_first, cases[i].n, false,
                                     cases[i].rewind, ref, true, NULL) == 0);
        const int64_t diff = mellum_group_run(model, cases[i].k, cases[i].six_gib ? 0u : 64u, cases[i].six_gib ? 6ull << 30 : 0u,
                                              cases[i].chunk, cases[i].ctx, cases[i].n_first, cases[i].n, cases[i].two,
                                              cases[i].rewind, ref, false, NULL);
        TEST_ASSERT(diff == 0);
        printf("mellum-stream-group: %s, %s: %lld of 98304 last-row logit words differ\n", cases[i].name,
               cases[i].six_gib ? "6 GiB" : "64 slots", (long long)diff);
    }
    uint64_t loads1 = 0, loads2 = 0;
    TEST_ASSERT(mellum_group_run(model, NULL, 0, 0, 256, 8192, 0, 1100, false, 0, ref, true, NULL) == 0);
    TEST_ASSERT(mellum_group_run(model, "1", 64, 0, 256, 8192, 0, 1100, false, 0, ref, false, &loads1) == 0);
    TEST_ASSERT(mellum_group_run(model, "2", 64, 0, 256, 8192, 0, 1100, false, 0, ref, false, &loads2) == 0);
    TEST_ASSERT(loads2 < loads1);
    printf("mellum-stream-group: 1,100 tokens at 64 slots load %llu experts at K 1 and %llu at K 2\n",
           (unsigned long long)loads1, (unsigned long long)loads2);
    free(ref);
}

/* S5 grouped prefill's events (DS4_TEST_MELLUM_MODEL), 1,100 tokens at chunk 256, K 2, 64 slots, so three
 * groups (512, 512, 76 rows): prefill_chunk once per group, prefill_display after each of a group's 28 layer
 * passes, cancellation only between groups (one asked for inside the second group lets it finish: the checkpoint
 * is 1,024), and a load that fails inside the second group fails the sync and invalidates the checkpoint, though
 * the first group had finished; a re-sync then matches resident. */
typedef struct {
    int chunks[8], n_chunks, displays, last_display, cancel_at_display;
    bool ascending, cancel, fail_after_first_chunk;
    ds4_session *s;
} mellum_group_events;

static void mellum_group_on_progress(void *ud, const char *event, int current, int total) {
    (void)total;
    mellum_group_events *ev = ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (ev->n_chunks < 8) ev->chunks[ev->n_chunks] = current;
    ev->n_chunks++;
    if (ev->fail_after_first_chunk && ev->n_chunks == 1) TEST_ASSERT(ds4_test_mellum_stream_fail_load(ev->s, 3));
}

static void mellum_group_on_display(void *ud, const char *event, int current, int total) {
    (void)total;
    mellum_group_events *ev = ud;
    if (strcmp(event, "prefill_display")) return;
    if (current < ev->last_display) ev->ascending = false;
    ev->last_display = current;
    ev->displays++;
    if (ev->cancel_at_display && ev->displays == ev->cancel_at_display) ev->cancel = true;
}

static bool mellum_group_cancel(void *ud) {
    return ((mellum_group_events *)ud)->cancel;
}

static void test_mellum_stream_group_events(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-group-events: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    float *ref = malloc(98304u * sizeof(float)), *got = malloc(98304u * sizeof(float));
    TEST_ASSERT(ref && got);
    if (!ref || !got) return;
    TEST_ASSERT(mellum_group_run(model, NULL, 0, 0, 256, 8192, 0, 1100, false, 0, ref, true, NULL) == 0);
    char *saved = test_save_env("DS4_MELLUM_STREAM_PREFILL_GROUP");
    setenv("DS4_MELLUM_STREAM_PREFILL_GROUP", "2", 1);
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 8192, .prefill_chunk = 256,
                               .ssd_streaming = true, .ssd_streaming_cache_experts = 64u };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine);
    ds4_tokens prompt = {0};
    for (int i = 0; i < 1100; i++) ds4_tokens_push(&prompt, test_mellum_token(i));
    char err[256] = {0};
    for (int arm = 0; engine && arm < 3; arm++) {
        ds4_session *s = NULL;
        TEST_ASSERT(ds4_session_create(&s, engine, 8192) == 0 && s);
        if (!s) break;
        /* arm 1 asks to cancel at the second group's 10th layer pass; arm 2 fails its 3rd load after the first group */
        mellum_group_events ev = { .ascending = true, .cancel_at_display = arm == 1 ? 28 + 10 : 0,
                                   .fail_after_first_chunk = arm == 2, .s = s };
        ds4_session_set_progress(s, mellum_group_on_progress, &ev);
        ds4_session_set_display_progress(s, mellum_group_on_display, &ev);
        ds4_session_set_cancel(s, mellum_group_cancel, &ev);
        if (arm == 0) {
            TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
            TEST_ASSERT(ev.n_chunks == 3 && ev.chunks[0] == 512 && ev.chunks[1] == 1024 && ev.chunks[2] == 1100);
            TEST_ASSERT(ev.displays == 3 * 28 && ev.ascending && ev.last_display == 1100);
            printf("mellum-stream-group-events: %d prefill_chunk events (%d, %d, %d), %d prefill_display events\n",
                   ev.n_chunks, ev.chunks[0], ev.chunks[1], ev.chunks[2], ev.displays);
        } else if (arm == 1) {
            const int rc = ds4_session_sync(s, &prompt, err, sizeof(err));
            TEST_ASSERT(rc == DS4_SESSION_SYNC_INTERRUPTED);
            TEST_ASSERT(ds4_session_pos(s) == 1024 && ds4_session_checkpoint_valid(s));
            printf("mellum-stream-group-events: cancelled inside the second group: rc %d, pos %d\n", rc, ds4_session_pos(s));
            ev.cancel_at_display = 0;
            ev.cancel = false;
            TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
        } else {
            TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) != 0);
            TEST_ASSERT(ev.n_chunks == 1 && !ds4_session_checkpoint_valid(s));
            printf("mellum-stream-group-events: a failed load inside the second group: %s; checkpoint valid %d\n", err,
                   ds4_session_checkpoint_valid(s));
            ev.fail_after_first_chunk = false;
            TEST_ASSERT(ds4_test_mellum_stream_fail_load(s, 0));
            TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
        }
        if (arm > 0) {
            TEST_ASSERT(ds4_session_copy_logits(s, got, 98304) == 98304);
            uint64_t diff = 0;
            for (uint32_t w = 0; w < 98304u; w++) diff += memcmp(&got[w], &ref[w], sizeof(float)) != 0;
            TEST_ASSERT(diff == 0);
            printf("mellum-stream-group-events: %s, then a re-sync: %llu of 98304 words differ\n",
                   arm == 1 ? "cancelled" : "failed", (unsigned long long)diff);
        }
        ds4_session_free(s);
    }
    ds4_tokens_free(&prompt);
    if (engine) ds4_engine_close(engine);
    test_restore_env("DS4_MELLUM_STREAM_PREFILL_GROUP", saved);
    free(ref);
    free(got);
}

static void test_mellum_stream_identity(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-identity: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    int toks[MELLUM_STREAM_STEPS];
    for (int i = 0; i < MELLUM_STREAM_PROMPT; i++) toks[i] = test_mellum_token(i);
    float *ref = malloc((size_t)MELLUM_STREAM_STEPS * 98304u * sizeof(float));
    TEST_ASSERT(ref != NULL);
    if (!ref) return;
    TEST_ASSERT(mellum_stream_run(model, false, 0, 0, toks, ref, true, NULL, NULL, NULL) == 0);
    static const struct { const char *name; uint32_t experts; uint64_t bytes; } arms[] = {
        { "6 GiB", 0, 6ull << 30 }, { "64 slots", 64, 0 },
    };
    for (size_t a = 0; a < sizeof(arms) / sizeof(arms[0]); a++) {
        uint64_t stops = 0, loads = 0;
        uint32_t stopped = 0;
        const int64_t diff = mellum_stream_run(model, true, arms[a].experts, arms[a].bytes, toks, ref, false, &stops, &stopped,
                                               &loads);
        TEST_ASSERT(diff == 0);
        /* every stop found a missing expert, so it read at least one into a slot */
        TEST_ASSERT(loads >= stops && loads > 0);
        if (arms[a].experts == 64) TEST_ASSERT(stopped * 2u >= MELLUM_STREAM_STEPS);
        printf("mellum-stream-identity: %s: %lld of %d logit words differ; %llu stops over %d tokens, %u tokens stopped, "
               "%llu experts loaded\n", arms[a].name, (long long)diff, MELLUM_STREAM_STEPS * 98304, (unsigned long long)stops,
               MELLUM_STREAM_STEPS, stopped, (unsigned long long)loads);
    }
    free(ref);
}

/* A stop whose reads the early-load path does not start (here: turned off) must still load before the resumed
 * pass is encoded: with 16 MiB slabs a decode stop can allocate a slab, and a pass encoded before that slab
 * existed would not have made it resident (it would read the stopped layer's experts from it regardless). */
static void test_mellum_stream_slabs(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-stream-slabs: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    int toks[MELLUM_STREAM_STEPS];
    for (int i = 0; i < MELLUM_STREAM_PROMPT; i++) toks[i] = test_mellum_token(i);
    float *ref = malloc((size_t)MELLUM_STREAM_STEPS * 98304u * sizeof(float));
    TEST_ASSERT(ref != NULL);
    if (!ref) return;
    TEST_ASSERT(mellum_stream_run(model, false, 0, 0, toks, ref, true, NULL, NULL, NULL) == 0);
    char *early = test_save_env("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD");
    char *slab = test_save_env("DS4_METAL_STREAMING_EXPERT_SLAB_MB");
    setenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD", "1", 1);
    setenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB", "16", 1);
    uint64_t stops = 0, loads = 0;
    uint32_t stopped = 0;
    const int64_t diff = mellum_stream_run(model, true, 64, 0, toks, ref, false, &stops, &stopped, &loads);
    test_restore_env("DS4_METAL_STREAMING_EXPERT_SLAB_MB", slab);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD", early);
    TEST_ASSERT(diff == 0 && stops > 0);
    printf("mellum-stream-slabs: early load off, 16 MiB slabs, 64 slots: %lld of %d logit words differ; %llu stops\n",
           (long long)diff, MELLUM_STREAM_STEPS * 98304, (unsigned long long)stops);
    free(ref);
}

static void test_mellum_rewind_case(ds4_engine *engine, int len, int k) {
    ds4_session *s = NULL;
    TEST_ASSERT(ds4_session_create(&s, engine, 4096) == 0 && s != NULL);
    if (!s) return;
    ds4_tokens prompt = {0};
    for (int i = 0; i < len; i++) ds4_tokens_push(&prompt, test_mellum_token(i));
    char err[256] = {0};
    float *first = malloc((size_t)k * 98304u * sizeof(float)), *again = malloc(98304u * sizeof(float));
    TEST_ASSERT(first && again);
    TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    for (int j = 0; j < k; j++) {
        TEST_ASSERT(ds4_session_eval(s, test_mellum_token(len + j), err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_session_copy_logits(s, first + (size_t)j * 98304u, 98304) == 98304);
    }
    ds4_session_rewind(s, len);
    TEST_ASSERT(ds4_session_pos(s) == len);
    uint32_t graph_pos = UINT32_MAX;
    TEST_ASSERT(ds4_test_session_graph_pos(s, &graph_pos));
    TEST_ASSERT(graph_pos == (uint32_t)len);   /* kept (k == 1) or restored from the sync's end, not reset */
    uint64_t diff = 0;
    for (int j = 0; j < k; j++) {
        TEST_ASSERT(ds4_session_eval(s, test_mellum_token(len + j), err, sizeof(err)) == 0);
        TEST_ASSERT(ds4_session_copy_logits(s, again, 98304) == 98304);
        for (uint32_t i = 0; i < 98304u; i++) diff += memcmp(&again[i], &first[(size_t)j * 98304u + i], sizeof(float)) != 0;
    }
    TEST_ASSERT(diff == 0);
    TEST_ASSERT(ds4_session_pos(s) == len + k);
    printf("mellum-rewind: prompt %d, k %d: graph at %u after the rewind (%s); %llu of %d logit words differ from the "
           "straight decode\n", len, k, graph_pos, k == 1 ? "both caches kept" : "rings restored from the sync's end",
           (unsigned long long)diff, k * 98304);
    free(first); free(again);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
}

/* A rewind deeper than one token resets the session, and the next eval prefills the kept tokens again: the
 * logits after it are a fresh session's synced to the kept tokens, byte for byte, which are the logits the
 * session gave before only when the kept tokens were evaluated the same way the first time.  The review's case:
 * sync 31 (an exact chunk), eval 3, rewind to 32; the replay is one 32-row chunk, which takes the batched and
 * tiled paths, so the next eval equals the fresh session's and differs from the straight run's. */
static void test_mellum_rewind_fresh_case(ds4_engine *engine, int len, int k, int to) {
    float *rewound = malloc(98304u * sizeof(float)), *fresh = malloc(98304u * sizeof(float));
    float *straight = malloc(98304u * sizeof(float));
    TEST_ASSERT(rewound && fresh && straight);
    char err[256] = {0};
    ds4_tokens prompt = {0}, kept = {0};
    for (int i = 0; i < len; i++) ds4_tokens_push(&prompt, test_mellum_token(i));
    for (int i = 0; i < to; i++) ds4_tokens_push(&kept, test_mellum_token(i));
    ds4_session *s = NULL;
    TEST_ASSERT(ds4_session_create(&s, engine, 4096) == 0 && s != NULL);
    if (!s || !rewound || !fresh || !straight) return;
    TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    for (int j = 0; j < k; j++) {
        TEST_ASSERT(ds4_session_eval(s, test_mellum_token(len + j), err, sizeof(err)) == 0);
        if (len + j == to) TEST_ASSERT(ds4_session_copy_logits(s, straight, 98304) == 98304);
    }
    ds4_session_rewind(s, to);
    TEST_ASSERT(ds4_session_pos(s) == to);
    TEST_ASSERT(ds4_session_eval(s, test_mellum_token(to), err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_copy_logits(s, rewound, 98304) == 98304);
    ds4_session_free(s);
    s = NULL;
    TEST_ASSERT(ds4_session_create(&s, engine, 4096) == 0 && s != NULL);
    if (!s) return;
    TEST_ASSERT(ds4_session_sync(s, &kept, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_eval(s, test_mellum_token(to), err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_copy_logits(s, fresh, 98304) == 98304);
    uint64_t d_fresh = 0, d_straight = 0;
    for (uint32_t i = 0; i < 98304u; i++) {
        d_fresh += memcmp(&rewound[i], &fresh[i], sizeof(float)) != 0;
        d_straight += memcmp(&rewound[i], &straight[i], sizeof(float)) != 0;
    }
    printf("mellum-rewind: sync %d, eval %d, rewind to %d, eval: %llu of 98304 logit words differ from a fresh sync of "
           "%d, %llu from the straight run\n", len, k, to, (unsigned long long)d_fresh, to, (unsigned long long)d_straight);
    TEST_ASSERT(d_fresh == 0);
    TEST_ASSERT(d_straight > 0);   /* the replay's 32-row chunk is not the straight run's arithmetic */
    free(rewound); free(fresh); free(straight);
    ds4_tokens_free(&prompt); ds4_tokens_free(&kept);
    ds4_session_free(s);
}

/* The rewind frontier (DS4_TEST_MELLUM_MODEL): where a sync ends the session saves its sliding rings and logits, so
 * a rewind there after k decoded tokens (a thinking answer the next request's template drops) is exact at any length:
 * the graph stays at len with the sync's logits, and a next sync extending the prompt equals a fresh session's sync
 * of the prompt and then of the extension, word for word.  The sibling: a rewind below the frontier and a different
 * token there make the saved rings stale, so a later rewind to the old frontier must reset, not restore them. */
static void test_mellum_frontier_case(ds4_engine *engine, int len, int k) {
    float *at_sync = malloc(98304u * sizeof(float)), *rewound = malloc(98304u * sizeof(float));
    float *next = malloc(98304u * sizeof(float)), *fresh = malloc(98304u * sizeof(float));
    TEST_ASSERT(at_sync && rewound && next && fresh);
    char err[256] = {0};
    ds4_tokens prompt = {0}, ext = {0};
    for (int i = 0; i < len; i++) ds4_tokens_push(&prompt, test_mellum_token(i));
    for (int i = 0; i < len + 5; i++) ds4_tokens_push(&ext, i < len ? test_mellum_token(i) : test_mellum_token(50000 + i));
    ds4_session *s = NULL;
    TEST_ASSERT(ds4_session_create(&s, engine, 4096) == 0 && s != NULL);
    if (!s || !at_sync || !rewound || !next || !fresh) return;
    TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_copy_logits(s, at_sync, 98304) == 98304);
    for (int j = 0; j < k; j++) TEST_ASSERT(ds4_session_eval(s, test_mellum_token(len + j), err, sizeof(err)) == 0);
    ds4_session_rewind(s, len);
    uint32_t graph_pos = UINT32_MAX;
    TEST_ASSERT(ds4_test_session_graph_pos(s, &graph_pos) && graph_pos == (uint32_t)len);
    TEST_ASSERT(ds4_session_copy_logits(s, rewound, 98304) == 98304);
    TEST_ASSERT(ds4_session_sync(s, &ext, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_copy_logits(s, next, 98304) == 98304);
    /* the sibling: one back below the frontier at len + 5 keeps the caches, a different token goes there, and two
     * more; the rewind to len + 5 is then two back from somewhere the frontier no longer describes */
    ds4_session_rewind(s, len + 4);
    TEST_ASSERT(ds4_session_eval(s, test_mellum_token(70000), err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_eval(s, test_mellum_token(70001), err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_eval(s, test_mellum_token(70002), err, sizeof(err)) == 0);
    ds4_session_rewind(s, len + 5);
    uint32_t stale_pos = UINT32_MAX;
    TEST_ASSERT(ds4_test_session_graph_pos(s, &stale_pos) && stale_pos == 0);
    ds4_session_free(s);
    s = NULL;
    TEST_ASSERT(ds4_session_create(&s, engine, 4096) == 0 && s != NULL);
    if (!s) return;
    TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_sync(s, &ext, err, sizeof(err)) == 0);
    TEST_ASSERT(ds4_session_copy_logits(s, fresh, 98304) == 98304);
    uint64_t d_sync = 0, d_next = 0;
    for (uint32_t i = 0; i < 98304u; i++) {
        d_sync += memcmp(&rewound[i], &at_sync[i], sizeof(float)) != 0;
        d_next += memcmp(&next[i], &fresh[i], sizeof(float)) != 0;
    }
    printf("mellum-rewind: sync %d, eval %d, rewind to %d: graph at %u, %llu of 98304 logit words differ from the "
           "sync's; the next sync of 5 differs in %llu from a fresh session's; a stale frontier resets (graph at %u)\n",
           len, k, len, graph_pos, (unsigned long long)d_sync, (unsigned long long)d_next, stale_pos);
    TEST_ASSERT(d_sync == 0);
    TEST_ASSERT(d_next == 0);
    free(at_sync); free(rewound); free(next); free(fresh);
    ds4_tokens_free(&prompt); ds4_tokens_free(&ext);
    ds4_session_free(s);
}

/* The exact-prefill diagnostic (DS4_TEST_MELLUM_MODEL): with DS4_MELLUM_EXACT_PREFILL=1 set when the session is
 * created, a prefill of 1,100 tokens (chunks of 1,024 and 76 rows) runs the exact paths, so its logits equal those
 * of the same tokens decoded one by one, byte for byte.  Without it the 1,024-row chunk takes the batched and tiled
 * paths and they do not (the sibling, which shows the switch is what makes the difference). */
static bool test_mellum_prefill_logits(ds4_engine *engine, bool exact, bool decode, float *logits) {
    if (exact) setenv("DS4_MELLUM_EXACT_PREFILL", "1", 1);
    else unsetenv("DS4_MELLUM_EXACT_PREFILL");
    ds4_session *s = NULL;
    const bool created = ds4_session_create(&s, engine, 4096) == 0 && s != NULL;
    unsetenv("DS4_MELLUM_EXACT_PREFILL");   /* read once, at session start */
    if (!created) return false;
    ds4_tokens prompt = {0};
    char err[256] = {0};
    bool ok = true;
    for (int i = 0; i < (decode ? 1 : 1100); i++) ds4_tokens_push(&prompt, test_mellum_token(i));
    ok = ds4_session_sync(s, &prompt, err, sizeof(err)) == 0;
    for (int i = 1; ok && decode && i < 1100; i++) ok = ds4_session_eval(s, test_mellum_token(i), err, sizeof(err)) == 0;
    ok = ok && ds4_session_copy_logits(s, logits, 98304) == 98304;
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    return ok;
}

static void test_mellum_exact_prefill(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-exact-prefill: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 4096 };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine != NULL);
    if (!engine) return;
    float *dec = malloc(98304u * sizeof(float)), *exact = malloc(98304u * sizeof(float)), *fast = malloc(98304u * sizeof(float));
    TEST_ASSERT(dec && exact && fast);
    TEST_ASSERT(test_mellum_prefill_logits(engine, false, true, dec));
    TEST_ASSERT(test_mellum_prefill_logits(engine, true, false, exact));
    TEST_ASSERT(test_mellum_prefill_logits(engine, false, false, fast));
    uint64_t d_exact = 0, d_fast = 0;
    for (uint32_t i = 0; i < 98304u; i++) {
        d_exact += memcmp(&exact[i], &dec[i], sizeof(float)) != 0;
        d_fast += memcmp(&fast[i], &dec[i], sizeof(float)) != 0;
    }
    printf("mellum-exact-prefill: 1,100-token prefill vs decode: %llu of 98304 logit words differ with "
           "DS4_MELLUM_EXACT_PREFILL=1, %llu without\n", (unsigned long long)d_exact, (unsigned long long)d_fast);
    TEST_ASSERT(d_exact == 0);
    TEST_ASSERT(d_fast > 0);
    free(dec); free(exact); free(fast);
    ds4_engine_close(engine);
}

/* The tiled prefill attention (DS4_TEST_MELLUM_MODEL): a 1,100-token prefill's last-row logits by default (the
 * half-operand tiles), with DS4_MELLUM_ATTN_TILED=0 (the sliding and full kernels the decode steps use) and =1 (any
 * value but 0 keeps the default), and under DS4_MELLUM_EXACT_PREFILL=1 with and without the switch.  Each setting is read once, when the
 * session is created. */
static bool test_mellum_attn_logits(ds4_engine *engine, const char *tiled, bool exact, float *logits) {
    if (tiled) setenv("DS4_MELLUM_ATTN_TILED", tiled, 1);
    if (exact) setenv("DS4_MELLUM_EXACT_PREFILL", "1", 1);
    ds4_session *s = NULL;
    const bool created = ds4_session_create(&s, engine, 4096) == 0 && s != NULL;
    unsetenv("DS4_MELLUM_ATTN_TILED");
    unsetenv("DS4_MELLUM_EXACT_PREFILL");
    if (!created) return false;
    ds4_tokens prompt = {0};
    char err[256] = {0};
    for (int i = 0; i < 1100; i++) ds4_tokens_push(&prompt, test_mellum_token(i));
    const bool ok = ds4_session_sync(s, &prompt, err, sizeof(err)) == 0 && ds4_session_copy_logits(s, logits, 98304) == 98304;
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    return ok;
}

/* The oracle's per-position rule (tests/mellum_oracle_compare.sh): the same top-1 token, or a different one where
 * the reference's top two are within 0.75 nats; max|d logit| <= 10; KL(ref || x) <= 1; |d log p(ref top-1)| <= 1. */
static bool test_mellum_within_rule(const float *ref, const float *x, double *kl_out) {
    uint32_t top = 0, second = 1, xtop = 0;
    double mr = ref[0], mx = x[0], maxd = 0.0;
    for (uint32_t i = 0; i < 98304u; i++) {
        if (ref[i] > ref[top]) { second = top; top = i; }
        else if (i != top && (second == top || ref[i] > ref[second])) second = i;
        if (x[i] > x[xtop]) xtop = i;
        mr = fmax(mr, ref[i]);
        mx = fmax(mx, x[i]);
        maxd = fmax(maxd, fabs((double)ref[i] - x[i]));
    }
    double zr = 0.0, zx = 0.0;
    for (uint32_t i = 0; i < 98304u; i++) {
        zr += exp(ref[i] - mr);
        zx += exp(x[i] - mx);
    }
    const double lzr = mr + log(zr), lzx = mx + log(zx);
    double kl = 0.0;
    for (uint32_t i = 0; i < 98304u; i++) {
        const double lp = ref[i] - lzr, lq = x[i] - lzx;
        kl += exp(lp) * (lp - lq);
    }
    const double dlogp = fabs((ref[top] - lzr) - (x[top] - lzx));
    *kl_out = kl;
    return (xtop == top || (double)ref[top] - ref[second] <= 0.75) && maxd <= 10.0 && kl <= 1.0 && dlogp <= 1.0;
}

static void test_mellum_attn_tiled(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-attn-tiled: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 4096 };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine != NULL);
    if (!engine) return;
    float *def = malloc(98304u * sizeof(float)), *off = malloc(98304u * sizeof(float));
    float *exact = malloc(98304u * sizeof(float)), *exact_off = malloc(98304u * sizeof(float));
    float *one = malloc(98304u * sizeof(float));
    TEST_ASSERT(def && off && exact && exact_off && one);
    TEST_ASSERT(test_mellum_attn_logits(engine, NULL, false, def));
    TEST_ASSERT(test_mellum_attn_logits(engine, "0", false, off));
    TEST_ASSERT(test_mellum_attn_logits(engine, "1", false, one));
    TEST_ASSERT(test_mellum_attn_logits(engine, NULL, true, exact));
    TEST_ASSERT(test_mellum_attn_logits(engine, "0", true, exact_off));
    uint64_t d_switch = 0, d_exact = 0, d_one = 0;
    for (uint32_t i = 0; i < 98304u; i++) {
        d_switch += memcmp(&def[i], &off[i], sizeof(float)) != 0;
        d_exact += memcmp(&exact[i], &exact_off[i], sizeof(float)) != 0;
        d_one += memcmp(&def[i], &one[i], sizeof(float)) != 0;
    }
    double kl_def = 0.0, kl_off = 0.0;
    const bool rule_def = test_mellum_within_rule(exact, def, &kl_def), rule_off = test_mellum_within_rule(exact, off, &kl_off);
    printf("mellum-attn-tiled: 1,100-token prefill, logit words of 98304 that differ: the default against "
           "DS4_MELLUM_ATTN_TILED=0 %llu, against =1 %llu; under DS4_MELLUM_EXACT_PREFILL=1 %llu; KL to the exact "
           "prefill: default %.2e, =0 %.2e\n", (unsigned long long)d_switch, (unsigned long long)d_one,
           (unsigned long long)d_exact, kl_def, kl_off);
    TEST_ASSERT(d_switch > 0);   /* the default tiles the prefill attention */
    TEST_ASSERT(d_one == 0);     /* =1 is the default: only =0 changes the attention */
    TEST_ASSERT(d_exact == 0);   /* the exact prefill never tiles */
    TEST_ASSERT(rule_def);
    TEST_ASSERT(rule_off);
    free(def); free(off); free(exact); free(exact_off); free(one);
    ds4_engine_close(engine);
}

/* B0 (phase 8b, D2): a two-row verify against two ordinary decode steps (DS4_TEST_MELLUM_MODEL).  At each context
 * of DS4_TEST_MELLUM_B0_CTX (comma-separated; default 2048), the session holds that many ids of
 * speed-bench/promessi_sposi.txt, tokenized plainly, and the two drafted ids are the text's next two.  Arm D is two
 * one-row forwards; arm V one two-row forward and the LM head on its first row.  Each arm returns the session to
 * where it was, so every run is at the same context.  Runs go D V V D for DS4_TEST_MELLUM_B0_N rounds (default 4),
 * the first round dropped from the timing.  Every run's two logit rows must equal the first D run's bytes: below
 * 32 rows a chunk computes what its decode steps do, so a verify can accept exactly what decode would. */
bool ds4_test_mellum_b0_arm(ds4_session *s, const int tokens[2], bool verify, float *logits0, float *logits1,
                            double *ms, double *head_ms);   /* ds4.c test hook */

static int test_double_cmp(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double test_median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), test_double_cmp);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void test_mellum_b0_at(ds4_engine *engine, const ds4_tokens *text, int ctx, int rounds) {
    ds4_session *s = NULL;
    TEST_ASSERT(ctx > 0 && ctx + 2 <= text->len);
    TEST_ASSERT(ds4_session_create(&s, engine, (ctx + 66 + 1023) / 1024 * 1024) == 0 && s != NULL);
    if (!s || ctx + 2 > text->len) return;
    ds4_tokens prompt = {0};
    char err[256] = {0};
    for (int i = 0; i < ctx; i++) ds4_tokens_push(&prompt, text->v[i]);
    TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    const int draft[2] = { text->v[ctx], text->v[ctx + 1] };
    float *ref0 = malloc(98304u * sizeof(float)), *ref1 = malloc(98304u * sizeof(float));
    float *l0 = malloc(98304u * sizeof(float)), *l1 = malloc(98304u * sizeof(float));
    double *d_ms = calloc((size_t)rounds * 2, sizeof(double)), *v_ms = calloc((size_t)rounds * 2, sizeof(double));
    double *h_ms = calloc((size_t)rounds * 2, sizeof(double));
    TEST_ASSERT(ref0 && ref1 && l0 && l1 && d_ms && v_ms && h_ms);
    int nd = 0, nv = 0;
    uint64_t differ = 0;
    bool have_ref = false, ok = ref0 && ref1 && l0 && l1 && d_ms && v_ms && h_ms;
    for (int r = 0; r < rounds && ok; r++) {
        static const bool order[4] = { false, true, true, false };
        for (int k = 0; k < 4 && ok; k++) {
            double ms = 0.0, head = 0.0;
            ok = ds4_test_mellum_b0_arm(s, draft, order[k], have_ref ? l0 : ref0, have_ref ? l1 : ref1, &ms, &head);
            if (!ok) break;
            if (have_ref) {
                for (uint32_t i = 0; i < 98304u; i++)
                    differ += (memcmp(&l0[i], &ref0[i], sizeof(float)) != 0) + (memcmp(&l1[i], &ref1[i], sizeof(float)) != 0);
            }
            have_ref = true;
            if (r == 0) continue;
            if (order[k]) { v_ms[nv] = ms; h_ms[nv++] = head; }
            else d_ms[nd++] = ms;
        }
    }
    TEST_ASSERT(ok);
    TEST_ASSERT(differ == 0);
    if (ok && nd > 0 && nv > 0) {
        const double d = test_median(d_ms, nd), v = test_median(v_ms, nv), h = test_median(h_ms, nv);
        printf("mellum-b0: context %d: n %d per arm; two decode steps %.3f ms (%.3f-%.3f), verify %.3f ms (%.3f-%.3f) "
               "of which the row-0 head %.3f ms; verify / one step %.3f; logit words that differ from decode %llu\n",
               ctx, nd, d, d_ms[0], d_ms[nd - 1], v, v_ms[0], v_ms[nv - 1], h, 2.0 * v / d, (unsigned long long)differ);
    }
    free(ref0); free(ref1); free(l0); free(l1); free(d_ms); free(v_ms); free(h_ms);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
}

static void test_mellum_b0(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-b0: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    const char *ctx_env = getenv("DS4_TEST_MELLUM_B0_CTX"), *n_env = getenv("DS4_TEST_MELLUM_B0_N");
    const char *ctxs = ctx_env && ctx_env[0] ? ctx_env : "2048";
    const int rounds = n_env && atoi(n_env) > 1 ? atoi(n_env) : 4;
    int max_ctx = 0;
    for (const char *p = ctxs; *p; p += strcspn(p, ",") + (p[strcspn(p, ",")] == ',')) {
        if (atoi(p) > max_ctx) max_ctx = atoi(p);
    }
    FILE *f = fopen("speed-bench/promessi_sposi.txt", "rb");
    TEST_ASSERT(f != NULL);
    if (!f) return;
    char *buf = malloc(1u << 21);
    const size_t n = buf ? fread(buf, 1, (1u << 21) - 1, f) : 0;
    fclose(f);
    if (!buf) return;
    buf[n] = '\0';
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL,
                               .context_size = (max_ctx + 66 + 1023) / 1024 * 1024 };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine != NULL);
    if (engine) {
        ds4_tokens text = {0};
        ds4_tokenize_text(engine, buf, &text);
        for (const char *p = ctxs; *p; p += strcspn(p, ",") + (p[strcspn(p, ",")] == ','))
            test_mellum_b0_at(engine, &text, atoi(p), rounds);
        ds4_tokens_free(&text);
        ds4_engine_close(engine);
    }
    free(buf);
}

/* --mellum-bk: what a k-row verify costs.  At a fixed context, for k in 1, 2, 3, 4, 6 and 8, rounds of D C C D
 * (D: k one-row forwards, C: one k-row forward, each followed by restoring the frontier, outside the timing); the
 * first round dropped.  Every run's last-row logits must equal the first D run's bytes for that k: below 32 rows
 * a chunk computes what its decode steps do.  Prints, per k, one decode step (D / k), the chunk, and their ratio. */
bool ds4_test_mellum_rows_arm(ds4_session *s, const int *tokens, uint32_t k, bool chunk, float *logits_last,
                              double *ms);   /* ds4.c test hook */

static void test_mellum_bk_at(ds4_engine *engine, const ds4_tokens *text, int ctx, int rounds) {
    enum { KMAX = 8 };
    static const uint32_t ks[] = { 1, 2, 3, 4, 6, 8 };
    ds4_session *s = NULL;
    TEST_ASSERT(ctx > 0 && ctx + KMAX <= text->len);
    TEST_ASSERT(ds4_session_create(&s, engine, (ctx + 66 + 1023) / 1024 * 1024) == 0 && s != NULL);
    if (!s || ctx + KMAX > text->len) return;
    ds4_tokens prompt = {0};
    char err[256] = {0};
    for (int i = 0; i < ctx; i++) ds4_tokens_push(&prompt, text->v[i]);
    TEST_ASSERT(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    float *ref = malloc(98304u * sizeof(float)), *got = malloc(98304u * sizeof(float));
    double *d_ms = calloc((size_t)rounds * 2, sizeof(double)), *c_ms = calloc((size_t)rounds * 2, sizeof(double));
    bool ok = ref && got && d_ms && c_ms;
    TEST_ASSERT(ok);
    for (size_t ki = 0; ok && ki < sizeof(ks) / sizeof(ks[0]); ki++) {
        const uint32_t k = ks[ki];
        int nd = 0, nc = 0;
        uint64_t differ = 0;
        bool have_ref = false;
        for (int r = 0; r < rounds && ok; r++) {
            static const bool order[4] = { false, true, true, false };
            for (int j = 0; j < 4 && ok; j++) {
                double ms = 0.0;
                ok = ds4_test_mellum_rows_arm(s, &text->v[ctx], k, order[j], have_ref ? got : ref, &ms);
                if (!ok) break;
                if (have_ref) {
                    for (uint32_t i = 0; i < 98304u; i++) differ += memcmp(&got[i], &ref[i], sizeof(float)) != 0;
                }
                have_ref = true;
                if (r == 0) continue;
                if (order[j]) c_ms[nc++] = ms;
                else d_ms[nd++] = ms;
            }
        }
        TEST_ASSERT(ok);
        TEST_ASSERT(differ == 0);
        if (ok && nd > 0 && nc > 0) {
            const double d = test_median(d_ms, nd) / k, c = test_median(c_ms, nc);
            printf("mellum-bk: context %d k %u: n %d per arm; one decode step %.3f ms, %u-row forward %.3f ms "
                   "(%.3f-%.3f); forward / one step %.3f; logit words that differ from decode %llu\n",
                   ctx, k, nd, d, k, c, c_ms[0], c_ms[nc - 1], c / d, (unsigned long long)differ);
        }
    }
    free(ref); free(got); free(d_ms); free(c_ms);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
}

static void test_mellum_bk(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-bk: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    const char *ctx_env = getenv("DS4_TEST_MELLUM_BK_CTX"), *n_env = getenv("DS4_TEST_MELLUM_BK_N");
    const int ctx = ctx_env && atoi(ctx_env) > 0 ? atoi(ctx_env) : 2048;
    const int rounds = n_env && atoi(n_env) > 1 ? atoi(n_env) : 4;
    FILE *f = fopen("speed-bench/promessi_sposi.txt", "rb");
    TEST_ASSERT(f != NULL);
    if (!f) return;
    char *buf = malloc(1u << 21);
    const size_t n = buf ? fread(buf, 1, (1u << 21) - 1, f) : 0;
    fclose(f);
    if (!buf) return;
    buf[n] = '\0';
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL,
                               .context_size = (ctx + 66 + 1023) / 1024 * 1024 };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine != NULL);
    if (engine) {
        ds4_tokens text = {0};
        ds4_tokenize_text(engine, buf, &text);
        test_mellum_bk_at(engine, &text, ctx, rounds);
        ds4_tokens_free(&text);
        ds4_engine_close(engine);
    }
    free(buf);
}

static void test_mellum_rewind(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-rewind: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 4096 };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine != NULL);
    if (!engine) return;
    test_mellum_rewind_case(engine, 500, 1);    /* inside the window */
    test_mellum_rewind_case(engine, 1100, 1);   /* past it: the ring has wrapped */
    test_mellum_rewind_case(engine, 1100, 3);   /* deeper than the ring allows: back to the sync's end */
    test_mellum_rewind_fresh_case(engine, 31, 3, 32);
    test_mellum_frontier_case(engine, 1100, 40);
    ds4_engine_close(engine);
}

#if defined(__APPLE__)
#include "mellum_moe_ref.inc"

bool ds4_test_mellum_moe_weights(ds4_engine *engine, uint32_t layer, const void **map,
                                 uint64_t *router, uint64_t *gate, uint64_t *up, uint64_t *down,
                                 uint32_t *expert_type);
bool ds4_test_mellum_moe(ds4_engine *engine, uint32_t layer, const float *x, uint32_t T,
                         float *out, float *out1, int32_t *selected, float *weights);

static float test_mellum_frand(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return ((float)(*state & 0xffffffu) / 8388608.0f) - 1.0f;
}

/* mellum_graph_moe on the real model's layers (DS4_TEST_MELLUM_MODEL): rows through layers 0, 13 and 27 against the
 * double reference (tests/mellum_moe_ref.inc), whose weights are found by their GGUF names and not through the layer
 * struct the driver reads, so a swapped or misbound tensor shows.  16 rows take the per-token experts, so row 0 alone
 * (T=1) equals row 0 of the 16 bit for bit; 64 rows take the tiled expert GEMMs (half-staged activations), which
 * must select the same experts and stay within MOE_TILED_TOL of the reference. */
#define MOE_TILED_TOL 2e-3

/* Real activations (review fix R7): the MoE input rows of a real prompt (tests/mellum_heldout_unicode.tokens, where
 * the release prefill's router flips were found), read from the trace's ffn_in files at 64 positions, through
 * layers 0, 6, 10, 13, 14 and 27 as 64 tiled rows and as four calls of 16 exact rows, against the double reference.
 * A row must select the reference's experts in its order, unless the reference's own 8th and 9th router logits
 * are within MOE_TIE_GAP: such a near-tie the f32 router may break either way, so it is counted, not failed.
 * Where the selection agrees, the weights are held to the random-row test's tolerance, and the output to 1e-5 of
 * the double reference for exact rows; tiled rows stage half operands, so theirs is held to MOE_STAGED_TOL of a
 * reference that stages the same way (mref_moe_row_staged), and its distance from the unstaged one is printed. */
#define MOE_TIE_GAP 1e-4
#define MOE_STAGED_TOL 2.5e-4
static bool test_mellum_read_tokens(const char *path, ds4_tokens *out) {
    FILE *f = fopen(path, "r");
    if (!f) return false;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        for (char *p = line, *end; *p; p = end) {
            const long v = strtol(p, &end, 10);
            if (end == p) { if (*p) end = p + 1; continue; }
            ds4_tokens_push(out, (int)v);
        }
    }
    fclose(f);
    return out->len > 0;
}

static void test_mellum_moe_real_rows(ds4_engine *engine) {
    enum { NPOS = 64, NL = 28 };
    ds4_tokens toks = {0};
    TEST_ASSERT(test_mellum_read_tokens("tests/mellum_heldout_unicode.tokens", &toks) && toks.len >= NPOS);
    const char *tmp = getenv("TMPDIR");
    char dir[512], prefix[600], at[1024] = "", path[700];
    snprintf(dir, sizeof(dir), "%s/mellum-moe-rows.XXXXXX", tmp && tmp[0] ? tmp : "/tmp");
    TEST_ASSERT(mkdtemp(dir) != NULL);
    snprintf(prefix, sizeof(prefix), "%s/t", dir);
    int pos[NPOS];
    for (int i = 0; i < NPOS; i++) {
        pos[i] = (int)lround((double)i * (toks.len - 1) / (NPOS - 1));
        snprintf(at + strlen(at), sizeof(at) - strlen(at), "%s%d", i ? "," : "", pos[i]);
    }
    setenv("DS4_MELLUM_TRACE", prefix, 1);
    setenv("DS4_MELLUM_TRACE_AT", at, 1);
    ds4_session *s = NULL;
    char err[256] = {0};
    TEST_ASSERT(ds4_session_create(&s, engine, 2048) == 0 && s != NULL);
    if (s) TEST_ASSERT(ds4_session_sync(s, &toks, err, sizeof(err)) == 0);
    ds4_session_free(s);
    unsetenv("DS4_MELLUM_TRACE");
    unsetenv("DS4_MELLUM_TRACE_AT");
    float *rows = malloc((size_t)NPOS * NL * MREF_E * sizeof(float));
    float *x = malloc((size_t)NPOS * MREF_E * sizeof(float)), *out = malloc((size_t)NPOS * MREF_E * sizeof(float));
    float *out1 = malloc(MREF_E * sizeof(float));
    int32_t (*sel)[MREF_USED] = malloc(sizeof(int32_t[NPOS][MREF_USED]));
    float (*w)[MREF_USED] = malloc(sizeof(float[NPOS][MREF_USED]));
    mref_t *ref = malloc(sizeof(*ref));
    TEST_ASSERT(rows && x && out && out1 && sel && w && ref);
    for (int i = 0; rows && i < NPOS; i++) {
        static const char *const what[] = {"hidden", "qk", "ffn_in", "logits"};
        snprintf(path, sizeof(path), "%s.p%d.ffn_in.f32", prefix, pos[i]);
        FILE *f = fopen(path, "rb");
        TEST_ASSERT(f && fread(rows + (size_t)i * NL * MREF_E, sizeof(float), (size_t)NL * MREF_E, f) == (size_t)NL * MREF_E);
        if (f) fclose(f);
        for (int k = 0; k < 4; k++) {
            snprintf(path, sizeof(path), "%s.p%d.%s.f32", prefix, pos[i], what[k]);
            unlink(path);
        }
    }
    snprintf(path, sizeof(path), "%s.session.logits.f32", prefix);
    unlink(path);
    rmdir(dir);
    static const uint32_t layers[] = { 0, 6, 10, 13, 14, 27 };
    for (int tiled = 1; tiled >= 0 && rows && x && out && out1 && sel && w && ref; tiled--) {
        uint32_t near = 0, near_differ = 0, sel_bad = 0;
        double worst_w = 0.0, worst_out = 0.0, min_gap = 1e30, worst_xmax = 0.0, unstaged = 0.0;
        uint32_t worst_layer = 0;
        int worst_pos = -1;
        for (size_t li = 0; li < sizeof(layers) / sizeof(layers[0]); li++) {
            const uint32_t il = layers[li];
            mref_weights mw = { 0 };   /* the expert type comes from the file */
            const void *map = NULL;
            TEST_ASSERT(ds4_test_mellum_moe_weights(engine, il, &map, &mw.router, &mw.gate, &mw.up, &mw.down, &mw.expert_type));
            mw.base = map;
            if (!map) break;
            for (int i = 0; i < NPOS; i++)
                memcpy(x + (size_t)i * MREF_E, rows + ((size_t)i * NL + il) * MREF_E, MREF_E * sizeof(float));
            const uint32_t T = tiled ? NPOS : 16u;
            for (uint32_t r0 = 0; r0 < NPOS; r0 += T)
                TEST_ASSERT(ds4_test_mellum_moe(engine, il, x + (size_t)r0 * MREF_E, T, out + (size_t)r0 * MREF_E, out1,
                                                &sel[r0][0], &w[r0][0]));
            const float *rw = (const float *)(mw.base + mw.router);
            for (int r = 0; r < NPOS; r++) {
                const float *xr = x + (size_t)r * MREF_E;
                mref_moe_row(&mw, xr, ref);
                double logit[MREF_NEXP];
                for (uint32_t e = 0; e < MREF_NEXP; e++) {
                    double acc = 0.0;
                    for (uint32_t i = 0; i < MREF_E; i++) acc += (double)rw[(uint64_t)e * MREF_E + i] * xr[i];
                    logit[e] = acc;
                }
                double l8 = 1e30, l9 = -1e30;   /* the reference's 8th logit, and the best one it left out */
                bool chosen[MREF_NEXP] = { false };
                for (uint32_t k = 0; k < MREF_USED; k++) { chosen[ref->sel[k]] = true; l8 = fmin(l8, logit[ref->sel[k]]); }
                for (uint32_t e = 0; e < MREF_NEXP; e++) if (!chosen[e]) l9 = fmax(l9, logit[e]);
                const double gap = l8 - l9;
                bool same = true;
                for (uint32_t k = 0; k < MREF_USED; k++) same = same && sel[r][k] == ref->sel[k];
                if (gap < MOE_TIE_GAP) near++;
                if (!same) {
                    if (gap < MOE_TIE_GAP) near_differ++;
                    else sel_bad++;
                    min_gap = fmin(min_gap, gap);
                    continue;
                }
                if (tiled) {   /* the staging's cost on real rows, measured against the unstaged reference */
                    double sc = 1e-30, e = 0.0;
                    for (uint32_t i = 0; i < MREF_E; i++) sc = fmax(sc, fabs(ref->out[i]));
                    for (uint32_t i = 0; i < MREF_E; i++) e = fmax(e, fabs((double)out[(size_t)r * MREF_E + i] - ref->out[i]));
                    unstaged = fmax(unstaged, isfinite(e) ? e / sc : 1e30);
                    mref_moe_row_staged(&mw, xr, ref);
                }
                double scale = 1e-30, d = 0.0;
                for (uint32_t i = 0; i < MREF_E; i++) scale = fmax(scale, fabs(ref->out[i]));
                for (uint32_t i = 0; i < MREF_E; i++) {
                    const double e = fabs((double)out[(size_t)r * MREF_E + i] - ref->out[i]);
                    d = isfinite(e) ? fmax(d, e) : 1e30;
                }
                if (d / scale > worst_out) {
                    worst_out = d / scale;
                    worst_layer = il;
                    worst_pos = pos[r];
                    worst_xmax = 0.0;
                    for (uint32_t i = 0; i < MREF_E; i++) worst_xmax = fmax(worst_xmax, fabs((double)xr[i]));
                }
                for (uint32_t k = 0; k < MREF_USED; k++) worst_w = fmax(worst_w, fabs((double)w[r][k] - ref->w[k]));
            }
        }
        printf("mellum-moe: real rows (%s, %d positions of a Unicode prompt x layers 0/6/10/13/14/27) vs double: "
               "%u near-ties (8th/9th logit gap < %g), %u of them selected differently; %u other selections differ; "
               "where the selection agrees, weights %.2e, output %.2e%s (layer %u, position %d, max |x| %.1f)\n",
               tiled ? "64 tiled rows" : "16 exact rows", NPOS, near, MOE_TIE_GAP, near_differ, sel_bad, worst_w,
               worst_out, tiled ? " against the staged reference" : "", worst_layer, worst_pos, worst_xmax);
        if (tiled) printf("mellum-moe: real rows, 64 tiled rows against the unstaged reference: output %.2e (the "
                          "half staging's cost)\n", unstaged);
        if (sel_bad) printf("mellum-moe: the smallest gap among differing selections: %.3e\n", min_gap);
        TEST_ASSERT(sel_bad == 0);
        TEST_ASSERT(worst_w <= 2.5e-6);
        TEST_ASSERT(worst_out <= (tiled ? MOE_STAGED_TOL : 1e-5));
    }
    free(rows); free(x); free(out); free(out1); free(sel); free(w); free(ref);
    ds4_tokens_free(&toks);
}

static void test_mellum_moe_real(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-moe: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    enum { TMAX = 64 };
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 2048, .prefill_chunk = TMAX };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine != NULL);
    if (!engine) return;
    float *x = malloc((uint64_t)TMAX * MREF_E * sizeof(float));
    float *out = malloc((uint64_t)TMAX * MREF_E * sizeof(float));
    float *out1 = malloc((uint64_t)MREF_E * sizeof(float));
    int32_t sel[TMAX][MREF_USED];
    float w[TMAX][MREF_USED];
    mref_t *ref = malloc(sizeof(*ref));
    TEST_ASSERT(x && out && out1 && ref);
    static const uint32_t layers[] = { 0, 13, 27 };
    static const uint32_t counts[] = { 16, 64 };
    uint32_t rng = 0x1234567u;
    for (size_t c = 0; x && c < sizeof(counts) / sizeof(counts[0]); c++) {
        const uint32_t T = counts[c];
        const uint32_t rows[] = { 0, 7, T / 2u - 1u, T - 1u };
        double worst_out = 0.0, worst_w = 0.0;
        uint32_t sel_bad = 0;
        uint64_t row0_diff = 0;
        for (size_t li = 0; li < sizeof(layers) / sizeof(layers[0]); li++) {
            const uint32_t il = layers[li];
            mref_weights mw = { 0 };   /* the expert type comes from the file */
            const void *map = NULL;
            TEST_ASSERT(ds4_test_mellum_moe_weights(engine, il, &map, &mw.router, &mw.gate, &mw.up, &mw.down, &mw.expert_type));
            mw.base = map;
            if (!map) break;
            for (uint64_t i = 0; i < (uint64_t)T * MREF_E; i++) x[i] = 1.7f * test_mellum_frand(&rng);
            TEST_ASSERT(ds4_test_mellum_moe(engine, il, x, T, out, out1, &sel[0][0], &w[0][0]));
            for (size_t ri = 0; ri < sizeof(rows) / sizeof(rows[0]); ri++) {
                const uint32_t r = rows[ri];
                mref_moe_row(&mw, x + (uint64_t)r * MREF_E, ref);
                double scale = 1e-30, d = 0.0;
                for (uint32_t i = 0; i < MREF_E; i++) scale = fmax(scale, fabs(ref->out[i]));
                for (uint32_t i = 0; i < MREF_E; i++) {
                    const double e = fabs((double)out[(uint64_t)r * MREF_E + i] - ref->out[i]);
                    d = isfinite(e) ? fmax(d, e) : 1e30;   /* a non-finite word fails the tolerance */
                }
                worst_out = fmax(worst_out, d / scale);
                for (uint32_t k = 0; k < MREF_USED; k++) {
                    sel_bad += sel[r][k] != ref->sel[k];
                    worst_w = fmax(worst_w, fabs((double)w[r][k] - ref->w[k]));
                }
            }
            for (uint32_t i = 0; i < MREF_E; i++) row0_diff += memcmp(&out1[i], &out[i], sizeof(float)) != 0;
        }
        const bool tiled = T >= 32u;
        printf("mellum-moe: T=%u (%s) layers 0/13/27, 4 rows each vs double: selected differ %u, weights %.2e, output %.2e; "
               "T=1 vs row 0 of T=%u: %llu words differ\n", T, tiled ? "tiled experts" : "per-token experts", sel_bad,
               worst_w, worst_out, T, (unsigned long long)row0_diff);
        TEST_ASSERT(sel_bad == 0);
        TEST_ASSERT(worst_w <= 2.5e-6);
        TEST_ASSERT(worst_out <= (tiled ? MOE_TILED_TOL : 1e-5));
        if (!tiled) TEST_ASSERT(row0_diff == 0);
        else TEST_ASSERT(row0_diff > 0);   /* the tiled path is not the decode arithmetic */
    }
    if (x && out && out1 && ref) {
        /* Experiment E1: by default 64 rows take the Metal 4 tensor tiles where the device has them, which sum in
         * another order than the simdgroup tiles DS4_MELLUM_MOE_NAX=0 keeps; both stage the same half operands, so
         * both stay within MOE_STAGED_TOL of the staged reference */
        enum { T = 64 };
        const uint32_t il = 13, rows[] = { 0, 7, 31, 63 };
        float *simd = malloc((uint64_t)T * MREF_E * sizeof(float));
        mref_weights mw = { 0 };   /* the expert type comes from the file */
        const void *map = NULL;
        TEST_ASSERT(simd && ds4_test_mellum_moe_weights(engine, il, &map, &mw.router, &mw.gate, &mw.up, &mw.down, &mw.expert_type));
        mw.base = map;
        for (uint64_t i = 0; i < (uint64_t)T * MREF_E; i++) x[i] = 1.7f * test_mellum_frand(&rng);
        setenv("DS4_MELLUM_MOE_NAX", "0", 1);
        TEST_ASSERT(simd && ds4_test_mellum_moe(engine, il, x, T, simd, out1, &sel[0][0], &w[0][0]));
        unsetenv("DS4_MELLUM_MOE_NAX");
        TEST_ASSERT(ds4_test_mellum_moe(engine, il, x, T, out, out1, &sel[0][0], &w[0][0]));
        uint64_t differ = 0;
        double worst_simd = 0.0, worst_tiles = 0.0;
        for (uint64_t i = 0; simd && i < (uint64_t)T * MREF_E; i++) differ += memcmp(&simd[i], &out[i], sizeof(float)) != 0;
        for (size_t ri = 0; simd && map && ri < sizeof(rows) / sizeof(rows[0]); ri++) {
            const uint32_t r = rows[ri];
            mref_moe_row_staged(&mw, x + (uint64_t)r * MREF_E, ref);
            double scale = 1e-30, ds = 0.0, dt = 0.0;
            for (uint32_t i = 0; i < MREF_E; i++) scale = fmax(scale, fabs(ref->out[i]));
            for (uint32_t i = 0; i < MREF_E; i++) {
                ds = fmax(ds, fabs((double)simd[(uint64_t)r * MREF_E + i] - ref->out[i]));
                dt = fmax(dt, fabs((double)out[(uint64_t)r * MREF_E + i] - ref->out[i]));
            }
            worst_simd = fmax(worst_simd, ds / scale);
            worst_tiles = fmax(worst_tiles, dt / scale);
        }
        /* E1's switch (ds4_gpu_qwen4_moe_q8_tiles) moves Q8_0 experts only; MXFP4 experts (S6) take the qwen4
         * default level whatever DS4_MELLUM_MOE_NAX says, so the two runs must be the same bits */
        const bool tiles = ds4_gpu_tensor_tiles_available() != 0, switched = tiles && mw.expert_type == MREF_TYPE_Q8_0;
        printf("mellum-moe: T=64 layer 13, %s experts, the default (%s) against DS4_MELLUM_MOE_NAX=0: %llu of %llu "
               "words differ; staged reference: NAX=0 %.2e, default %.2e\n",
               mw.expert_type == MREF_TYPE_MXFP4 ? "MXFP4" : "Q8_0",
               switched ? "tensor tiles" : tiles ? "the qwen4 default, switch inert" : "no tensor API",
               (unsigned long long)differ, (unsigned long long)T * MREF_E, worst_simd, worst_tiles);
        TEST_ASSERT(switched ? differ > 0 : differ == 0);
        TEST_ASSERT(worst_simd <= MOE_STAGED_TOL && worst_tiles <= MOE_STAGED_TOL);
        free(simd);
    }
    free(x); free(out); free(out1); free(ref);
    test_mellum_moe_real_rows(engine);
    ds4_engine_close(engine);
}
#else
static void test_mellum_moe_real(void) { puts("mellum-moe: Metal only, skipped"); }
#endif
