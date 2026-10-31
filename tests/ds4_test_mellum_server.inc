/* Mellum 2.1, included by ds4_test.c where this code sits: not a separate translation unit. */

/* ---- ds4-server and Mellum 2.1: the Hermes syntax, pinned against the model's own template ---- */

typedef struct { const char *name, *arguments; } msc_call;
typedef struct { const char *role, *content, *reasoning; int ncalls; const msc_call *calls; } msc_msg;
typedef struct { const char *name; bool think, tools; int nmsg; const msc_msg *msgs; const char *text; int nids;
                 const int *ids; } msc_case;
#include "mellum_server_cases.inc"

/* a case's request as the server holds it: its messages, and its tools as parse_tools_value keeps them */
static void mellum_server_case_request(const msc_case *c, chat_msgs *msgs, char **schemas, tool_schema_orders *orders) {
    memset(msgs, 0, sizeof(*msgs));
    memset(orders, 0, sizeof(*orders));
    *schemas = NULL;
    for (int j = 0; j < c->nmsg; j++) {
        const msc_msg *m = &c->msgs[j];
        chat_msg msg = {0};
        msg.role = xstrdup(m->role);
        msg.content = xstrdup(m->content);
        if (m->reasoning) msg.reasoning = xstrdup(m->reasoning);
        for (int k = 0; k < m->ncalls; k++) {
            char id[16];
            snprintf(id, sizeof(id), "c%d", k + 1);
            tool_calls_push(&msg.calls, (tool_call){.id = xstrdup(id), .name = xstrdup(m->calls[k].name),
                                                    .arguments = xstrdup(m->calls[k].arguments)});
        }
        chat_msgs_push(msgs, msg);
    }
    if (c->tools) {
        const char *p = msc_tools_json;
        TEST_ASSERT(parse_tools_value(&p, schemas, orders));
    }
}

/* each case renders to the template's text, byte for byte */
static void test_mellum_server_render(void) {
    int bad = 0;
    for (size_t i = 0; i < sizeof(msc_cases) / sizeof(msc_cases[0]); i++) {
        const msc_case *c = &msc_cases[i];
        chat_msgs msgs;
        char *schemas;
        tool_schema_orders orders;
        mellum_server_case_request(c, &msgs, &schemas, &orders);
        char *got = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_HERMES, &msgs, schemas, &orders,
                                                       c->think ? DS4_THINK_HIGH : DS4_THINK_NONE);
        if (!got || strcmp(got, c->text) != 0) {
            size_t at = 0;
            while (got && got[at] && got[at] == c->text[at]) at++;
            fprintf(stderr, "mellum-server: render \"%s\" differs from the template at byte %zu\n", c->name, at);
            bad++;
        }
        free(got);
        free(schemas);
        tool_schema_orders_free(&orders);
        chat_msgs_free(&msgs);
    }
    TEST_ASSERT(bad == 0);
    if (!bad) printf("mellum-server: %zu renders equal 2.1's template text\n", sizeof(msc_cases) / sizeof(msc_cases[0]));
}

/* Mellum's generated text, parsed: reasoning (the model writes its own <think>), content, and Hermes calls whose
 * arguments keep the model's JSON text, so the next request renders the turn back byte for byte */
static void mellum_parse(const char *text, bool closed, char **content, char **reasoning, tool_calls *calls) {
    *content = *reasoning = NULL;
    memset(calls, 0, sizeof(*calls));
    TEST_ASSERT(parse_generated_message_ex_for_syntax(SERVER_MODEL_SYNTAX_HERMES, text, closed, content, reasoning, calls));
}

static void test_mellum_server_parse(void) {
    const int failures_before = test_failures;
    char *content, *reasoning;
    tool_calls calls;
    mellum_parse("<think>\nThe user greets me.\n</think>\n\nHello!", true, &content, &reasoning, &calls);
    TEST_ASSERT(reasoning && !strcmp(reasoning, "The user greets me.") && content && !strcmp(content, "Hello!") && calls.len == 0);
    free(content); free(reasoning); tool_calls_free(&calls);
    /* thinking off: the prompt closed the think block, so the text is the answer */
    mellum_parse("Hello!", false, &content, &reasoning, &calls);
    TEST_ASSERT(!reasoning && content && !strcmp(content, "Hello!") && calls.len == 0);
    free(content); free(reasoning); tool_calls_free(&calls);

    const char *one = "<think>\nRead it first.\n</think>\n\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"hello.py\"}}\n</tool_call>";
    mellum_parse(one, true, &content, &reasoning, &calls);
    TEST_ASSERT(reasoning && !strcmp(reasoning, "Read it first.") && content && !strcmp(content, ""));
    TEST_ASSERT(calls.len == 1 && !strcmp(calls.v[0].name, "read") && !strcmp(calls.v[0].arguments, "{\"path\": \"hello.py\"}"));
    {   /* the round trip: rendered back within its tool cycle, the turn is the model's own text */
        chat_msgs msgs = {0};
        chat_msgs_push(&msgs, (chat_msg){.role = xstrdup("user"), .content = xstrdup("Q")});
        chat_msg a = {.role = xstrdup("assistant"), .content = content, .reasoning = reasoning, .calls = calls};
        chat_msgs_push(&msgs, a);
        char *text = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_HERMES, &msgs, NULL, NULL, DS4_THINK_HIGH);
        buf want = {0};
        buf_puts(&want, "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n");
        buf_puts(&want, one);
        buf_puts(&want, "<|im_end|>\n<|im_start|>assistant\n");     /* and the generation prompt, as always */
        TEST_ASSERT(text && !strcmp(text, want.ptr));
        free(text); buf_free(&want); chat_msgs_free(&msgs);
    }
    {   /* the client sends the arguments back as the server returned them (minified); tool memory restores the
         * sampled call text, and the render uses it, so the turn still matches the KV */
        chat_msgs msgs = {0};
        chat_msgs_push(&msgs, (chat_msg){.role = xstrdup("user"), .content = xstrdup("Q")});
        chat_msg a = {.role = xstrdup("assistant"), .content = xstrdup(""), .reasoning = xstrdup("Read it first.")};
        tool_calls_push(&a.calls, (tool_call){.id = xstrdup("c1"), .name = xstrdup("read"),
                                              .arguments = xstrdup("{\"path\":\"hello.py\"}")});
        a.calls.raw_tool_text = xstrdup("<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"hello.py\"}}\n</tool_call>");
        chat_msgs_push(&msgs, a);
        char *text = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_HERMES, &msgs, NULL, NULL, DS4_THINK_HIGH);
        TEST_ASSERT(text && strstr(text, "{\"path\": \"hello.py\"}") != NULL && strstr(text, "{\"path\":\"hello.py\"}") == NULL);
        free(text);
        chat_msgs_free(&msgs);
    }
    {   /* P6-R7: tool memory keeps the sampled two-call block under both ids.  The render uses it only when the
         * client sends back exactly those calls (names, order, decoded arguments); a subset or an edited call is
         * rendered as sent, and the cache misses */
        static const char raw[] = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"OLD\"}}\n</tool_call>\n"
                                  "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"b.txt\"}}\n</tool_call>";
        static const struct { int n; const char *args0, *want, *not_want; } cases[] = {
            {2, "{\"command\":\"OLD\"}", raw, "{\"command\":\"OLD\"}"},                     /* the same calls: the raw block */
            {1, "{\"command\":\"OLD\"}", "{\"command\":\"OLD\"}}\n</tool_call><|im_end|>", "b.txt"},  /* c1 alone */
            {2, "{\"command\":\"NEW\"}", "{\"command\":\"NEW\"}", "OLD"},                   /* c1 edited */
        };
        for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
            server srv = {0};
            pthread_mutex_init(&srv.tool_mu, NULL);
            tool_memory_put(&srv, "c1", raw);
            tool_memory_put(&srv, "c2", raw);
            chat_msgs msgs = {0};
            chat_msgs_push(&msgs, (chat_msg){.role = xstrdup("user"), .content = xstrdup("Q")});
            chat_msg a = {.role = xstrdup("assistant"), .content = xstrdup("")};
            tool_calls_push(&a.calls, (tool_call){.id = xstrdup("c1"), .name = xstrdup("bash"), .arguments = xstrdup(cases[k].args0)});
            if (cases[k].n == 2)
                tool_calls_push(&a.calls, (tool_call){.id = xstrdup("c2"), .name = xstrdup("read"),
                                                      .arguments = xstrdup("{\"path\":\"b.txt\"}")});
            chat_msgs_push(&msgs, a);
            chat_msgs_push(&msgs, (chat_msg){.role = xstrdup("tool"), .content = xstrdup("OUT")});
            tool_memory_attach_to_messages(&srv, &msgs, NULL);
            char *text = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_HERMES, &msgs, NULL, NULL, DS4_THINK_HIGH);
            if (!text || !strstr(text, cases[k].want) || strstr(text, cases[k].not_want))
                fprintf(stderr, "mellum-server: tool memory case %zu rendered %s\n", k, text ? text : "(null)");
            TEST_ASSERT(text && strstr(text, cases[k].want) && !strstr(text, cases[k].not_want));
            free(text);
            chat_msgs_free(&msgs);
            tool_memory_free(&srv.tool_mem);
            pthread_mutex_destroy(&srv.tool_mu);
        }
    }
    /* content, then two calls; a call's end marker inside a JSON string is text */
    mellum_parse("<think>\nTwo.\n</think>\n\nWriting.\n<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"a\", \"content\": \"x </tool_call> y\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
                 true, &content, &reasoning, &calls);
    TEST_ASSERT(content && !strcmp(content, "Writing.") && calls.len == 2);
    TEST_ASSERT(calls.len == 2 && !strcmp(calls.v[0].arguments, "{\"path\": \"a\", \"content\": \"x </tool_call> y\"}") &&
                !strcmp(calls.v[1].name, "bash"));
    free(content); free(reasoning); tool_calls_free(&calls);
    {   /* P6-R5: a protocol marker inside a JSON string argument is text: the call is still found, whole */
        static const char *const markers[] = {"</think>", "<think>", "<tool_call>", "</tool_call>", "<|im_end|>",
                                              "<|im_start|>", "<tool_response>", "</tool_response>"};
        for (size_t k = 0; k < sizeof(markers) / sizeof(markers[0]); k++) {
            buf text = {0}, args = {0};
            buf_printf(&args, "{\"path\": \"a\", \"content\": \"x %s y\"}", markers[k]);
            buf_printf(&text, "<think>\nwrite it\n</think>\n\n<tool_call>\n{\"name\": \"write\", \"arguments\": %s}\n</tool_call>",
                       args.ptr);
            content = reasoning = NULL;
            memset(&calls, 0, sizeof(calls));
            const bool ok = parse_generated_message_ex_for_syntax(SERVER_MODEL_SYNTAX_HERMES, text.ptr, true,
                                                                  &content, &reasoning, &calls);
            if (!ok || calls.len != 1 || strcmp(calls.v[0].arguments, args.ptr) || !reasoning || strcmp(reasoning, "write it"))
                fprintf(stderr, "mellum-server: a call holding %s in a string was not parsed whole\n", markers[k]);
            TEST_ASSERT(ok && calls.len == 1 && !strcmp(calls.v[0].arguments, args.ptr));
            free(content); free(reasoning); tool_calls_free(&calls);
            buf_free(&text); buf_free(&args);
        }
    }
    /* arguments sent as a JSON string, as json.loads would take them; an unknown key is skipped */
    mellum_parse("<think>\nx\n</think>\n\n<tool_call>\n{\"name\": \"bash\", \"id\": 7, \"arguments\": \"{\\\"command\\\": \\\"ls\\\"}\"}\n</tool_call>",
                 true, &content, &reasoning, &calls);
    TEST_ASSERT(calls.len == 1 && !strcmp(calls.v[0].arguments, "{\"command\": \"ls\"}"));
    free(content); free(reasoning); tool_calls_free(&calls);
    {   /* P6-R8: string-encoded arguments must decode to one whole JSON object, with nothing after it */
        static const char *const bad_args[] = {"\"oops\"", "\"{\\\"path\\\": \\\"a\\\"} trailing\"", "\"[1, 2]\"",
                                               "\"{\\\"path\\\": \""};
        for (size_t k = 0; k < sizeof(bad_args) / sizeof(bad_args[0]); k++) {
            buf text = {0};
            buf_printf(&text, "<think>\nx\n</think>\n\n<tool_call>\n{\"name\": \"write\", \"arguments\": %s}\n</tool_call>",
                       bad_args[k]);
            content = reasoning = NULL;
            memset(&calls, 0, sizeof(calls));
            const bool ok = parse_generated_message_ex_for_syntax(SERVER_MODEL_SYNTAX_HERMES, text.ptr, true,
                                                                  &content, &reasoning, &calls);
            if (ok) fprintf(stderr, "mellum-server: arguments %s were accepted\n", bad_args[k]);
            TEST_ASSERT(!ok);
            free(content); free(reasoning); tool_calls_free(&calls);
            buf_free(&text);
        }
    }
    /* not JSON (single quotes): refused, as json.loads refuses it */
    content = reasoning = NULL;
    memset(&calls, 0, sizeof(calls));
    TEST_ASSERT(!parse_generated_message_ex_for_syntax(SERVER_MODEL_SYNTAX_HERMES,
        "<think>\nx\n</think>\n\n<tool_call>\n{\"name\": \"edit\", \"arguments\": {'a': 1}}\n</tool_call>", true,
        &content, &reasoning, &calls));
    free(content); free(reasoning); tool_calls_free(&calls);
    if (test_failures == failures_before)
        printf("mellum-server: Hermes output parsed into reasoning, content and calls, and rendered back byte for byte\n");
}

/* The text the server remembers for a finished Mellum turn is the text the next request renders for it: after a
 * tool-call turn the tool cycle keeps the reasoning; after a final answer a new user message drops it. */
static char *mellum_render(const char *const *roles, const char *const *contents, const char *reasoning,
                           const tool_calls *calls, int n, bool think) {
    chat_msgs msgs = {0};
    for (int i = 0; i < n; i++) {
        chat_msg m = {.role = xstrdup(roles[i]), .content = xstrdup(contents[i])};
        if (!strcmp(roles[i], "assistant")) {
            if (reasoning) m.reasoning = xstrdup(reasoning);
            if (calls) {
                for (int k = 0; k < calls->len; k++)
                    tool_calls_push(&m.calls, (tool_call){.id = xstrdup("c1"), .name = xstrdup(calls->v[k].name),
                                                         .arguments = xstrdup(calls->v[k].arguments)});
            }
        }
        chat_msgs_push(&msgs, m);
    }
    char *text = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_HERMES, &msgs, NULL, NULL,
                                                    think ? DS4_THINK_HIGH : DS4_THINK_NONE);
    chat_msgs_free(&msgs);
    return text;
}

static bool starts_with(const char *s, const char *prefix) {
    return s && prefix && !strncmp(s, prefix, strlen(prefix));
}

static void test_mellum_server_visible_text(void) {
    const int failures_before = test_failures;
    for (int think = 0; think <= 1; think++) {
        const char *q[] = {"user"}, *qc[] = {"Q"};
        char *prompt = mellum_render(q, qc, NULL, NULL, 1, think);
        request r = {0};
        r.kind = REQ_CHAT;
        r.model_syntax = SERVER_MODEL_SYNTAX_HERMES;
        r.think_mode = think ? DS4_THINK_HIGH : DS4_THINK_NONE;
        r.prompt_text = prompt;
        tool_calls calls = {0};
        tool_calls_push(&calls, (tool_call){.id = xstrdup("c1"), .name = xstrdup("bash"),
                                            .arguments = xstrdup("{\"command\": \"ls\"}")});
        /* a tool-call turn: the next request carries it and the tool's result */
        const char *reasoning = think ? "Use bash." : NULL;
        char *suffix = build_tool_checkpoint_suffix(&r, "", reasoning, &calls);
        const char *n1[] = {"user", "assistant", "tool"}, *n1c[] = {"Q", "", "a.txt"};
        char *next = mellum_render(n1, n1c, reasoning, &calls, 3, think);
        buf want = {0};
        buf_puts(&want, prompt);
        buf_puts(&want, suffix ? suffix : "<none>");
        if (think) {
            TEST_ASSERT(starts_with(next, want.ptr));
        } else {
            /* thinking off, the prompt's empty think block is not in the next render of this turn, so no suffix
             * makes a prefix of it: the suffix is the KV's own text after the prompt */
            TEST_ASSERT(suffix && !strcmp(suffix, "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n"
                                                  "</tool_call><|im_end|>"));
        }
        free(suffix); free(next); buf_free(&want);
        /* a final answer: the next request adds a user message and drops the turn's reasoning, which
         * the KV keeps, so no visible key stands for the live session (P6-R1; test_mellum_frontends
         * checks the reused prompt itself) */
        TEST_ASSERT(build_thinking_visible_text(&r, "The answer.") == NULL);
        tool_calls_free(&calls);
        free(prompt);
    }
    if (test_failures == failures_before)
        printf("mellum-server: a finished tool turn's remembered text is the next request's render of it\n");
}

/* After a Mellum tool-call turn the session continues: the live tail appends the tool results to the KV (the turn's
 * <|im_end|> was the stop token), and a malformed call is answered with an error result in the same shape */
static void test_mellum_server_continuations(void) {
    const int failures_before = test_failures;
    const char *q[] = {"user"}, *qc[] = {"Q"};
    char *prompt = mellum_render(q, qc, NULL, NULL, 1, true);
    request r = {0};
    r.kind = REQ_CHAT;
    r.model_syntax = SERVER_MODEL_SYNTAX_HERMES;
    r.think_mode = DS4_THINK_HIGH;
    r.prompt_text = prompt;
    tool_calls calls = {0};
    tool_calls_push(&calls, (tool_call){.id = xstrdup("c1"), .name = xstrdup("bash"),
                                        .arguments = xstrdup("{\"command\": \"ls\"}")});
    char *turn = build_tool_checkpoint_suffix(&r, "", "Use bash.", &calls);
    turn[strlen(turn) - strlen("<|im_end|>")] = '\0';              /* the KV stops before the stop token */
    const char *roles[] = {"user", "assistant", "tool", "tool"}, *contents[] = {"Q", "", "a.txt", "b.txt"};
    chat_msgs msgs = {0};
    for (int i = 0; i < 4; i++) {
        chat_msg m = {.role = xstrdup(roles[i]), .content = xstrdup(contents[i])};
        if (i == 1) {
            m.reasoning = xstrdup("Use bash.");
            tool_calls_push(&m.calls, (tool_call){.id = xstrdup("c1"), .name = xstrdup("bash"),
                                                  .arguments = xstrdup("{\"command\": \"ls\"}")});
        }
        chat_msgs_push(&msgs, m);
    }
    char *full = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_HERMES, &msgs, NULL, NULL, DS4_THINK_HIGH);
    char *tail = render_live_tool_tail_for_syntax(SERVER_MODEL_SYNTAX_HERMES, &msgs, 2, NULL, DS4_THINK_HIGH);
    buf live = {0};
    buf_puts(&live, prompt);
    buf_puts(&live, turn);
    buf_puts(&live, tail ? tail : "<none>");
    TEST_ASSERT(full && !strcmp(live.ptr, full));
    buf_free(&live); free(full); free(tail); free(turn); chat_msgs_free(&msgs);

    thinking_state thinking = {0};
    char *bad = build_invalid_tool_call_error_suffix(&r, &thinking, "invalid JSON");
    TEST_ASSERT(starts_with(bad, "<|im_end|>\n<|im_start|>user\n<tool_response>\nTool error: invalid Hermes tool call: invalid JSON\n"));
    TEST_ASSERT(bad && strstr(bad, "\n</tool_response><|im_end|>\n<|im_start|>assistant\n") != NULL);
    TEST_ASSERT(bad && strlen(bad) > 20 && !strcmp(bad + strlen(bad) - 22, "<|im_start|>assistant\n"));
    free(bad);
    thinking.inside = true;                       /* a call made inside the think block closes it first */
    bad = build_invalid_tool_call_error_suffix(&r, &thinking, NULL);
    TEST_ASSERT(starts_with(bad, "\n</think>\n\n<|im_end|>\n<|im_start|>user\n<tool_response>\nTool error: invalid Hermes tool call\n"));
    free(bad);
    tool_calls_free(&calls);
    free(prompt);
    if (test_failures == failures_before)
        printf("mellum-server: a live tool tail continues the KV to the full render; a malformed call gets a Hermes error turn\n");
}

/* A streamed Mellum answer with tools: the reasoning starts after the model's own "<think>\n", the content is sent as it
 * comes (Mellum has one think block, so nothing waits for a second), and the call arrives as tool_calls at the end */
static void test_mellum_server_stream(void) {
    const int failures_before = test_failures;
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_HIGH;
    r.has_tools = true;
    r.model_syntax = SERVER_MODEL_SYNTAX_HERMES;
    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_test", NULL, NULL));
    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw1 = "<think>\nneed a tool\n</think>\n\nChecking";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st, raw1, strlen(raw1), false));
    TEST_ASSERT(st.sent_content);                       /* not held back */
    const char *raw = "<think>\nneed a tool\n</think>\n\nChecking.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>";
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex_for_syntax(SERVER_MODEL_SYNTAX_HERMES, raw, true, &content, &reasoning, &calls));
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st, raw, strlen(raw), false));
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_test", &st, raw, strlen(raw), &calls, "tool_calls", 10, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(out && strstr(out, "\"reasoning_content\":\"need a tool") != NULL);
    TEST_ASSERT(out && strstr(out, "\"reasoning_content\":\"\\n") == NULL);
    /* the arguments go out minified, as for every model */
    TEST_ASSERT(out && strstr(out, "\"tool_calls\"") != NULL && strstr(out, "\\\"command\\\":\\\"ls\\\"") != NULL);
    TEST_ASSERT(out && strstr(out, "<tool_call>") == NULL);
    free(out);
    free(content); free(reasoning); tool_calls_free(&calls);
    close(sv[0]);
    close(sv[1]);
    request_free(&r);
    if (test_failures == failures_before)
        printf("mellum-server: a streamed answer sends reasoning, content as it comes, and the call as tool_calls\n");
}

/* P6-R6: while Mellum generates, the tool-marker tracker sees a Hermes call start and end (a </tool_call> inside one
 * of its JSON strings is text), and a malformed call it saw start is recovered, not returned as an empty answer */
static void test_mellum_server_tracker(void) {
    const int failures_before = test_failures;
    const char *raw = "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"content\": \"a </tool_call> b\"}}\n</tool_call>";
    dsml_decode_tracker dt;
    dsml_decode_tracker_init(&dt);
    dt.model_syntax = SERVER_MODEL_SYNTAX_HERMES;
    bool start = false, end = false, orphan = false, saw_string = false, end_early = false;
    const size_t len = strlen(raw);
    for (size_t n = 1; n <= len; n++) {
        char *prefix = xstrndup(raw, n);
        dsml_decode_tracker_update(&dt, prefix, n);
        observe_tool_markers(&dt, prefix, &start, &end, &orphan);
        free(prefix);
        if (dt.decode == DSML_DECODE_JSON_STRING) saw_string = true;
        if (end && n < len) end_early = true;
    }
    TEST_ASSERT(start && end && saw_string && !end_early && !orphan);

    const char *bad = "<think>\nx\n</think>\n\n<tool_call>\n{\"name\": \"write\", \"arguments\": {bad}}\n</tool_call>";
    dsml_decode_tracker_init(&dt);
    dt.model_syntax = SERVER_MODEL_SYNTAX_HERMES;
    dt.pos = (size_t)(strstr(bad, "</think>") + 8 - bad);
    start = end = orphan = false;
    dsml_decode_tracker_update(&dt, bad, strlen(bad));
    observe_tool_markers(&dt, bad + dt.pos, &start, &end, &orphan);
    TEST_ASSERT(start);
    const char *finish = "stop";
    char err[64] = "", *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    bool recovered = false;
    TEST_ASSERT(!parse_generated_message_for_response_for_syntax(SERVER_MODEL_SYNTAX_HERMES, bad, true, start, true,
                                                                 &finish, err, sizeof(err), &content, &reasoning,
                                                                 &calls, &recovered, NULL));
    TEST_ASSERT(recovered && !strcmp(err, "invalid tool call"));
    free(content); free(reasoning); tool_calls_free(&calls);
    if (test_failures == failures_before)
        printf("mellum-server: the generation tracker follows a Hermes call, and a malformed one is recovered\n");
}

/* With the real vocabulary (DS4_TEST_MELLUM_MODEL): each rendered case, tokenized as the server tokenizes a prompt
 * (ds4_tokenize_rendered_chat), gives the template's ids. */
static void test_mellum_server_ids(void) {
    const char *model = getenv("DS4_TEST_MELLUM_MODEL");
    if (!model || !model[0]) {
        puts("mellum-server-ids: DS4_TEST_MELLUM_MODEL unset, skipped");
        return;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = { .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = 2048 };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0 && engine != NULL);
    if (!engine) return;
    int bad = 0;
    for (size_t i = 0; i < sizeof(msc_cases) / sizeof(msc_cases[0]); i++) {
        const msc_case *c = &msc_cases[i];
        chat_msgs msgs;
        char *schemas;
        tool_schema_orders orders;
        mellum_server_case_request(c, &msgs, &schemas, &orders);
        char *text = render_chat_prompt_text_for_syntax(SERVER_MODEL_SYNTAX_HERMES, &msgs, schemas, &orders,
                                                        c->think ? DS4_THINK_HIGH : DS4_THINK_NONE);
        ds4_tokens got = {0};
        ds4_tokenize_rendered_chat(engine, text ? text : "", &got);
        bool same = got.len == c->nids;
        for (int k = 0; same && k < c->nids; k++) same = got.v[k] == c->ids[k];
        if (!same) {
            fprintf(stderr, "mellum-server-ids: \"%s\": %d ids, the template's %d\n", c->name, got.len, c->nids);
            bad++;
        }
        ds4_tokens_free(&got);
        free(text);
        free(schemas);
        tool_schema_orders_free(&orders);
        chat_msgs_free(&msgs);
    }
    TEST_ASSERT(bad == 0);
    if (!bad) printf("mellum-server-ids: %zu rendered requests tokenize to 2.1's template ids\n",
                     sizeof(msc_cases) / sizeof(msc_cases[0]));
    ds4_engine_close(engine);
}
