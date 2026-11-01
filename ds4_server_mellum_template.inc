/* Mellum 2.1, included by ds4_server.c where this code sits: not a separate translation unit. */

/* ---- Mellum 2.1: its chat template, Hermes <tool_call> JSON ----
 *
 * The template renders a request's tools with `tool | tojson`, which is Python's
 * json.dumps(ensure_ascii=False): ", " and ": " separators, key order kept, non-ASCII
 * text as is.  The server keeps each tool's function object raw (parse_tools_value), so
 * the wrapper is rebuilt around it and the whole value written again in that form. */
static bool append_py_json_value(buf *b, const char **p, int depth) {
    json_ws(p);
    if (depth > JSON_MAX_NESTING) return false;
    if (**p == '"') {
        char *s = NULL;
        if (!json_string(p, &s)) return false;
        buf_putc(b, '"');
        for (const unsigned char *c = (const unsigned char *)s; *c; c++) {
            switch (*c) {
            case '"': buf_puts(b, "\\\""); break;
            case '\\': buf_puts(b, "\\\\"); break;
            case '\n': buf_puts(b, "\\n"); break;
            case '\r': buf_puts(b, "\\r"); break;
            case '\t': buf_puts(b, "\\t"); break;
            case '\b': buf_puts(b, "\\b"); break;
            case '\f': buf_puts(b, "\\f"); break;
            default:
                if (*c < 0x20) buf_printf(b, "\\u%04x", *c);
                else buf_putc(b, (char)*c);
            }
        }
        buf_putc(b, '"');
        free(s);
        return true;
    }
    if (**p == '{' || **p == '[') {
        const bool object = **p == '{';
        const char close = object ? '}' : ']';
        (*p)++;
        buf_putc(b, object ? '{' : '[');
        json_ws(p);
        bool first = true;
        while (**p && **p != close) {
            if (!first) buf_puts(b, ", ");
            first = false;
            if (object) {
                if (!append_py_json_value(b, p, depth + 1)) return false;
                json_ws(p);
                if (**p != ':') return false;
                (*p)++;
                buf_puts(b, ": ");
            }
            if (!append_py_json_value(b, p, depth + 1)) return false;
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != close) return false;
        (*p)++;
        buf_putc(b, close);
        return true;
    }
    /* numbers and literals as the client wrote them */
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    buf_append(b, start, (size_t)(*p - start));
    return true;
}

static void append_hermes_tools_block(buf *b, const char *tool_schemas) {
    buf_puts(b, "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>");
    const char *p = tool_schemas ? tool_schemas : "";
    json_ws(&p);
    while (*p) {
        const size_t mark = b->len;
        buf_puts(b, "\n{\"type\": \"function\", \"function\": ");
        if (!append_py_json_value(b, &p, 0)) {
            b->len = mark;
            break;
        }
        buf_putc(b, '}');
        json_ws(&p);
    }
    buf_puts(b, "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
                "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
                "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
}

/* Python's str.strip('\n') / lstrip('\n') / rstrip('\n'), into b */
static void append_strip_newlines(buf *b, const char *s, bool left, bool right) {
    size_t n = strlen(s);
    while (left && n && *s == '\n') { s++; n--; }
    while (right && n && s[n - 1] == '\n') n--;
    buf_append(b, s, n);
}

/* An assistant turn's calls as the template writes them: a newline before each but a first
 * one with no content, the client's argument text as is. */
static void append_hermes_tool_calls_text(buf *b, const tool_calls *calls, bool has_content) {
    for (int i = 0; calls && i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i > 0 || has_content) buf_putc(b, '\n');
        buf_puts(b, "<tool_call>\n{\"name\": \"");
        buf_puts(b, tc->name ? tc->name : "");
        buf_puts(b, "\", \"arguments\": ");
        buf_puts(b, tc->arguments && tc->arguments[0] ? tc->arguments : "{}");
        buf_puts(b, "}\n</tool_call>");
    }
}

static bool parse_hermes_generated_message_ex(const char *text, bool require_thinking_closed, char **content_out,
                                              char **reasoning_out, tool_calls *calls);

/* Tool memory restores a sampled block for every call id in it.  It is the turn's text only when it holds exactly
 * the calls the client sent back: names, order and decoded arguments.  Otherwise the sent calls are rendered. */
static bool hermes_raw_calls_match(const char *raw, const tool_calls *calls) {
    char *content = NULL, *reasoning = NULL;
    tool_calls parsed = {0};
    bool same = raw && raw[0] && parse_hermes_generated_message_ex(raw, false, &content, &reasoning, &parsed) &&
                parsed.len == calls->len;
    for (int i = 0; same && i < calls->len; i++) {
        const char *want = calls->v[i].arguments && calls->v[i].arguments[0] ? calls->v[i].arguments : "{}";
        const char *got = parsed.v[i].arguments && parsed.v[i].arguments[0] ? parsed.v[i].arguments : "{}";
        buf w = {0}, g = {0};
        same = calls->v[i].name && parsed.v[i].name && !strcmp(calls->v[i].name, parsed.v[i].name) &&
               append_py_json_value(&w, &want, 0) && append_py_json_value(&g, &got, 0) && !strcmp(w.ptr, g.ptr);
        buf_free(&w);
        buf_free(&g);
    }
    free(content);
    free(reasoning);
    tool_calls_free(&parsed);
    return same;
}

static void append_hermes_assistant_message(buf *b, const chat_msg *m, bool keep_reasoning) {
    const char *content = m->content ? m->content : "";
    char *split_reasoning = NULL, *split_content = NULL;
    const char *reasoning = m->reasoning;
    if (!reasoning) {
        /* no reasoning_content: the template takes it from a <think> block in the content */
        const char *close = strstr(content, "</think>");
        if (close) {
            /* the reasoning: before the first </think>, after the last <think> there; the content:
             * after the last </think>; both without the newlines next to the tags */
            const char *r = content;
            for (const char *o; (o = strstr(r, "<think>")) != NULL && o < close;) r = o + strlen("<think>");
            char *between = xstrndup(r, (size_t)(close - r));
            buf rb = {0};
            append_strip_newlines(&rb, between, true, true);
            free(between);
            split_reasoning = rb.ptr ? buf_take(&rb) : xstrdup("");
            const char *after = close;
            for (const char *o; (o = strstr(after, "</think>")) != NULL;) after = o + strlen("</think>");
            while (*after == '\n') after++;
            split_content = xstrdup(after);
            reasoning = split_reasoning;
            content = split_content;
        }
    }
    buf_puts(b, "<|im_start|>assistant\n");
    if (keep_reasoning && reasoning && reasoning[0]) {
        buf_puts(b, "<think>\n");
        append_strip_newlines(b, reasoning, true, true);
        buf_puts(b, "\n</think>\n\n");
        append_strip_newlines(b, content, true, false);
    } else {
        buf_puts(b, content);
    }
    const char *raw = m->calls.len && hermes_raw_calls_match(m->calls.raw_tool_text, &m->calls) ?
        m->calls.raw_tool_text : NULL;
    if (raw) {
        /* the calls as sampled (restored by tool memory when the client sent them back): the
         * KV's bytes.  With no content the template's separator is already written. */
        if (!content[0]) while (*raw == '\n' || *raw == ' ') raw++;
        buf_puts(b, raw);
    } else {
        append_hermes_tool_calls_text(b, &m->calls, content[0] != '\0');
    }
    buf_puts(b, "<|im_end|>\n");
    free(split_reasoning);
    free(split_content);
}

static bool hermes_is_tool_message(const chat_msg *m) {
    return !strcmp(m->role, "tool") || !strcmp(m->role, "function");
}

/* The template's message loop from message start on, and the generation prompt.  Reasoning is
 * kept after the latest user message that is not a wrapped tool response, wherever start is. */
static void append_hermes_conversation(buf *b, const chat_msgs *msgs, int start, ds4_think_mode think_mode) {
    const int n = msgs ? msgs->len : 0;
    int last_query = n - 1;
    for (int i = n - 1; i >= 0; i--) {
        const chat_msg *m = &msgs->v[i];
        const char *c = m->content ? m->content : "";
        const size_t cl = strlen(c);
        const bool wrapped = !strncmp(c, "<tool_response>", 15) && cl >= 16 &&
                             !strcmp(c + cl - 16, "</tool_response>");
        if (!strcmp(m->role, "user") && !wrapped) {
            last_query = i;
            break;
        }
    }
    for (int i = start; i < n; i++) {
        const chat_msg *m = &msgs->v[i];
        const char *content = m->content ? m->content : "";
        if (!strcmp(m->role, "user") || (role_is_system(m->role) && i > 0)) {
            /* a developer message is the system turn, as role_is_system has it */
            buf_printf(b, "<|im_start|>%s\n", role_is_system(m->role) ? "system" : "user");
            buf_puts(b, content);
            buf_puts(b, "<|im_end|>\n");
        } else if (!strcmp(m->role, "assistant")) {
            append_hermes_assistant_message(b, m, i > last_query);
        } else if (hermes_is_tool_message(m)) {
            if (i == 0 || !hermes_is_tool_message(&msgs->v[i - 1])) buf_puts(b, "<|im_start|>user");
            buf_puts(b, "\n<tool_response>\n");
            buf_puts(b, content);
            buf_puts(b, "\n</tool_response>");
            if (i == n - 1 || !hermes_is_tool_message(&msgs->v[i + 1])) buf_puts(b, "<|im_end|>\n");
        }
    }
    /* the template appends the generation prompt whenever add_generation_prompt is set, after a
     * final assistant turn too: the server has no assistant-prefill semantics for Mellum */
    buf_puts(b, "<|im_start|>assistant\n");
    if (!ds4_think_mode_enabled(think_mode)) buf_puts(b, "<think>\n\n</think>\n\n");
}

/* Mellum 2.1's chat_template.jinja: the first system message (with the tools block when
 * there are tools), user and later system turns as given, an assistant turn's reasoning
 * kept only after the latest user message that is not a wrapped tool response (so within
 * the current tool cycle), Hermes calls with the client's argument text, consecutive tool
 * messages in one user turn, and the generation prompt (an empty think block when thinking
 * is off).  Pinned against the template's text in tests/ds4_test.c --server. */
static char *render_hermes_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                            ds4_think_mode think_mode) {
    /* Anthropic keeps tool results in a user message (with <tool_result> wrappers and recorded
     * spans): as V4.1 does, render each as the tool message it is, and any text around it as user */
    int start = 0;
    chat_msgs fragments = {0};
    chat_msgs views = ds41_message_views(msgs, &fragments, &start);
    msgs = &views;
    buf out = {0};
    const int n = msgs ? msgs->len : 0;
    const bool first_system = n > 0 && role_is_system(msgs->v[0].role);
    if (tool_schemas && tool_schemas[0]) {
        buf_puts(&out, "<|im_start|>system\n");
        if (first_system) {
            buf_puts(&out, msgs->v[0].content ? msgs->v[0].content : "");
            buf_puts(&out, "\n\n");
        }
        append_hermes_tools_block(&out, tool_schemas);
    } else if (first_system) {
        buf_puts(&out, "<|im_start|>system\n");
        buf_puts(&out, msgs->v[0].content ? msgs->v[0].content : "");
        buf_puts(&out, "<|im_end|>\n");
    }
    append_hermes_conversation(&out, msgs, 0, think_mode);
    free(views.v);
    chat_msgs_free(&fragments);
    return buf_take(&out);
}

/* Continue a live session after the model's tool-call turn: its <|im_end|> was the stop token
 * and is not in the KV yet.  The tail is the full render from message start on. */
static char *render_hermes_live_tool_tail(const chat_msgs *msgs, int start, ds4_think_mode think_mode) {
    chat_msgs fragments = {0};
    chat_msgs views = ds41_message_views(msgs, &fragments, &start);
    buf out = {0};
    buf_puts(&out, "<|im_end|>\n");
    append_hermes_conversation(&out, &views, start, think_mode);
    free(views.v);
    chat_msgs_free(&fragments);
    return buf_take(&out);
}
