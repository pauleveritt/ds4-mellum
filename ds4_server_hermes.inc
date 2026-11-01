/* Mellum 2.1, included by ds4_server.c where this code sits: not a separate translation unit. */

/* Mellum's answer follows "</think>\n\n", and its reasoning sits between "<think>\n" and
 * "\n</think>" (the model writes the opening tag itself): strip those newlines. */
static void hermes_trim_split(char **content_out, char **reasoning_out) {
    if (reasoning_out && *reasoning_out) {
        const char *r = *reasoning_out;
        while (*r == '\n') r++;
        char *t = xstrdup(r);
        size_t n = strlen(t);
        while (n > 0 && t[n - 1] == '\n') t[--n] = '\0';
        free(*reasoning_out);
        *reasoning_out = t;
    }
    qwen_trim_split(content_out, NULL);
}

/* One Hermes call object, {"name": ..., "arguments": ...}, read as json.loads reads it: the
 * arguments' JSON text is kept as the model wrote it (an object), or taken from a JSON string
 * holding it; other keys are skipped; a missing name or a malformed value refuses the call. */
static bool hermes_parse_call(const char **p, tool_call *out) {
    char *name = NULL, *args = NULL;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto bad;
        json_ws(p);
        if (**p != ':') { free(key); goto bad; }
        (*p)++;
        json_ws(p);
        bool ok = true;
        if (!strcmp(key, "name")) {
            free(name);
            name = NULL;
            ok = json_string(p, &name);
        } else if (!strcmp(key, "arguments")) {
            free(args);
            args = NULL;
            if (**p == '"') {
                /* a JSON string holding the arguments: it must decode to one whole object */
                ok = json_string(p, &args);
                if (ok) {
                    const char *q = args;
                    json_ws(&q);
                    ok = *q == '{' && json_skip_value(&q);
                    if (ok) {
                        json_ws(&q);
                        ok = *q == '\0';
                    }
                }
            } else if (**p == '{') {
                const char *begin = *p;
                ok = json_skip_value(p);
                if (ok) args = xstrndup(begin, (size_t)(*p - begin));
            } else {
                ok = false;
            }
        } else {
            ok = json_skip_value(p);
        }
        free(key);
        if (!ok) goto bad;
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            json_ws(p);
            if (**p == '}') goto bad;          /* a trailing comma */
        } else if (**p != '}') {
            goto bad;
        }
    }
    if (**p != '}' || !name || !name[0]) goto bad;
    (*p)++;
    out->name = name;
    out->arguments = args ? args : xstrdup("{}");
    return true;
bad:
    free(name);
    free(args);
    return false;
}

/* Mellum 2.1's output: reasoning in a think block, the answer, then any number of
 * <tool_call>{JSON}</tool_call> blocks.  A call is read as JSON, so a marker inside one of its
 * strings is text.  raw_tool_text keeps the blocks as generated for the next request's render. */
static bool parse_hermes_generated_message_ex(const char *text,
                                              bool require_thinking_closed,
                                              char **content_out,
                                              char **reasoning_out,
                                              tool_calls *calls) {
    static const char tool_start[] = "<tool_call>";
    static const char tool_end[] = "</tool_call>";
    text = text ? text : "";
    const char *tool_search = text;
    bool recovered_unclosed_tool = false;
    if (require_thinking_closed) {
        /* the first </think>, as the template splits it: the model's own close comes before any
         * call, and a later one is text inside a call's JSON string */
        const char *think_end = strstr(text, "</think>");
        if (!think_end) {
            const char *candidate = strstr(text, tool_start);
            if (!candidate || !strstr(candidate, tool_end)) {
                fprintf(stderr, "ds4-server: thinking not closed, ignoring incomplete Hermes tool calls in reasoning\n");
                ds4_local_unterminated_reasoning(text, content_out, reasoning_out);
                return true;
            }
            tool_search = candidate;
            recovered_unclosed_tool = true;
        } else {
            tool_search = think_end + strlen("</think>");
        }
    }
    const char *start = strstr(tool_search, tool_start);
    if (!start) {
        split_reasoning_content(text, strlen(text), content_out, reasoning_out);
        hermes_trim_split(content_out, reasoning_out);
        return true;
    }
    const char *raw_block_start = start;
    while (raw_block_start > text && (raw_block_start[-1] == '\n' || raw_block_start[-1] == ' '))
        raw_block_start--;
    const size_t content_len = trim_tool_separator_ws(text, 0, (size_t)(raw_block_start - text));
    const char *p = start;
    for (;;) {
        p = skip_ascii_ws(p);
        if (strncmp(p, tool_start, strlen(tool_start)) != 0) break;
        p += strlen(tool_start);
        tool_call tc = {0};
        if (!hermes_parse_call(&p, &tc)) return false;
        p = skip_ascii_ws(p);
        if (strncmp(p, tool_end, strlen(tool_end)) != 0) {
            free(tc.name);
            free(tc.arguments);
            return false;
        }
        p += strlen(tool_end);
        tool_calls_push(calls, tc);
    }
    if (calls->len == 0) return false;
    free(calls->raw_tool_text);
    calls->raw_tool_text = xstrndup(raw_block_start, (size_t)(p - raw_block_start));
    if (recovered_unclosed_tool) {
        ds4_unterminated_reasoning_before_tool(text, content_len, content_out, reasoning_out);
    } else {
        split_reasoning_content(text, content_len, content_out, reasoning_out);
    }
    hermes_trim_split(content_out, reasoning_out);
    return true;
}
