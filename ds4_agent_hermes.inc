/* Mellum 2.1, included by ds4_agent.c where this code sits: not a separate translation unit. */

/* ---- Hermes: <tool_call>\n{"name": "read", "arguments": {"path": "hello.py"}}\n</tool_call> ---- */

enum {
    AGENT_HERMES_OPEN,        /* the call's opening brace */
    AGENT_HERMES_TOP_KEY,
    AGENT_HERMES_TOP_COLON,
    AGENT_HERMES_TOP_VALUE,
    AGENT_HERMES_TOP_AFTER,
    AGENT_HERMES_ARGS_KEY,
    AGENT_HERMES_ARGS_COLON,
    AGENT_HERMES_ARGS_VALUE,
    AGENT_HERMES_ARGS_AFTER,
    AGENT_HERMES_CLOSE,       /* the call object is closed; </tool_call> is next */
    AGENT_HERMES_PROSE,       /* after a call: prose, skipped until the next <tool_call> */
    AGENT_HERMES_DEFERRED,    /* a write or edit failed the strict parse: wait for </tool_call>, then repair */
};

#define AGENT_HERMES_MAX_DEPTH 256

static const char *agent_memmem(const char *hay, size_t hay_len,
                                const char *needle, size_t needle_len);

/* Is the JSON string at s (which starts with a quote) complete?  Gives the offset just past its
 * closing quote.  A backslash makes the next byte inert, which is all that matters to find the end. */
static bool agent_hermes_string_end(const char *s, const char *end, const char **past) {
    bool escape = false;
    for (const char *q = s + 1; q < end; q++) {
        if (escape) escape = false;
        else if (*q == '\\') escape = true;
        else if (*q == '"') { *past = q + 1; return true; }
    }
    return false;
}

/* Decode the complete string at s (starting at its quote); NULL when its escapes are invalid, or
 * when it spells \u0000: a C string would end there and the argument would quietly lose its tail. */
static char *agent_hermes_decode_string(const char *s) {
    for (const char *q = s + 1; *q && *q != '"'; q++) {
        if (*q != '\\') continue;
        if (q[1] == 'u' && !strncmp(q + 2, "0000", 4)) return NULL;
        if (q[1]) q++;
    }
    char *out = NULL;
    const char *q = s;
    if (!json_string(&q, &out)) return NULL;
    return out;
}

/* A strict JSON validator (RFC 8259 numbers and literals, no trailing commas, no hex, nan or inf),
 * for the raw values: json_number takes whatever strtod does.  Raw control characters inside
 * strings are let through, as the argument strings are; \u0000 is not, since every string in a
 * call is decoded to a C string sooner or later, and it would end there. */
static bool agent_hermes_valid_value(const char **p, const char *end, int depth) {
    if (depth > AGENT_HERMES_MAX_DEPTH) return false;
    *p = agent_skip_ascii_space(*p, end);
    if (*p >= end) return false;
    const char *q = *p;
    if (*q == '"') {
        for (q++; q < end && *q != '"'; q++) {
            if (*q != '\\') continue;
            if (++q >= end) return false;
            if (*q == 'u') {
                for (int i = 1; i <= 4; i++) if (q + i >= end || json_hex(q[i]) < 0) return false;
                if (!strncmp(q + 1, "0000", 4)) return false;
                q += 4;
            } else if (!strchr("\"\\/bfnrt", *q)) {
                return false;
            }
        }
        if (q >= end) return false;
        *p = q + 1;
        return true;
    }
    if (*q == '{' || *q == '[') {
        const char close = *q == '{' ? '}' : ']';
        q = agent_skip_ascii_space(q + 1, end);
        if (q < end && *q == close) { *p = q + 1; return true; }
        for (;;) {
            q = agent_skip_ascii_space(q, end);
            if (close == '}') {
                if (q >= end || *q != '"' || !agent_hermes_valid_value(&q, end, depth + 1)) return false;
                q = agent_skip_ascii_space(q, end);
                if (q >= end || *q != ':') return false;
                q++;
            }
            if (!agent_hermes_valid_value(&q, end, depth + 1)) return false;
            q = agent_skip_ascii_space(q, end);
            if (q >= end) return false;
            if (*q == close) { *p = q + 1; return true; }
            if (*q != ',') return false;
            q++;
        }
    }
    if (end - q >= 4 && !strncmp(q, "true", 4)) { *p = q + 4; return true; }
    if (end - q >= 5 && !strncmp(q, "false", 5)) { *p = q + 5; return true; }
    if (end - q >= 4 && !strncmp(q, "null", 4)) { *p = q + 4; return true; }
    if (q < end && *q == '-') q++;
    if (q >= end || *q < '0' || *q > '9') return false;
    if (*q == '0') q++;
    else while (q < end && *q >= '0' && *q <= '9') q++;
    if (q < end && *q == '.') {
        q++;
        if (q >= end || *q < '0' || *q > '9') return false;
        while (q < end && *q >= '0' && *q <= '9') q++;
    }
    if (q < end && (*q == 'e' || *q == 'E')) {
        q++;
        if (q < end && (*q == '+' || *q == '-')) q++;
        if (q >= end || *q < '0' || *q > '9') return false;
        while (q < end && *q >= '0' && *q <= '9') q++;
    }
    *p = q;
    return true;
}

static bool agent_hermes_valid_json(const char *s, size_t n) {
    const char *p = s, *end = s + n;
    if (!agent_hermes_valid_value(&p, end, 0)) return false;
    return agent_skip_ascii_space(p, end) == end;
}

/* Arguments sent as a JSON string: the decoded text is an object; read its members as arguments. */
static bool agent_hermes_add_args_from_json(agent_dsml_parser *p, const char *json) {
    if (!agent_hermes_valid_json(json, strlen(json))) return false;
    const char *q = json;
    json_ws(&q);
    if (*q != '{') return false;
    q++;
    json_ws(&q);
    while (*q && *q != '}') {
        char *key = NULL;
        if (!json_string(&q, &key)) return false;
        json_ws(&q);
        q++;                                   /* the colon: the validator saw it */
        json_ws(&q);
        char *value = NULL;
        bool is_string = *q == '"';
        bool ok = is_string ? json_string(&q, &value) : json_raw_value(&q, &value);
        if (!ok) { free(key); return false; }
        agent_tool_call_add_arg_owned(&p->current, key, value, is_string);
        free(key);
        json_ws(&q);
        if (*q == ',') { q++; json_ws(&q); }
    }
    return true;
}

/* Pathology 148: a value delimited with backticks, as a JavaScript template string is.  Reported at once, not held
 * for a repair: a quote repair has nothing to do with it. */
static void agent_hermes_backtick_error(agent_dsml_parser *p, const char *key) {
    char msg[160];
    snprintf(msg, sizeof(msg), "argument \"%.32s\" is delimited with backticks: a JSON string needs double quotes, "
             "with each quote inside written \\\"", key ? key : "");
    p->hermes_deferred = true;
    agent_dsml_set_error(p, msg);
}

/* The key whose value starts at `at` inside a raw value ("key": `), or "" when there is none to name. */
static char *agent_hermes_key_before(const char *from, const char *at) {
    const char *q = at;
    while (q > from && (q[-1] == ' ' || q[-1] == '\t' || q[-1] == '\r' || q[-1] == '\n')) q--;
    if (q <= from || q[-1] != ':') return xstrdup("");
    q--;
    while (q > from && (q[-1] == ' ' || q[-1] == '\t' || q[-1] == '\r' || q[-1] == '\n')) q--;
    if (q <= from || q[-1] != '"') return xstrdup("");
    const char *close = --q;
    while (q > from && q[-1] != '"') q--;
    return q > from ? xstrndup(q, (size_t)(close - q)) : xstrdup("");
}

/* Scan the raw (non-string) value being read since p->param_value_start.  Returns 1 when complete
 * (end set), 0 when it needs more bytes, -1 when it cannot be JSON, -2 at a backtick outside a
 * string (end set to it). */
static int agent_hermes_raw_value_end(agent_dsml_parser *p, size_t *end) {
    const bool bracketed = p->raw[p->param_value_start] == '{' || p->raw[p->param_value_start] == '[';
    for (size_t i = p->hermes_scan; i < p->raw_len; i++) {
        const char c = p->raw[i];
        p->hermes_scan = i + 1;
        if (p->hermes_in_string) {
            if (p->hermes_escape) p->hermes_escape = false;
            else if (c == '\\') p->hermes_escape = true;
            else if (c == '"') p->hermes_in_string = false;
            continue;
        }
        if (c == '"') { p->hermes_in_string = true; continue; }
        if (c == '`') { *end = i; return -2; }
        if (c == '{' || c == '[') {
            if (++p->hermes_depth > AGENT_HERMES_MAX_DEPTH) return -1;
            continue;
        }
        if (c == '}' || c == ']') {
            if (bracketed) {
                if (--p->hermes_depth == 0) { *end = i + 1; return 1; }
                continue;
            }
            p->hermes_scan = i;
            *end = i;
            return 1;
        }
        if (!bracketed && (c == ',' || c == ' ' || c == '\t' || c == '\r' || c == '\n')) {
            p->hermes_scan = i;
            *end = i;
            return 1;
        }
    }
    return 0;
}

/* Finish the argument value in PARAM_VALUE state, when its end has arrived.  False: wait for bytes. */
static bool agent_hermes_finish_value(agent_dsml_parser *p) {
    const char *name = p->param_name ? p->param_name : "";
    if (p->param_is_string) {
        const char *past = NULL;
        bool escape = p->hermes_escape;
        for (const char *q = p->raw + p->hermes_scan; q < p->raw + p->raw_len; q++) {
            if (escape) escape = false;
            else if (*q == '\\') escape = true;
            else if (*q == '"') { past = q + 1; break; }
        }
        if (!past) {
            p->hermes_scan = p->raw_len;
            p->hermes_escape = escape;
            return false;
        }
        char *value = agent_hermes_decode_string(p->raw + p->param_value_start - 1);
        if (!value) {
            agent_dsml_set_error(p, "invalid string in a tool call argument (bad escape or \\u0000)");
            return false;
        }
        if (p->hermes_skip) free(value);
        else agent_tool_call_add_arg_owned(&p->current, name, value, true);
        p->parse_pos = (size_t)(past - p->raw);
    } else {
        size_t end = 0;
        int rc = agent_hermes_raw_value_end(p, &end);
        if (rc == -2) {
            char *key = agent_hermes_key_before(p->raw + p->param_value_start, p->raw + end);
            agent_hermes_backtick_error(p, key);
            free(key);
            return false;
        }
        if (rc < 0) {
            agent_dsml_set_error(p, "tool call argument nests too deeply");
            return false;
        }
        if (rc == 0) return false;
        char *value = xstrndup(p->raw + p->param_value_start, end - p->param_value_start);
        if (!agent_hermes_valid_json(value, strlen(value))) {
            free(value);
            agent_dsml_set_error(p, "invalid JSON value in a tool call argument");
            return false;
        }
        if (p->hermes_skip) free(value);
        else agent_tool_call_add_arg_owned(&p->current, name, value, false);
        p->parse_pos = end;
    }
    free(p->param_name);
    p->param_name = NULL;
    p->param_close_prefix = false;
    p->hermes_phase = p->hermes_skip ? AGENT_HERMES_TOP_AFTER : AGENT_HERMES_ARGS_AFTER;
    p->hermes_skip = false;
    p->state = AGENT_DSML_STRUCTURAL;
    return true;
}

/* Start reading the value at cur (an argument's, or a skipped top-level key's) in PARAM_VALUE. */
static bool agent_hermes_begin_value(agent_dsml_parser *p, const char *cur, bool skip) {
    free(p->param_name);
    p->param_name = skip ? NULL : xstrdup(p->hermes_key ? p->hermes_key : "");
    p->hermes_skip = skip;
    p->param_close_prefix = false;
    p->hermes_scan = p->parse_pos;
    p->hermes_depth = 0;
    p->hermes_escape = false;
    p->hermes_in_string = false;
    if (*cur == '"') {
        p->param_is_string = true;
        p->param_value_start = p->parse_pos + 1;
        p->hermes_scan = p->param_value_start;
    } else if (*cur == '{' || *cur == '[' || *cur == '-' || (*cur >= '0' && *cur <= '9') ||
               *cur == 't' || *cur == 'f' || *cur == 'n') {
        p->param_is_string = false;
        p->param_value_start = p->parse_pos;
    } else if (*cur == '`') {
        agent_hermes_backtick_error(p, skip ? "" : p->hermes_key);
        return false;
    } else {
        agent_dsml_set_error(p, "invalid JSON value in a tool call argument");
        return false;
    }
    p->state = AGENT_DSML_PARAM_VALUE;
    return true;
}

static void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n);
static void agent_dsml_finish(agent_dsml_parser *p);

/* Match pattern at s: a space skips JSON whitespace, any other byte must be there.  The end, or NULL. */
static const char *agent_hermes_eat(const char *s, const char *end, const char *pattern) {
    for (; *pattern; pattern++) {
        if (*pattern == ' ') s = agent_skip_ascii_space(s, end);
        else if (s < end && *s == *pattern) s++;
        else return NULL;
    }
    return s;
}

/* Is the quote at q unescaped (an even run of backslashes before it, back to from)? */
static bool agent_hermes_bare_quote(const char *from, const char *q) {
    size_t backslashes = 0;
    while (q - backslashes > from && q[-1 - (ptrdiff_t)backslashes] == '\\') backslashes++;
    return *q == '"' && backslashes % 2 == 0;
}

/* A value that holds a tool key's boundary (", "path": and the like) has folded another member into itself:
 * that is no bare quote, so the call is not repaired. */
static bool agent_hermes_folds_member(const char *s, const char *end) {
    static const char *const keys[] = {"path", "content", "edits", "oldText", "newText"};
    for (const char *q = s; q < end; q++) {
        if (!agent_hermes_bare_quote(s, q)) continue;
        for (size_t k = 0; k < sizeof(keys) / sizeof(keys[0]); k++) {
            char pattern[32];
            snprintf(pattern, sizeof(pattern), "\" , \"%s\" :", keys[k]);
            if (agent_hermes_eat(q, end, pattern)) return true;
        }
    }
    return false;
}

/* A value's end in an edit's items: the quote that opens the item boundary b (" , "newText" : " or "}, {"oldText": ")
 * at q, if any.  The boundary's end, or NULL. */
static const char *agent_hermes_edit_boundary(const char *q, const char *end, bool to_new) {
    return agent_hermes_eat(q, end, to_new ? "\" , \"newText\" : \"" : "\" } , { \"oldText\" : \"");
}

/* The call body (from its opening brace to </tool_call>) of a write or an edit whose string values hold bare
 * quotes, with them escaped; NULL when the body is not exactly {"name": "write", "arguments": {"path": <a valid
 * string>, "content": "..."}} or {"name": "edit", "arguments": {"path": <a valid string>, "edits": [{"oldText":
 * "...", "newText": "..."}, ...]}} (or the older single-edit form, "oldText": "...", "newText": "..." after the
 * path), when an edit's item boundaries are ambiguous, or when the rewrite changes nothing.  The path is not
 * repaired: a wrong path is another file. */
static char *agent_hermes_repair_quotes(const char *s, size_t n) {
    const char *end = s + n, *past = NULL;
    const char *q = agent_hermes_eat(s, end, " { \"name\" : \"write\" , \"arguments\" : { \"path\" : ");
    const bool edit = !q;
    if (edit) q = agent_hermes_eat(s, end, " { \"name\" : \"edit\" , \"arguments\" : { \"path\" : ");
    if (!q || q >= end || *q != '"' || !agent_hermes_string_end(q, end, &past)) return NULL;
    char *path = agent_hermes_decode_string(q);
    if (!path) return NULL;
    free(path);
    const char *v = agent_hermes_eat(past, end, edit ? " , \"edits\" : [ { \"oldText\" : \"" : " , \"content\" : \"");
    const char *legacy = edit && !v ? agent_hermes_eat(past, end, " , \"oldText\" : \"") : NULL;
    if (legacy) v = legacy;
    if (!v) return NULL;
    /* the tail, read backwards: " } } for a write or a single edit, " } ] } } for edits, and whitespace */
    const char *v_end = end;
    for (const char *lit = edit && !legacy ? "}}]}\"" : "}}\""; *lit; lit++) {
        while (v_end > v && (v_end[-1] == ' ' || v_end[-1] == '\t' || v_end[-1] == '\r' || v_end[-1] == '\n')) v_end--;
        if (v_end <= v || v_end[-1] != *lit) return NULL;
        v_end--;
    }
    /* the values: [cut[2i], cut[2i+1]).  An edit's alternate oldText and newText, cut at each boundary; a boundary
     * that comes out of turn (in the single-edit form, any after its one newText), or a list that does not end in a
     * newText, is ambiguous */
    const char **cut = xmalloc(((size_t)(v_end - v) + 2) * sizeof(*cut));
    size_t ncut = 0;
    cut[ncut++] = v;
    bool to_new = true;
    for (const char *c = v; edit && c < v_end; c++) {
        if (!agent_hermes_bare_quote(v, c)) continue;
        const char *b = agent_hermes_edit_boundary(c, v_end, to_new);
        if (b && !(legacy && !to_new)) {
            cut[ncut++] = c;
            cut[ncut++] = b;
            c = b - 1;
            to_new = !to_new;
        } else if (b || agent_hermes_edit_boundary(c, v_end, !to_new)) {
            ncut = 0;
            break;
        }
    }
    if (edit && to_new) ncut = 0;
    if (ncut) cut[ncut++] = v_end;
    for (size_t i = 0; i < ncut; i += 2) {
        if (agent_hermes_folds_member(cut[i], cut[i + 1])) ncut = 0;
    }
    if (!ncut) { free(cut); return NULL; }
    char *out = xmalloc(2 * n + 1), *o = out;
    const char *from = s;
    bool changed = false;
    for (size_t i = 0; i < ncut; i += 2) {
        memcpy(o, from, (size_t)(cut[i] - from));
        o += cut[i] - from;
        for (const char *c = cut[i]; c < cut[i + 1]; c++) {
            if (agent_hermes_bare_quote(cut[i], c)) { *o++ = '\\'; changed = true; }
            *o++ = *c;
        }
        from = cut[i + 1];
    }
    memcpy(o, from, (size_t)(end - from));
    o[end - from] = '\0';
    free(cut);
    if (!changed) { free(out); return NULL; }
    return out;
}

/* The deferred call has reached </tool_call> at close_at: parse its repair strictly.  True: the call is pushed. */
static bool agent_hermes_repaired_call(agent_dsml_parser *p, size_t close_at) {
    char *body = agent_hermes_repair_quotes(p->raw + p->hermes_call_start, close_at - p->hermes_call_start);
    if (!body) return false;
    agent_dsml_parser q = {.syntax = AGENT_TOOL_SYNTAX_HERMES, .state = AGENT_DSML_SEARCH, .hermes_deferred = true};
    agent_dsml_feed(&q, "<tool_call>", strlen("<tool_call>"));
    agent_dsml_feed(&q, body, strlen(body));
    agent_dsml_feed(&q, "</tool_call>", strlen("</tool_call>"));
    agent_dsml_finish(&q);
    free(body);
    const bool ok = q.state == AGENT_DSML_DONE && q.calls.len == 1;
    if (ok) {
        agent_tool_call_free(&p->current);
        agent_tool_calls_push(&p->calls, &q.calls.v[0]);
    }
    agent_dsml_parser_free(&q);
    return ok;
}

/* Streaming parser for the Hermes call object.  Only an argument's value streams through
 * PARAM_VALUE (a string carries the code being written; numbers, literals and nested values are
 * scanned to their end and kept as raw JSON text); everything else structural is parsed once the
 * bytes of that token have arrived.  Incomplete input waits; malformed input errors. */
static void agent_hermes_tool_parse(agent_dsml_parser *p) {
    static const char start[] = "<tool_call>";
    static const char close[] = "</tool_call>";

    if (p->raw_len < sizeof(start) - 1 || memcmp(p->raw, start, sizeof(start) - 1) != 0)
        return;
    while (p->state == AGENT_DSML_STRUCTURAL || p->state == AGENT_DSML_PARAM_VALUE) {
        if (p->state == AGENT_DSML_PARAM_VALUE) {
            if (!agent_hermes_finish_value(p)) return;
            continue;
        }
        const char *raw = p->raw;
        const char *end = p->raw + p->raw_len;
        const char *cur = agent_skip_ascii_space(raw + p->parse_pos, end);
        p->parse_pos = (size_t)(cur - raw);
        if (cur >= end) return;

        if (p->glm_after_call) {
            if (agent_bytes_starts_with(cur, end, start)) {
                p->parse_pos += sizeof(start) - 1;
                p->glm_after_call = false;
                p->hermes_phase = AGENT_HERMES_OPEN;
                continue;
            }
            if (agent_bytes_partial_prefix_at(cur, end, start)) return;
            /* prose after a call is tolerated, as in training; the turn ends when the model stops,
             * and a later <tool_call> is still a call */
            p->glm_after_call = false;
            p->hermes_phase = AGENT_HERMES_PROSE;
            continue;
        }

        switch (p->hermes_phase) {
        case AGENT_HERMES_PROSE: {
            const char *at = agent_memmem(cur, (size_t)(end - cur), start, sizeof(start) - 1);
            if (at) {
                p->parse_pos = (size_t)(at - raw) + sizeof(start) - 1;
                p->hermes_phase = AGENT_HERMES_OPEN;
                break;
            }
            /* keep a possible half-written opener for the next byte */
            size_t keep = sizeof(start) - 2 < (size_t)(end - cur) ? sizeof(start) - 2 : (size_t)(end - cur);
            p->parse_pos = (size_t)(end - raw) - keep;
            return;
        }
        case AGENT_HERMES_OPEN:
            p->hermes_call_start = p->parse_pos;
            if (*cur != '{') {
                agent_dsml_set_error(p, "expected a JSON object after <tool_call>");
                return;
            }
            p->parse_pos++;
            p->hermes_comma = false;
            p->hermes_phase = AGENT_HERMES_TOP_KEY;
            break;
        case AGENT_HERMES_TOP_KEY:
        case AGENT_HERMES_ARGS_KEY: {
            const bool top = p->hermes_phase == AGENT_HERMES_TOP_KEY;
            if (*cur == '}') {
                if (p->hermes_comma) {
                    agent_dsml_set_error(p, "trailing comma in the tool call");
                    return;
                }
                p->parse_pos++;
                p->hermes_phase = top ? AGENT_HERMES_CLOSE : AGENT_HERMES_TOP_AFTER;
                break;
            }
            if (*cur != '"') {
                agent_dsml_set_error(p, "expected a quoted key in the tool call");
                return;
            }
            const char *past = NULL;
            if (!agent_hermes_string_end(cur, end, &past)) return;
            char *key = agent_hermes_decode_string(cur);
            if (!key) {
                agent_dsml_set_error(p, "invalid string escape in a tool call key");
                return;
            }
            free(p->hermes_key);
            p->hermes_key = key;
            p->hermes_comma = false;
            p->parse_pos = (size_t)(past - raw);
            p->hermes_phase = top ? AGENT_HERMES_TOP_COLON : AGENT_HERMES_ARGS_COLON;
            break;
        }
        case AGENT_HERMES_TOP_COLON:
        case AGENT_HERMES_ARGS_COLON:
            if (*cur != ':') {
                agent_dsml_set_error(p, "expected ':' after a key in the tool call");
                return;
            }
            p->parse_pos++;
            p->hermes_phase = p->hermes_phase == AGENT_HERMES_TOP_COLON ?
                              AGENT_HERMES_TOP_VALUE : AGENT_HERMES_ARGS_VALUE;
            break;
        case AGENT_HERMES_TOP_VALUE: {
            const char *key = p->hermes_key ? p->hermes_key : "";
            if (!strcmp(key, "name")) {
                if (*cur != '"') {
                    agent_dsml_set_error(p, "tool call name must be a string");
                    return;
                }
                const char *past = NULL;
                if (!agent_hermes_string_end(cur, end, &past)) return;
                char *name = agent_hermes_decode_string(cur);
                if (!name || !name[0]) {
                    free(name);
                    agent_dsml_set_error(p, "tool call without name");
                    return;
                }
                free(p->current.name);
                p->current.name = name;
                p->parse_pos = (size_t)(past - raw);
                p->hermes_phase = AGENT_HERMES_TOP_AFTER;
            } else if (!strcmp(key, "arguments")) {
                /* a repeated "arguments" replaces the earlier object, as json.loads keeps the last */
                for (int i = 0; i < p->current.argc; i++) {
                    free(p->current.args[i].name);
                    free(p->current.args[i].value);
                }
                p->current.argc = 0;
                if (*cur == '"') {
                    /* arguments as a JSON string: decode it, then read the object inside */
                    const char *past = NULL;
                    if (!agent_hermes_string_end(cur, end, &past)) return;
                    char *text = agent_hermes_decode_string(cur);
                    if (!text || !agent_hermes_add_args_from_json(p, text)) {
                        free(text);
                        agent_dsml_set_error(p, "tool call arguments must be an object");
                        return;
                    }
                    free(text);
                    p->parse_pos = (size_t)(past - raw);
                    p->hermes_phase = AGENT_HERMES_TOP_AFTER;
                    break;
                }
                if (*cur != '{') {
                    agent_dsml_set_error(p, "tool call arguments must be an object");
                    return;
                }
                p->parse_pos++;
                p->hermes_comma = false;
                p->hermes_phase = AGENT_HERMES_ARGS_KEY;
            } else {
                /* an unknown top-level key (an id, say): its value is read and dropped */
                if (!agent_hermes_begin_value(p, cur, true)) return;
            }
            break;
        }
        case AGENT_HERMES_TOP_AFTER:
        case AGENT_HERMES_ARGS_AFTER: {
            const bool top = p->hermes_phase == AGENT_HERMES_TOP_AFTER;
            if (*cur == ',') {
                p->parse_pos++;
                p->hermes_comma = true;
                p->hermes_phase = top ? AGENT_HERMES_TOP_KEY : AGENT_HERMES_ARGS_KEY;
            } else if (*cur == '}') {
                p->parse_pos++;
                p->hermes_phase = top ? AGENT_HERMES_CLOSE : AGENT_HERMES_TOP_AFTER;
            } else {
                agent_dsml_set_error(p, top ? "expected ',' or '}' in the tool call object"
                                            : "expected ',' or '}' in the tool call arguments");
                return;
            }
            break;
        }
        case AGENT_HERMES_ARGS_VALUE:
            if (!agent_hermes_begin_value(p, cur, false)) return;
            break;
        case AGENT_HERMES_DEFERRED: {
            /* resume the search where the last one stopped, short of a half-arrived </tool_call> */
            size_t from = p->hermes_scan > sizeof(close) - 2 ? p->hermes_scan - (sizeof(close) - 2) : 0;
            if (from < p->parse_pos) from = p->parse_pos;
            const char *at = agent_memmem(raw + from, p->raw_len - from, close, sizeof(close) - 1);
            p->hermes_scan = p->raw_len;
            if (!at) return;
            if (!agent_hermes_repaired_call(p, (size_t)(at - raw))) {
                agent_dsml_set_error(p, p->hermes_deferred_error);
                return;
            }
            p->hermes_deferred = false;
            p->hermes_phase = AGENT_HERMES_CLOSE;
            p->parse_pos = (size_t)(at - raw) + sizeof(close) - 1;
            p->glm_after_call = true;
            break;
        }
        case AGENT_HERMES_CLOSE:
            if (agent_bytes_starts_with(cur, end, close)) {
                if (!p->current.name) {
                    agent_dsml_set_error(p, "tool call without name");
                    return;
                }
                p->parse_pos += sizeof(close) - 1;
                agent_tool_calls_push(&p->calls, &p->current);
                p->glm_after_call = true;
                break;
            }
            if (agent_bytes_partial_prefix_at(cur, end, close)) return;
            agent_dsml_set_error(p, "expected </tool_call> after the call object");
            return;
        }
    }
}
