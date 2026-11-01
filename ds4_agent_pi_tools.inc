/* Mellum 2.1, included by ds4_agent.c where this code sits: not a separate translation unit. */

/* ============================================================================
 * Pi's Four Tools, For Mellum
 * ============================================================================
 *
 * Mellum 2.1 was trained on Pi's read, bash, edit and write, with Pi's argument names and Pi's
 * result texts, so for that model the dispatcher maps those four onto this agent's machinery.
 * The behaviour and the texts follow Pi (@earendil-works/pi-coding-agent 1.1.0, MIT licensed per
 * its package.json): dist/core/tools/read.js, write.js, edit.js, edit-diff.js, bash.js and
 * truncate.js.  Results are plain text; an error is the message Pi's tool would throw, returned the
 * same way, so the model reads Pi's words in both cases.
 */

#define PI_MAX_LINES 2000
#define PI_MAX_BYTES (50 * 1024)
#define PI_READ_MAX_BYTES (64 * 1024 * 1024)

static void pi_format_size(size_t bytes, char *out, size_t cap) {
    if (bytes < 1024) snprintf(out, cap, "%zuB", bytes);
    else if (bytes < 1024 * 1024) snprintf(out, cap, "%.1fKB", (double)bytes / 1024.0);
    else snprintf(out, cap, "%.1fMB", (double)bytes / (1024.0 * 1024.0));
}

static const char *pi_errno_code(int e) {
    switch (e) {
    case ENOENT: return "ENOENT";
    case EACCES: return "EACCES";
    case EISDIR: return "EISDIR";
    case ENOTDIR: return "ENOTDIR";
    case EEXIST: return "EEXIST";
    case EPERM: return "EPERM";
    case ENOSPC: return "ENOSPC";
    case EROFS: return "EROFS";
    case ELOOP: return "ELOOP";
    case ENAMETOOLONG: return "ENAMETOOLONG";
    default: return "EIO";
    }
}

static const char *pi_errno_text(int e) {
    switch (e) {
    case ENOENT: return "no such file or directory";
    case EACCES: return "permission denied";
    case EISDIR: return "illegal operation on a directory";
    case ENOTDIR: return "not a directory";
    case EEXIST: return "file already exists";
    case EPERM: return "operation not permitted";
    case ENOSPC: return "no space left on device";
    case EROFS: return "read-only file system";
    default: return strerror(e);
    }
}

/* The path as the file system should see it: Pi strips a leading @ and expands ~. */
static char *pi_resolve_path(const char *path) {
    if (path[0] == '@') path++;
    if (path[0] == '~' && (path[1] == '/' || !path[1])) {
        const char *home = getenv("HOME");
        if (home && home[0]) {
            size_t n = strlen(home) + strlen(path);
            char *out = xmalloc(n + 1);
            snprintf(out, n + 1, "%s%s", home, path + 1);
            return out;
        }
    }
    return xstrdup(path);
}

/* Node's message for a failed system call: "ENOENT: no such file or directory, open '/abs/path'". */
static char *pi_fs_error(int e, const char *syscall_name, const char *path) {
    char abs[PATH_MAX + 1] = {0};
    if (path[0] == '/') {
        snprintf(abs, sizeof(abs), "%s", path);
    } else {
        char cwd[PATH_MAX] = {0};
        if (!getcwd(cwd, sizeof(cwd))) cwd[0] = '\0';
        snprintf(abs, sizeof(abs), "%s/%s", cwd, path);
    }
    char *out = xmalloc(strlen(abs) + 128);
    sprintf(out, "%s: %s, %s '%s'", pi_errno_code(e), pi_errno_text(e), syscall_name, abs);
    return out;
}

/* Read a whole file (any size up to PI_READ_MAX_BYTES).  errno is set when it returns false. */
static bool pi_read_all(const char *path, agent_buf *out) {
    FILE *fp = agent_open_regular_file(path);
    if (!fp) return false;
    char tmp[16384];
    for (;;) {
        size_t n = fread(tmp, 1, sizeof(tmp), fp);
        if (n) {
            if (out->len + n > PI_READ_MAX_BYTES) {
                fclose(fp);
                errno = EFBIG;
                return false;
            }
            agent_buf_append(out, tmp, n);
        }
        if (n < sizeof(tmp)) {
            if (ferror(fp)) {
                int saved = errno;
                fclose(fp);
                errno = saved;
                return false;
            }
            break;
        }
    }
    fclose(fp);
    return true;
}

/* An argument Pi's schema makes a string: NULL when it is absent or was another JSON type, which
 * the tools refuse before touching a file or starting a shell. */
static const char *pi_string_arg(const agent_tool_call *call, const char *name) {
    for (int i = 0; i < call->argc; i++) {
        if (call->args[i].name && !strcmp(call->args[i].name, name))
            return call->args[i].is_string ? (call->args[i].value ? call->args[i].value : "") : NULL;
    }
    return NULL;
}

/* Parse an optional number argument.  Returns false when it is absent. */
static bool pi_number_arg(const agent_tool_call *call, const char *name, double *out) {
    const char *v = agent_tool_arg_value(call, name);
    if (!v || !v[0]) return false;
    char *end = NULL;
    double d = strtod(v, &end);
    if (end == v || !isfinite(d)) return false;
    *out = d;
    return true;
}

/* ---- truncation (truncate.js) ---- */

typedef struct {
    bool truncated;
    bool by_lines;
    bool first_line_exceeds;
    size_t total_lines;
    size_t out_lines;
    size_t out_bytes;   /* of the head kept: the first out_bytes bytes of the content */
} pi_truncation;

/* splitLinesForCounting: no lines in "", and a final newline does not start another. */
static size_t pi_count_lines(const char *s, size_t n) {
    if (n == 0) return 0;
    size_t lines = 1;
    for (size_t i = 0; i < n; i++) if (s[i] == '\n') lines++;
    if (s[n - 1] == '\n') lines--;
    return lines;
}

static pi_truncation pi_truncate_head(const char *s, size_t n) {
    pi_truncation t = {.by_lines = true};
    t.total_lines = pi_count_lines(s, n);
    if (t.total_lines <= PI_MAX_LINES && n <= PI_MAX_BYTES) {
        t.out_lines = t.total_lines;
        t.out_bytes = n;
        return t;
    }
    t.truncated = true;
    const char *nl = memchr(s, '\n', n);
    size_t first = nl ? (size_t)(nl - s) : n;
    if (first > PI_MAX_BYTES) {
        t.by_lines = false;
        t.first_line_exceeds = true;
        return t;
    }
    size_t pos = 0, bytes = 0, count = 0;
    bool by_bytes = false;
    for (size_t i = 0; i < t.total_lines && i < PI_MAX_LINES; i++) {
        const char *line_end = memchr(s + pos, '\n', n - pos);
        size_t line_len = line_end ? (size_t)(line_end - (s + pos)) : n - pos;
        size_t lb = line_len + (i > 0 ? 1 : 0);
        if (bytes + lb > PI_MAX_BYTES) { by_bytes = true; break; }
        bytes += lb;
        count++;
        pos += line_len + 1;
        if (pos > n) pos = n;
    }
    t.by_lines = !by_bytes || (count >= PI_MAX_LINES && bytes <= PI_MAX_BYTES);
    t.out_lines = count;
    t.out_bytes = bytes;
    return t;
}

/* Byte offset where line k (0-based) starts; k must not exceed the number of newlines. */
static size_t pi_line_start(const char *s, size_t n, size_t k) {
    size_t pos = 0;
    while (k-- > 0) {
        const char *nl = memchr(s + pos, '\n', n - pos);
        pos = nl ? (size_t)(nl - s) + 1 : n;
    }
    return pos;
}

static char *pi_tool_read(const agent_tool_call *call) {
    const char *path_arg = pi_string_arg(call, "path");
    if (!path_arg || !path_arg[0]) return xstrdup("read requires a path string.");
    char *path = pi_resolve_path(path_arg);
    agent_buf file = {0};
    if (!pi_read_all(path, &file)) {
        int e = errno;
        struct stat st;
        char *msg;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
            msg = xstrdup("EISDIR: illegal operation on a directory, read");
        } else if (e == EINVAL || e == EFBIG) {
            msg = xstrdup(e == EFBIG ? "File is too large to read." : "Not a regular file.");
        } else {
            msg = pi_fs_error(e, "access", path);
        }
        free(path);
        free(file.ptr);
        return msg;
    }
    free(path);
    const char *d = file.ptr ? file.ptr : "";
    const size_t n = file.len;

    size_t total = 1;                                  /* allLines.length: split("\n") */
    for (size_t i = 0; i < n; i++) if (d[i] == '\n') total++;

    double offset = 0, limit = 0;
    const bool has_offset = pi_number_arg(call, "offset", &offset) && offset != 0;
    const bool has_limit = pi_number_arg(call, "limit", &limit);
    /* clamp before converting: a double past the range of size_t is undefined behaviour, and past
     * the file's end every offset and limit reads the same; a negative limit reads nothing */
    if (offset > (double)total + 1.0) offset = (double)total + 1.0;
    if (limit > (double)total) limit = (double)total;
    if (limit < 0) limit = 0;
    size_t start_line = has_offset && offset > 1 ? (size_t)offset - 1 : 0;
    if (start_line >= total) {
        const char *given = agent_tool_arg_value(call, "offset");
        char *msg = xmalloc(strlen(given) + 96);
        sprintf(msg, "Offset %s is beyond end of file (%zu lines total)", given, total);
        free(file.ptr);
        return msg;
    }
    const size_t sb = pi_line_start(d, n, start_line);
    size_t eb = n, user_limited = 0;
    if (has_limit) {
        const size_t lim = (size_t)limit;
        size_t end_line = start_line + lim < total ? start_line + lim : total;
        user_limited = end_line - start_line;
        if (end_line == start_line) eb = sb;
        else if (end_line == total) eb = n;
        else eb = pi_line_start(d, n, end_line) - 1;
    }
    const char *sel = d + sb;
    const size_t sel_len = eb - sb;
    pi_truncation t = pi_truncate_head(sel, sel_len);

    agent_buf out = {0};
    const size_t start_display = start_line + 1;
    char size_a[32], size_b[32];
    if (t.first_line_exceeds) {
        const char *nl = memchr(sel, '\n', sel_len);
        size_t first = nl ? (size_t)(nl - sel) : sel_len;
        pi_format_size(first, size_a, sizeof(size_a));
        pi_format_size(PI_MAX_BYTES, size_b, sizeof(size_b));
        char *line = xmalloc(strlen(path_arg) + 256);
        sprintf(line, "[Line %zu is %s, exceeds %s limit. Use bash: sed -n '%zup' %s | head -c %d]",
                start_display, size_a, size_b, start_display, path_arg, PI_MAX_BYTES);
        agent_buf_puts(&out, line);
        free(line);
    } else if (t.truncated) {
        const size_t end_display = start_display + t.out_lines - 1;
        char note[256];
        agent_buf_append(&out, sel, t.out_bytes);
        if (t.by_lines) {
            snprintf(note, sizeof(note),
                     "\n\n[Showing lines %zu-%zu of %zu. Use offset=%zu to continue.]",
                     start_display, end_display, total, end_display + 1);
        } else {
            pi_format_size(PI_MAX_BYTES, size_b, sizeof(size_b));
            snprintf(note, sizeof(note),
                     "\n\n[Showing lines %zu-%zu of %zu (%s limit). Use offset=%zu to continue.]",
                     start_display, end_display, total, size_b, end_display + 1);
        }
        agent_buf_puts(&out, note);
    } else if (has_limit && start_line + user_limited < total) {
        char note[160];
        snprintf(note, sizeof(note), "\n\n[%zu more lines in file. Use offset=%zu to continue.]",
                 total - (start_line + user_limited), start_line + user_limited + 1);
        agent_buf_append(&out, sel, sel_len);
        agent_buf_puts(&out, note);
    } else {
        agent_buf_append(&out, sel, sel_len);
    }
    free(file.ptr);
    return agent_buf_take(&out);
}

/* ---- write ---- */

/* mkdir -p for the directory part of path; false (errno set) on failure. */
static bool pi_mkdir_parents(const char *path) {
    char *dir = xstrdup(path);
    char *slash = strrchr(dir, '/');
    if (!slash) { free(dir); return true; }
    if (slash == dir) { free(dir); return true; }
    *slash = '\0';
    for (char *q = dir + 1; ; q++) {
        if (*q == '/' || *q == '\0') {
            char saved = *q;
            *q = '\0';
            if (mkdir(dir, 0777) != 0 && errno != EEXIST) {
                int e = errno;
                free(dir);
                errno = e;
                return false;
            }
            *q = saved;
            if (!saved) break;
        }
    }
    free(dir);
    return true;
}

static char *pi_tool_write(const agent_tool_call *call) {
    const char *path_arg = pi_string_arg(call, "path");
    const char *content = pi_string_arg(call, "content");
    if (!path_arg || !path_arg[0]) return xstrdup("write requires a path string.");
    if (!content) return xstrdup("write requires a content string.");
    char *path = pi_resolve_path(path_arg);
    if (!pi_mkdir_parents(path)) {
        char *msg = pi_fs_error(errno, "mkdir", path);
        free(path);
        return msg;
    }
    const size_t len = strlen(content);
    char err[256];
    if (agent_replace_file(path, content, len, NULL, 0, err, sizeof(err)) != 0) {
        free(path);
        return xstrdup(err);
    }
    free(path);
    /* the training-era text (Pi 0.84.4; 1.1.0 says "Successfully wrote to P"), whose count is
     * content.length: UTF-16 units, so a code point above U+FFFF counts two */
    size_t units = 0;
    for (size_t i = 0; i < len; i++) {
        const unsigned char c = (unsigned char)content[i];
        if ((c & 0xc0) != 0x80) units += c >= 0xf0 ? 2 : 1;
    }
    char *msg = xmalloc(strlen(path_arg) + 64);
    sprintf(msg, "Successfully wrote %zu bytes to %s", units, path_arg);
    return msg;
}

/* ---- edit (edit.js, edit-diff.js) ---- */

typedef struct {
    char *old_text;
    char *new_text;
} pi_edit;

typedef struct {
    pi_edit *v;
    size_t len;
} pi_edits;

static void pi_edits_free(pi_edits *e) {
    for (size_t i = 0; i < e->len; i++) { free(e->v[i].old_text); free(e->v[i].new_text); }
    free(e->v);
    memset(e, 0, sizeof(*e));
}

static void pi_edits_add(pi_edits *e, char *old_text, char *new_text) {
    e->v = xrealloc(e->v, (e->len + 1) * sizeof(e->v[0]));
    e->v[e->len].old_text = old_text;
    e->v[e->len].new_text = new_text;
    e->len++;
}

/* One {"oldText": "...", "newText": "..."} object, with the cursor on its '{'. */
static bool pi_parse_edit_object(const char **p, pi_edit *out) {
    char *old_text = NULL, *new_text = NULL;
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
        if (!strcmp(key, "oldText") || !strcmp(key, "newText")) {
            char *val = NULL;
            if (!json_string(p, &val)) { free(key); goto bad; }
            if (!strcmp(key, "oldText")) { free(old_text); old_text = val; }
            else { free(new_text); new_text = val; }
        } else if (!json_skip_value(p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(p);
        if (**p == ',') { (*p)++; json_ws(p); }
        else if (**p != '}') goto bad;
    }
    if (**p != '}' || !old_text || !new_text) goto bad;
    (*p)++;
    out->old_text = old_text;
    out->new_text = new_text;
    return true;
bad:
    free(old_text);
    free(new_text);
    return false;
}

/* edits as Pi's prepareArguments accepts them: an array, or a single edit object. */
static bool pi_parse_edits_json(const char *json, pi_edits *out) {
    const char *p = json;
    json_ws(&p);
    if (*p == '{') {
        pi_edit e;
        if (!pi_parse_edit_object(&p, &e)) return false;
        pi_edits_add(out, e.old_text, e.new_text);
        json_ws(&p);
        return *p == '\0';
    }
    if (*p != '[') return false;
    p++;
    json_ws(&p);
    while (*p && *p != ']') {
        pi_edit e;
        if (!pi_parse_edit_object(&p, &e)) return false;
        pi_edits_add(out, e.old_text, e.new_text);
        json_ws(&p);
        if (*p == ',') { p++; json_ws(&p); }
        else if (*p != ']') return false;
    }
    if (*p != ']') return false;
    p++;
    json_ws(&p);
    return *p == '\0';
}

static char *pi_normalize_to_lf(const char *s, size_t n, size_t *out_len) {
    char *out = xmalloc(n + 1);
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\r') {
            out[o++] = '\n';
            if (i + 1 < n && s[i + 1] == '\n') i++;
        } else {
            out[o++] = s[i];
        }
    }
    out[o] = '\0';
    if (out_len) *out_len = o;
    return out;
}

static size_t pi_utf8_decode(const unsigned char *s, size_t n, uint32_t *cp) {
    if (s[0] < 0x80) { *cp = s[0]; return 1; }
    if ((s[0] & 0xe0) == 0xc0 && n >= 2 && (s[1] & 0xc0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x1f) << 6) | (s[1] & 0x3f);
        return 2;
    }
    if ((s[0] & 0xf0) == 0xe0 && n >= 3 && (s[1] & 0xc0) == 0x80 && (s[2] & 0xc0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x0f) << 12) | ((uint32_t)(s[1] & 0x3f) << 6) | (s[2] & 0x3f);
        return 3;
    }
    if ((s[0] & 0xf8) == 0xf0 && n >= 4 && (s[1] & 0xc0) == 0x80 && (s[2] & 0xc0) == 0x80 &&
        (s[3] & 0xc0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x07) << 18) | ((uint32_t)(s[1] & 0x3f) << 12) |
              ((uint32_t)(s[2] & 0x3f) << 6) | (s[3] & 0x3f);
        return 4;
    }
    /* an invalid byte is a unit of its own that is neither space nor punctuation to map: copied as is */
    *cp = 0x110000u + s[0];
    return 1;
}

/* JavaScript's trimEnd whitespace, minus what a line cannot contain. */
static bool pi_is_js_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\r' || cp == 0x0b || cp == 0x0c || cp == 0xa0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202f || cp == 0x205f || cp == 0x3000 || cp == 0xfeff;
}

/* normalizeForFuzzyMatch, without NFKC: trailing whitespace per line, smart quotes, Unicode
 * dashes and Unicode spaces to their ASCII forms.  Never changes the number of lines. */
static char *pi_normalize_fuzzy(const char *s, size_t n, size_t *out_len) {
    char *out = xmalloc(n + 1);
    size_t o = 0, i = 0;
    while (i <= n) {
        const char *nl = i < n ? memchr(s + i, '\n', n - i) : NULL;
        size_t line_end = nl ? (size_t)(nl - s) : n;
        /* where the line's trailing whitespace begins */
        size_t keep = i;                 /* a line of only whitespace trims to nothing */
        for (size_t k = i; k < line_end;) {
            uint32_t cp;
            size_t w = pi_utf8_decode((const unsigned char *)s + k, line_end - k, &cp);
            if (!pi_is_js_space(cp)) keep = k + w;
            k += w;
        }
        for (size_t k = i; k < keep;) {
            uint32_t cp;
            size_t w = pi_utf8_decode((const unsigned char *)s + k, keep - k, &cp);
            if (cp >= 0x2018 && cp <= 0x201b) out[o++] = '\'';
            else if (cp >= 0x201c && cp <= 0x201f) out[o++] = '"';
            else if ((cp >= 0x2010 && cp <= 0x2015) || cp == 0x2212) out[o++] = '-';
            else if (cp == 0xa0 || (cp >= 0x2002 && cp <= 0x200a) || cp == 0x202f ||
                     cp == 0x205f || cp == 0x3000) out[o++] = ' ';
            else { memcpy(out + o, s + k, w); o += w; }
            k += w;
        }
        if (!nl) break;
        out[o++] = '\n';
        i = line_end + 1;
    }
    out[o] = '\0';
    if (out_len) *out_len = o;
    return out;
}

typedef struct {
    size_t index;
    size_t length;
    const char *new_text;
    size_t new_len;
    size_t edit_index;
} pi_match;

static int pi_match_cmp(const void *a, const void *b) {
    const pi_match *x = a, *y = b;
    return x->index < y->index ? -1 : x->index > y->index;
}

/* How many times needle occurs in hay, counted left to right without overlap (split().length - 1). */
static size_t pi_count_occurrences(const char *hay, size_t hay_len, const char *needle, size_t needle_len) {
    /* split("") cuts between every character: one fewer piece-boundary than there are characters */
    if (needle_len == 0) return hay_len ? hay_len - 1 : 0;
    size_t count = 0;
    const char *p = hay, *end = hay + hay_len;
    while (p < end) {
        const char *m = agent_memmem(p, (size_t)(end - p), needle, needle_len);
        if (!m) break;
        count++;
        p = m + needle_len;
    }
    return count;
}

static void pi_apply_replacements(agent_buf *out, const char *content, size_t len,
                                  const pi_match *m, size_t count, size_t offset) {
    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        size_t at = m[i].index - offset;
        agent_buf_append(out, content + pos, at - pos);
        agent_buf_append(out, m[i].new_text, m[i].new_len);
        pos = at + m[i].length;
    }
    agent_buf_append(out, content + pos, len - pos);
}

typedef struct { size_t start, end; } pi_span;

/* splitLinesWithEndings: lines keep their "\n"; a last line without one is still a line. */
static pi_span *pi_line_spans(const char *s, size_t n, size_t *count) {
    size_t cap = 16, c = 0, pos = 0;
    pi_span *v = xmalloc(cap * sizeof(v[0]));
    while (pos < n) {
        const char *nl = memchr(s + pos, '\n', n - pos);
        size_t end = nl ? (size_t)(nl - s) + 1 : n;
        if (c == cap) { cap *= 2; v = xrealloc(v, cap * sizeof(v[0])); }
        v[c].start = pos;
        v[c].end = end;
        c++;
        pos = end;
    }
    *count = c;
    return v;
}

/* applyReplacementsPreservingUnchangedLines: the matches were made in the normalized base; the
 * lines they touch are rewritten from it and every other line is copied from the original, so a
 * fuzzy edit does not rewrite the file's smart quotes elsewhere.  Returns false on a line-count
 * mismatch. */
static bool pi_apply_preserving_lines(agent_buf *out, const char *orig, size_t orig_len,
                                      const char *base, size_t base_len,
                                      const pi_match *m, size_t count) {
    size_t n_orig = 0, n_base = 0;
    pi_span *ol = pi_line_spans(orig, orig_len, &n_orig);
    pi_span *bl = pi_line_spans(base, base_len, &n_base);
    bool ok = n_orig == n_base;
    size_t *gs = xmalloc((count + 1) * sizeof(size_t));
    size_t *ge = xmalloc((count + 1) * sizeof(size_t));
    size_t *gfirst = xmalloc((count + 1) * sizeof(size_t));
    size_t *gcount = xmalloc((count + 1) * sizeof(size_t));
    size_t groups = 0;
    for (size_t i = 0; ok && i < count; i++) {
        size_t rs = m[i].index, re = m[i].index + m[i].length, start_line = n_base;
        for (size_t l = 0; l < n_base; l++) {
            if (rs >= bl[l].start && rs < bl[l].end) { start_line = l; break; }
        }
        if (start_line == n_base) { ok = false; break; }
        size_t end_line = start_line;
        while (end_line < n_base && bl[end_line].end < re) end_line++;
        if (end_line >= n_base) { ok = false; break; }
        end_line++;
        if (groups && start_line < ge[groups - 1]) {
            if (end_line > ge[groups - 1]) ge[groups - 1] = end_line;
            gcount[groups - 1]++;
        } else {
            gs[groups] = start_line;
            ge[groups] = end_line;
            gfirst[groups] = i;
            gcount[groups] = 1;
            groups++;
        }
    }
    if (ok) {
        size_t orig_line = 0;
        for (size_t g = 0; g < groups; g++) {
            if (gs[g] > orig_line)
                agent_buf_append(out, orig + ol[orig_line].start,
                                 ol[gs[g] - 1].end - ol[orig_line].start);
            size_t b0 = bl[gs[g]].start, b1 = bl[ge[g] - 1].end;
            pi_apply_replacements(out, base + b0, b1 - b0, m + gfirst[g], gcount[g], b0);
            orig_line = ge[g];
        }
        if (orig_line < n_orig)
            agent_buf_append(out, orig + ol[orig_line].start, orig_len - ol[orig_line].start);
    }
    free(gs); free(ge); free(gfirst); free(gcount);
    free(ol); free(bl);
    return ok;
}

/* One side at a missed edit's departure: the rest of its line in backticks (40 bytes at most, whole UTF-8
 * characters), or what ends it there. */
static void pi_departure_side(const char *s, const char *end, char *out, size_t out_size) {
    if (s >= end) { snprintf(out, out_size, "the end of the text"); return; }
    if (*s == '\n') { snprintf(out, out_size, "the end of the line"); return; }
    size_t n = 0;
    while (s + n < end && s[n] != '\n' && n < 40) n++;
    while (s + n < end && n > 0 && ((unsigned char)s[n] & 0xc0) == 0x80) n--;
    snprintf(out, out_size, "`%.*s`", (int)n, s);
}

/* The byte in s[0..n) that pi_normalize_fuzzy's output offset off came from: the start of the character it lies in,
 * or the line's end (its newline, or n) when off is where the normalised line ends, since trimmed trailing
 * whitespace is no text to point at.  Every mapped character becomes one byte, every other keeps its bytes. */
static size_t pi_fuzzy_raw_offset(const char *s, size_t n, size_t off) {
    size_t o = 0, i = 0;
    for (;;) {
        const char *nl = i < n ? memchr(s + i, '\n', n - i) : NULL;
        const size_t line_end = nl ? (size_t)(nl - s) : n;
        size_t keep = i;
        for (size_t k = i; k < line_end;) {
            uint32_t cp;
            const size_t w = pi_utf8_decode((const unsigned char *)s + k, line_end - k, &cp);
            if (!pi_is_js_space(cp)) keep = k + w;
            k += w;
        }
        for (size_t k = i; k < keep;) {
            uint32_t cp;
            const size_t w = pi_utf8_decode((const unsigned char *)s + k, keep - k, &cp);
            /* the characters pi_normalize_fuzzy maps to one ASCII byte */
            const bool mapped = (cp >= 0x2010 && cp <= 0x201f && cp != 0x2016 && cp != 0x2017) || cp == 0x2212 ||
                                cp == 0xa0 || (cp >= 0x2002 && cp <= 0x200a) || cp == 0x202f || cp == 0x205f ||
                                cp == 0x3000;
            const size_t ow = mapped ? 1 : w;
            if (off < o + ow) return k;
            o += ow;
            k += w;
        }
        if (off <= o || !nl) return line_end;
        o++;                                       /* the newline */
        i = line_end + 1;
    }
}

/* Step 45: where a missed edit's oldText departs from the file, found under Pi's fuzzy normalisation (step 49: the
 * one the edit itself tries, so a curly quote typed straight or dropped trailing spaces are no departure) and told in
 * the file's own bytes.  The longest start of oldText the file holds (a longer start can only occur where a shorter
 * one does, so a binary search finds it), at its first place; appended to msg as a sentence.  Nothing when not even
 * oldText's first character is in the file.  The normalisation keeps the line count, so lines agree. */
static void pi_edit_departure(const char *content, size_t content_len, const char *old, size_t old_len,
                              char *msg, size_t msg_size) {
    size_t fc_len = 0, fo_len = 0;
    char *fc = pi_normalize_fuzzy(content, content_len, &fc_len), *fo = pi_normalize_fuzzy(old, old_len, &fo_len);
    size_t lo = 0, hi = fo_len;
    while (hi - lo > 1) {
        const size_t mid = lo + (hi - lo) / 2;
        if (agent_memmem(fc, fc_len, fo, mid)) lo = mid;
        else hi = mid;
    }
    if (lo == 0 && fo_len && agent_memmem(fc, fc_len, fo, 1)) lo = 1;
    if (lo > 0) {
        const size_t f_at = (size_t)(agent_memmem(fc, fc_len, fo, lo) - fc) + lo;
        const char *at = content + pi_fuzzy_raw_offset(content, content_len, f_at);
        const char *old_at = old + pi_fuzzy_raw_offset(old, old_len, lo);
        int line = 1;
        const char *line_start = content;
        for (const char *c = content; c < at; c++) {
            if (*c == '\n') { line++; line_start = c + 1; }
        }
        char old_side[64], file_side[64];
        pi_departure_side(old_at, old + old_len, old_side, sizeof(old_side));
        pi_departure_side(at, content + content_len, file_side, sizeof(file_side));
        const size_t used = strlen(msg);
        snprintf(msg + used, msg_size - used,
                 " The file matches oldText's start up to line %d, column %d, where oldText has %s and the file has %s.",
                 line, (int)(at - line_start) + 1, old_side, file_side);
    }
    free(fc);
    free(fo);
}

/* applyEditsToNormalizedContent: every edit is matched against the same original text, exactly
 * first, then fuzzily; each must be unique and none may overlap.  Returns the new text, or NULL with
 * a Pi error message in *error. */
static char *pi_apply_edits(const char *content, size_t content_len, const pi_edits *edits,
                            const char *path, size_t *new_len, char **error) {
    const size_t n = edits->len;
    char **old_lf = xmalloc(n * sizeof(char *)), **new_lf = xmalloc(n * sizeof(char *));
    size_t *old_len = xmalloc(n * sizeof(size_t)), *nlen = xmalloc(n * sizeof(size_t));
    pi_match *matches = xmalloc(n * sizeof(pi_match));
    char *result = NULL, *fuzzy_content = NULL;
    size_t fuzzy_len = 0;
    char msg[PATH_MAX + 512];
    msg[0] = '\0';
    for (size_t i = 0; i < n; i++) { old_lf[i] = new_lf[i] = NULL; }
    for (size_t i = 0; i < n; i++) {
        old_lf[i] = pi_normalize_to_lf(edits->v[i].old_text, strlen(edits->v[i].old_text), &old_len[i]);
        new_lf[i] = pi_normalize_to_lf(edits->v[i].new_text, strlen(edits->v[i].new_text), &nlen[i]);
    }
    for (size_t i = 0; i < n; i++) {
        if (old_len[i] == 0) {
            if (n == 1) snprintf(msg, sizeof(msg), "oldText must not be empty in %s.", path);
            else snprintf(msg, sizeof(msg), "edits[%zu].oldText must not be empty in %s.", i, path);
            goto done;
        }
    }
    bool used_fuzzy = false;
    for (size_t i = 0; i < n && !used_fuzzy; i++) {
        if (agent_memmem(content, content_len, old_lf[i], old_len[i])) continue;
        size_t fl = 0, ol = 0;
        char *fc = pi_normalize_fuzzy(content, content_len, &fl);
        char *fo = pi_normalize_fuzzy(old_lf[i], old_len[i], &ol);
        if (agent_memmem(fc, fl, fo, ol)) used_fuzzy = true;
        free(fc);
        free(fo);
    }
    const char *base = content;
    size_t base_len = content_len;
    if (used_fuzzy) {
        fuzzy_content = pi_normalize_fuzzy(content, content_len, &fuzzy_len);
        base = fuzzy_content;
        base_len = fuzzy_len;
    }
    for (size_t i = 0; i < n; i++) {
        /* fuzzyFindText on the base: exact, else both sides normalized (the base already is) */
        const char *needle = old_lf[i];
        size_t needle_len = old_len[i];
        char *fuzzy_needle = NULL;
        const char *at = agent_memmem(base, base_len, needle, needle_len);
        if (!at) {
            size_t ol = 0, fl = 0;
            fuzzy_needle = pi_normalize_fuzzy(old_lf[i], old_len[i], &ol);
            char *fb = pi_normalize_fuzzy(base, base_len, &fl);
            const char *fat = agent_memmem(fb, fl, fuzzy_needle, ol);
            if (fat) {
                /* the base is fuzzy-normalized already, so offsets agree */
                at = base + (fat - fb);
                needle = fuzzy_needle;
                needle_len = ol;
            }
            free(fb);
        }
        if (!at) {
            free(fuzzy_needle);
            if (n == 1)
                snprintf(msg, sizeof(msg), "Could not find the exact text in %s. The old text must match exactly including all whitespace and newlines.", path);
            else
                snprintf(msg, sizeof(msg), "Could not find edits[%zu] in %s. The oldText must match exactly including all whitespace and newlines.", i, path);
            pi_edit_departure(content, content_len, old_lf[i], old_len[i], msg, sizeof(msg));
            goto done;
        }
        size_t fl = 0, ol = 0;
        char *fb = pi_normalize_fuzzy(base, base_len, &fl);
        char *fo = pi_normalize_fuzzy(old_lf[i], old_len[i], &ol);
        size_t occurrences = pi_count_occurrences(fb, fl, fo, ol);
        free(fb);
        free(fo);
        if (occurrences > 1) {
            free(fuzzy_needle);
            if (n == 1)
                snprintf(msg, sizeof(msg), "Found %zu occurrences of the text in %s. The text must be unique. Please provide more context to make it unique.", occurrences, path);
            else
                snprintf(msg, sizeof(msg), "Found %zu occurrences of edits[%zu] in %s. Each oldText must be unique. Please provide more context to make it unique.", occurrences, i, path);
            goto done;
        }
        matches[i] = (pi_match){.index = (size_t)(at - base), .length = needle_len,
                                .new_text = new_lf[i], .new_len = nlen[i], .edit_index = i};
        free(fuzzy_needle);
    }
    qsort(matches, n, sizeof(matches[0]), pi_match_cmp);
    for (size_t i = 1; i < n; i++) {
        if (matches[i - 1].index + matches[i - 1].length > matches[i].index) {
            snprintf(msg, sizeof(msg), "edits[%zu] and edits[%zu] overlap in %s. Merge them into one edit or target disjoint regions.",
                     matches[i - 1].edit_index, matches[i].edit_index, path);
            goto done;
        }
    }
    {
        agent_buf out = {0};
        if (used_fuzzy) {
            if (!pi_apply_preserving_lines(&out, content, content_len, base, base_len, matches, n)) {
                free(out.ptr);
                snprintf(msg, sizeof(msg), "Cannot preserve unchanged lines because the base content has a different line count.");
                goto done;
            }
        } else {
            pi_apply_replacements(&out, base, base_len, matches, n, 0);
        }
        size_t rl = out.len;
        result = agent_buf_take(&out);
        if (rl == content_len && memcmp(result, content, rl) == 0) {
            free(result);
            result = NULL;
            if (n == 1)
                snprintf(msg, sizeof(msg), "No changes made to %s. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected.", path);
            else
                snprintf(msg, sizeof(msg), "No changes made to %s. The replacements produced identical content.", path);
            goto done;
        }
        *new_len = rl;
    }
done:
    for (size_t i = 0; i < n; i++) { free(old_lf[i]); free(new_lf[i]); }
    free(old_lf); free(new_lf); free(old_len); free(nlen); free(matches);
    free(fuzzy_content);
    if (!result) *error = xstrdup(msg);
    return result;
}

static char *pi_tool_edit(const agent_tool_call *call) {
    const char *path_arg = pi_string_arg(call, "path");
    if (!path_arg || !path_arg[0]) return xstrdup("edit requires a path string.");
    const char *legacy_old = agent_tool_arg_value(call, "oldText");
    const char *legacy_new = agent_tool_arg_value(call, "newText");
    if ((legacy_old && !pi_string_arg(call, "oldText")) || (legacy_new && !pi_string_arg(call, "newText")))
        return xstrdup("Edit tool input is invalid. oldText and newText must be strings.");
    pi_edits edits = {0};
    bool parsed = true;
    const char *edits_arg = agent_tool_arg_value(call, "edits");
    /* edits sent as a JSON string were not validated as the call was parsed */
    if (edits_arg) parsed = agent_hermes_valid_json(edits_arg, strlen(edits_arg)) && pi_parse_edits_json(edits_arg, &edits);
    /* the legacy single-edit form: top-level oldText and newText */
    if (parsed && legacy_old && legacy_new) pi_edits_add(&edits, xstrdup(legacy_old), xstrdup(legacy_new));
    if (!parsed) {
        pi_edits_free(&edits);
        return xstrdup("Edit tool input is invalid. edits must be an array of {oldText, newText} string pairs.");
    }
    if (edits.len == 0) {
        pi_edits_free(&edits);
        return xstrdup("Edit tool input is invalid. edits must contain at least one replacement.");
    }
    char *path = pi_resolve_path(path_arg);
    if (access(path, R_OK | W_OK) != 0) {
        char *msg = xmalloc(strlen(path_arg) + 96);
        sprintf(msg, "Could not edit file: %s. Error code: %s.", path_arg, pi_errno_code(errno));
        free(path);
        pi_edits_free(&edits);
        return msg;
    }
    agent_buf file = {0};
    if (!pi_read_all(path, &file)) {
        char *msg = pi_fs_error(errno, "open", path);
        free(path);
        free(file.ptr);
        pi_edits_free(&edits);
        return msg;
    }
    const char *raw = file.ptr ? file.ptr : "";
    size_t raw_len = file.len;
    /* splitBom: the model never includes the invisible BOM in oldText */
    const bool bom = raw_len >= 3 && !memcmp(raw, "\xef\xbb\xbf", 3);
    if (bom) { raw += 3; raw_len -= 3; }
    /* detectLineEnding: whichever of \r\n and \n comes first */
    const char *crlf = agent_memmem(raw, raw_len, "\r\n", 2);
    const char *lf = memchr(raw, '\n', raw_len);
    const bool crlf_file = crlf && lf && crlf < lf;
    size_t lf_len = 0;
    char *normalized = pi_normalize_to_lf(raw, raw_len, &lf_len);
    char *error = NULL;
    size_t new_len = 0;
    const size_t blocks = edits.len;
    char *updated = pi_apply_edits(normalized, lf_len, &edits, path_arg, &new_len, &error);
    free(normalized);
    pi_edits_free(&edits);
    if (!updated) {
        free(path);
        free(file.ptr);
        return error;
    }
    agent_buf out = {0};
    if (bom) agent_buf_append(&out, "\xef\xbb\xbf", 3);
    for (size_t i = 0; i < new_len; i++) {
        if (crlf_file && updated[i] == '\n') agent_buf_append(&out, "\r\n", 2);
        else agent_buf_append(&out, updated + i, 1);
    }
    free(updated);
    char err[256];
    int rc = agent_replace_file(path, out.ptr ? out.ptr : "", out.len, NULL, 0, err, sizeof(err));
    free(out.ptr);
    free(path);
    free(file.ptr);
    if (rc != 0) return xstrdup(err);
    char *msg = xmalloc(strlen(path_arg) + 64);
    sprintf(msg, "Successfully replaced %zu block(s) in %s.", blocks, path_arg);
    return msg;
}

/* ---- bash (bash.js, truncate.js truncateTail) ---- */

/* Size of the output's last line: scan the file backwards for a newline. */
static size_t pi_last_line_bytes(const char *path, size_t total) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t line = 0, pos = total;
    char chunk[4096];
    bool first = true;
    while (pos > 0) {
        size_t take = pos < sizeof(chunk) ? pos : sizeof(chunk);
        if (fseeko(fp, (off_t)(pos - take), SEEK_SET) != 0) break;
        if (fread(chunk, 1, take, fp) != take) break;
        size_t i = take;
        while (i > 0) {
            if (chunk[i - 1] == '\n' && !(first && i == take && pos == total)) {
                fclose(fp);
                return line;
            }
            line++;
            i--;
        }
        first = false;
        pos -= take;
    }
    fclose(fp);
    return line;
}

/* The job's timeout in whole seconds: a positive timeout argument rounded up, at most a day.  Without
 * one it is 0, no deadline, as Pi's bash.js sets no timer. */
static int pi_bash_timeout(const agent_tool_call *call) {
    double t = 0;
    int timeout = 0;
    if (pi_number_arg(call, "timeout", &t) && t > 0)
        timeout = t > 24.0 * 3600.0 ? 24 * 3600 : (int)ceil(t);
    return timeout;
}

static char *pi_tool_bash(agent_worker *w, const agent_tool_call *call) {
    const char *cmd = pi_string_arg(call, "command");
    if (!cmd || !cmd[0]) return xstrdup("bash requires a command string.");
    const int timeout = pi_bash_timeout(call);
    char err[256] = {0};
    agent_bash_job *job = agent_bash_start(w, cmd, timeout, err, sizeof(err));
    if (!job) return xstrdup(err[0] ? err : "failed to start the command");

    bool aborted = false;
    while (agent_bash_is_running(job)) {
        if (worker_should_interrupt(w)) {
            aborted = true;
            agent_bash_signal(job, SIGTERM);
            double s = now_sec();
            while (agent_bash_is_running(job) && now_sec() - s < 1.0) usleep(20000);
            agent_bash_signal(job, SIGKILL);
            s = now_sec();
            while (agent_bash_is_running(job) && now_sec() - s < 3.0) usleep(20000);
            break;
        }
        usleep(20000);
    }

    pthread_mutex_lock(&job->mu);
    const size_t total_bytes = job->bytes;
    const int exit_status = job->exit_status;
    const bool timed_out = job->timed_out;
    const int child_status = job->child_status;
    const bool reaped = job->reaped;
    const int total_lines = agent_bash_display_lines(job);
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s", job->path);
    pthread_mutex_unlock(&job->mu);

    /* the last PI_MAX_BYTES + 1 bytes are enough for a tail of PI_MAX_BYTES */
    char *text = NULL;
    size_t text_len = 0;
    const bool truncated = (size_t)total_lines > PI_MAX_LINES || total_bytes > PI_MAX_BYTES;
    bool by_lines = true, last_partial = false;
    size_t out_lines = 0;
    if (total_bytes == 0) {
        text = xstrdup("");
    } else {
        size_t take = total_bytes < (size_t)PI_MAX_BYTES + 1 ? total_bytes : (size_t)PI_MAX_BYTES + 1;
        char *buf = xmalloc(take + 1);
        FILE *fp = fopen(path, "rb");
        size_t got = 0;
        if (fp && fseeko(fp, (off_t)(total_bytes - take), SEEK_SET) == 0) got = fread(buf, 1, take, fp);
        if (fp) fclose(fp);
        buf[got] = '\0';
        if (!truncated) {
            text = buf;
            text_len = got;
        } else {
            /* truncateTail: whole lines from the end, up to PI_MAX_LINES and PI_MAX_BYTES */
            const bool whole_file = got == total_bytes;
            size_t end = got;
            if (end && buf[end - 1] == '\n') end--;           /* a final newline starts no line */
            size_t kept_start = end, bytes = 0, line_end = end;
            for (;;) {
                size_t ls = line_end;
                while (ls > 0 && buf[ls - 1] != '\n') ls--;
                const bool partial_first = ls == 0 && !whole_file;   /* cut by the read, length unknown */
                const size_t lb = (line_end - ls) + (out_lines > 0 ? 1 : 0);
                if (partial_first || bytes + lb > PI_MAX_BYTES) {
                    by_lines = false;
                    if (out_lines == 0) {
                        /* even the last line alone is too long: its final PI_MAX_BYTES bytes */
                        size_t from = end > (size_t)PI_MAX_BYTES ? end - PI_MAX_BYTES : 0;
                        while (from < end && ((unsigned char)buf[from] & 0xc0) == 0x80) from++;
                        kept_start = from;
                        out_lines = 1;
                        last_partial = true;
                    }
                    break;
                }
                bytes += lb;
                out_lines++;
                kept_start = ls;
                if (out_lines >= PI_MAX_LINES || ls == 0) break;
                line_end = ls - 1;                                 /* step over the newline */
            }
            text_len = end - kept_start;
            text = xmalloc(text_len + 1);
            memcpy(text, buf + kept_start, text_len);
            text[text_len] = '\0';
            free(buf);
        }
    }
    (void)text_len;

    agent_buf out = {0};
    const bool no_exit = reaped && WIFSIGNALED(child_status) && !timed_out && !aborted;
    const bool failed_status = aborted || timed_out || no_exit || exit_status != 0;
    const char *body = text[0] ? text : (failed_status && (aborted || timed_out) ? "" : "(no output)");
    agent_buf_puts(&out, body);
    if (truncated && text[0]) {
        char size_b[32], note[PATH_MAX + 200];
        pi_format_size(PI_MAX_BYTES, size_b, sizeof(size_b));
        if (last_partial) {
            char size_a[32], size_l[32];
            pi_format_size(strlen(text), size_a, sizeof(size_a));
            pi_format_size(pi_last_line_bytes(path, total_bytes), size_l, sizeof(size_l));
            snprintf(note, sizeof(note), "\n\n[Showing last %s of line %d (line is %s). Full output: %s]",
                     size_a, total_lines, size_l, path);
        } else if (by_lines) {
            snprintf(note, sizeof(note), "\n\n[Showing lines %zu-%d of %d. Full output: %s]",
                     (size_t)total_lines - out_lines + 1, total_lines, total_lines, path);
        } else {
            snprintf(note, sizeof(note), "\n\n[Showing lines %zu-%d of %d (%s limit). Full output: %s]",
                     (size_t)total_lines - out_lines + 1, total_lines, total_lines, size_b, path);
        }
        agent_buf_puts(&out, note);
    }
    if (aborted || timed_out || no_exit || exit_status != 0) {
        char status[96];
        if (aborted) snprintf(status, sizeof(status), "Command aborted");
        else if (timed_out) snprintf(status, sizeof(status), "Command timed out after %d seconds", timeout);
        else if (no_exit) snprintf(status, sizeof(status), "Command terminated without an exit code");
        else snprintf(status, sizeof(status), "Command exited with code %d", exit_status);
        if (out.len) agent_buf_puts(&out, "\n\n");
        agent_buf_puts(&out, status);
    }
    free(text);
    char *result = agent_buf_take(&out);
    /* show the user what the command printed, as the agent's own bash does */
    if (result[0]) {
        size_t shown = strlen(result) > 8192 ? 8192 : strlen(result);
        char *safe = agent_terminal_safe_text(result, shown);
        agent_publish(w, "\x1b[90m", 5);
        agent_publish(w, safe, strlen(safe));
        if (safe[0] && safe[strlen(safe) - 1] != '\n') agent_publish(w, "\n", 1);
        agent_publish(w, "\x1b[0m", 4);
        free(safe);
    }
    agent_bash_remove_job(w, job);
    return result;
}
