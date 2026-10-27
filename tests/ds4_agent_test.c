#define DS4_AGENT_TEST
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"
#include "mellum_tool_text.h"
#include <sys/resource.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/xattr.h>
#endif

static const char *test_output_dir;

static void test_fixture(const char *name, const char *data, size_t len) {
    if (!test_output_dir) return;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", test_output_dir, name);
    FILE *fp = fopen(path, "wb");
    AGENT_TEST_ASSERT(fp != NULL);
    if (!fp) return;
    AGENT_TEST_ASSERT(fwrite(data, 1, len, fp) == len);
    AGENT_TEST_ASSERT(fclose(fp) == 0);
}

static void test_tool_arg(agent_tool_call *call, const char *name, const char *value) {
    agent_tool_call_add_arg(call, name, value, strlen(value), true, "</arg_value>");
}

static int test_write_file(const char *path, const char *data, size_t len,
                           char *err, size_t errlen) {
    return agent_replace_file(path, data, len, NULL, 0, err, errlen);
}

static void test_atomic_file_tools(void) {
    char dir[] = "/tmp/ds4-agent-files-XXXXXX";
    AGENT_TEST_ASSERT(mkdtemp(dir) != NULL);
    char path[PATH_MAX], linkpath[PATH_MAX], err[256];
    snprintf(path, sizeof(path), "%s/file", dir);
    snprintf(linkpath, sizeof(linkpath), "%s/link", dir);
    char original[4096];
    memset(original, 'x', sizeof(original));
    AGENT_TEST_ASSERT(test_write_file(path, original, sizeof(original), err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(chmod(path, 0751) == 0);
#ifdef __APPLE__
    AGENT_TEST_ASSERT(setxattr(path, "com.ds4.agent-test", "keep", 4, 0, 0) == 0);
#elif defined(__linux__)
    AGENT_TEST_ASSERT(setxattr(path, "user.ds4-agent-test", "keep", 4, 0) == 0);
#endif

    pid_t child = fork();
    AGENT_TEST_ASSERT(child >= 0);
    if (child == 0) {
        signal(SIGXFSZ, SIG_IGN);
        struct rlimit limit = {64, 64};
        if (setrlimit(RLIMIT_FSIZE, &limit)) _exit(2);
        int rc = agent_replace_file(path, original, sizeof(original),
                                     original, sizeof(original), err, sizeof(err));
        _exit(rc == -1 ? 0 : 3);
    }
    int status = 0;
    if (child > 0) waitpid(child, &status, 0);
    AGENT_TEST_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    char *data = NULL;
    size_t len = 0;
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(len == sizeof(original) && !memcmp(data, original, len));
    free(data);
    AGENT_TEST_ASSERT(agent_replace_file(path, "bad", 3, "stale", 5, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "changed") != NULL);

    AGENT_TEST_ASSERT(symlink("file", linkpath) == 0);
    AGENT_TEST_ASSERT(test_write_file(linkpath, "new", 3, err, sizeof(err)) == 0);
    struct stat st;
    AGENT_TEST_ASSERT(lstat(linkpath, &st) == 0 && S_ISLNK(st.st_mode));
    AGENT_TEST_ASSERT(stat(path, &st) == 0 && (st.st_mode & 0777) == 0751);
    AGENT_TEST_ASSERT(st.st_uid == getuid());
#ifdef __APPLE__
    char attribute[16];
    AGENT_TEST_ASSERT(getxattr(path, "com.ds4.agent-test", attribute, sizeof(attribute), 0, 0) == 4);
    AGENT_TEST_ASSERT(!memcmp(attribute, "keep", 4));
#elif defined(__linux__)
    char attribute[16];
    AGENT_TEST_ASSERT(getxattr(path, "user.ds4-agent-test", attribute, sizeof(attribute)) == 4);
    AGENT_TEST_ASSERT(!memcmp(attribute, "keep", 4));
#endif
    AGENT_TEST_ASSERT(agent_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(len == 3 && !memcmp(data, "new", 3));
    free(data);
    unlink(linkpath);
    AGENT_TEST_ASSERT(link(path, linkpath) == 0);
    AGENT_TEST_ASSERT(test_write_file(path, "bad", 3, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "hard-linked") != NULL);
    unlink(linkpath);
    unlink(path);
    /* Failed replacements must not leave temporary files behind. */
    AGENT_TEST_ASSERT(rmdir(dir) == 0);

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = true;
    AGENT_TEST_ASSERT(agent_edit_find_old_span("literal [upto] text", 19,
                         "[upto]", false, &match, &match_len, &anchored, err, sizeof(err)));
    AGENT_TEST_ASSERT(!anchored && match_len == 6 && !memcmp(match, "[upto]", 6));
}

static void test_streaming_file_tools(void) {
    char path[] = "/tmp/ds4-agent-read-XXXXXX";
    int fd = mkstemp(path);
    AGENT_TEST_ASSERT(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    /* Larger than the old whole-file cap, with matches at both ends. */
    fputs("needle first\r\n", fp);
    for (int i = 0; i < 1024 * 1024; i++) fputs("0123456789abcdef\n", fp);
    fputs("needle last\r", fp);
    fclose(fp);
    agent_worker w = {0};
    char *text = agent_read_range(&w, path, 1, 1, false, false, true);
    AGENT_TEST_ASSERT(strstr(text, "needle first") && w.more_valid);
    AGENT_TEST_ASSERT(w.more_next_line == 2 && w.more_byte_offset == 14);
    free(text);
    agent_tool_call more = {0};
    test_tool_arg(&more, "count", "1");
    text = agent_tool_more(&w, &more);
    AGENT_TEST_ASSERT(strstr(text, "2 0123456789abcdef"));
    free(text);
    agent_tool_call_free(&more);

    agent_tool_call call = {0};
    test_tool_arg(&call, "path", path);
    test_tool_arg(&call, "query", "needle");
    char linkpath[PATH_MAX];
    snprintf(linkpath, sizeof(linkpath), "%s-link", path);
    AGENT_TEST_ASSERT(symlink(path, linkpath) == 0);
    agent_tool_call linked = {0};
    test_tool_arg(&linked, "path", linkpath);
    test_tool_arg(&linked, "query", "needle");
    text = agent_tool_search(&w, &linked);
    AGENT_TEST_ASSERT(strstr(text, "2 matches") && strstr(text, "needle last"));
    free(text);
    agent_tool_call_free(&linked);
    unlink(linkpath);
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "2 matches") && strstr(text, "needle last"));
    free(text);
    test_tool_arg(&call, "mode", "regexp");
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:"));
    free(text);
    agent_tool_call_free(&call);

    fp = fopen(path, "wb");
    for (int i = 0; i < 256 * 1024; i++) fputc('x', fp);
    fclose(fp);
    size_t total = 0;
    text = agent_read_range(&w, path, 1, 1, false, true, true);
    do {
        size_t n = strspn(text, "x");
        total += n;
        AGENT_TEST_ASSERT(n > 0 && n < AGENT_TOOL_MAX_BYTES);
        if (w.more_valid) AGENT_TEST_ASSERT(strstr(text, "Read truncated") != NULL);
        free(text);
        if (!w.more_valid) break;
        text = agent_tool_more(&w, &more);
    } while (total < 512 * 1024);
    AGENT_TEST_ASSERT(total == 256 * 1024 && !w.more_valid);
    text = agent_read_range(&w, path, 1, INT_MAX, true, true, true);
    AGENT_TEST_ASSERT(strstr(text, "Tool error: whole read") && !w.more_valid);
    free(text);
    fp = fopen(path, "wb");
    for (int i = 0; i < 256 * 1024; i++) fputc(0x80, fp);
    fclose(fp);
    text = agent_read_range(&w, path, 1, 1, false, true, true);
    AGENT_TEST_ASSERT(strlen(text) < AGENT_TOOL_MAX_BYTES && w.more_valid);
    free(text);
    unlink(path);
    AGENT_TEST_ASSERT(mkfifo(path, 0600) == 0);
    double started = now_sec();
    text = agent_read_range(&w, path, 1, 1, false, false, true);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:") && now_sec() - started < 0.5);
    free(text);
    unlink(path);
    test_tool_arg(&call, "path", path);
    test_tool_arg(&call, "query", "needle");
    text = agent_tool_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:"));
    free(text);
    agent_tool_call_free(&call);

    agent_buf b = {0};
    char *large = xmalloc(200000);
    memset(large, 'q', 200000);
    agent_buf_append(&b, large, 200000);
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(strlen(text) == 200000);
    free(text);
    b.limit = 100;
    agent_buf_append(&b, large, 200000);
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(strstr(text, "Output truncated") != NULL);
    free(large);
    free(text);
    b.limit = 3;
    agent_buf_puts(&b, "a\xe4\xb8\xad" "b");
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(!strncmp(text, "a\n[Output truncated", 19));
    free(text);
}

static void test_shell_spawn(void) {
    agent_worker w = {0};
    pthread_mutex_init(&w.mu, NULL);
    w.wake_fd[0] = w.wake_fd[1] = -1;
    char err[256], cwd[PATH_MAX];
    AGENT_TEST_ASSERT(getcwd(cwd, sizeof(cwd)) != NULL);
    setenv("DS4_TEST_SHELL_ENV", "inherited", 1);
    for (int closed_stdio = 0; closed_stdio <= 1; closed_stdio++) {
        int saved[3];
        if (closed_stdio) {
            fflush(NULL);
            for (int i = 0; i < 3; i++) {
                saved[i] = fcntl(i, F_DUPFD_CLOEXEC, 3);
                AGENT_TEST_ASSERT(saved[i] >= 0);
                if (saved[i] < 0) exit(1);
            }
            for (int i = 0; i < 3; i++) close(i);
        }
        agent_bash_job *job = agent_bash_start(&w,
            "read line || printf 'stdin-eof\\n'; pwd; "
            "printf 'env=%s\\n' \"$DS4_TEST_SHELL_ENV\"; "
            "printf 'stderr-captured\\n' >&2; sleep 0.1; exit 23",
            5, err, sizeof(err));
        bool group_ok = job && getpgid(job->pid) == job->pid;
        double start = now_sec();
        while (job && agent_bash_is_running(job) && now_sec() - start < 6)
            usleep(10000);
        bool finished = false;
        char *obs = job ? agent_bash_observation(job, true, &finished) : NULL;
        if (job) {
            unlink(job->path);
            agent_bash_remove_job(&w, job);
        }
        if (closed_stdio) {
            for (int i = 0; i < 3; i++) {
                if (dup2(saved[i], i) < 0) _exit(1);
                close(saved[i]);
            }
        }
        AGENT_TEST_ASSERT(group_ok && finished && obs);
        if (obs) {
            AGENT_TEST_ASSERT(strstr(obs, "exit_status=23"));
            AGENT_TEST_ASSERT(strstr(obs, "stdin-eof"));
            AGENT_TEST_ASSERT(strstr(obs, cwd));
            AGENT_TEST_ASSERT(strstr(obs, "env=inherited"));
            AGENT_TEST_ASSERT(strstr(obs, "stderr-captured"));
        }
        free(obs);
    }
    unsetenv("DS4_TEST_SHELL_ENV");
    free(w.out);
    pthread_mutex_destroy(&w.mu);
}

static void test_background_jobs(void) {
    agent_worker w = {0};
    pthread_mutex_init(&w.mu, NULL);
    w.wake_fd[0] = w.wake_fd[1] = -1;
    char err[256], marker[] = "/tmp/ds4-agent-deadline-XXXXXX";
    int fd = mkstemp(marker);
    close(fd);
    unlink(marker);
    char cmd[PATH_MAX + 128];
    snprintf(cmd, sizeof(cmd), "sleep 2; printf late > %s", marker);
    agent_bash_job *job = agent_bash_start(&w, cmd, 1, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    /* Deliberately no status polling: this stands in for model generation. */
    usleep(2400000);
    AGENT_TEST_ASSERT(access(marker, F_OK) != 0);
    bool finished = false;
    char *obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "timed_out=1"));
    free(obs);
    unlink(job->path);
    agent_bash_remove_job(&w, job);

    job = agent_bash_start(&w, "(sleep 0.1; printf descendant-output) &", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    usleep(350000);
    obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "descendant-output"));
    free(obs);
    unlink(job->path);
    agent_bash_remove_job(&w, job);

    job = agent_bash_start(&w, "head -c 2097152 /dev/zero | tr '\\000' x", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    usleep(900000);
    obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "exit_status=0"));
    AGENT_TEST_ASSERT(job->bytes == 2097152);
    free(obs);
    obs = agent_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(strlen(obs) < AGENT_BASH_TAIL_BYTES + 2048);
    free(obs);
    unlink(job->path);
    agent_bash_remove_job(&w, job);

    job = agent_bash_start(&w, "sleep 0.3; printf completed", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    char output_path[PATH_MAX];
    snprintf(output_path, sizeof(output_path), "%s", job->path);
    agent_tool_call call = {.name = xstrdup("bash_status")};
    char id[32];
    snprintf(id, sizeof(id), "%d", job->id);
    test_tool_arg(&call, "job", id);
    test_tool_arg(&call, "refresh_sec", "1");
    double start = now_sec();
    obs = agent_execute_tool_call(&w, &call);
    AGENT_TEST_ASSERT(now_sec() - start >= 0.2);
    AGENT_TEST_ASSERT(strstr(obs, "status=done") && strstr(obs, "completed"));
    AGENT_TEST_ASSERT(w.bash_jobs == NULL);
    free(obs);
    agent_tool_call_free(&call);
    unlink(output_path);

    /* Two monitors must make progress together, and stopping one must neither
     * block shutdown nor kill the other job's process group. */
    job = agent_bash_start(&w, "sleep 30", 60, err, sizeof(err));
    agent_bash_job *other = agent_bash_start(&w, "sleep 0.2; printf independent", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job && other);
    if (!job || !other) goto done;
    start = now_sec();
    agent_bash_signal(job, SIGKILL);
    unlink(job->path);
    agent_bash_remove_job(&w, job);
    AGENT_TEST_ASSERT(now_sec() - start < 2);
    while (agent_bash_is_running(other) && now_sec() - start < 3) usleep(10000);
    obs = agent_bash_observation(other, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "exit_status=0") && strstr(obs, "independent"));
    free(obs);
    unlink(other->path);
    agent_bash_remove_job(&w, other);
done:
    agent_bash_jobs_free(&w);
    unlink(marker);
    free(w.out);
    pthread_mutex_destroy(&w.mu);
}

static void test_completion(const char *text, linenoiseCompletions *completions) {
    (void)text;
    linenoiseAddCompletion(completions, "example");
}

static void test_fragmented_terminal_input(void) {
    int input[2];
    AGENT_TEST_ASSERT(pipe(input) == 0);
    fcntl(input[0], F_SETFL, O_NONBLOCK);
    FILE *sink = tmpfile();
    AGENT_TEST_ASSERT(sink != NULL);
    if (!sink) { close(input[0]); close(input[1]); return; }
    setenv("LINENOISE_ASSUME_TTY", "1", 1);
    struct linenoiseState l = {0};
    char buffer[1024] = "";
    l.ifd = input[0]; l.ofd = fileno(sink);
    l.buf = buffer; l.buflen = sizeof(buffer) - 1;
    l.cols = 80; l.prompt = "";
    const char *samples[] = {"\xc3\xa9", "\xe4\xb8\xad", "\xf0\x9f\x98\x80"};
    for (size_t s = 0; s < sizeof(samples)/sizeof(samples[0]); s++) {
        const char *sample = samples[s];
        for (size_t split = 1; split < strlen(sample); split++) {
            linenoiseEditClear(&l);
            for (size_t i = 0; i < split; i++)
                AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, sample[i]) == linenoiseEditMore);
            AGENT_TEST_ASSERT(l.len == 0);
            AGENT_TEST_ASSERT(linenoiseEditFeed(&l) == linenoiseEditMore);
            AGENT_TEST_ASSERT(l.len == 0);
            for (size_t i = split; i < strlen(sample); i++)
                AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, sample[i]) == linenoiseEditMore);
            AGENT_TEST_ASSERT(!strcmp(l.buf, sample));
        }
    }
    linenoiseEditClear(&l);
    const char sequence[] = "ab\x1b[DZ\x1b[3~\x1b[H!";
    for (size_t i = 0; i < sizeof(sequence) - 1; i++) {
        AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, sequence[i]) == linenoiseEditMore);
        AGENT_TEST_ASSERT(linenoiseEditFeed(&l) == linenoiseEditMore);
    }
    AGENT_TEST_ASSERT(!strcmp(l.buf, "!aZ"));
    linenoiseEditClear(&l);
    const char paste[] = "\x1b[200~one\r\n\xe4\xb8\xad\nthree\x1b[201~";
    for (size_t i = 0; i < sizeof(paste) - 1; i++) {
        AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, paste[i]) == linenoiseEditMore);
        AGENT_TEST_ASSERT(linenoiseEditFeed(&l) == linenoiseEditMore);
        if (i < sizeof(paste) - 2) AGENT_TEST_ASSERT(l.len == 0);
    }
    AGENT_TEST_ASSERT(!strcmp(l.buf, "one\n\xe4\xb8\xad\nthree"));
    linenoiseEditClear(&l);
    linenoiseEditFeedByte(&l, '\xe4');
    linenoiseEditFeedByte(&l, 'X');
    AGENT_TEST_ASSERT(!strcmp(l.buf, "\xef\xbf\xbdX"));
    linenoiseEditClear(&l);
    const char invalid_paste[] = "\x1b[200~bad\xe4\x1b[201~";
    for (size_t i = 0; i < sizeof(invalid_paste) - 1; i++)
        AGENT_TEST_ASSERT(linenoiseEditFeedByte(&l, invalid_paste[i]) == linenoiseEditMore);
    AGENT_TEST_ASSERT(l.len == 0 && !l.paste_active);
    linenoiseSetCompletionCallback(test_completion);
    linenoiseEditFeedByte(&l, 'e');
    linenoiseEditFeedByte(&l, '\t');
    AGENT_TEST_ASSERT(l.in_completion);
    linenoiseEditFeedByte(&l, '\x1b');
    AGENT_TEST_ASSERT(!l.in_completion && !strcmp(l.buf, "e"));
    linenoiseEditFeedByte(&l, '[');
    linenoiseEditFeedByte(&l, 'D');
    AGENT_TEST_ASSERT(l.pos == 0);
    linenoiseEditClear(&l);
    linenoiseEditFeedByte(&l, '\t');
    const char unicode[] = "\xe4\xb8\xad";
    for (size_t i = 0; i < sizeof(unicode) - 1; i++) linenoiseEditFeedByte(&l, unicode[i]);
    AGENT_TEST_ASSERT(!l.in_completion && !strcmp(l.buf, "example\xe4\xb8\xad"));
    linenoiseEditClear(&l);
    l.buflen = 3;
    linenoiseEditFeedByte(&l, 'e');
    linenoiseEditFeedByte(&l, '\t');
    linenoiseEditFeedByte(&l, ' ');
    AGENT_TEST_ASSERT(!l.in_completion && !strcmp(l.buf, "e") && l.len == 1);
    l.buflen = sizeof(buffer) - 1;
    linenoiseSetCompletionCallback(NULL);
    free(l.queued_input);
    free(l.paste_buf);
    close(input[0]); close(input[1]); fclose(sink);
    unsetenv("LINENOISE_ASSUME_TTY");
}

static void test_shell_terminal_controls(void) {
    const char malicious[] = "before\x1b[2Jafter\x1b[H!\x1b]52;c;secret\a"
                             "\x1bPdata\x1b\\\x1b[31mred\x1b[0m\b\n";
    char *safe = agent_terminal_safe_text(malicious, sizeof(malicious) - 1);
    AGENT_TEST_ASSERT(!strcmp(safe, "beforeafter!\x1b[31mred\x1b[0m\\x08\n"));
    free(safe);
    const char c1[] = "\xe4\xb8\xad\xc2\x9b" "2J\x9b" "2J";
    safe = agent_terminal_safe_text(c1, sizeof(c1) - 1);
    AGENT_TEST_ASSERT(!strcmp(safe, "\xe4\xb8\xad\\xc2\\x9b2J\\x9b2J"));
    free(safe);
    for (size_t i = 0; i < sizeof(malicious); i++) {
        safe = agent_terminal_safe_text(malicious, i);
        AGENT_TEST_ASSERT(!strstr(safe, "\x1b[2J") && !strstr(safe, "\x1b]52"));
        free(safe);
    }
}

static void test_markdown_literals(void) {
    const char *input[] = {"Use *.c files.", "The literal is \\*.", "An unmatched `tick",
                          "**bold** and *italic* and `code`.", "``a ` b``", "*unclosed",
                          "* list item\n", "trailing \\", "**unclosed", "`a``", "\\`literal\\`",
                          "> **Hint:** Check `errno`.\n"};
    const char *expected[] = {"Use *.c files.", "The literal is *.", "An unmatched `tick",
                             "bold and italic and code.", "a ` b", "*unclosed",
                             "* list item\n", "trailing \\", "**unclosed", "`a``", "`literal`",
                             "> Hint: Check errno.\n"};
    for (size_t i = 0; i < sizeof(input)/sizeof(input[0]); i++) {
        agent_tail_capture capture = {.cap = 16384};
        agent_token_renderer r = {.capture = &capture, .format_markdown = true};
        for (size_t j = 0; j < strlen(input[i]); j++) renderer_markdown_feed(&r, input[i][j]);
        renderer_markdown_finish(&r);
        renderer_flush_utf8(&r);
        size_t len;
        char *out = agent_tail_capture_take(&capture, &len);
        AGENT_TEST_ASSERT(!strcmp(out, expected[i]));
        if (strcmp(out, expected[i])) fprintf(stderr, "markdown: %s => %s\n", input[i], out);
        free(out);
    }
    agent_tail_capture capture = {.cap = 20000};
    agent_token_renderer r = {.capture = &capture, .format_markdown = true};
    renderer_markdown_feed(&r, '*');
    for (int i = 0; i < 8192; i++) renderer_markdown_feed(&r, 'x');
    renderer_markdown_finish(&r);
    size_t len;
    char *out = agent_tail_capture_take(&capture, &len);
    AGENT_TEST_ASSERT(len == 8193 && out[0] == '*');
    free(out);
}

static char *test_hint_capture(const char *text, size_t split, bool markdown,
                               bool color, agent_tool_syntax syntax, int *calls) {
    agent_tail_capture capture = {.cap = 32768};
    agent_token_renderer renderer = {.capture = &capture, .format_thinking = true,
        .format_markdown = markdown, .use_color = color, .last_output_newline = true};
    agent_dsml_parser parser = {.syntax = syntax, .state = AGENT_DSML_SEARCH};
    agent_stream_renderer stream = {.renderer = &renderer, .parser = &parser, .syntax = syntax};
    size_t n = strlen(text);
    if (split <= n) {
        agent_stream_text(&stream, text, split, false);
        /* A prompt redraw resets terminal colors between generated fragments. */
        if (color && renderer.wrote_visible_output) renderer_write(&renderer, "\x1b[0m", 4);
        agent_stream_text(&stream, text + split, n - split, false);
    } else {
        for (size_t i = 0; i < n; i++) {
            agent_stream_text(&stream, text + i, 1, false);
            if (color && renderer.wrote_visible_output) renderer_write(&renderer, "\x1b[0m", 4);
        }
    }
    agent_stream_text(&stream, NULL, 0, true);
    renderer_finish(&renderer);
    AGENT_TEST_ASSERT(!renderer.md_hint && !renderer.md_hint_prefix_len && !renderer.color_open);
    if (calls) {
        *calls = (int)parser.calls.len;
        AGENT_TEST_ASSERT(parser.calls.len == 1);
        if (parser.calls.len == 1) {
            AGENT_TEST_ASSERT(!strcmp(parser.calls.v[0].name, "bash"));
            AGENT_TEST_ASSERT(parser.calls.v[0].argc == 1);
            if (parser.calls.v[0].argc == 1)
                AGENT_TEST_ASSERT(!strcmp(parser.calls.v[0].args[0].value, "printf HINT_OK"));
        }
    } else AGENT_TEST_ASSERT(parser.calls.len == 0);
    agent_dsml_parser_free(&parser);
    return agent_tail_capture_take(&capture, NULL);
}

static void test_hint_rendering(void) {
    const char *badge = "\x1b[1;97;48;5;23m Hint \x1b[0m";
    const char *sample =
        "The error path is fixed.\n\n"
        "> **Hint:** Save `errno` before **cleanup**; calls can overwrite it.\n"
        "> Keep the original error for reporting.\n"
        "> Unicode stays intact: caf\xc3\xa9, \xe4\xb8\xad.\n"
        "Normal prose resumes here.\n\n"
        "```text\n> **Hint:** This is literal code.\n```\n"
        "Another normal line.\n";
    for (size_t split = 0; split <= strlen(sample) + 1; split++) {
        char *out = test_hint_capture(sample, split, true, true, AGENT_TOOL_SYNTAX_DSML, NULL);
        const char *label = strstr(out, badge);
        AGENT_TEST_ASSERT(label && !strstr(label + strlen(badge), badge));
        AGENT_TEST_ASSERT(strstr(out, "\x1b[1mcleanup") || split > strlen(sample));
        if (split == strlen(sample)) test_fixture("hints.ansi", out, strlen(out));
        if (split > strlen(sample)) test_fixture("hints-fragmented.ansi", out, strlen(out));
        free(out);
    }
    const char *literal[] = {"> quoted text", "> **Hinting:** not a hint",
        "Inline > **Hint:** not an aside", "`> **Hint:** literal`",
        "\\> **Hint:** escaped", "<think>> **Hint:** hidden reasoning</think>Normal."};
    for (size_t i = 0; i < sizeof(literal) / sizeof(literal[0]); i++) {
        char *out = test_hint_capture(literal[i], strlen(literal[i]), true, true,
                                      AGENT_TOOL_SYNTAX_DSML, NULL);
        AGENT_TEST_ASSERT(!strstr(out, badge));
        free(out);
    }
    const char marker[] = "> **Hint:**";
    for (size_t i = 0; i < sizeof(marker) - 1; i++) {
        char partial[sizeof(marker)];
        memcpy(partial, marker, i);
        partial[i] = 0;
        char *out = test_hint_capture(partial, i, true, true, AGENT_TOOL_SYNTAX_GLM, NULL);
        AGENT_TEST_ASSERT(!strstr(out, badge) && !strncmp(out, partial, i));
        free(out);
    }
    char *out = test_hint_capture(sample, strlen(sample), false, false, AGENT_TOOL_SYNTAX_DSML, NULL);
    AGENT_TEST_ASSERT(!strncmp(out, sample, strlen(sample)) && !strchr(out, '\x1b'));
    free(out);
    out = test_hint_capture("> **Hint:** Check `errno`.\n", 0, true, false, AGENT_TOOL_SYNTAX_DSML, NULL);
    AGENT_TEST_ASSERT(!strcmp(out, "> Hint: Check errno.\n\n"));
    free(out);
    const char *tool[] = {
        "> **Hint:** Keep shell checks reproducible.\n"
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">"
        "<｜DSML｜parameter name=\"command\" string=\"true\">printf HINT_OK"
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
        "> **Hint:** Keep shell checks reproducible.\n"
        "<tool_call>bash<arg_key>command</arg_key><arg_value>printf HINT_OK</arg_value></tool_call>"
    };
    for (int glm = 0; glm < 2; glm++) {
        for (size_t split = 0; split <= strlen(tool[glm]); split++) {
            int calls = 0;
            out = test_hint_capture(tool[glm], split, true, true,
                glm ? AGENT_TOOL_SYNTAX_GLM : AGENT_TOOL_SYNTAX_DSML, &calls);
            AGENT_TEST_ASSERT(calls == 1 && strstr(out, badge));
            if (split == strlen(tool[glm]))
                test_fixture(glm ? "hints-glm-tool.ansi" : "hints-dsml-tool.ansi", out, strlen(out));
            free(out);
        }
    }
}

static void test_unicode_output_and_footer(void) {
    agent_editor ed = {0};
    ed.edit.cols = 80;
    char ascii[78];
    memset(ascii, 'a', sizeof(ascii));
    editor_note_output(&ed, ascii, sizeof(ascii));
    editor_note_output(&ed, "\xe4", 1);
    AGENT_TEST_ASSERT(ed.output_col == 78 && ed.output_utf8_len == 1);
    editor_note_output(&ed, "\xb8\xad", 2);
    AGENT_TEST_ASSERT(ed.output_col == 0 && ed.output_pending_wrap);
    editor_note_output(&ed, "\xcc\x81", 2);
    AGENT_TEST_ASSERT(ed.output_col == 0 && ed.output_pending_wrap);
    editor_note_output(&ed, "x", 1);
    AGENT_TEST_ASSERT(ed.output_col == 1 && !ed.output_pending_wrap);
    editor_note_output(&ed, "\r", 1);
    AGENT_TEST_ASSERT(ed.output_col == 0 && !ed.output_pending_wrap);
    const char family[] = "\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x92\xbb";
    for (size_t i = 0; i < sizeof(family) - 1; i++) editor_note_output(&ed, family + i, 1);
    AGENT_TEST_ASSERT(ed.output_col == 2);
    int width;
    AGENT_TEST_ASSERT(linenoiseNextGrapheme(family, sizeof(family) - 1, &width) == sizeof(family) - 1 && width == 2);
    const char flag[] = "\xf0\x9f\x87\xae\xf0\x9f\x87\xb9";
    AGENT_TEST_ASSERT(linenoiseNextGrapheme(flag, sizeof(flag) - 1, &width) == sizeof(flag) - 1 && width == 2);
    editor_note_output(&ed, flag, sizeof(flag) - 1);
    AGENT_TEST_ASSERT(ed.output_col == 4);
    editor_note_output(&ed, "\x1b[", 2);
    AGENT_TEST_ASSERT(ed.output_escape == 2);
    editor_note_output(&ed, "31mX", 4);
    AGENT_TEST_ASSERT(ed.output_col == 5 && !ed.output_escape);

    agent_prompt_queue q = {0};
    char queued[181];
    for (int i = 0; i < 60; i++) memcpy(queued + i * 3, "\xe4\xb8\xad", 3);
    queued[180] = 0;
    agent_prompt_queue_push(&q, queued);
    agent_status st = {0};
    char footer[4096];
    build_footer_text(&st, &q, 40, footer, sizeof(footer));
    test_fixture("queue-footer.txt", footer, strlen(footer));
    for (size_t pos = 0; pos < strlen(footer);) {
        uint32_t cp;
        size_t n = linenoiseUtf8Decode(footer + pos, strlen(footer) - pos, &cp);
        AGENT_TEST_ASSERT(n && cp != 0xfffd);
        if (!n) break;
        pos += n;
    }
    agent_prompt_queue_free(&q);
}

static void test_footer_only_updates(void) {
    FILE *sink = tmpfile();
    AGENT_TEST_ASSERT(sink != NULL);
    if (!sink) return;
    int saved = dup(STDOUT_FILENO);
    dup2(fileno(sink), STDOUT_FILENO);
    agent_editor ed = {.active = true, .scroll_region = true, .term_rows = 24,
                       .term_cols = 80, .output_bottom = 22, .prompt_row = 23};
    snprintf(ed.prompt, sizeof(ed.prompt), "ds4-agent> ");
    snprintf(ed.status, sizeof(ed.status), "generation 0");
    char buffer[] = "draft";
    ed.edit = (struct linenoiseState){.ifd = -1, .ofd = STDOUT_FILENO,
        .buf = buffer, .buflen = sizeof(buffer), .len = 5, .pos = 5, .oldpos = 5,
        .prompt = ed.prompt, .plen = strlen(ed.prompt), .cols = 80,
        .oldrows = 1, .oldstatusrows = 1, .oldrpos = 1,
        .screen_cursor_row = 23, .screen_cursor_col = 17};
    linenoiseEditSetStatus(&ed.edit, ed.status, "", "");
    const char initial[] = "\x1b[23;1Hds4-agent> draft\r\ngeneration 0\x1b[23;17H";
    write_all(STDOUT_FILENO, initial, sizeof(initial) - 1);
    editor_set_prompt_status(&ed, ed.prompt, "generation 1");
    off_t first = lseek(STDOUT_FILENO, 0, SEEK_CUR);
    editor_set_prompt_status(&ed, ed.prompt, "generation 2");
    AGENT_TEST_ASSERT(lseek(STDOUT_FILENO, 0, SEEK_CUR) == first && ed.status_dirty);
    ed.last_prompt_redraw_time -= 1;
    editor_set_prompt_status(&ed, ed.prompt, "generation 2");
    AGENT_TEST_ASSERT(!ed.status_dirty);
    editor_set_prompt_status(&ed, ed.prompt, "done");
    editor_flush_prompt_status(&ed, true);
    AGENT_TEST_ASSERT(!ed.status_dirty && !strcmp(buffer, "draft"));
    dup2(saved, STDOUT_FILENO);
    close(saved);
    fseek(sink, 0, SEEK_END);
    size_t len = (size_t)ftell(sink);
    rewind(sink);
    char *text = xmalloc(len + 1);
    AGENT_TEST_ASSERT(fread(text, 1, len, sink) == len);
    text[len] = 0;
    AGENT_TEST_ASSERT(strstr(text, "\x1b[?2026h") && !strstr(text, "\x1b[0K"));
    test_fixture("status.ansi", text, len);
    free(text);
    free(ed.edit.status); free(ed.edit.status_start); free(ed.edit.status_end);
    fclose(sink);
}

static void test_tool_contracts(void) {
    for (int glm = 0; glm < 3; glm++) {
        for (int vision = 0; vision < 2; vision++) {
            char *prompt = glm == 2 ? agent_build_qwen_tools_prompt(false, vision) :
                           glm ? agent_build_glm_tools_prompt(false, vision) :
                                 agent_build_dsml_tools_prompt(false, vision);
            AGENT_TEST_ASSERT((strstr(prompt, "view_image") != NULL) == vision);
            AGENT_TEST_ASSERT(strstr(prompt, "POSIX extended") && strstr(prompt, "128 KiB"));
            AGENT_TEST_ASSERT(strstr(prompt, "&amp;lt;/"));
            char name[64];
            snprintf(name, sizeof(name), "prompt-%s-%d.txt", glm == 2 ? "qwen" : glm ? "glm" : "dsml", vision);
            test_fixture(name, prompt, strlen(prompt));
            free(prompt);
        }
    }
}

/* Model-free real-PTY driver for tests/ds4_agent_terminal_test.py. */
static int test_terminal_driver(void) {
    agent_editor ed = {0};
    linenoiseSetMultiLine(1);
    if (editor_start(&ed, "ds4-agent> ", "ready", NULL)) return 2;
    double start = now_sec(), next = start;
    char *answer = NULL;
    unsigned tick = 0;
    while (now_sec() - start < 10 && !answer) {
        struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
        poll(&pfd, 1, 10);
        if (pfd.revents & POLLIN) editor_read_stdin(&ed);
        while (linenoiseEditQueuedInput(&ed.edit)) {
            char *line = linenoiseEditFeed(&ed.edit);
            if (line == linenoiseEditMore) continue;
            if (line) answer = line;
            else answer = xstrdup("<input error>");
            break;
        }
        if (now_sec() >= next) {
            char status[80];
            snprintf(status, sizeof(status), "generation %u", ++tick);
            if (tick % 4 == 0) {
                const char output[] = "model output \xe4\xb8\xad\n";
                editor_write_async(&ed, output, sizeof(output) - 1, "ds4-agent> ", status, false);
            } else editor_set_prompt_status(&ed, "ds4-agent> ", status);
            next = now_sec() + 0.05;
        }
    }
    editor_stop(&ed);
    editor_restore_terminal_layout(&ed);
    if (!answer) return 3;
    printf("\nRESULT:");
    for (size_t i = 0; i < strlen(answer); i++) printf("%02x", (unsigned char)answer[i]);
    puts("");
    free(answer);
    return 0;
}

static void test_observation_error_is_not_context_exhaustion(void) {
    agent_worker worker = {0};
    ds4_tokens_push(&worker.transcript, 42);
    agent_tool_observation observation;
    agent_tool_observation_init(&observation);
    agent_tool_observation_puts(&observation, "image result");
    ds4_vision_embedding invalid = {0};
    agent_tool_observation_add_image(&observation, &invalid);
    char err[160] = {0};
    int count = -1;
    AGENT_TEST_ASSERT(agent_tool_observation_fits(&worker, &observation, 16,
                                                 &count, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "invalid image observation") != NULL);
    AGENT_TEST_ASSERT(count == -1);
    AGENT_TEST_ASSERT(worker.transcript.len == 1 && worker.transcript.v[0] == 42);
    agent_tool_observation_free(&observation);
    ds4_tokens_free(&worker.transcript);
}

static void test_compaction_boundaries(void) {
    agent_dsml_parser parser = {.state = AGENT_DSML_SEARCH};
    agent_stream_renderer stream = {.parser = &parser};
    AGENT_TEST_ASSERT(!agent_stream_compaction_needs_lookahead(&stream));
    stream.pending_len = 1;
    AGENT_TEST_ASSERT(agent_stream_compaction_needs_lookahead(&stream));
    stream.pending_len = 0;
    stream.dsml_start_len = 1;
    AGENT_TEST_ASSERT(agent_stream_compaction_needs_lookahead(&stream));
    stream.dsml_active = true;
    AGENT_TEST_ASSERT(!agent_stream_compaction_needs_lookahead(&stream));
    stream.dsml_active = false;
    parser.state = AGENT_DSML_PARAM_VALUE;
    AGENT_TEST_ASSERT(!agent_stream_compaction_needs_lookahead(&stream));
    int data[1000] = {0};
    ds4_tokens tokens = {.v = data, .len = 1000};
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, 42) == 900);
    data[850] = 42;
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, 42) == 850);
    data[950] = 42;
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, 42) == 950);
    data[850] = data[950] = 0;
    data[799] = 42;
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, 42) == 900);
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 150, 100, 100, 42) == 100);
    AGENT_TEST_ASSERT(agent_compact_tail_boundary(&tokens, 1000, 100, 100, -1) == 900);
    AGENT_TEST_ASSERT(agent_compact_summary_budget(4096) == 512);
    AGENT_TEST_ASSERT(agent_compact_summary_budget(100000) == 4096);
    AGENT_TEST_ASSERT(agent_compact_summary_budget(1024) == 256);
    ds4_vision_span spans[2] = {
        {.token_start = 100, .embedding = {.token_count = 50}},
        {.token_start = 200, .embedding = {.token_count = 30}},
    };
    for (int glm = 0; glm <= 1; glm++) {
        for (int pos = 0; pos < 260; pos++) {
            int expected = pos;
            for (size_t i = 0; i < 2; i++) {
                int start = (int)spans[i].token_start - glm;
                int end = (int)(spans[i].token_start + spans[i].embedding.token_count) + glm;
                if (pos > start && pos < end) expected = start;
            }
            AGENT_TEST_ASSERT(agent_compact_image_boundary(spans, 2, glm, pos) == expected);
        }
    }
    AGENT_TEST_ASSERT(agent_compact_image_boundary(NULL, 0, false, 77) == 77);
}

static int test_v41_thinking(const char *model) {
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = 512,
        .context_size = 256, .power_percent = 100};
    agent_config cfg = {.gen = {.ctx_size = 256, .think_mode = DS4_THINK_HIGH}};
    agent_worker w = {.cfg = &cfg, .initialized = true,
        .wake_fd = {-1, -1}, .status = {.state = AGENT_WORKER_IDLE}};
    pthread_mutex_init(&w.mu, NULL);
    AGENT_TEST_ASSERT(ds4_engine_open(&w.engine, &opt) == 0);
    if (!w.engine) return 1;
    AGENT_TEST_ASSERT(ds4_session_create(&w.session, w.engine, 256) == 0);
    if (!w.session) { ds4_engine_close(w.engine); return 1; }
    /* Simulate a restored session whose effort differs from the CLI default. */
    agent_numeric_think_prefix(w.engine, DS4_THINK_MAX, &w.transcript);
    ds4_tokens suffix = {0};
    ds4_tokenize_text(w.engine, "System text.\n\n", &suffix);
    ds4_chat_append_message(w.engine, &suffix, "user", "Hello");
    ds4_chat_append_message(w.engine, &suffix, "assistant", "Hello again.");
    const int first_image_offset = w.transcript.len;
    for (int i = 0; i < suffix.len; i++) ds4_tokens_push(&w.transcript, suffix.v[i]);
    w.images = calloc(1, sizeof(*w.images));
    w.image_count = 1;
    w.images[0].token_start = (uint32_t)first_image_offset;
    const int levels[] = {25, 0, 100, 1, 75, 75, 0};
    for (size_t i = 0; i < sizeof(levels) / sizeof(*levels); i++) {
        char err[160] = {0};
        AGENT_TEST_ASSERT(ds4_session_sync(w.session, &w.transcript, err, sizeof(err)) == 0);
        const int before = w.transcript.len;
        w.requested_think = (ds4_think_mode)(DS4_THINK_LEVEL_BASE + levels[i]);
        const bool changed = effective_think_mode(&cfg) != w.requested_think;
        w.think_requested = true;
        AGENT_TEST_ASSERT(!worker_is_idle(&w));
        AGENT_TEST_ASSERT(!worker_submit(&w, "must wait"));
        worker_apply_requested_think(&w);
        AGENT_TEST_ASSERT(worker_is_idle(&w));
        AGENT_TEST_ASSERT(ds4_session_pos(w.session) == (changed ? 0 : before));
        ds4_tokens expected = {0};
        agent_numeric_think_prefix(w.engine, w.requested_think, &expected);
        AGENT_TEST_ASSERT(w.images[0].token_start == (uint32_t)expected.len);
        for (int j = 0; j < suffix.len; j++) ds4_tokens_push(&expected, suffix.v[j]);
        AGENT_TEST_ASSERT(w.transcript.len == expected.len &&
                           ds4_tokens_starts_with(&w.transcript, &expected));
        ds4_tokens_free(&expected);
    }
    cfg.gen.raw_prompt = true;
    w.requested_think = DS4_THINK_MAX;
    worker_apply_requested_think(&w);
    AGENT_TEST_ASSERT(ds4_think_mode_level(cfg.gen.think_mode) == 0);
    AGENT_TEST_ASSERT(strstr(w.out, "requires a V4.1 chat session"));
    cfg.gen.raw_prompt = false;
    while (w.transcript.len < 250) ds4_tokens_push(&w.transcript, ds4_token_eos(w.engine));
    worker_apply_requested_think(&w);
    AGENT_TEST_ASSERT(ds4_think_mode_level(cfg.gen.think_mode) == 0 && w.transcript.len == 250);
    AGENT_TEST_ASSERT(strstr(w.out, "no context room"));
    free(w.out); free(w.images);
    ds4_tokens_free(&suffix); ds4_tokens_free(&w.transcript);
    ds4_session_free(w.session); ds4_engine_close(w.engine);
    pthread_mutex_destroy(&w.mu);
    puts("V4.1 agent thinking levels, restored prefix, cache invalidation: done");
    return agent_test_failures ? 1 : 0;
}

static void test_qwen_tool_syntax(void) {
    const char text[] =
        "<think>Plan.</think>\n<tool_call>\n<function=list>\n"
        "<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>";
    for (size_t split = 0; split < sizeof(text); split++) {
        char *first = xstrndup(text, split);
        const char *chunks[] = {first, text + split};
        agent_dsml_parser p;
        char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_QWEN, chunks, 2, &p, NULL);
        AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
        if (p.calls.len == 1) AGENT_TEST_ASSERT(!strcmp(p.calls.v[0].name, "list"));
        free(first);
        free(out);
        agent_dsml_parser_free(&p);
    }
    AGENT_TEST_ASSERT(agent_syntax_is_xml_tool_call(AGENT_TOOL_SYNTAX_QWEN));
    AGENT_TEST_ASSERT(agent_syntax_is_xml_tool_call(AGENT_TOOL_SYNTAX_GLM));
    AGENT_TEST_ASSERT(!agent_syntax_is_xml_tool_call(AGENT_TOOL_SYNTAX_DSML41));
    AGENT_TEST_ASSERT(!agent_syntax_is_xml_tool_call(AGENT_TOOL_SYNTAX_DSML));
}

/* ---- Hermes (Mellum) tool syntax ---- */

/* a Hermes call opens with <tool_call> and renders as chat messages, like Qwen's and GLM's */
static void test_hermes_syntax_basics(void) {
    AGENT_TEST_ASSERT(!strcmp(agent_tool_start(AGENT_TOOL_SYNTAX_HERMES), "<tool_call>"));
    AGENT_TEST_ASSERT(agent_syntax_is_xml_tool_call(AGENT_TOOL_SYNTAX_HERMES));
}

static const char hermes_read_call[] =
    "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"hello.py\"}}\n</tool_call>";

static agent_dsml_parser hermes_parser(void) {
    return (agent_dsml_parser){.syntax = AGENT_TOOL_SYNTAX_HERMES, .state = AGENT_DSML_SEARCH};
}

/* "<tool_call>\n" + body, fed whole and finished: the state, the error and the call count */
static agent_dsml_state hermes_parse_all(const char *body, char *err, size_t err_len, int *ncalls) {
    char text[1024];
    snprintf(text, sizeof(text), "<tool_call>\n%s", body);
    agent_dsml_parser p = hermes_parser();
    agent_dsml_feed(&p, text, strlen(text));
    agent_dsml_finish(&p);
    agent_dsml_state state = p.state;
    if (err) snprintf(err, err_len, "%s", p.error);
    if (ncalls) *ncalls = p.calls.len;
    agent_dsml_parser_free(&p);
    return state;
}

/* every split point of a reasoning-then-call turn ends DONE with the one call, and the call's
 * text is not printed as prose */
static void test_hermes_split_points(void) {
    const int failures_before = agent_test_failures;
    char text[512];
    snprintf(text, sizeof(text), "<think>\nThe user wants hello.py.\n</think>\n\n%s", hermes_read_call);
    for (size_t split = 0; split <= strlen(text); split++) {
        char *first = xstrndup(text, split);
        const char *chunks[] = {first, text + split};
        agent_dsml_parser p;
        char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_HERMES, chunks, 2, &p, NULL);
        AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
        AGENT_TEST_ASSERT(p.calls.len == 1);
        if (p.calls.len == 1) {
            const agent_tool_call *c = &p.calls.v[0];
            AGENT_TEST_ASSERT(!strcmp(c->name, "read"));
            AGENT_TEST_ASSERT(c->argc == 1 && !strcmp(c->args[0].name, "path") && c->args[0].is_string);
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(c, "path"), "hello.py"));
        }
        AGENT_TEST_ASSERT(strstr(out, "<tool_call>") == NULL);
        free(first);
        free(out);
        agent_dsml_parser_free(&p);
        if (agent_test_failures != failures_before) return;
    }
}

/* JSON string escapes decode, one byte at a time or whole; two adjacent calls are two */
static void test_hermes_strings_and_two_calls(void) {
    const char *text =
        "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.py\", "
        "\"content\": \"q=\\\"x\\\"\\nline2\\t\\\\ \\ud83d\\ude00 \\u00e9 caf\xc3\xa9\"}}\n</tool_call>\n"
        "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>";
    const size_t text_len = strlen(text);
    for (size_t step = 1; step <= text_len; step += text_len - 1) {   /* a byte at a time, then whole */
        agent_dsml_parser p = hermes_parser();
        for (size_t i = 0; i < text_len; i += step)
            agent_dsml_feed(&p, text + i, text_len - i < step ? text_len - i : step);
        agent_dsml_finish(&p);
        AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
        AGENT_TEST_ASSERT(p.calls.len == 2);
        if (p.calls.len == 2) {
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "content"),
                "q=\"x\"\nline2\t\\ \xf0\x9f\x98\x80 \xc3\xa9 caf\xc3\xa9"));
            AGENT_TEST_ASSERT(p.calls.v[0].args[1].is_string);
            AGENT_TEST_ASSERT(!strcmp(p.calls.v[1].name, "bash"));
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[1], "command"), "ls"));
        }
        agent_dsml_parser_free(&p);
    }
}

/* a literal </tool_call>, </think> or "} inside a string does not close anything */
static void test_hermes_markers_inside_strings(void) {
    const char *text =
        "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"p\", "
        "\"content\": \"</tool_call> </think> <think> \\\"} }} </tool_call>\"}}\n</tool_call>";
    agent_dsml_parser p = hermes_parser();
    for (size_t i = 0; text[i]; i++) {
        agent_dsml_feed(&p, text + i, 1);
        AGENT_TEST_ASSERT(p.state != AGENT_DSML_ERROR);
        /* the call completes only with the final </tool_call>, never at one in the string */
        AGENT_TEST_ASSERT(text[i + 1] ? p.calls.len == 0 : p.calls.len == 1);
        if (p.state == AGENT_DSML_ERROR) break;
    }
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
    if (p.calls.len == 1)
        AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "content"),
                                  "</tool_call> </think> <think> \"} }} </tool_call>"));
    agent_dsml_parser_free(&p);

    /* through the stream renderer too: a </think> inside the string is not a think close */
    const char *chunks[] = {"<think>x</think><tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"p\", \"content\": \"</th",
                            "ink> end\"}}\n</tool_call>"};
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_HERMES, chunks, 2, &p, NULL);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
    if (p.calls.len == 1)
        AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "content"), "</think> end"));
    free(out);
    agent_dsml_parser_free(&p);
}

/* malformed calls are errors, each beside a sibling that parses */
static void test_hermes_errors(void) {
    char err[160];
    int n = -1;
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": {\"path\": \"x\"}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"arguments\": {\"path\": \"x\"}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(strstr(err, "name") != NULL);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"\", \"arguments\": {}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": 5, \"arguments\": {}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    /* an empty arguments object is a call with no arguments */
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"list\", \"arguments\": {}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": \"path=x\"}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(strstr(err, "arguments") != NULL);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": [1]}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": {\"path\": \"x\"}} junk</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": {\"path\": \"x\"} junk}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": {\"path\": \"a\\qb\"}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    /* an unterminated string at the end of input is neither done nor a call */
    agent_dsml_parser p = hermes_parser();
    const char *cut = "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"x\", \"content\": \"abc";
    agent_dsml_feed(&p, cut, strlen(cut));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE && p.calls.len == 0);
    agent_dsml_parser_free(&p);
}

/* Step 44 (ds4-engine pathology 140): a write whose content holds bare quotes fails the strict parse.  The error
 * waits for </tool_call> instead of ending the turn, and the call's shape is unambiguous, so the bare quotes are
 * escaped and the call runs.  Siblings: a strict call parses as before; a bare quote in the path is not repaired (a
 * wrong path is another file); a content that has folded another member into itself is refused; a turn that ends
 * before </tool_call> reports the strict parse's error. */
static void test_hermes_quote_repair_write(void) {
    const char *text = "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.py\", "
                       "\"content\": \"print(\"hi\")\nx = \",\".join(r)\n\"}}\n</tool_call>";
    agent_dsml_parser p = hermes_parser();
    for (size_t i = 0; text[i]; i++) {
        agent_dsml_feed(&p, text + i, 1);
        AGENT_TEST_ASSERT(p.state != AGENT_DSML_ERROR);   /* the turn goes on to </tool_call> */
        if (p.state == AGENT_DSML_ERROR) break;
    }
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
    if (p.calls.len == 1) {
        AGENT_TEST_ASSERT(!strcmp(p.calls.v[0].name, "write"));
        AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "a.py"));
        AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "content"), "print(\"hi\")\nx = \",\".join(r)\n"));
    }
    agent_dsml_parser_free(&p);

    char err[160];
    int n = -1;
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"write\", \"arguments\": {\"path\": \"a.py\", "
                                       "\"content\": \"print(\\\"hi\\\")\"}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"write\", \"arguments\": {\"path\": \"a\"b.py\", "
                                       "\"content\": \"x\"}}\n</tool_call>", err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"write\", \"arguments\": {\"path\": \"a.py\", "
                                       "\"content\": \"x\", \"path\": \"b.py\", \"content\": \"say \"y\"\"}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    /* cut off inside the broken value: the strict parse's own error */
    p = hermes_parser();
    const char *cut = "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.py\", \"content\": \"print(\"hi";
    agent_dsml_feed(&p, cut, strlen(cut));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_ERROR && p.calls.len == 0);
    AGENT_TEST_ASSERT(strstr(p.error, "expected ',' or '}'") != NULL);
    agent_dsml_parser_free(&p);
}

/* Step 44, edit (pathology 140's own call): bare quotes in each edits[].oldText and newText are escaped when the
 * items' boundaries are unambiguous, also when an odd count of them left the raw edits value open until the turn
 * ended.  Siblings: a strict edit parses as before; a value holding an item boundary or another member's boundary is
 * refused, as is a bare quote in the path. */
static void test_hermes_quote_repair_edit(void) {
    char text[1024];
    snprintf(text, sizeof(text), "<tool_call>\n%s\n</tool_call>",
             "{\"name\": \"edit\", \"arguments\": {\"path\": \"r.py\", \"edits\": ["
             "{\"oldText\": \"\",\".join(str(v) for v in r)\", \"newText\": \"\", \".join(str(v) for v in r)\"}, "
             "{\"oldText\": \"say \"a\"\", \"newText\": \"say 'a'\"}]}}");
    agent_dsml_parser p = hermes_parser();
    for (size_t i = 0; text[i]; i++) {
        agent_dsml_feed(&p, text + i, 1);
        AGENT_TEST_ASSERT(p.state != AGENT_DSML_ERROR);
        if (p.state == AGENT_DSML_ERROR) break;
    }
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
    if (p.calls.len == 1) {
        AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "r.py"));
        AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "edits"),
            "[{\"oldText\": \"\\\",\\\".join(str(v) for v in r)\", \"newText\": \"\\\", \\\".join(str(v) for v in r)\"}, "
            "{\"oldText\": \"say \\\"a\\\"\", \"newText\": \"say 'a'\"}]"));
    }
    agent_dsml_parser_free(&p);

    char err[160];
    int n = -1;
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"edit\", \"arguments\": {\"path\": \"r.py\", \"edits\": "
                                       "[{\"oldText\": \"say \\\"a\\\"\", \"newText\": \"b\"}]}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"edit\", \"arguments\": {\"path\": \"r.py\", \"edits\": "
                                       "[{\"oldText\": \"say \"a\"\", \"newText\": \"b\"}, {\"oldText\": \"z\"}]}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    /* an odd count of bare quotes leaves the raw edits value open to the end: no call, never a repaired one */
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"edit\", \"arguments\": {\"path\": \"r.py\", \"edits\": "
                                       "[{\"oldText\": \"say \"a\", \"path\": \"x\", \"newText\": \"b\"}]}}\n</tool_call>",
                                       err, sizeof(err), &n) != AGENT_DSML_DONE && n == 0);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"edit\", \"arguments\": {\"path\": \"r.py\", \"edits\": "
                                       "[{\"oldText\": \"say \"a\", \"newText\": \"b\"}]}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"edit\", \"arguments\": {\"path\": \"r\"q.py\", \"edits\": "
                                       "[{\"oldText\": \"a\", \"newText\": \"b\"}]}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
}

/* greedy while the call's grammar is being written, sampled inside a string */
static void test_hermes_greedy_boundaries(void) {
    agent_tail_capture tail = {.cap = 4096};
    agent_token_renderer renderer = {.format_thinking = true, .last_output_newline = true, .capture = &tail};
    agent_dsml_parser p = hermes_parser();
    agent_stream_renderer stream = {.renderer = &renderer, .parser = &p, .syntax = AGENT_TOOL_SYNTAX_HERMES};
    #define FEED(s) agent_stream_text(&stream, (s), strlen(s), false)
    FEED("<tool_call>\n{\"name\": \"wri");
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));     /* the tool name */
    FEED("te\", \"arguments\": {\"path\": ");
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_STRUCTURAL);
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));     /* before a value */
    FEED("\"a.");
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(!agent_stream_wants_greedy_sampling(&stream));    /* inside a string */
    FEED("py\", \"con");
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));     /* the next key */
    FEED("tent\": \"x = [1, \\\"");
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(!agent_stream_wants_greedy_sampling(&stream));    /* an escaped quote stays inside */
    FEED("}\"}");
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));     /* the object's close */
    FEED("}\n</tool_call>");
    #undef FEED
    agent_stream_text(&stream, NULL, 0, true);
    renderer_finish(&renderer);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
    AGENT_TEST_ASSERT(!agent_stream_wants_greedy_sampling(&stream));    /* done: sampling is back */
    free(agent_tail_capture_take(&tail, NULL));
    agent_dsml_parser_free(&p);
}

/* numbers, booleans, null and nested values are kept as their raw JSON text */
static void test_hermes_raw_values(void) {
    const char *text =
        "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"x\", \"offset\": 12, \"limit\": 3}}\n</tool_call>\n"
        "<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"f\", \"edits\": "
        "[{\"oldText\": \"a\\\"b]}\", \"newText\": \"c\"}, {\"oldText\": \"x\", \"newText\": \"y\"}]}}\n</tool_call>\n"
        "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\", \"timeout\": 5, \"bg\": false}}\n</tool_call>";
    const size_t text_len = strlen(text);
    for (size_t step = 1; step <= text_len; step += text_len - 1) {   /* a byte at a time, then whole */
        agent_dsml_parser p = hermes_parser();
        for (size_t i = 0; i < text_len; i += step)
            agent_dsml_feed(&p, text + i, text_len - i < step ? text_len - i : step);
        agent_dsml_finish(&p);
        AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
        AGENT_TEST_ASSERT(p.calls.len == 3);
        if (p.calls.len == 3) {
            const agent_tool_call *r = &p.calls.v[0], *e = &p.calls.v[1], *b = &p.calls.v[2];
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(r, "offset"), "12"));
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(r, "limit"), "3"));
            AGENT_TEST_ASSERT(r->args[0].is_string && !r->args[1].is_string && !r->args[2].is_string);
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(e, "edits"),
                "[{\"oldText\": \"a\\\"b]}\", \"newText\": \"c\"}, {\"oldText\": \"x\", \"newText\": \"y\"}]"));
            AGENT_TEST_ASSERT(!e->args[1].is_string);
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(b, "timeout"), "5"));
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(b, "bg"), "false"));
        }
        agent_dsml_parser_free(&p);
    }

    /* what json.loads refuses as a value is refused; what it takes is taken */
    char err[160];
    int n;
    const char *bad[] = {"1x", "0x10", "nan", "01", "1.", "-", "+1", "inf", "1e", "[0x1]", "[1,]", "tru", "{\"a\" 1}"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        char body[200];
        snprintf(body, sizeof(body), "{\"name\": \"read\", \"arguments\": {\"offset\": %s}}\n</tool_call>", bad[i]);
        AGENT_TEST_ASSERT(hermes_parse_all(body, err, sizeof(err), &n) == AGENT_DSML_ERROR);
    }
    const char *good[] = {"0", "-0", "7", "-1.5e3", "2E+2", "true", "false", "null", "[1, 2.5, {\"a\": null}]", "{}"};
    for (size_t i = 0; i < sizeof(good) / sizeof(good[0]); i++) {
        char body[200];
        snprintf(body, sizeof(body), "{\"name\": \"read\", \"arguments\": {\"offset\": %s}}\n</tool_call>", good[i]);
        AGENT_TEST_ASSERT(hermes_parse_all(body, err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    }
    /* nesting past the limit is refused; at the limit it is taken */
    char deep[700], ok[700];
    int k = 0;
    k += snprintf(deep + k, sizeof(deep) - (size_t)k, "{\"name\": \"read\", \"arguments\": {\"x\": ");
    memcpy(ok, deep, (size_t)k);
    for (int i = 0; i < 257; i++) deep[k + i] = '[';
    for (int i = 0; i < 257; i++) deep[k + 257 + i] = ']';
    snprintf(deep + k + 514, sizeof(deep) - (size_t)k - 514, "}}\n</tool_call>");
    for (int i = 0; i < 256; i++) ok[k + i] = '[';
    for (int i = 0; i < 256; i++) ok[k + 256 + i] = ']';
    snprintf(ok + k + 512, sizeof(ok) - (size_t)k - 512, "}}\n</tool_call>");
    AGENT_TEST_ASSERT(hermes_parse_all(deep, err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(hermes_parse_all(ok, err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);

    /* a nested edits array is sampled, not greedy: it holds the code being written */
    agent_tail_capture tail = {.cap = 4096};
    agent_token_renderer renderer = {.format_thinking = true, .last_output_newline = true, .capture = &tail};
    agent_dsml_parser p = hermes_parser();
    agent_stream_renderer stream = {.renderer = &renderer, .parser = &p, .syntax = AGENT_TOOL_SYNTAX_HERMES};
    const char *edit = "<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"f\", \"edits\": [{\"oldText\": \"a";
    agent_stream_text(&stream, edit, strlen(edit), false);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(!agent_stream_wants_greedy_sampling(&stream));
    agent_stream_text(&stream, NULL, 0, true);
    renderer_finish(&renderer);
    free(agent_tail_capture_take(&tail, NULL));
    agent_dsml_parser_free(&p);
}

/* the call object as training's json.loads-based parser read it: key order, extra keys,
 * arguments as a JSON string, repeated keys; and what it refuses */
static void test_hermes_json_loads(void) {
    char err[160];
    int n;
    AGENT_TEST_ASSERT(hermes_parse_all("{\"arguments\": {\"path\": \"x\"}, \"name\": \"read\"}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"id\": \"c1\", \"name\": \"read\", \"arguments\": {\"path\": \"x\"}, \"meta\": {\"a\": [1, \"}\"]}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"id\": 7, \"arguments\": {\"path\": \"x\"}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);   /* still no name */

    /* arguments given as a JSON string are decoded and read as the object */
    const char *as_string = "<tool_call>\n{\"name\": \"read\", \"arguments\": \"{\\\"path\\\": \\\"a b.py\\\", \\\"offset\\\": 3}\"}\n</tool_call>";
    agent_dsml_parser p = hermes_parser();
    agent_dsml_feed(&p, as_string, strlen(as_string));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
    if (p.calls.len == 1) {
        AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "a b.py"));
        AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "offset"), "3"));
        AGENT_TEST_ASSERT(p.calls.v[0].args[0].is_string && !p.calls.v[0].args[1].is_string);
    }
    agent_dsml_parser_free(&p);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": \"not json\"}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": \"[1]\"}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);

    /* a repeated key keeps its last value, as in Python */
    p = hermes_parser();
    const char *dup = "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"first\", \"path\": \"second\"}}\n</tool_call>";
    agent_dsml_feed(&p, dup, strlen(dup));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.calls.len == 1 && p.calls.v[0].argc == 1 &&
                      !strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "second"));
    agent_dsml_parser_free(&p);

    /* \u0000 would cut a C string short: refused; an escaped backslash before u0000 is no NUL */
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"write\", \"arguments\": {\"path\": \"x\", \"content\": \"a\\u0000b\"}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"write\", \"arguments\": {\"path\": \"x\", \"content\": \"a\\\\u0000b\"}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    /* trailing commas */
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": {\"path\": \"x\",}}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": {\"path\": \"x\"},}\n</tool_call>",
                                       err, sizeof(err), &n) == AGENT_DSML_ERROR);
}

/* prose around calls, as training tolerated it: it neither ends the turn early nor drops a call */
static void test_hermes_prose_around_calls(void) {
    char err[160];
    int n;
    const char *call1 = "{\"name\": \"read\", \"arguments\": {\"path\": \"a.py\"}}\n</tool_call>";

    /* prose after a call: the parser is not DONE until the model stops, the call is kept, and the
     * prose is sampled, not greedy */
    agent_dsml_parser p = hermes_parser();
    char text[512];
    snprintf(text, sizeof(text), "<tool_call>\n%s\nNow I wait for the result.", call1);
    for (size_t i = 0; text[i]; i++) {
        agent_dsml_feed(&p, text + i, 1);
        AGENT_TEST_ASSERT(p.state != AGENT_DSML_DONE && p.state != AGENT_DSML_ERROR);
    }
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(!agent_stream_wants_greedy_sampling(&(agent_stream_renderer){.parser = &p, .dsml_active = true}));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
    agent_dsml_parser_free(&p);

    /* prose between two calls does not drop the second */
    char two[512];
    snprintf(two, sizeof(two), "%s\nand then, a second look\n<tool_call>\n%s", call1,
             "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>");
    AGENT_TEST_ASSERT(hermes_parse_all(two, err, sizeof(err), &n) == AGENT_DSML_DONE && n == 2);
    /* a half-written opener inside the prose is only prose */
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": {}}\n</tool_call>\nsee <tool_ca",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
    /* sibling: whitespace alone after a call is the end of the turn */
    AGENT_TEST_ASSERT(hermes_parse_all("{\"name\": \"read\", \"arguments\": {}}\n</tool_call>\n\n",
                                       err, sizeof(err), &n) == AGENT_DSML_DONE && n == 1);
}

/* the system turn: the agent's preamble, a blank line, then the tools block 2.1's template writes
 * for Pi's four tools (mellum_tool_text.h, which test_mellum_config checks against jinja2) */
static void test_hermes_system_turn(void) {
    char *want = mtt_system_text(NULL);
    char *got = agent_build_hermes_tools_block();
    AGENT_TEST_ASSERT(!strcmp(got, want));
    free(got);
    free(want);

    const char *preamble = "You are a coding agent.\n\nCurrent working directory: /testbed";
    char *sys = agent_build_hermes_system_text(preamble);
    char *ref = mtt_system_text(preamble);
    AGENT_TEST_ASSERT(!strcmp(sys, ref));
    AGENT_TEST_ASSERT(strstr(sys, "<|im_start|>") == NULL && strstr(sys, "<|im_end|>") == NULL);
    free(sys);
    free(ref);
    sys = agent_build_hermes_system_text(NULL);     /* no preamble: the block alone */
    ref = mtt_system_text(NULL);
    AGENT_TEST_ASSERT(!strcmp(sys, ref));
    free(sys);
    free(ref);

    /* what the agent puts ahead of the block, in order, and nothing from the tools in it */
    char *pre = agent_build_hermes_preamble("EXTRA-SYS", "DATETIME-LINE", "/work/dir");
    const char *extra = strstr(pre, "EXTRA-SYS");
    const char *dt = strstr(pre, "DATETIME-LINE");
    const char *cwd = strstr(pre, "Current working directory: /work/dir");
    AGENT_TEST_ASSERT(extra && dt && cwd && pre < extra && extra < dt && dt < cwd);
    AGENT_TEST_ASSERT(strstr(pre, "# Tools") == NULL && strstr(pre, "<tool_call>") == NULL);
    free(pre);
    pre = agent_build_hermes_preamble(NULL, NULL, NULL);
    AGENT_TEST_ASSERT(pre[0] && strstr(pre, "Current working directory") == NULL);
    free(pre);
}

/* a call inside <think> is ignored; Mellum's prefix has no <think>, so a turn starts outside it */
static void test_hermes_think_state(void) {
    const char *inside[] = {
        "<think>plan <tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"x\"}}\n</tool_call>",
        "</think>done",
    };
    agent_dsml_parser p;
    bool in_think = false;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_HERMES, inside, 2, &p, &in_think);
    AGENT_TEST_ASSERT(in_think);
    AGENT_TEST_ASSERT(p.calls.len == 0);
    AGENT_TEST_ASSERT(strstr(out, "[tool call ignored:") != NULL);
    free(out);
    agent_dsml_parser_free(&p);

    AGENT_TEST_ASSERT(!agent_turn_starts_in_think(AGENT_TOOL_SYNTAX_HERMES, DS4_THINK_HIGH));
    AGENT_TEST_ASSERT(!agent_turn_starts_in_think(AGENT_TOOL_SYNTAX_HERMES, DS4_THINK_NONE));
    /* siblings: the other models' prefixes open the block when thinking is on */
    AGENT_TEST_ASSERT(agent_turn_starts_in_think(AGENT_TOOL_SYNTAX_QWEN, DS4_THINK_HIGH));
    AGENT_TEST_ASSERT(!agent_turn_starts_in_think(AGENT_TOOL_SYNTAX_QWEN, DS4_THINK_NONE));
    AGENT_TEST_ASSERT(agent_turn_starts_in_think(AGENT_TOOL_SYNTAX_DSML, DS4_THINK_HIGH));

    const char *direct[] = {hermes_read_call};
    bool direct_in_think = true;
    out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_HERMES, direct, 1, &p, &direct_in_think);
    AGENT_TEST_ASSERT(!direct_in_think);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
    free(out);
    agent_dsml_parser_free(&p);
}

/* the turn ends on the EOS token and then a newline, as the template writes <|im_end|>\n */
static void test_hermes_turn_end(void) {
    AGENT_TEST_ASSERT(agent_tool_syntax_assistant_turn_uses_eos(AGENT_TOOL_SYNTAX_HERMES));
    AGENT_TEST_ASSERT(!strcmp(agent_turn_end_suffix(AGENT_TOOL_SYNTAX_HERMES), "\n"));
    /* siblings: the other syntaxes' turn end is unchanged */
    AGENT_TEST_ASSERT(!strcmp(agent_turn_end_suffix(AGENT_TOOL_SYNTAX_GLM), ""));
    AGENT_TEST_ASSERT(!strcmp(agent_turn_end_suffix(AGENT_TOOL_SYNTAX_QWEN), ""));
    AGENT_TEST_ASSERT(!strcmp(agent_turn_end_suffix(AGENT_TOOL_SYNTAX_DSML), ""));
}

/* only the EOS class ends a Hermes turn cleanly; id 0 and the protocol ids are abnormal */
static void test_hermes_stop_classes(void) {
    AGENT_TEST_ASSERT(agent_stop_is_clean_end(AGENT_TOOL_SYNTAX_HERMES, DS4_STOP_EOS));
    AGENT_TEST_ASSERT(!agent_stop_is_clean_end(AGENT_TOOL_SYNTAX_HERMES, DS4_STOP_EOS_ALT));
    AGENT_TEST_ASSERT(!agent_stop_is_clean_end(AGENT_TOOL_SYNTAX_HERMES, DS4_STOP_PROTOCOL));
    AGENT_TEST_ASSERT(!agent_stop_is_clean_end(AGENT_TOOL_SYNTAX_HERMES, DS4_STOP_MARKER));
    AGENT_TEST_ASSERT(!agent_stop_is_clean_end(AGENT_TOOL_SYNTAX_HERMES, DS4_STOP_NONE));
    /* siblings: the other syntaxes treat every stop as the end of the reply, as before (a think
     * tag stops a --nothink generation with class NONE) */
    for (int c = DS4_STOP_NONE; c <= DS4_STOP_MARKER; c++) {
        AGENT_TEST_ASSERT(agent_stop_is_clean_end(AGENT_TOOL_SYNTAX_QWEN, c));
        AGENT_TEST_ASSERT(agent_stop_is_clean_end(AGENT_TOOL_SYNTAX_DSML, c));
    }
}

/* step 37: a Mellum turn the token cap ended (no stop token, no complete call) is cut off, not an answer; a stop,
 * a call, a malformed call, an early tool error, an interruption or a turn short of the cap is not; and the other
 * syntaxes accept a capped turn as before */
static void test_hermes_turn_cut_off(void) {
    const agent_tool_syntax H = AGENT_TOOL_SYNTAX_HERMES;
    AGENT_TEST_ASSERT(agent_turn_cut_off(H, false, false, false, false, false, 512, 512));
    AGENT_TEST_ASSERT(!agent_turn_cut_off(H, true, false, false, false, false, 512, 512));
    AGENT_TEST_ASSERT(!agent_turn_cut_off(H, false, true, false, false, false, 512, 512));
    AGENT_TEST_ASSERT(!agent_turn_cut_off(H, false, false, true, false, false, 512, 512));
    AGENT_TEST_ASSERT(!agent_turn_cut_off(H, false, false, false, true, false, 512, 512));
    AGENT_TEST_ASSERT(!agent_turn_cut_off(H, false, false, false, false, true, 512, 512));
    AGENT_TEST_ASSERT(!agent_turn_cut_off(H, false, false, false, false, false, 100, 512));
    AGENT_TEST_ASSERT(!agent_turn_cut_off(AGENT_TOOL_SYNTAX_QWEN, false, false, false, false, false, 512, 512));
    AGENT_TEST_ASSERT(!agent_turn_cut_off(AGENT_TOOL_SYNTAX_DSML, false, false, false, false, false, 512, 512));
}

/* step 39: the compaction summary is a generation too; for Mellum only EOS ends it, id 0 and the protocol ids are an
 * abnormal end that keeps the previous conversation */
static void test_hermes_compaction_stop(void) {
    const agent_tool_syntax H = AGENT_TOOL_SYNTAX_HERMES;
    AGENT_TEST_ASSERT(agent_compaction_stop_is_clean(H, DS4_STOP_EOS, false));
    AGENT_TEST_ASSERT(!agent_compaction_stop_is_clean(H, DS4_STOP_EOS_ALT, false));
    AGENT_TEST_ASSERT(!agent_compaction_stop_is_clean(H, DS4_STOP_PROTOCOL, false));
    /* siblings: the other syntaxes end the summary on any stop, and on the DSML marker */
    AGENT_TEST_ASSERT(agent_compaction_stop_is_clean(AGENT_TOOL_SYNTAX_QWEN, DS4_STOP_EOS_ALT, false));
    AGENT_TEST_ASSERT(agent_compaction_stop_is_clean(AGENT_TOOL_SYNTAX_DSML, DS4_STOP_NONE, true));
    AGENT_TEST_ASSERT(agent_compaction_stop_is_clean(AGENT_TOOL_SYNTAX_QWEN, DS4_STOP_PROTOCOL, false));
}

/* the [upto] forcer rewrites an edit's old text; Pi's edit has no place for it, so it stays off */
static void test_hermes_upto_forcer_off(void) {
    AGENT_TEST_ASSERT(!agent_edit_upto_enabled(true, AGENT_TOOL_SYNTAX_HERMES));
    AGENT_TEST_ASSERT(!agent_edit_upto_enabled(false, AGENT_TOOL_SYNTAX_HERMES));
    AGENT_TEST_ASSERT(agent_edit_upto_enabled(true, AGENT_TOOL_SYNTAX_QWEN));
    AGENT_TEST_ASSERT(!agent_edit_upto_enabled(false, AGENT_TOOL_SYNTAX_QWEN));
}

/* step 36: the terminal projection of a Hermes call shows the decoded arguments, not their JSON escapes */
static void test_hermes_visualizer_decodes(void) {
    const char *chunks[] = {
        "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"printf \\\"a\\\\tb\\\"\\nls \\u00e9 \\ud83d\\ude00\", ",
        "\"timeout\": 15}}\n</tool_call>",
    };
    agent_dsml_parser p;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_HERMES, chunks, 2, &p, NULL);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 1);
    AGENT_TEST_ASSERT(strstr(out, "printf \"a\\tb\"\nls \xc3\xa9 \xf0\x9f\x98\x80") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "\\u00e9") == NULL && strstr(out, "\\n") == NULL);
    AGENT_TEST_ASSERT(strstr(out, "15") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "<tool_call>") == NULL && strstr(out, "\"arguments\"") == NULL);
    free(out);
    agent_dsml_parser_free(&p);

    /* a read shows Pi's path, offset and limit, the first digit included */
    const char *read[] = {"<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"src/a.py\", \"offset\": 120, \"limit\": 40}}\n</tool_call>"};
    out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_HERMES, read, 1, &p, NULL);
    AGENT_TEST_ASSERT(strstr(out, "Reading src/a.py 120:40...") != NULL);
    free(out);
    agent_dsml_parser_free(&p);
    /* Pi's read with a path alone reads the file (to Pi's own cap), so no range is shown, not the
     * agent's own read defaults; an offset alone reads on from there */
    const char *whole[] = {"<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"hello.py\"}}\n</tool_call>"};
    out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_HERMES, whole, 1, &p, NULL);
    AGENT_TEST_ASSERT(strstr(out, "Reading hello.py...") != NULL);
    free(out);
    agent_dsml_parser_free(&p);
    const char *from[] = {"<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.py\", \"offset\": 50}}\n</tool_call>"};
    out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_HERMES, from, 1, &p, NULL);
    AGENT_TEST_ASSERT(strstr(out, "Reading a.py 50:EOF...") != NULL);
    free(out);
    agent_dsml_parser_free(&p);
}

static void test_hermes_tool_syntax(void) {
    test_hermes_syntax_basics();
    test_hermes_split_points();
    test_hermes_strings_and_two_calls();
    test_hermes_markers_inside_strings();
    test_hermes_errors();
    test_hermes_quote_repair_write();
    test_hermes_quote_repair_edit();
    test_hermes_greedy_boundaries();
    test_hermes_raw_values();
    test_hermes_json_loads();
    test_hermes_prose_around_calls();
    test_hermes_system_turn();
    test_hermes_think_state();
    test_hermes_turn_end();
    test_hermes_stop_classes();
    test_hermes_upto_forcer_off();
    test_hermes_visualizer_decodes();
    test_hermes_turn_cut_off();
    test_hermes_compaction_stop();
}

static void test_v41_tool_syntax(void) {
    const char text[] =
        "<think>Plan.</think>\n\n<｜DSML｜ calls>\n"
        "<｜DSML｜ invoke name=\"write\">\n"
        "<｜DSML｜ parameter name=\"path\" string=\"true\">a.txt</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"content\" string=\"true\">x </think> "
        "&lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter></｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"list\">\n"
        "<｜DSML｜ parameter name=\"path\" string=\"true\">.</｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const char expected[] = "x </think> </｜DSML｜ parameter> &lt;/｜DSML｜ parameter>";
    for (size_t split = 0; split < sizeof(text); split++) {
        char *first = xstrndup(text, split);
        const char *chunks[] = {first, text + split};
        agent_dsml_parser p;
        char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_DSML41, chunks, 2, &p, NULL);
        AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE && p.calls.len == 2);
        if (p.state != AGENT_DSML_DONE || p.calls.len != 2) {
            fprintf(stderr, "V4.1 split=%zu state=%d calls=%d error=%s raw=%s\n",
                    split, p.state, p.calls.len, p.error, p.raw ? p.raw : "(none)");
            free(first); free(out); agent_dsml_parser_free(&p);
            break;
        }
        if (p.calls.len == 2) {
            AGENT_TEST_ASSERT(!strcmp(p.calls.v[0].name, "write"));
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "a.txt"));
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "content"), expected));
            AGENT_TEST_ASSERT(!strcmp(p.calls.v[1].name, "list"));
        }
        AGENT_TEST_ASSERT(!strstr(out, "<｜DSML｜ calls>"));
        AGENT_TEST_ASSERT(p.raw && strstr(p.raw, "<｜DSML｜ calls>") == p.raw);
        free(first); free(out); agent_dsml_parser_free(&p);
    }
    const char *inside[] = {"<think><｜DSML｜ calls><｜DSML｜ invoke name=\"list\">"
        "</｜DSML｜ invoke></｜DSML｜ calls></think>Done"};
    agent_dsml_parser p;
    bool early = false;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_DSML41, inside, 1, &p, &early);
    AGENT_TEST_ASSERT(early && p.calls.len == 0 && strstr(out, "tool call ignored"));
    free(out); agent_dsml_parser_free(&p);
    for (int upto = 0; upto < 2; upto++) {
        char *old = agent_build_dsml_tools_prompt(upto, false);
        char *prompt = agent_dsml41_tools_prompt(old);
        AGENT_TEST_ASSERT(strstr(prompt, "<｜DSML｜ calls>"));
        AGENT_TEST_ASSERT(strstr(prompt, "&amp;lt;/｜DSML｜ parameter>"));
        AGENT_TEST_ASSERT(!strstr(prompt, "｜DSML｜tool_calls") &&
                           !strstr(prompt, "｜DSML｜invoke") &&
                           !strstr(prompt, "｜DSML｜parameter"));
        free(old); free(prompt);
    }
}

/* Model-backed regression: save an unsynchronizable transcript, then rebuild
 * it in a larger context using the normal stripped-session loader. */
static int test_full_context_save(const char *model) {
    ds4_engine_options opt = {.model_path = model, .backend = default_backend(),
        .context_size = 512, .power_percent = 100};
    agent_config cfg = {.gen = {.ctx_size = 256}, .non_interactive = true};
    agent_worker w = {.cfg = &cfg, .initialized = true, .user_activity = true,
        .wake_fd = {-1, -1}, .status = {.state = AGENT_WORKER_IDLE}};
    pthread_mutex_init(&w.mu, NULL);
    char dir[] = "/tmp/ds4-agent-full-save-XXXXXX";
    AGENT_TEST_ASSERT(mkdtemp(dir) != NULL);
    w.cache_dir = dir;
    w.session_title = xstrdup("Full transcript recovery");
    w.session_created_at = 123456;
    AGENT_TEST_ASSERT(ds4_engine_open(&w.engine, &opt) == 0);
    if (!w.engine) return 1;
    ds4_tokens complete = {0};
    ds4_chat_begin(w.engine, &complete);
    for (int i = 0; i < 60; i++)
        ds4_chat_append_message(w.engine, &complete, "user", "Explain the colors of a rainbow.");
    AGENT_TEST_ASSERT(complete.len > 257);
    for (int length = 256; length <= 257; length++) {
        AGENT_TEST_ASSERT(ds4_session_create(&w.session, w.engine, 256) == 0);
        if (!w.session) break;
        ds4_tokens slice = complete;
        slice.len = length;
        ds4_tokens_copy(&w.transcript, &slice);
        w.session_dirty = true;
        char err[256] = {0}, sha[41];
        int saved_tokens = 0;
        AGENT_TEST_ASSERT(agent_worker_save_session_now(&w, sha, &saved_tokens, err, sizeof(err)));
        AGENT_TEST_ASSERT(saved_tokens == length && !w.session_dirty);
        AGENT_TEST_ASSERT(ds4_session_pos(w.session) == 0);
        char *path = agent_kv_path_for_sha(dir, sha);
        FILE *fp = fopen(path, "rb");
        AGENT_TEST_ASSERT(fp != NULL);
        if (!fp) { free(path); ds4_session_free(w.session); w.session = NULL; break; }
        ds4_kvstore_entry hdr = {0};
        uint32_t text_bytes = 0;
        AGENT_TEST_ASSERT(ds4_kvstore_read_header(fp, &hdr, &text_bytes));
        AGENT_TEST_ASSERT(hdr.payload_bytes == 0 && hdr.tokens == (uint32_t)length);
        char *text = NULL, *title = NULL;
        AGENT_TEST_ASSERT(agent_kv_read_text(fp, text_bytes, &text, err, sizeof(err)));
        AGENT_TEST_ASSERT(agent_kv_read_title_trailer(fp, &hdr, &title, err, sizeof(err)));
        AGENT_TEST_ASSERT(title && !strcmp(title, w.session_title));
        fclose(fp);
        size_t expected_len = 0;
        char *expected = ds4_kvstore_render_tokens_text(w.engine, &w.transcript, &expected_len);
        AGENT_TEST_ASSERT(text && expected && text_bytes == expected_len && !strcmp(text, expected));
        free(expected); free(text); free(title);
        ds4_session_free(w.session);
        w.session = NULL;
        AGENT_TEST_ASSERT(ds4_session_create(&w.session, w.engine, 512) == 0);
        cfg.gen.ctx_size = 512;
        ds4_tokens restored = {0};
        agent_kv_session_meta meta = {0};
        AGENT_TEST_ASSERT(agent_kv_load_path(&w, path, sha, NULL, 0, &restored, &meta, err, sizeof(err)));
        AGENT_TEST_ASSERT(agent_tokens_equal(&w.transcript, &restored));
        AGENT_TEST_ASSERT(meta.created_at == w.session_created_at && meta.title && !strcmp(meta.title, w.session_title));
        ds4_token_score score;
        AGENT_TEST_ASSERT(ds4_session_top_logprobs(w.session, &score, 1) == 1);
        agent_kv_session_meta_free(&meta);
        ds4_tokens_free(&restored);
        ds4_tokens_free(&w.transcript);
        ds4_session_free(w.session); w.session = NULL;
        cfg.gen.ctx_size = 256;
        unlink(path); free(path);
    }
    ds4_tokens_free(&complete);
    ds4_engine_close(w.engine);
    free(w.session_title);
    pthread_mutex_destroy(&w.mu);
    rmdir(dir);
    return agent_test_failures ? 1 : 0;
}

/* A headless run ends when the worker is idle, and worker_is_idle counts AGENT_WORKER_ERROR as
 * idle.  The loop's status snapshot is taken earlier in the iteration, so an abnormal end that
 * lands between the snapshot and the idle check must still exit 1 with the error on stderr
 * (the old port's ladder run2 r4s2 exited 0 with no line; r5s2 and r5s3, the same cut, exited 1).
 * Step 38. */
static void test_noninteractive_exit_after_late_error(void) {
    agent_worker w;
    memset(&w, 0, sizeof(w));
    AGENT_TEST_ASSERT(pthread_mutex_init(&w.mu, NULL) == 0);
    AGENT_TEST_ASSERT(pipe(w.wake_fd) == 0);
    w.initialized = true;
    w.status.state = AGENT_WORKER_GENERATING;

    pthread_mutex_lock(&w.mu);
    agent_status snapshot = w.status;                 /* the loop's worker_consume */
    pthread_mutex_unlock(&w.mu);
    agent_set_error(&w, "turn cut off by the token cap after 2048 tokens, before the end-of-turn token");
    AGENT_TEST_ASSERT(snapshot.state != AGENT_WORKER_ERROR);
    AGENT_TEST_ASSERT(worker_is_idle(&w));            /* the loop breaks here */

    FILE *err = tmpfile();
    AGENT_TEST_ASSERT(err != NULL);
    AGENT_TEST_ASSERT(agent_noninteractive_exit_code(&w, 0, err) == 1);
    char line[512] = {0};
    rewind(err);
    AGENT_TEST_ASSERT(fgets(line, sizeof(line), err) != NULL);
    AGENT_TEST_ASSERT(!strcmp(line, "ds4-agent: turn cut off by the token cap after 2048 tokens, "
                                    "before the end-of-turn token\n"));
    fclose(err);

    /* sibling: the loop already reported the error it saw (rc 1); nothing is printed twice */
    err = tmpfile();
    AGENT_TEST_ASSERT(agent_noninteractive_exit_code(&w, 1, err) == 1);
    AGENT_TEST_ASSERT(ftell(err) == 0);
    fclose(err);

    /* sibling: a run that ended idle without an error exits 0 and prints nothing */
    w.status.state = AGENT_WORKER_IDLE;
    w.status.error[0] = '\0';
    err = tmpfile();
    AGENT_TEST_ASSERT(agent_noninteractive_exit_code(&w, 0, err) == 0);
    AGENT_TEST_ASSERT(ftell(err) == 0);
    fclose(err);

    close(w.wake_fd[0]);
    close(w.wake_fd[1]);
    pthread_mutex_destroy(&w.mu);
}


int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--full-context-save")) return test_full_context_save(argv[2]);
    if (argc == 3 && !strcmp(argv[1], "--think-fixture")) return test_v41_thinking(argv[2]);
    if (argc == 2 && !strcmp(argv[1], "--terminal-driver")) return test_terminal_driver();
    if (argc == 3 && !strcmp(argv[1], "--terminal-fixtures")) test_output_dir = argv[2];
    char *options[] = {"ds4-agent", "--model", "qwen.gguf",
                    "--vision", "mmproj.gguf", "--non-interactive", "-p", "test"};
    agent_config cfg = parse_options((int)(sizeof(options) / sizeof(options[0])), options);
    AGENT_TEST_ASSERT(cfg.engine.vision_path && !strcmp(cfg.engine.vision_path, "mmproj.gguf"));
    AGENT_TEST_ASSERT(cfg.engine.model_path && !strcmp(cfg.engine.model_path, "qwen.gguf"));
    ds4_agent_unit_tests_run();
    test_v41_tool_syntax();
    test_qwen_tool_syntax();
    test_hermes_tool_syntax();
    test_noninteractive_exit_after_late_error();
    test_compaction_boundaries();
    test_observation_error_is_not_context_exhaustion();
    test_atomic_file_tools();
    test_streaming_file_tools();
    test_shell_spawn();
    test_background_jobs();
    test_fragmented_terminal_input();
    test_shell_terminal_controls();
    test_markdown_literals();
    test_hint_rendering();
    test_unicode_output_and_footer();
    test_footer_only_updates();
    test_tool_contracts();
    if (agent_test_failures) {
        fprintf(stderr, "ds4-agent tests: %d failure(s)\n",
                agent_test_failures);
        return 1;
    }
    puts("ds4-agent tests: ok");
    return 0;
}
