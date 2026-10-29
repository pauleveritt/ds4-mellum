/* mellum_trace -- run Mellum 2.1's forward pass over a token file and let the
 * engine write its trace.  The engine does the writing: with
 * DS4_MELLUM_TRACE=<prefix> and DS4_MELLUM_TRACE_AT=<pos,pos,...> set, every
 * forward that covers a listed absolute position writes, for that position,
 *
 *   <prefix>.p<pos>.hidden.f32   28 x 2304 f32: each layer's output row
 *   <prefix>.p<pos>.qk.f32       28 x 4608 f32: post-norm, post-RoPE Q (4096)
 *                                then K (512), layer by layer
 *   <prefix>.p<pos>.ffn_in.f32   28 x 2304 f32: each layer's MoE input row
 *   <prefix>.p<pos>.logits.f32   1 x 98304 f32
 *
 * The last row of a hidden file is the last layer's output before the output
 * norm, the state an MTP head would read.
 * This driver only runs the model and checks the files.  It also writes the
 * session's own last-row logits (what ds4_session_copy_logits returns after
 * the last forward) to <prefix>.session.logits.f32, 1 x 98304 f32.  A traced
 * forward ends its command buffer after every layer, so it cannot show a
 * hazard between layers of the one buffer an ordinary run uses; with only
 * DS4_MELLUM_TRACE set (no DS4_MELLUM_TRACE_AT) the driver clears it for the
 * engine, the run is untraced, and its session logits file is the one to `cmp`
 * against a traced run's (e.g. both chunks of a two-chunk prefill traced).
 *
 *   tests/mellum_trace MODEL.gguf TOKENS_FILE MODE
 *
 * TOKENS_FILE holds whitespace-separated non-negative decimal ids; `#` starts
 * a comment to the end of the line.  MODE is `prefill` (one
 * ds4_session_sync of every token), `decode` (a sync of the first token,
 * then one ds4_session_eval per remaining token) or `prefill:N` (a sync of
 * the first N tokens, then one eval per remaining token: decode steps that
 * read a prefilled cache).  Prints one line on stdout:
 * the token count, the mode, the argmax of the last position's logits, how
 * many of them are not finite, and the wall seconds of the forward passes.
 * MELLUM_TRACE_REPEAT=N (default 1) runs the whole thing N times on a fresh
 * session, so the first, cold run (pipelines are compiled on first use) can be
 * told from the warm ones.  MELLUM_TRACE_CONTEXT fixes the context (decimal,
 * >= tokens + 64); absent keeps max(4096, tokens + 64).  MELLUM_TRACE_GREEDY=K
 * (0 to 64) then decodes K greedy tokens from the last run's cache, after its
 * session logits are written, and prints them on a second line.
 *
 * Every file the run is to write is removed before it starts, so a file from an
 * earlier run with the same prefix cannot pass for this run's.
 * Exit 0 when the run succeeded, every file of every listed position (and the
 * session logits file) exists with its exact size, and every listed position's
 * logits are finite.  Exit 1 on a failed run, a listed position that is not one
 * of the tokens (the error names it), a missing or short file (the error names
 * the position and the file) or a non-finite logit.  Exit 2 on bad arguments.
 * Never --raw-prompt: this feeds token ids, not text.
 */
#include "../ds4.h"
#include "mellum_trace_context.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { N_LAYER = 28, HIDDEN = 2304, QK = 4608, VOCAB = 98304, MAX_AT = 64 };

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ids from the token file: false on any byte that is not a digit, space or comment */
static bool read_tokens(const char *path, ds4_tokens *out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "mellum_trace: cannot read %s\n", path);
        return false;
    }
    bool ok = true, comment = false, in_num = false;
    unsigned long long v = 0;
    int c;
    while (ok && (c = fgetc(f)) != EOF) {
        if (comment) {
            if (c == '\n') comment = false;
            continue;
        }
        if (isdigit(c)) {
            v = v * 10u + (unsigned)(c - '0');
            in_num = true;
            if (v > INT32_MAX) ok = false;
            continue;
        }
        if (in_num) {
            ds4_tokens_push(out, (int)v);
            v = 0;
            in_num = false;
        }
        if (c == '#') comment = true;
        else if (!isspace(c)) ok = false;
    }
    if (ok && in_num) ds4_tokens_push(out, (int)v);
    fclose(f);
    if (!ok) fprintf(stderr, "mellum_trace: %s is not a list of non-negative decimal token ids\n", path);
    if (ok && out->len <= 0) {
        fprintf(stderr, "mellum_trace: %s holds no token\n", path);
        ok = false;
    }
    return ok;
}

/* DS4_MELLUM_TRACE_AT: comma-separated absolute positions */
static int parse_positions(const char *text, unsigned *pos) {
    int n = 0;
    const char *p = text;
    while (*p) {
        if (!isdigit((unsigned char)*p)) return -1;
        char *end = NULL;
        errno = 0;
        const unsigned long v = strtoul(p, &end, 10);
        if (errno || v > INT32_MAX || n >= MAX_AT) return -1;
        pos[n++] = (unsigned)v;
        if (*end == ',') end++;
        else if (*end) return -1;
        p = end;
    }
    return n;
}

static bool check_file(const char *prefix, unsigned pos, const char *what, uint64_t bytes) {
    char path[4096];
    snprintf(path, sizeof(path), "%s.p%u.%s.f32", prefix, pos, what);
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "mellum_trace: position %u: %s was not written\n", pos, path);
        return false;
    }
    if ((uint64_t)st.st_size != bytes) {
        fprintf(stderr, "mellum_trace: position %u: %s is %llu bytes, expected %llu\n", pos, path,
                (unsigned long long)st.st_size, (unsigned long long)bytes);
        return false;
    }
    return true;
}

/* MELLUM_TRACE_SESSION_AT: the session's current logits (the last forward's last row) as <prefix>.p<pos> */
static bool write_session_logits(ds4_session *s, const char *prefix, int pos) {
    float *v = malloc((size_t)VOCAB * sizeof(float));
    char path[4096];
    snprintf(path, sizeof(path), "%s.p%d.logits.f32", prefix, pos);
    FILE *f = NULL;
    const bool ok = v && ds4_session_copy_logits(s, v, VOCAB) == VOCAB && (f = fopen(path, "wb")) != NULL &&
                    fwrite(v, sizeof(float), VOCAB, f) == VOCAB;
    if (f && fclose(f) != 0) { free(v); return false; }
    if (!ok) fprintf(stderr, "mellum_trace: cannot write %s\n", path);
    free(v);
    return ok;
}

/* finite logits files at every listed position are the run's evidence the logits are real */
static bool report_logits(const char *prefix, unsigned pos) {
    char path[4096];
    snprintf(path, sizeof(path), "%s.p%u.logits.f32", prefix, pos);
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    float *v = malloc((size_t)VOCAB * sizeof(float));
    const bool read_ok = v && fread(v, sizeof(float), VOCAB, f) == VOCAB;
    fclose(f);
    if (!read_ok) { free(v); return false; }
    unsigned bad = 0, arg = 0;
    for (unsigned i = 0; i < VOCAB; i++) {
        if (!isfinite(v[i])) bad++;
        else if (!isfinite(v[arg]) || v[i] > v[arg]) arg = i;
    }
    printf("mellum_trace: position %u logits: %u non-finite of %d, trace argmax %u (%.6g)\n", pos, bad, VOCAB, arg,
           (double)v[arg]);
    free(v);
    return bad == 0;
}

/* MODE: prefill (0), decode (1), or prefill:N (N); -1 when it is none of them */
static int parse_mode(const char *mode) {
    if (!strcmp(mode, "prefill")) return 0;
    if (!strcmp(mode, "decode")) return 1;
    if (strncmp(mode, "prefill:", 8) != 0 || !isdigit((unsigned char)mode[8])) return -1;
    char *end = NULL;
    errno = 0;
    const long n = strtol(mode + 8, &end, 10);
    return errno == 0 && *end == '\0' && n >= 1 && n <= INT_MAX ? (int)n : -1;
}

int main(int argc, char **argv) {
    const int mode_sync = argc == 4 ? parse_mode(argv[3]) : -1;
    if (mode_sync < 0) {
        fprintf(stderr, "usage: mellum_trace MODEL.gguf TOKENS_FILE prefill|decode|prefill:N\n");
        return 2;
    }
    ds4_tokens toks = {0};
    if (!read_tokens(argv[2], &toks)) return 1;
    /* how many tokens the first sync takes; the rest are evaluated one by one */
    const int n_sync = mode_sync == 0 ? toks.len : mode_sync;
    if (n_sync > toks.len) {
        fprintf(stderr, "mellum_trace: %s syncs more than the %d tokens of %s\n", argv[3], toks.len, argv[2]);
        return 2;
    }
    const char *greedy_env = getenv("MELLUM_TRACE_GREEDY");
    const int greedy = greedy_env && greedy_env[0] ? atoi(greedy_env) : 0;
    if (greedy < 0 || greedy > 64) {
        fprintf(stderr, "mellum_trace: MELLUM_TRACE_GREEDY must be 0 to 64\n");
        return 2;
    }

    const char *prefix = getenv("DS4_MELLUM_TRACE");
    const char *at = getenv("DS4_MELLUM_TRACE_AT");
    unsigned pos[MAX_AT];
    int n_pos = 0;
    if (prefix && !prefix[0]) prefix = NULL;
    if (prefix) prefix = strdup(prefix);   /* unsetenv below must not pull the string out from under us */
    if (!prefix && at && at[0]) {
        fprintf(stderr, "mellum_trace: DS4_MELLUM_TRACE_AT needs DS4_MELLUM_TRACE=<prefix>\n");
        return 2;
    }
    if (prefix && !(at && at[0])) {
        /* an untraced run: the prefix only names the session logits file; the engine must not see it */
        unsetenv("DS4_MELLUM_TRACE");
    }
    /* MELLUM_TRACE_SESSION_AT (S6): in an untraced run, the session's logits after the forward that ends at each
     * listed position, written as <prefix>.p<pos>.logits.f32 by the driver (ds4_session_copy_logits), not by the
     * engine's trace; so a streamed run, which the engine will not trace, still gives per-position logits.
     * Positions inside the first sync are not reachable; prefill:N with N <= the first position, or decode. */
    const char *session_at = getenv("MELLUM_TRACE_SESSION_AT");
    unsigned spos[MAX_AT];
    int n_spos = 0;
    if (session_at && session_at[0]) {
        n_spos = parse_positions(session_at, spos);
        if (!prefix || (at && at[0]) || n_spos <= 0) {
            fprintf(stderr, "mellum_trace: MELLUM_TRACE_SESSION_AT needs DS4_MELLUM_TRACE=<prefix>, no "
                            "DS4_MELLUM_TRACE_AT, and a list of at most %d positions\n", MAX_AT);
            return 2;
        }
        for (int i = 0; i < n_spos; i++) {
            if (spos[i] >= (unsigned)toks.len || spos[i] + 1u < (unsigned)n_sync) {
                fprintf(stderr, "mellum_trace: session position %u is outside tokens %d..%d of %s\n", spos[i],
                        n_sync - 1, toks.len - 1, argv[2]);
                return 2;
            }
        }
    }
    if (prefix && at && at[0]) {
        n_pos = parse_positions(at, pos);
        if (n_pos <= 0) {
            fprintf(stderr, "mellum_trace: DS4_MELLUM_TRACE_AT=%s is not a list of at most %d positions\n", at, MAX_AT);
            return 2;
        }
        for (int i = 0; i < n_pos; i++) {
            if (pos[i] >= (unsigned)toks.len) {
                fprintf(stderr, "mellum_trace: trace position %u is not one of the %d tokens of %s\n", pos[i],
                        toks.len, argv[2]);
                return 1;
            }
        }
    }
    if (prefix) {
        char path[4096];
        for (int i = 0; i < n_pos; i++) {
            static const char *const what[] = {"hidden", "qk", "ffn_in", "logits"};
            for (int w = 0; w < 4; w++) {
                snprintf(path, sizeof(path), "%s.p%u.%s.f32", prefix, pos[i], what[w]);
                if (unlink(path) != 0 && errno != ENOENT) {
                    fprintf(stderr, "mellum_trace: cannot remove %s\n", path);
                    return 1;
                }
            }
        }
        snprintf(path, sizeof(path), "%s.session.logits.f32", prefix);
        if (unlink(path) != 0 && errno != ENOENT) {
            fprintf(stderr, "mellum_trace: cannot remove %s\n", path);
            return 1;
        }
    }
    const char *rep_env = getenv("MELLUM_TRACE_REPEAT");
    const int repeat = rep_env && rep_env[0] ? atoi(rep_env) : 1;
    if (repeat < 1 || repeat > 16) {
        fprintf(stderr, "mellum_trace: MELLUM_TRACE_REPEAT must be 1 to 16\n");
        return 2;
    }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_METAL;
    opt.context_size = mellum_trace_context(getenv("MELLUM_TRACE_CONTEXT"), toks.len);
    if (opt.context_size < 0) {
        fprintf(stderr, "mellum_trace: MELLUM_TRACE_CONTEXT must be decimal and >= tokens + 64\n");
        return 2;
    }
    uint64_t stream_bytes = 0;
    if (mellum_trace_stream_bytes(getenv("MELLUM_TRACE_STREAM_BYTES"), &stream_bytes) != 0) {
        fprintf(stderr, "mellum_trace: MELLUM_TRACE_STREAM_BYTES must be a decimal byte count of at least 1\n");
        return 2;
    }
    if (stream_bytes) {   /* S6: streamed experts, as on an 8 GB Mac */
        opt.ssd_streaming = true;
        opt.ssd_streaming_cache_bytes = stream_bytes;
    }
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) {
        fprintf(stderr, "mellum_trace: cannot open %s\n", argv[1]);
        return 1;
    }

    int rc = 0, argmax = -1;
    for (int r = 0; r < repeat && rc == 0; r++) {
        ds4_session *s = NULL;
        char err[512] = {0};
        if (ds4_session_create(&s, e, opt.context_size) != 0 || !s) {
            fprintf(stderr, "mellum_trace: cannot create a session\n");
            rc = 1;
            break;
        }
        const double t0 = now_sec();
        ds4_tokens first = {0};
        for (int i = 0; i < n_sync; i++) ds4_tokens_push(&first, toks.v[i]);
        if (ds4_session_sync(s, &first, err, sizeof(err)) != 0) {
            fprintf(stderr, "mellum_trace: sync failed: %s\n", err);
            rc = 1;
        }
        for (int i = n_sync - 1; rc == 0 && i < toks.len; i++) {
            if (i >= n_sync && ds4_session_eval(s, toks.v[i], err, sizeof(err)) != 0) {
                fprintf(stderr, "mellum_trace: eval of token %d (position %d) failed: %s\n", toks.v[i], i, err);
                rc = 1;
            }
            for (int k = 0; rc == 0 && k < n_spos; k++) {
                if (spos[k] != (unsigned)i) continue;
                if (!write_session_logits(s, prefix, i)) rc = 1;
            }
        }
        const double secs = now_sec() - t0;
        if (rc == 0) {
            argmax = ds4_session_argmax(s);
            float *logits = malloc((size_t)VOCAB * sizeof(float));
            unsigned bad = VOCAB;
            if (logits && ds4_session_copy_logits(s, logits, VOCAB) == VOCAB) {
                bad = 0;
                for (unsigned i = 0; i < VOCAB; i++) bad += !isfinite(logits[i]);
            }
            if (bad == 0 && prefix) {
                char path[4096];
                snprintf(path, sizeof(path), "%s.session.logits.f32", prefix);
                FILE *f = fopen(path, "wb");
                const bool wrote = f && fwrite(logits, sizeof(float), VOCAB, f) == VOCAB;
                if (f) fclose(f);
                if (!wrote) { fprintf(stderr, "mellum_trace: cannot write %s\n", path); rc = 1; }
            }
            free(logits);
            printf("mellum_trace: run %d of %d: %d tokens, %s, argmax %d, %u non-finite logits, %.3f s (%.1f tokens/s)\n",
                   r + 1, repeat, toks.len, argv[3], argmax, bad, secs, secs > 0.0 ? (double)toks.len / secs : 0.0);
            if (bad != 0) {
                fprintf(stderr, "mellum_trace: the session's logits are not finite\n");
                rc = 1;
            }
        }
        if (rc == 0 && greedy > 0 && r + 1 == repeat) {
            /* after the session logits are written: greedy tokens decoded from the cache this run built */
            printf("mellum_trace: greedy %d:", greedy);
            for (int k = 0; rc == 0 && k < greedy; k++) {
                const int tok = ds4_session_argmax(s);
                printf(" %d", tok);
                if (ds4_session_eval(s, tok, err, sizeof(err)) != 0) {
                    fprintf(stderr, "mellum_trace: greedy eval %d failed: %s\n", k, err);
                    rc = 1;
                }
            }
            printf("\n");
        }
        ds4_tokens_free(&first);
        ds4_session_free(s);
    }
    ds4_engine_close(e);
    if (rc != 0) {
        ds4_tokens_free(&toks);
        return rc;
    }
    if (rc == 0 && prefix) {
        char path[4096];
        struct stat st;
        snprintf(path, sizeof(path), "%s.session.logits.f32", prefix);
        if (stat(path, &st) != 0 || (uint64_t)st.st_size != (uint64_t)VOCAB * 4u) {
            fprintf(stderr, "mellum_trace: %s was not written with %llu bytes\n", path, (unsigned long long)VOCAB * 4ull);
            rc = 1;
        }
    }
    for (int i = 0; i < n_pos && rc == 0; i++) {
        if (!check_file(prefix, pos[i], "hidden", (uint64_t)N_LAYER * HIDDEN * 4u) ||
            !check_file(prefix, pos[i], "qk", (uint64_t)N_LAYER * QK * 4u) ||
            !check_file(prefix, pos[i], "ffn_in", (uint64_t)N_LAYER * HIDDEN * 4u) ||
            !check_file(prefix, pos[i], "logits", (uint64_t)VOCAB * 4u)) {
            rc = 1;
        }
    }
    for (int i = 0; i < n_pos && rc == 0; i++) {
        if (!report_logits(prefix, pos[i])) {
            fprintf(stderr, "mellum_trace: the logits at position %u are not finite\n", pos[i]);
            rc = 1;
        }
    }
    ds4_tokens_free(&toks);
    return rc;
}
