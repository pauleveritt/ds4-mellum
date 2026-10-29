/* mellum_llama_logits -- llama.cpp's logits for a Mellum token file, at the positions tests/mellum_trace traces,
 * in its file layout, so tools/mellum_logits_agree.py can compare the two engines (S6 phase B, item b).
 *
 *   tools/mellum_llama_logits MODEL.gguf TOKENS_FILE PREFIX POSITIONS
 *
 * TOKENS_FILE is tests/mellum_trace's format: whitespace-separated non-negative decimal ids, `#` to the end of
 * a line is a comment.  The ids are fed as they are: llama.cpp neither tokenizes nor applies a template.  They
 * run as one llama_decode of every token (a prefill), with logits on at each listed position and the last.
 * Writes PREFIX.p<pos>.logits.f32 (n_vocab little-endian f32) for each position, and PREFIX.session.logits.f32
 * for the last.  Exit 0; 1 when llama.cpp fails; 2 on bad usage, an unreadable file, an id outside the model's
 * vocabulary or a position past the last token. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

#define MAX_TOKENS 65536
#define MAX_POS 256

static int usage(void) {
    fprintf(stderr, "usage: mellum_llama_logits MODEL.gguf TOKENS_FILE PREFIX POSITIONS\n");
    return 2;
}

static int read_tokens(const char *path, llama_token *ids, int cap) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int n = 0, c;
    long v = -1;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n') {} }
        if (c >= '0' && c <= '9') {
            v = (v < 0 ? 0 : v) * 10 + (c - '0');
            if (v > 0x7fffffffL) { fclose(f); return -1; }
            continue;
        }
        if (v >= 0) {
            if (n == cap) { fclose(f); return -1; }
            ids[n++] = (llama_token)v;
            v = -1;
        }
        if (c != EOF && c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '#') { fclose(f); return -1; }
    }
    if (v >= 0 && n < cap) ids[n++] = (llama_token)v;
    fclose(f);
    return n;
}

static int write_row(const char *prefix, const char *what, const float *row, int n) {
    char path[4096];
    snprintf(path, sizeof(path), "%s.%s.logits.f32", prefix, what);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(row, sizeof(float), (size_t)n, f) != (size_t)n) {
        if (f) fclose(f);
        fprintf(stderr, "mellum_llama_logits: cannot write %s\n", path);
        return 0;
    }
    return fclose(f) == 0;
}

int main(int argc, char **argv) {
    if (argc != 5) return usage();
    const char *model_path = argv[1], *tokens_path = argv[2], *prefix = argv[3];
    FILE *probe = fopen(model_path, "rb");
    if (!probe) { fprintf(stderr, "mellum_llama_logits: cannot open %s\n", model_path); return 2; }
    fclose(probe);
    static llama_token ids[MAX_TOKENS];
    const int n = read_tokens(tokens_path, ids, MAX_TOKENS);
    if (n <= 0) { fprintf(stderr, "mellum_llama_logits: %s is not a token file\n", tokens_path); return 2; }
    int pos[MAX_POS], n_pos = 0;
    for (char *s = argv[4], *end; *s; s = *end == ',' ? end + 1 : end) {
        const long p = strtol(s, &end, 10);
        if (end == s || p < 0 || p >= n || n_pos == MAX_POS || (*end && *end != ',')) {
            fprintf(stderr, "mellum_llama_logits: bad position list '%s' for %d tokens\n", argv[4], n);
            return 2;
        }
        pos[n_pos++] = (int)p;
    }

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "mellum_llama_logits: llama.cpp could not load %s\n", model_path); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= n_vocab) {
            fprintf(stderr, "mellum_llama_logits: id %d at %d is outside the vocabulary of %d\n", ids[i], i, n_vocab);
            llama_model_free(model);
            return 2;
        }
    }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)((n + 255) / 256 * 256);
    cp.n_batch = (uint32_t)n;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "mellum_llama_logits: no context\n"); llama_model_free(model); return 1; }
    fprintf(stderr, "mellum_llama_logits: %d tokens, n_ctx %u, n_batch %u, n_ubatch %u, flash_attn %s, vocab %d\n", n,
            cp.n_ctx, cp.n_batch, cp.n_ubatch, llama_flash_attn_type_name(cp.flash_attn_type), n_vocab);

    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        bool want = i == n - 1;
        for (int k = 0; k < n_pos && !want; k++) want = pos[k] == i;
        batch.logits[i] = want;
    }
    batch.n_tokens = n;
    int rc = llama_decode(ctx, batch) == 0 ? 0 : 1;
    if (rc) fprintf(stderr, "mellum_llama_logits: llama_decode failed\n");
    for (int k = 0; !rc && k <= n_pos; k++) {
        const int at = k < n_pos ? pos[k] : n - 1;
        const float *row = llama_get_logits_ith(ctx, at);
        char what[32];
        if (k < n_pos) snprintf(what, sizeof(what), "p%d", at);
        else snprintf(what, sizeof(what), "session");
        if (!row || !write_row(prefix, what, row, n_vocab)) rc = 1;
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
