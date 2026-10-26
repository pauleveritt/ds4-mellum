/* A sparse Mellum 2.1 GGUF for model-free tests: the real key set and the real
 * tensor directory (every tensor at offset 0, so the file is a few hundred
 * megabytes of holes), and a tiny vocabulary with 2.1's special tokens at their
 * real ids followed by the 256 GPT-2 byte pieces and no merges, so text
 * tokenizes to one id per byte (byte c is id 35 + c).  A mutation changes one
 * thing, for a refusal test.  Self-contained: it does not include ds4.c, so the
 * GGUF constants are restated here. */
#ifndef DS4_TESTS_MELLUM_FIXTURE_H
#define DS4_TESTS_MELLUM_FIXTURE_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    MF_GGUF_MAGIC = 0x46554747,   /* "GGUF" */
    MF_UINT32 = 4, MF_INT32 = 5, MF_FLOAT32 = 6, MF_BOOL = 7, MF_STRING = 8, MF_ARRAY = 9,
    MF_F32 = 0, MF_Q8_0 = 8,
};

typedef enum {
    MF_NONE = 0,
} mf_mutation;

typedef struct { char *p; size_t n, cap; } mf_buf;

static void mf_put(mf_buf *b, const void *d, size_t n) {
    if (b->n + n > b->cap) {
        b->cap = (b->n + n) * 2 + 64;
        b->p = realloc(b->p, b->cap);
        assert(b->p);
    }
    memcpy(b->p + b->n, d, n);
    b->n += n;
}
static void mf_u32(mf_buf *b, uint32_t v) { mf_put(b, &v, 4); }
static void mf_u64(mf_buf *b, uint64_t v) { mf_put(b, &v, 8); }
static void mf_str(mf_buf *b, const char *s) { mf_u64(b, strlen(s)); mf_put(b, s, strlen(s)); }

typedef struct { mf_buf kv, tn; uint64_t n_kv, n_tensor; } mf_gguf;

static void mf_kv_u32(mf_gguf *g, const char *k, uint32_t v) {
    mf_str(&g->kv, k); mf_u32(&g->kv, MF_UINT32); mf_u32(&g->kv, v); g->n_kv++;
}
static void mf_kv_f32(mf_gguf *g, const char *k, float v) {
    mf_str(&g->kv, k); mf_u32(&g->kv, MF_FLOAT32); mf_put(&g->kv, &v, 4); g->n_kv++;
}
static void mf_kv_str(mf_gguf *g, const char *k, const char *v) {
    mf_str(&g->kv, k); mf_u32(&g->kv, MF_STRING); mf_str(&g->kv, v); g->n_kv++;
}
static void mf_kv_strs(mf_gguf *g, const char *k, const char *const *v, uint64_t n) {
    mf_str(&g->kv, k); mf_u32(&g->kv, MF_ARRAY); mf_u32(&g->kv, MF_STRING); mf_u64(&g->kv, n);
    for (uint64_t i = 0; i < n; i++) mf_str(&g->kv, v[i]);
    g->n_kv++;
}
static void mf_kv_i32s(mf_gguf *g, const char *k, uint32_t etype, const int32_t *v, uint64_t n) {
    mf_str(&g->kv, k); mf_u32(&g->kv, MF_ARRAY); mf_u32(&g->kv, etype); mf_u64(&g->kv, n);
    for (uint64_t i = 0; i < n; i++) mf_put(&g->kv, &v[i], 4);
    g->n_kv++;
}
static void mf_kv_bools(mf_gguf *g, const char *k, const bool *v, uint64_t n) {
    mf_str(&g->kv, k); mf_u32(&g->kv, MF_ARRAY); mf_u32(&g->kv, MF_BOOL); mf_u64(&g->kv, n);
    for (uint64_t i = 0; i < n; i++) { uint8_t x = v[i]; mf_put(&g->kv, &x, 1); }
    g->n_kv++;
}

static void mf_tensor(mf_gguf *g, const char *name, uint32_t type, uint32_t ndim,
                      uint64_t d0, uint64_t d1, uint64_t d2) {
    mf_str(&g->tn, name); mf_u32(&g->tn, ndim);
    mf_u64(&g->tn, d0);
    if (ndim > 1) mf_u64(&g->tn, d1);
    if (ndim > 2) mf_u64(&g->tn, d2);
    mf_u32(&g->tn, type); mf_u64(&g->tn, 0);
    g->n_tensor++;
}

static void mf_layer_tensor(mf_gguf *g, uint32_t il, const char *suffix, uint32_t type, uint32_t ndim,
                            uint64_t d0, uint64_t d1, uint64_t d2) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
    mf_tensor(g, name, type, ndim, d0, d1, d2);
}

/* 2.1's hyperparameters, as its GGUF carries them. */
static void mf_keys(mf_gguf *g, mf_mutation mu) {
    (void)mu;
    bool pattern[28];
    for (int i = 0; i < 28; i++) pattern[i] = (i % 4) != 3;
    mf_kv_str(g, "general.architecture", "mellum");
    mf_kv_u32(g, "mellum.block_count", 28);
    mf_kv_u32(g, "mellum.context_length", 131072);
    mf_kv_u32(g, "mellum.embedding_length", 2304);
    mf_kv_u32(g, "mellum.feed_forward_length", 7168);
    mf_kv_u32(g, "mellum.attention.head_count", 32);
    mf_kv_u32(g, "mellum.attention.head_count_kv", 4);
    mf_kv_str(g, "mellum.rope.scaling.type", "yarn");
    mf_kv_f32(g, "mellum.rope.scaling.factor", 16.0f);
    mf_kv_u32(g, "mellum.rope.scaling.original_context_length", 8192);
    mf_kv_f32(g, "mellum.rope.scaling.yarn_attn_factor", 1.2772588729858398f);
    mf_kv_f32(g, "mellum.rope.scaling.yarn_beta_fast", 32.0f);
    mf_kv_f32(g, "mellum.rope.scaling.yarn_beta_slow", 1.0f);
    mf_kv_f32(g, "mellum.rope.freq_base", 500000.0f);
    mf_kv_f32(g, "mellum.rope.freq_base_swa", 500000.0f);
    mf_kv_f32(g, "mellum.attention.layer_norm_rms_epsilon", 1.0e-6f);
    mf_kv_u32(g, "mellum.expert_count", 64);
    mf_kv_u32(g, "mellum.expert_used_count", 8);
    mf_kv_u32(g, "mellum.attention.key_length", 128);
    mf_kv_u32(g, "mellum.attention.value_length", 128);
    mf_kv_u32(g, "mellum.expert_feed_forward_length", 896);
    mf_kv_u32(g, "mellum.attention.sliding_window", 1024);
    mf_kv_bools(g, "mellum.attention.sliding_window_pattern", pattern, 28);
}

/* The special ids at their real places and spellings (2.1's tokenizer_config.json),
 * then the 256 byte-level pieces. */
static void mf_vocab(mf_gguf *g, mf_mutation mu) {
    (void)mu;
    static char fill[35][24], bytes[256][8];
    const char *tok[35 + 256];
    const int ntok = 35 + 256;
    for (int i = 0; i < 35; i++) { snprintf(fill[i], sizeof(fill[i]), "<fill_%d>", i); tok[i] = fill[i]; }
    tok[0] = "<|endoftext|>"; tok[1] = "<assistant>"; tok[2] = "</assistant>";
    tok[3] = "<commit_before>"; tok[4] = "<commit_after>"; tok[5] = "<commit_msg>";
    tok[6] = "<empty_output>"; tok[7] = "<filename>"; tok[8] = "<fim_prefix>";
    tok[9] = "<fim_middle>"; tok[10] = "<fim_suffix>"; tok[11] = "<fim_pad>";
    tok[12] = "<gh_stars>"; tok[13] = "<issue_start>"; tok[14] = "<issue_comment>";
    tok[15] = "<issue_closed>"; tok[16] = "<jupyter_text>"; tok[17] = "<jupyter_code>";
    tok[18] = "<jupyter_start>"; tok[19] = "<jupyter_output>"; tok[20] = "<reponame>";
    tok[21] = "<system>"; tok[22] = "</system>"; tok[23] = "<think>"; tok[24] = "</think>";
    tok[25] = "<user>"; tok[26] = "</user>"; tok[27] = "<|im_start|>"; tok[28] = "<|im_end|>";
    tok[29] = "<tool_call>"; tok[30] = "</tool_call>";
    tok[31] = "<tool_response>"; tok[32] = "</tool_response>";
    tok[33] = "<|extra_token_7|>"; tok[34] = "<|extra_token_8|>";
    int extra = 0;
    for (int c = 0; c < 256; c++) {
        const bool direct = (c >= 33 && c <= 126) || (c >= 161 && c <= 172) || (c >= 174);
        const int cp = direct ? c : 256 + extra++;
        if (cp < 0x80) { bytes[c][0] = (char)cp; bytes[c][1] = 0; }
        else { bytes[c][0] = (char)(0xC0 | (cp >> 6)); bytes[c][1] = (char)(0x80 | (cp & 0x3F)); bytes[c][2] = 0; }
        tok[35 + c] = bytes[c];
    }
    mf_kv_str(g, "tokenizer.ggml.model", "gpt2");
    mf_kv_str(g, "tokenizer.ggml.pre", "mellum2");
    mf_kv_strs(g, "tokenizer.ggml.tokens", tok, (uint64_t)ntok);
    mf_kv_strs(g, "tokenizer.ggml.merges", NULL, 0);
    /* as 2.1's file has them: 0-22, 25-28, 33, 34 CONTROL (3); 23, 24, 29-32
     * USER_DEFINED (4); the byte pieces NORMAL (1) */
    static int32_t types[35 + 256];
    for (int i = 0; i < 35 + 256; i++) types[i] = i >= 35 ? 1 : 3;
    types[23] = types[24] = types[29] = types[30] = types[31] = types[32] = 4;
    mf_kv_i32s(g, "tokenizer.ggml.token_type", MF_INT32, types, (uint64_t)ntok);
    mf_kv_u32(g, "tokenizer.ggml.bos_token_id", 0);
    mf_kv_u32(g, "tokenizer.ggml.eos_token_id", 28);
    mf_kv_u32(g, "tokenizer.ggml.unknown_token_id", 0);
    mf_kv_u32(g, "tokenizer.ggml.padding_token_id", 0);
}

/* 2.1's tensor directory (Q8_0 projections and experts, F32 norms and router). */
static void mf_tensors(mf_gguf *g, mf_mutation mu) {
    (void)mu;
    const uint32_t Q8 = MF_Q8_0, F32 = MF_F32;
    mf_tensor(g, "output.weight", Q8, 2, 2304, 98304, 0);
    mf_tensor(g, "output_norm.weight", F32, 1, 2304, 0, 0);
    mf_tensor(g, "token_embd.weight", Q8, 2, 2304, 98304, 0);
    for (uint32_t il = 0; il < 28; il++) {
        mf_layer_tensor(g, il, "attn_k.weight", Q8, 2, 2304, 512, 0);
        mf_layer_tensor(g, il, "attn_k_norm.weight", F32, 1, 128, 0, 0);
        mf_layer_tensor(g, il, "attn_norm.weight", F32, 1, 2304, 0, 0);
        mf_layer_tensor(g, il, "attn_output.weight", Q8, 2, 4096, 2304, 0);
        mf_layer_tensor(g, il, "attn_q.weight", Q8, 2, 2304, 4096, 0);
        mf_layer_tensor(g, il, "attn_q_norm.weight", F32, 1, 128, 0, 0);
        mf_layer_tensor(g, il, "attn_v.weight", Q8, 2, 2304, 512, 0);
        mf_layer_tensor(g, il, "ffn_down_exps.weight", Q8, 3, 896, 2304, 64);
        mf_layer_tensor(g, il, "ffn_gate_exps.weight", Q8, 3, 2304, 896, 64);
        mf_layer_tensor(g, il, "ffn_gate_inp.weight", F32, 2, 2304, 64, 0);
        mf_layer_tensor(g, il, "ffn_norm.weight", F32, 1, 2304, 0, 0);
        mf_layer_tensor(g, il, "ffn_up_exps.weight", Q8, 3, 2304, 896, 64);
    }
}

/* Write the fixture to a fresh temporary file; path receives its name (the caller unlinks). */
static void mellum_fixture_write(mf_mutation mu, char *path, size_t path_cap) {
    mf_gguf g;
    memset(&g, 0, sizeof(g));
    mf_keys(&g, mu);
    mf_vocab(&g, mu);
    mf_tensors(&g, mu);

    mf_buf head = {0};
    mf_u32(&head, MF_GGUF_MAGIC);
    mf_u32(&head, 3);
    mf_u64(&head, g.n_tensor);
    mf_u64(&head, g.n_kv);
    mf_put(&head, g.kv.p, g.kv.n);
    mf_put(&head, g.tn.p, g.tn.n);
    const uint64_t data_pos = (head.n + 31u) & ~(uint64_t)31u;
    const uint64_t file_size = data_pos + 260ull * 1000 * 1000;   /* > output.weight, 240 MB */

    const char *tmp = getenv("TMPDIR");
    snprintf(path, path_cap, "%s/mellum-fixture.XXXXXX", tmp && tmp[0] ? tmp : "/tmp");
    const int fd = mkstemp(path);
    assert(fd >= 0);
    assert(write(fd, head.p, head.n) == (ssize_t)head.n);
    assert(ftruncate(fd, (off_t)file_size) == 0);
    assert(close(fd) == 0);
    free(g.kv.p); free(g.tn.p); free(head.p);
}

#endif
