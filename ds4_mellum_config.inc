/* Mellum 2.1, included by ds4.c where this code sits: not a separate translation unit. */

/* Refuse a Mellum key at open, naming it (config_expect_* name theirs too). */
static void config_refuse_mellum_key(const char *key, const char *why) {
    fprintf(stderr, "ds4: %s: %s\n", key, why);
    exit(1);
}

static void config_mellum_range(const char *key, uint32_t v, uint32_t lo, uint32_t hi) {
    if (v >= lo && v <= hi) return;
    fprintf(stderr, "ds4: %s=%u is outside what the Mellum kernels support (%u..%u)\n", key, v, lo, hi);
    exit(1);
}

static float config_mellum_positive_f32(const ds4_model *m, const char *key) {
    const float v = required_f32(m, key);
    if (!(v > 0.0f) || !isfinite(v)) config_refuse_mellum_key(key, "must be a positive finite number");
    return v;
}

/* Mellum 2.1 (arch `mellum`).  Widths come from the GGUF with range checks;
 * what the Metal kernels need as a constant is pinned to the validated value
 * (window 1024, 64 experts with 8 used, 32 heads over 4 KV heads of width 128,
 * YaRN 16 over 8192).
 * The layer count is read, not pinned.  The dense `feed_forward_length`
 * (7168) is deliberately not read: every Mellum layer is a routed MoE, so
 * n_ff_dense stays 0.  Never calls qwen4_rope_configure: its 32-pair table
 * cannot hold Mellum's 64 rotary pairs, and with DS4_QWEN4_YARN_FACTOR set it
 * would print a misleading "Qwen3.8 YaRN" line. */
static void config_validate_mellum_model(const ds4_model *m) {
    g_ds4_shape = DS4_SHAPE_MELLUM;
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
    memset(&g_ds4_qwen4_ple, 0, sizeof(g_ds4_qwen4_ple));
    /* the Qwen rope state belongs to qwen4_rope_configure, which Mellum never
     * calls; a Qwen model opened earlier in this process must not leave it behind */
    memset(g_qwen4_rope_freq, 0, sizeof(g_qwen4_rope_freq));
    g_qwen4_rope_mscale = 1.0f;
    g_qwen4_native_ctx = 0;
    g_qwen4_rope_yarn = false;

    const uint32_t n_block = required_u32(m, "mellum.block_count");
    config_mellum_range("mellum.block_count", n_block, 1u, DS4_MAX_LAYER);
    g_ds4_shape.n_layer = n_block;
    g_ds4_shape.n_embd = required_u32(m, "mellum.embedding_length");
    config_mellum_range("mellum.embedding_length", g_ds4_shape.n_embd, 128u, DS4_MAX_EMBD);
    if (g_ds4_shape.n_embd % 128u) {
        config_refuse_mellum_key("mellum.embedding_length", "must be a multiple of 128 (the MoE kernels' rows)");
    }
    /* the sliding attention kernel is written for 32 query heads over 4 KV heads */
    config_expect_u32("mellum.attention.head_count", required_u32(m, "mellum.attention.head_count"), 32u);
    config_expect_u32("mellum.attention.head_count_kv", required_u32(m, "mellum.attention.head_count_kv"), 4u);
    g_ds4_shape.n_head = 32u;
    g_ds4_shape.n_head_kv = 4u;
    config_expect_u32("mellum.attention.key_length", required_u32(m, "mellum.attention.key_length"), 128u);
    config_expect_u32("mellum.attention.value_length", required_u32(m, "mellum.attention.value_length"), 128u);
    g_ds4_shape.n_head_dim = 128u;
    g_ds4_shape.n_value_dim = 128u;
    g_ds4_shape.n_rot = 128u;               /* rope covers the whole head: no rope.dimension_count key */

    config_expect_u32("mellum.expert_count", required_u32(m, "mellum.expert_count"), 64u);
    config_expect_u32("mellum.expert_used_count", required_u32(m, "mellum.expert_used_count"), 8u);
    g_ds4_shape.n_expert = 64u;
    g_ds4_shape.n_expert_used = 8u;
    g_ds4_shape.n_ff_exp = required_u32(m, "mellum.expert_feed_forward_length");
    config_mellum_range("mellum.expert_feed_forward_length", g_ds4_shape.n_ff_exp, 32u, DS4_MAX_FF_EXP);
    if (g_ds4_shape.n_ff_exp % 32u) {
        config_refuse_mellum_key("mellum.expert_feed_forward_length", "must be a multiple of 32 (32-value quantization blocks)");
    }

    config_expect_u32("mellum.attention.sliding_window", required_u32(m, "mellum.attention.sliding_window"), 1024u);
    g_ds4_shape.n_swa = 1024u;
    {
        const char *key = "mellum.attention.sliding_window_pattern";
        ds4_array_ref arr = {0};
        if (!model_get_array(m, key, &arr) || arr.type != GGUF_VALUE_BOOL || arr.len != n_block) {
            config_refuse_mellum_key(key, "must be a bool array with one entry per block");
        }
        ds4_cursor c = cursor_at(m, arr.data_pos);
        for (uint32_t il = 0; il < n_block; il++) {
            uint8_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if ((v != 0) != ds4_mellum_layer_is_sliding(il)) {
                fprintf(stderr, "ds4: %s: layer %u is %s, but Mellum slides every layer except il %% 4 == 3\n",
                        key, il, v ? "sliding" : "full");
                exit(1);
            }
        }
    }

    {
        const char *key = "mellum.rope.scaling.type";
        ds4_str type = {0};
        if (!model_get_string(m, key, &type) || !ds4_streq(type, "yarn")) {
            config_refuse_mellum_key(key, "must be \"yarn\"");
        }
    }
    config_expect_f32("mellum.rope.scaling.factor", required_f32(m, "mellum.rope.scaling.factor"), 16.0f);
    config_expect_u64("mellum.rope.scaling.original_context_length",
                      required_u64_compat(m, "mellum.rope.scaling.original_context_length"), 8192u);
    /* 1 + 0.1 ln(16): checked here, and not applied, because the RoPE host
     * computes the same magnitude from the factor (see the shape field) */
    config_expect_f32("mellum.rope.scaling.yarn_attn_factor",
                      required_f32(m, "mellum.rope.scaling.yarn_attn_factor"), 1.2772589f);
    config_expect_f32("mellum.rope.scaling.yarn_beta_fast", required_f32(m, "mellum.rope.scaling.yarn_beta_fast"), 32.0f);
    config_expect_f32("mellum.rope.scaling.yarn_beta_slow", required_f32(m, "mellum.rope.scaling.yarn_beta_slow"), 1.0f);
    g_ds4_shape.rope_scale_factor = 16.0f;
    g_ds4_shape.rope_orig_ctx = 8192u;
    g_ds4_shape.rope_yarn_attn_factor = 1.2772589f;
    g_ds4_shape.rope_yarn_beta_fast = 32.0f;
    g_ds4_shape.rope_yarn_beta_slow = 1.0f;
    g_ds4_shape.rope_freq_base = config_mellum_positive_f32(m, "mellum.rope.freq_base");
    g_ds4_shape.rope_freq_base_swa = config_mellum_positive_f32(m, "mellum.rope.freq_base_swa");
    g_ds4_shape.rms_eps = config_mellum_positive_f32(m, "mellum.attention.layer_norm_rms_epsilon");
    if (required_u64_compat(m, "mellum.context_length") < DS4_ROPE_ORIG_CTX) {
        config_refuse_mellum_key("mellum.context_length", "is shorter than the YaRN original context");
    }

    const ds4_tensor *embd = model_find_tensor(m, "token_embd.weight");
    if (!embd || embd->ndim != 2 || embd->dim[1] == 0 || embd->dim[1] > DS4_MAX_VOCAB) {
        ds4_die("Mellum token_embd.weight is missing or its vocabulary exceeds the compiled limit");
    }
    g_ds4_shape.n_vocab = (uint32_t)embd->dim[1];

    /* Every Mellum layer is routed-only; a shared-expert tensor means a different model. */
    static const char *const shexp[] = {
        "blk.%u.ffn_gate_inp_shexp.weight", "blk.%u.ffn_gate_shexp.weight",
        "blk.%u.ffn_up_shexp.weight", "blk.%u.ffn_down_shexp.weight",
    };
    for (uint32_t il = 0; il < n_block; il++) {
        for (size_t i = 0; i < sizeof(shexp) / sizeof(shexp[0]); i++) {
            char name[96];
            snprintf(name, sizeof(name), shexp[i], il);
            if (model_find_tensor(m, name)) {
                fprintf(stderr, "ds4: Mellum has no shared expert, but the GGUF carries tensor %s\n", name);
                exit(1);
            }
        }
    }
}
