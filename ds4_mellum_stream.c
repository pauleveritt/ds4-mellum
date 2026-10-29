#include "ds4_mellum_stream.h"

#include <stdio.h>
#include <stdlib.h>

uint64_t ds4_mellum_expert_bytes(uint32_t expert_type) {
    return expert_type == 8u ? DS4_MELLUM_Q8_EXPERT_BYTES : expert_type == 39u ? DS4_MELLUM_MXFP4_EXPERT_BYTES : 0ull;
}

uint32_t ds4_mellum_stream_slots(uint64_t cache_bytes, uint64_t expert_bytes) {
    if (!expert_bytes) return 0;
    const uint64_t slots = cache_bytes / expert_bytes;
    return slots > UINT32_MAX ? UINT32_MAX : (uint32_t)slots;
}

int ds4_mellum_stream_config(bool streaming, uint64_t cache_bytes, uint64_t expert_bytes, uint32_t *slots,
                             char *err, size_t n) {
    if (!slots) return -1;
    *slots = 0;
    if (!streaming) return 0;
    if (!expert_bytes) {
        if (err && n) snprintf(err, n, "Mellum --ssd-streaming: unsupported Mellum expert type");
        return -1;
    }
    const uint32_t got = ds4_mellum_stream_slots(cache_bytes, expert_bytes);
    if (got < DS4_MELLUM_STREAM_MIN_SLOTS) {
        if (err && n) {
            snprintf(err, n, "Mellum --ssd-streaming needs --ssd-streaming-cache-experts of at least %u experts "
                             "(%llu bytes), one layer's 64: got %u", DS4_MELLUM_STREAM_MIN_SLOTS,
                     (unsigned long long)DS4_MELLUM_STREAM_MIN_SLOTS * expert_bytes, got);
        }
        return -1;
    }
    *slots = got;
    return 0;
}

int ds4_mellum_stream_decode(const ds4_mellum_stream_ops *ops, uint32_t n_layer, uint32_t *stops) {
    if (!ops || !ops->encode || !ops->finish || !ops->load || !ops->cancelled || !stops || n_layer == 0 ||
        (ops->load_begin && !ops->discard)) {
        return DS4_MELLUM_STREAM_PROTOCOL;
    }
    *stops = 0;
    uint32_t layer = 0;
    uint64_t loaded_here = 0;  /* experts loaded at `layer` since the first stop there */
    bool resume = false, encoded = false;
    for (;;) {
        bool stopped = false;
        ds4_mellum_stream_stop stop;
        if ((!encoded && !ops->encode(ops->ctx, layer, resume)) || !ops->finish(ops->ctx, &stopped, &stop)) {
            return DS4_MELLUM_STREAM_GPU;
        }
        encoded = false;
        if (!stopped) return DS4_MELLUM_STREAM_OK;
        if (*stops >= n_layer * 8u || stop.layer >= n_layer || stop.n_missing == 0 || stop.n_missing > 8u ||
            (resume && stop.layer < layer)) {
            return DS4_MELLUM_STREAM_PROTOCOL;
        }
        uint64_t missing = 0;
        for (uint32_t i = 0; i < stop.n_missing; i++) {
            if (stop.missing[i] < 0 || stop.missing[i] >= 64) return DS4_MELLUM_STREAM_PROTOCOL;
            missing |= 1ull << stop.missing[i];
        }
        if (resume && stop.layer == layer) {
            if (missing & loaded_here) return DS4_MELLUM_STREAM_PROTOCOL;
        } else {
            loaded_here = 0;
        }
        (*stops)++;
        if (ops->load_begin) {
            /* the reads run while the resumed pass is encoded; the pass is committed only once they finish */
            if (!ops->load_begin(ops->ctx, stop.layer, stop.missing, stop.n_missing)) return DS4_MELLUM_STREAM_LOAD;
            if (!ops->encode(ops->ctx, stop.layer, true)) {
                ops->discard(ops->ctx);
                return DS4_MELLUM_STREAM_GPU;
            }
            encoded = true;
            if (!ops->load(ops->ctx, stop.layer, stop.missing, stop.n_missing)) {
                ops->discard(ops->ctx);
                return DS4_MELLUM_STREAM_LOAD;
            }
            if (ops->cancelled(ops->ctx)) {
                ops->discard(ops->ctx);
                return DS4_MELLUM_STREAM_CANCELLED;
            }
        } else {
            if (!ops->load(ops->ctx, stop.layer, stop.missing, stop.n_missing)) return DS4_MELLUM_STREAM_LOAD;
            if (ops->cancelled(ops->ctx)) return DS4_MELLUM_STREAM_CANCELLED;
        }
        loaded_here |= missing;
        layer = stop.layer;
        resume = true;
    }
}

uint32_t ds4_mellum_stream_unique(const int32_t *selected, uint32_t rows, uint32_t top_k, int32_t out[64]) {
    if (!selected || !out) return UINT32_MAX;
    uint64_t seen = 0;
    for (uint64_t i = 0; i < (uint64_t)rows * top_k; i++) {
        if (selected[i] < 0 || selected[i] >= 64) return UINT32_MAX;
        seen |= 1ull << selected[i];
    }
    uint32_t n = 0;
    for (int32_t e = 0; e < 64; e++) if (seen & (1ull << e)) out[n++] = e;
    return n;
}

uint32_t ds4_mellum_stream_prefill_group(const char *env) {
    if (!env) return DS4_MELLUM_STREAM_GROUP_DEFAULT;
    if (env[0] < '0' || env[0] > '9') return 0;
    char *end = NULL;
    const unsigned long k = strtoul(env, &end, 10);
    return *end == 0 && k >= 1ul && k <= DS4_MELLUM_STREAM_GROUP_MAX ? (uint32_t)k : 0u;
}

uint32_t ds4_mellum_stream_group_rows(uint32_t remaining, uint32_t cap, uint32_t k) {
    if (!remaining || !cap || !k) return 0;
    const uint64_t most = (uint64_t)k * cap;
    uint32_t rows = most < remaining ? (uint32_t)most : remaining;
    if (rows == remaining && rows > 1u && rows % cap == 1u) rows--;
    return rows;
}
