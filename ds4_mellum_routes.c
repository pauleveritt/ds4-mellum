#include "ds4_mellum_routes.h"

bool mellum_routes_layout(uint32_t cap, uint32_t rows, uint32_t layer,
                          mellum_routes_layout_info *out) {
    if (!out || !cap || cap > MELLUM_ROUTES_MAX_CAP || !rows || rows > cap ||
        layer >= MELLUM_ROUTES_LAYERS) return false;
    const uint64_t stride = (uint64_t)cap * MELLUM_ROUTES_TOP_K * 4u;
    *out = (mellum_routes_layout_info){MELLUM_ROUTES_LAYERS * stride, layer * stride,
                                     (uint64_t)rows * MELLUM_ROUTES_TOP_K * 4u};
    return true;
}

static bool routes_bytes(mellum_routes_writer *w, const void *data, size_t bytes) {
    if (w->state != 1) return false;
    if (w->sink(w->ctx, data, bytes) == bytes) return true;
    w->state = 0;
    return false;
}

static bool routes_word(mellum_routes_writer *w, uint64_t value, unsigned width) {
    unsigned char bytes[8];
    for (unsigned i = 0; i < width; i++) bytes[i] = (unsigned char)(value >> (i * 8));
    return routes_bytes(w, bytes, width);
}

bool mellum_routes_begin(mellum_routes_writer *w, uint32_t cap, mellum_routes_sink sink, void *ctx) {
    if (!w) return false;
    *w = (mellum_routes_writer){.sink = sink, .ctx = ctx, .cap = cap};
    if (!sink || !cap || cap > MELLUM_ROUTES_MAX_CAP) return false;
    w->state = 1;
    return routes_bytes(w, "MLRT0001", 8) && routes_word(w, 28, 4) && routes_word(w, 64, 4) &&
           routes_word(w, 8, 4) && routes_word(w, cap, 4);
}

bool mellum_routes_forward(mellum_routes_writer *w, uint64_t forward, uint64_t position,
                           uint32_t rows, uint32_t phase, uint32_t segment_id, uint32_t execution_id,
                           const uint32_t *tokens, size_t token_count,
                           const uint32_t *selected, size_t selected_count) {
    if (!w || w->state != 1) return false;
    if (!tokens || !selected || !rows || rows > w->cap || (phase != 1 && phase != 2) ||
        (phase == 2 && rows != 1) || token_count < rows || selected_count < 28u * w->cap * 8u ||
        forward != w->forwards || forward == UINT64_MAX || position > UINT64_MAX - rows ||
        w->positions > UINT64_MAX - rows || (forward && position != w->next_position)) {
        w->state = 0;
        return false;
    }
    /* Check the entire active forward before its prefix. Padding is never consumed. */
    for (unsigned layer = 0; layer < 28; layer++) {
        for (unsigned row = 0; row < rows; row++) {
            uint64_t seen = 0;
            for (unsigned slot = 0; slot < 8; slot++) {
                uint32_t id = selected[(layer * w->cap + row) * 8 + slot];
                if (id >= 64 || (seen & (UINT64_C(1) << id))) {
                    w->state = 0;
                    return false;
                }
                seen |= UINT64_C(1) << id;
            }
        }
    }
    if (!routes_bytes(w, "FWD1", 4) || !routes_word(w, forward, 8) || !routes_word(w, position, 8) ||
        !routes_word(w, rows, 4) || !routes_word(w, phase, 4) || !routes_word(w, segment_id, 4) ||
        !routes_word(w, execution_id, 4)) return false;
    for (unsigned row = 0; row < rows; row++)
        if (!routes_word(w, tokens[row], 4)) return false;
    for (unsigned layer = 0; layer < 28; layer++)
        for (unsigned i = 0; i < rows * 8u; i++)
            if (!routes_word(w, selected[layer * w->cap * 8u + i], 4)) return false;
    w->forwards++;
    w->positions += rows;
    w->next_position = position + rows;
    return true;
}

bool mellum_routes_finish(mellum_routes_writer *w) {
    if (!w || w->state != 1) return false;
    if (!w->forwards) { w->state = 0; return false; }
    if (!routes_bytes(w, "END1", 4) || !routes_word(w, w->forwards, 8) ||
        !routes_word(w, w->positions, 8)) return false;
    w->state = 2;
    return true;
}
