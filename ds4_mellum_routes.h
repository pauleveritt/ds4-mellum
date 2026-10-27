#ifndef DS4_MELLUM_ROUTES_H
#define DS4_MELLUM_ROUTES_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum { MELLUM_ROUTES_LAYERS = 28, MELLUM_ROUTES_EXPERTS = 64,
       MELLUM_ROUTES_TOP_K = 8, MELLUM_ROUTES_MAX_CAP = 1024 };

typedef struct {
    uint64_t buffer_bytes, layer_offset, copy_bytes;
} mellum_routes_layout_info;

bool mellum_routes_layout(uint32_t cap, uint32_t rows, uint32_t layer,
                          mellum_routes_layout_info *out);

/* The caller owns the sink and buffers. Begin only on a fresh/unowned writer. */
typedef size_t (*mellum_routes_sink)(void *ctx, const void *data, size_t bytes);
typedef struct {
    mellum_routes_sink sink;
    void *ctx;
    uint64_t forwards, positions, next_position;
    uint32_t cap, state; /* 0 failed/uninitialized, 1 active, 2 finished */
} mellum_routes_writer;

bool mellum_routes_begin(mellum_routes_writer *w, uint32_t cap, mellum_routes_sink sink, void *ctx);
bool mellum_routes_forward(mellum_routes_writer *w, uint64_t forward, uint64_t position,
                           uint32_t rows, uint32_t phase, uint32_t segment_id, uint32_t execution_id,
                           const uint32_t *tokens, size_t token_count,
                           const uint32_t *selected, size_t selected_count);
#endif
