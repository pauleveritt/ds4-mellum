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
#endif
