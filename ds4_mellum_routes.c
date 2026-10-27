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
