#include "ds4_mellum_stream.h"

#include <stdio.h>

uint32_t ds4_mellum_stream_slots(uint64_t cache_bytes) {
    const uint64_t slots = cache_bytes / DS4_MELLUM_EXPERT_BYTES;
    return slots > UINT32_MAX ? UINT32_MAX : (uint32_t)slots;
}

int ds4_mellum_stream_config(bool streaming, uint64_t cache_bytes, uint32_t *slots, char *err, size_t n) {
    if (!slots) return -1;
    *slots = 0;
    if (!streaming) return 0;
    const uint32_t got = ds4_mellum_stream_slots(cache_bytes);
    if (got == 0) {
        if (err && n) {
            snprintf(err, n, "Mellum --ssd-streaming needs --ssd-streaming-cache-experts of at least one "
                             "expert (%llu bytes)", (unsigned long long)DS4_MELLUM_EXPERT_BYTES);
        }
        return -1;
    }
    *slots = got;
    return 0;
}
