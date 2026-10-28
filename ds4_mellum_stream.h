/* S4 Mellum expert streaming: the host-pure pieces, with no GPU or model calls, so CPU tests can pin
 * them.  See docs/superpowers/specs/2026-10-28-mellum-s4-cached-execution-design.md. */
#ifndef DS4_MELLUM_STREAM_H
#define DS4_MELLUM_STREAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* One routed Q8_0 expert: gate, up and down together. */
#define DS4_MELLUM_EXPERT_BYTES 6580224ull

/* Slots a cache of cache_bytes holds, rounded down; 0 below one expert. */
uint32_t ds4_mellum_stream_slots(uint64_t cache_bytes);

/* Validates a streaming request.  Off: *slots = 0, returns 0.  On: *slots from cache_bytes, or -1 with
 * a message in err when the cache holds no expert.  The whole budget is slots; prefill shares them. */
int ds4_mellum_stream_config(bool streaming, uint64_t cache_bytes, uint32_t *slots, char *err, size_t n);

#endif
