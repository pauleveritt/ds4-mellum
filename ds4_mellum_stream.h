/* S4 Mellum expert streaming: the host-pure pieces, with no GPU or model calls, so CPU tests can pin
 * them.  See docs/superpowers/specs/2026-10-28-mellum-s4-cached-execution-design.md. */
#ifndef DS4_MELLUM_STREAM_H
#define DS4_MELLUM_STREAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* One routed Q8_0 expert: gate, up and down together. */
#define DS4_MELLUM_EXPERT_BYTES 6580224ull

/* The smallest cache: a prefill chunk can need all 64 of one layer's experts resident at once. */
#define DS4_MELLUM_STREAM_MIN_SLOTS 64u

/* Slots a cache of cache_bytes holds, rounded down; 0 below one expert. */
uint32_t ds4_mellum_stream_slots(uint64_t cache_bytes);

/* Validates a streaming request.  Off: *slots = 0, returns 0.  On: *slots from cache_bytes, or -1 with
 * a message in err when the cache holds fewer than DS4_MELLUM_STREAM_MIN_SLOTS experts.  The whole budget is slots; prefill shares them. */
int ds4_mellum_stream_config(bool streaming, uint64_t cache_bytes, uint32_t *slots, char *err, size_t n);

/* Decode stop/resume.  A token runs as one command buffer that stops at its first layer with a missing
 * expert; the host loads that layer's missing experts and resumes at the layer's expert compute (its
 * attention and KV write already ran).  Mellum's expert ids are 0..63. */
typedef struct { uint32_t layer; uint32_t n_missing; int32_t missing[8]; } ds4_mellum_stream_stop;
typedef struct ds4_mellum_stream_ops {
    void *ctx;
    bool (*encode)(void *ctx, uint32_t from_layer, bool resume_at_experts);
    bool (*finish)(void *ctx, bool *stopped, ds4_mellum_stream_stop *stop);  /* commit, wait, read the gate */
    bool (*load)(void *ctx, uint32_t layer, const int32_t *ids, uint32_t n);  /* returns once resident */
    bool (*cancelled)(void *ctx);
} ds4_mellum_stream_ops;
enum { DS4_MELLUM_STREAM_OK = 0, DS4_MELLUM_STREAM_LOAD = -1,
       DS4_MELLUM_STREAM_PROTOCOL = -2, DS4_MELLUM_STREAM_CANCELLED = -3, DS4_MELLUM_STREAM_GPU = -4 };

/* Runs one decode token over n_layer layers; *stops counts the stops taken.  A stop must not come before
 * the previous one, and a stop at the same layer must miss none of the experts just loaded there; a
 * token takes at most n_layer * 8 stops. */
int ds4_mellum_stream_decode(const ds4_mellum_stream_ops *ops, uint32_t n_layer, uint32_t *stops);

/* Streamed prefill: the distinct experts `rows` rows of top_k ids chose, sorted into out (at most 64).  Returns
 * the count, or UINT32_MAX for an id outside 0..63. */
uint32_t ds4_mellum_stream_unique(const int32_t *selected, uint32_t rows, uint32_t top_k, int32_t out[64]);

/* S5 grouped layer-major prefill: a streamed sync runs its suffix in groups of K chunks, every layer over
 * the group's chunks before the next layer.  K comes from DS4_MELLUM_STREAM_PREFILL_GROUP. */
#define DS4_MELLUM_STREAM_GROUP_DEFAULT 8u
#define DS4_MELLUM_STREAM_GROUP_MAX 64u

/* K from the setting's text: NULL gives the default; a decimal 1..64 gives itself; anything else 0 (refuse). */
uint32_t ds4_mellum_stream_prefill_group(const char *env);

/* Rows for the next group: min(k * cap, remaining), minus the last row when that would take the whole
 * suffix and end in a one-row chunk (a one-row chunk runs on the decode path); 0 when any argument is 0. */
uint32_t ds4_mellum_stream_group_rows(uint32_t remaining, uint32_t cap, uint32_t k);

#endif
