#ifndef DS4_MELLUM_CAPTURE_STATE_H
#define DS4_MELLUM_CAPTURE_STATE_H
#include "ds4_mellum_routes.h"
#include "ds4_mellum_profile.h"
enum { MELLUM_CAPTURE_FAILED, MELLUM_CAPTURE_READY, MELLUM_CAPTURE_BUSY, MELLUM_CAPTURE_FINISHED };
/* Caller-owned core; no GPU/session resources or implicit footer on destruction. */
typedef struct {
    mellum_routes_writer writer;
    mellum_profiles profiles;
    uint64_t position;
    uint32_t state,phase,segment,rows,layers;
    bool forward_active,logits;
} mellum_capture_state;
bool mellum_capture_state_init(mellum_capture_state*, uint32_t cap, mellum_routes_sink, void*, const mellum_kernel_catalog*);
bool mellum_capture_begin_sync(mellum_capture_state*, const uint32_t*, size_t count, const uint32_t*, size_t length, uint32_t segment);
bool mellum_capture_begin_eval(mellum_capture_state*, uint64_t position, uint32_t segment);
int mellum_capture_end_operation(mellum_capture_state*, mellum_profile_scope*, int status);
void mellum_capture_abort(mellum_capture_state*, mellum_profile_scope*);
bool mellum_capture_has_logits(const mellum_capture_state*);
#endif
