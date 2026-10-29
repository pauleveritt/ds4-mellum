#ifndef DS4_MELLUM_CAPTURE_H
#define DS4_MELLUM_CAPTURE_H
#include "ds4.h"
#include "ds4_mellum_routes.h"
#include "ds4_mellum_profile.h"
/* Diagnostic-only, serialized/non-reentrant. Engine and sinks remain caller-owned.
 * The opaque handle owns a fresh session; no reset/rewind or retry after failure.
 * Sinks must not invoke capture/inference APIs. Check flush/close before export. */
typedef struct ds4_mellum_capture ds4_mellum_capture;
typedef struct {
    mellum_routes_sink spool_sink;
    void *spool_context;
    ds4_session_cancel_fn cancel;
    void *cancel_context;
} ds4_mellum_capture_options;
int ds4_mellum_capture_create(ds4_mellum_capture **out, ds4_engine *, int context_tokens,
    const ds4_mellum_capture_options *, char *err, size_t errlen);
int ds4_mellum_capture_sync(ds4_mellum_capture *, const ds4_tokens *, uint32_t segment, char *, size_t);
int ds4_mellum_capture_eval(ds4_mellum_capture *, int token, uint32_t segment, char *, size_t);
int ds4_mellum_capture_copy_logits(ds4_mellum_capture *, float *, int capacity);
int ds4_mellum_capture_finish(ds4_mellum_capture *, mellum_profile_sink, void *, char *, size_t);
/* Null safe, never writes a footer or calls sinks. Engine must outlive handle. */
void ds4_mellum_capture_free(ds4_mellum_capture *);
#endif
