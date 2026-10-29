#ifndef DS4_MELLUM_OBSERVER_H
#define DS4_MELLUM_OBSERVER_H
#include "ds4_mellum_profile.h"
/* Serialized/non-reentrant. Scope must remain active/alive until matching leave.
 * Encode is a bookkeeping notification after dispatch, never a GPU barrier.
 * A failure is sticky in the profile: callers must still drain GPU work. */
const mellum_kernel_catalog *mellum_observer_catalog(void);
bool mellum_observer_enter(mellum_profile_scope *);
bool mellum_observer_leave(mellum_profile_scope *);
void mellum_observer_projection_role(bool head);
bool mellum_observer_encode(const char *name, size_t count, const uint32_t *values);
#endif
