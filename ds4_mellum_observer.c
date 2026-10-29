#include "ds4_mellum_observer.h"
#include <string.h>

#define MELLUM_KERNEL(name,role,n,...) {#name,n,{__VA_ARGS__}},
static const mellum_kernel_def definitions[]={
#include "ds4_mellum_kernels.inc"
};
#undef MELLUM_KERNEL
#define MELLUM_KERNEL(name,role,n,...) MELLUM_ROLE_##role,
static const uint8_t roles[]={
#include "ds4_mellum_kernels.inc"
};
#undef MELLUM_KERNEL
static const mellum_kernel_catalog catalog={definitions,sizeof(definitions)/sizeof(definitions[0])};
static mellum_profile_scope *active;
static bool head_projection;

const mellum_kernel_catalog *mellum_observer_catalog(void) { return &catalog; }

bool mellum_observer_enter(mellum_profile_scope *scope) {
    if (active || !scope || !scope->owner || !scope->owner->active || scope->owner->failed) return false;
    active=scope;
    head_projection=false;
    return true;
}

bool mellum_observer_leave(mellum_profile_scope *scope) {
    if (!active || active!=scope) return false;
    active=NULL;
    head_projection=false;
    return true;
}

void mellum_observer_projection_role(bool head) {
    if (active) head_projection=head;
}

bool mellum_observer_encode(const char *name, size_t count, const uint32_t *values) {
    if (!active) return true;
    mellum_profiles *r=active->owner;
    if (!r || r->failed) return false;
    if (name && (!count || values)) {
        for (size_t i=0;i<catalog.count;i++) {
            if (strcmp(name,definitions[i].name)) continue;
            if (count!=definitions[i].value_count) break;
            mellum_kernel_descriptor d={.function=(uint16_t)i,.role=roles[i],.value_count=(uint8_t)count};
            if (d.role==MELLUM_ROLE_PROJECTION && head_projection) d.role=MELLUM_ROLE_HEAD;
            for (size_t j=0;j<count;j++) d.values[j]=values[j];
            return mellum_profile_scope_record(active,d);
        }
    }
    r->current.failed=true;
    r->failed=true;
    return false;
}
