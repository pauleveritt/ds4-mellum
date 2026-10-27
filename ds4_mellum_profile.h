#ifndef DS4_MELLUM_PROFILE_H
#define DS4_MELLUM_PROFILE_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
enum { MELLUM_PROFILE_VALUES=6, MELLUM_PROFILE_ITEMS=64, MELLUM_PROFILE_LIMIT=128,
       MELLUM_PROFILE_TEXT=16384 };
enum { MELLUM_ROLE_EMBED, MELLUM_ROLE_PROJECTION, MELLUM_ROLE_ROUTER,
       MELLUM_ROLE_MID, MELLUM_ROLE_DOWN, MELLUM_ROLE_REDUCE,
       MELLUM_ROLE_ATTENTION, MELLUM_ROLE_HEAD, MELLUM_ROLE_OTHER };
typedef struct {
    const char *name;
    uint8_t value_count;
    const char *value_names[MELLUM_PROFILE_VALUES];
} mellum_kernel_def;
/* Definitions must stay immutable and alive for the whole capture. */
typedef struct { const mellum_kernel_def *defs; size_t count; } mellum_kernel_catalog;
typedef struct {
    uint16_t function;
    uint8_t role, value_count;
    uint32_t values[MELLUM_PROFILE_VALUES];
} mellum_kernel_descriptor;
typedef struct {
    const mellum_kernel_catalog *catalog;
    uint32_t count;
    bool with_head, failed;
    mellum_kernel_descriptor items[MELLUM_PROFILE_ITEMS];
} mellum_profile;
bool mellum_kernel_catalog_init(mellum_kernel_catalog*, const mellum_kernel_def*, size_t);
bool mellum_profile_init(mellum_profile*, const mellum_kernel_catalog*, bool with_head);
bool mellum_profile_record(mellum_profile*, mellum_kernel_descriptor);
bool mellum_profile_valid(const mellum_profile*);
typedef struct {
    const mellum_kernel_catalog *catalog;
    bool failed, active;
    uint32_t count;
    mellum_profile current, profiles[MELLUM_PROFILE_LIMIT];
} mellum_profiles;
typedef struct { mellum_profiles *owner; } mellum_profile_scope;
bool mellum_profiles_init(mellum_profiles*, const mellum_kernel_catalog*);
bool mellum_profile_scope_enter(mellum_profile_scope*, mellum_profiles*, bool head);
bool mellum_profile_scope_record(mellum_profile_scope*, mellum_kernel_descriptor);
bool mellum_profile_scope_exit(mellum_profile_scope*, mellum_profiles*, bool success, uint32_t *id);
#endif
