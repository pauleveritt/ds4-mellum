#include "ds4_mellum_profile.h"
#include <string.h>

static bool identifier(const char *s, size_t max) {
    if (!s || !s[0]) return false;
    for (size_t i=0; i<=max; i++) {
        unsigned char c=(unsigned char)s[i];
        if (!c) return true;
        if (!(c=='_' || (c>='a' && c<='z') || (c>='A' && c<='Z') ||
              (i && c>='0' && c<='9'))) return false;
    }
    return false;
}

bool mellum_kernel_catalog_init(mellum_kernel_catalog *c, const mellum_kernel_def *defs, size_t n) {
    if (!c) return false;
    *c=(mellum_kernel_catalog){0};
    if (!defs || !n || n>65536) return false;
    for (size_t i=0;i<n;i++) {
        if (!identifier(defs[i].name,95) || defs[i].value_count>MELLUM_PROFILE_VALUES) return false;
        for (size_t j=0;j<i;j++) if (!strcmp(defs[i].name,defs[j].name)) return false;
        for (unsigned j=0;j<defs[i].value_count;j++) {
            if (!identifier(defs[i].value_names[j],31)) return false;
            for (unsigned k=0;k<j;k++)
                if (!strcmp(defs[i].value_names[j],defs[i].value_names[k])) return false;
        }
    }
    *c=(mellum_kernel_catalog){defs,n};
    return true;
}

bool mellum_profile_init(mellum_profile *p, const mellum_kernel_catalog *c, bool head) {
    if (!p) return false;
    *p=(mellum_profile){.catalog=c,.with_head=head,.failed=true};
    if (!c || !c->defs || !c->count || c->count>65536) return false;
    p->failed=false;
    return true;
}

/* Compare semantic words, never structure padding or unused constants. */
static int descriptor_compare(const mellum_kernel_descriptor *a, const mellum_kernel_descriptor *b) {
    if (a->role!=b->role) return a->role<b->role ? -1:1;
    if (a->function!=b->function) return a->function<b->function ? -1:1;
    if (a->value_count!=b->value_count) return a->value_count<b->value_count ? -1:1;
    for (unsigned i=0;i<a->value_count;i++)
        if (a->values[i]!=b->values[i]) return a->values[i]<b->values[i] ? -1:1;
    return 0;
}

bool mellum_profile_record(mellum_profile *p, mellum_kernel_descriptor d) {
    if (!p || p->failed) return false;
    if (d.function>=p->catalog->count || d.role>MELLUM_ROLE_OTHER ||
        d.value_count!=p->catalog->defs[d.function].value_count) {
        p->failed=true;
        return false;
    }
    for (unsigned i=d.value_count;i<MELLUM_PROFILE_VALUES;i++) d.values[i]=0;
    unsigned at=0;
    while (at<p->count && descriptor_compare(&p->items[at],&d)<0) at++;
    if (at<p->count && descriptor_compare(&p->items[at],&d)==0) return true;
    if (p->count==MELLUM_PROFILE_ITEMS) { p->failed=true; return false; }
    memmove(p->items+at+1,p->items+at,(p->count-at)*sizeof(d));
    p->items[at]=d;
    p->count++;
    return true;
}

bool mellum_profile_valid(const mellum_profile *p) {
    if (!p || p->failed || !p->catalog) return false;
    unsigned roles=0;
    for (unsigned i=0;i<p->count;i++) roles|=1u<<p->items[i].role;
    const unsigned required=(1u<<MELLUM_ROLE_HEAD)-1u;
    return (roles&required)==required && ((roles>>MELLUM_ROLE_HEAD)&1u)==p->with_head;
}

bool mellum_profiles_init(mellum_profiles *r, const mellum_kernel_catalog *c) {
    if (!r) return false;
    *r=(mellum_profiles){.catalog=c,.failed=true};
    if (!mellum_profile_init(&r->current,c,false)) return false;
    r->failed=false;
    return true;
}

bool mellum_profile_scope_enter(mellum_profile_scope *s, mellum_profiles *r, bool head) {
    /* Refusing a foreign/nested scope must not poison its current owner. */
    if (!s || s->owner || !r || r->failed || r->active) return false;
    if (!mellum_profile_init(&r->current,r->catalog,head)) { r->failed=true; return false; }
    r->active=true;
    s->owner=r;
    return true;
}

bool mellum_profile_scope_record(mellum_profile_scope *s, mellum_kernel_descriptor d) {
    if (!s || !s->owner || s->owner->failed) return false;
    if (mellum_profile_record(&s->owner->current,d)) return true;
    s->owner->failed=true;
    return false;
}

static bool profile_equal(const mellum_profile *a, const mellum_profile *b) {
    if (a->with_head!=b->with_head || a->count!=b->count) return false;
    for (unsigned i=0;i<a->count;i++)
        if (descriptor_compare(&a->items[i],&b->items[i])) return false;
    return true;
}

bool mellum_profile_scope_exit(mellum_profile_scope *s, mellum_profiles *r, bool ok, uint32_t *id) {
    if (!s || !r || s->owner!=r) return false;
    s->owner=NULL;
    r->active=false;
    if (!ok || !id || r->failed || !mellum_profile_valid(&r->current)) {
        r->failed=true;
        return false;
    }
    for (unsigned i=0;i<r->count;i++) {
        if (profile_equal(&r->current,&r->profiles[i])) { *id=i; return true; }
    }
    if (r->count==MELLUM_PROFILE_LIMIT) { r->failed=true; return false; }
    *id=r->count;
    r->profiles[r->count++]=r->current;
    return true;
}
