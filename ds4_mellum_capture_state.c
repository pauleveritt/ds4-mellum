#include "ds4_mellum_capture_state.h"
#include <string.h>

void mellum_capture_abort(mellum_capture_state *c, mellum_profile_scope *scope) {
    if (!c || c->state==MELLUM_CAPTURE_FINISHED) return;
    if (c->scope && c->scope->owner==&c->profiles)
        (void)mellum_profile_scope_exit(c->scope,&c->profiles,false,NULL);
    c->scope=NULL;
    if (scope && scope->owner==&c->profiles)
        (void)mellum_profile_scope_exit(scope,&c->profiles,false,NULL);
    c->profiles.active=false;
    c->profiles.failed=true;
    c->writer.state=0;
    c->forward_active=false;
    c->state=MELLUM_CAPTURE_FAILED;
}

bool mellum_capture_state_init(mellum_capture_state *c, uint32_t cap, mellum_routes_sink sink,
                               void *ctx, const mellum_kernel_catalog *catalog) {
    if (!c) return false;
    *c=(mellum_capture_state){0};
    mellum_routes_layout_info layout;
    if (!sink || !mellum_routes_layout(cap,1,0,&layout) ||
        !mellum_profiles_init(&c->profiles,catalog) ||
        !mellum_routes_begin(&c->writer,cap,sink,ctx)) {
        mellum_capture_abort(c,NULL);
        return false;
    }
    c->state=MELLUM_CAPTURE_READY;
    return true;
}

static bool operation_ready(mellum_capture_state *c) {
    if (c && c->state==MELLUM_CAPTURE_READY && c->writer.state==1 &&
        !c->profiles.failed && !c->profiles.active) return true;
    mellum_capture_abort(c,NULL);
    return false;
}

bool mellum_capture_begin_sync(mellum_capture_state *c, const uint32_t *committed, size_t n,
                               const uint32_t *prompt, size_t length, uint32_t segment) {
    if (!operation_ready(c)) return false;
    if (!prompt || !length || n!=c->writer.positions || n>length ||
        n>SIZE_MAX/sizeof(uint32_t) || (n && (!committed || memcmp(committed,prompt,n*sizeof(uint32_t))))) {
        mellum_capture_abort(c,NULL);
        return false;
    }
    c->operation_end=length;
    c->phase=1;
    c->segment=segment;
    c->state=MELLUM_CAPTURE_BUSY;
    return true;
}

bool mellum_capture_begin_eval(mellum_capture_state *c, uint64_t position, uint32_t segment) {
    if (!operation_ready(c)) return false;
    if (position!=c->writer.positions || position==UINT64_MAX) { mellum_capture_abort(c,NULL); return false; }
    c->operation_end=position+1;
    c->phase=2;
    c->segment=segment;
    c->state=MELLUM_CAPTURE_BUSY;
    return true;
}

int mellum_capture_end_operation(mellum_capture_state *c, mellum_profile_scope *scope, int status) {
    if (!c || status || c->state!=MELLUM_CAPTURE_BUSY || c->forward_active ||
        c->profiles.active || c->profiles.failed || c->writer.state!=1 ||
        c->writer.positions!=c->operation_end) {
        mellum_capture_abort(c,scope);
        return status ? status:1;
    }
    c->state=MELLUM_CAPTURE_READY;
    return 0;
}

bool mellum_capture_has_logits(const mellum_capture_state *c) {
    return c && (c->state==MELLUM_CAPTURE_READY || c->state==MELLUM_CAPTURE_FINISHED) && c->logits;
}

bool mellum_capture_forward_begin(mellum_capture_state *c, mellum_profile_scope *s,
                                  uint64_t position, uint32_t rows, bool head) {
    mellum_routes_layout_info layout;
    if (!c || c->state!=MELLUM_CAPTURE_BUSY || c->forward_active ||
        c->writer.state!=1 || c->profiles.failed ||
        position!=c->writer.positions || position>UINT64_MAX-rows ||
        position+rows>c->operation_end || (c->phase==2 && rows!=1) ||
        !mellum_routes_layout(c->writer.cap,rows,0,&layout) ||
        !mellum_profile_scope_enter(s,&c->profiles,head)) {
        mellum_capture_abort(c,NULL);
        return false;
    }
    c->scope=s;
    c->position=position;
    c->rows=rows;
    c->layers=0;
    c->forward_active=true;
    return true;
}

bool mellum_capture_layer(mellum_capture_state *c, uint32_t layer) {
    if (!c || c->state!=MELLUM_CAPTURE_BUSY || !c->forward_active ||
        layer!=c->layers || layer>=MELLUM_ROUTES_LAYERS || c->profiles.failed) {
        mellum_capture_abort(c,NULL);
        return false;
    }
    c->layers++;
    return true;
}

bool mellum_capture_forward_end(mellum_capture_state *c, mellum_profile_scope *s, bool ok,
    const uint32_t *tokens, size_t token_count, const uint32_t *selected, size_t selected_count) {
    uint32_t id;
    if (!c || c->state!=MELLUM_CAPTURE_BUSY || !c->forward_active || !ok ||
        s!=c->scope || c->layers!=MELLUM_ROUTES_LAYERS ||
        !mellum_profile_scope_exit(s,&c->profiles,true,&id)) {
        mellum_capture_abort(c,NULL);
        return false;
    }
    c->scope=NULL;
    if (!mellum_routes_forward(&c->writer,c->writer.forwards,c->position,c->rows,
                              c->phase,c->segment,id,tokens,token_count,selected,selected_count)) {
        mellum_capture_abort(c,NULL);
        return false;
    }
    c->forward_active=false;
    c->logits=c->profiles.current.with_head;
    return true;
}

bool mellum_capture_finish(mellum_capture_state *c, mellum_profile_sink sink, void *ctx) {
    if (!operation_ready(c)) return false;
    if (!c->writer.forwards || !mellum_profiles_emit(&c->profiles,sink,ctx) ||
        !mellum_routes_finish(&c->writer)) {
        mellum_capture_abort(c,NULL);
        return false;
    }
    c->state=MELLUM_CAPTURE_FINISHED;
    return true;
}
