#include "ds4_mellum_capture_state.h"
bool mellum_capture_state_init(mellum_capture_state *s,uint32_t cap,mellum_routes_sink f,void *ctx,const mellum_kernel_catalog *c) {(void)s;(void)cap;(void)f;(void)ctx;(void)c;return false;}
bool mellum_capture_begin_sync(mellum_capture_state *s,const uint32_t *a,size_t n,const uint32_t *b,size_t m,uint32_t seg) {(void)s;(void)a;(void)n;(void)b;(void)m;(void)seg;return false;}
bool mellum_capture_begin_eval(mellum_capture_state *s,uint64_t p,uint32_t seg) {(void)s;(void)p;(void)seg;return false;}
int mellum_capture_end_operation(mellum_capture_state *s,mellum_profile_scope *p,int status) {(void)s;(void)p;(void)status;return 1;}
void mellum_capture_abort(mellum_capture_state *s,mellum_profile_scope *p) {(void)s;(void)p;}
bool mellum_capture_has_logits(const mellum_capture_state *s) {(void)s;return false;}
