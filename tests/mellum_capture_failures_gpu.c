/* Live allocation-boundary/cancellation checks; never included in model-free gates. */
#include "../ds4_gpu.h"
#include <stdlib.h>
static bool injecting;
static size_t allocation_index,fail_at;
static bool fail_allocation(void) { return injecting && allocation_index++==fail_at; }
static void *fault_malloc(size_t);
static void *fault_calloc(size_t,size_t);
static void *fault_realloc(void *,size_t);
static ds4_gpu_tensor *fault_tensor(uint64_t);
static ds4_gpu_tensor *fault_managed(uint64_t);
static void fault_free(void *);
static void fault_tensor_free(ds4_gpu_tensor *);
/* Only resources returned while create injection is enabled are owned here.
 * Metal-internal resources and allocations in separately compiled TUs are not. */
static void *live[256];
static size_t live_count;
static void remember(void *p) {
    if (!p || !injecting) return;
    if (live_count==sizeof(live)/sizeof(*live)) abort();
    live[live_count++]=p;
}
static void forget(void *p) {
    if (!p) return;
    for (size_t i=0;i<live_count;i++) if (live[i]==p) {
        live[i]=live[--live_count];return;
    }
}
#define malloc fault_malloc
#define calloc fault_calloc
#define realloc fault_realloc
#define ds4_gpu_tensor_alloc fault_tensor
#define ds4_gpu_tensor_alloc_managed fault_managed
#define free fault_free
#define ds4_gpu_tensor_free fault_tensor_free
#include "../ds4.c"
#undef malloc
#undef calloc
#undef realloc
#undef ds4_gpu_tensor_alloc
#undef ds4_gpu_tensor_alloc_managed
#undef free
#undef ds4_gpu_tensor_free
static void *fault_malloc(size_t n) {
    void *p=fail_allocation() ? NULL:malloc(n);remember(p);return p;
}
static void *fault_calloc(size_t n,size_t s) {
    void *p=fail_allocation() ? NULL:calloc(n,s);remember(p);return p;
}
static void *fault_realloc(void *p,size_t n) {
    if (fail_allocation()) return NULL;
    /* Creation never requests realloc-to-zero; keep tracking unambiguous. */
    if (!n) abort();
    size_t slot=live_count;
    for (size_t i=0;i<live_count;i++) if (live[i]==p) { slot=i;break; }
    void *next=realloc(p,n);
    if (next) {
        if (slot<live_count) live[slot]=next;
        else remember(next);
    }
    return next;
}
static ds4_gpu_tensor *fault_tensor(uint64_t n) {
    ds4_gpu_tensor *p=fail_allocation() ? NULL:ds4_gpu_tensor_alloc(n);remember(p);return p;
}
static ds4_gpu_tensor *fault_managed(uint64_t n) {
    ds4_gpu_tensor *p=fail_allocation() ? NULL:ds4_gpu_tensor_alloc_managed(n);remember(p);return p;
}
static void fault_free(void *p) { forget(p);free(p); }
static void fault_tensor_free(ds4_gpu_tensor *p) { forget(p);ds4_gpu_tensor_free(p); }
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s (allocation %zu)\n",__FILE__,__LINE__,#c,fail_at);exit(1); } } while(0)
static size_t written;
static size_t sink(void *ctx,const void *data,size_t n) { (void)ctx;(void)data;written+=n;return n; }
static bool cancel_after_chunk(void *ctx) { return ++*(unsigned *)ctx==3; }
static bool profile(void *ctx,uint32_t id,const char *text,size_t n) {
    (void)ctx;return id==0 && n && !strncmp(text,"profile-v1;head=encoded;",24);
}
static void allocation_matrix(ds4_engine *engine) {
    ds4_mellum_capture_options options={.spool_sink=sink};
    ds4_mellum_capture *c=NULL;char err[512];written=0;
    injecting=true;allocation_index=0;fail_at=SIZE_MAX;
    int rc=ds4_mellum_capture_create(&c,engine,8192,&options,err,sizeof(err));
    injecting=false;CHECK(!rc && c && written==24);
    size_t count=allocation_index;CHECK(count>2 && live_count>2);ds4_mellum_capture_free(c);
    CHECK(!live_count);
    for(size_t i=0;i<count;i++) {
        c=NULL;written=0;allocation_index=0;fail_at=i;injecting=true;
        rc=ds4_mellum_capture_create(&c,engine,8192,&options,err,sizeof(err));
        injecting=false;
        CHECK(rc && !c && !written && allocation_index>i && err[0]);
        CHECK(!ds4_gpu_commands_active() && !live_count);
    }
    printf("owned allocation failures: %zu boundaries PASS\n",count);
}
static void chunk_cancellation(ds4_engine *engine) {
    unsigned calls=100;char err[512];ds4_mellum_capture *c=NULL;
    int vocab=ds4_engine_vocab_size(engine);CHECK(vocab>0);
    float *logits=malloc((size_t)vocab*sizeof(float));CHECK(logits);
    ds4_mellum_capture_options options={.spool_sink=sink,.cancel=cancel_after_chunk,.cancel_context=&calls};
    CHECK(!ds4_mellum_capture_create(&c,engine,8192,&options,err,sizeof(err)));
    int cap=(int)c->session->qwen4_graph.cap_tokens;
    int *tokens=malloc((size_t)(cap+2)*sizeof(int));CHECK(tokens);
    for(int i=0;i<cap+2;i++)tokens[i]=10;
    ds4_tokens prompt={.v=tokens,.len=1};
    CHECK(!ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err)));
    CHECK(ds4_mellum_capture_copy_logits(c,logits,vocab)==vocab);
    /* Leave valid prior logits behind, then interrupt a later multi-chunk sync. */
    calls=0;prompt.len=cap+2;
    CHECK(ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err))==DS4_SESSION_SYNC_INTERRUPTED);
    CHECK(calls==3 && c->core.writer.forwards==2 && c->core.writer.positions==(uint64_t)cap+1);
    CHECK(c->core.state==MELLUM_CAPTURE_FAILED && !c->scope.owner && !ds4_gpu_commands_active());
    size_t before=written;
    CHECK(ds4_mellum_capture_finish(c,profile,NULL,err,sizeof(err))!=0 && written==before);
    CHECK(!ds4_mellum_capture_copy_logits(c,logits,vocab));ds4_mellum_capture_free(c);CHECK(written==before);
    options.cancel=NULL;options.cancel_context=NULL;
    CHECK(!ds4_mellum_capture_create(&c,engine,8192,&options,err,sizeof(err)));
    prompt.len=1;CHECK(!ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err)));
    CHECK(c->core.profiles.count==1 && !c->scope.owner);
    CHECK(ds4_mellum_capture_copy_logits(c,logits,vocab)==vocab);
    CHECK(!ds4_mellum_capture_finish(c,profile,NULL,err,sizeof(err)));
    ds4_mellum_capture_free(c);free(tokens);free(logits);
    printf("between-chunk cancellation and fresh-handle isolation: PASS (cap=%d)\n",cap);
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    ds4_engine *engine=NULL;ds4_engine_options options={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=8192,.prefill_chunk=32};
    CHECK(!ds4_engine_open(&engine,&options));
    allocation_matrix(engine);chunk_cancellation(engine);ds4_engine_close(engine);
    puts("owned capture live failure matrix: PASS");return 0;
}
