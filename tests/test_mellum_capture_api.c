/* Public unsupported CPU API: no model or GPU is created. */
#include <stdlib.h>
static void *checkpoint_test_realloc(void *, size_t);
#define realloc checkpoint_test_realloc
#include "../ds4.c"
#undef realloc
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); } } while (0)
static bool fail_realloc;
static unsigned failed_realloc_calls;
static void *checkpoint_test_realloc(void *p,size_t n) {
    if (fail_realloc) { failed_realloc_calls++;return NULL; }
    return realloc(p,n);
}
static size_t bytes;
static size_t spool(void *ctx,const void *p,size_t n) { (void)ctx;(void)p;bytes+=n;return n; }
static ds4_mellum_capture *fixture(void) {
    ds4_mellum_capture *c=calloc(1,sizeof(*c));CHECK(c);
    c->session=calloc(1,sizeof(*c->session));CHECK(c->session);c->session->ctx_size=4;
    CHECK(mellum_capture_state_init(&c->core,2,spool,NULL,mellum_observer_catalog()));return c;
}
static void release_fixture(ds4_mellum_capture *c) {
    free(c->session->checkpoint.v);free(c->session->logits);free(c->session);c->session=NULL;
    size_t before=bytes;ds4_mellum_capture_free(c);CHECK(bytes==before);
}
static void invalid_operations(void) {
    for(unsigned bad=0;bad<7;bad++) {
        ds4_mellum_capture *c=fixture();int tokens[]={10,11,12,13};
        ds4_tokens prompt={.v=tokens,.len=3};char err[128]={0};size_t before=bytes;
        switch(bad) {
        case 0:CHECK(ds4_mellum_capture_eval(c,-1,0,err,sizeof(err))!=0);break;
        case 1:CHECK(ds4_mellum_capture_eval(c,(int)DS4_N_VOCAB,0,err,sizeof(err))!=0);break;
        case 2:prompt.len=4;CHECK(ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err))!=0);break;
        case 3:tokens[2]=-1;CHECK(ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err))!=0);break;
        case 4:prompt.v=NULL;CHECK(ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err))!=0);break;
        case 5:prompt.len=0;CHECK(ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err))!=0);break;
        case 6:CHECK(ds4_mellum_capture_sync(c,NULL,0,err,sizeof(err))!=0);break;
        }
        CHECK(c->core.state==MELLUM_CAPTURE_FAILED && err[0] && bytes==before);
        CHECK(ds4_mellum_capture_sync(c,&prompt,0,NULL,0)!=0);
        CHECK(ds4_mellum_capture_eval(c,10,0,NULL,0)!=0);
        CHECK(ds4_mellum_capture_copy_logits(c,NULL,0)==0);
        CHECK(ds4_mellum_capture_finish(c,NULL,NULL,NULL,0)!=0 && bytes==before);release_fixture(c);
    }
    ds4_mellum_capture *c=fixture();c->core.state=MELLUM_CAPTURE_FINISHED;c->core.logits=true;
    c->session->logits=calloc(DS4_N_VOCAB,sizeof(float));float *out=malloc(DS4_N_VOCAB*sizeof(float));
    CHECK(c->session->logits && out);c->session->logits[0]=42;
    CHECK(ds4_mellum_capture_copy_logits(c,out,(int)DS4_N_VOCAB)==(int)DS4_N_VOCAB && out[0]==42);
    CHECK(!ds4_mellum_capture_copy_logits(c,out,(int)DS4_N_VOCAB-1));
    size_t before=bytes;CHECK(ds4_mellum_capture_eval(c,10,0,NULL,0)!=0);
    CHECK(ds4_mellum_capture_finish(c,NULL,NULL,NULL,0)!=0);
    CHECK(c->core.state==MELLUM_CAPTURE_FINISHED && bytes==before);free(out);release_fixture(c);
    c=fixture();size_t n=bytes;release_fixture(c);CHECK(bytes==n);
}
static bool cancelled(void *ctx) { unsigned *calls=ctx;(*calls)++;return true; }
static void cancellation_and_noop(void) {
    ds4_engine engine={.backend=DS4_BACKEND_CPU};
    ds4_mellum_capture *c=fixture();c->session->engine=&engine;
    unsigned calls=0;ds4_session_set_cancel(c->session,cancelled,&calls);
    int token=10;ds4_tokens prompt={.v=&token,.len=1,.cap=1};char err[128];size_t before=bytes;
    CHECK(ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err))==DS4_SESSION_SYNC_INTERRUPTED);
    CHECK(calls==1 && c->core.state==MELLUM_CAPTURE_FAILED && bytes==before);release_fixture(c);
    c=fixture();c->session->engine=&engine;
    uint32_t selected[448];for(unsigned i=0;i<448;i++)selected[i]=i%8;
    CHECK(mellum_routes_forward(&c->core.writer,0,0,1,1,0,0,(const uint32_t *)&token,1,selected,448));
    c->session->checkpoint=prompt;c->session->checkpoint_valid=true;c->core.logits=true;
    before=bytes;CHECK(!ds4_mellum_capture_sync(c,&prompt,123,err,sizeof(err)));
    CHECK(bytes==before && mellum_capture_has_logits(&c->core));
    token=11;int old=10;c->session->checkpoint.v=&old;
    CHECK(ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err))!=0);
    CHECK(c->core.state==MELLUM_CAPTURE_FAILED && bytes==before);
    c->session->checkpoint.v=NULL;release_fixture(c);
    CHECK(sizeof(ds4_mellum_capture)+2*917504u+MELLUM_PROFILE_TEXT<=2200000u);
}
static void checkpoint_allocation_failure(void) {
    unsigned failures=0;
    for (unsigned eval=0;eval<2;eval++) {
        ds4_engine engine={.backend=eval ? DS4_BACKEND_METAL:DS4_BACKEND_CPU};
        ds4_mellum_capture *c=fixture();c->session->engine=&engine;
        unsigned cancelled_calls=0;ds4_session_set_cancel(c->session,cancelled,&cancelled_calls);
        int token=10;ds4_tokens prompt={.v=&token,.len=1};char err[128];size_t before=bytes;
        fail_realloc=true;failed_realloc_calls=0;
        int rc=eval ? ds4_mellum_capture_eval(c,token,0,err,sizeof(err)):
                      ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err));
        fail_realloc=false;
        if (rc!=1 || failed_realloc_calls!=1 || cancelled_calls || !strstr(err,"checkpoint allocation")) {
            fprintf(stderr,"%s failed to reject checkpoint OOM before inference (rc=%d, realloc=%u, cancel=%u)\n",
                    eval ? "eval":"sync",rc,failed_realloc_calls,cancelled_calls);failures++;
        }
        CHECK(!c->session->checkpoint.v && !c->session->checkpoint.cap);
        CHECK(c->core.state==MELLUM_CAPTURE_FAILED && bytes==before);
        CHECK(ds4_mellum_capture_finish(c,NULL,NULL,NULL,0)!=0 && bytes==before);
        release_fixture(c);
    }
    CHECK(!failures);
}
int main(void) {
    ds4_mellum_capture_options opts={.spool_sink=spool};
    ds4_mellum_capture *sentinel=(ds4_mellum_capture *)&opts,*out=sentinel;
    char err[128]={0};
    CHECK(ds4_mellum_capture_create(&out,NULL,8192,&opts,err,sizeof(err))!=0);
    CHECK(out==NULL && strstr(err,"unsupported") && !bytes);
    for(unsigned bad=0;bad<6;bad++) {
        char tiny[2]={'x','z'};out=sentinel;
        CHECK(ds4_mellum_capture_create(bad==0 ? NULL:&out,NULL,bad==1 ? -1:1,
              bad==2 ? NULL:&opts,bad==3 ? NULL:tiny,bad==4 ? 0:1)!=0);
        CHECK(bad==0 || !out);CHECK(tiny[1]=='z');CHECK(!bytes);
    }
    CHECK(ds4_mellum_capture_sync(NULL,NULL,0,NULL,0)!=0);
    CHECK(ds4_mellum_capture_eval(NULL,0,0,NULL,0)!=0);
    CHECK(ds4_mellum_capture_copy_logits(NULL,NULL,0)==0);
    CHECK(ds4_mellum_capture_finish(NULL,NULL,NULL,NULL,0)!=0);
    ds4_mellum_capture_free(NULL);CHECK(!bytes);
    invalid_operations();cancellation_and_noop();checkpoint_allocation_failure();
    puts("Mellum capture API boundary checks: PASS");return 0;
}
