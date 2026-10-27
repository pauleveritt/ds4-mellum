/* Public unsupported CPU API: no model or GPU is created. */
#include "../ds4.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); } } while (0)
static size_t bytes;
static size_t spool(void *ctx,const void *p,size_t n) { (void)ctx;(void)p;bytes+=n;return n; }
static ds4_mellum_capture *fixture(void) {
    ds4_mellum_capture *c=calloc(1,sizeof(*c));CHECK(c);
    c->session=calloc(1,sizeof(*c->session));CHECK(c->session);c->session->ctx_size=4;
    CHECK(mellum_capture_state_init(&c->core,2,spool,NULL,mellum_observer_catalog()));return c;
}
static void release_fixture(ds4_mellum_capture *c) {
    free(c->session->logits);free(c->session);c->session=NULL;
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
    invalid_operations();
    puts("Mellum capture API boundary checks: PASS");return 0;
}
