/* No GPU: real lifecycle/writer/profile code, explicitly supplied fixture events. */
#include "ds4_mellum_capture_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); } } while (0)
typedef struct { unsigned char bytes[32768];size_t n,limit,calls; } memory_sink;
static size_t append(void *ctx,const void *data,size_t n) {
    memory_sink *s=ctx;s->calls++;
    if(n>s->limit-s->n)n=s->limit-s->n;
    memcpy(s->bytes+s->n,data,n);s->n+=n;return n;
}
static const mellum_kernel_def definitions[]={{"fixture_kernel",1,{"tile"}}};
static mellum_kernel_catalog catalog;
static uint32_t selected[448];
static mellum_capture_state *fresh(memory_sink *sink) {
    *sink=(memory_sink){.limit=32768};
    mellum_capture_state *c=calloc(1,sizeof(*c));
    CHECK(c && mellum_capture_state_init(c,2,append,sink,&catalog));return c;
}
static void seed_prefix(mellum_capture_state *c) {
    /* Real native-writer state supplies a committed prefix for prefix checks. */
    const uint32_t tokens[]={10,11};
    CHECK(mellum_routes_forward(&c->writer,0,0,2,1,7,0,tokens,2,selected,448));
}
static void check_operations(void) {
    memory_sink sink;mellum_capture_state *c=fresh(&sink);mellum_profile_scope scope={0};
    const unsigned char header[]="MLRT0001" "\x1c\0\0\0" "\x40\0\0\0" "\x08\0\0\0" "\x02\0\0\0";
    CHECK(sink.n==24 && !memcmp(sink.bytes,header,24));
    uint32_t prompt[]={10,11,12};
    CHECK(!mellum_capture_has_logits(c));
    CHECK(mellum_capture_begin_sync(c,NULL,0,prompt,1,99));
    CHECK(c->phase==1 && c->segment==99);
    CHECK(mellum_capture_end_operation(c,&scope,0)==0);
    seed_prefix(c);
    CHECK(mellum_capture_begin_sync(c,prompt,2,prompt,2,100));
    CHECK(mellum_capture_end_operation(c,&scope,0)==0);
    CHECK(mellum_capture_begin_sync(c,prompt,2,prompt,3,101));
    CHECK(c->phase==1 && c->segment==101);
    CHECK(mellum_capture_end_operation(c,&scope,0)==0);
    CHECK(mellum_capture_begin_eval(c,2,102));CHECK(c->phase==2 && c->segment==102);
    size_t before=sink.n;
    CHECK(mellum_capture_end_operation(c,&scope,2)==2);
    CHECK(!mellum_capture_begin_eval(c,2,1));CHECK(!mellum_capture_has_logits(c));
    CHECK(!mellum_routes_finish(&c->writer) && sink.n==before);
    mellum_capture_abort(c,&scope);CHECK(sink.n==before);free(c);
    for(unsigned bad=0;bad<8;bad++) {
        c=fresh(&sink);seed_prefix(c);
        uint32_t changed[]={10,99,12};
        const uint32_t *old=prompt,*next=prompt;size_t n=2,len=3;
        switch(bad) {
        case 0:len=1;break;
        case 1:next=changed;break;
        case 2:old=NULL;break;
        case 3:next=NULL;break;
        case 4:len=0;break;
        case 5:n=1;break;
        case 6:n=SIZE_MAX;break;
        case 7:CHECK(mellum_capture_begin_eval(c,2,0));break;
        }
        before=sink.n;
        CHECK(!mellum_capture_begin_sync(c,old,n,next,len,0));
        CHECK(c->state==MELLUM_CAPTURE_FAILED && sink.n==before);
        CHECK(mellum_capture_end_operation(c,&scope,0)!=0);
        CHECK(!mellum_capture_begin_eval(c,2,0));CHECK(sink.n==before);free(c);
    }
    c=fresh(&sink);CHECK(!mellum_capture_begin_eval(c,1,0));free(c);
    c=fresh(&sink);CHECK(mellum_capture_begin_eval(c,0,0));
    CHECK(mellum_capture_end_operation(c,&scope,1)==1);CHECK(!mellum_routes_finish(&c->writer));free(c);
    c=fresh(&sink);CHECK(mellum_capture_end_operation(c,&scope,0)!=0);free(c);
}
static void check_initialization(void) {
    memory_sink sink; mellum_capture_state *c=fresh(&sink);free(c);
    CHECK(sizeof(mellum_capture_state)+2*917504u+16384u<=2200000u);
    c=calloc(1,sizeof(*c));CHECK(c);
    for(unsigned bad=0;bad<5;bad++) {
        sink=(memory_sink){.limit=bad==4 ? 10:32768};
        CHECK(!mellum_capture_state_init(c,bad==0 ? 0:bad==1 ? 1025:2,
              bad==2 ? NULL:append,&sink,bad==3 ? NULL:&catalog));
        CHECK(c->state==MELLUM_CAPTURE_FAILED);
        if(bad!=4)CHECK(sink.n==0);
        size_t n=sink.n;mellum_capture_abort(c,NULL);CHECK(sink.n==n);
    }
    free(c);
}
int main(void) {
    CHECK(mellum_kernel_catalog_init(&catalog,definitions,1));
    for(unsigned l=0;l<28;l++)for(unsigned r=0;r<2;r++)for(unsigned k=0;k<8;k++)selected[(l*2+r)*8+k]=k;
    check_initialization();check_operations();puts("Mellum capture operation checks: PASS");return 0;
}
