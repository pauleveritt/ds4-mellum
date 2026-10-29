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
static void record_roles(mellum_profile_scope *scope, bool head) {
    for(unsigned role=0;role<(head ? 8u:7u);role++)
        CHECK(mellum_profile_scope_record(scope,(mellum_kernel_descriptor){.role=role,.value_count=1,.values={32}}));
}
static void complete_forward(mellum_capture_state *c, uint64_t pos, uint32_t rows,
                             const uint32_t *tokens, bool head) {
    mellum_profile_scope scope={0};
    CHECK(mellum_capture_forward_begin(c,&scope,pos,rows,head));
    record_roles(&scope,head);
    for(unsigned l=0;l<28;l++)CHECK(mellum_capture_layer(c,l));
    CHECK(mellum_capture_forward_end(c,&scope,true,tokens,rows,selected,448));
    CHECK(!scope.owner);
}
static void seed_prefix(mellum_capture_state *c) {
    const uint32_t tokens[]={10,11};
    CHECK(mellum_capture_begin_sync(c,NULL,0,tokens,2,7));
    complete_forward(c,0,2,tokens,true);
    CHECK(mellum_capture_end_operation(c,NULL,0)==0);
}
static void check_operations(void) {
    memory_sink sink;mellum_capture_state *c=fresh(&sink);mellum_profile_scope scope={0};
    const unsigned char header[]="MLRT0001" "\x1c\0\0\0" "\x40\0\0\0" "\x08\0\0\0" "\x02\0\0\0";
    CHECK(sink.n==24 && !memcmp(sink.bytes,header,24));
    uint32_t prompt[]={10,11,12};
    CHECK(!mellum_capture_has_logits(c));
    CHECK(mellum_capture_begin_sync(c,NULL,0,prompt,1,99));
    CHECK(c->phase==1 && c->segment==99);
    complete_forward(c,0,1,prompt,true);
    CHECK(mellum_capture_end_operation(c,&scope,0)==0);
    free(c);c=fresh(&sink);seed_prefix(c);
    CHECK(mellum_capture_begin_sync(c,prompt,2,prompt,2,100));
    CHECK(mellum_capture_end_operation(c,&scope,0)==0);
    CHECK(mellum_capture_begin_sync(c,prompt,2,prompt,3,101));
    CHECK(c->phase==1 && c->segment==101);
    complete_forward(c,2,1,prompt+2,true);
    CHECK(mellum_capture_end_operation(c,&scope,0)==0);
    CHECK(mellum_capture_begin_eval(c,3,102));CHECK(c->phase==2 && c->segment==102);
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
typedef struct { memory_sink *spool; size_t calls; bool fail; FILE *file; } profile_output;
static bool profile_append(void *ctx,uint32_t id,const char *text,size_t n) {
    profile_output *out=ctx;
    CHECK(out->spool->n>=24 && memcmp(out->spool->bytes+out->spool->n-20,"END1",4));
    CHECK(id==out->calls && n==strlen(text));
    CHECK(strstr(text,"profile-v1;head=encoded;EMBED:fixture_kernel(tile=32)")==text);
    CHECK(strstr(text,";HEAD:fixture_kernel(tile=32)"));
    out->calls++;
    if(out->file)CHECK(fwrite(text,1,n,out->file)==n && fputc('\n',out->file)!=EOF);
    return !out->fail;
}
static void fixture_operations(mellum_capture_state *c) {
    uint32_t tokens[]={10,11,12,13,14};
    CHECK(mellum_capture_begin_sync(c,NULL,0,tokens,3,0));
    complete_forward(c,0,2,tokens,true);
    complete_forward(c,2,1,tokens+2,true);
    CHECK(!mellum_capture_has_logits(c)); /* BUSY hides even a completed head. */
    CHECK(mellum_capture_end_operation(c,NULL,0)==0);
    CHECK(mellum_capture_has_logits(c));
    CHECK(mellum_capture_begin_eval(c,3,1));complete_forward(c,3,1,tokens+3,true);
    CHECK(mellum_capture_end_operation(c,NULL,0)==0);
    CHECK(mellum_capture_begin_sync(c,tokens,4,tokens,5,2));complete_forward(c,4,1,tokens+4,true);
    CHECK(mellum_capture_end_operation(c,NULL,0)==0);
}
static void check_forwards(void) {
    memory_sink sink;mellum_capture_state *c=fresh(&sink);
    profile_output out={.spool=&sink};fixture_operations(c);
    CHECK(c->writer.forwards==4 && c->writer.positions==5 && c->profiles.count==1);
    CHECK(sink.n==4668 && out.calls==0);
    uint32_t tokens[]={10,11,12,13,14};size_t n=sink.n;
    CHECK(mellum_capture_begin_sync(c,tokens,5,tokens,5,3));
    CHECK(mellum_capture_end_operation(c,NULL,0)==0 && sink.n==n && mellum_capture_has_logits(c));
    CHECK(mellum_capture_finish(c,profile_append,&out));
    CHECK(out.calls==1 && sink.n==4688 && c->state==MELLUM_CAPTURE_FINISHED);
    CHECK(!memcmp(sink.bytes+4668,"END1",4) && mellum_capture_has_logits(c));
    size_t calls=sink.calls;
    CHECK(!mellum_capture_finish(c,profile_append,&out));
    CHECK(!mellum_capture_begin_eval(c,5,0));CHECK(!mellum_capture_begin_sync(c,tokens,5,tokens,5,0));
    CHECK(!mellum_capture_forward_begin(c,NULL,5,1,true));CHECK(!mellum_capture_layer(c,0));
    CHECK(!mellum_capture_forward_end(c,NULL,true,tokens,1,selected,448));
    CHECK(mellum_capture_end_operation(c,NULL,0)!=0);mellum_capture_abort(c,NULL);
    CHECK(sink.n==4688 && sink.calls==calls && out.calls==1 && c->state==MELLUM_CAPTURE_FINISHED);
    free(c);
    c=fresh(&sink);seed_prefix(c);CHECK(mellum_capture_begin_eval(c,2,0));
    complete_forward(c,2,1,tokens+2,false);CHECK(mellum_capture_end_operation(c,NULL,0)==0);
    CHECK(!mellum_capture_has_logits(c));
    CHECK(c->profiles.count==2);free(c);
}
static void check_failures(void) {
    const uint32_t tokens[]={10,11,12};
    for(unsigned bad=0;bad<21;bad++) {
        memory_sink sink;mellum_capture_state *c=fresh(&sink);mellum_profile_scope scope={0},wrong={0};
        CHECK(mellum_capture_begin_eval(c,0,0));size_t before=sink.n;
        if(bad==0)CHECK(!mellum_capture_forward_begin(c,&scope,1,1,true));
        else if(bad==1)CHECK(!mellum_capture_forward_begin(c,&scope,0,2,true));
        else if(bad==2)CHECK(!mellum_capture_forward_begin(c,&scope,UINT64_MAX,1,true));
        else if(bad==3)CHECK(mellum_capture_end_operation(c,&scope,0)!=0);
        else if(bad==4)CHECK(!mellum_capture_forward_begin(c,NULL,0,1,true));
        else {
            CHECK(mellum_capture_forward_begin(c,&scope,0,1,true));
            if(bad!=5)record_roles(&scope,true);
            if(bad==6)CHECK(!mellum_capture_layer(c,1));
            else if(bad==7)CHECK(!mellum_capture_layer(c,28));
            else if(bad==8)CHECK(mellum_capture_end_operation(c,&wrong,2)==2);
            else if(bad==9)CHECK(!mellum_capture_forward_begin(c,&wrong,0,1,true));
            else if(bad==10)mellum_capture_abort(c,&wrong);
            else {
                for(unsigned l=0;l<(bad==11 ? 27u:28u);l++)CHECK(mellum_capture_layer(c,l));
                if(bad==12)CHECK(!mellum_capture_layer(c,28));
                else if(bad==13)CHECK(!mellum_capture_finish(c,profile_append,NULL));
                else {
                    if(bad==20)selected[0]=64;
                    CHECK(!mellum_capture_forward_end(c,bad==14 ? &wrong:&scope,bad!=15,
                          bad==16 ? NULL:tokens,bad==17 ? 0:1,bad==18 ? NULL:selected,bad==19 ? 447:448));
                    selected[0]=0;
                }
            }
        }
        CHECK(c->state==MELLUM_CAPTURE_FAILED && !scope.owner && !c->profiles.active);
        CHECK(sink.n==before && !mellum_capture_has_logits(c));
        CHECK(!mellum_capture_finish(c,profile_append,NULL));
        CHECK(!mellum_capture_begin_eval(c,0,0));CHECK(sink.n==before);free(c);
    }
    /* Prefix completeness: no missing tail, no forward past the promised frontier. */
    for(unsigned bad=0;bad<3;bad++) {
        memory_sink sink;mellum_capture_state *c=fresh(&sink);mellum_profile_scope scope={0};
        CHECK(mellum_capture_begin_sync(c,NULL,0,tokens,bad==2 ? 1:3,0));
        if(bad==0)complete_forward(c,0,2,tokens,true);
        if(bad==2)CHECK(!mellum_capture_forward_begin(c,&scope,0,2,true));
        else CHECK(mellum_capture_end_operation(c,NULL,0)!=0);
        CHECK(c->state==MELLUM_CAPTURE_FAILED);free(c);
    }
    /* A foreign active scope remains usable when this controller fails. */
    memory_sink sink,other_sink;mellum_capture_state *c=fresh(&sink),*other=fresh(&other_sink);
    mellum_profile_scope foreign={0};CHECK(mellum_capture_begin_eval(other,0,0));
    CHECK(mellum_capture_forward_begin(other,&foreign,0,1,true));
    CHECK(mellum_capture_begin_eval(c,0,0));CHECK(!mellum_capture_forward_begin(c,&foreign,0,1,true));
    CHECK(foreign.owner==&other->profiles && other->profiles.active);
    record_roles(&foreign,true);for(unsigned l=0;l<28;l++)CHECK(mellum_capture_layer(other,l));
    CHECK(mellum_capture_forward_end(other,&foreign,true,tokens,1,selected,448));
    CHECK(mellum_capture_end_operation(other,NULL,0)==0);free(c);free(other);
    /* Complete frames followed by metadata/footer failure never succeed or retry. */
    for(unsigned bad=0;bad<4;bad++) {
        c=fresh(&sink);seed_prefix(c);profile_output out={.spool=&sink,.fail=bad==0};
        size_t n=sink.n;
        if(bad==2)sink.limit=n+7;
        if(bad==3)sink.limit=n;
        CHECK(!mellum_capture_finish(c,bad==1 ? NULL:profile_append,&out));
        CHECK(c->state==MELLUM_CAPTURE_FAILED && !mellum_capture_has_logits(c));
        if(bad<2)CHECK(sink.n==n);
        n=sink.n;size_t calls=sink.calls;
        CHECK(!mellum_capture_finish(c,profile_append,&out));
        CHECK(sink.n==n && sink.calls==calls);free(c);
    }
    c=fresh(&sink);CHECK(!mellum_capture_finish(c,profile_append,NULL));CHECK(sink.n==24);free(c);
    c=fresh(&sink);CHECK(mellum_capture_begin_eval(c,0,0));sink.limit=29;
    mellum_profile_scope scope={0};CHECK(mellum_capture_forward_begin(c,&scope,0,1,true));record_roles(&scope,true);
    for(unsigned l=0;l<28;l++)CHECK(mellum_capture_layer(c,l));
    CHECK(!mellum_capture_forward_end(c,&scope,true,tokens,1,selected,448));
    CHECK(sink.n==29 && c->writer.forwards==0 && !scope.owner && c->state==MELLUM_CAPTURE_FAILED);free(c);
}
static int emit_fixture(const char *mode,const char *path) {
    CHECK(!strcmp(mode,"--emit-fixture") || !strcmp(mode,"--emit-incomplete"));
    memory_sink sink;mellum_capture_state *c=fresh(&sink);fixture_operations(c);
    char profile_path[4096];CHECK(snprintf(profile_path,sizeof(profile_path),"%s.profiles",path)<(int)sizeof(profile_path));
    FILE *profiles=fopen(profile_path,"wb");CHECK(profiles);
    profile_output out={.spool=&sink,.file=profiles};
    if(!strcmp(mode,"--emit-fixture"))CHECK(mellum_capture_finish(c,profile_append,&out));
    else mellum_capture_abort(c,NULL);
    CHECK(!fclose(profiles));FILE *file=fopen(path,"wb");CHECK(file);
    CHECK(fwrite(sink.bytes,1,sink.n,file)==sink.n && !fclose(file));free(c);return 0;
}
int main(int argc, char **argv) {
    CHECK(mellum_kernel_catalog_init(&catalog,definitions,1));
    for(unsigned l=0;l<28;l++)for(unsigned r=0;r<2;r++)for(unsigned k=0;k<8;k++)selected[(l*2+r)*8+k]=k;
    if(argc==3)return emit_fixture(argv[1],argv[2]);
    check_initialization();check_operations();check_forwards();check_failures();puts("Mellum capture operation checks: PASS");return 0;
}
