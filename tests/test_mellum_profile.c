/* Supplied kernel definitions are synthetic fixtures, not Metal observations. */
#include "ds4_mellum_profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c); exit(1); } } while (0)
static const mellum_kernel_def definitions[] = {
    {"kernel_short", 1, {"tile"}}, {"kernel_tail", 2, {"tile", "type"}}
};
static mellum_kernel_catalog catalog;
static mellum_kernel_descriptor desc(unsigned role, unsigned tile) {
    return (mellum_kernel_descriptor){.function=0, .role=role, .value_count=1, .values={tile}};
}
static void complete(mellum_profile *p, bool head, unsigned tile) {
    CHECK(mellum_profile_init(p, &catalog, head));
    for (unsigned r=0; r<7; r++) CHECK(mellum_profile_record(p,desc(r,tile)));
    if (head) CHECK(mellum_profile_record(p,(mellum_kernel_descriptor){1,MELLUM_ROLE_HEAD,2,{64,8}}));
}
static void check_catalog(void) {
    CHECK(mellum_kernel_catalog_init(&catalog,definitions,2));
    CHECK(!mellum_kernel_catalog_init(NULL,definitions,2));
    mellum_kernel_catalog bad;
    CHECK(!mellum_kernel_catalog_init(&bad,NULL,2));
    CHECK(!mellum_kernel_catalog_init(&bad,definitions,0));
    CHECK(!mellum_kernel_catalog_init(&bad,definitions,65537));
    char long_name[97], long_label[33];
    memset(long_name,'a',96);long_name[96]=0;
    memset(long_label,'a',32);long_label[32]=0;
    for (unsigned i=0;i<10;i++) {
        mellum_kernel_def d[2]={definitions[0],definitions[1]};
        switch(i) {
        case 0:d[0].name=NULL;break;
        case 1:d[0].name="";break;
        case 2:d[0].name="0kernel";break;
        case 3:d[0].name="kernel bad";break;
        case 4:d[0].name=long_name;break;
        case 5:d[0].value_count=7;break;
        case 6:d[0].value_names[0]=NULL;break;
        case 7:d[1].value_names[1]="tile";break;
        case 8:d[1].name="kernel_short";break;
        case 9:d[0].value_names[0]=long_label;break;
        }
        CHECK(!mellum_kernel_catalog_init(&bad,d,2));
        CHECK(!mellum_profile_init(&(mellum_profile){0},&bad,false));
    }
}
static void check_profiles(void) {
    mellum_profile p,q;
    complete(&p,true,32);CHECK(mellum_profile_valid(&p));
    CHECK(mellum_profile_init(&q,&catalog,true));
    CHECK(!mellum_profile_valid(&q));
    CHECK(mellum_profile_record(&q,(mellum_kernel_descriptor){1,MELLUM_ROLE_HEAD,2,{64,8}}));
    for (int r=6;r>=0;r--) CHECK(mellum_profile_record(&q,desc(r,32)));
    CHECK(mellum_profile_valid(&q) && q.count==8);
    for (unsigned i=0;i<8;i++) {
        CHECK(p.items[i].function==q.items[i].function && p.items[i].role==q.items[i].role);
        CHECK(p.items[i].values[0]==q.items[i].values[0]);
    }
    mellum_kernel_descriptor duplicate=desc(0,32);
    for(unsigned i=1;i<6;i++) duplicate.values[i]=UINT32_MAX;
    CHECK(mellum_profile_record(&p,duplicate) && p.count==8);
    CHECK(mellum_profile_record(&p,desc(0,33)) && p.count==9);
    complete(&q,false,0);CHECK(mellum_profile_valid(&q));
    CHECK(mellum_profile_record(&q,desc(MELLUM_ROLE_HEAD,0)));
    CHECK(!mellum_profile_valid(&q));
    complete(&q,true,0);q.count--;CHECK(!mellum_profile_valid(&q));
    CHECK(sizeof(mellum_kernel_descriptor)<=32);
}
static void check_refusals(void) {
    mellum_profile p;
    for (unsigned i=0;i<3;i++) {
        complete(&p,false,0);
        mellum_kernel_descriptor d=desc(0,0);
        if(i==0)d.function=2;
        if(i==1)d.role=9;
        if(i==2)d.value_count=2;
        CHECK(!mellum_profile_record(&p,d));
        CHECK(!mellum_profile_valid(&p));
        CHECK(!mellum_profile_record(&p,desc(0,0)));
    }
    complete(&p,false,0);
    for(unsigned i=0;i<57;i++) CHECK(mellum_profile_record(&p,desc(MELLUM_ROLE_OTHER,i)));
    CHECK(p.count==64 && mellum_profile_valid(&p));
    CHECK(mellum_profile_record(&p,desc(MELLUM_ROLE_OTHER,56)));
    CHECK(!mellum_profile_record(&p,desc(MELLUM_ROLE_OTHER,57)));
    CHECK(!mellum_profile_valid(&p));
}
static void observe(mellum_profile_scope *scope, bool head, unsigned tile, bool reverse) {
    for (int i=0;i<7;i++) {
        mellum_kernel_descriptor d=desc(reverse ? 6-i:i,tile);
        d.values[5]=UINT32_MAX;
        CHECK(mellum_profile_scope_record(scope,d));
    }
    if(head) CHECK(mellum_profile_scope_record(scope,(mellum_kernel_descriptor){1,MELLUM_ROLE_HEAD,2,{64,8}}));
}
static uint32_t register_profile(mellum_profiles *r, mellum_profile_scope *scope,
                                 bool head, unsigned tile, bool reverse) {
    uint32_t id=UINT32_MAX;
    CHECK(mellum_profile_scope_enter(scope,r,head));
    observe(scope,head,tile,reverse);
    CHECK(mellum_profile_scope_exit(scope,r,true,&id));
    CHECK(!scope->owner && !r->active);
    return id;
}
static void check_registry(void) {
    mellum_profiles *a=calloc(1,sizeof(*a)),*b=calloc(1,sizeof(*b));
    mellum_profile_scope scope={0},second={0};
    CHECK(a && b && mellum_profiles_init(a,&catalog) && mellum_profiles_init(b,&catalog));
    CHECK(register_profile(a,&scope,true,0,false)==0);
    CHECK(register_profile(a,&scope,true,0,true)==0 && a->count==1);
    CHECK(register_profile(a,&scope,false,0,false)==1);
    CHECK(register_profile(a,&scope,true,1,false)==2);
    CHECK(register_profile(b,&scope,true,1,false)==0 && b->count==1);
    CHECK(mellum_profile_scope_enter(&scope,a,true));
    CHECK(!mellum_profile_scope_enter(&scope,b,true) && scope.owner==a);
    CHECK(!mellum_profile_scope_enter(&second,a,true) && !second.owner);
    CHECK(!mellum_profile_scope_exit(&scope,b,false,NULL) && scope.owner==a);
    observe(&scope,true,0,false);
    uint32_t id=99;
    CHECK(mellum_profile_scope_exit(&scope,a,true,&id) && id==0);
    free(a);free(b);
    a=calloc(1,sizeof(*a));CHECK(a && mellum_profiles_init(a,&catalog));
    for(unsigned i=0;i<128;i++) CHECK(register_profile(a,&scope,true,i,false)==i);
    CHECK(a->count==128 && register_profile(a,&scope,true,127,true)==127);
    CHECK(mellum_profile_scope_enter(&scope,a,true));observe(&scope,true,128,false);
    CHECK(!mellum_profile_scope_exit(&scope,a,true,&id));
    CHECK(!scope.owner && a->failed && !a->active);
    CHECK(!mellum_profile_scope_enter(&scope,a,true));free(a);
    for(unsigned failure=0;failure<4;failure++) {
        a=calloc(1,sizeof(*a));CHECK(a && mellum_profiles_init(a,&catalog));
        CHECK(mellum_profile_scope_enter(&scope,a,true));
        if(failure==0) CHECK(!mellum_profile_scope_record(&scope,desc(9,0)));
        else if(failure>1) observe(&scope,true,0,false);
        CHECK(!mellum_profile_scope_exit(&scope,a,failure!=2,failure==3 ? NULL:&id));
        CHECK(!scope.owner && !a->active && a->failed);
        CHECK(!mellum_profile_scope_record(&scope,desc(0,0)));
        CHECK(!mellum_profile_scope_enter(&scope,a,true));free(a);
    }
}
typedef struct { unsigned calls, fail_at; char first[MELLUM_PROFILE_TEXT]; } catalog_sink;
static bool collect_profile(void *ctx,uint32_t id,const char *text,size_t n) {
    catalog_sink *sink=ctx;
    CHECK(id==sink->calls && text[n]==0 && strlen(text)==n);
    if(sink->calls++==0) memcpy(sink->first,text,n+1);
    return sink->calls!=sink->fail_at;
}
static void check_descriptions(void) {
    mellum_profile p;complete(&p,true,32);
    char text[MELLUM_PROFILE_TEXT];
    CHECK(mellum_profile_describe(&p,text,sizeof(text)));
    CHECK(!strcmp(text,"profile-v1;head=encoded;EMBED:kernel_short(tile=32);PROJECTION:kernel_short(tile=32);ROUTER:kernel_short(tile=32);MID:kernel_short(tile=32);DOWN:kernel_short(tile=32);REDUCE:kernel_short(tile=32);ATTENTION:kernel_short(tile=32);HEAD:kernel_tail(tile=64,type=8)"));
    complete(&p,false,0);CHECK(mellum_profile_describe(&p,text,sizeof(text)));
    CHECK(!strncmp(text,"profile-v1;head=not-encoded;",27));
    struct { char tiny[8],canary[4]; } buffer;
    memset(&buffer,'X',sizeof(buffer));
    CHECK(!mellum_profile_describe(&p,buffer.tiny,sizeof(buffer.tiny)));
    CHECK(buffer.tiny[0]==0 && !memcmp(buffer.canary,"XXXX",4));
    CHECK(!mellum_profile_describe(&p,NULL,0));
    CHECK(!mellum_profile_describe(&p,text,0));
    mellum_profiles *r=calloc(1,sizeof(*r));mellum_profile_scope scope={0};
    CHECK(r && mellum_profiles_init(r,&catalog));catalog_sink sink={0};
    CHECK(!mellum_profiles_emit(r,collect_profile,&sink) && sink.calls==0);
    CHECK(register_profile(r,&scope,true,0,false)==0);
    CHECK(register_profile(r,&scope,true,0,true)==0);
    CHECK(register_profile(r,&scope,true,1,false)==1);
    CHECK(mellum_profiles_emit(r,collect_profile,&sink) && sink.calls==2);
    sink=(catalog_sink){.fail_at=1};
    CHECK(!mellum_profiles_emit(r,collect_profile,&sink) && sink.calls==1 && r->failed);
    CHECK(!mellum_profiles_emit(r,collect_profile,&sink) && sink.calls==1);free(r);
    /* A valid, bounded descriptor set can still exceed the text output cap. */
    char name[96],label[32];memset(name,'n',95);name[95]=0;memset(label,'v',31);label[31]=0;
    char labels[6][32];mellum_kernel_def large={.name=name,.value_count=6};
    for(unsigned i=0;i<6;i++) {memcpy(labels[i],label,32);labels[i][0]='a'+i;large.value_names[i]=labels[i];}
    mellum_kernel_catalog huge;CHECK(mellum_kernel_catalog_init(&huge,&large,1));
    r=calloc(1,sizeof(*r));CHECK(r && mellum_profiles_init(r,&huge));
    CHECK(mellum_profile_scope_enter(&scope,r,false));
    for(unsigned i=0;i<64;i++) {
        mellum_kernel_descriptor d={0,i<7 ? i:MELLUM_ROLE_OTHER,6,{i,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}};
        CHECK(mellum_profile_scope_record(&scope,d));
    }
    uint32_t id;CHECK(mellum_profile_scope_exit(&scope,r,true,&id));
    sink=(catalog_sink){0};CHECK(!mellum_profiles_emit(r,collect_profile,&sink));
    CHECK(r->failed && sink.calls==0);free(r);
}
int main(void) {
    check_catalog();check_profiles();check_refusals();check_registry();check_descriptions();
    puts("Mellum profile set checks: PASS");return 0;
}
