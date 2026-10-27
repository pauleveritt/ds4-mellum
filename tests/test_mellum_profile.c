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
int main(void) {
    check_catalog();check_profiles();check_refusals();
    puts("Mellum profile set checks: PASS");return 0;
}
