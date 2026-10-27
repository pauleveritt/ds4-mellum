/* Real observer/registry, supplied encoding notifications; never GPU evidence. */
#include "ds4_mellum_observer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); } } while (0)
static mellum_profiles *registry(void) {
    mellum_profiles *r=calloc(1,sizeof(*r));CHECK(r);
    CHECK(mellum_profiles_init(r,mellum_observer_catalog()));return r;
}
static void exact(mellum_profile_scope *scope,uint32_t nsg,bool head,bool reverse) {
    (void)scope;
    const char *names[]={"kernel_get_rows_q8_0_f32","kernel_rms_norm_mul_f32_4","kernel_qwen4_multi_gemv",
        "kernel_qwen4_router_topk","kernel_qwen4_moe_reduce","kernel_mellum_attn_swa",
        "kernel_mellum_qk_norm_rope","kernel_mellum_ring_commit","kernel_cpy_contig_f32_f16_4","kernel_add2_f32"};
    for(unsigned j=0;j<10;j++)CHECK(mellum_observer_encode(names[reverse ? 9-j:j],0,NULL));
    uint32_t mv[]={nsg},moe[]={0,8,0,2048,2};
    CHECK(mellum_observer_encode("kernel_mul_mv_q8_0_f32",1,mv));
    CHECK(mellum_observer_encode("kernel_qwen4_moe_mid",5,moe));
    CHECK(mellum_observer_encode("kernel_qwen4_moe_down",5,moe));
    if(head) {
        mellum_observer_projection_role(true);
        CHECK(mellum_observer_encode("kernel_mul_mv_q8_0_f32",1,mv));
        mellum_observer_projection_role(false);
    }
}
static uint32_t observed(mellum_profiles *r,uint32_t nsg,bool head,bool reverse) {
    mellum_profile_scope scope={0};uint32_t id;
    CHECK(mellum_profile_scope_enter(&scope,r,head));CHECK(mellum_observer_enter(&scope));
    exact(&scope,nsg,head,reverse);CHECK(mellum_observer_leave(&scope));
    CHECK(mellum_profile_scope_exit(&scope,r,true,&id));return id;
}
int main(void) {
    CHECK(mellum_observer_catalog());
    const mellum_kernel_catalog *catalog=mellum_observer_catalog();mellum_kernel_catalog validated;
    CHECK(mellum_kernel_catalog_init(&validated,catalog->defs,catalog->count));
    CHECK(mellum_observer_encode(NULL,SIZE_MAX,NULL));
    mellum_observer_projection_role(true); /* must reset at next enter */
    mellum_profiles *r=registry();
    CHECK(observed(r,4,true,false)==0);CHECK(observed(r,4,true,true)==0);
    CHECK(observed(r,8,true,false)==1);CHECK(observed(r,4,false,false)==2);
    char description[16384];CHECK(mellum_profile_describe(&r->profiles[0],description,sizeof(description)));
    CHECK(strstr(description,";PROJECTION:kernel_mul_mv_q8_0_f32(nsg=4)"));
    CHECK(strstr(description,";HEAD:kernel_mul_mv_q8_0_f32(nsg=4)"));
    CHECK(strstr(description,";MID:kernel_qwen4_moe_mid(specialized=0,type=8,shared_type=0,dim=2048,rows=2)"));
    CHECK(mellum_profile_describe(&r->profiles[2],description,sizeof(description)));
    CHECK(strstr(description,"head=not-encoded") && !strstr(description,";HEAD:"));free(r);
    for(unsigned bad=0;bad<5;bad++) {
        r=registry();mellum_profiles *other=registry();mellum_profile_scope scope={0},foreign={0};uint32_t id;
        CHECK(mellum_profile_scope_enter(&scope,r,true));CHECK(mellum_observer_enter(&scope));
        CHECK(mellum_profile_scope_enter(&foreign,other,true));
        CHECK(!mellum_observer_enter(&foreign));CHECK(!mellum_observer_leave(&foreign));
        CHECK(!mellum_observer_enter(&scope));CHECK(!other->failed && other->active);
        uint32_t v[]={4};
        switch(bad) {
        case 0:CHECK(!mellum_observer_encode("not_a_kernel",0,NULL));break;
        case 1:CHECK(!mellum_observer_encode(NULL,0,NULL));break;
        case 2:CHECK(!mellum_observer_encode("kernel_mul_mv_q8_0_f32",0,v));break;
        case 3:CHECK(!mellum_observer_encode("kernel_mul_mv_q8_0_f32",1,NULL));break;
        case 4:CHECK(!mellum_observer_encode("kernel_get_rows_q8_0_f32",7,v));break;
        }
        CHECK(r->failed && !mellum_observer_encode("kernel_get_rows_q8_0_f32",0,NULL));
        CHECK(mellum_observer_leave(&scope));CHECK(!mellum_profile_scope_exit(&scope,r,true,&id));
        CHECK(!scope.owner && !r->active);
        CHECK(mellum_observer_enter(&foreign));CHECK(mellum_observer_leave(&foreign));
        CHECK(!mellum_profile_scope_exit(&foreign,other,true,&id)); /* missing roles */
        CHECK(mellum_observer_encode("unknown_when_disabled",0,NULL));free(r);free(other);
    }
    r=registry();CHECK(observed(r,4,false,false)==0);free(r);
    CHECK(!mellum_observer_enter(NULL) && !mellum_observer_leave(NULL));
    puts("Mellum encoding observer checks: PASS");return 0;
}
