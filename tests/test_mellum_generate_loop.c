#include "../tools/mellum_generate_loop.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void ds4_tokens_push(ds4_tokens *v,int token) { v->v=realloc(v->v,(size_t)(v->len+1)*sizeof(int));assert(v->v);v->v[v->len++]=token; }
typedef struct { int position,mode,fail; } fake;
static int sync_tokens(void *ctx,const ds4_tokens *v,unsigned segment) { (void)segment;((fake *)ctx)->position=v->len;return 0; }
static int eval(void *ctx,int token,unsigned segment) { (void)token;(void)segment;((fake *)ctx)->position++;return 0; }
static int logits(void *ctx,float *values,int n) {
    fake *f=ctx;for(int i=0;i<n;i++) values[i]=0;
    values[f->position<3 ? 2:5]=1;
    if(f->mode==1) values[0]=NAN;
    if(f->mode==2 && f->position<3) values[3]=1; /* equal logits, lowest id wins */
    if(f->mode==3) values[7]=2; /* parity mismatch */
    return n;
}
static int stop(void *ctx,int token) { return token==5 ? (((fake *)ctx)->fail ? DS4_STOP_PROTOCOL:DS4_STOP_EOS):0; }
int main(void) {
    fake f={.mode=2};mellum_generated_backend backend={&f,sync_tokens,eval,logits,stop};
    ds4_tokens history={0};ds4_tokens_push(&history,1);
    FILE *ops=tmpfile(),*gold=tmpfile(),*emitted=tmpfile();assert(ops && gold && emitted);
    float values[8];int count=0,reason=0;
    assert(!mellum_generated_sync(&backend,&history,0,ops,gold,values,8));
    f.mode=2;
    assert(!mellum_generated_turn(&backend,&history,10,1,ops,gold,emitted,values,8,&count,&reason));
    assert(count==3 && reason==DS4_STOP_EOS && history.len==4 && history.v[1]==2 && history.v[3]==5);
    fflush(ops);fflush(gold);rewind(ops);rewind(gold);
    f=(fake){.mode=2};assert(!mellum_generated_parity(&backend,&history,ops,gold,values,8));
    rewind(ops);rewind(gold);f=(fake){.mode=3};assert(mellum_generated_parity(&backend,&history,ops,gold,values,8));
    rewind(ops);rewind(gold);f=(fake){.mode=2};fputc(0,gold);fflush(gold);rewind(gold);
    assert(mellum_generated_parity(&backend,&history,ops,gold,values,8));
    f=(fake){.mode=1};assert(mellum_generated_sync(&backend,&history,0,ops,gold,values,8));
    f=(fake){.fail=1,.position=3};assert(!logits(&f,values,8) || values[5]==1);
    assert(mellum_generated_turn(&backend,&history,2,1,ops,gold,emitted,values,8,&count,&reason));
    f=(fake){0};assert(logits(&f,values,8)==8);
    assert(!mellum_generated_turn(&backend,&history,1,3,ops,gold,emitted,values,8,&count,&reason));
    assert(count==1 && reason==0);
    FILE *readonly=fopen("/dev/null","rb");assert(readonly);
    assert(mellum_generated_sync(&backend,&history,0,readonly,gold,values,8));fclose(readonly);
    fclose(ops);fclose(gold);fclose(emitted);free(history.v);
    puts("generated loop/parity boundaries: PASS");return 0;
}
