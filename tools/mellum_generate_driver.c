#include "mellum_generate_driver.h"
#include "../ds4_gpu.h"
#include "../tests/mellum_capture_receipt.h"
#include "../ds4_mellum_capture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
typedef struct { ds4_engine *engine;ds4_mellum_capture *capture;ds4_session *ordinary;char err[512]; } backend;
static int sync_tokens(void *ctx,const ds4_tokens *tokens,unsigned segment) {
    backend *b=ctx;
    return b->capture ? ds4_mellum_capture_sync(b->capture,tokens,segment,b->err,sizeof(b->err)):
        ds4_session_sync(b->ordinary,tokens,b->err,sizeof(b->err));
}
static int eval(void *ctx,int token,unsigned segment) {
    backend *b=ctx;
    return b->capture ? ds4_mellum_capture_eval(b->capture,token,segment,b->err,sizeof(b->err)):
        ds4_session_eval(b->ordinary,token,b->err,sizeof(b->err));
}
static int logits(void *ctx,float *out,int n) {
    backend *b=ctx;return b->capture ? ds4_mellum_capture_copy_logits(b->capture,out,n):ds4_session_copy_logits(b->ordinary,out,n);
}
static int stop(void *ctx,int token) { return ds4_token_stop_class(((backend *)ctx)->engine,token); }
static size_t spool(void *ctx,const void *data,size_t n) { return fwrite(data,1,n,ctx); }
static bool profile(void *ctx,uint32_t id,const char *text,size_t n) { return n==strlen(text) && fprintf(ctx,"%u %s\n",id,text)>0; }
static FILE *open_output(const char *dir,const char *name,const char *mode) {
    char p[4096];int n=snprintf(p,sizeof(p),"%s/%s",dir,name);
    return n<0 || (size_t)n>=sizeof(p) ? NULL:fopen(p,mode);
}
static int load(const char *dir,const char *name,ds4_tokens *out) {
    char p[4096];int n=snprintf(p,sizeof(p),"%s/%s",dir,name);
    return n<0 || (size_t)n>=sizeof(p) ? -1:mellum_generated_read_tokens(p,out);
}
static int same(const ds4_tokens *a,const ds4_tokens *b) { return a->len==b->len && !memcmp(a->v,b->v,(size_t)a->len*sizeof(int)); }
int mellum_generated_collect(ds4_engine *engine,const char *input,const char *output) {
    /* Recreate every prepared token/rendered input in a separate private child. */
    char check[4096];int nn=snprintf(check,sizeof(check),"%s/recreated",output);
    if(nn<0 || (size_t)nn>=sizeof(check) || mkdir(check,0700) || mellum_generated_prepare(engine,input,check)) return -1;
    const char *names[]={"initial.tokens","append.tokens","close.tokens","newline.tokens"};
    ds4_tokens prepared[4]={{0}},derived[4]={{0}},history={0};int rc=-1;
    for(int i=0;i<4;i++) if(load(input,names[i],&prepared[i]) || load(check,names[i],&derived[i]) || !same(&prepared[i],&derived[i])) goto clean_tokens;
    if(prepared[0].len+prepared[1].len+prepared[2].len+5120+64>8192) goto clean_tokens;
    for(int i=0;i<prepared[0].len;i++) ds4_tokens_push(&history,prepared[0].v[i]);
    const char *files[]={"owned.routes","owned.profiles","gold.f32","operations.txt","emitted.txt"};
    FILE *streams[5]={0};for(int i=0;i<5;i++) { streams[i]=open_output(output,files[i],i==2 || i==3 ? "w+bx":"wbx");if(!streams[i]) goto close_streams; }
    backend ctx={.engine=engine};
    mellum_generated_backend b={&ctx,sync_tokens,eval,logits,stop};
    ds4_mellum_capture_options options={.spool_sink=spool,.spool_context=streams[0]};
    float *values=malloc(98304*sizeof(float));if(!values) goto close_streams;
    int counts[2]={0},reasons[2]={0},host_tokens=0;
    if(ds4_mellum_capture_create(&ctx.capture,engine,8192,&options,ctx.err,sizeof(ctx.err)) ||
        mellum_generated_sync(&b,&history,0,streams[3],streams[2],values,98304) ||
        mellum_generated_turn(&b,&history,4096,1,streams[3],streams[2],streams[4],values,98304,&counts[0],&reasons[0])) goto close_handle;
    int response0_end=history.len;
    ds4_tokens *separator=reasons[0] ? &prepared[3]:&prepared[2];host_tokens=separator->len;
    for(int i=0;i<separator->len;i++) ds4_tokens_push(&history,separator->v[i]);
    for(int i=0;i<prepared[1].len;i++) ds4_tokens_push(&history,prepared[1].v[i]);
    int response1_start=history.len;
    if(mellum_generated_sync(&b,&history,2,streams[3],streams[2],values,98304) ||
        mellum_generated_turn(&b,&history,1024,3,streams[3],streams[2],streams[4],values,98304,&counts[1],&reasons[1]) ||
        ds4_mellum_capture_finish(ctx.capture,profile,streams[1],ctx.err,sizeof(ctx.err))) goto close_handle;
    ds4_mellum_capture_free(ctx.capture);ctx.capture=NULL;
    if(fflush(streams[2]) || fflush(streams[3]) || fseek(streams[2],0,SEEK_SET) || fseek(streams[3],0,SEEK_SET) ||
        ds4_session_create(&ctx.ordinary,engine,8192) || mellum_generated_parity(&b,&history,streams[3],streams[2],values,98304)) goto close_handle;
    char p[4096];snprintf(p,sizeof(p),"%s/tokens.txt",output);
    if(mellum_generated_write_tokens(p,&history)) goto close_handle;
    for(int turn=0;turn<2;turn++) {
        ds4_tokens answer={.v=history.v+(turn ? response1_start:prepared[0].len),.len=counts[turn]};
        snprintf(p,sizeof(p),"%s/response%d.txt",output,turn);
        if(mellum_generated_write_rendered(p,engine,&answer)) goto close_handle;
    }
    FILE *runtime=open_output(output,"runtime.json","wbx");
    if(!runtime) goto close_handle;
    bool wrote=mellum_capture_receipt_write(runtime,8192,ds4_engine_prefill_chunk(engine),ds4_gpu_tensor_tiles_available(),
        ds4_gpu_device_is_pre_m5_apple_silicon(),ds4_gpu_device_is_m5_apple_silicon());
    if(fclose(runtime) || !wrote) goto close_handle;
    FILE *result=open_output(output,"result.json","wbx");if(!result) goto close_handle;
    int written=fprintf(result,"{\"complete\":true,\"parity\":true,\"counts\":[%d,%d],\"stop_classes\":[%d,%d],\"host_separator_tokens\":%d,\"response0_end\":%d,\"response1_start\":%d,\"tokens\":%d}\n",counts[0],counts[1],reasons[0],reasons[1],host_tokens,response0_end,response1_start,history.len);
    if(fclose(result) || written<0) goto close_handle;
    rc=0;
close_handle:
    ds4_mellum_capture_free(ctx.capture);ds4_session_free(ctx.ordinary);free(values);
close_streams:
    for(int i=0;i<5;i++) if(streams[i] && fclose(streams[i])) rc=-1;
clean_tokens:
    for(int i=0;i<4;i++) { ds4_tokens_free(&prepared[i]);ds4_tokens_free(&derived[i]); }
    ds4_tokens_free(&history);return rc;
}
