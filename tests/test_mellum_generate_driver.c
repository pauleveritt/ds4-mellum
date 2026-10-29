/* Fake engine, real research driver: no model, lock, backend, or GPU. */
#include "../tools/mellum_generate_driver.h"
#include "../ds4_mellum_capture.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static int mode;
static FILE *fault_stream;
FILE *mellum_test_open(const char *path,const char *flags) {
    FILE *stream=fopen(path,flags);
    if((mode==1 && strstr(path,"/owned.routes")) || (mode==4 && strstr(path,"/result"))) fault_stream=stream;
    return stream;
}
int mellum_test_close(FILE *stream) {
    int fail=stream==fault_stream && (mode==1 || mode==4);
    int rc=fclose(stream);if(fail) { fault_stream=NULL;return EOF; }return rc;
}
void ds4_tokens_push(ds4_tokens *tokens,int token) {
    tokens->v=realloc(tokens->v,(size_t)(tokens->len+1)*sizeof(int));assert(tokens->v);tokens->v[tokens->len++]=token;
}
void ds4_tokens_free(ds4_tokens *tokens) { free(tokens->v);*tokens=(ds4_tokens){0}; }
char *ds4_token_text(ds4_engine *engine,int token,size_t *n) { (void)engine;(void)token;*n=1;return strdup("x"); }
int ds4_token_stop_class(ds4_engine *engine,int token) { (void)engine;(void)token;return mode==2 ? DS4_STOP_PROTOCOL:DS4_STOP_EOS; }
uint32_t ds4_engine_prefill_chunk(ds4_engine *engine) { (void)engine;return 1024; }
int ds4_gpu_tensor_tiles_available(void) { return 0; }
int ds4_gpu_device_is_pre_m5_apple_silicon(void) { return 0; }
int ds4_gpu_device_is_m5_apple_silicon(void) { return 0; }
struct ds4_mellum_capture { int unused; };
struct ds4_session { int unused; };
int ds4_mellum_capture_create(ds4_mellum_capture **out,ds4_engine *engine,int n,const ds4_mellum_capture_options *options,char *error,size_t size) {
    (void)engine;(void)n;(void)options;(void)error;(void)size;*out=calloc(1,sizeof(**out));return !*out;
}
int ds4_mellum_capture_sync(ds4_mellum_capture *capture,const ds4_tokens *tokens,uint32_t segment,char *error,size_t size) {
    (void)capture;(void)tokens;(void)segment;(void)error;(void)size;return 0;
}
int ds4_mellum_capture_eval(ds4_mellum_capture *capture,int token,uint32_t segment,char *error,size_t size) {
    (void)capture;(void)token;(void)segment;
    if(mode==3) { snprintf(error,size,"fixture backend eval failed");return -1; }return 0;
}
static int values(float *out,int n) { for(int i=0;i<n;i++) out[i]=0;out[7]=1;return n; }
int ds4_mellum_capture_copy_logits(ds4_mellum_capture *capture,float *out,int n) { (void)capture;return values(out,n); }
int ds4_mellum_capture_finish(ds4_mellum_capture *capture,mellum_profile_sink sink,void *ctx,char *error,size_t size) {
    (void)capture;(void)sink;(void)ctx;(void)error;(void)size;return 0;
}
void ds4_mellum_capture_free(ds4_mellum_capture *capture) { free(capture); }
int ds4_session_create(ds4_session **out,ds4_engine *engine,int n) { (void)engine;(void)n;*out=calloc(1,sizeof(**out));return !*out; }
int ds4_session_sync(ds4_session *session,const ds4_tokens *tokens,char *error,size_t size) { (void)session;(void)tokens;(void)error;(void)size;return 0; }
int ds4_session_eval(ds4_session *session,int token,char *error,size_t size) { (void)session;(void)token;(void)error;(void)size;return 0; }
int ds4_session_copy_logits(ds4_session *session,float *out,int n) { (void)session;return values(out,n); }
void ds4_session_free(ds4_session *session) { free(session); }
int mellum_generated_prepare(ds4_engine *engine,const char *input,const char *output) {
    (void)engine;(void)input;
    const char *names[]={"initial","append","close","newline"};const char *text[]={"1\n","3\n","5 6\n","6\n"};
    for(int i=0;i<4;i++) {
        char path[512];snprintf(path,sizeof(path),"%s/%s.tokens",output,names[i]);FILE *f=fopen(path,"wbx");assert(f);fputs(text[i],f);assert(!fclose(f));
        snprintf(path,sizeof(path),"%s/%s.rendered",output,names[i]);f=fopen(path,"wbx");assert(f);fputs("x",f);assert(!fclose(f));
    }return 0;
}
static int exists(const char *dir,const char *name) { char path[512];snprintf(path,sizeof(path),"%s/%s",dir,name);return !access(path,F_OK); }
static void contains(const char *dir,const char *name,const char *needle) {
    char path[512];snprintf(path,sizeof(path),"%s/%s",dir,name);char *text=mellum_generated_read_text(path);assert(text);assert(strstr(text,needle));free(text);
}
int main(int argc,char **argv) {
    char root[]="/tmp/mellum-gen-driver-XXXXXX";assert(mkdtemp(root));char input[512];snprintf(input,sizeof(input),"%s/input",root);assert(!mkdir(input,0700));
    assert(!mellum_generated_prepare(NULL,input,input));
    for(mode=0;mode<=4;mode++) {
        if(argc==2 && mode!=atoi(argv[1])) continue;
        char output[512];snprintf(output,sizeof(output),"%s/run%d",root,mode);assert(!mkdir(output,0700));
        int rc=mellum_generated_collect(NULL,input,output);
        if(!mode) { assert(!rc);assert(exists(output,"result.json"));continue; }
        assert(rc);assert(!exists(output,"result.json"));
        assert(exists(output,"failure.json"));
        if(mode==2) { contains(output,"failure.json","protocol-stop");contains(output,"failure.json","\"segment\":1");contains(output,"failure.json","\"token\":7"); }
        if(mode==3) contains(output,"failure.json","fixture backend eval failed");
    }
    printf("native driver close/protocol/backend failure boundaries: PASS (tiny retained fixtures %s)\n",root);return 0;
}
