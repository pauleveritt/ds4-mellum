/* Separate research executable; prepare is inspect-only, collect owns model lock. */
#include "mellum_generate_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int path(char *out,size_t size,const char *directory,const char *name) {
    int n=snprintf(out,size,"%s/%s",directory,name);return n<0 || (size_t)n>=size ? -1:0;
}
int mellum_generated_prepare(ds4_engine *e,const char *input,const char *output) {
    const char *names[]={"system.txt","user.txt","followup.txt"};char *texts[3]={0};
    char p[4096];ds4_tokens tokens[4]={{0}};int rc=-1;
    for(int i=0;i<3;i++) { if(path(p,sizeof(p),input,names[i])) goto finish;texts[i]=mellum_generated_read_text(p);if(!texts[i]) goto finish; }
    ds4_encode_chat_prompt(e,texts[0],texts[1],DS4_THINK_HIGH,&tokens[0]);
    ds4_chat_append_message(e,&tokens[1],"user",texts[2]);
    ds4_chat_append_assistant_prefix(e,&tokens[1],DS4_THINK_HIGH);
    ds4_tokenize_rendered_chat(e,"<|im_end|>\n",&tokens[2]);
    ds4_tokenize_rendered_chat(e,"\n",&tokens[3]);
    if(tokens[0].len+tokens[1].len+tokens[2].len+4096+1024+64>8192) goto finish;
    const char *base[]={"initial","append","close","newline"};
    for(int i=0;i<4;i++) {
        char name[64];snprintf(name,sizeof(name),"%s.tokens",base[i]);
        if(path(p,sizeof(p),output,name) || mellum_generated_write_tokens(p,&tokens[i])) goto finish;
        snprintf(name,sizeof(name),"%s.rendered",base[i]);
        if(path(p,sizeof(p),output,name) || mellum_generated_write_rendered(p,e,&tokens[i])) goto finish;
    }
    rc=0;
finish:
    for(int i=0;i<4;i++) ds4_tokens_free(&tokens[i]);
    for(int i=0;i<3;i++) free(texts[i]);
    return rc;
}
int main(int argc,char **argv) {
    if(argc!=5 || (strcmp(argv[1],"prepare") && strcmp(argv[1],"collect"))) {
        fprintf(stderr,"usage: mellum_generate prepare|collect MODEL INPUT_DIR OUTPUT_DIR\n");return 2;
    }
    int preparing=!strcmp(argv[1],"prepare");
#ifdef DS4_NO_GPU
    if(!preparing) { fprintf(stderr,"collect requires the Metal build\n");return 2; }
#endif
    ds4_engine_options options={0};options.model_path=argv[2];options.context_size=8192;
    options.prefill_chunk=1024;options.backend=preparing ? DS4_BACKEND_CPU:DS4_BACKEND_METAL;options.inspect_only=preparing;
    ds4_engine *engine=NULL;
#ifdef DS4_NO_GPU
    if(mellum_generated_inspect_open(&engine,&options)) return 1;
#else
    if(ds4_engine_open(&engine,&options)) return 1;
#endif
    int rc=-1;
    if(ds4_engine_vocab_size(engine)==98304 && ds4_engine_layer_count(engine)==28) {
        if(preparing) rc=mellum_generated_prepare(engine,argv[3],argv[4]);
#ifndef DS4_NO_GPU
        else rc=mellum_generated_collect(engine,argv[3],argv[4]);
#endif
    }
    ds4_engine_close(engine);
    if(rc) fprintf(stderr,"generated native operation failed; partial output retained\n");
    return rc ? 1:0;
}
