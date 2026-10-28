/* Native sparse fixture exercises the actual inspect/tokenizer boundary. */
#include "../tools/mellum_generate_inspect.c"
#define main mellum_generated_cli_main
#include "../tools/mellum_generate.c"
#undef main
#include "mellum_fixture.h"
#include <assert.h>
int main(void) {
    char fixture[512];mellum_fixture_write(MF_NONE,fixture,sizeof(fixture));
    ds4_engine_options opt={0};opt.model_path=fixture;opt.backend=DS4_BACKEND_CPU;opt.inspect_only=true;opt.context_size=8192;opt.prefill_chunk=1024;
    ds4_engine *plain=NULL;assert(!ds4_engine_open(&plain,&opt));
    assert(ds4_engine_vocab_size(plain)==0);ds4_engine_close(plain);
    ds4_engine *ready=NULL;assert(!mellum_generated_inspect_open(&ready,&opt));
    assert(ds4_engine_vocab_size(ready)==291 && ds4_engine_layer_count(ready)==28);
    char root[]="/tmp/mellum-gen-prepare-XXXXXX";assert(mkdtemp(root));
    const char *names[]={"system.txt","user.txt","followup.txt"};
    for(int i=0;i<3;i++) { char p[512];snprintf(p,sizeof(p),"%s/%s",root,names[i]);FILE *f=fopen(p,"wbx");assert(f);fputs("Hello",f);assert(!fclose(f)); }
    assert(!mellum_generated_prepare(ready,root,root));
    char p[512];snprintf(p,sizeof(p),"%s/initial.tokens",root);ds4_tokens ids={0};assert(!mellum_generated_read_tokens(p,&ids));assert(ids.len>0);ds4_tokens_free(&ids);
    snprintf(p,sizeof(p),"%s/close.tokens",root);assert(!mellum_generated_read_tokens(p,&ids));assert(ids.len==2 && ids.v[0]==28 && ids.v[1]==45);ds4_tokens_free(&ids);
    snprintf(p,sizeof(p),"%s/append.rendered",root);char *text=mellum_generated_read_text(p);assert(text && strstr(text,"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"));free(text);
    ds4_engine_close(ready);unlink(fixture);
    printf("inspect-only native template preparation: PASS (sparse fixture only, %s)\n",root);return 0;
}
