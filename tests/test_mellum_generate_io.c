#include "../tools/mellum_generate_io.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
void ds4_tokens_push(ds4_tokens *v,int t) {
    v->v=realloc(v->v,(size_t)(v->len+1)*sizeof(int));assert(v->v);v->v[v->len++]=t;
}
char *ds4_token_text(ds4_engine *e,int token,size_t *n) { (void)e;(void)token;*n=1;return strdup("x"); }
int main(void) {
    char directory[]="/tmp/mellum-gen-io-XXXXXX";assert(mkdtemp(directory));
    char path[256];snprintf(path,sizeof(path),"%s/tokens",directory);
    ds4_tokens tokens={0};int ids[]={0,98303};tokens.v=ids;tokens.len=2;
    assert(!mellum_generated_write_tokens(path,&tokens));assert(mellum_generated_write_tokens(path,&tokens));
    ds4_tokens actual={0};assert(!mellum_generated_read_tokens(path,&actual));assert(actual.len==2 && actual.v[1]==98303);free(actual.v);
    const char *bad[]={"","-1","98304","1x","1 9999999999999999999999"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++) {
        FILE *f=fopen(path,"wb");assert(f);fputs(bad[i],f);assert(!fclose(f));
        actual=(ds4_tokens){0};assert(mellum_generated_read_tokens(path,&actual));free(actual.v);
    }
    FILE *f=fopen(path,"wb");assert(f);fwrite("a\0b",1,3,f);assert(!fclose(f));
    assert(!mellum_generated_read_text(path));
    unlink(path);rmdir(directory);puts("generated bounded I/O: PASS");return 0;
}
