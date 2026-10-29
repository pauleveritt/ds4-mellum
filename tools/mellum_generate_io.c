#include "mellum_generate_io.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char *mellum_generated_read_text(const char *path) {
    FILE *f=fopen(path,"rb");if (!f) return NULL;
    char *text=malloc(65537);if (!text) { fclose(f);return NULL; }
    size_t n=fread(text,1,65537,f);int bad=ferror(f) || n>65536 || memchr(text,0,n)!=NULL;
    if (fclose(f)) bad=1;
    if (bad) { free(text);return NULL; }
    text[n]=0;return text;
}
int mellum_generated_read_tokens(const char *path,ds4_tokens *out) {
    char *text=mellum_generated_read_text(path);if (!text) return -1;
    const unsigned char *p=(const unsigned char *)text;int failed=0;
    while (*p) {
        while (isspace(*p)) p++;
        if (!*p) break;
        if (!isdigit(*p) || out->len>=8192) { failed=1;break; }
        unsigned token=0;
        while (isdigit(*p)) {
            token=token*10+(*p++-'0');if (token>=98304) { failed=1;break; }
        }
        if (failed || (*p && !isspace(*p))) { failed=1;break; }
        ds4_tokens_push(out,(int)token);
    }
    if (!out->len) failed=1;
    free(text);return failed ? -1:0;
}
int mellum_generated_write_tokens(const char *path,const ds4_tokens *tokens) {
    FILE *f=fopen(path,"wbx");if (!f) return -1;
    int bad=0;
    for (int i=0;i<tokens->len;i++) if (fprintf(f,"%d\n",tokens->v[i])<0) { bad=1;break; }
    if (ferror(f)) bad=1;
    if (fclose(f)) bad=1;
    return bad ? -1:0;
}
int mellum_generated_write_rendered(const char *path,ds4_engine *e,const ds4_tokens *tokens) {
    FILE *f=fopen(path,"wbx");if (!f) return -1;
    int bad=0;
    for (int i=0;i<tokens->len;i++) {
        size_t n=0;char *piece=ds4_token_text(e,tokens->v[i],&n);
        if (!piece || fwrite(piece,1,n,f)!=n) bad=1;
        free(piece);if (bad) break;
    }
    if (ferror(f)) bad=1;
    if (fclose(f)) bad=1;
    return bad ? -1:0;
}
