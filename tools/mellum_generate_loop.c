#include "mellum_generate_loop.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static int snapshot(mellum_generated_backend *b,FILE *gold,float *v,int n) {
    if (b->logits(b->context,v,n)!=n) return -1;
    for(int i=0;i<n;i++) if(!isfinite(v[i])) return -1;
    return fwrite(v,sizeof(float),(size_t)n,gold)==(size_t)n ? 0:-1;
}
int mellum_generated_sync(mellum_generated_backend *b,const ds4_tokens *history,unsigned segment,
                          FILE *ops,FILE *gold,float *v,int n) {
    if (n<1 || n>98304 || history->len<1 || history->len>8192 || (segment!=0 && segment!=2) ||
        b->sync(b->context,history,segment) || snapshot(b,gold,v,n) ||
        fprintf(ops,"S %u %d\n",segment,history->len)<0) return -1;
    return 0;
}
int mellum_generated_turn(mellum_generated_backend *b,ds4_tokens *history,int limit,unsigned segment,
        FILE *ops,FILE *gold,FILE *emitted,float *v,int n,int *count,int *reason) {
    *count=*reason=0;
    if (limit<1 || limit>4096 || n<1 || n>98304 || (segment!=1 && segment!=3)) return -1;
    for(int step=0;step<limit;step++) {
        int token=0;
        for(int i=0;i<n;i++) { if(!isfinite(v[i])) return -1;if(v[i]>v[token]) token=i; }
        if(history->len>=8192 || b->eval(b->context,token,segment)) return -1;
        ds4_tokens_push(history,token);
        struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t)) return -1;
        if(snapshot(b,gold,v,n) || fprintf(ops,"E %u %d\n",segment,token)<0 ||
            fprintf(emitted,"%u %d %llu\n",segment,token,(unsigned long long)t.tv_sec*1000000000+(unsigned long long)t.tv_nsec)<0) return -1;
        (*count)++;
        int stop=b->stop_class(b->context,token);
        if(stop==DS4_STOP_PROTOCOL) { *reason=stop;return -1; }
        if(stop==DS4_STOP_EOS || stop==DS4_STOP_EOS_ALT) { *reason=stop;break; }
    }
    return 0;
}
int mellum_generated_parity(mellum_generated_backend *b,const ds4_tokens *history,FILE *ops,FILE *gold,float *v,int n) {
    if(n<1 || n>98304 || history->len<1 || history->len>8192) return -1;
    float *expected=malloc((size_t)n*sizeof(float));if(!expected) return -1;
    char kind;unsigned segment;int argument,cursor=0,records=0,fields,rc=-1;
    while((fields=fscanf(ops," %c %u %d",&kind,&segment,&argument))==3) {
        if(++records>8194) goto finish;
        if(kind=='S') {
            if((segment!=0 && segment!=2) || argument<=cursor || argument>history->len) goto finish;
            ds4_tokens prefix={.v=history->v,.len=argument,.cap=argument};
            if(b->sync(b->context,&prefix,segment)) goto finish;
            cursor=argument;
        } else if(kind=='E') {
            if((segment!=1 && segment!=3) || cursor>=history->len || history->v[cursor]!=argument ||
                b->eval(b->context,argument,segment)) goto finish;
            cursor++;
        } else goto finish;
        if(b->logits(b->context,v,n)!=n || fread(expected,sizeof(float),(size_t)n,gold)!=(size_t)n ||
            memcmp(expected,v,(size_t)n*sizeof(float))) goto finish;
        for(int i=0;i<n;i++) if(!isfinite(v[i])) goto finish;
    }
    if(fields!=EOF || ferror(ops) || cursor!=history->len || fgetc(gold)!=EOF || ferror(gold)) goto finish;
    rc=0;
finish:
    free(expected);return rc;
}
