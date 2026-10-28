#ifndef MELLUM_GENERATE_LOOP_H
#define MELLUM_GENERATE_LOOP_H
#include "../ds4.h"
#include <stdio.h>
typedef struct {
    void *context;
    int (*sync)(void *,const ds4_tokens *,unsigned);
    int (*eval)(void *,int,unsigned);
    int (*logits)(void *,float *,int);
    int (*stop_class)(void *,int);
} mellum_generated_backend;
int mellum_generated_sync(mellum_generated_backend *,const ds4_tokens *,unsigned,FILE *,FILE *,float *,int);
int mellum_generated_turn(mellum_generated_backend *,ds4_tokens *,int,unsigned,FILE *,FILE *,FILE *,float *,int,int *,int *);
int mellum_generated_parity(mellum_generated_backend *,const ds4_tokens *,FILE *,FILE *,float *,int);
#endif
