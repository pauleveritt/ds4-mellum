#ifndef MELLUM_READ_FAULTS_H
#define MELLUM_READ_FAULTS_H
#include <stdio.h>
#include <stdlib.h>
void *mellum_test_calloc(size_t, size_t);
int mellum_test_fclose(FILE *);
#define MELLUM_READ_CALLOC mellum_test_calloc
#define MELLUM_READ_FCLOSE mellum_test_fclose
#endif
