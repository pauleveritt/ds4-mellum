#ifndef MELLUM_GENERATE_DRIVER_FAULT_H
#define MELLUM_GENERATE_DRIVER_FAULT_H
#include <stdio.h>
FILE *mellum_test_open(const char *,const char *);
int mellum_test_close(FILE *);
#define fopen mellum_test_open
#define fclose mellum_test_close
#endif
