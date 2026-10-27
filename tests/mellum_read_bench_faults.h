#ifndef MELLUM_READ_BENCH_FAULTS_H
#define MELLUM_READ_BENCH_FAULTS_H
#include <sys/stat.h>
int mellum_test_close(int);
int mellum_test_stat(int, struct stat *);
#define MELLUM_READ_CLOSE mellum_test_close
#define MELLUM_READ_FSTAT mellum_test_stat
#endif
