#ifndef MELLUM_READ_RUN_FAULTS_H
#define MELLUM_READ_RUN_FAULTS_H
#include <pthread.h>
#include <stddef.h>
#include <sys/types.h>
int mellum_test_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int mellum_test_join(pthread_t, void **);
ssize_t mellum_test_pread(int, void *, size_t, off_t);
void mellum_test_complete(unsigned, const unsigned char *, size_t);
#define MELLUM_READ_CREATE mellum_test_create
#define MELLUM_READ_JOIN mellum_test_join
#define MELLUM_READ_PREAD mellum_test_pread
#define MELLUM_READ_COMPLETE mellum_test_complete
#endif
