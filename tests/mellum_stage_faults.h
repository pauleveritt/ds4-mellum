#ifndef MELLUM_STAGE_FAULTS_H
#define MELLUM_STAGE_FAULTS_H
#include <stddef.h>
#include <sys/types.h>
#include <sys/stat.h>
ssize_t mellum_stage_test_read(int,void *,size_t,off_t);
ssize_t mellum_stage_test_write(int,const void *,size_t,off_t);
int mellum_stage_test_control(int,int,int);
int mellum_stage_test_sync(int);
int mellum_stage_test_close(int);
int mellum_stage_test_stat(int,struct stat *);
#define MELLUM_STAGE_PREAD mellum_stage_test_read
#define MELLUM_STAGE_PWRITE mellum_stage_test_write
#define MELLUM_STAGE_FCNTL mellum_stage_test_control
#define MELLUM_STAGE_FSYNC mellum_stage_test_sync
#define MELLUM_STAGE_CLOSE mellum_stage_test_close
#define MELLUM_STAGE_FSTAT mellum_stage_test_stat
#endif
