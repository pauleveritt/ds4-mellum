#ifndef MELLUM_READ_IO_FAULTS_H
#define MELLUM_READ_IO_FAULTS_H
int mellum_test_fcntl(int, int, int);
#define MELLUM_READ_FCNTL mellum_test_fcntl
#endif
