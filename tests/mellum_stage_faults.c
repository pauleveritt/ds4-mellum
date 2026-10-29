#include "mellum_stage_faults.h"
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static unsigned reads,writes,stats;
static int mode(const char *value) { const char *v=getenv("MELLUM_STAGE_FAULT");return v && !strcmp(v,value); }
ssize_t mellum_stage_test_read(int fd,void *b,size_t n,off_t o) {
    if (++reads==1 && mode("partial")) { errno=EINTR;return -1; }
    if (mode("eof")) return 0;
    return pread(fd,b,mode("partial") && n>1 ? 1 : n,o);
}
ssize_t mellum_stage_test_write(int fd,const void *b,size_t n,off_t o) {
    if (!writes) { if ((uintptr_t)b%16384 || n%16384 || o%16384) abort(); }
    if (++writes==1 && mode("partial")) { errno=EINTR;return -1; }
    if (mode("write-zero") && writes>1) return 0;
    if (mode("write-zero") && n>32) n=32;
    if (mode("partial") && n>4096) n=4096;
    return pwrite(fd,b,n,o);
}
int mellum_stage_test_control(int fd,int cmd,int value) {
    if (mode("flag")) { errno=EINVAL;return -1; }
    return fcntl(fd,cmd,value);
}
int mellum_stage_test_sync(int fd) { if (mode("fsync")) { errno=EIO;return -1; } return fsync(fd); }
int mellum_stage_test_close(int fd) { int rc=close(fd);if (mode("close")) { errno=EIO;return -1; }return rc; }
int mellum_stage_test_stat(int fd,struct stat *s) {
    int rc=fstat(fd,s);if (++stats>2 && mode("stat") && !rc) ++s->st_size;return rc;
}
