#include "../tools/mellum_read.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int actual_fd, controls, control_fail, calls, fault;
int mellum_test_fcntl(int fd, int cmd, int value) {
    assert(fd==actual_fd);
#ifdef __APPLE__
    assert((controls==0 && cmd==F_NOCACHE && value==1) ||
           (controls==1 && cmd==F_RDAHEAD && value==0));
#else
    (void)cmd; (void)value;
#endif
    ++controls;
    if (control_fail) { errno=EINVAL; return -1; }
    return 0;
}
static ssize_t injected(int fd, void *buf, size_t size, off_t offset) {
    ++calls;
    if (fault==1 && calls==1) { errno=EINTR; return -1; }
    if (fault==2 && calls==2) { errno=EIO; return -1; }
    if (fault==3 && calls==2) return 0;
    if (fault==4) return (ssize_t)size+1;
    return pread(fd,buf,fault==1 ? 1 : size,offset);
}
int main(void) {
    char path[]="/tmp/mellum-io-XXXXXX", err[160];
    actual_fd=mkstemp(path); assert(actual_fd>=0);
    unsigned char data[64], buffer[6], expected[]={0,1,8,9,16,17};
    for (unsigned i=0;i<64;++i) data[i]=(unsigned char)i;
    assert(write(actual_fd,data,64)==64);
    mellum_read_job job={.expert=0,.nranges=3,.bytes=6,.ranges={{0,2,0},{8,2,2},{16,2,4}}};
    mellum_read_counts counts={0};
    assert(mellum_read_job_run(actual_fd,&job,buffer,&counts,pread,err,sizeof(err))==0);
    assert(!memcmp(buffer,expected,6) && counts.bytes==6 && counts.syscalls==3);
    fault=1; calls=0; counts=(mellum_read_counts){0};
    assert(mellum_read_job_run(actual_fd,&job,buffer,&counts,injected,err,sizeof(err))==0);
    assert(!memcmp(buffer,expected,6) && counts.bytes==6 && counts.syscalls==7);
    for (fault=2;fault<=4;++fault) {
        calls=0; counts=(mellum_read_counts){0};
        assert(mellum_read_job_run(actual_fd,&job,buffer,&counts,injected,err,sizeof(err))!=0);
        assert(err[0]);
        if (fault!=4) assert(counts.bytes==2 && counts.syscalls==2);
    }
    counts=(mellum_read_counts){0};
    assert(ftruncate(actual_fd,1)==0);
    assert(mellum_read_job_run(actual_fd,&job,buffer,&counts,pread,err,sizeof(err))!=0);
    assert(counts.bytes==1 && counts.syscalls==2);
    assert(mellum_read_cache_mode(actual_fd,false,err,sizeof(err))==0 && controls==0);
#ifdef __APPLE__
    assert(mellum_read_cache_mode(actual_fd,true,err,sizeof(err))==0 && controls==2);
    controls=0; control_fail=1;
    assert(mellum_read_cache_mode(actual_fd,true,err,sizeof(err))!=0 && controls==1 && err[0]);
#else
    assert(mellum_read_cache_mode(actual_fd,true,err,sizeof(err))!=0 && err[0]);
#endif
    assert(close(actual_fd)==0 && unlink(path)==0);
    puts("complete range I/O and cache control checks: PASS");
    return 0;
}
