#include "../tools/mellum_read.h"
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static pthread_mutex_t guard=PTHREAD_MUTEX_INITIALIZER;
static int seen[12], created, joined, fail_create, fail_io;
static const void *buffers[4];
static pthread_t owners[4];
static unsigned owner_count;
int mellum_test_create(pthread_t *thread,const pthread_attr_t *attr,void *(*start)(void *),void *arg) {
    if (fail_create && created==1) return EAGAIN;
    int rc=pthread_create(thread,attr,start,arg);
    if (!rc) ++created;
    return rc;
}
int mellum_test_join(pthread_t thread,void **value) { int rc=pthread_join(thread,value); if (!rc) ++joined; return rc; }
ssize_t mellum_test_pread(int fd,void *buf,size_t size,off_t offset) {
    if (fail_io && offset==8) { errno=EIO; return -1; }
    if (offset<8) { struct timespec delay={0,10000000}; nanosleep(&delay,NULL); }
    return pread(fd,buf,size,offset);
}
void mellum_test_complete(unsigned id,const unsigned char *buf,size_t size) {
    assert(id<12 && size==6);
    const unsigned char offsets[]={0,1,3,4,6,7};
    for (unsigned i=0;i<6;++i) assert(buf[i]==id*8+offsets[i]);
    pthread_mutex_lock(&guard);
    ++seen[id];
    unsigned owner=owner_count;
    for (unsigned i=0;i<owner_count;++i) {
        if (pthread_equal(owners[i],pthread_self())) owner=i;
        else assert(buffers[i]!=buf);
    }
    if (owner==owner_count) {
        assert(owner_count<4); owners[owner_count]=pthread_self(); buffers[owner_count++]=buf;
    } else assert(buffers[owner]==buf);
    pthread_mutex_unlock(&guard);
}
static int hook_phase, hook_fail;
static int ready(void *context,char *err,size_t n) {
    (void)err; (void)n;
    assert(context==&hook_phase && created==4 && joined==0);
    for (unsigned i=0;i<12;++i) assert(seen[i]==0);
    hook_phase=1; return hook_fail==1 ? -1 : 0;
}
static int done(void *context,char *err,size_t n) {
    (void)err; (void)n;
    assert(context==&hook_phase && hook_phase==1 && created==joined);
    for (unsigned i=0;i<12;++i) assert(seen[i]==1);
    hook_phase=2; return hook_fail==2 ? -1 : 0;
}
int main(void) {
    char path[]="/tmp/mellum-run-XXXXXX", err[160];
    int fd=mkstemp(path); assert(fd>=0);
    unsigned char data[96]; for (unsigned i=0;i<96;++i) data[i]=(unsigned char)i;
    assert(write(fd,data,96)==96);
    mellum_read_job jobs[12];
    for (unsigned i=0;i<12;++i) jobs[i]=(mellum_read_job){.expert=i,.nranges=3,.bytes=6,
        .ranges={{i*8,2,0},{i*8+3,2,2},{i*8+6,2,4}}};
    mellum_read_plan plan={.count=12,.file_bytes=96,.total_bytes=72,.max_job_bytes=6,.jobs=jobs};
    for (unsigned workers=1;workers<=4;workers*=2) {
        memset(seen,0,sizeof(seen)); owner_count=0; created=joined=0;
        mellum_read_sample samples[12]; mellum_read_summary summary={0};
        assert(mellum_read_execute(fd,&plan,workers,samples,&summary,err,sizeof(err))==0);
        assert(created==(int)workers && joined==created && summary.buffer_bytes==workers*6);
        for (unsigned i=0;i<12;++i) {
            assert(seen[i]==1 && samples[i].bytes==6 && samples[i].syscalls==3);
            assert(samples[i].enqueue_ns==summary.start_ns);
            assert(samples[i].enqueue_ns<=samples[i].start_ns && samples[i].start_ns<=samples[i].done_ns);
            assert(samples[i].done_ns<=summary.end_ns);
        }
        if (workers>1) assert(samples[0].done_ns>samples[1].done_ns && owner_count>1);
    }
    for (int mode=0;mode<2;++mode) {
        memset(seen,0,sizeof(seen)); owner_count=0; created=joined=0;
        fail_create=mode==0; fail_io=mode==1;
        mellum_read_sample samples[12]; mellum_read_summary summary={0};
        assert(mellum_read_execute(fd,&plan,4,samples,&summary,err,sizeof(err))!=0);
        assert(created==joined && created>0 && err[0]);
    }
    fail_create=fail_io=0;
    for (hook_fail=0;hook_fail<3;++hook_fail) {
        memset(seen,0,sizeof(seen)); owner_count=0; created=joined=hook_phase=0;
        mellum_read_sample samples[12]; mellum_read_summary summary={0};
        mellum_read_observers hooks={ready,done,&hook_phase};
        int rc=mellum_read_execute_observed(fd,&plan,4,samples,&summary,&hooks,err,sizeof(err));
        assert((rc==0)==(hook_fail==0));
        assert(created==joined && hook_phase==(hook_fail==1 ? 1 : 2));
    }
    assert(close(fd)==0 && unlink(path)==0);
    puts("worker content, timing and join checks: PASS"); return 0;
}
