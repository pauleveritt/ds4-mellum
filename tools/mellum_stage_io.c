#include "mellum_stage.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifndef MELLUM_STAGE_PREAD
#define MELLUM_STAGE_PREAD pread
#endif
#ifndef MELLUM_STAGE_PWRITE
#define MELLUM_STAGE_PWRITE pwrite
#endif
#ifndef MELLUM_STAGE_FSTAT
#define MELLUM_STAGE_FSTAT fstat
#endif
int mellum_stage_copy(int source,int scratch,const mellum_read_plan *plan,size_t page,
                      mellum_stage_counts *c,char *err,size_t n) {
    unsigned char *buffer=NULL;
    struct stat in,out;
    uint64_t total=0,stride=0;
    int rc=-1;
    memset(c,0,sizeof(*c));
    if (n) snprintf(err,n,"invalid staging plan or file");
    if (!plan || !plan->jobs || !plan->count || plan->count>4096 || page!=16384 ||
        !plan->max_job_bytes || plan->max_job_bytes>6580224 || plan->file_bytes>INT64_MAX) return -1;
    stride=(plan->max_job_bytes+page-1)/page*page;
    if (plan->count*stride>16ULL*1024*1024*1024) return -1;
    for (unsigned i=0;i<plan->count;++i) {
        const mellum_read_job *j=&plan->jobs[i];
        if (j->nranges!=1 || j->expert>=1792 || (i && j->expert<=plan->jobs[i-1].expert) ||
            !j->bytes || j->bytes>plan->max_job_bytes || j->ranges[0].dest || j->ranges[0].bytes!=j->bytes ||
            j->ranges[0].offset>plan->file_bytes || j->bytes>plan->file_bytes-j->ranges[0].offset) return -1;
        total+=j->bytes;
    }
    if (total!=plan->total_bytes || MELLUM_STAGE_FSTAT(source,&in) || MELLUM_STAGE_FSTAT(scratch,&out) ||
        !S_ISREG(in.st_mode) || !S_ISREG(out.st_mode) || in.st_size<0 || (uint64_t)in.st_size!=plan->file_bytes ||
        out.st_size || (in.st_dev==out.st_dev && in.st_ino==out.st_ino)) return -1;
    if (posix_memalign((void **)&buffer,page,(size_t)stride)) {
        if (n) snprintf(err,n,"aligned staging allocation failed");
        return -1;
    }
    c->buffer_bytes=stride;
    for (unsigned i=0;i<plan->count;++i) {
        mellum_read_counts read_counts={0};
        memset(buffer,0,(size_t)stride);
        int failed=mellum_read_job_run(source,&plan->jobs[i],buffer,&read_counts,MELLUM_STAGE_PREAD,err,n);
        c->source_bytes+=read_counts.bytes;c->source_syscalls+=read_counts.syscalls;
        if (failed) goto finish;
        uint64_t done=0;
        while (done<stride) {
            ++c->write_syscalls;
            ssize_t got=MELLUM_STAGE_PWRITE(scratch,buffer+done,(size_t)(stride-done),(off_t)(i*stride+done));
            if (got<0 && errno==EINTR) continue;
            if (got<=0 || (uint64_t)got>stride-done) {
                if (n) snprintf(err,n,"staging write incomplete or failed");
                goto finish;
            }
            done+=(uint64_t)got;c->write_bytes+=(uint64_t)got;
        }
        ++c->slots_completed;
    }
    rc=0;
finish:
    free(buffer);return rc;
}
