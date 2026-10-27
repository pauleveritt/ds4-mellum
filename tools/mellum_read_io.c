#include "mellum_read.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>
#ifndef MELLUM_READ_FCNTL
#define MELLUM_READ_FCNTL fcntl
#endif

int mellum_read_job_run(int fd, const mellum_read_job *job, unsigned char *buffer,
                       mellum_read_counts *counts, ssize_t (*reader)(int,void *,size_t,off_t),
                       char *err, size_t n) {
    if (!job->nranges || job->nranges>3 || !job->bytes || job->bytes>6580224) goto invalid;
    uint64_t dest=0;
    for (unsigned i=0; i<job->nranges; ++i) {
        mellum_read_range range=job->ranges[i];
        if (range.dest!=dest || !range.bytes || range.bytes>job->bytes-dest ||
            range.offset>INT64_MAX || range.bytes>(uint64_t)INT64_MAX-range.offset) goto invalid;
        dest+=range.bytes;
    }
    if (dest!=job->bytes) goto invalid;
    for (unsigned i=0; i<job->nranges; ++i) {
        mellum_read_range range=job->ranges[i];
        uint64_t done=0;
        while (done<range.bytes) {
            size_t wanted=(size_t)(range.bytes-done);
            ++counts->syscalls;
            ssize_t got=reader(fd,buffer+range.dest+done,wanted,(off_t)(range.offset+done));
            if (got<0 && errno==EINTR) continue;
            if (got<=0 || (uint64_t)got>wanted) {
                if (n) snprintf(err,n,"pread incomplete or failed (errno %d)",got<0 ? errno : 0);
                return -1;
            }
            counts->bytes+=(uint64_t)got;
            done+=(uint64_t)got;
        }
    }
    return 0;
invalid:
    if (n) snprintf(err,n,"invalid compound read job");
    return -1;
}

int mellum_read_cache_mode(int fd, bool avoid, char *err, size_t n) {
    if (!avoid) return 0;
#ifdef __APPLE__
    if (MELLUM_READ_FCNTL(fd,F_NOCACHE,1)==0 && MELLUM_READ_FCNTL(fd,F_RDAHEAD,0)==0) return 0;
    if (n) snprintf(err,n,"cache avoidance controls failed (errno %d)",errno);
#else
    (void)fd;
    if (n) snprintf(err,n,"cache avoidance unsupported on this platform");
#endif
    return -1;
}
