#include "mellum_read.h"
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef MELLUM_READ_CLOSE
#define MELLUM_READ_CLOSE close
#endif
#ifndef MELLUM_READ_FSTAT
#define MELLUM_READ_FSTAT fstat
#endif
#ifdef __APPLE__
#define MTIME(s) ((s).st_mtimespec)
#define CTIME(s) ((s).st_ctimespec)
#else
#define MTIME(s) ((s).st_mtim)
#define CTIME(s) ((s).st_ctim)
#endif
static int same(struct stat a,struct stat b) {
    return S_ISREG(a.st_mode) && S_ISREG(b.st_mode) && a.st_dev==b.st_dev && a.st_ino==b.st_ino &&
        a.st_size==b.st_size && MTIME(a).tv_sec==MTIME(b).tv_sec && MTIME(a).tv_nsec==MTIME(b).tv_nsec &&
        CTIME(a).tv_sec==CTIME(b).tv_sec && CTIME(a).tv_nsec==CTIME(b).tv_nsec;
}
static void report(const mellum_read_plan *p,unsigned workers,bool avoid,struct stat st,
                   const mellum_read_sample *samples,mellum_read_summary s,uint64_t prewarm_calls) {
    const char *control=avoid ? "true" : "false";
    mellum_read_file_identity id=p->identity;
    printf("{\"plan\":{\"dev\":%" PRIu64 ",\"ino\":%" PRIu64 ",\"size\":%" PRIu64
           ",\"mtime_ns\":%" PRIu64 ",\"ctime_ns\":%" PRIu64 "},",
           id.dev,id.ino,id.size,id.mtime_ns,id.ctime_ns);
    printf("\"schema_version\":1,\"mode\":\"%s\",\"physical_io_verified\":false,"
           "\"cache_controls\":{\"nocache\":%s,\"rdahead_disabled\":%s},\"workers\":%u,"
           "\"jobs\":%u,\"logical_bytes\":%" PRIu64 ",\"file\":{\"dev\":%" PRIu64
           ",\"ino\":%" PRIu64 ",\"size\":%" PRIu64 ",\"mtime_ns\":%" PRIu64 ",\"ctime_ns\":%" PRIu64 "},",
           avoid ? "avoid" : "buffered",control,control,workers,p->count,p->total_bytes,
           (uint64_t)st.st_dev,(uint64_t)st.st_ino,(uint64_t)st.st_size,
           (uint64_t)MTIME(st).tv_sec*1000000000+(uint64_t)MTIME(st).tv_nsec,
           (uint64_t)CTIME(st).tv_sec*1000000000+(uint64_t)CTIME(st).tv_nsec);
    printf("\"summary\":{\"start_ns\":%" PRIu64 ",\"end_ns\":%" PRIu64 ",\"buffer_bytes\":%" PRIu64
           ",\"prewarm_bytes\":%" PRIu64 ",\"prewarm_syscalls\":%" PRIu64 "},\"samples\":[",
           s.start_ns,s.end_ns,s.buffer_bytes,s.prewarm_bytes,prewarm_calls);
    for (unsigned i=0;i<p->count;++i) {
        const mellum_read_sample *v=&samples[i];
        printf("%s{\"job_id\":%u,\"expert_id\":%u,\"enqueue_ns\":%" PRIu64 ",\"start_ns\":%" PRIu64
               ",\"done_ns\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"syscalls\":%" PRIu64 "}",
               i ? "," : "",i,p->jobs[i].expert,v->enqueue_ns,v->start_ns,v->done_ns,v->bytes,v->syscalls);
    }
    puts("]}");
}
static int pipe_fd(const char *text,int access) {
    if (!text || !*text) return -1;
    unsigned long value=0;
    for (const char *p=text;*p;++p) {
        if (*p<'0' || *p>'9' || value>((unsigned long)INT_MAX-(unsigned)(*p-'0'))/10) return -1;
        value=value*10+(unsigned)(*p-'0');
    }
    struct stat st;
    int fd=(int)value,flags=fcntl(fd,F_GETFL);
    if (fd<=2 || flags<0 || (flags&O_ACCMODE)!=access || fstat(fd,&st) || !S_ISFIFO(st.st_mode)) return -1;
    return fd;
}
int main(int argc,char **argv) {
    const char *input=NULL,*plan_path=NULL,*mode=NULL,*worker_text=NULL;
    const char *control_text=NULL,*event_text=NULL;
    mellum_read_gate gate={-1,-1};
    mellum_read_observers hooks={mellum_read_gate_ready,mellum_read_gate_done,&gate};
    mellum_read_plan plan={0}; mellum_read_sample *samples=NULL;
    unsigned char *buffer=NULL; int fd=-1,rc=1;
    char err[160]="invalid arguments";
    if (argc!=9 && argc!=13) goto finish;
    for (int i=1;i<argc;i+=2) {
        const char **slot=NULL;
        if (!strcmp(argv[i],"--input")) slot=&input;
        else if (!strcmp(argv[i],"--plan")) slot=&plan_path;
        else if (!strcmp(argv[i],"--mode")) slot=&mode;
        else if (!strcmp(argv[i],"--workers")) slot=&worker_text;
        else if (!strcmp(argv[i],"--control-fd")) slot=&control_text;
        else if (!strcmp(argv[i],"--event-fd")) slot=&event_text;
        if (!slot || *slot) goto finish;
        *slot=argv[i+1];
    }
    if (!input || !plan_path || !mode || !worker_text ||
        (strcmp(mode,"buffered") && strcmp(mode,"avoid")) ||
        (strcmp(worker_text,"1") && strcmp(worker_text,"2") && strcmp(worker_text,"4"))) goto finish;
    if (control_text || event_text) {
        gate.control_fd=pipe_fd(control_text,O_RDONLY); gate.event_fd=pipe_fd(event_text,O_WRONLY);
        if (gate.control_fd<0 || gate.event_fd<0 || gate.control_fd==gate.event_fd || signal(SIGPIPE,SIG_IGN)==SIG_ERR) goto finish;
    }
    unsigned workers=(unsigned)(worker_text[0]-'0'); bool avoid=!strcmp(mode,"avoid");
    if (mellum_read_plan_load(plan_path,&plan,err,sizeof(err))) goto finish;
    struct stat before,opened,after,path_after;
    snprintf(err,sizeof(err),"input size/identity/open/close check failed");
    if (sizeof(off_t)<8 || lstat(input,&before) || !S_ISREG(before.st_mode)) goto finish;
    fd=open(input,O_RDONLY|O_NOFOLLOW);
    if (fd<0 || MELLUM_READ_FSTAT(fd,&opened) || !same(before,opened) ||
        opened.st_size<0 || (uint64_t)opened.st_size!=plan.file_bytes) goto finish;
    if (mellum_read_cache_mode(fd,avoid,err,sizeof(err))) goto finish;
    mellum_read_counts prewarm={0};
    if (!avoid) {
        buffer=malloc((size_t)plan.max_job_bytes);
        if (!buffer) goto finish;
        for (unsigned i=0;i<plan.count;++i)
            if (mellum_read_job_run(fd,&plan.jobs[i],buffer,&prewarm,pread,err,sizeof(err))) goto finish;
        free(buffer); buffer=NULL;
    }
    samples=calloc(plan.count,sizeof(*samples));
    if (!samples) goto finish;
    mellum_read_summary summary={0};
    if (mellum_read_execute_observed(fd,&plan,workers,samples,&summary,control_text ? &hooks : NULL,err,sizeof(err))) goto finish;
    summary.prewarm_bytes=prewarm.bytes;
    snprintf(err,sizeof(err),"input identity/close changed after reads");
    if (MELLUM_READ_FSTAT(fd,&after) || lstat(input,&path_after) || !same(opened,after) || !same(opened,path_after)) goto finish;
    int closed=MELLUM_READ_CLOSE(fd); fd=-1;
    if (closed) goto finish;
    report(&plan,workers,avoid,opened,samples,summary,prewarm.syscalls);
    if (fflush(stdout) || ferror(stdout)) goto finish;
    rc=0;
finish:
    if (fd>=0 && MELLUM_READ_CLOSE(fd)) rc=1;
    if (rc) fprintf(stderr,"%s\n",err);
    free(buffer); free(samples); mellum_read_plan_free(&plan);
    return rc;
}
