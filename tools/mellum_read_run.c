#include "mellum_read.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifndef MELLUM_READ_CREATE
#define MELLUM_READ_CREATE pthread_create
#endif
#ifndef MELLUM_READ_JOIN
#define MELLUM_READ_JOIN pthread_join
#endif
#ifndef MELLUM_READ_PREAD
#define MELLUM_READ_PREAD pread
#endif
#ifndef MELLUM_READ_COMPLETE
#define MELLUM_READ_COMPLETE(id,buffer,bytes) ((void)0)
#endif

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t gate;
    int fd, stop, released;
    unsigned ready, next;
    uint64_t start;
    const mellum_read_plan *plan;
    mellum_read_sample *samples;
} shared;
typedef struct { shared *state; unsigned char *buffer; } worker;

static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)!=0) return 0;
    return (uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec;
}
static void *work(void *arg) {
    worker *w=arg; shared *s=w->state;
    pthread_mutex_lock(&s->lock);
    ++s->ready;
    pthread_cond_broadcast(&s->gate);
    while (!s->released && !s->stop) {
        if (pthread_cond_wait(&s->gate,&s->lock)!=0) { s->stop=1; pthread_cond_broadcast(&s->gate); }
    }
    for (;;) {
        if (s->stop || s->next==s->plan->count) break;
        unsigned id=s->next++;
        uint64_t start=now_ns();
        pthread_mutex_unlock(&s->lock);
        mellum_read_counts counts={0}; char err[128];
        int failed=!start || mellum_read_job_run(s->fd,&s->plan->jobs[id],w->buffer,&counts,
                                                MELLUM_READ_PREAD,err,sizeof(err));
        if (!failed) MELLUM_READ_COMPLETE(id,w->buffer,(size_t)counts.bytes);
        uint64_t done=now_ns();
        s->samples[id]=(mellum_read_sample){s->start,start,done,counts.bytes,counts.syscalls};
        pthread_mutex_lock(&s->lock);
        if (failed || !done) s->stop=1;
    }
    pthread_mutex_unlock(&s->lock);
    return NULL;
}
int mellum_read_execute_observed(int fd,const mellum_read_plan *plan,unsigned count,mellum_read_sample *samples,
                        mellum_read_summary *summary,const mellum_read_observers *hooks,char *err,size_t n) {
    shared s={.fd=fd,.plan=plan,.samples=samples};
    worker workers[4]={{0}};
    pthread_t threads[4];
    unsigned created=0;
    int failed=0;
    memset(summary,0,sizeof(*summary));
    if ((count!=1 && count!=2 && count!=4) || !plan->count || plan->count>4096 ||
        !plan->max_job_bytes || plan->max_job_bytes>6580224) goto early;
    memset(samples,0,plan->count*sizeof(*samples));
    if (pthread_mutex_init(&s.lock,NULL)!=0) goto early;
    if (pthread_cond_init(&s.gate,NULL)!=0) { pthread_mutex_destroy(&s.lock); goto early; }
    for (unsigned i=0;i<count;++i) {
        workers[i].state=&s;
        workers[i].buffer=malloc((size_t)plan->max_job_bytes);
        if (!workers[i].buffer) { failed=1; break; }
        volatile unsigned char *touch=workers[i].buffer;
        for (uint64_t pos=0;pos<plan->max_job_bytes;pos+=4096) touch[pos]=0;
        touch[plan->max_job_bytes-1]=0;
    }
    if (!failed) {
        for (unsigned i=0;i<count;++i) {
            if (MELLUM_READ_CREATE(&threads[i],NULL,work,&workers[i])!=0) { failed=1; break; }
            ++created;
        }
    }
    pthread_mutex_lock(&s.lock);
    while (!failed && !s.stop && s.ready<count) {
        if (pthread_cond_wait(&s.gate,&s.lock)!=0) failed=1;
    }
    if (!failed && !s.stop && hooks && hooks->ready && hooks->ready(hooks->context,err,n)) failed=1;
    s.start=now_ns();
    if (failed || !s.start) s.stop=1;
    s.released=1;
    pthread_cond_broadcast(&s.gate);
    pthread_mutex_unlock(&s.lock);
    for (unsigned i=0;i<created;++i) if (MELLUM_READ_JOIN(threads[i],NULL)!=0) failed=1;
    summary->end_ns=now_ns(); summary->start_ns=s.start;
    summary->buffer_bytes=count*plan->max_job_bytes;
    failed=failed || s.stop || !summary->end_ns || s.next!=plan->count;
    if (!failed && hooks && hooks->done && hooks->done(hooks->context,err,n)) failed=1;
    for (unsigned i=0;i<count;++i) free(workers[i].buffer);
    if (pthread_cond_destroy(&s.gate)!=0) failed=1;
    if (pthread_mutex_destroy(&s.lock)!=0) failed=1;
    if (!failed) return 0;
early:
    if (n) snprintf(err,n,"worker execution failed; no complete report");
    return -1;
}

int mellum_read_execute(int fd,const mellum_read_plan *plan,unsigned count,mellum_read_sample *samples,
                       mellum_read_summary *summary,char *err,size_t n) {
    return mellum_read_execute_observed(fd,plan,count,samples,summary,NULL,err,n);
}
