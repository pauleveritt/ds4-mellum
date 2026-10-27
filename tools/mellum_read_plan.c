#include "mellum_read.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef MELLUM_READ_CALLOC
#define MELLUM_READ_CALLOC calloc
#endif
#ifndef MELLUM_READ_FCLOSE
#define MELLUM_READ_FCLOSE fclose
#endif

int mellum_read_identity_get(const struct stat *s,mellum_read_file_identity *out) {
    if (!S_ISREG(s->st_mode) || s->st_size<0) return -1;
#ifdef __APPLE__
    struct timespec m=s->st_mtimespec,c=s->st_ctimespec;
#else
    struct timespec m=s->st_mtim,c=s->st_ctim;
#endif
    *out=(mellum_read_file_identity){(uint64_t)s->st_dev,(uint64_t)s->st_ino,(uint64_t)s->st_size,
          (uint64_t)m.tv_sec*1000000000+(uint64_t)m.tv_nsec,
          (uint64_t)c.tv_sec*1000000000+(uint64_t)c.tv_nsec};
    return 0;
}
int mellum_read_identity_equal(mellum_read_file_identity a,mellum_read_file_identity b) {
    return a.dev==b.dev && a.ino==b.ino && a.size==b.size && a.mtime_ns==b.mtime_ns && a.ctime_ns==b.ctime_ns;
}
static int plan_identity(FILE *f,const char *path,mellum_read_file_identity *out) {
    struct stat fd,path_stat;
    mellum_read_file_identity linked;
    return fstat(fileno(f),&fd) || lstat(path,&path_stat) || mellum_read_identity_get(&fd,out) ||
        mellum_read_identity_get(&path_stat,&linked) || !mellum_read_identity_equal(*out,linked);
}

static uint64_t le64(const unsigned char *p) {
    uint64_t value=0;
    for (unsigned i=0; i<8; ++i) value |= (uint64_t)p[i] << (8*i);
    return value;
}
static uint32_t le32(const unsigned char *p) {
    uint32_t value=0;
    for (unsigned i=0; i<4; ++i) value |= (uint32_t)p[i] << (8*i);
    return value;
}
void mellum_read_plan_free(mellum_read_plan *plan) {
    if (plan) { free(plan->jobs); memset(plan,0,sizeof(*plan)); }
}
int mellum_read_plan_load(const char *path, mellum_read_plan *out, char *err, size_t n) {
    mellum_read_plan plan={0};
    FILE *f=NULL;
    const char *why="invalid plan";
    unsigned char h[64], record[32];
    uint32_t ranges=0, previous=0;
    uint64_t total=0, maximum=0;
    memset(out,0,sizeof(*out));
    if (n) err[0]=0;
    f=fopen(path,"rb");
    if (!f) { why="cannot open plan"; goto fail; }
    if (plan_identity(f,path,&plan.identity) || plan.identity.size>64+4096*3*32) goto fail;
    if (fread(h,1,64,f)!=64 || memcmp(h,"MLREAD01",8) || le32(h+8)!=1 || le32(h+12)!=64) goto fail;
    plan.count=le32(h+16); ranges=le32(h+20);
    plan.file_bytes=le64(h+24); plan.total_bytes=le64(h+32); plan.max_job_bytes=le64(h+40);
    if (!plan.count || plan.count>4096 || ranges<plan.count || ranges>3*plan.count ||
        !plan.file_bytes || plan.file_bytes>INT64_MAX || !plan.max_job_bytes ||
        plan.max_job_bytes>6580224 || le64(h+48) || le64(h+56)) goto fail;
    plan.jobs=MELLUM_READ_CALLOC(plan.count,sizeof(*plan.jobs));
    if (!plan.jobs) { why="plan allocation failed"; goto fail; }
    for (uint32_t i=0; i<ranges; ++i) {
        if (fread(record,1,32,f)!=32) goto fail;
        uint32_t id=le32(record), expert=le32(record+4);
        uint64_t offset=le64(record+8), bytes=le64(record+16), dest=le64(record+24);
        if (id>=plan.count || expert>=1792 || (i==0 && id!=0) ||
            (i && id!=previous && id!=previous+1)) goto fail;
        mellum_read_job *job=&plan.jobs[id];
        if (job->nranges>=3 || (job->nranges && job->expert!=expert) ||
            dest!=job->bytes || !bytes || bytes>plan.max_job_bytes-job->bytes ||
            offset>plan.file_bytes || bytes>plan.file_bytes-offset) goto fail;
        job->expert=expert;
        job->ranges[job->nranges++]=(mellum_read_range){offset,bytes,dest};
        job->bytes+=bytes;
        total+=bytes;
        if (job->bytes>maximum) maximum=job->bytes;
        previous=id;
    }
    if (previous!=plan.count-1 || total!=plan.total_bytes || maximum!=plan.max_job_bytes ||
        fgetc(f)!=EOF || ferror(f)) goto fail;
    mellum_read_file_identity final_identity;
    if (plan_identity(f,path,&final_identity) || !mellum_read_identity_equal(plan.identity,final_identity)) goto fail;
    if (MELLUM_READ_FCLOSE(f)!=0) { f=NULL; why="plan close failed"; goto fail; }
    *out=plan;
    return 0;
fail:
    if (f && MELLUM_READ_FCLOSE(f)!=0) why="plan close failed";
    mellum_read_plan_free(&plan);
    if (n) snprintf(err,n,"%s",why);
    return -1;
}
