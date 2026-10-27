#include "mellum_stage.h"
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifndef MELLUM_STAGE_FSTAT
#define MELLUM_STAGE_FSTAT fstat
#endif
#ifndef MELLUM_STAGE_FCNTL
#define MELLUM_STAGE_FCNTL fcntl
#endif
#ifndef MELLUM_STAGE_FSYNC
#define MELLUM_STAGE_FSYNC fsync
#endif
#ifndef MELLUM_STAGE_CLOSE
#define MELLUM_STAGE_CLOSE close
#endif
static uint64_t now(void) {
    struct timespec t;if (clock_gettime(CLOCK_MONOTONIC,&t)) return 0;
    return (uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec;
}
static int identity(int fd,const char *path,mellum_read_file_identity *value) {
    struct stat f,p;mellum_read_file_identity linked;
    return MELLUM_STAGE_FSTAT(fd,&f) || lstat(path,&p) || mellum_read_identity_get(&f,value) ||
        mellum_read_identity_get(&p,&linked) || !mellum_read_identity_equal(*value,linked);
}
static void print_id(const char *name,mellum_read_file_identity id) {
    printf("\"%s\":{\"dev\":%" PRIu64 ",\"ino\":%" PRIu64 ",\"size\":%" PRIu64
           ",\"mtime_ns\":%" PRIu64 ",\"ctime_ns\":%" PRIu64 "},",
           name,id.dev,id.ino,id.size,id.mtime_ns,id.ctime_ns);
}
int main(int argc,char **argv) {
    const char *input=NULL,*wire=NULL,*output=NULL;
    int source=-1,scratch=-1,rc=1;
    bool controlled=false,synced=false,closed=true;
    mellum_read_plan plan={0};mellum_stage_counts c={0};
    mellum_read_file_identity src={0},dst={0},after={0};
    uint64_t start=now(),end=0;
    char err[160]="invalid staging arguments";
    if (argc!=7 || !start) goto finish;
    for (int i=1;i<argc;i+=2) {
        const char **slot=NULL;
        if (!strcmp(argv[i],"--input")) slot=&input;
        else if (!strcmp(argv[i],"--plan")) slot=&wire;
        else if (!strcmp(argv[i],"--output")) slot=&output;
        if (!slot || *slot) goto finish;
        *slot=argv[i+1];
    }
    if (!input || !wire || !output) goto finish;
    if (sysconf(_SC_PAGESIZE)!=16384) { snprintf(err,sizeof(err),"unsupported staging page size");goto finish; }
    if (mellum_read_plan_load(wire,&plan,err,sizeof(err))) goto finish;
    source=open(input,O_RDONLY|O_NOFOLLOW);
    if (source<0 || identity(source,input,&src) || src.size!=plan.file_bytes) {
        snprintf(err,sizeof(err),"staging source identity/open failed");goto finish;
    }
    scratch=open(output,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if (scratch<0) { snprintf(err,sizeof(err),"exclusive scratch creation failed");goto finish; }
#ifdef __APPLE__
    if (MELLUM_STAGE_FCNTL(scratch,F_NOCACHE,1)) {
        snprintf(err,sizeof(err),"scratch cache control failed");goto finish;
    }
    controlled=true;
#else
    snprintf(err,sizeof(err),"macOS scratch cache controls required");goto finish;
#endif
    if (mellum_stage_copy(source,scratch,&plan,16384,&c,err,sizeof(err))) goto finish;
    if (MELLUM_STAGE_FSYNC(scratch)) { snprintf(err,sizeof(err),"scratch fsync failed");goto finish; }
    synced=true;
    if (identity(source,input,&after) || !mellum_read_identity_equal(src,after) ||
        identity(scratch,output,&dst) || dst.size!=c.write_bytes) {
        snprintf(err,sizeof(err),"staging source/scratch changed");goto finish;
    }
    struct stat st;
    if (lstat(wire,&st) || mellum_read_identity_get(&st,&after) || !mellum_read_identity_equal(after,plan.identity)) {
        snprintf(err,sizeof(err),"staging plan changed");goto finish;
    }
    rc=0;
finish:
    if (scratch>=0) {
        if (rc && output) (void)identity(scratch,output,&dst);
        if (MELLUM_STAGE_CLOSE(scratch)) closed=false;
    }
    if (source>=0 && MELLUM_STAGE_CLOSE(source)) closed=false;
    if (!closed) { rc=1;snprintf(err,sizeof(err),"staging checked close failed"); }
    end=now();if (!end) { rc=1;snprintf(err,sizeof(err),"staging clock failed"); }
    printf("{\"complete\":%s,",rc ? "false" : "true");
    print_id("source",src);print_id("scratch",dst);print_id("plan",plan.identity);
    printf("\"counts\":{\"source_bytes\":%" PRIu64 ",\"source_syscalls\":%" PRIu64
           ",\"write_bytes\":%" PRIu64 ",\"write_syscalls\":%" PRIu64
           ",\"slots_completed\":%" PRIu64 ",\"buffer_bytes\":%" PRIu64 "},",
           c.source_bytes,c.source_syscalls,c.write_bytes,c.write_syscalls,c.slots_completed,c.buffer_bytes);
    printf("\"page_bytes\":16384,\"stride_bytes\":%" PRIu64 ",\"start_ns\":%" PRIu64
           ",\"end_ns\":%" PRIu64 ",\"cache_controls\":{\"nocache\":%s},"
           "\"fsync_ok\":%s,\"close_ok\":%s,\"error_type\":\"%s\",\"error\":\"%s\"}\n",
           c.buffer_bytes,start,end,controlled ? "true" : "false",synced ? "true" : "false",
           closed ? "true" : "false",rc ? "NativeStageError" : "",rc ? err : "");
    if (fflush(stdout) || ferror(stdout)) rc=1;
    mellum_read_plan_free(&plan);return rc;
}
