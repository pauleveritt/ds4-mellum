#include "../tools/mellum_read.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fail_alloc, fail_close;
void *mellum_test_calloc(size_t n, size_t size) { return fail_alloc ? NULL : calloc(n, size); }
int mellum_test_fclose(FILE *f) { int result = fclose(f); return fail_close ? EOF : result; }
static unsigned char golden[192];

static void put64(unsigned char *data, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) data[i] = (unsigned char)(value >> (8*i));
}
static void put32(unsigned char *data, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) data[i] = (unsigned char)(value >> (8*i));
}
static void invalid(unsigned char *data, size_t size) {
    char path[] = "/tmp/mellum-plan-XXXXXX", err[160];
    int fd = mkstemp(path);
    assert(fd >= 0 && write(fd, data, size) == (ssize_t)size && close(fd) == 0);
    mellum_read_plan plan = {0};
    assert(mellum_read_plan_load(path, &plan, err, sizeof(err)) != 0);
    assert(plan.jobs == NULL && plan.count == 0 && err[0]);
    mellum_read_plan_free(&plan);
    assert(unlink(path) == 0);
}
int main(void) {
    FILE *f = fopen("tests/mellum_read_plan_v1.bin", "rb");
    assert(f && fread(golden, 1, sizeof(golden), f) == sizeof(golden) && fclose(f) == 0);
    mellum_read_plan plan = {0}; char err[160];
    assert(mellum_read_plan_load("tests/mellum_read_plan_v1.bin", &plan, err, sizeof(err)) == 0);
    assert(plan.count == 2 && plan.total_bytes == 10 && plan.max_job_bytes == 6);
    assert(plan.jobs[0].nranges == 3 && plan.jobs[0].ranges[2].offset == 16);
    assert(plan.jobs[1].expert == 1791 && plan.jobs[1].bytes == 4);
    mellum_read_plan_free(&plan);
    mellum_read_plan_free(NULL);
    for (size_t size = 0; size < sizeof(golden); ++size) invalid(golden, size);
    const struct { unsigned offset; uint64_t value; unsigned width; } bad[] = {
        {0,0,8},{8,2,4},{12,63,4},{16,0,4},{16,4097,4},{20,3,4},{20,7,4},
        {24,UINT64_MAX,8},{32,11,8},{40,6580225,8},{48,1,8},{56,1,8},
        {64,1,4},{68,1792,4},{72,63,8},{80,0,8},{88,1,8},{96,1,4},
        {100,1,4},{120,0,8},{128,0,4},{152,3,8},{160,2,4},
        {168,UINT64_MAX,8},{176,UINT64_MAX,8}
    };
    for (unsigned i=0; i<sizeof(bad)/sizeof(bad[0]); ++i) {
        unsigned char data[192]; memcpy(data,golden,sizeof(data));
        if (bad[i].width == 4) put32(data+bad[i].offset,(uint32_t)bad[i].value);
        else put64(data+bad[i].offset,bad[i].value);
        invalid(data,sizeof(data));
    }
    unsigned char extra[193]; memcpy(extra,golden,192); extra[192]=0; invalid(extra,193);
    fail_alloc=1; invalid(golden,sizeof(golden)); fail_alloc=0;
    fail_close=1; invalid(golden,sizeof(golden)); fail_close=0;
    puts("native range plan checks: PASS");
    return 0;
}
