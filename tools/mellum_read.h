#ifndef MELLUM_READ_H
#define MELLUM_READ_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

typedef struct { uint64_t offset, bytes, dest; } mellum_read_range;
typedef struct {
    uint32_t expert, nranges;
    uint64_t bytes;
    mellum_read_range ranges[3];
} mellum_read_job;
typedef struct {
    uint32_t count;
    uint64_t file_bytes, total_bytes, max_job_bytes;
    mellum_read_job *jobs;
} mellum_read_plan;
int mellum_read_plan_load(const char *, mellum_read_plan *, char *, size_t);
void mellum_read_plan_free(mellum_read_plan *);
#endif
