#ifndef MELLUM_READ_H
#define MELLUM_READ_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>

typedef struct { uint64_t offset, bytes, dest; } mellum_read_range;
typedef struct {
    uint32_t expert, nranges;
    uint64_t bytes;
    mellum_read_range ranges[3];
} mellum_read_job;
typedef struct { uint64_t dev, ino, size, mtime_ns, ctime_ns; } mellum_read_file_identity;
int mellum_read_identity_get(const struct stat *, mellum_read_file_identity *);
int mellum_read_identity_equal(mellum_read_file_identity, mellum_read_file_identity);
typedef struct {
    uint32_t count;
    uint64_t file_bytes, total_bytes, max_job_bytes;
    mellum_read_job *jobs;
    mellum_read_file_identity identity;
} mellum_read_plan;
int mellum_read_plan_load(const char *, mellum_read_plan *, char *, size_t);
void mellum_read_plan_free(mellum_read_plan *);
typedef struct { uint64_t bytes, syscalls; } mellum_read_counts;
int mellum_read_job_run(int, const mellum_read_job *, unsigned char *, mellum_read_counts *,
                        ssize_t (*)(int, void *, size_t, off_t), char *, size_t);
int mellum_read_cache_mode(int, bool, char *, size_t);
typedef struct { uint64_t enqueue_ns, start_ns, done_ns, bytes, syscalls; } mellum_read_sample;
typedef struct { uint64_t start_ns, end_ns, prewarm_bytes, buffer_bytes; } mellum_read_summary;
int mellum_read_execute(int, const mellum_read_plan *, unsigned, mellum_read_sample *,
                         mellum_read_summary *, char *, size_t);
#endif
