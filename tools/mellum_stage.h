#ifndef MELLUM_STAGE_H
#define MELLUM_STAGE_H
#include "mellum_read.h"
typedef struct {
    uint64_t source_bytes,source_syscalls,write_bytes,write_syscalls,slots_completed,buffer_bytes;
} mellum_stage_counts;
int mellum_stage_copy(int,int,const mellum_read_plan *,size_t,mellum_stage_counts *,char *,size_t);
#endif
