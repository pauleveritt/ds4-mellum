#ifndef MELLUM_GENERATE_DRIVER_H
#define MELLUM_GENERATE_DRIVER_H
#include "mellum_generate_io.h"
#include "mellum_generate_loop.h"
int mellum_generated_inspect_open(ds4_engine **,const ds4_engine_options *);
int mellum_generated_collect(ds4_engine *,const char *,const char *);
int mellum_generated_prepare(ds4_engine *,const char *,const char *);
#endif
