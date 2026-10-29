#ifndef MELLUM_GENERATE_IO_H
#define MELLUM_GENERATE_IO_H
#include "../ds4.h"
char *mellum_generated_read_text(const char *);
int mellum_generated_read_tokens(const char *,ds4_tokens *);
int mellum_generated_write_tokens(const char *,const ds4_tokens *);
int mellum_generated_write_rendered(const char *,ds4_engine *,const ds4_tokens *);
#endif
