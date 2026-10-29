#ifndef MELLUM_CAPTURE_RECEIPT_H
#define MELLUM_CAPTURE_RECEIPT_H
#include <stdbool.h>
#include <stdio.h>
static bool mellum_capture_receipt_write(FILE *out,unsigned context,unsigned cap,
        bool tensor_tiles,bool pre_m5,bool m5) {
    if (!out || !context || !cap || cap>1024) return false;
    return fprintf(out,"{\"receipt_version\":1,\"context_tokens\":%u,\"cap_tokens\":%u,"
        "\"capabilities\":{\"tensor_tiles_available\":%s,\"pre_m5_apple\":%s,\"m5_apple\":%s}}\n",
        context,cap,tensor_tiles ? "true":"false",pre_m5 ? "true":"false",m5 ? "true":"false")>0 && !ferror(out);
}
#endif
