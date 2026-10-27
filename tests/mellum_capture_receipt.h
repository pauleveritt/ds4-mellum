#ifndef MELLUM_CAPTURE_RECEIPT_H
#define MELLUM_CAPTURE_RECEIPT_H
#include <stdbool.h>
#include <stdio.h>
static bool mellum_capture_receipt_write(FILE *out,unsigned context,unsigned cap,
        bool tensor_tiles,bool pre_m5,bool m5) {
    (void)out;(void)context;(void)cap;(void)tensor_tiles;(void)pre_m5;(void)m5;
    return false;
}
#endif
