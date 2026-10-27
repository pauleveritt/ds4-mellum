/* Model-free route capture tests: no engine or GPU linkage. */
#include "ds4_mellum_routes.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

static void check_layout(void) {
    mellum_routes_layout_info info;
    CHECK(mellum_routes_layout(1024, 1024, 27, &info));
    CHECK(info.buffer_bytes == 917504 && info.layer_offset == 884736 && info.copy_bytes == 32768);
    CHECK(info.layer_offset + info.copy_bytes == info.buffer_bytes);
    CHECK(mellum_routes_layout(1024, 1, 27, &info));
    CHECK(info.buffer_bytes == 917504 && info.layer_offset == 884736 && info.copy_bytes == 32);
    CHECK(mellum_routes_layout(2, 1, 1, &info));
    CHECK(info.buffer_bytes == 1792 && info.layer_offset == 64 && info.copy_bytes == 32);
    CHECK(!mellum_routes_layout(0, 1, 0, &info));
    CHECK(!mellum_routes_layout(1025, 1, 0, &info));
    CHECK(!mellum_routes_layout(UINT32_MAX, 1, 0, &info));
    CHECK(!mellum_routes_layout(2, 0, 0, &info));
    CHECK(!mellum_routes_layout(2, 3, 0, &info));
    CHECK(!mellum_routes_layout(2, 1, 28, &info));
    CHECK(!mellum_routes_layout(2, 1, UINT32_MAX, &info));
    CHECK(!mellum_routes_layout(2, 1, 0, NULL));
    puts("ok layout");
}

int main(void) {
    check_layout();
    return 0;
}
