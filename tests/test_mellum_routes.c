/* Model-free route capture tests: no engine or GPU linkage. */
#include "ds4_mellum_routes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

typedef struct { unsigned char bytes[8192]; size_t n, limit; } memory_sink;

static size_t append_bytes(void *ctx, const void *data, size_t bytes) {
    memory_sink *s = ctx;
    size_t n = bytes < s->limit - s->n ? bytes : s->limit - s->n;
    memcpy(s->bytes + s->n, data, n);
    s->n += n;
    return n;
}

static void padded_ids(uint32_t ids[452]) {
    for (unsigned i = 0; i < 452; i++) ids[i] = UINT32_MAX;
    for (unsigned layer = 0; layer < 28; layer++)
        for (unsigned slot = 0; slot < 8; slot++) ids[layer * 16 + slot] = 7 - slot;
}

static bool emit_fixture(mellum_routes_writer *w, const uint32_t *ids) {
    const uint32_t tokens[] = {1234, 1235};
    return mellum_routes_forward(w, 0, 42, 1, 2, 7, 11, tokens, 2, ids, 452);
}

static void check_frames(void) {
    memory_sink sink = {.limit = 8192};
    mellum_routes_writer w;
    uint32_t ids[452];
    padded_ids(ids);
    CHECK(mellum_routes_begin(&w, 2, append_bytes, &sink));
    const unsigned char header[] = "MLRT0001" "\x1c\0\0\0" "\x40\0\0\0" "\x08\0\0\0" "\x02\0\0\0";
    CHECK(sink.n == 24 && memcmp(sink.bytes, header, 24) == 0);
    CHECK(emit_fixture(&w, ids));
    const unsigned char prefix[] = "FWD1" "\0\0\0\0\0\0\0\0" "\x2a\0\0\0\0\0\0\0"
        "\x01\0\0\0" "\x02\0\0\0" "\x07\0\0\0" "\x0b\0\0\0";
    CHECK(sink.n == 960 && memcmp(sink.bytes + 24, prefix, 36) == 0);
    CHECK(memcmp(sink.bytes + 60, "\xd2\x04\0\0", 4) == 0);
    for (unsigned layer = 0; layer < 28; layer++) {
        for (unsigned slot = 0; slot < 8; slot++) {
            unsigned char want[4] = {(unsigned char)(7 - slot), 0, 0, 0};
            CHECK(memcmp(sink.bytes + 64 + layer * 32 + slot * 4, want, 4) == 0);
        }
    }
    const uint32_t token = 1235;
    CHECK(mellum_routes_forward(&w, 1, 43, 1, 1, 8, 12, &token, 1, ids, 452));
    CHECK(mellum_routes_forward(&w, 2, 44, 1, 2, 8, 12, &token, 1, ids, 452));
    CHECK(!mellum_routes_begin(NULL, 2, append_bytes, &sink));
    CHECK(!mellum_routes_begin(&w, 0, append_bytes, &sink));
    CHECK(!emit_fixture(&w, ids));
    CHECK(!mellum_routes_begin(&w, 2, NULL, &sink));
    CHECK(!emit_fixture(&w, ids));
    for (unsigned bad = 0; bad < 13; bad++) {
        sink.n = 0;
        CHECK(mellum_routes_begin(&w, 2, append_bytes, &sink));
        padded_ids(ids);
        uint64_t number = 0, position = 42;
        uint32_t rows = 1, phase = 2;
        size_t nt = 1, ni = 448;
        const uint32_t *tokens = &token, *selected = ids;
        switch (bad) {
        case 0: tokens = NULL; break;
        case 1: selected = NULL; break;
        case 2: nt = 0; break;
        case 3: ni = 447; break;
        case 4: phase = 3; break;
        case 5: rows = 0; break;
        case 6: rows = 3; break;
        case 7: rows = 2; break;
        case 8: number = 1; break;
        case 9: position = UINT64_MAX; break;
        case 10: ids[1] = ids[0]; break;
        case 11: ids[0] = 64; break;
        case 12: ids[27 * 16 + 7] = UINT32_MAX; break;
        }
        CHECK(!mellum_routes_forward(&w, number, position, rows, phase, 0, 0, tokens, nt, selected, ni));
        CHECK(sink.n == 24);
        CHECK(!emit_fixture(&w, ids) && sink.n == 24);
    }
    for (unsigned pos = 41; pos <= 44; pos++) {
        if (pos == 43) continue;
        sink.n = 0;
        padded_ids(ids);
        CHECK(mellum_routes_begin(&w, 2, append_bytes, &sink) && emit_fixture(&w, ids));
        CHECK(!mellum_routes_forward(&w, 1, pos, 1, 2, 0, 0, &token, 1, ids, 448));
        CHECK(sink.n == 960 && !emit_fixture(&w, ids));
    }
    CHECK(!emit_fixture(NULL, ids));
    puts("ok frames");
}

static void check_finish(void) {
    memory_sink sink = {.limit = 8192};
    mellum_routes_writer w;
    uint32_t ids[452];
    padded_ids(ids);
    CHECK(mellum_routes_begin(&w, 2, append_bytes, &sink) && emit_fixture(&w, ids));
    CHECK(mellum_routes_finish(&w));
    const char footer[] = "END1" "\x01\0\0\0\0\0\0\0" "\x01\0\0\0\0\0\0\0";
    CHECK(sink.n == 980 && memcmp(sink.bytes + 960, footer, 20) == 0);
    CHECK(!mellum_routes_finish(&w) && !emit_fixture(&w, ids) && sink.n == 980);
    sink.n = 0;
    CHECK(mellum_routes_begin(&w, 2, append_bytes, &sink));
    CHECK(!mellum_routes_finish(&w) && !emit_fixture(&w, ids) && sink.n == 24);
    sink.n = 0;
    CHECK(mellum_routes_begin(&w, 2, append_bytes, &sink) && emit_fixture(&w, ids));
    const uint32_t token = 1235;
    CHECK(mellum_routes_forward(&w, 1, 43, 1, 1, 8, 12, &token, 1, ids, 452));
    CHECK(mellum_routes_finish(&w));
    const char twice[] = "END1" "\x02\0\0\0\0\0\0\0" "\x02\0\0\0\0\0\0\0";
    CHECK(sink.n == 1916 && memcmp(sink.bytes + 1896, twice, 20) == 0);
    for (size_t limit = 0; limit < 980; limit++) {
        sink.n = 0;
        sink.limit = limit;
        bool began = mellum_routes_begin(&w, 2, append_bytes, &sink);
        bool framed = emit_fixture(&w, ids);
        CHECK(began == (limit >= 24) && framed == (limit >= 960));
        CHECK(w.forwards == (framed ? 1u : 0u));
        CHECK(!mellum_routes_finish(&w) && sink.n == limit);
        CHECK(!emit_fixture(&w, ids) && !mellum_routes_finish(&w) && sink.n == limit);
    }
    CHECK(!mellum_routes_finish(NULL));
    puts("ok finish and short writes");
}

static size_t file_sink(void *ctx, const void *data, size_t bytes) {
    return fwrite(data, 1, bytes, ctx);
}

static int write_fixture(const char *path) {
    FILE *file = fopen(path, "wb");
    if (!file) return 1;
    mellum_routes_writer w;
    uint32_t ids[452];
    padded_ids(ids);
    bool ok = mellum_routes_begin(&w, 2, file_sink, file) && emit_fixture(&w, ids) && mellum_routes_finish(&w);
    if (fflush(file) != 0) ok = false;
    if (fclose(file) != 0) ok = false;
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--emit-fixture") == 0) return write_fixture(argv[2]);
    if (argc != 1) return 2;
    check_layout();
    check_frames();
    check_finish();
    return 0;
}
