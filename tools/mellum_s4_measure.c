/* S4 measurement driver: one Mellum run from a frozen prompt, resident or streamed from a bounded expert-slot
 * cache, reporting what the preregistered measurement reads.  The prompt is one user turn rendered with the
 * model's chat template (thinking on); decode is greedy.  Output: one JSON object.
 *
 *   tools/mellum_s4_measure --model M --prompt FILE --gen N --context C --out OUT.json
 *                           [--stream-experts N | --stream-bytes B]
 *
 * ttft_ns covers the prompt's sync up to its logits; each token's ns covers its eval; stops and loads are
 * the session's cumulative counts after that token (0 resident); memory is the process's peak phys_footprint
 * and resident size, and the largest Metal allocation sampled after the prompt and after each token. */
#include <errno.h>
#include <mach/mach.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_test_mellum_stream_stops(ds4_session *s, uint64_t *stops);   /* ds4.c hooks */
bool ds4_test_mellum_stream_loads(ds4_session *s, uint64_t *loads);

#define VOCAB 98304

static uint64_t now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

/* the process's peak physical footprint and peak resident size */
static bool memory_peaks(uint64_t *footprint, uint64_t *rss) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return false;
    *footprint = info.ledger_phys_footprint_peak;
    *rss = info.resident_size_peak;
    return true;
}

static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *buf = malloc((1u << 20) + 1u);
    const size_t n = buf ? fread(buf, 1, 1u << 20, f) : 0;
    const bool whole = buf && feof(f) && !ferror(f);
    fclose(f);
    if (!whole || n == 0 || memchr(buf, 0, n)) { free(buf); return NULL; }
    buf[n] = 0;
    return buf;
}

static int argmax(const float *l) {
    int best = 0;
    for (int i = 1; i < VOCAB; i++) if (l[i] > l[best]) best = i;
    return best;
}

static int usage(void) {
    fprintf(stderr, "usage: mellum_s4_measure --model M --prompt FILE --gen N --context C --out OUT.json "
                    "[--stream-experts N | --stream-bytes B]\n");
    return 2;
}

int main(int argc, char **argv) {
    const char *model = NULL, *prompt_path = NULL, *out_path = NULL;
    long gen = 0, context = 0;
    unsigned long long experts = 0, bytes = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--model")) model = argv[i + 1];
        else if (!strcmp(argv[i], "--prompt")) prompt_path = argv[i + 1];
        else if (!strcmp(argv[i], "--out")) out_path = argv[i + 1];
        else if (!strcmp(argv[i], "--gen")) gen = strtol(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "--context")) context = strtol(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "--stream-experts")) experts = strtoull(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "--stream-bytes")) bytes = strtoull(argv[i + 1], NULL, 10);
        else return usage();
    }
    if (!model || !prompt_path || !out_path || gen <= 0 || gen > 65536 || context <= 0 || (argc % 2) == 0 ||
        (experts && bytes)) {
        return usage();
    }
    const bool streaming = experts || bytes;
    char *text = read_text(prompt_path);
    if (!text) { fprintf(stderr, "mellum_s4_measure: cannot read %s\n", prompt_path); return 1; }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = model, .backend = DS4_BACKEND_METAL, .context_size = (int)context,
        .ssd_streaming = streaming, .ssd_streaming_cache_experts = (uint32_t)experts, .ssd_streaming_cache_bytes = bytes,
    };
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) { free(text); return 1; }
    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", text);
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_HIGH);
    free(text);
    if (prompt.len + gen > context) {
        fprintf(stderr, "mellum_s4_measure: %d prompt tokens + %ld generated exceed context %ld\n", prompt.len, gen, context);
        return 1;
    }
    ds4_session *s = NULL;
    if (ds4_session_create(&s, engine, (int)context) != 0 || !s) return 1;
    char err[256] = {0};
    float *logits = malloc(VOCAB * sizeof(float));
    uint64_t *ns = calloc((size_t)gen, sizeof(uint64_t)), *stops = calloc((size_t)gen, sizeof(uint64_t));
    uint64_t *loads = calloc((size_t)gen, sizeof(uint64_t));
    int *ids = calloc((size_t)gen, sizeof(int));
    if (!logits || !ns || !stops || !loads || !ids) return 1;
    const uint64_t t0 = now_ns();
    if (ds4_session_sync(s, &prompt, err, sizeof(err)) != 0 || ds4_session_copy_logits(s, logits, VOCAB) != VOCAB) {
        fprintf(stderr, "mellum_s4_measure: prompt sync failed: %s\n", err);
        return 1;
    }
    const uint64_t ttft = now_ns() - t0;
    uint64_t metal_peak = ds4_gpu_current_allocated_size();
    for (long i = 0; i < gen; i++) {
        ids[i] = argmax(logits);
        const uint64_t t = now_ns();
        if (ds4_session_eval(s, ids[i], err, sizeof(err)) != 0 || ds4_session_copy_logits(s, logits, VOCAB) != VOCAB) {
            fprintf(stderr, "mellum_s4_measure: token %ld failed: %s\n", i, err);
            return 1;
        }
        ns[i] = now_ns() - t;
        const uint64_t metal = ds4_gpu_current_allocated_size();
        if (metal > metal_peak) metal_peak = metal;
        if (streaming && (!ds4_test_mellum_stream_stops(s, &stops[i]) || !ds4_test_mellum_stream_loads(s, &loads[i]))) return 1;
    }
    uint64_t peak = 0, rss_peak = 0;
    if (!memory_peaks(&peak, &rss_peak)) return 1;
    /* the engine's effective slots (a byte budget is floored to whole experts) */
    const unsigned long long slots = streaming ? ds4_gpu_stream_expert_cache_configured_count() : 0u;
    FILE *out = fopen(out_path, "wx");
    if (!out) { fprintf(stderr, "mellum_s4_measure: cannot create %s: %s\n", out_path, strerror(errno)); return 1; }
    fprintf(out, "{\"streaming\":%s,\"slots\":%llu,\"stream_bytes\":%llu,\"context\":%ld,\"prompt_tokens\":%d,\"ttft_ns\":%llu,"
                 "\"stops\":%llu,\"loads\":%llu,\"phys_footprint_peak\":%llu,\"rss_peak\":%llu,"
                 "\"metal_allocated_peak\":%llu,\"tokens\":[",
            streaming ? "true" : "false", slots, bytes, context, prompt.len, (unsigned long long)ttft,
            (unsigned long long)stops[gen - 1], (unsigned long long)loads[gen - 1], (unsigned long long)peak,
            (unsigned long long)rss_peak, (unsigned long long)metal_peak);
    for (long i = 0; i < gen; i++) {
        fprintf(out, "%s{\"id\":%d,\"ns\":%llu,\"stops\":%llu,\"loads\":%llu}", i ? "," : "", ids[i],
                (unsigned long long)ns[i], (unsigned long long)stops[i], (unsigned long long)loads[i]);
    }
    fprintf(out, "]}\n");
    const bool closed = fclose(out) == 0;
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    ds4_engine_close(engine);
    return closed ? 0 : 1;
}
