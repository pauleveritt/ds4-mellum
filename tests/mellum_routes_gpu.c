/* Dedicated diagnostic translation unit. Hooks do not exist in shipping builds.
 * Usage: mellum_routes_gpu MODEL TOKENS FIRST_PREFILL OUTPUT_DIRECTORY
 * Every remaining token is decoded except the last, which is a one-token prompt append.
 * Spools are experimental until actual kernel provenance is implemented. */
#include "../ds4_gpu.h"
#include "../ds4_mellum_routes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool route_layer(uint32_t cap, uint32_t layer, uint32_t rows,
                        const ds4_gpu_tensor *selected);
static bool route_done(uint32_t pos, uint32_t rows, const int *tokens);
#define DS4_MELLUM_ROUTES_TEST 1
#include "../ds4.c"
/* Reuse the existing token reader, without duplicating its token grammar. */
#define main mellum_trace_unused_main
#define now_sec mellum_trace_now_sec
#include "mellum_trace.c"
#undef now_sec
#undef main

static struct {
    int mode; /* 0 baseline, 1 buffered, 2 synchronous reference */
    uint32_t cap, phase, segment, layers;
    uint64_t bytes, copies, waits;
    ds4_gpu_tensor *gpu;
    uint32_t *host;
    mellum_routes_writer writer;
} capture;

static size_t file_sink(void *ctx, const void *data, size_t bytes) {
    return fwrite(data, 1, bytes, ctx);
}

static bool route_layer(uint32_t cap, uint32_t layer, uint32_t rows,
                        const ds4_gpu_tensor *selected) {
    if (!capture.mode) return true;
    mellum_routes_layout_info layout;
    if (cap != capture.cap || layer != capture.layers ||
        !mellum_routes_layout(cap, rows, layer, &layout)) return false;
    capture.layers++;
    if (capture.mode == 2) {
        /* Independent reference reads the original selected scratch directly. */
        capture.waits++;
        return ds4_gpu_end_commands() &&
            ds4_gpu_tensor_read(selected, 0, (char *)capture.host + layout.layer_offset,
                                layout.copy_bytes) && ds4_gpu_begin_commands();
    }
    capture.copies++;
    return ds4_gpu_tensor_copy(capture.gpu, layout.layer_offset, selected, 0, layout.copy_bytes);
}

static bool route_done(uint32_t pos, uint32_t rows, const int *tokens) {
    if (!capture.mode) return true;
    if (ds4_gpu_commands_active() || capture.layers != MELLUM_ROUTES_LAYERS) return false;
    if (capture.mode == 1 && !ds4_gpu_tensor_read(capture.gpu, 0, capture.host, capture.bytes)) return false;
    capture.layers = 0;
    return mellum_routes_forward(&capture.writer, capture.writer.forwards, pos, rows,
        capture.phase, capture.segment, 0, (const uint32_t *)tokens, rows,
        capture.host, capture.bytes / sizeof(uint32_t));
}

static bool same_files(const char *a, const char *b) {
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    bool ok = fa && fb;
    int ca = 0, cb = 0;
    while (ok) {
        ca = fgetc(fa); cb = fgetc(fb);
        if (ca != cb) { ok = false; break; }
        if (ca == EOF) break;
    }
    if (fa) { ok = ok && !ferror(fa); fclose(fa); }
    if (fb) { ok = ok && !ferror(fb); fclose(fb); }
    return ok;
}

/* Live gate for the public owned handle. Compiling it does not run a GPU. */
typedef struct { FILE *file; uint32_t count; } owned_profiles;
static bool owned_profile(void *ctx,uint32_t id,const char *text,size_t n) {
    owned_profiles *p=ctx;
    const char *roles[]={";EMBED:",";PROJECTION:",";ROUTER:",";MID:",";DOWN:",";REDUCE:",";ATTENTION:"};
    if(id!=p->count || n!=strlen(text) || strncmp(text,"profile-v1;head=",16))return false;
    for(unsigned i=0;i<7;i++)if(!strstr(text,roles[i]))return false;
    if(!strstr(text,";MID:kernel_qwen4_moe_") || !strstr(text,";DOWN:kernel_qwen4_moe_"))return false;
    if(strstr(text,"head=encoded") && !strstr(text,";HEAD:"))return false;
    p->count++;
    return fprintf(p->file,"%u %s\n",id,text)>0;
}
static bool run_owned(ds4_engine *engine,int context,const ds4_tokens *tokens,int first,
                       const char *directory,FILE *gold) {
    char path[4096],err[512]={0};
    snprintf(path,sizeof(path),"%s/owned.routes",directory);FILE *spool=fopen(path,"wbx");
    snprintf(path,sizeof(path),"%s/owned.profiles",directory);FILE *profiles=fopen(path,"wbx");
    ds4_mellum_capture *handle=NULL;
    ds4_mellum_capture_options options={.spool_sink=file_sink,.spool_context=spool};
    bool ok=spool && profiles && !ds4_mellum_capture_create(&handle,engine,context,&options,err,sizeof(err));
    float logits[VOCAB],expected[VOCAB];rewind(gold);
    for(int pos=0;ok && pos<tokens->len;) {
        if(pos==0 || pos==tokens->len-1) {
            ds4_tokens prompt={.v=tokens->v,.len=pos==0 ? first:tokens->len};
            ok=!ds4_mellum_capture_sync(handle,&prompt,pos==0 ? 0:2,err,sizeof(err));pos=prompt.len;
        } else {
            ok=!ds4_mellum_capture_eval(handle,tokens->v[pos],1,err,sizeof(err));pos++;
        }
        ok=ok && ds4_mellum_capture_copy_logits(handle,logits,VOCAB)==VOCAB &&
           fread(expected,sizeof(expected),1,gold)==1 && !memcmp(logits,expected,sizeof(logits));
    }
    owned_profiles out={.file=profiles};
    if(ok)ok=!ds4_mellum_capture_finish(handle,owned_profile,&out,err,sizeof(err)) && out.count>0;
    if(ok)ok=ds4_mellum_capture_copy_logits(handle,logits,VOCAB)==VOCAB &&
        ds4_mellum_capture_eval(handle,0,1,err,sizeof(err))!=0 &&
        ds4_mellum_capture_finish(handle,owned_profile,&out,err,sizeof(err))!=0;
    ds4_mellum_capture_free(handle);
    if(spool && fclose(spool))ok=false;if(profiles && fclose(profiles))ok=false;
    fprintf(stderr,"owned capture logits/profile gate: %s (%s)\n",ok ? "PASS":"FAIL",err);
    return ok;
}

typedef struct { size_t bytes,calls,limit; } owned_sink;
static size_t owned_bytes(void *ctx,const void *data,size_t n) {
    owned_sink *s=ctx;(void)data;s->calls++;
    if(n>s->limit-s->bytes)n=s->limit-s->bytes;s->bytes+=n;return n;
}
static bool owned_cancel(void *ctx) { (void)ctx;return true; }
static bool owned_failures(ds4_engine *engine,int context,int token) {
    owned_sink header={.limit=0};ds4_mellum_capture *failed=(void *)&header;char error[128];
    ds4_mellum_capture_options bad={.spool_sink=owned_bytes,.spool_context=&header};
    if(!ds4_mellum_capture_create(&failed,engine,context,&bad,error,sizeof(error)) ||
       failed || header.bytes || header.calls!=1 || ds4_gpu_commands_active())return false;
    for(unsigned fault=0;fault<4;fault++) {
        owned_sink sink={.limit=SIZE_MAX};ds4_mellum_capture *c=NULL;char err[512]={0};
        ds4_mellum_capture_options options={.spool_sink=owned_bytes,.spool_context=&sink,
            .cancel=fault==0 ? owned_cancel:NULL};
        if(ds4_mellum_capture_create(&c,engine,context,&options,err,sizeof(err)))return false;
        uint32_t *host=c->host;ds4_gpu_tensor *gpu=c->gpu;
        if(fault==1)sink.limit=sink.bytes+4;
        if(fault==2)c->gpu=NULL; /* Actual copy rejects NULL, then ordinary GPU cleanup must run. */
        if(fault==3)c->host=NULL; /* Actual read rejects NULL, after successful GPU completion. */
        ds4_tokens prompt={.v=&token,.len=1};
        int status=ds4_mellum_capture_sync(c,&prompt,0,err,sizeof(err));
        c->host=host;c->gpu=gpu;
        size_t calls=sink.calls;
        bool ok=status==(fault==0 ? DS4_SESSION_SYNC_INTERRUPTED:1) &&
            c->core.state==MELLUM_CAPTURE_FAILED && !c->scope.owner && !ds4_gpu_commands_active() &&
            ds4_mellum_capture_finish(c,NULL,NULL,err,sizeof(err))!=0 &&
            !ds4_mellum_capture_copy_logits(c,NULL,0) && sink.calls==calls;
        ds4_mellum_capture_free(c);if(!ok || sink.calls!=calls)return false;
    }
    return true;
}

static bool owned_nohead(ds4_engine *engine,int context,const ds4_tokens *tokens) {
    FILE *spool=tmpfile(),*profiles=tmpfile();ds4_mellum_capture *c=NULL;ds4_session *base=NULL;
    ds4_mellum_capture_options options={.spool_sink=file_sink,.spool_context=spool};char err[512]={0};
    bool ok=spool && profiles && !ds4_mellum_capture_create(&c,engine,context,&options,err,sizeof(err)) &&
        !ds4_session_create(&base,engine,context);
    float expected[VOCAB],actual[VOCAB];
    if(ok)ok=mellum_capture_begin_sync(&c->core,NULL,0,(const uint32_t *)tokens->v,2,0) &&
        mellum_graph_forward_tokens(&c->session->qwen4_graph,&engine->model,&engine->weights,tokens->v,1,NULL,false) &&
        !c->core.logits && !ds4_mellum_capture_copy_logits(c,actual,VOCAB) &&
        mellum_graph_forward_tokens(&base->qwen4_graph,&engine->model,&engine->weights,tokens->v,1,NULL,false) &&
        mellum_graph_forward_tokens(&c->session->qwen4_graph,&engine->model,&engine->weights,tokens->v+1,1,c->session->logits,false) &&
        mellum_graph_forward_tokens(&base->qwen4_graph,&engine->model,&engine->weights,tokens->v+1,1,expected,false) &&
        !mellum_capture_end_operation(&c->core,&c->scope,0) &&
        ds4_mellum_capture_copy_logits(c,actual,VOCAB)==VOCAB && !memcmp(actual,expected,sizeof(actual));
    owned_profiles out={.file=profiles};
    if(ok)ok=c->core.profiles.count==2 && !ds4_mellum_capture_finish(c,owned_profile,&out,err,sizeof(err)) && out.count==2;
    ds4_mellum_capture_free(c);ds4_session_free(base);
    if(spool && fclose(spool))ok=false;if(profiles && fclose(profiles))ok=false;
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 5 || getenv("DS4_MELLUM_TRACE_AT") || getenv("DS4_MELLUM_TRACE")) return 2;
    ds4_tokens tokens = {0};
    if (!read_tokens(argv[2], &tokens)) return 2;
    char *end = NULL;
    long first = strtol(argv[3], &end, 10);
    if (!end || *end || first < 1 || first > tokens.len - 2) return 2;
    ds4_engine_options opt = {0};
    opt.model_path = argv[1]; opt.backend = DS4_BACKEND_METAL;
    opt.context_size = mellum_trace_context(getenv("MELLUM_TRACE_CONTEXT"), tokens.len);
    ds4_engine *engine = NULL;
    if (opt.context_size < 0 || ds4_engine_open(&engine, &opt)) return 1;
    FILE *gold = tmpfile();
    float logits[VOCAB], expected[VOCAB];
    char paths[2][4096];
    snprintf(paths[0], sizeof(paths[0]), "%s/buffered.routes", argv[4]);
    snprintf(paths[1], sizeof(paths[1]), "%s/reference.routes", argv[4]);
    bool ok = gold != NULL;
    /* Cold baseline, buffered, reference, then interleaved warm samples. */
    const int modes[] = {0, 1, 2, 0, 1, 0, 1};
    for (size_t run = 0; ok && run < sizeof(modes)/sizeof(modes[0]); run++) {
        ds4_session *session = NULL;
        char err[512] = {0};
        FILE *spool = NULL;
        memset(&capture, 0, sizeof(capture)); capture.mode = modes[run];
        ok = ds4_session_create(&session, engine, opt.context_size) == 0;
        if (!ok) break;
        capture.cap = session->qwen4_graph.cap_tokens;
        if (capture.mode) {
            mellum_routes_layout_info layout;
            ok = mellum_routes_layout(capture.cap, 1, 0, &layout);
            if (ok) {
                capture.bytes = layout.buffer_bytes;
                capture.host = malloc(capture.bytes);
                if (capture.mode == 1) capture.gpu = ds4_gpu_tensor_alloc(capture.bytes);
                /* Never overwrite a previous invocation's evidence. Warm runs use private files. */
                spool = run <= 2 ? fopen(paths[capture.mode - 1], "wbx") : tmpfile();
                ok = capture.host && (capture.mode == 2 || capture.gpu) && spool &&
                    mellum_routes_begin(&capture.writer, capture.cap, file_sink, spool);
            }
        }
        rewind(gold);
        const double start = now_sec();
        double prefill = 0, decode = 0, append = 0;
        for (int pos = 0; ok && pos < tokens.len;) {
            const double step = now_sec();
            capture.phase = pos == 0 || pos == tokens.len - 1 ? 1 : 2;
            capture.segment = pos == tokens.len - 1 ? 2 : pos == 0 ? 0 : 1;
            if (pos == 0 || pos == tokens.len - 1) {
                ds4_tokens prompt = { .v = tokens.v, .len = pos == 0 ? (int)first : tokens.len };
                ok = ds4_session_sync(session, &prompt, err, sizeof(err)) == 0;
                pos = prompt.len;
            } else {
                ok = ds4_session_eval(session, tokens.v[pos], err, sizeof(err)) == 0;
                pos++;
            }
            double elapsed = now_sec() - step;
            if (capture.segment == 0) prefill += elapsed;
            else if (capture.segment == 1) decode += elapsed;
            else append += elapsed;
            ok = ok && ds4_session_copy_logits(session, logits, VOCAB) == VOCAB;
            for (int i = 0; ok && i < VOCAB; i++) ok = isfinite(logits[i]);
            if (ok && run == 0) ok = fwrite(logits, sizeof(logits), 1, gold) == 1;
            else if (ok) ok = fread(expected, sizeof(expected), 1, gold) == 1 &&
                              memcmp(logits, expected, sizeof(logits)) == 0;
        }
        if (ok && capture.mode) {
            const uint64_t expected_forwards = ((uint64_t)first + capture.cap - 1) / capture.cap + tokens.len - first;
            ok = capture.writer.positions == (uint64_t)tokens.len &&
                 capture.writer.forwards == expected_forwards && mellum_routes_finish(&capture.writer);
        }
        if (spool && fclose(spool) != 0) ok = false;
        printf("run=%zu mode=%d ok=%d cap=%u forwards=%llu positions=%llu bytes=%llu copies=%llu added_waits=%llu total=%.6f prefill=%.6f decode=%.6f append=%.6f error=%s\n",
            run, capture.mode, ok, capture.cap, (unsigned long long)capture.writer.forwards,
            (unsigned long long)capture.writer.positions, (unsigned long long)capture.bytes,
            (unsigned long long)capture.copies, (unsigned long long)capture.waits,
            now_sec()-start, prefill, decode, append, err);
        ds4_gpu_tensor_free(capture.gpu); free(capture.host);
        capture.mode = 0;
        ds4_session_free(session);
    }
    if (ok) ok = same_files(paths[0], paths[1]);
    if (ok) ok=run_owned(engine,opt.context_size,&tokens,(int)first,argv[4],gold);
    if (ok) ok=owned_failures(engine,opt.context_size,tokens.v[0]);
    if (ok) ok=owned_nohead(engine,opt.context_size,&tokens);
    if (gold) fclose(gold);
    ds4_engine_close(engine); ds4_tokens_free(&tokens);
    fprintf(stderr, "routing GPU proof: %s\n", ok ? "PASS (all logits and ordered spool bytes match)" : "FAIL");
    return ok ? 0 : 1;
}
