/* Mellum 2.1 recognition, validation and binding, model-free.  Each case writes
 * the sparse GGUF of tests/mellum_fixture.h and opens it through model_open /
 * config_validate_model / weights_bind, or ds4_engine_open.  The validators
 * exit(1) with a message, so a refusal is checked in a forked child whose
 * stderr is read back and must name what it refuses. */
#include "../ds4.c"
#include "mellum_fixture.h"
#include <sys/wait.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

/* Run fn in a child; return its exit status and what it wrote to stderr. */
typedef void (*child_fn)(const char *path);

static int run_child(child_fn fn, const char *path, char *err, size_t cap) {
    int pipefd[2];
    CHECK(pipe(pipefd) == 0);
    fflush(stdout);   /* a child's exit(1) would otherwise flush a copy of our pending output */
    fflush(stderr);
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], 2);
        close(pipefd[1]);
        fn(path);
        _exit(0);
    }
    close(pipefd[1]);
    size_t n = 0;
    ssize_t r;
    while (n + 1 < cap && (r = read(pipefd[0], err + n, cap - 1 - n)) > 0) n += (size_t)r;
    err[n] = '\0';
    close(pipefd[0]);
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

/* general.architecture = mellum names the variant, inside the QWEN4_EXP family. */
static void check_variant(void) {
    char path[512];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    CHECK(ds4_model_is_mellum());
    CHECK(ds4_model_is_qwen4());
    CHECK(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN4_EXP);
    CHECK(DS4_MODEL_VARIANT == DS4_VARIANT_MELLUM);
    model_close(&m);
    unlink(path);

    /* sibling: the Qwen3.8 shapes are not Mellum */
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    CHECK(!ds4_model_is_mellum() && ds4_model_is_qwen4());
    g_ds4_shape = DS4_SHAPE_QWEN4_MINI;
    CHECK(!ds4_model_is_mellum() && ds4_model_is_qwen4());
    printf("ok   variant: mellum is its own variant in the QWEN4_EXP family\n");
}

static void child_engine_open(const char *path) {
    ds4_engine *e = NULL;
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = path;
    opt.backend = DS4_BACKEND_CPU;
    opt.context_size = 4096;
    const int rc = ds4_engine_open(&e, &opt);
    _exit(rc == 0 ? 0 : 7);
}

/* Until the forward pass is ported, an open is refused, by name. */
static void check_open_refused(void) {
    char path[512], err[4096];
    mellum_fixture_write(MF_NONE, path, sizeof(path));
    const int rc = run_child(child_engine_open, path, err, sizeof(err));
    unlink(path);
    if (rc != 7 || !strstr(err, "Mellum 2.1") || !strstr(err, "not ported")) {
        fprintf(stderr, "FAIL open: exit %d, stderr [%s]\n", rc, err);
        exit(1);
    }
    printf("ok   open: refused by name until the forward pass is ported\n");
}

int main(void) {
    check_variant();
    check_open_refused();
    printf("test-mellum-config: ok\n");
    return 0;
}
