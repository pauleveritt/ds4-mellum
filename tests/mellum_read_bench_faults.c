#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
static int stats;
int mellum_test_close(int fd) { int rc=close(fd); return getenv("MELLUM_TEST_CLOSE_FAIL") ? -1 : rc; }
int mellum_test_stat(int fd, struct stat *s) {
    int rc=fstat(fd,s);
    if (++stats>1 && getenv("MELLUM_TEST_STAT_FAIL")) ++s->st_size;
    return rc;
}
