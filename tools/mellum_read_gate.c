#include "mellum_read.h"
#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#ifndef MELLUM_GATE_TIMEOUT_MS
#define MELLUM_GATE_TIMEOUT_MS 30000
#endif
#ifndef MELLUM_GATE_POLL
#define MELLUM_GATE_POLL poll
#endif
static int64_t milliseconds(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)) return -1;
    return (int64_t)t.tv_sec*1000+t.tv_nsec/1000000;
}
static int exchange(void *context,char event,char command,char *err,size_t n) {
    mellum_read_gate *gate=context;
    int64_t start=milliseconds();
    if (start<0) goto fail;
    for (int phase=0;phase<2;++phase) {
        for (;;) {
            int64_t now=milliseconds(),remaining=start+MELLUM_GATE_TIMEOUT_MS-now;
            if (now<0 || remaining<=0) goto fail;
            struct pollfd p={phase ? gate->control_fd : gate->event_fd,phase ? POLLIN : POLLOUT,0};
            int rc=MELLUM_GATE_POLL(&p,1,(int)remaining);
            if (rc<0 && errno==EINTR) continue;
            if (rc<=0 || !(p.revents & p.events)) goto fail;
            char received=0;
            ssize_t count=phase ? read(p.fd,&received,1) : write(p.fd,&event,1);
            if (count<0 && errno==EINTR) continue;
            if (count!=1 || (phase && received!=command)) goto fail;
            break;
        }
    }
    return 0;
fail:
    if (n) snprintf(err,n,"observation pipe closed, invalid or timed out");
    return -1;
}
int mellum_read_gate_ready(void *p,char *e,size_t n) { return exchange(p,'R','G',e,n); }
int mellum_read_gate_done(void *p,char *e,size_t n) { return exchange(p,'D','L',e,n); }
