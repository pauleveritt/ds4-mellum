#include "../tools/mellum_read.h"
#include <assert.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
static int interrupted;
int mellum_gate_test_poll(struct pollfd *p,nfds_t n,int ms) {
    if (!interrupted++) { errno=EINTR; return -1; }
    return poll(p,n,ms);
}
int main(void) {
    signal(SIGPIPE,SIG_IGN);
    for (int phase=0;phase<2;++phase) for (int mode=0;mode<5;++mode) {
        int control[2],event[2]; char err[128],value;
        assert(pipe(control)==0 && pipe(event)==0);
        mellum_read_gate gate={control[0],event[1]};
        if (mode==0) assert(write(control[1],phase ? "L" : "G",1)==1);
        if (mode==1) assert(write(control[1],"X",1)==1);
        if (mode==2) { close(control[1]); control[1]=-1; }
        if (mode==4) { close(event[0]); event[0]=-1; }
        interrupted=0;
        int rc=(phase ? mellum_read_gate_done : mellum_read_gate_ready)(&gate,err,sizeof(err));
        assert((rc==0)==(mode==0));
        if (event[0]>=0) { assert(read(event[0],&value,1)==1 && value==(phase ? 'D' : 'R')); close(event[0]); }
        close(event[1]); close(control[0]); if (control[1]>=0) close(control[1]);
    }
    puts("bounded gate tokens, EINTR, EOF and deadlines: PASS");
}
