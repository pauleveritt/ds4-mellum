#include "../tools/mellum_stage.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(void) {
    char input[]="/tmp/mellum-stage-source-XXXXXX",output[]="/tmp/mellum-stage-output-XXXXXX",err[160];
    int source=mkstemp(input),out=mkstemp(output);assert(source>=0 && out>=0);
    unsigned char bytes[64],actual[32768];for (unsigned i=0;i<64;++i) bytes[i]=(unsigned char)i;
    assert(write(source,bytes,64)==64);
    mellum_read_job jobs[2]={{.expert=1,.nranges=1,.bytes=4,.ranges={{4,4,0}}},
                             {.expert=7,.nranges=1,.bytes=4,.ranges={{12,4,0}}}};
    mellum_read_plan plan={.count=2,.file_bytes=64,.total_bytes=8,.max_job_bytes=4,.jobs=jobs};
    mellum_stage_counts c;
    assert(mellum_stage_copy(source,out,&plan,16384,&c,err,sizeof(err))==0);
    assert(c.source_bytes==8 && c.write_bytes==32768 && c.slots_completed==2 && c.buffer_bytes==16384);
    assert(pread(out,actual,sizeof(actual),0)==sizeof(actual));
    assert(!memcmp(actual,bytes+4,4) && !memcmp(actual+16384,bytes+12,4));
    for (unsigned i=4;i<16384;++i) assert(!actual[i] && !actual[16384+i]);
    for (unsigned bad=0;bad<4;++bad) {
        mellum_read_job saved=jobs[1];
        if (bad==0) jobs[1].expert=1;
        if (bad==1) jobs[1].nranges=2;
        if (bad==2) jobs[1].ranges[0].offset=63;
        if (bad==3) plan.file_bytes=63;
        assert(mellum_stage_copy(source,out,&plan,16384,&c,err,sizeof(err))!=0);
        assert(!c.source_bytes && !c.write_bytes && err[0]);
        jobs[1]=saved;plan.file_bytes=64;
    }
    assert(close(source)==0 && close(out)==0 && unlink(input)==0 && unlink(output)==0);
    puts("aligned staging slots and validation: PASS");return 0;
}
