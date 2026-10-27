/* Public unsupported CPU API: no model or GPU is created. */
#include "ds4_mellum_capture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); } } while (0)
static size_t bytes;
static size_t spool(void *ctx,const void *p,size_t n) { (void)ctx;(void)p;bytes+=n;return n; }
int main(void) {
    ds4_mellum_capture_options opts={.spool_sink=spool};
    ds4_mellum_capture *sentinel=(ds4_mellum_capture *)&opts,*out=sentinel;
    char err[128]={0};
    CHECK(ds4_mellum_capture_create(&out,NULL,8192,&opts,err,sizeof(err))!=0);
    CHECK(out==NULL && strstr(err,"unsupported") && !bytes);
    for(unsigned bad=0;bad<6;bad++) {
        char tiny[2]={'x','z'};out=sentinel;
        CHECK(ds4_mellum_capture_create(bad==0 ? NULL:&out,NULL,bad==1 ? -1:1,
              bad==2 ? NULL:&opts,bad==3 ? NULL:tiny,bad==4 ? 0:1)!=0);
        CHECK(bad==0 || !out);CHECK(tiny[1]=='z');CHECK(!bytes);
    }
    CHECK(ds4_mellum_capture_sync(NULL,NULL,0,NULL,0)!=0);
    CHECK(ds4_mellum_capture_eval(NULL,0,0,NULL,0)!=0);
    CHECK(ds4_mellum_capture_copy_logits(NULL,NULL,0)==0);
    CHECK(ds4_mellum_capture_finish(NULL,NULL,NULL,NULL,0)!=0);
    ds4_mellum_capture_free(NULL);CHECK(!bytes);
    puts("Mellum unsupported capture API checks: PASS");return 0;
}
