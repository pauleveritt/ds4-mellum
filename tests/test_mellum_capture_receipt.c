/* Serialization only: no backend symbols, GPU initialization or model. */
#include "mellum_capture_receipt.h"
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1); } } while(0)
int main(void) {
    FILE *file=tmpfile();CHECK(file);
    CHECK(mellum_capture_receipt_write(file,8192,1024,true,false,true));
    CHECK(!fflush(file));rewind(file);
    char text[512]={0};size_t n=fread(text,1,sizeof(text)-1,file);
    CHECK(n && !strcmp(text,"{\"receipt_version\":1,\"context_tokens\":8192,\"cap_tokens\":1024,\"capabilities\":{\"tensor_tiles_available\":true,\"pre_m5_apple\":false,\"m5_apple\":true}}\n"));
    CHECK(!fclose(file));
    file=fopen("/dev/null","r");CHECK(file);
    CHECK(!mellum_capture_receipt_write(file,8192,1024,false,false,false));fclose(file);
    puts("Mellum capture receipt checks: PASS");return 0;
}
