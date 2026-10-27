/* Explicit GPU test: C tensor access must not retain freed backing buffers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(1); \
} } while (0)

static void check_access(id<MTLDevice> device,unsigned kind) {
    uint64_t before=device.currentAllocatedSize;
    /* One enclosing pool deliberately stays alive until after the assertion.
     * The helper must release its temporaries at its own C API boundary. */
    @autoreleasepool {
        for (unsigned i=0;i<4;i++) {
            ds4_gpu_tensor *t=ds4_gpu_tensor_alloc(1024*1024);CHECK(t);
            float data[4]={1,2,3,4};
            if (kind==1) CHECK(ds4_gpu_tensor_read(t,0,data,sizeof(data)));
            if (kind==2) CHECK(ds4_gpu_tensor_write(t,0,data,sizeof(data)));
            if (kind==3) {
                float *contents=ds4_gpu_tensor_contents(t);CHECK(contents);
                /* The returned pointer must remain valid after the helper returns. */
                contents[0]=42;CHECK(contents[0]==42);
            }
            ds4_gpu_tensor_free(t);
        }
        uint64_t after=device.currentAllocatedSize;
        fprintf(stderr,"tensor lifetime kind=%u before=%llu after=%llu\n",kind,
                (unsigned long long)before,(unsigned long long)after);
        CHECK(after==before);
    }
}

int main(void) {
    @autoreleasepool {
        CHECK(ds4_gpu_init());
    }
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();CHECK(device);
    for (unsigned kind=0;kind<4;kind++) check_access(device,kind);
    ds4_gpu_cleanup();
    puts("Metal tensor access lifetime: PASS");return 0;
}
