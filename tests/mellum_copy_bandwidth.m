// D1's measured memory bandwidth (phase 8b): what a simple Metal kernel moves, to set bytes per decode step against.
//   clang -O2 -fobjc-arc -framework Metal -framework Foundation tests/mellum_copy_bandwidth.m -o tests/mellum_copy_bandwidth
//   tests/mellum_copy_bandwidth [MiB [N]]        default 1024 MiB, N = 20
// Two kernels on buffers of MiB each (private storage), each dispatch in its own command buffer, timed by its
// GPUStartTime and GPUEndTime; three warmup dispatches each, then N timed, in turn (copy, read, copy, read, ...):
//   copy: dst[i] = src[i] over uint4, reported as (read + written) bytes / time;
//   read: a grid-stride XOR of src over uint4 into one word, reported as read bytes / time.
// Prints the median, minimum and maximum of each.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void bw_copy(device const uint4 *src [[buffer(0)]], device uint4 *dst [[buffer(1)]],\n"
     "                    uint i [[thread_position_in_grid]]) { dst[i] = src[i]; }\n"
     "kernel void bw_read(device const uint4 *src [[buffer(0)]], device atomic_uint *out [[buffer(1)]],\n"
     "                    constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]],\n"
     "                    uint grid [[threads_per_grid]]) {\n"
     "    uint4 acc = 0;\n"
     "    for (uint j = i; j < n; j += grid) acc ^= src[j];\n"
     "    const uint x = acc.x ^ acc.y ^ acc.z ^ acc.w;\n"
     "    if (x == 0x9e3779b9u) atomic_fetch_add_explicit(out, 1u, memory_order_relaxed);\n"
     "}\n";

static int cmp(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const unsigned long mib = argc > 1 ? strtoul(argv[1], NULL, 10) : 1024;
        const int n = argc > 2 ? atoi(argv[2]) : 20;
        if (mib == 0 || mib > 8192 || n < 1) {
            fprintf(stderr, "usage: %s [MiB (1-8192) [N]]\n", argv[0]);
            return 2;
        }
        const NSUInteger bytes = (NSUInteger)mib << 20, words = bytes / 16;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> copy = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bw_copy"] error:&err];
        id<MTLComputePipelineState> rd = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bw_read"] error:&err];
        id<MTLBuffer> src = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> dst = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        id<MTLBuffer> out = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!copy || !rd || !src || !dst || !out || !q) {
            fprintf(stderr, "setup failed\n");
            return 1;
        }
        {   // fill src so the reads are of written pages
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
            [b fillBuffer:src range:NSMakeRange(0, bytes) value:0x5a];
            [b endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        const uint32_t nwords = (uint32_t)words;
        const NSUInteger read_threads = (NSUInteger)dev.maxThreadsPerThreadgroup.width * 1024;
        double *tc = calloc((size_t)n, sizeof(double)), *tr = calloc((size_t)n, sizeof(double));
        for (int it = -3; it < n; it++) {
            for (int k = 0; k < 2; k++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                if (k == 0) {
                    [e setComputePipelineState:copy];
                    [e setBuffer:src offset:0 atIndex:0];
                    [e setBuffer:dst offset:0 atIndex:1];
                    [e dispatchThreads:MTLSizeMake(words, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    [e setComputePipelineState:rd];
                    [e setBuffer:src offset:0 atIndex:0];
                    [e setBuffer:out offset:0 atIndex:1];
                    [e setBytes:&nwords length:sizeof(nwords) atIndex:2];
                    [e dispatchThreads:MTLSizeMake(read_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    fprintf(stderr, "command buffer failed\n");
                    return 1;
                }
                if (it >= 0) (k == 0 ? tc : tr)[it] = cb.GPUEndTime - cb.GPUStartTime;
            }
        }
        qsort(tc, (size_t)n, sizeof(double), cmp);
        qsort(tr, (size_t)n, sizeof(double), cmp);
        printf("device: %s; buffers %lu MiB; n %d per kernel\n", dev.name.UTF8String, mib, n);
        printf("copy: median %.1f GB/s (read + written), range %.1f-%.1f\n", 2.0 * bytes / tc[n / 2] / 1e9,
               2.0 * bytes / tc[n - 1] / 1e9, 2.0 * bytes / tc[0] / 1e9);
        printf("read: median %.1f GB/s, range %.1f-%.1f\n", bytes / tr[n / 2] / 1e9, bytes / tr[n - 1] / 1e9,
               bytes / tr[0] / 1e9);
        free(tc);
        free(tr);
    }
    return 0;
}
