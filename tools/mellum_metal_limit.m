/* mellum_metal_limit -- the default Metal device's facts that bound a streamed Mellum run (S6 phase C): its name,
 * recommendedMaxWorkingSetSize, unified memory, the largest buffer, the Apple GPU families and, on macOS 26 or
 * later, MTLGPUFamilyMetal4 (the engine's Metal 4 check, ds4_metal.m:2889-2893).  Creates no queue and runs
 * nothing.  Build: make tools/mellum_metal_limit.  Usage: tools/mellum_metal_limit */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d) { fprintf(stderr, "mellum_metal_limit: no Metal device\n"); return 1; }
        printf("name: %s\n", d.name.UTF8String);
        printf("recommendedMaxWorkingSetSize: %llu\n", (unsigned long long)d.recommendedMaxWorkingSetSize);
        printf("hasUnifiedMemory: %d\n", d.hasUnifiedMemory ? 1 : 0);
        printf("maxBufferLength: %llu\n", (unsigned long long)d.maxBufferLength);
        const struct { const char *name; MTLGPUFamily family; } families[] = {
            { "Apple7", MTLGPUFamilyApple7 }, { "Apple8", MTLGPUFamilyApple8 }, { "Apple9", MTLGPUFamilyApple9 },
        };
        for (size_t i = 0; i < sizeof(families) / sizeof(families[0]); i++)
            printf("supportsFamily %s: %d\n", families[i].name, [d supportsFamily:families[i].family] ? 1 : 0);
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
        if (@available(macOS 26.0, *)) {
            printf("supportsFamily Metal4: %d\n", [d supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0);
            printf("newMTL4CommandQueue: %d\n", [d respondsToSelector:@selector(newMTL4CommandQueue)] ? 1 : 0);
        } else {
            printf("supportsFamily Metal4: unavailable (macOS < 26)\n");
        }
#else
        printf("supportsFamily Metal4: not in this SDK\n");
#endif
    }
    return 0;
}
