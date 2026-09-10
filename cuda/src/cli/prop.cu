#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int dev = 0;

    if (argc > 1) {
        dev = atoi(argv[1]);
    }

    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("%s", prop.name);
    printf("%d", prop.major);
    print_prop(prop);

    return 0;
}