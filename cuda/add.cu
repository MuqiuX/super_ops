#include <cuda_runtime.h>

__global__ void add(float *input, float *other, float *out) {
    int i = threadIdx.x;
    out[i] = input[i] + other[i];
}