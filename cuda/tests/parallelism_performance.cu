#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

__global__ void addmm(float *a, float *b, float *c, int m, int n) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = ix + iy * m;
    if (ix < n && iy < m) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char **argv) {
    // 设备信息
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // 大小
    int m = 1 << 14;
    int n = 1 << 14;
    int size = m * n;
    int n_bytes = size * sizeof(float);

    // Malloc
    float *A_host = (float *)malloc(n_bytes);
    float *B_host = (float *)malloc(n_bytes);
    float *C_host = (float *)malloc(n_bytes);
    float *C_from_gpu = (float *)malloc(n_bytes);
    initialData(A_host, size);
    initialData(B_host, size);

    // cudaMalloc
    float *A_dev = NULL;
    float *B_dev = NULL;
    float *C_dev = NULL;
    CHECK(cudaMalloc((void **)&A_dev, n_bytes));
    CHECK(cudaMalloc((void **)&B_dev, n_bytes));
    CHECK(cudaMalloc((void **)&C_dev, n_bytes));

    CHECK(cudaMemcpy(A_dev, A_host, n_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, n_bytes, cudaMemcpyHostToDevice));

    int dimx = argc > 2 ? atoi(argv[1]) : 32;
    int dimy = argc > 2 ? atoi(argv[2]) : 32;

    double iStart, iElaps;

    // 2d block and 2d grid
    dim3 block(dimx, dimy);
    dim3 grid(cdiv(n, block.x), cdiv(m, block.y));
    iStart = cpuSecond();
    addmm<<<grid, block>>>(A_dev, B_dev, C_dev, n, m);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)|%f sec\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaMemcpy(C_from_gpu, C_dev, n_bytes, cudaMemcpyDeviceToHost));

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);
    cudaDeviceReset();
    return 0;
}