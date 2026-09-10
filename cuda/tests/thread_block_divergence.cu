#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void warmup(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f * 2.0f;
    } else {
        b = 200.0f * 2.0f;
    }

    c[tid] = a + b;
}

__global__ void kernel1(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;

    if (tid % 2 == 0) {
        a = 100.0f * 2.0f;
    } else {
        b = 200.0f * 2.0f;
    }

    c[tid] = a + b;
}

__global__ void kernel2(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f * 2.0f;
    } else {
        b = 200.0f * 2.0f;
    }

    c[tid] = a + b;
}

__global__ void kernel3(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;
    int ipred = (tid % 2) == 0;

    if (ipred) {
        a = 100.0f * 2.0f;
    } else {
        b = 200.0f * 2.0f;
    }

    c[tid] = a + b;
}

int main(int argc, char **argv) {
    // 设备信息
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // 大小
    int size = 64;
    int block_size = 64;

    if (argc > 1) {
        block_size = atoi(argv[1]);
    }
    if (argc > 2) {
        size = atoi(argv[2]);
    }
    printf("Data size %d ", size);

    dim3 block(block_size, 1);
    dim3 grid((size - 1) / block.x + 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // 分配内存
    float *C_dev;
    size_t n_bytes = size * sizeof(float);
    float *C_host = (float *)malloc(n_bytes);
    cudaMalloc((float **)&C_dev, n_bytes);

    // warmup
    for (size_t i = 0; i < 2; i++) {
        cudaDeviceSynchronize();
        warmup<<<grid, block>>>(C_dev);
        cudaDeviceSynchronize();
    }

    // kernel1
    cudaDeviceSynchronize();
    kernel1<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();
    cudaMemcpy(C_host, C_dev, n_bytes, cudaMemcpyDeviceToHost);

    // kernel2
    cudaDeviceSynchronize();
    kernel2<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();

    // kernel3
    cudaDeviceSynchronize();
    kernel3<<<grid, block>>>(C_dev);
    cudaDeviceSynchronize();

    free(C_host);
    cudaFree(C_dev);

    return 0;
}