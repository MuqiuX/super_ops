/**
 * 线程束分化
 */
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * cpu归约
 */
float sum_cpu(float *data, const int size) {
    if (size == 1) {
        return data[0];
    }

    int stride = size / 2;

    for (size_t i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    return sum_cpu(data, stride);
}


__global__ void warmup(float *input, float *out, unsigned int n) {

    int tid = threadIdx.x;

    if (tid + blockIdx.x * blockDim.x >= n) {
        return;
    }

    input += blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tid % (stride << 1) == 0) {
            input[tid] += input[tid + stride];
        }

        // 等待一个block中的所有线程完成之后进入下一步
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = input[0];
    }
}


/**
 * 存在严重的线程分化, 线程id对应内存地址
 */
__global__ void sum_kernel(float *input, float *out, unsigned int n) {

    int tid = threadIdx.x;

    if (tid + blockIdx.x * blockDim.x >= n) {
        return;
    }

    input += blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tid % (stride << 1) == 0) {
            input[tid] += input[tid + stride];
        }

        // 等待一个block中的所有线程完成之后进入下一步
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = input[0];
    }
}


/**
 * 优化后的
 */
__global__ void sum_kernel_better(float *input, float *out, unsigned int n) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    input += blockDim.x * blockIdx.x;

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            input[index] += input[index + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = input[0];
    }
}

int main(int argc, char **argv) {
    // 设备信息
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // 大小
    int size = 1 << 24;
    int n_bytes = size * sizeof(float);

    int block_size = 512;

    if (argc > 1) {
        block_size = atoi(argv[1]);
    }

    int grid_size = cdiv(size, block_size);
    int out_size = grid_size;
    int out_bytes = grid_size * sizeof(float);

    dim3 block(block_size, 1);
    dim3 grid(grid_size, 1);

    // Malloc
    float *input_host = (float *)malloc(n_bytes);
    float *out_host = (float *)malloc(out_bytes);
    initialData(input_host, size);
    initialData(out_host, out_size);

    // cudaMalloc
    float *input_dev = NULL;
    float *out_dev = NULL;
    CHECK(cudaMalloc((void **)&input_dev, n_bytes));
    CHECK(cudaMalloc((void **)&out_dev, out_bytes));
    CHECK(cudaMemcpy(input_dev, input_host, n_bytes, cudaMemcpyHostToDevice));

    // warmup
    warmup<<<grid, block>>>(input_dev, out_dev, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost));
    printf("GPU WARMUP OUT: %f <<<grid: %d, block: %d>>>\n", sum_cpu(out_host, out_size), grid.x, block.x);

    // sum_kernel
    CHECK(cudaMemcpy(input_dev, input_host, n_bytes, cudaMemcpyHostToDevice));
    sum_kernel<<<grid, block>>>(input_dev, out_dev, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost));
    printf("GPU OUT: %f <<<grid: %d, block: %d>>>\n", sum_cpu(out_host, out_size), grid.x, block.x);

    // sum_kernel_better
    CHECK(cudaMemcpy(input_dev, input_host, n_bytes, cudaMemcpyHostToDevice));
    sum_kernel_better<<<grid, block>>>(input_dev, out_dev, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost));
    printf("GPU BETTER OUT: %f <<<grid: %d, block: %d>>>\n", sum_cpu(out_host, out_size), grid.x, block.x);

    // sum_cpu
    printf("CPU OUT: %f\n", sum_cpu(input_host, size));

    cudaFree(input_dev);
    cudaFree(out_dev);
    free(input_host);
    free(out_host);

    cudaDeviceReset();
    return 0;
}