/**
 * 线程束分化
 */
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * baseline cpu归约
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

/**
 * warmup
 */
__global__ void warmup(float *input, float *out, unsigned int n) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    input += blockDim.x * blockIdx.x;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            input[tid] += input[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = input[0];
    }
}

/**
 * 交错配对归约
 */
__global__ void sum_kernel(float *input, float *out, unsigned int n) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    input += blockDim.x * blockIdx.x;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            input[tid] += input[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = input[0];
    }
}

/**
 * 交错配对归约-展开每block计算两block
 */
__global__ void sum_kernel_extended_2(float *input, float *out, unsigned int n) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    if (idx >= n) {
        return;
    }

    input += blockDim.x * blockIdx.x * 2;

    // 通过先与下一个block进行求和，减少一个block的归约
    if (idx + blockDim.x < n) {
        input[tid] += input[tid + blockDim.x];
    }

    // 注意得等所有线程的求和全部完成才能进入归约
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            input[tid] += input[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = input[0];
    }
}


/**
 * 交错配对归约-展开每block计算4block
 */
__global__ void sum_kernel_extended_4(float *input, float *out, unsigned int n) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;

    if (idx >= n) {
        return;
    }

    input += blockDim.x * blockIdx.x * 4;

    // 通过先与下一个block进行求和，减少一个block的归约
    if (idx + blockDim.x < n) {
        input[tid] += input[tid + blockDim.x * 1];
        input[tid] += input[tid + blockDim.x * 2];
        input[tid] += input[tid + blockDim.x * 3];
    }

    // 注意得等所有线程的求和全部完成才能进入归约
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            input[tid] += input[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = input[0];
    }
}


/**
 * 交错配对归约-展开每block计算8block
 */
__global__ void sum_kernel_extended_8(float *input, float *out, unsigned int n) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;

    if (idx >= n) {
        return;
    }

    input += blockDim.x * blockIdx.x * 8;

    // 通过先与下一个block进行求和，减少一个block的归约
    if (idx + blockDim.x < n) {
        input[tid] += input[tid + blockDim.x * 1];
        input[tid] += input[tid + blockDim.x * 2];
        input[tid] += input[tid + blockDim.x * 3];
        input[tid] += input[tid + blockDim.x * 4];
        input[tid] += input[tid + blockDim.x * 5];
        input[tid] += input[tid + blockDim.x * 6];
        input[tid] += input[tid + blockDim.x * 7];
    }

    // 注意得等所有线程的求和全部完成才能进入归约
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            input[tid] += input[tid + stride];
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

    // sum_kernel_extended_2
    int grid_size_2 = (grid_size + 1) / 2;
    dim3 block_2(block_size, 1);
    dim3 grid_2(grid_size_2, 1);

    CHECK(cudaMemcpy(input_dev, input_host, n_bytes, cudaMemcpyHostToDevice));
    sum_kernel_extended_2<<<grid_2, block_2>>>(input_dev, out_dev, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost));
    printf("GPU EXTENDED_2 OUT: %f <<<grid: %d, block: %d>>>\n", sum_cpu(out_host, grid_size_2), grid_2.x, block_2.x);

    // sum_kernel_extended_4
    int grid_size_4 = (grid_size + 3) / 4;
    dim3 block_4(block_size, 1);
    dim3 grid_4(grid_size_4, 1);

    CHECK(cudaMemcpy(input_dev, input_host, n_bytes, cudaMemcpyHostToDevice));
    sum_kernel_extended_4<<<grid_4, block_4>>>(input_dev, out_dev, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost));
    printf("GPU EXTENDED_4 OUT: %f <<<grid: %d, block: %d>>>\n", sum_cpu(out_host, grid_size_4), grid_4.x, block_4.x);

    // sum_kernel_extended_8
    int grid_size_8 = (grid_size + 7) / 8;
    dim3 block_8(block_size, 1);
    dim3 grid_8(grid_size_8, 1);

    CHECK(cudaMemcpy(input_dev, input_host, n_bytes, cudaMemcpyHostToDevice));
    sum_kernel_extended_8<<<grid_8, block_8>>>(input_dev, out_dev, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost));
    printf("GPU EXTENDED_8 OUT: %f <<<grid: %d, block: %d>>>\n", sum_cpu(out_host, grid_size_8), grid_8.x, block_8.x);

    // sum_cpu
    printf("CPU OUT: %f\n", sum_cpu(input_host, size));

    cudaFree(input_dev);
    cudaFree(out_dev);
    free(input_host);
    free(out_host);

    cudaDeviceReset();
    return 0;
}