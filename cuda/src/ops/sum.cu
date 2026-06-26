#include "ops/sum.h"
#include "utils/utils.h"

__global__ void sum_kernel(const float *__restrict__ input,
                           float *__restrict__ partial_sums, int N) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  sdata[tid] = idx < N ? input[idx] : 0.0f;
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride <<= 1) {
    int index = stride * 2 * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

float sum(const float *input, int N) {
  const int block_size = 256;
  const int grid_size = cdiv(N, block_size);

  const int n_bytes = N * sizeof(float);
  const int partial_bytes = grid_size * sizeof(float);

  float *d_input, *d_partial;
  cudaMalloc(&d_input, n_bytes);
  cudaMalloc(&d_partial, partial_bytes);

  cudaMemcpy(d_input, input, n_bytes, cudaMemcpyHostToDevice);

  sum_kernel<<<grid_size, block_size, block_size * 2 * sizeof(float)>>>(
      d_input, d_partial, N);

  float *partial = (float *)malloc(partial_bytes);
  cudaMemcpy(partial, d_partial, partial_bytes, cudaMemcpyDeviceToHost);

  float sum = 0.0f;
  for (int i = 0; i < grid_size; i++) {
    sum += partial[i];
  }

  free(partial);
  cudaFree(d_input);
  cudaFree(d_partial);

  return sum;
}

float sum_r(float *input, int N) {
  if (N == 1) {
    return input[0];
  }

  const int n_bytes = N * sizeof(float);
  float *data = (float *)malloc(n_bytes);
  memcpy(data, input, n_bytes);

  for (int stride = 1; stride < N; stride <<= 1) {
    for (int i = 0; i < N; i += stride * 2) {
      if (i + stride < N) {
        data[i] += data[i + stride];
      }
    }
  }

  float sum = data[0];
  free(data);
  return sum;
}