#include "ops/add.h"
#include "utils/utils.h"
#include <stdio.h>

__global__ void add_kernel(const float *__restrict__ a,
                           const float *__restrict__ b, float *__restrict__ c,
                           int M, int N) {
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < N && row < M) {
    int idx = col + row * N;
    c[idx] = a[idx] + b[idx];
  }
}

cudaError_t add(const float *a, const float *b, float *c, int M, int N) {
  int n_bytes = sizeof(float) * M * N;

  float *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, n_bytes);
  cudaMalloc(&d_b, n_bytes);
  cudaMalloc(&d_out, n_bytes);

  cudaMemcpy(d_a, a, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n_bytes, cudaMemcpyHostToDevice);

  dim3 block(32, 8);
  dim3 grid(cdiv(N, block.x), cdiv(M, block.y));
  add_kernel<<<grid, block>>>(d_a, d_b, d_out, M, N);

  cudaMemcpy(c, d_out, n_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  return cudaGetLastError();
}

void add_r(float *a, float *b, float *c, const int size) {
  for (size_t i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}