#include "utils/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "ops/add.h"

int main() {
  cudaSetDevice(0);
  
  int nx = 4096, ny = 4066;
  int N = nx * ny;
  int n_bytes = sizeof(float) * (nx * ny);

  float *h_a;
  float *h_b;
  float *h_out;
  h_a = (float *)malloc(n_bytes);
  h_b = (float *)malloc(n_bytes);
  h_out = (float *)malloc(n_bytes);
  memset(h_a, 0, n_bytes);
  memset(h_b, 0, n_bytes);
  memset(h_out, 0, n_bytes);

  float *d_a;
  float *d_b;
  float *d_out;
  CHECK(cudaMalloc((float **)&d_a, n_bytes));
  CHECK(cudaMalloc((float **)&d_b, n_bytes));
  CHECK(cudaMalloc((float **)&d_out, n_bytes));
  cudaMemset(d_a, 0, n_bytes);
  cudaMemset(d_b, 0, n_bytes);
  cudaMemset(d_out, 0, n_bytes);

  CHECK(cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice));

  dim3 block(8, 32);
  dim3 grid(1, 1);

  add<<<grid, block>>>(d_a, d_b, d_out, nx, ny);

  CHECK(cudaMemcpy(h_out, d_out, n_bytes, cudaMemcpyDeviceToHost));

  float *ref_out;
  ref_out = (float *)malloc(n_bytes);
  memset(ref_out, 0, n_bytes);

  add_r(h_a, h_b, ref_out, N);

  checkResult(ref_out, h_out, N);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  free(h_a);
  free(h_b);
  free(h_out);

  return 0;
}