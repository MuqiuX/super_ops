#include "ops/add.h"

__global__ void add(float *input, float *other, float *out, int M, int N) {
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;

  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;

  for (int row = iy; row < M; row += stride_y) {
    for (int col = ix; col < N; col += stride_x) {
      int idx = row * N + col;
      out[idx] = input[idx] + other[idx];
    }
  }
}

void add_r(float *input, float *other, float *out, const int size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = input[i] + other[i];
  }
}