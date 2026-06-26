#include "ops/add.h"
#include "utils/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  cudaSetDevice(0);

  int nx = 4096, ny = 4066;
  int N = nx * ny;
  int n_bytes = sizeof(float) * N;

  float *h_a = (float *)malloc(n_bytes);
  float *h_b = (float *)malloc(n_bytes);
  float *h_out = (float *)malloc(n_bytes);
  float *ref_out = (float *)malloc(n_bytes);

  for (int i = 0; i < N; i++) {
    h_a[i] = (float)(i % 100) / 3.0f;
    h_b[i] = (float)((i * 7) % 50) / 2.0f;
  }

  add(h_a, h_b, h_out, ny, nx);

  add_r(h_a, h_b, ref_out, N);

  checkResult(ref_out, h_out, N);

  free(h_a);
  free(h_b);
  free(h_out);
  free(ref_out);
  return 0;
}