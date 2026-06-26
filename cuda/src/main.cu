#include "utils/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "ops/sum.h"

int main(int argc, char ** argv) {
  cudaSetDevice(0);
  
  int N = 524288;
  int n_bytes = sizeof(float) * N;

  float *h_a;
  h_a = (float *)malloc(n_bytes);
  memset(h_a, 0, n_bytes);

  float res_out = sum(h_a, N);

  float ref_out = sum_r(h_a, N);

  checkResult(ref_out, res_out, N);

  free(h_a);

  return 0;
}