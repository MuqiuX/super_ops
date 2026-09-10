#include "ops/sum.h"
#include "accuracy_utils.h"
#include "utils.h"
#include <cuda_runtime.h>

int main(int argc, char **argv) {
  cudaSetDevice(0);
  run_accuracy_test(SUM_SHAPES, sum, sum_r);
  return 0;
}