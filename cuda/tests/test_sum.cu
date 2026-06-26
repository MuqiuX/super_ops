#include "ops/sum.h"
#include "utils/accuracy_utils.h"
#include "utils/utils.h"
#include <cuda_runtime.h>

int main(int argc, char **argv) {
  cudaSetDevice(0);
  run_accuracy_test(SUM_SHAPES, sum, sum_r);
  return 0;
}