#include "utils/utils.h"
#include <cmath>
#include <stdio.h>

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i],
             i);
      return;
    }
  }
  printf("Check result success!\n");
}

int cdiv(int x, int y) { return (x + y - 1) / y; }