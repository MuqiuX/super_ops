#include "utils/utils.h"
#include <cmath>
#include <stdio.h>

void checkResult(float *ref, float *res, const int N) {
  double epsilon = 1.0E-8;
  for (int i = 0; i < N; i++) {
    if (abs(ref[i] - res[i]) > epsilon) {
      printf("Results don\'t match!\n");
      printf("%f(ref[%d] )!= %f(res[%d])\n", ref[i], i, res[i],
             i);
      return;
    }
  }
  printf("Check result success!\n");
}

void checkResult(float ref, float res, const int N) {
  double epsilon = 1.0E-8;
  if (abs(ref - res) > epsilon) {
    printf("Results don\'t match!\n");
    printf("%f(ref )!= %f(res)\n", ref, res);
    return;
  }
  printf("Check result success!\n");
}

int cdiv(int x, int y) { return (x + y - 1) / y; }