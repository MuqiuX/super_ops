#include "utils/accuracy_utils.h"
#include <cmath>
#include <random>
#include <stdio.h>

void init_data(float *data, int N, InitMode mode, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 2.0f);

  switch (mode) {
  case InitMode::ZERO:
    memset(data, 0, N * sizeof(float));
    break;
  case InitMode::ONES:
    std::fill_n(data, N, 1.0f);
    break;
  case InitMode::UNIFORM:
    for (int i = 0; i < N; i++)
      data[i] = dist(rng);
    break;
  case InitMode::RAMP:
    for (int i = 0; i < N; i++)
      data[i] = (float)i;
    break;
  case InitMode::MOD:
    for (int i = 0; i < N; i++)
      data[i] = (float)(i % 100) / 3.0f;
    break;
  }
}

std::pair<int, int> check_accuracy(const float *actual, const float *expected,
                                   int N, double rtol, double atol,
                                   const std::string &name) {
  int pass = 0, fail = 0;
  for (int i = 0; i < N; i++) {
    double diff = std::abs(actual[i] - expected[i]);
    double tol = atol + rtol * std::abs(expected[i]);
    if (diff > tol) {
      if (fail < 5) // only print first 5 failures
        printf("[FAIL] %s[%d]: actual=%f expected=%f diff=%e\n", name.c_str(),
               i, actual[i], expected[i], diff);
      fail++;
    } else {
      pass++;
    }
  }
  return {pass, fail};
}

bool check_scalar(float actual, float expected, double rtol, double atol,
                  const std::string &name) {
  double diff = std::abs(actual - expected);
  double tol = atol + rtol * std::abs(expected);
  if (diff > tol) {
    printf("[FAIL] %s: actual=%f expected=%f diff=%e\n", name.c_str(), actual,
           expected, diff);
    return false;
  }
  return true;
}

void run_accuracy_test(const std::vector<Shape> &shapes,
                       std::function<float(float *, int)> gpu_op,
                       std::function<float(float *, int)> cpu_ref) {
  printf("\n%-12s %8s %12s %12s %s\n", "shape", "N", "GPU", "CPU", "result");
  printf("-----------------------------------------------------\n");

  int total = 0, passed = 0;

  for (auto &s : shapes) {
    int N = s.M * s.N;
    float *data = (float *)malloc(N * sizeof(float));
    init_data(data, N, InitMode::MOD);

    float gpu_result = gpu_op(data, N);
    float cpu_result = cpu_ref(data, N);
    bool ok = check_scalar(gpu_result, cpu_result, 1e-5, 1e-6, s.desc);

    printf("%-12s %8d %12.4f %12.4f [%s]\n", s.desc.c_str(), N, gpu_result,
           cpu_result, ok ? "PASS" : "FAIL");

    total++;
    if (ok)
      passed++;
    free(data);
  }

  printf("-----------------------------------------------------\n");
  printf("Summary: %d/%d passed\n\n", passed, total);
}