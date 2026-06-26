#pragma once
#include <functional>
#include <string>
#include <vector>

struct Shape {
  int M, N;
  std::string desc;
};

const std::vector<Shape> DEFAULT_SHAPES = {
    {1, 1024, "tiny"},
    {1024, 1024, "1K×1K"},
    {256, 4096, "medium"},
    {4096, 4096, "square"},
};

const std::vector<Shape> SUM_SHAPES = {
    {1024, 1, "1K"},    {65536, 1, "64K"},   {524288, 1, "512K"},
    {1 << 20, 1, "1M"}, {1 << 24, 1, "16M"},
};

enum class InitMode {
  ZERO,
  ONES,
  UNIFORM,
  RAMP,
  MOD,
};

void init_data(float *data, int N, InitMode mode = InitMode::MOD,
               unsigned seed = 42);

std::pair<int, int> check_accuracy(const float *actual, const float *expected,
                                   int N, double rtol = 1e-5,
                                   double atol = 1e-8,
                                   const std::string &name = "");

bool check_scalar(float actual, float expected, double rtol = 1e-5,
                  double atol = 1e-8, const std::string &name = "");

void run_accuracy_test(const std::vector<Shape> &shapes,
                       std::function<float(float *, int)> gpu_op,
                       std::function<float(float *, int)> cpu_ref);