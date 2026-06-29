#include <torch/extension.h>

__global__ void add_kernel(const float *__restrict__ a,
                           const float *__restrict__ b, float *__restrict__ c,
                           int M, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N)
    c[row * N + col] = a[row * N + col] + b[row * N + col];
}

torch::Tensor add_op(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
  auto out = torch::empty_like(a);
  int M = a.size(0), N = a.size(1);
  dim3 block(32, 8);
  dim3 grid((N + 31) / 32, (M + 7) / 8);
  add_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                              out.data_ptr<float>(), M, N);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_op", &add_op, "custom add");
}