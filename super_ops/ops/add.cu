#include <ATen/ops/add.h>
#include <torch/extension.h>

__global__ void add_kernel(const float *__restrict__ a,
                           const float *__restrict__ b, float *__restrict__ c,
                           int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

at::Tensor add_forward(const at::Tensor &a, const at::Tensor &b,
                       const at::Scalar &alpha = 1) {
  if (a.scalar_type() != at::kFloat || !a.is_cuda())
    return at::add(a, b, alpha);

  auto out = at::empty_like(a);
  auto a_flat = a.reshape(-1).contiguous();
  auto b_flat = alpha.equal(1) ? b.reshape(-1).contiguous()
                               : b.mul(alpha).reshape(-1).contiguous();
  int N = a.numel();
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(a_flat.data_ptr<float>(),
                                  b_flat.data_ptr<float>(),
                                  out.data_ptr<float>(), N);
  return out;
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) { m.impl("add.Tensor", add_forward); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}