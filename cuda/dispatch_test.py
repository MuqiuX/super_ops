import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.cpp_extension import load

my_ops = load(
    name="my_ops",
    sources=["torch_bridge/torch_bridge.cu"],
    extra_cuda_cflags=["--use_fast_math"],
    verbose=True,
)

class MyDispatch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.add.Tensor:
            a, b = args[0].contiguous(), args[1].contiguous()
            if a.dtype == torch.float32 and a.is_cuda:
                return my_ops.add_op(a, b)
        return func(*args, **kwargs)

# test
a = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)

with MyDispatch():
    c = a + b

ref = a + b
print("max diff:", (c - ref).abs().max().item())
print("PASS" if (c - ref).abs().max().item() < 1e-5 else "FAIL")