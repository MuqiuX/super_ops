import torch
import super_ops


super_ops.enable()

for shape in [(100,), (100, 200), (2, 3, 64), (1, 2, 3, 128)]:
    a = torch.randn(*shape, device='cuda', dtype=torch.float32)
    b = torch.randn(*shape, device='cuda', dtype=torch.float32)
    c = torch.add(a, b)
    ref = a.float() + b.float()
    d = (c.cpu() - ref.cpu()).abs().max().item()
    print(f"{str(shape):20s}  max diff: {d:.2e}  {'PASS' if d < 1e-5 else 'FAIL'}")

super_ops.disable()