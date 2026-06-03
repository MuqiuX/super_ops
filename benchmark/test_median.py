import torch
import triton
import 

SHAPES = [
    ((32768, 1), (1, 32)),
    ((1024, 1), (1, 1024)),
    ((10000, 1), (1, 64)),
    ((1024, 1024), (1024, 1024)),
    ((4096, 4096), (4096, 4096)),
]


def bench_mul(a_shape, b_shape, warmup=25, rep=100):
    a = torch.randn(a_shape, device="cuda", dtype=torch.float32)
    b = torch.randn(b_shape, device="cuda", dtype=torch.float32)

    # --- latency_base: PyTorch native ---
    def native():
        return torch.mul(a, b)

    latency_base = triton.testing.do_bench(
        native, warmup=warmup, rep=rep, return_mode="median"
    )

    # --- latency: FlagGems ---
    with flag_gems.use_gems():
        latency_gems = triton.testing.do_bench(
            native, warmup=warmup, rep=rep, return_mode="median"
        )

    # --- correctness ---
    ref = native()
    with flag_gems.use_gems():
        res = native()
    correct = torch.allclose(res, ref, atol=1e-4, rtol=1.3e-6)
    if not correct:
        mismatch = (res - ref).abs()
        print(f"  ❌ MISMATCH! max_diff={mismatch.max():.6e}, shape={tuple(res.shape)} vs ref={tuple(ref.shape)}")

    return latency_base, latency_gems, correct


print("=" * 72)
print("mul benchmark — PyTorch native vs FlagGems")
print(f"triton {triton.__version__}, torch {torch.__version__}")
print("=" * 72)
print(f"{'a_shape':>16} {'b_shape':>16} {'native_ms':>10} {'gems_ms':>10} {'speedup':>8} {'correct':>8}")
print("-" * 72)

for a_shape, b_shape in SHAPES:
    base, gems, correct = bench_mul(a_shape, b_shape)
    speedup = base / gems
    print(
        f"{str(a_shape):>16} {str(b_shape):>16} "
        f"{base:10.4f} {gems:10.4f} "
        f"{speedup:7.2f}x "
        f"{'✓' if correct else '✗':>8}"
    )
