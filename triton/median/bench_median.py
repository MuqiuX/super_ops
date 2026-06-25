"""Benchmark for triton-based median_dim kernel vs torch.median, following flaggems benchmark patterns."""
import torch
import triton

from median import median_dim

WARMUP = 100
REPETITION = 100

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

SHAPES = [
    # 1D shapes
    (1024 * 1024,),
    (16 * 1024 * 1024,),
    # 2D shapes
    (64, 64),
    (1024, 256),
    (4096, 4096),
    (1024, 16384),
    # 3D shapes
    (64, 512, 128),
    (200, 2560, 3),
    (20, 8, 10000),
]

HEADER_FMT = "{:<12} {:<20} {:<16} {:<16} {:<16} {:<12}"
ROW_FMT = "{:<12} {:<20} {:<16.3f} {:<16.3f} {:<16.2f} {:<12.1f}"


def _run(op, *args, **kwargs):
    """Measure median latency in ms using triton.testing.do_bench."""
    fn = lambda: op(*args, **kwargs)
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REPETITION, return_mode="median")


def _gbps(inp, latency_ms):
    """Compute effective bandwidth in GB/s for median reduction (read + write)."""
    io_bytes = inp.numel() * inp.element_size() * 2
    return io_bytes * 1e-9 / (latency_ms * 1e-3)


def bench_median_overall(dtype):
    """Benchmark median() over the whole tensor (no dim)."""
    print(f"\n── median(input)  ——  dtype={dtype} ──")
    print(HEADER_FMT.format("shape", "shape_desc", "torch_ms", "triton_ms", "speedup", "triton_gbps"))
    print("-" * 92)

    for shape in SHAPES:
        inp = torch.randn(shape, dtype=dtype, device="cuda")

        t_torch = _run(torch.median, inp)
        t_triton = _run(median_dim, inp, dim=None)
        speedup = t_torch / t_triton
        bw = _gbps(inp, t_triton)

        print(ROW_FMT.format(str(shape), f"{inp.numel():,}", t_torch, t_triton, speedup, bw))


def bench_median_dim(dtype, dim=-1):
    """Benchmark median(input, dim=dim)."""
    shapes_2d_plus = [s for s in SHAPES if len(s) >= 2]

    print(f"\n── median(input, dim={dim})  ——  dtype={dtype} ──")
    print(HEADER_FMT.format("shape", "shape_desc", "torch_ms", "triton_ms", "speedup", "triton_gbps"))
    print("-" * 92)

    for shape in shapes_2d_plus:
        inp = torch.randn(shape, dtype=dtype, device="cuda")

        t_torch = _run(torch.median, inp, dim=dim)
        t_triton = _run(median_dim, inp, dim=dim)
        speedup = t_torch / t_triton
        bw = _gbps(inp, t_triton)

        print(ROW_FMT.format(str(shape), f"{inp.numel():,}", t_torch, t_triton, speedup, bw))


if __name__ == "__main__":
    for dtype in FLOAT_DTYPES:
        bench_median_overall(dtype)
        bench_median_dim(dtype, dim=-1)
