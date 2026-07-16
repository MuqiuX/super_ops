# AGENTS.md — super_ops

## Overview

Custom CUDA ops as drop-in replacements for PyTorch via `TorchDispatchMode`.
Wraps low-level kernels (CUDA, Triton) so that `torch.add`, `torch.median`, etc.
transparently dispatch to optimized implementations.

## Architecture

- **`super_ops/ops/`** — JIT-compiled CUDA extensions loaded via `torch.utils.cpp_extension.load`.
  Each `.cu` file registers a `TORCH_LIBRARY_IMPL(aten, CUDA, m)` override.
  Calling `super_ops.enable()` activates the dispatch mode; `disable()` restores default.
- **`cuda/`** — Standalone CMake project (targets SM89) for the same CUDA kernels plus tests/benchmarks.
  Separate from the PyTorch-extension path; build with `cmake` + `make`.
- **`triton/`** — Triton-language kernels (median, sort, topk) using `flag_gems` internal framework
  and TLE (Triton Language Extensions for explicit shared-memory programming).
- **`tests/`** — End-to-end tests for the `super_ops` package.
- **`benchmarks/`** — Empty (benchmarking lives in `triton/median/bench_median.py` for now).

## Commands

```bash
# Run the only in-package test (requires CUDA GPU)
python tests/test_add.py

# Triton median tests (uses pytest markers, needs flag_gems + CUDA)
pytest triton/median/test_median.py -v -m median

# Triton median benchmark
python triton/median/bench_median.py

# Standalone CUDA build & test
cd cuda && mkdir -p build && cd build && cmake .. && make
./test_add
./test_sum
```

## Dependencies

- Python >=3.10, PyTorch >=2.0
- `flag_gems` — internal framework used by triton kernels (provides `device`, `testing.assert_close`, `dim_compress`)
- TLE (`triton.experimental.tle`) — required for shared-memory Triton kernels; installed alongside Triton
- CUDA toolkit (for `cuda/` CMake build; SM89 by default)
- Conda is the project's Python env manager (see `.vscode/settings.json`)

## Conventions

- **Op filenames match PyTorch op names** (e.g., `add.cu` overrides `torch.add`).
- **CUDA JIT flags**: `--use_fast_math` is always set in `super_ops/ops/__init__.py`.
- **Tests need a CUDA GPU** — no CPU fallback; all test tensors are created on `cuda`.
- **Triton tests** use `@pytest.mark.median`, `@pytest.mark.sort` markers; `assert_close` from `flag_gems.testing`.
- **No linter/formatter configured** yet. Use standard Python style.

## Gotchas

- `super_ops.enable()` must be called before tensor operations to activate overrides.
  The test pattern is: enable → run ops → disable.
- The `super_ops/__init__.py` `disable()` is a no-op stub; dispatch cleanup isn't implemented yet.
- `triton/median/median.py` has `if __name__ == '__main__'` that serves as a quick smoke test.
- Triton code in `triton/` is NOT automatically loaded by `super_ops.enable()` —
  only the CUDA extensions in `super_ops/ops/` are registered via the dispatch mode.
