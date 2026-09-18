# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Learning/project repo for hand-written CUDA/Triton kernels. Long-term goal (per commit history): replace PyTorch implementations with self-written CUDA kernels to optimize Qwen3-8B inference on an RTX 4090D (SM89).

## Remote-development workflow (important)

Code is built and run on a **remote Linux GPU machine**, then synced back to this Windows machine. Locally there is no GPU and no CUDA toolkit:

- Never build CUDA or run CUDA-dependent tests locally — it cannot work here.
- Artifacts such as `cuda/build/libutils.a` are synced Linux binaries, not local builds.
- When execution is needed, provide the shell commands for the user to run remotely.

Repo docs (`cuda/NSIGHT.md`, `cuda/tests/RESULTS.md`, `triton/TLE.md`, etc.) are written in Chinese.

## Three kernel tracks

1. **`super_ops/`** — PyTorch drop-in ops. `super_ops/ops/*.cu` are JIT-compiled via `torch.utils.cpp_extension.load` (`super_ops/ops/__init__.py`). Each `.cu` registers `TORCH_LIBRARY_IMPL(aten, CUDA, m)`, so once loaded the custom kernel **globally overrides the ATen op for the whole process** — this is not a TorchDispatchMode. `super_ops.enable()` only triggers JIT loading (`load_all()`); `disable()` is a no-op stub. Non-float32/non-CUDA inputs fall back to `at::add` inside the op. Adding a new op requires both a new `.cu` and an entry in the `_ALL` list in `super_ops/ops/__init__.py`.
2. **`cuda/`** — standalone CMake project (pure CUDA, no PyTorch) for learning kernels. The `utils` static lib plus active targets `prop` (device-property CLI) and `parallelism_performance`; `main`, `test_add`, `test_sum`, `thread_block_divergence` sources exist but their targets are commented out in CMakeLists. CUDA 14, `CMAKE_CUDA_ARCHITECTURES=86`, `--ptxas-options=-v` for register-usage output. `.clang-format` lives in this dir.
3. **`triton/`** — Triton kernels built on `flag_gems` utilities (`dim_compress`, `device`, `testing.assert_close`) and TLE (`triton.experimental.tle`) for explicit shared-memory programming. `median/` (radix-select kth → `median_dim`, plus benchmark), `sort/bitonic_sort.py`, `topk.py`. `triton/TLE.md` is a full TLE API reference (Chinese).

`cuda/dispatch_test.py` + `cuda/torch_bridge/torch_bridge.cu` are the earlier TorchDispatchMode prototype that the `super_ops` package superseded.

## Commands (require the remote GPU machine)

```bash
# super_ops package test (JIT-compiles add.cu, then runs)
python tests/test_add.py

# Triton median tests (markers: median, sort)
pytest triton/median/test_median.py -v -m median

# Triton median benchmark vs torch.median
python triton/median/bench_median.py

# Standalone CUDA build (build/ is gitignored but may be synced from remote)
cd cuda && cmake -S . -B build && cmake --build build
./build/prop
./build/parallelism_performance

# Profiling (see cuda/NSIGHT.md)
ncu --set full build/<target>
nsys profile --stats=true build/<target>
```

## Conventions

- Op source files are named after the PyTorch op they override (`add.cu` → `torch.add.Tensor`).
- `--use_fast_math` is always set for super_ops JIT builds.
- All tests create tensors on `cuda` — no CPU fallback paths.
- Triton tests use pytest markers (`@pytest.mark.median`, `@pytest.mark.sort`) and `assert_close` from `flag_gems.testing`, comparing against CPU `torch` references.
- No Python linter/formatter configured; conda is the env manager.

## Gotchas

- `triton/__init__.py` imports `super_ops.ops.triton.median`, which does not exist — importing the `triton` package from the repo root fails. Run scripts from their own subdirectory instead.
- Triton kernels are NOT wired into `super_ops` dispatch — the package only registers the CUDA extensions in `super_ops/ops/`.
- CMake arch is pinned to SM86; sm_86 cubins are binary-compatible with the 4090D (SM89), but set 89 when using SM89-specific features.
- `triton/median/test_median.py` and `bench_median.py` do a sibling import (`from median import median_dim`) — works because pytest inserts the test file's directory into `sys.path`.
