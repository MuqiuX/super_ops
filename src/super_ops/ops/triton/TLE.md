# Triton Language Extensions (TLE) 使用说明书

"""
## 概述

TLE（Triton Language Extensions）是 FlagTree 项目在 Triton 之上的扩展层，解决两个核心问题：
1. **显式 shared memory 编程** - 可分配、动态索引、原子操作
2. **跨 block 同步与协作** - grid barrier、cluster barrier、DSMEM 远程访问

三层架构：

import triton.experimental.tle as tled               # 分布式/同步
import triton.experimental.tle.language.gpu as tle   # 显存编程
from triton.experimental.tle.raw import cuda, mlir   # 底层代码生成

---

## 第一层：tle - Shared Memory 显式编程

### 1.1 tle.alloc - 分配 shared memory buffer

tle.alloc(
    shape: tuple,              # buffer 形状，如 [256] 或 [64, 128]
    dtype: tl.dtype,           # 元素类型，如 tl.int32, tl.float16
    layout=None,               # 内存布局编码，None 则自动选择
    scope=tle.smem,            # tle.smem（shared memory）或 tle.tmem（tensor memory）
    nv_mma_shared_layout=False,# False=swizzled layout, True=nv_mma layout
) -> buffered_tensor

返回值：buffered_tensor - 不是普通 tl.tensor，不能直接做算术。只代表"一块已分配的 shared memory"。

示例：

```python
# 分配一个 256 元素的 int32 histogram
hist = tle.alloc([256], dtype=tl.int32, scope=tle.smem, nv_mma_shared_layout=False)

# 分配二维 buffer
buf = tle.alloc([64, 128], dtype=tl.float16, scope=tle.smem)
```

Layout 说明：

| layout | 适用场景 |
|--------|---------|
| swizzled_shared_layout (默认) | 通用 shared memory，通过 swizzle 避免 bank conflict |
| nv_mma_shared_layout | NVIDIA MMA 指令专用，自动选择 swizzle 模式 |
| tensor_memory_layout | Tensor memory（Hopper TMA 场景） |

---

### 1.2 tle.local_ptr - 将 buffer 转换为可读写的指针（最重要的 API）

tle.local_ptr(
    buffer: buffered_tensor,    # tle.alloc 返回的 buffer
    indices: tuple,             # 索引张量元组，长度必须等于 buffer 的 rank
) -> tl.tensor                  # 返回指针张量，可用于 tl.load/tl.store/tl.atomic_add

支持用运行时变量做 shared memory 的动态索引。

**示例1：静态索引遍历（清零整个 buffer）**

```python
expert_offsets = tl.arange(0, BLOCK_EXPERT)
ptrs = tle.local_ptr(local_counts, (expert_offsets,))
tl.store(ptrs, 0, mask=expert_offsets < num_experts)
# 等价于: for i in range(BLOCK_EXPERT): local_counts[i] = 0
```

**示例2：动态索引（radix histogram 的关键！）**

```python
# 每个线程根据自己拿到的 key 值，原子加到对应 bin
expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
count_ptrs = tle.local_ptr(local_counts, (expert_id,))   # 动态索引！
tl.atomic_add(count_ptrs, 1, mask=mask, sem="relaxed", scope="cta")
# 等价于 CUDA: hist[expert_id] += 1（在 shared memory 上）
```

**示例3：标量索引（读写单个元素）**

```python
# 读 buffer 中第 expert_id 个元素
ptr = tle.local_ptr(buf, (expert_id,))
val = tl.load(ptr, mask=valid, other=0)
```

限制：
- 索引元组长度必须等于 buffer rank
- 所有非标量索引必须有相同的 shape
- 不能混用标量和张量索引（要么全标量，要么全同 shape 张量）

---

### 1.3 tle.copy - Global <-> Shared Memory 批量拷贝

tle.copy(
    src,          # tl.tensor（global memory）或 tle.buffered_tensor
    dst,          # tle.buffered_tensor 或 tl.tensor
    shape,        # 拷贝的形状
    offsets=None, # TMA 模式下的起始坐标（普通模式不需要）
)

方向自动推断：
- src 是 global tensor, dst 是 buffered_tensor -> GM -> Local
- src 是 buffered_tensor, dst 是 global tensor -> Local -> GM

示例：

```python
# 从 global memory 批量加载到 shared memory
row_ptrs = tokens_cnts_ptr + off_t + tl.arange(0, BLOCK_EXPERT)
tle.copy(row_ptrs, local_counts, [num_experts])
```

TMA 模式（需要 tl.tensor_descriptor，Hopper+）：

```python
desc = tl.make_tensor_descriptor(global_tensor, [64, 64])
tle.copy(desc, local_buf, [64, 64], [x_offset, y_offset])
```

---

### 1.4 tle.smem / tle.tmem - scope 常量

```python
tle.smem   # scope('share_memory')  - 标准 shared memory
tle.tmem   # scope('tensor_memory') - Tensor Memory（Hopper 架构）
```

---

## 第二层：tled - 分布式同步与协作

### 2.1 tled.device_mesh - 声明设备拓扑

tled.device_mesh(topology: dict) -> device_mesh

**示例1：Grid 拓扑（跨所有 block）**

```python
mesh = tled.device_mesh({"block": [("block_x", NUM_BLOCKS)]})
# 声明一个 shape=[NUM_BLOCKS] 的一维 block mesh
```

**示例2：Cluster 拓扑（SM90+）**

```python
mesh = tled.device_mesh({"block_cluster": [("cluster_x", 8)]})
# 声明一个包含 8 个 block 的 cluster
```

**示例3：多维拓扑**

```python
mesh = tled.device_mesh({
    "block": [("x", 4), ("y", 4)],       # 4x4=16 blocks
})
# mesh.shape = (4, 4), mesh.dim_names = ("x", "y")
```

mesh 的切片操作：

```python
sub = mesh[0, :]     # 取第 0 行的 4 个 block，shape=(4,)
sub = mesh[..., 2]   # 取第 2 列，shape=(4,)
```

---

### 2.2 tled.distributed_barrier - 跨 block 同步

tled.distributed_barrier(mesh: device_mesh | None = None)

这是 Triton 原生完全缺失的能力 - 等待 mesh 内所有 block 到达同一点。

**示例1：Grid barrier（所有 block 同步）**

```python
mesh = tled.device_mesh({"block": [("block_x", NUM_BLOCKS)]})

# Stage 1: 每个 block 各自做 local histogram
# ...

tled.distributed_barrier(mesh)
# 等所有 block 都完成 Stage 1，然后才进入 Stage 2

# Stage 2: rank 0 做全局 prefix sum
if pid == 0:
    ...
tled.distributed_barrier(mesh)

# Stage 3: 所有 block 用 prefix sum 结果继续
# ...
```

要求：需要 launch_cooperative_grid=True，且 cluster_dims=(1,1,1)。

**示例2：Cluster barrier（只同步一个 cluster 内的 block）**

```python
mesh = tled.device_mesh({"block_cluster": [("cluster_x", 8)]})
tled.distributed_barrier(mesh)
```

**示例3：Sub-mesh barrier（mesh 的子集同步）**

```python
full_mesh = tled.device_mesh({"block": [("x", 4), ("y", 4)]})
sub = full_mesh[0, :]  # 只取第 0 行
tled.distributed_barrier(sub)  # 只在第 0 行的 4 个 block 间同步
```

---

### 2.3 tled.shard_id - 获取当前 block 的坐标

tled.shard_id(
    mesh: device_mesh,
    axis: str | int,    # axis 名称或索引
) -> tl.tensor          # 返回标量 int32

示例：

```python
mesh = tled.device_mesh({"block_cluster": [("cluster_x", 8)]})

cluster_rank = tled.shard_id(mesh, "cluster_x")  # 0~7
is_rank0 = cluster_rank == 0

if is_rank0:
    # 只有 rank 0 做全局操作
    tl.store(num_tokens_post_pad_ptr, total_tokens)
```

---

### 2.4 tled.remote - 跨 block 远程访问 shared memory（DSMEM）

tled.remote(
    tensor: buffered_tensor,  # 本地 shared memory buffer
    shard_id: int,            # 目标 block 的编号
    scope: device_mesh,       # 用于验证拓扑
) -> buffered_tensor          # 带有远程标记的 buffer

仅在 SM90+（Hopper）上可用，利用 Distributed Shared Memory 实现 cluster 内跨 block 直接访问。

完整示例：

```python
mesh = tled.device_mesh({"block_cluster": [("cluster_x", 8)]})

cluster_rank = tled.shard_id(mesh, "cluster_x")
is_rank0 = cluster_rank == 0

# 每个 block 各自分配 shared memory
cumsum_local = tle.alloc([BLOCK_EXPERT], dtype=tl.int32, scope=tle.smem,
                         nv_mma_shared_layout=False)

# rank 0 先把 cumsum_local 清零
rank0_ptrs = tle.local_ptr(cumsum_local, (tl.arange(0, BLOCK_EXPERT),))
if is_rank0:
    tl.store(rank0_ptrs, 0, mask=expert_mask)

tled.distributed_barrier(mesh)  # 等 rank0 清零完成

# 每个 block 做 local histogram
local_counts_vals = ...  # 256 个 bin 的计数结果

# 通过 DSMEM 直接原子加到 rank0 的共享内存上！（不需要写 global memory）
rank0_cumsum_remote = tled.remote(cumsum_local, 0, scope=mesh)
rank0_remote_ptrs = tle.local_ptr(rank0_cumsum_remote, (expert_offsets,))
prefix_before = tl.atomic_add(rank0_remote_ptrs, local_counts_vals,
                               mask=expert_mask, sem="relaxed", scope="cta")

tled.distributed_barrier(mesh)

# 现在 rank0 的 cumsum_local 里有全局 prefix sum
# 非 rank0 的 block 通过 remote 读取 rank0 的结果
rank0_data = tl.load(tle.local_ptr(
    tled.remote(cumsum_local, 0, scope=mesh), (expert_offsets,)),
    mask=expert_mask, other=0)
```

关键点：全程不需要 global memory 做中间存储，所有跨 block 通信都在 shared memory / DSMEM 上完成。

---

## 第三层：tle.raw - 底层代码生成

### 3.1 raw.cuda - JIT 编译 CUDA kernel

```python
from triton.experimental.tle.raw import cuda

@cuda.jit
def my_cuda_kernel(a, b, c, N):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < N:
        c[tid] = a[tid] + b[tid]

# 启动
my_cuda_kernel[grid, block](a, b, c, N)
```

适用场景：需要直接写 CUDA C++ 级别的内核，Triton 表达能力不够时使用。

---

### 3.2 raw.mlir - MLIR 代码生成

```python
from triton.experimental.tle.raw import mlir

@mlir.jit
def my_mlir_kernel(...):
    ...
```

适用场景：需要手写 MLIR 进行底层优化。

---

## 实战模式总结

### 模式 A：单 block shared memory histogram

```python
@triton.jit
def single_block_histogram(inp_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    # 1. 分配 shared memory histogram
    hist = tle.alloc([256], dtype=tl.int32, scope=tle.smem, nv_mma_shared_layout=False)
    hist_ptrs = tle.local_ptr(hist, (tl.arange(0, 256),))
    tl.store(hist_ptrs, 0)  # 清零

    # 2. 加载数据，动态索引写入 shared memory histogram
    offsets = tl.arange(0, BLOCK_N)
    for start in range(0, N, BLOCK_N):
        data = tl.load(inp_ptr + start + offsets, mask=start+offsets < N)
        keys = (data >> bit_offset) & 0xFF
        count_ptrs = tle.local_ptr(hist, (keys,))  # 动态索引！
        tl.atomic_add(count_ptrs, 1, sem="relaxed", scope="cta")

    tl.debug_barrier()  # block 内部同步

    # 3. 从 shared memory 读取结果
    # ...
```

### 模式 B：多 block 协作 + grid barrier

```python
@triton.jit
def multi_block_histogram(inp_ptr, out_ptr, N, mesh: tl.constexpr,
                           BLOCK_N: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    pid = tl.program_id(0)

    # 1. 每 block 分配自己的 shared memory histogram
    local_hist = tle.alloc([256], dtype=tl.int32, scope=tle.smem,
                           nv_mma_shared_layout=False)
    hist_ptrs = tle.local_ptr(local_hist, (tl.arange(0, 256),))
    tl.store(hist_ptrs, 0)

    # 2. 各 block 处理自己那部分数据
    for start in range(pid * BLOCK_N, N, NUM_BLOCKS * BLOCK_N):
        offsets = start + tl.arange(0, BLOCK_N)
        mask = offsets < N
        data = tl.load(inp_ptr + offsets, mask=mask)
        keys = (data >> bit_offset) & 0xFF
        count_ptrs = tle.local_ptr(local_hist, (keys,))
        tl.atomic_add(count_ptrs, 1, mask=mask, sem="relaxed", scope="cta")

    tl.debug_barrier()

    # 3. 读取并原子归约到 global memory
    local_vals = tl.load(hist_ptrs)
    global_prefix = tl.atomic_add(global_cumsum + tl.arange(0, 256),
                                   local_vals, sem="acq_rel", scope="gpu")

    # 4. 所有 block 同步，等归约完成
    tled.distributed_barrier(mesh)

    # 5. 确定中位数落在哪个 bin（只需要 rank 0 做）
    # ...
```

### 模式 C：Cluster + DSMEM 远程访问

SM90+（Hopper），8 个 block 在同一个 cluster 内：

```python
@triton.jit
def cluster_histogram(inp_ptr, out_ptr, N, mesh: tl.constexpr,
                       CLUSTER_SIZE: tl.constexpr):
    cluster_rank = tled.shard_id(mesh, "cluster_x")
    is_rank0 = cluster_rank == 0

    local_hist = tle.alloc([256], dtype=tl.int32, scope=tle.smem,
                           nv_mma_shared_layout=False)

    # rank 0 的 shared memory 作为全局累加器
    rank0_cumsum = tle.alloc([256], dtype=tl.int32, scope=tle.smem,
                              nv_mma_shared_layout=False)
    if is_rank0:
        tl.store(tle.local_ptr(rank0_cumsum, (tl.arange(0, 256),)), 0)

    tled.distributed_barrier(mesh)

    # 各 rank 原子加到 rank0 的共享内存（DSMEM 直通）
    rank0_remote = tled.remote(rank0_cumsum, 0, scope=mesh)
    rank0_ptrs = tle.local_ptr(rank0_remote, (keys,))
    tl.atomic_add(rank0_ptrs, 1, sem="relaxed", scope="cta")

    tled.distributed_barrier(mesh)

    # 非 rank0 从 rank0 读取全局结果
    if not is_rank0:
        remote_vals = tl.load(tle.local_ptr(
            tled.remote(rank0_cumsum, 0, scope=mesh),
            (tl.arange(0, 256),)))
    # ...
```

---

## API 速查表

| API | 层 | 功能 | 硬件要求 |
|-----|---|---|---------|
| tle.alloc(shape, dtype, scope) | L1 | 分配 shared memory buffer | 所有 GPU |
| tle.local_ptr(buf, indices) | L1 | 获取 buffer 指针（支持动态索引） | 所有 GPU |
| tle.copy(src, dst, shape) | L1 | GM<->Local 批量拷贝（含 TMA） | 所有 GPU / TMA 需 Hopper+ |
| tle.smem | L1 | shared memory scope 常量 | 所有 GPU |
| tle.tmem | L1 | tensor memory scope 常量 | Hopper+ |
| tled.device_mesh(topology) | L2 | 声明设备拓扑 | 所有 GPU |
| tled.distributed_barrier(mesh) | L2 | 跨 block 同步 | Grid: cooperative launch / Cluster: SM90+ |
| tled.shard_id(mesh, axis) | L2 | 获取当前 block 坐标 | 所有 GPU |
| tled.remote(buf, id, scope) | L2 | DSMEM 远程访问 | SM90+ |
| tled.sharding(mesh, split) | L2 | SPMD 切分标注 | 标注阶段（M4 待实现） |
| raw.cuda.jit | L3 | JIT CUDA kernel | 所有 CUDA GPU |
| raw.mlir.jit | L3 | MLIR 代码生成 | 所有 GPU |

---

## 环境要求

| 功能 | 最低 SM 版本 | 说明 |
|------|------------|------|
| tle.alloc + tle.local_ptr | SM70 (Volta) | 基础 shared memory 操作 |
| tle.copy (普通) | SM70 | GM<->Local 批量拷贝 |
| tle.copy (TMA) | SM90 (Hopper) | Tensor Memory Accelerator |
| tled.distributed_barrier (grid) | SM70 | 需要 cooperative launch |
| tled.distributed_barrier (cluster) | SM90 (Hopper) | cluster 内同步 |
| tled.remote (DSMEM) | SM90 (Hopper) | Distributed Shared Memory |
| raw.cuda.jit | SM70 | CUDA JIT 编译 |
"""
