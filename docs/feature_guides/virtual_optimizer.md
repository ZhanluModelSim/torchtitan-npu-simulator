# Virtual Optimizer - 优化器显存虚拟化特性

## 背景与挑战

在大规模分布式训练中，模型参数、梯度和优化器状态都会占用大量 NPU 显存。对于 Adam/AdamW，`exp_avg` 和 `exp_avg_sq` 两组 moments 通常与模型参数规模相当，并且在前向和反向计算期间仍会长期驻留在显存中。

随着模型规模、流水线并行 stage 数量或梯度累积步数增加，optimizer moments 可能成为显存瓶颈。与此同时，这些状态主要在 optimizer step 阶段使用，因此可以利用 Ascend NPU 的 swap memory 能力，将它们存储在 Host 内存中，同时保留可供 NPU 算子访问的地址，从而降低 optimizer state 对 NPU HBM 的长期占用。

## 特性概述

Virtual Optimizer 是一套基于 Ascend NPU 虚拟内存能力的优化器显存优化方案。它将 Adam/AdamW 的一阶和二阶 moments（`exp_avg`、`exp_avg_sq`）创建为 Host-backed、NPU 可访问的 tensor，在不修改训练循环的前提下减少 NPU 显存占用。

**核心思想**：使用 Ascend NPU 提供的 `torch_npu.empty_with_swapped_memory()` 申请实际存储位于 Host、但地址可由 NPU 访问的 tensor，并让 optimizer 直接使用这些 tensor 更新 moments。

<p align="center">
<img src="../assets/virtual_optimizer.png" width="80%" >
</p>

## 解决方案

### 核心实现原理

标准 Adam/AdamW 会在第一次 optimizer step 时按照参数创建 state：

```python
state = optimizer.state[p]
state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
state["exp_avg"] = torch.zeros_like(p)
state["exp_avg_sq"] = torch.zeros_like(p)
```

Virtual Optimizer 为每个底层 optimizer 注册 step pre-hook，在 Adam/AdamW 创建普通 moments 之前改用 swap memory：

```python
state = optimizer.state[p]
state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
state["exp_avg"] = torch_npu.empty_with_swapped_memory(
    p.size(), dtype=p.dtype, device=p.device
).zero_()
state["exp_avg_sq"] = torch_npu.empty_with_swapped_memory(
    p.size(), dtype=p.dtype, device=p.device
).zero_()
```

Hook 只处理已有 gradient 且 state 尚未初始化的参数，不会覆盖已经存在的 optimizer state。

对于 DTensor 参数，Virtual Optimizer 仅按照当前 rank 的 local shard 分配 moments，并保留原有 device mesh、placements、全局 shape 和 stride。部分 rank 可能持有 zero-sized local shard，而 swap allocator 不接受空 tensor，因此该情况使用普通的 `torch.empty_like()`。

## 启用方式

Virtual Optimizer 通过 TorchTitan 的配置级 override 启用，不增加 `virtual_optimizer_size`、`swap_optimizer_times` 等配置字段。

### 命令行示例

```bash
python -m torchtitan_npu.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --override.imports "torchtitan_npu.override.common.optimizer.virtual,torchtitan_npu.override.common.optimizer.checkpoint_virtual" \
  --checkpoint.enable \
  --checkpoint.async-mode disabled
```

如果 recipe 已经启用了 attention、RoPE、RMSNorm 等 override，应将以上两个入口追加到现有 `override.imports` 列表，不要覆盖模型原有入口。

也可以在 Python 配置中添加：

```python
cfg.override.imports.extend(
    [
        "torchtitan_npu.override.common.optimizer.virtual",
        "torchtitan_npu.override.common.optimizer.checkpoint_virtual",
    ]
)
cfg.checkpoint.enable = True
cfg.checkpoint.async_mode = "disabled"
```

两个入口分别替换不同的 TorchTitan 配置节点：

| Override | 配置节点 | 作用 |
| --- | --- | --- |
| `optimizer.virtual` | `OptimizersContainer.Config` | 创建 swap-backed `exp_avg` 和 `exp_avg_sq` |
| `optimizer.checkpoint_virtual` | `CheckpointManager.Config` | 兼容同步 native DCP 保存 live swap tensors |

只需要使用 Virtual Optimizer、但不保存 checkpoint 时，可以只启用 `optimizer.virtual`。需要同步保存并恢复完整训练状态时，应同时启用两个入口。

## Virtual Optimizer 与 Checkpoint

TorchTitan 在保存 optimizer checkpoint 前会初始化尚未创建的 optimizer state，因此即使保存发生在第一次真实 optimizer step 之前，也能先建立形状和分片正确的 swap-backed moments。

PyTorch DCP 默认 `FileSystemWriter` 的 copy-ahead 路径与当前 NPU swap storage 不兼容。`optimizer.checkpoint_virtual` 因此只对同步 native DCP 本地保存使用以下 writer：

```python
dcp.FileSystemWriter(
    checkpoint_id,
    per_thread_copy_ahead=0,
)
```

该 checkpoint 修改只用于解决 Virtual Optimizer state 的保存兼容问题，不是独立的 checkpoint 特性。DCP 加载不需要单独改写：fresh optimizer 会先创建 swap-backed load targets，再由 DCP 将 checkpoint 数据写入这些 targets；已有 state 和 repeated load 会复用当前 state。

## 适用范围与限制

- 当前支持由 TorchTitan `OptimizersContainer` 构建的 Adam 和 AdamW。
- 当前只接管 `exp_avg` 和 `exp_avg_sq`，`step` 仍为参数 device 上的普通 FP32 scalar。
- 不包含 AMSGrad 的 `max_exp_avg_sq`，也不是通用 optimizer-state offload 实现。
- 不提供 CPU cache、分块 Load/Update/Offload、多 stream 更新流水线或可配置 swap 容量。
- checkpoint writer 特化仅覆盖同步 native DCP 的本地文件系统保存。
- async、async-with-pinned-memory 和 Hugging Face 保存继续使用 TorchTitan 上游路径，不经过当前 writer 特化。
- 多卡组合、async 和 Hugging Face 路径尚未在本特性中完成完整验证。
- 本文不声明未经独立验证的最低驱动版本、性能提升或 HBM 节省比例。

## 常见问题

| 现象 | 检查项 |
| --- | --- |
| Virtual Optimizer 未生效 | 确认 `override.imports` 使用完整的 `module.function` 路径，并检查启动日志中的 override 应用记录 |
| 只启用 `checkpoint_virtual` 后没有 swap state | `checkpoint_virtual` 只处理保存兼容；还需启用 `optimizer.virtual` |
| 同步保存仍使用默认 writer | 确认使用 native DCP、`async_mode="disabled"`，且不是 Hugging Face 保存 |
| 模型原有 override 丢失 | 将两个入口追加到现有列表，不要用新列表覆盖原配置 |
| zero-sized shard 分配失败 | 检查当前 rank 的 local shard 是否进入普通 `empty_like` 分支 |

## 验证状态

CPU 单元测试覆盖 swap allocation、zero-sized fallback、lazy state 初始化、同步 checkpoint writer 以及 async/Hugging Face 上游委托。此前的 Ascend NPU 单卡验证覆盖同步 save/load、fresh/existing/repeated load 和 load 后继续执行 optimizer step；多卡、async 和 Hugging Face 路径不在当前完整验证范围内。

## 相关链接

- [Virtual Optimizer override](../../torchtitan_npu/override/common/optimizer.py)
- [Virtual Optimizer 单元测试](../../tests/unit_tests/override/common/test_optimizer.py)
- [Override 机制与入口](../../torchtitan_npu/override/README.md)
- [软件安装](../user-guides/installation.md)
- [快速上手](../user-guides/quickstart.md)
