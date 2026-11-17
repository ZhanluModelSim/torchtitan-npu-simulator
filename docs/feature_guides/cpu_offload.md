# CPU Offload - 权重与梯度 CPU 常驻，优化器状态可选 CPU/NPU 驻留面

## 背景与挑战

在大规模分布式训练中，模型参数、梯度和优化器状态全部驻留 NPU HBM，显存占用为三者的总和。对于 MoE 模型（如 DeepSeek-V4），仅参数和梯度就可达数百 GB，远超单卡 HBM 容量。

Virtual Optimizer（见 [virtual_optimizer.md](virtual_optimizer.md)）已将优化器 moments 移到 Host 内存，但参数和梯度仍在 NPU 上。当参数+梯度的规模本身成为瓶颈时，需要更彻底的方案：**将三者全部常驻 CPU，仅在计算时按需搬到 NPU**。

## 特性概述

CPU Offload 是一套"CPU 常驻 + NPU 计算"的显存优化方案：

- **参数（权重）**：CPU pinned 存储，forward 前异步预取到 NPU
- **梯度**：backward 后 reduce-scatter 结果直接落 CPU（pinned），第二个 microbatch 起在 NPU 上做分块累加
- **优化器状态**：两种驻留面——默认 **NPU 常驻、step 原地更新**（无逐 step 往返）；
  列出 `swap_optimizer` override 时退回 CPU 常驻 + 每 step 在 NPU 上执行更新计算

**核心思想**：PyTorch FSDP2 的 `CPUOffloadPolicy` 负责参数/梯度的 CPU offload 基础设施；本仓在其上补齐 NPU 侧的性能关键路径（异步预取、NPU 梯度累加、NPU 侧 clip、CPU 常驻优化器），使全 offload 训练达到可用的吞吐。

## 启用方式

单一开关，通过命令行：

```bash
# 命令行
--training.enable-cpu-offload
```

优化器状态的驻留面由是否列出 `torchtitan_npu.override.common.optimizer.swap_optimizer`
的 override 决定（数据面始终 offload）：

| 配置 | 参数/梯度 | 优化器状态 |
|------|------|------|
| 仅 `--training.enable-cpu-offload`（无需列 override） | CPU 常驻 | **NPU 常驻**，step 原地更新，无逐 step H2D/D2H 往返 |
| 再列出 `swap_optimizer` override | CPU 常驻 | CPU 常驻（历史行为），每 step H2D → NPU 计算 → D2H |

该开关同时驱动两条路径：

| 路径 | 机制 |
|------|------|
| FSDP 数据面 | `parallelize_fn` 读取 flag → `CPUOffloadPolicy()` → FSDPParam 的 `offload_to_cpu=True` |
| 优化器容器 | `TrainerEx` 默认派生 `CpuOffloadNpuStateOptimizersContainer`；列出 `swap_optimizer` 时 override 再派生 `CpuOffloadOptimizersContainer`（两者都安装 FSDP/grad-clip 补丁） |

未开启时，相关补丁模块**不被 import**，训练行为与上游完全一致。

## 架构

```
--training.enable-cpu-offload
├── parallelize_fn → CPUOffloadPolicy → FSDPParam.offload_to_cpu = True
│   ├── forward: unshard 时 CPU pinned 参数 → NPU（多 rank: 上游内建 / 单 rank: 本仓预取补丁）
│   └── backward: foreach_reduce → 梯度 reduce-scatter → D2H 落 CPU pinned
│                                    → 第二个 microbatch 起在 NPU 分块累加（本仓补丁）
│
├── TrainerEx 默认派生 → CpuOffloadNpuStateOptimizersContainer（列出 swap_optimizer 时
│   override 再派生 → CpuOffloadOptimizersContainer）
│   ├── 安装 grad_accum.install() + grad_clip.install()（显式、幂等）
│   ├── clip: NPU 侧范数计算 + 系数发布 → 优化器直接消费已缓存的 NPU 梯度
│   └── optimizer: CpuOffloadAdamW / CpuOffloadDistributedMuon
│       ├── 状态 CPU 常驻: 每 step CPU 参数/状态 H2D → NPU ping-pong buffer → 计算 → D2H 回 CPU
│       └── 状态 NPU 常驻: 仅 CPU 参数 H2D，状态在 NPU 原地更新，D2H 仅回写参数
│
└── TrainerEx → cpu_dtensor_init.install()
    └── init_weights 时 CPU DTensor 参数在 NPU 上初始化后拷回
```

## 组件清单

| 模块 | 职责 |
|------|------|
| `extensions/cpu_offload/staging.py` | `CpuStaging`：有界 pinned 传输环（owner/wait/close 协议） |
| `extensions/cpu_offload/runtime.py` | `GradientClipChannel`：clip→optimizer 显式交接 + `stage_gradient` |
| `extensions/cpu_offload/cpu_offload_adamw.py` | `CpuOffloadAdamW`：公开算子实现，状态驻留面可选（CPU 常驻 / NPU 常驻原地更新）+ NPU 计算 |
| `extensions/cpu_offload/cpu_offload_muon.py` | `CpuOffloadDistributedMuon`：FlexShard 双路径（local + redistribution），momentum 驻留面同上 |
| `extensions/distributed/grad_accum.py` | 梯度累加 + 单 rank 参数预取（FSDP 集成） |
| `extensions/distributed/grad_clip.py` | clip 补丁：NPU 侧范数计算（cached / bounded 双模式） |
| `patches/torch_npu/cpu_dtensor_init.py` | `Module._init_param` 包装：CPU DTensor 参数 NPU 初始化 |
| `override/common/optimizer.py` | `CpuOffload*OptimizersContainer`：唯一 owner，显式安装/关闭；NPU-state 默认与 CPU-state（含 HostSparse）变体按 `swap_optimizer` 选择 |

## 支持组合

| 并行模式 | 支持 | 说明 |
|----------|------|------|
| DP shard (FSDP) | ✅ | 主路径 |
| HSDP (shard + replicate) | ✅ | 单 rank AG 组走预取补丁 |
| EP (expert parallel) | ✅ | clip 支持跨 EP mesh 的范数归约 |
| CP (context parallel) | ✅ | CP 维度参与范数归约 |
| PP | ✅ | pp_mesh 参与 clip 总范数 all_reduce |
| TP | ❌ | 未测试 |

## 显存与带宽权衡

**cached clip 模式**（默认，有 consumer 时）：clip 时全量梯度 H2D 一次，NPU 上算范数后**驻留**，优化器直接消费——每个梯度每 step 只 H2D 一次，代价是 NPU 峰值显存 `O(全部梯度)`。

关键性质：offload 后 NPU 占用（梯度 + buffer）**严格低于**非 offload 基线（梯度 + 参数 + 优化器状态），因此不存在"比不 offload 更早 OOM"的场景。

**bounded fallback**（无 consumer 时自动退化）：64MB 分块算范数，NPU 峰值 `O(64MB)` 常数，代价是梯度 H2D 两次（clip 一次 + 优化器一次）。实测性能下降明显（全量梯度二次 H2D 为每步秒级），故未暴露为用户选项；如未来超大模型确有需求，可低成本启用。

Chunk 大小（64 MiB）为内部调优常量，非用户配置。

## 与其他特性的关系

| 特性 | 关系 |
|------|------|
| OptimizerStateSwapContainer | 在优化器部分会swap权重和梯度 |
| CpuOffloadOptimizersContainer | 在优化器部分仅swap权重 |
| FullAC (全激活检查点) | 正交：AC 管激活，offload 管参数/梯度/状态；可组合使用 |

## 限制

- 优化器固定支持 `CpuOffloadAdamW` 和 `CpuOffloadDistributedMuon`（Muon）
- 不支持 `amsgrad`、`fused`、`capturable`、`differentiable`、`foreach`、复数张量
- 梯度 clip 仅支持 L2 和 inf 范数
- DistMuon 需要 DTensor 参数 + FlexShard compute layout 配置
