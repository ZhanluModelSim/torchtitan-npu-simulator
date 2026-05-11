# Muon 优化器特性

在大规模语言模型的训练中，优化器的选择对收敛速度和最终性能有着重要影响。传统的 Adam/AdamW 优化器虽然通用性强，但将参数视为独立的一维向量进行更新，忽略了矩阵参数的结构信息。Muon 优化器通过引入动量正交化（Momentum Orthogonalization）技术，针对 2D/3D 矩阵参数实现了更高效的梯度下降策略，在大模型训练中展现出更快的收敛速度。

## 实现原理

torchtitan-npu 采用了 **Muon + AdamW 混合优化器**策略，核心代码定义在 `torchtitan_npu/patches/optimizer/muon_optimizer.py` 和 `torchtitan_npu/patches/optimizer/muon_distributed.py`。

### 参数分配策略

模型参数根据维度和语义自动路由到不同的优化器：

- **2D 参数**（如 Linear 层的权重矩阵）→ 使用 Muon 优化器
  - 例外：名称中包含 `embed`、`lm_head`、`output` 的 2D 参数 → 使用 AdamW（此类层不适合正交化更新）
- **3D 参数**（如 MoE 专家权重）→ 使用 Muon 优化器
- **1D 及其他参数**（如偏置、LayerNorm 参数）→ 使用 AdamW 优化器

两种优化器被统一封装在 `MuonHybridOptimizersContainer` 中，对外提供与标准优化器一致的 `step()`、`zero_grad()`、`state_dict()` / `load_state_dict()` 接口。

### Newton-Schulz 正交化

Muon 的核心操作是对动量梯度进行正交化（LMO，Low-orthogonal Matrix Operation），通过 Newton-Schulz 迭代算法实现矩阵的零次幂近似（即投影到正交矩阵流形上）：

$$X_{k+1} = a \cdot X_k + b \cdot X_k X_k^T X_k + c \cdot (X_k X_k^T)^2 X_k$$

其中主系数为 $(a, b, c) = (3.4445, -4.7750, 2.0315)$，在零点处具有最大斜率，确保快速收敛。

#### 混合 Newton-Schulz（hybrid_ns）

当启用 `muon_hybrid_ns = true` 时，采用 DeepSeek-V4 提出的混合迭代策略：

- **前 8 步**：使用主系数 $(3.4445, -4.7750, 2.0315)$
- **后 2 步**：切换到次系数 $(2.0, -1.5, 0.5)$

此策略在保持收敛速度的同时，提升了正交化结果的数值稳定性。

### 学习率调整模式

Muon 优化器支持两种学习率调整模式（通过 `muon_adjust_lr_fn` 配置），其核心区别在于如何根据矩阵形状调整学习率，以及是否需要独立的超参数调优。

| 模式 | 调整公式 | 说明 |
|------|----------|------|
| `original` | $\gamma \leftarrow \gamma \cdot \sqrt{\max(1, A/B)}$ | Keller Jordan 原始实现，根据矩阵宽高比调整 |
| `match_rms_adamw` | $\gamma \leftarrow 0.18 \cdot \gamma \cdot \sqrt{\max(A, B)}$ | Moonshot 实现，直接复用 AdamW 的 lr 和 weight_decay |

#### original 模式

该模式源自 Muon 创始人 Keller Jordan 的原始实现。调整公式为：

$$\gamma_{\text{adjusted}} = \gamma \times \sqrt{\max\left(1, \frac{A}{B}\right)}$$

其中 $A$ 和 $B$ 是矩阵的两个维度。这个调整的目的是：**让正交化后的梯度更新在不同形状的矩形矩阵上具有一致的 RMS（Root Mean Square）**。

- 当 $A \le B$（宽矩阵，如 FFN 中的中间层）时，系数为 1，不做额外调整
- 当 $A > B$（高矩阵，如输出层）时，按 $\sqrt{A/B}$ 缩放

由于调整幅度较大，通常需要单独为 Muon 调优学习率（即配置 `muon_lr`），一般来说可以将 AdamW 的学习率放大 10 倍来作为 Muon 的学习率。

#### match_rms_adamw 模式

该模式来自 Moonshot 团队的论文 [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982)。调整公式为：

$$\gamma_{\text{adjusted}} = 0.18 \times \gamma \times \sqrt{\max(A, B)}$$

> 注：当前实现使用系数 0.18（与 DeepSeek-V4 一致），Moonshot 原始论文使用 0.2。

这个模式的设计目标是：**让 Muon 可以直接复用已经为 AdamW 调优好的学习率和权重衰减超参数**，无需额外的超参数搜索。

#### 模式选择建议

- **使用 `match_rms_adamw`**（默认）：如果你已经为 AdamW 调优好了超参数，希望直接尝试 Muon 而不想重新调参
- **使用 `original`**：如果你愿意投入时间单独调优 Muon 的学习率，追求可能的更好收敛效果

### 分布式通信机制

`DistributedMuon` 优化器针对不同的并行策略实现了三种参数更新路径：

| 并行策略 | 通信方式 | 适用场景 |
|----------|----------|----------|
| FSDP | all_to_all 双向通信 | FSDP 分片的 2D 参数 |
| DDP | all_gather 通信 | DDP 复制的 2D 参数 |
| Expert/MoE | 本地 LMO（无跨卡通信） | 3D 专家权重（已通过 EP 切分） |

**FSDP 路径**（`step_fsdp`）：采用 owner-per-bucket 方案，每个 rank 负责一组参数的完整 LMO 计算。通过两轮 all_to_all 通信：第一轮将各 rank 的梯度分片汇聚到参数 owner，owner 完成正交化后，第二轮将更新分片发回各 rank。

**DDP 路径**（`step_ddp`）：每个 rank 先对本地负责的参数计算 LMO 更新，然后通过 all_gather 将更新广播给所有 rank。支持与 TP 的联合使用，在 LMO 计算前先通过 all_gather 收集 TP 分片。

**Expert 路径**（`step_experts`）：专家权重已通过 Expert Parallel 切分到各 rank，每个 rank 独立对本地专家参数执行 LMO，无需跨卡通信。

### Virtual Allocator（可选）

对于显存受限的场景，Muon 优化器支持通过 `virtual_allocator = true` 启用优化器状态 CPU 卸载。

- **初始化阶段**：Muon 的 `momentum_buffer` 和 AdamW 的 `exp_avg`、`exp_avg_sq` 均分配在 CPU 的 pinned memory 上
- **step 阶段**：更新前将状态从 CPU 异步拷贝到 Device，更新完成后再异步拷贝回 CPU，释放 Device 显存
- **Checkpoint 阶段**：保存/加载前自动将状态换入 Device，完成后换回 CPU

## 配置选项

在训练任务的 TOML 配置文件（例如 `torchtitan_npu/models/deepseek_v4/train_configs/deepseek_v4_285b_4layers_debug_muon.toml`，或实际启动训练时 `--job.config_file` 所指向的路径）中，找到对应的 `[optimizer]` 节，并添加以下配置以启用 Muon 优化器：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `name` | str | "AdamW" | 优化器类型，设置为 `"Muon"` 启用本特性 |
| `lr` | float | — | 基础学习率，AdamW 部分直接使用；Muon 部分取决于 `muon_adjust_lr_fn` |
| `muon_lr` | float | None | Muon 专用学习率。仅当 `muon_adjust_lr_fn = "original"` 时生效；若不设置则回退到 `lr`；当 `muon_adjust_lr_fn = "match_rms_adamw"` 时此值被忽略 |
| `muon_momentum` | float | 0.95 | Muon 的动量因子 |
| `muon_enable_nesterov` | bool | True | 是否启用 Nesterov 动量 |
| `muon_ns_steps` | int | 5 | Newton-Schulz 正交化迭代步数，影响正交化精度与计算开销。值越大正交化越精确，但计算量也越大 |
| `muon_adjust_lr_fn` | str | "match_rms_adamw" | 学习率调整模式：`"original"` 或 `"match_rms_adamw"` |
| `muon_hybrid_ns` | bool | False | 是否启用混合 Newton-Schulz 迭代（前 8 步用主系数，后续步用次系数） |
| `virtual_allocator` | bool | False | 是否启用 Virtual Allocator，将优化器状态卸载到 CPU 以节省显存 |

### 配置示例

#### 示例 1：match_rms_adamw 模式

最简配置，直接复用 AdamW 超参，无需额外调参即可使用 Muon：

```toml
[job]
custom_config_module = "torchtitan_npu.config.custom_config"    # 使能本代码仓的自定义配置

[optimizer]
name = "Muon"                            # 使用 Muon 混合优化器
lr = 2.2e-4                              # 基础学习率（Muon 和 AdamW 共用）
weight_decay = 0.1
muon_momentum = 0.95                     # Muon 动量因子
muon_enable_nesterov = true              # 启用 Nesterov 动量
muon_ns_steps = 10                       # 正交化步数
muon_adjust_lr_fn = "match_rms_adamw"    # 复用 AdamW 超参（默认值，可省略）
muon_hybrid_ns = true                    # 启用混合 NS
```

#### 示例 2：original 模式

需要为 Muon 单独设置学习率，适合愿意投入调参资源的场景：

```toml
[job]
custom_config_module = "torchtitan_npu.config.custom_config"    # 使能本代码仓的自定义配置

[optimizer]
name = "Muon"                            # 使用 Muon 混合优化器
lr = 3e-4                                # AdamW 部分的学习率
muon_lr = 3e-3                           # Muon 专用学习率（通常为 AdamW lr 的 10 倍）
weight_decay = 0.01
muon_momentum = 0.95                     # Muon 动量因子
muon_enable_nesterov = true              # 启用 Nesterov 动量
muon_ns_steps = 5                        # 正交化步数
muon_adjust_lr_fn = "original"           # 使用独立的 lr 调度器
```

#### 示例 3：启用 Virtual Allocator

在显存受限的场景下，将优化器状态卸载到 CPU：

```toml
[job]
custom_config_module = "torchtitan_npu.config.custom_config"    # 使能本代码仓的自定义配置

[optimizer]
name = "Muon"                            # 使用 Muon 混合优化器
lr = 2.2e-4
weight_decay = 0.1
muon_momentum = 0.95
muon_enable_nesterov = true
muon_ns_steps = 10
muon_adjust_lr_fn = "match_rms_adamw"
muon_hybrid_ns = true
virtual_allocator = true                 # 启用 Virtual Allocator，优化器状态卸载到 CPU
```

## 注意事项

 **`swap_optimizer` 与 Muon 互斥**：Muon 优化器不支持与 Swap Optimizer 特性同时启用。如需节省显存，请使用 `virtual_allocator = true` 替代 `swap_optimizer = true`。

## 参考文献

- [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982)
- [Muon 优化器指南：快速上手与关键细节](https://kexue.fm/archives/11416)
- [Muon 优化器赏析：从向量到矩阵的本质跨越](https://www.spaces.ac.cn/archives/10592)
