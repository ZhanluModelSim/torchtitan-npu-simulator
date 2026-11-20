# DeepSeek-V4 MTP

> **0.4.0 配置迁移说明：** 下文保留 MTP 原理和按需扩展方式。训练工厂已迁至 `torchtitan_npu_recipes`；当前启动方式见[快速上手](../user-guides/quickstart.md)，专用配置按 [Python 配置组合指南](../developer_guides/python_recipes.md)定义。

本文介绍 TorchTitan-NPU 中 DeepSeek-V4 的 Multi-Token Prediction（MTP）训练特性，内容包括 MTP 原理、代码结构和使能方式。

## 1. MTP 原理

标准自回归训练采用 Next-Token Prediction（NTP）：位置 `t` 的 hidden state 只预测下一个 token。MTP 在保留主预测任务的基础上，增加若干串联的辅助预测深度，让模型同时学习更远位置的 token。

设输入 token 为 `x`，主干最终 hidden state 为 `h^(0)`。第 `k` 个 MTP 深度接收：

- 主干或上一个 MTP 深度产生的 `h^(k-1)`；
- 左移 `k` 位后的 token embedding；
- 当前深度对应的有效位置掩码。

如果 `label[t]` 是输入位置 `t` 的下一个 token，那么第 `k` 个 MTP 深度会利用第 `k` 个未来 token 的 embedding，预测第 `k + 1` 个未来 token。当前深度产生的 hidden state 会继续传给下一个深度，因此多个 MTP 深度构成一条逐层推进的辅助预测链，而不是若干相互独立的输出 head。

### 1.1 序列移动和有效位置

`roll_mtp_sequence` 负责生成左移后的 token，并同时返回 `valid_mask`：

- 屏蔽序列尾部不存在足够未来 token 的位置；
- packed sequence 根据 `positions` 屏蔽跨文档边界的位置；
- 无效位置不参与 MTP loss，也不会把无效的上一深度 hidden state 传入当前深度。

模型返回主预测和各 MTP 深度组成的列表。使用普通 `MTPLoss` 时列表元素是 logits；使用默认的分块损失时列表元素是 LM head 之前的 hidden state：

```text
outputs[0]     主 NTP 任务的 logits 或 hidden state
outputs[k]     第 k 个 MTP 深度的 logits 或 hidden state
```

`MTPLoss` 接收该列表以及 `positions`，在 loss 内部按相同规则移动 labels，因而不需要模型额外返回有效位置掩码。设 MTP 深度数为 `K`，总损失为：

```text
L = L_main + mtp_scale / K * sum(L_mtp,k)
```

主损失和各层 MTP 损失均采用 token-sum 交叉熵，最后统一按照全局有效 token 数进行归一化。`mtp_scale` 用于控制辅助预测任务对总梯度的贡献。

### 1.2 DeepSeek-V4 MTP

DeepSeek-V4 主干使用 mHC，多流 hidden state 的形状为 `[batch, sequence, hc_mult, hidden]`。MTP 过程中不能先把这些 stream 合并成普通三维 hidden，否则会丢失 mHC 状态。每个 DeepSeek-V4 MTP block 执行：

1. 分别投影未来 token embedding 和上一深度的多流 hidden state；
2. 将 token embedding 投影广播到每个 mHC stream；
3. 通过 `HcPre -> Attention -> HcPost` 更新 attention 分支；
4. 通过 `HcPre -> FFN/MoE -> HcPost` 更新 FFN 分支；
5. 使用当前深度独立的 `HcHead` 生成三维 prediction hidden；
6. 三维 prediction hidden 进入共享 LM head，多流 hidden state 则继续传给下一
   个 MTP 深度。

DeepSeek-V4 MTP 保留 mHC 多流状态，但其 attention 使用非压缩形式，不启用主干 DSA 的 compressor 和 indexer。

### 1.3 DeepSeek-V4 Context Parallel

MTP 的 future-token shift 不能在 CP rank-local 序列上执行：局部移动会在 rank 边界读到不属于本 rank 的 token。V4 的 CP 布局由切分施加（`--parallelism.context-parallel-load-balancer doc_chunk`，见[Context Parallel](context_parallel.md)），实现因此保持在 `cp_shard` 之前对完整序列统一生成：

- 每个 MTP 深度的 future token；
- packed sequence 感知的有效位置 mask；
- 每个 MTP 深度的监督 label。

这些张量在 `preprocess_inputs` 中先于基座切分、在全局行上构建；随后 `super().preprocess_inputs()` 完成 `inputs`/`labels`/`positions` 的切分，MTP 三个张量再用**同一把连续切片**单独 `cp_shard`（本地切片，无通信），保证 token、label 和 mask 与 `inputs` 在每个 rank 上严格同位。

该流程只接入 DeepSeek-V4：继续复用已有压缩 metadata 和 CP plan 构造；MTP 张量的切分是独立的一次 `cp_shard` 调用，与 `inputs`/`labels`/`positions` 的切分使用同一把连续切片。DeepSeek-V3 保留社区 MTP 的原有限制，`context_parallel_degree > 1` 时在配置更新阶段报错；DeepSeek-V3 已有的非 MTP CP 路径不变。

该处理与 Megatron-LM MTP 在 CP 边界交换 future token/label 的数学目标一致；区别是 TorchTitan 的模型 metadata hook 位于 CP 切分之前，因此可直接在全局序列上生成，再按和输入相同的连续切法单独切分，无需额外点对点通信。

### 1.4 Chunked loss

DeepSeek-V4 启用 MTP 时使用 `MTPChunkedLossWrapper`。它继承并逐预测分支调用 TorchTitan 原生 `ChunkedLossWrapper`，因此复用其 sequence chunk、LM head FSDP、梯度累加和 decoder backward bridge。主分支权重为 `1`，每个 MTP 分支权重为 `mtp_scale / K`。

分支权重在分块 loss 内部、内部 `backward()` 之前应用，而不是在最终标量上事后相乘。这样 LM head 和 decoder hidden gradient 都与未分块 MTP loss 保持相同缩放，同时避免构造完整 `[batch, sequence, vocab]` logits。

### 1.5 梯度隔离（按需配置）

在 RL 等场景中，可以让 MTP 继续训练，同时避免 MTP 辅助损失直接改变主模型的策略梯度。梯度隔离只影响反向传播边界，不改变前向输出、loss 公式或 checkpoint 参数结构。

当前默认 recipe 不显式暴露 `mtp_isolate_gradients` CLI 参数。需要该策略时，在自定义 recipe 中同时调整 MTP 模型和 loss 的构造：截断主干 hidden state 到 MTP 的梯度，并让 MTP 分支使用与共享 token embedding、LM head 隔离的权重梯度。模型和 loss 必须使用同一策略，不能只修改其中一侧。

```python
config.model.mtp_isolate_gradients = True
config.loss.mtp_isolate_gradients = True
```

该策略只隔离直接梯度，仍使用现有的全局梯度裁剪，因此主模型和 MTP 的裁剪系数仍然关联。默认训练不启用梯度隔离；用户可以在自己的模型扩展和 recipe 中按需实现或接入。

## 2. 代码结构

### 2.1 公共 MTP 模块

公共能力由社区 `torchtitan.models.deepseek_v3.mtp` 提供：

| 对象 | 职责 |
|---|---|
| `roll_mtp_sequence` | 移动 token，处理序列尾部和 packed 文档边界 |
| `MTPLoss` | 计算主 NTP loss 和加权 MTP loss |
| `MTPDecoder` | 管理 MTP layers、forward 输出和 FSDP 接入 |

TorchTitan-NPU 只补充社区当前尚未提供的组合能力：

| 对象 | 职责 |
|---|---|
| `prepare_mtp_batch` | 为 DeepSeek-V4 在 CP 之前生成各深度 token、mask 和 label |
| `MTPChunkedLossWrapper` | 为 DeepSeek-V4 将原生 chunked loss 扩展到主预测和各 MTP 预测 |

DeepSeek-V4 复用 `roll_mtp_sequence` 和社区多预测列表接口，只实现 mHC 多流状态、CP 和 chunk loss 所需的专用 decoder 与 block。只有 DeepSeek-V4 的专用 decoder 配置放开 MTP 与 CP 的组合；没有修改社区 `MTPDecoder` 或 `MTPLoss`。

### 2.2 DeepSeek-V4 目录

| 文件 | MTP 相关职责 |
|---|---|
| `deepseek_v4/__init__.py` | 根据模型规格构造非压缩、mHC-aware 的 MTP blocks |
| `deepseek_v4/model.py` | 保留主干多流 state，并组织主预测和 MTP 预测 |
| `deepseek_v4/mtp.py` | 实现 DeepSeek-V4 mHC MTP block、CP batch 和 chunk loss |
| `torchtitan_npu_recipes/tests/models/deepseek_v4.py` | 按 `num_mtp_layers` 构造 MTP 模型并选择对应 loss |
| `deepseek_v4/parallelize.py` | 让 MTP blocks 复用现有 DSV4 并行流程 |
| `deepseek_v4/sharding.py` | 增加 MTP fusion、mHC head 和有效掩码布局 |
| `deepseek_v4/state_dict_adapter.py` | 转换本地 `mtp_layers` 与 HF `mtp.<depth>` 参数 |

### 2.3 与原模型解耦

DeepSeek-V4 在 `config.mtp_layers` 为空列表时关闭 MTP，运行时 `model.mtp_layers` 为 `None`。关闭时 forward 返回普通 logits，MTP 并行和 sharding 增量逻辑不会执行。

MTP 开启后，参数归属于独立的 `model.mtp_layers`。社区 DeepSeek-V3 并行入口负责让这些 blocks 沿用主干相同的 TP/SP、EP、activation checkpoint、compile和 FSDP 处理顺序；DeepSeek-V4 的专用 decoder 与 sharding 遵循相同约定。

DeepSeek-V4 使用`list[hidden] + MTPChunkedLossWrapper`；wrapper 是标准 `ChunkedLossWrapper` 子类，因此 `Trainer` 原有的 LM head 注入和 `_skip_lm_head` 流程无需修改。

当前并行限制如下：

- DeepSeek-V3 MTP 暂不支持 Context Parallel（CP）和
Pipeline Parallel（PP），启用时必须保持 `context_parallel_degree=1` 和 `pipeline_parallel_degree=1`。
- DeepSeek-V4 MTP 支持 CP，但暂不支持 PP；启用时必须保持 `pipeline_parallel_degree=1`。

不支持的并行组合会在模型配置更新阶段直接报错。

## 3. 使能方式

### 3.1 默认 recipe

DeepSeek-V4 的 recipe 通过 `num_mtp_layers` 控制 MTP，当前默认值为 `0`。需要 MTP 时，在自定义 recipe 中同时设置正的 `num_mtp_layers` 和对应的 MTP loss，再通过 `MODULE` / `CONFIG` 启动。

模型和 loss 必须同时切换。只构造 MTP layers 而继续使用普通 loss，或者只配置 `MTPLoss` 而没有构造 MTP layers，都会造成模型输出与 loss 输入不匹配。

### 3.2 自定义 MTP 深度

需要调整预测深度时，应在 recipe 中同时设置模型和对应的 loss。例如：

```python
config = deepseek_v4_debugmodel(num_mtp_layers=2)
config.loss = MTPChunkedLossWrapper.Config(
    mtp_scale=0.3,
    num_chunks=8,
    loss_fn=CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model),
    ),
)
```

主要参数如下：

| 参数 | 含义 |
|---|---|
| `num_mtp_layers` | MTP 辅助预测深度数；大于 0 时构造 MTP blocks |
| `mtp_scale` | 所有 MTP 辅助损失的总权重 |
| `global_vocab_size` | 词表并行交叉熵使用的完整词表大小 |
| `num_chunks` | DeepSeek-V4 每个主/MTP hidden sequence 的 loss 分块数 |
| `mtp_layers` | 模型内部展开后的 MTP block 配置列表，通常不手工构造 |

DeepSeek-V4 的 MTP loss 默认使用 `mtp_scale=0.3`。

### 3.3 关闭 MTP

关闭 MTP 时需要让模型构造保持 `num_mtp_layers=0`，并使用普通 loss。此时 `config.mtp_layers` 为空列表，运行时 `model.mtp_layers is None`；模型不创建 MTP 参数，训练只计算主 NTP loss。
