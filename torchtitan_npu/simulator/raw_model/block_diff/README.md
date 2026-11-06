# Block Diffusion 规范参考

**模型契约版本：3**（`model.py` 中的 `MODEL_CONTRACT_VERSION`）。本目录只包含
本规范与独立 PyTorch 参考实现 [`model.py`](model.py)，是本仓库 Block Diffusion
训练和推理语义的标准。网络配置决定规模，默认小配置不代表固定 checkpoint。

本版本包含标准 block diffusion 子契约 v1 和通用 LOOP 子契约 v1。
`ModelConfig` 定义共享 backbone；`BlockDiffusionModel` 提供标准训练／推理，
`LoopedModel` 提供与论文无关的循环计算机制。具体数据布局、Gate、停止／提交
策略和训练目标由外层配方组合，不再按论文向本文件追加模型实现。
Native 当前只对齐已验证的标准子契约，不自动获得 LOOP 支持。
本公开仓的 native 路径是训练适配。推理平台可以独立接收本文件作为模型定义，
其框架注册、调度、kernel 映射与验收属于接收方的适配范围。

## 结构与张量

记 batch 为 `B`，当前序列为 `S`，前缀为 `P`，块长为 `C`，hidden 为 `H`，
query/KV heads 为 `Nq/Nkv`，head dimension 为 `D`，词表为 `V`。

- 输入 token 是 int32/int64 `[B,S]`；embedding 输出 `[B,S,H]`。
- 每层执行 pre-norm GQA、残差、pre-norm FFN、残差；最后是 RMSNorm 和输出头。
- Q/K/V 形状为 `[B,Nq,S,D]`／`[B,Nkv,S,D]`／`[B,Nkv,S,D]`。
  `Nq` 必须可被 `Nkv` 整除，`D` 为正偶数；`H` 与 `Nq*D` 独立配置。
  attention 与 FFN 的线性层无 bias。
- `ModelConfig.qk_norm=True` 可在 RoPE 前对每个 Q/K head 的最后一维做 RMSNorm。
  每层 Q/K 各有 D 个共享于 heads 的权重；默认 False 保持旧参数格式与行为。
  这是 backbone 选项，不属于循环或某篇论文的专属配置。
  已验证的标准 v1／native 对照采用默认 `qk_norm=False`；True 不自动属于该支持范围。
- dense FFN 为 `down(silu(gate(x))*up(x))`。MoE 为逐 token top-k routed
  SwiGLU experts，加 `num_shared_experts` 个可选 shared expert 的输出之和。
  shared expert 数可以为零；`num_experts=0` 时为 dense，不启用 shared experts。
- router 即使在外层 autocast 下也在 FP32 计算 logits，选择 top-k 后对所选 logits 作 softmax。
  `RoutingPlan` 可显式回放每个 token 的不同专家 ID；每个 pair 计算一次，
  按专家分组执行，不为每个 token 复制整份专家权重。
- `tie_word_embeddings=True` 必须实际共享 embedding 与输出头的 Parameter。
- RoPE 按 split-half 配对 `(i, i+D/2)`，`0<=i<D/2`；逆频率为
  `rope_theta**(-2*i/D)`，不作额外频率缩放，默认 `rope_theta=1000000`。
  旋转在 FP32 中计算后转回激活 dtype；训练位置为 `0..S-1`，推理为
  `P..P+S-1`。不同配对方式、频率缩放或位置起点不视为数值等价。

## 训练契约

`forward_training(input_ids, labels, spec=TrainingSpec(...), routing_plan=...)`
保持梯度，使用完整窗口，不复用或 detach 推理 KV。labels 为同设备的 int64
`[B,S]`，与输入位置对齐，**不做 next-token shift**。调用方负责 corruption、
mask 比例和未监督位置的 `ignore_index`，模型不隐藏时间步或 loss weighting。

要求 `S>=C` 且 `S%C=0`。布尔 attention mask 中 True 表示可见：

| `attention_policy` | query `i` 可见 key `j` 的条件 |
| --- | --- |
| `block_causal` | `floor(j/C) <= floor(i/C)`：先前块与当前整个块可见 |
| `prefix_causal_canvas_bidirectional` | `j<=i or i>=S-C`：clean prefix 逐 token causal，最后 canvas 可见整个窗口 |

输出 `TrainingOutput(logits, loss, loss_sum, valid_tokens)`；logits 为 `[B,S,V]`，
输出头覆盖全部位置，包括 ignored positions。`loss_sum` 是 FP32 masked
cross-entropy 之和，`valid_tokens` 是非 ignore 标签数，
`loss=loss_sum/max(valid_tokens,1)`；零有效标签时返回可微零值。分布式全局
有效 token 归一化由训练驱动完成，不能直接平均不同 rank 的局部 mean loss。

## 推理契约

`forward(input_ids, cache=None, phase=..., attention_policy=..., logits_positions=...)`
为无梯度的单次前向；`model.train()` 不会将它变成训练入口。

- `KVCache` 每层保存独立 K/V，各为 `[B,Nkv,P,D]`。当前块 RoPE 位置从 `P`
  开始，所有层的 cache 长度一致。
- `prefill`／`commit` 返回追加当前 `S` 个位置后的新 cache，不修改输入 cache；
  `denoise` 读取前缀并重算当前块，保持 cache 不变。
- 推理 `causal` 表示前缀全部可见、当前块逐 token causal；`bidirectional`
  表示前缀与当前整个块可见。commit 采用哪种策略必须由调用方明确选择。
- 输出 `PhaseOutput(logits, cache)`。`logits_positions=None` 输出全部位置；
  空选择返回 `logits=None` 并跳过 final norm／head；位置序列共享于 batch，
  int64 `[B,K]` tensor 则按请求选择，输出 `[B,K,V]`。选择发生在 final norm 前。
- token 更新、停止、去噪步数和额外 commit 由外部调度，模型不假定统一的 `D+1`。

## 通用 LOOP 子契约 v1

`LoopSpec(loop_start, loop_end, max_depth, normalize_carry=False)` 只定义层区间、
循环上限和可选 carry normalization，不包含 mask token、Gate、loss 或提交规则。
区间 `[loop_start,loop_end)` 为非空共享 Core；之前是 Prelude，之后是 Coda。
同一个物理 backbone 被重复调用，循环次数增加不会复制或新增网络参数。
支持 dense 和 MoE；后者的路由可在每步通过 `RoutingPlan` 明确回放。

### 状态与执行接口

| 接口 | 语义 |
| --- | --- |
| `begin(input_ids, position_ids, attention_mask)` | embedding 加一次 Prelude，返回 depth=0 的 LoopState |
| `step(state, ...)` | 执行一次共享 Core，返回新状态及可选读出 |
| `forward(..., steps, readout)` | 无反馈的固定深度便利调用；readout 为 last（默认）、each_step 或 none |

`LoopState` 保存 pre-coda carry `[B,S,H]`、原始逻辑位置 `[B,S]`、布尔
attention mask `[S,S]`、已执行深度和可复用的 RoPE 表。RoPE 表不是 KV cache。
状态的几何信息在同一次循环内固定，调用方不得修改其 tensor 内容；更换
序列或位置应重新 begin。新状态保留梯度，step 不原地修改输入状态。

每次 step 的顺序是：

1. 若已有 Core pass 且配置启用，先 normalize 前一轮的 raw carry。
2. 若调用方提供 `feedback[B,S,H]` 和 `feedback_mask[B,S]`，用 where 注入。
3. 执行共享 Core；其输出成为下一轮的 raw carry。
4. 按 readout 请求执行 Coda、final norm 和可选 LMHead；读出不改变 carry。

`feedback` 是通用 hidden state，既可以是 token embedding，也可以是其他
可微变换；其选择、累计方式和 detach 决策属于调用方。Core 不认识 gold label、
已提交 token、confidence 或 halting probability。
`feedback_mask` 只替换值，不裁剪 Core 的计算行数；所有位置仍参与每次稠密前向。

`readout=False` 跳过 Coda、final norm 和输出头。
`readout=True, project_logits=False` 只返回 normalized features，允许外层接入
任意 head。Loop 本身不创建 Gate；普通固定深度循环无需任何策略模型或专用 loss。
训练／推理的 grad mode 由调用方控制，通用接口保持 autograd 可用。

### 负载语义

设执行 s 次 Core、请求 r 次 readout，则 Transformer 层调用数为：

```text
prelude_layers + s * core_layers + r * coda_layers
```

`layer_evaluations(steps, readout_count)` 返回这一计数，不包括 embedding、norm
或输出头，也不是 FLOPs/token 或时延。只在最后读出时 r=1，每轮读出时 r=s，
纯 latent 循环时 r=0。实际 mask、每步路由、head 和反馈决定其余工作量。

当前 Loop 是完整窗口、无持久 KV 的参考。Depth-aware cache、虚拟层缓存共享
和 token pruning 需要单独的后端策略；不在通用 Loop 中假定某种缓存数量公式。

### 具体配方

具体论文应描述如何组合 backbone、Loop 和外层策略。例如 ALoDLM 使用每轮
readout，再组合自己的 Gate、反馈规则、双流数据和训练目标；这些保存在
`scripts/block_diffusion/recipes/alodlm.py`，不是 raw 的独立模型范式。
其他循环方案可以复用同一 begin/step，无需在 raw 中增加论文名分支。

## 精度与边界

权重／激活 `dtype` 和 KV 存储 `kv_dtype` 支持 float32、float16、bfloat16；
KV 读写转换必须显式生效。router 参数／评分、RMSNorm reduction、RoPE 计算
使用 FP32；RMSNorm 参数使用模型 dtype。整数／量化配置不在当前契约范围内。

参考实现为单 rank 逻辑计算；TP/EP、服务调度、采样和后端 kernel 映射属于外部
适配。参数容量、KV 状态、临时 workspace、实际 HBM 流量与时延不能互相替代。

## 维护与来源

公开训练侧 `models/block_diffusion` 必须**人工维护**以符合所声明的本规范子集，不能运行时导入 raw
模型或工具 loader。修改通用 raw 语义时先审查并递增对应契约版本，再人工修改 native
实现，执行同配置、同权重的 logits／loss／梯度和适用状态的对照及后端检查；只有
对应证据通过后才能声明 native 对齐。纯格式或文件迁移不改变契约版本。
向推理平台交付的是同一份独立 `model.py`；公开训练侧的通过结果不代表
接收方推理框架已完成接入。

历史来源是 `ZhanluModelSim/torchtitan-npu-simulator` 的
[`feat/npu-simulator` raw 文件](https://github.com/ZhanluModelSim/torchtitan-npu-simulator/blob/2e988ab42d56c43c19c781eb8e9afc1f446693f4/torchtitan_npu/simulator/raw_model/block_diff/model.py)；
本地参考已修正其状态、数值与统计语义，不承诺公开 checkpoint 的训练等价。
遵循仓库 BSD 许可证。工具、迁移方式和执行命令见
[功能指南](../../../../docs/feature_guides/block_diffusion.md)，证据见
[参考验收记录](../../../../docs/test_guides/block_diffusion_raw_acceptance.md)。
