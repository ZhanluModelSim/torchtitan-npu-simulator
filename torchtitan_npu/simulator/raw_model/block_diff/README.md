# Block Diffusion 规范参考

**模型契约版本：1**（`model.py` 中的 `MODEL_CONTRACT_VERSION`）。本目录只包含
本规范与独立 PyTorch 参考实现 [`model.py`](model.py)，是本仓库 Block Diffusion
训练和推理语义的标准。网络配置决定规模，默认小配置不代表固定 checkpoint。

## 结构与张量

记 batch 为 `B`，当前序列为 `S`，前缀为 `P`，块长为 `C`，hidden 为 `H`，
query/KV heads 为 `Nq/Nkv`，head dimension 为 `D`，词表为 `V`。

- 输入 token 是 int32/int64 `[B,S]`；embedding 输出 `[B,S,H]`。
- 每层执行 pre-norm GQA、残差、pre-norm FFN、残差；最后是 RMSNorm 和输出头。
- Q/K/V 形状为 `[B,Nq,S,D]`／`[B,Nkv,S,D]`／`[B,Nkv,S,D]`。
  `Nq` 必须可被 `Nkv` 整除，`D` 为正偶数；`H` 与 `Nq*D` 独立配置。
  attention 与 FFN 的线性层无 bias。
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

## 精度与边界

权重／激活 `dtype` 和 KV 存储 `kv_dtype` 支持 float32、float16、bfloat16；
KV 读写转换必须显式生效。router 参数／评分、RMSNorm reduction、RoPE 计算
使用 FP32；RMSNorm 参数使用模型 dtype。整数／量化配置不在版本 1 范围内。

参考实现为单 rank 逻辑计算；TP/EP、服务调度、采样和后端 kernel 映射属于外部
适配。参数容量、KV 状态、临时 workspace、实际 HBM 流量与时延不能互相替代。

## 维护与来源

`models/block_diffusion` 必须**人工维护**以符合本规范，不能运行时导入 raw
模型或工具 loader。修改 raw 语义时先审查并递增契约版本，再人工修改 native
实现，执行同配置、同权重的 logits／loss／梯度／KV 对照及后端检查；只有
对应证据通过后才能声明 native 对齐。纯格式或文件迁移不改变契约版本。

历史来源是 `ZhanluModelSim/torchtitan-npu-simulator` 的
[`feat/npu-simulator` raw 文件](https://github.com/ZhanluModelSim/torchtitan-npu-simulator/blob/2e988ab42d56c43c19c781eb8e9afc1f446693f4/torchtitan_npu/simulator/raw_model/block_diff/model.py)；
本地参考已修正其状态、数值与统计语义，不承诺公开 checkpoint 的训练等价。
遵循仓库 BSD 许可证。工具、迁移方式和执行命令见
[功能指南](../../../../docs/feature_guides/block_diffusion.md)，证据见
[参考验收记录](../../../../docs/test_guides/block_diffusion_raw_acceptance.md)。
