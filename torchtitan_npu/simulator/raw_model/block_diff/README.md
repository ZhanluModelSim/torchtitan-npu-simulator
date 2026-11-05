# Block Diffusion 规范参考

**模型契约版本：2**（`model.py` 中的 `MODEL_CONTRACT_VERSION`）。本目录只包含
本规范与独立 PyTorch 参考实现 [`model.py`](model.py)，是本仓库 Block Diffusion
训练和推理语义的标准。网络配置决定规模，默认小配置不代表固定 checkpoint。

本版本包含两个明确分开的子契约：`STANDARD_BLOCK_DIFFUSION_CONTRACT_VERSION=1`
保持原有 `BlockDiffusionModel` 行为；`ALODLM_CONTRACT_VERSION=1` 新增下述
`ALoDLMModel`。Native 目前只对齐标准子契约 v1，不自动获得 ALoDLM 支持。
下方“结构、训练、推理”先定义标准子契约，ALoDLM 的差异在独立章节说明。

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

## ALoDLM 子契约 v1

依据 [ALoDLM 论文 v1](https://arxiv.org/html/2610.04198v1) 的循环架构及训练公式，
并核对作者发布代码
[`fe225a99`](https://github.com/amazon-science/ALoDLM/tree/fe225a99e0b5aeff351804e871d7581155d7f27d)。
本节为独立的稠密参考实现，不是作者代码移植或其 checkpoint 的数值兼容声明。

### 循环结构与状态

`ALoDLMModel(ModelConfig, ALoDLMSpec)` 当前只接受 dense backbone。Spec 用
`[loop_start, loop_end)` 指定共享 core；prelude 每次 outer 调用执行一次，
core 和 coda 在每个 inner pass 都执行。下一轮携带 **pre-coda** 隐状态，
coda 分支只用于本轮 readout；不将 coda 输出反馈成下一轮 carry。

- 主干物理参数不随循环次数复制。新增 ExitGate 为 `Linear(H,1,bias=True)`
  和 K 个可学习 depth bias；其输入 detach，不通过 gate 给主干传梯度。
- `qk_norm=True` 默认在每个 head 的 Q/K 上、RoPE 前增加 RMSNorm。
  `normalize_carry=True` 默认在轮间使用同一个 backbone final norm。
  这些是显式选项；标准 `BlockDiffusionModel` 不因此增加 QK norm 参数。
- 每次新 outer 调用重新从 embedding/prelude 初始化；carry 只在 inner
  passes 中保留。所有 physical rows 继续参与稠密主干和读出，不按退出深度
  自动剪掉 token 的后续计算。
- 输入必须给出原始逻辑 `position_ids[B,T]` 与显式布尔 mask `[T,T]`；不能
  把重排后的 physical row index 直接当成 RoPE 位置。

推理 `run_outer(...)` 是固定 schedule 回放：commit depth 与执行 `steps`
分别提供，Gate 停止概率不是 commit depth。`feedback_policy="cumulative"`
每轮回填累计提交位置；`"newly_committed"` 只在下一轮回填新提交位置，之后
保留其 latent 演化。这两个选项分别对应论文反馈公式和发布推理的不同细节，
不代表两个端到端引擎 profile。提供 `feedback_tokens` 可回放指定 token，
否则使用各轮 argmax。全部位置均执行 LMHead/ExitGate，包括最后一轮。

`alodlm_decision` 单独表达提交和停止：entropy 决定新提交集合，累积 hazard
更新为 `a+(1-a)*lambda`，剩余位置的平均值决定是否停止；到达末轮时有效
hazard 强制为 1。停止但本 outer 尚无提交时，至少提交剩余位置中 entropy
最低的一项。默认采用论文的 `<=` 条件；发布代码的严格 `<` 和位置惩罚必须
通过 `strict_entropy`、`position_penalty` 明确提供。

### 双流训练与目标

`build_alodlm_batch(clean_ids, labels, masked, mask_probability, ...)` 构造
`[B,2S]` 的 clean/noisy 双流，逐 noisy block 稳定排列为 observed-first、
masked-last，保留原始位置。每行可见同 stream、同 block 的 causal 前缀，
以及所有此前 clean blocks；noisy 行看不到同块 clean target。
每个 batch row 是独立序列；随机 corruption 和 packed 文档不在此 helper 中。
Batch 同时保存 `ignore_index`，目标函数继承这一口径，不能使用冲突的覆盖值。

`rollout_training(batch, exit_depths=...)` 始终执行完整 K 轮，输出所有深度的
logits/gates。训练采用累计 **gold embedding** 回填，必须在对应退出轮读出
之后才能注入标签。可给定退出轨迹回放，或采样条件 Bernoulli gate；有效
退出深度为 1..K，末轮强制退出。回放不等于从当前策略随机采样。

`alodlm_training_objective` 明确采用发布实现的 practical 数学口径：

- Denoiser 使用到采样退出轮为止的 detached 权重，并将剩余概率质量并入
  该退出轮；逆 mask probability 权重在本批监督 mask 内归一化。
- Actor 使用每条独立序列的 cost sum 与 decision-log-probability sum 的乘积，
  再除以监督 mask 总数；第一轮 CE 为 detached baseline。
- 深度 prior 为 `softmax(-c*depth)`；默认 individual/aggregate 系数为 0.1/1。
- AR loss 在 clean stream 上做 next-token shift，跨深度平均；最终为
  `(denoising + actor + AR)/2`。本参考将 CE/reduction 显式置于 FP32；发布代码
  的 denoiser CE 使用 backbone dtype，二者不声称低精度逐位一致。

这个 practical 目标不声称具有论文基础 joint-KL estimator 的完整无偏性。
零监督 mask 时 denoising/actor 为零，但只要 clean AR 标签有效，AR loss
仍然存在。所有 Tensor 变换支持固定形状 meta 回放和真实反向传播。

### 负载口径与首版边界

本版是 **uncached dense** 参考，拒绝普通和 depth-aware KV cache 参数。
不实现 streaming window、cache repair、packed 文档、activation checkpointing、
ragged batch 调度或优化 serving。完整 readout 历史被保留用于验收，不能据此
推断优化流式推理的峰值内存。
推理读出覆盖整个输入窗口，比发布引擎只对初始 masked rows 读出的工作更多；
不能将本参考的耗时或流量当成作者优化引擎结果。
本参考还先计算全部 2S 行的 denoising CE 再屏蔽，发布代码只计算监督 masked
rows；需要精确估计该发行实现的 loss 工作量时，应由后端显式做行选择。

`layer_evaluations(s)` 返回 `loop_start+s*(L-loop_start)` 层调用数；它不是
FLOPs/token。`logical_buffer_count()` 只给出潜在 depth-aware 布局的
`L+(K-1)*(loop_end-loop_start)` 个 KV 对，不分配或模拟这些 cache。
例如 L=36、core=[10,26)、K=4 时，两项分别为 114 次层调用和 84 个 KV 对。
论文明确将 deepest-available fallback 与 shared-coda cache 视为 serving
近似，未来适配不能把普通 KV cache 或 token 提交深度直接替代它们。

## 精度与边界

权重／激活 `dtype` 和 KV 存储 `kv_dtype` 支持 float32、float16、bfloat16；
KV 读写转换必须显式生效。router 参数／评分、RMSNorm reduction、RoPE 计算
使用 FP32；RMSNorm 参数使用模型 dtype。整数／量化配置不在当前契约范围内。

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
