# Block Diffusion 模型契约与支持范围

本文记录 `block_diffusion` 原生 TorchTitan 模型的来源、结构和验收边界。
接入流程与最终判定分别遵循[新模型接入开发流程](new_model_onboarding.md)和
[新模型最终验收规范](../test_guides/model_acceptance.md)。

## 来源与基线

- 结构来源：`torchtitan_npu/simulator/raw_model/block_diff/model.py`。
- 接入基线：`feat/npu-simulator`，基线 commit `3eeed771d70b733f4f7eadd60ccd38bb77ac3aea`。
- 原始文件没有声明公开 checkpoint、tokenizer 或可核对的参考仓库，因此当前
  state-dict adapter 只承诺与该 raw workload 的参数 schema 双向闭环，不声明
  与某个 Hugging Face checkpoint 数值兼容。

## 模型契约

正式规格使用以下结构：

| 项目 | 值 |
| --- | --- |
| layers | 97 |
| hidden size | 8192 |
| query heads / KV heads | 64 / 8 |
| head dim | 128 |
| vocabulary | 262144 |
| routed experts / top-k | 1024 / 8 |
| routed expert intermediate | 4096 |
| shared experts | 1 |
| shared expert intermediate | 14336 |
| training sequence | 4096 |
| block size | 256 |
| RoPE theta | 1000000 |
| tied embedding | false |

每层为 pre-norm GQA，加 top-k routed experts 和一个始终激活的 shared
SwiGLU expert。`num_experts=0` 时切换为 dense SwiGLU，用于单独验证 dense
路径。

训练数据使用固定长度滑动窗口，stride 等于 `block_size`。每个窗口的最后一个
block 是当前 canvas，之前的 token 是已完成的 clean prefix。正式配置因此把
4096 tokens 解释为 3840-token prefix 加 256-token canvas；每向前滑动 256
tokens，初始 prefix 之后的下一个 block 成为训练目标。

每条样本均匀采样 `[0.01, 1.0]` 的 mask ratio，在 canvas 中精确选择相应数量的
位置并替换为 `mask_token_id`。label 与原 token 同位置对齐；prefix 和未被 mask
的位置写为 `IGNORE_INDEX`，所以公共 sum-reduction cross-entropy 只在 masked
positions 上计算，并按全局有效 label 数归一化。

本接入的目标是计算量建模，不复现 block attention mask。CPU/reference 路径把整个
`seq_len` 作为一条普通自回归序列，执行一次 causal SDPA；NPU converter 对应执行一次
`npu_fusion_attention`，进入算子前将 BSND 仅作 metadata reshape 为三维 BSH，使用
压缩 causal mask 和 `sparse_mode=2`，返回后恢复 BSND。这样 reduced 的 seq 256 不再
拆成 192/64 两段。Simulator 不直接导出该融合节点，因为 Zhanlu 的
`FlashAttentionPrediction` 当前无法从该节点建立有效输入 shape；它改为按每头批化的
`matmul(Q,K^T) -> softmax -> matmul(P,V)` 导出，反向对应 4 个 matmul 和 1 个
softmax backward，使下游使用已有 Matmul/Softmax cost model。
真实 NPU 调用显式传入 `head_num`、`input_layout=BSH`、`scale`、`keep_prob`、
`pre_tockens=INT_MAX`、`next_tockens=0`、`inner_precise=0`、`sparse_mode=2`、
`gen_mask_parallel=true` 和 `sync=false`。模拟算子另外记录 `head_dim`、KV head 数、
Q/KV sequence length，并同时保留 torch-npu 与成本模型常见的参数名别名，避免
下游把 `[B,S,H]` 错解为 `[B,H,S]`。这些建模参数通过 L0 OpNode 顶层的
`parameter_inputs` 专用通道导出；`attrs` 中保留同值仅用于仓库内兼容。

配置项 `attention_compute_alpha` 只折算 Attention 的 QK/Softmax/PV FLOPs，不改变
数值前向，也不折算 QKV/输出投影、MoE、Norm 或优化器计算。默认值按照原两段融合
kernel 的 score-matrix 面积与全长 kernel 面积之比设定：
`alpha=((S-B)^2+B*S)/S^2`。因此 debug/dense_debug 为 `0.75`，reduced 为
`0.8125`，full 为 `0.94140625`。该值是可覆盖的计算建模参数，不是模型权重或训练
超参数。Simulator 先按 causal AR 取平均有效 key 长度，再将 alpha 直接折入分解节点的
key 维度；因此即使外部 Zhanlu 不读取自定义属性，Matmul/Softmax 的输入 shape 也已经
体现折算。完整融合信息仍同时保存在每个子节点的 `attrs` 和 `parameter_inputs` 中。

上述 corruption 和 loss 是根据 raw workload 的逐 block 生成语义补齐的训练
契约。由于原始文件没有给出权威训练代码或 checkpoint，本接入不声明 mask-ratio
分布、loss weighting 或梯度与某个外部实现数值等价。raw workload 中的多轮
置信度提交、causal boundary pass 和跨 block KV cache 仍属于推理调度范围。

## 参数量公式

记 vocabulary 为 `V`、hidden size 为 `D`、层数为 `L`、query/KV head 数为
`H/K`、head dim 为 `A`、shared intermediate 为 `I`、expert 数为 `E`、
routed intermediate 为 `M`。

```text
embedding = V × D
output = V × D                         # 未 tied
final_norm = D
attention_per_layer = 2 × D × A × (H + K)
norm_per_layer = 2 × D
router_per_layer = E × D
routed_experts_per_layer = 3 × E × D × M
shared_expert_per_layer = 3 × D × I

total_moe = embedding + output + final_norm
          + L × (attention_per_layer + norm_per_layer
                 + router_per_layer + routed_experts_per_layer
                 + shared_expert_per_layer)

total_dense = embedding + output + final_norm
            + L × (attention_per_layer + norm_per_layer + 3 × D × I)
```

参数字节数使用参数实际 storage dtype 计算。RoPE cache 是非持久 buffer，不计入
参数量。

## 配置规格

- `dense_debug`：两层 dense，seq 16 / block 8，小维度 CPU/meta 单测。
- `debug`：两层、8 experts、top-2，seq 32 / block 16。
- `reduced`：八层、32 experts、top-4，seq 256 / block 64。
- `full`：97 层、1024 experts、top-8，seq 4096 / block 256。

训练/模拟器配置名称为：

- `block_diffusion_dense_smoketest`
- `block_diffusion_smoketest`
- `block_diffusion_reduced`
- `block_diffusion_baseline`

## 声明支持矩阵

| 能力 | 状态 | 实现依据 |
| --- | --- | --- |
| sliding-window corruption | 支持 | 一个 canvas stride，最后 block 精确随机 mask |
| masked-only same-position loss | 支持 | `IGNORE_INDEX` labels 和有效 token 归一化 |
| full causal attention compute proxy | 支持 | 单次全长 SDPA/NPU 融合 attention，Attention FLOPs 按 alpha 折算 |
| 精确 block attention mask 数值语义 | 不支持 | 计算量建模主动省略该 mask |
| 单卡 meta 前向/反向 | 支持 | 原生 Module/ModelSpec 和 full causal SDPA |
| TP / sequence parallel | 支持 | TorchTitan sparse decoder 公共计划 |
| EP / ETP（分别启用） | 支持 | common MoE 的 ExpertParallel/ExpertTensorParallel |
| EP > 1 与 ETP > 1 同时启用 | 不支持、fail fast | 当前公共 ExpertParallel 不能处理二维 expert mesh |
| CP | 支持 | NPU Ulysses head/sequence all-to-all dispatcher |
| FSDP/eFSDP | 支持 | sparse decoder 公共 fully_shard 计划 |
| activation checkpoint | 支持 | 公共 sparse AC 计划 |
| PP | 支持 | `pipeline_llm`，需在最终组合单独验收 |
| NPU Attention/RMSNorm/RoPE/GMM/EP dispatch | 支持 | 全长 causal attention 专用 converter 和仓库通用 converter |
| raw schema state-dict round trip | 支持 | `BlockDiffusionStateDictAdapter` |
| 真实 NPU 数值训练 | 未声明 | 不属于 meta 模拟器验收范围 |
| DLM 多轮生成和 KV cache | 不支持 | 本次实现训练范式，不含推理解码循环 |
| 公开 HF checkpoint 数值兼容 | 不支持 | 原始实现未提供 checkpoint 契约 |

上述“支持”表示已接入对应公共实现，最终能否标记为 Ready 仍以验收证据矩阵
实际执行结果为准；任何未执行的组合不得仅根据代码路径宣称通过。

本次执行记录、环境、命令和最终结论见
[Block Diffusion 接入验收记录](../test_guides/block_diffusion_acceptance.md)。
