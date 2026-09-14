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
| block size | 256 |
| RoPE theta | 1000000 |
| tied embedding | false |

每层为 pre-norm GQA，加 top-k routed experts 和一个始终激活的 shared
SwiGLU expert。`num_experts=0` 时切换为 dense SwiGLU，用于单独验证 dense
路径。

一次训练 `forward` 表示对一个完整 canvas 的去噪网络求值，canvas 内使用
双向 attention。当前训练/meta 接口要求 `seq_len == block_size`。raw workload
中的多轮置信度提交、causal boundary pass 和跨 block KV cache 属于推理调度，
不在本次训练模拟器的声明范围内。

当前 recipe 使用仓库公共 cross-entropy loss，仅用于打通结构级 meta
前向、反向和内存/通信捕获。原始文件没有提供 mask corruption、目标构造或
训练 loss 的数值契约，因此本次接入不声明真实 Block Diffusion 训练语义或
loss 数值等价。

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

- `dense_debug`：两层 dense，小维度 CPU/meta 单测。
- `debug`：两层、8 experts、top-2，验证 MoE 和 converter 路径。
- `reduced`：八层、32 experts、top-4，用于核心并行组合。
- `full`：97 层、1024 experts、top-8，正式结构容量验证。

训练/模拟器配置名称为：

- `block_diffusion_dense_smoketest`
- `block_diffusion_smoketest`
- `block_diffusion_reduced`
- `block_diffusion_baseline`

## 声明支持矩阵

| 能力 | 状态 | 实现依据 |
| --- | --- | --- |
| 单卡 meta 前向/反向 | 支持 | 原生 Module/ModelSpec 和双向 SDPA |
| TP / sequence parallel | 支持 | TorchTitan sparse decoder 公共计划 |
| EP / ETP（分别启用） | 支持 | common MoE 的 ExpertParallel/ExpertTensorParallel |
| EP > 1 与 ETP > 1 同时启用 | 不支持、fail fast | 当前公共 ExpertParallel 不能处理二维 expert mesh |
| CP | 支持 | NPU Ulysses head/sequence all-to-all dispatcher |
| FSDP/eFSDP | 支持 | sparse decoder 公共 fully_shard 计划 |
| activation checkpoint | 支持 | 公共 sparse AC 计划 |
| PP | 支持 | `pipeline_llm`，需在最终组合单独验收 |
| NPU RMSNorm/RoPE/GMM/EP dispatch | 支持 | 仓库现有通用 converter |
| raw schema state-dict round trip | 支持 | `BlockDiffusionStateDictAdapter` |
| 真实 NPU 数值训练 | 未声明 | 不属于 meta 模拟器验收范围 |
| DLM 多轮生成和 KV cache | 不支持 | 训练模型 fail-fast 限定为单 canvas |
| 公开 HF checkpoint 数值兼容 | 不支持 | 原始实现未提供 checkpoint 契约 |

上述“支持”表示已接入对应公共实现，最终能否标记为 Ready 仍以验收证据矩阵
实际执行结果为准；任何未执行的组合不得仅根据代码路径宣称通过。

本次执行记录、环境、命令和最终结论见
[Block Diffusion 接入验收记录](../test_guides/block_diffusion_acceptance.md)。
