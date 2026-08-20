# Kimi K3 模型参数 Web 字段目录

本文档定义 Web 页面和上游服务可配置的 Kimi K3 **模型参数**。训练、并行、
重计算、优化器、通信、检查点和 simulator 参数不在本文范围内。

Kimi K3 的内部 `KimiK3Model.Config` 保存展开后的逐层 `layers[]`。该结构是
派生结果，不适合作为公共接口。公共接口使用稳定的
`KimiK3ModelOverrides` dataclass 一共包含 32 个字段，其中 `param_init` 是隐藏的
内部字段。Web 实际可编辑和提交的是其余 31 个模型超参数；后端校验后统一重建
每一层的 KDA/MLA 与 dense/MoE 配置。

## 请求格式与 CLI 映射

Web 请求使用 snake_case：

```json
{
  "model_overrides": {
    "n_layers": 16,
    "dim": 4096,
    "kda_layers": [0, 1, 2, 4, 5, 6],
    "num_experts": 256,
    "router_top_k": 8
  }
}
```

CLI 保持同一层级，仅将下划线转换为连字符：

```text
model_overrides.n_layers -> --model-overrides.n-layers
model_overrides.router_top_k -> --model-overrides.router-top-k
```

布尔字段使用 `--model-overrides.<字段>` / `--model-overrides.no-<字段>`；
可空字段传 `None`；列表字段在一个参数名后传多个值：

```bash
python3 -m torchtitan_npu.entry \
  --module torchtitan_npu.models.kimi_k3 \
  --config kimi_k3_smoketest \
  --model-overrides.n-layers 3 \
  --model-overrides.kda-layers 0 1 \
  --model-overrides.num-experts 16 \
  --model-overrides.router-top-k 4
```

训练模块与 `torchtitan_npu.simulator` 使用完全相同的字段和校验规则。

## 核心规模

| Web 字段 | English label | 中文名称 | 类型 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.vocab_size` | Vocabulary Size | 词表大小 | `int` | 基础 | 大于 0；需匹配 tokenizer/checkpoint |
| `model_overrides.dim` | Hidden Dimension | 隐藏层维度 | `int` | 基础 | 大于 0；同步作用于所有层 |
| `model_overrides.n_layers` | Transformer Layers | 层数 | `int` | 基础 | 大于 0；决定实际消费的 `kda_layers` 范围 |
| `model_overrides.n_dense_layers` | Dense Prefix Layers | Dense 前缀层数 | `int` | 基础 | `0 <= value <= n_layers`；其余层为 MoE |
| `model_overrides.norm_eps` | RMSNorm Epsilon | 归一化 Epsilon | `float` | 高级 | 大于 0；同步作用于模型、attention 和 MoE |
| `model_overrides.attn_res_block_size` | Attention Residual Block Size | Attention 残差块大小 | `int \| null` | 高级 | `None` 关闭；否则大于 0 |

## Gated MLA

| Web 字段 | English label | 中文名称 | 类型 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.n_heads` | Attention Heads | 注意力头数 | `int` | 基础 | 大于 0；同时用于 KDA 和 MLA |
| `model_overrides.q_lora_rank` | Query LoRA Rank | Query LoRA 秩 | `int` | 基础 | 大于 0 |
| `model_overrides.kv_lora_rank` | KV LoRA Rank | KV LoRA 秩 | `int` | 基础 | 大于 0 |
| `model_overrides.qk_nope_head_dim` | QK NoPE Head Dimension | 非 RoPE QK 头维度 | `int` | 高级 | 大于 0 |
| `model_overrides.qk_rope_head_dim` | QK RoPE Head Dimension | RoPE QK 头维度 | `int` | 高级 | 大于 0 |
| `model_overrides.v_head_dim` | Value Head Dimension | Value 头维度 | `int` | 高级 | 大于 0，且不超过两个 QK 维度之和 |

## KDA

| Web 字段 | English label | 中文名称 | 类型 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.kda_head_dim` | KDA Head Dimension | KDA 头维度 | `int` | 基础 | 大于 0 |
| `model_overrides.kda_layers` | KDA Layer Indices | KDA 层索引 | `list[int]` | 基础 | 0-based、非负、不能重复；未列出的活动层使用 Gated MLA |
| `model_overrides.conv_kernel_size` | Short Convolution Kernel | 短卷积核大小 | `int` | 高级 | 大于 0 |
| `model_overrides.gate_lower_bound` | KDA Gate Lower Bound | KDA 门控下界 | `float \| null` | 专家 | `None` 表示不添加下界 |
| `model_overrides.use_full_rank_gate` | Full-Rank Output Gate | 使用全秩输出门控 | `bool` | 高级 | 关闭后使用低秩 gate 投影 |

`kda_layers` 可以保留大于等于 `n_layers` 的尾部索引，这些值不会被当前模型
消费。因此从 93 层 preset 缩短层数时只改 `n_layers` 即可；增大层数时，上游
必须确认新增层需要使用 KDA 还是 MLA，并相应更新列表。

## Dense FFN、MoE 与路由

| Web 字段 | English label | 中文名称 | 类型 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.dense_hidden_dim` | Dense FFN Dimension | Dense FFN 中间维度 | `int` | 基础 | 大于 0 |
| `model_overrides.moe_inter_dim` | Expert Intermediate Dimension | 专家中间维度 | `int` | 基础 | 大于 0 |
| `model_overrides.num_experts` | Routed Experts | 路由专家数 | `int` | 基础 | 大于 0，且不小于 `router_top_k` |
| `model_overrides.num_shared_experts` | Shared Experts | 共享专家数 | `int` | 基础 | 大于等于 0；0 关闭共享专家 |
| `model_overrides.router_top_k` | Experts per Token | 每 Token 激活专家数 | `int` | 基础 | 大于 0，且不超过 `num_experts` |
| `model_overrides.router_score_func` | Router Score Function | 路由评分函数 | `enum` | 高级 | `sigmoid` 或 `softmax` |
| `model_overrides.num_expert_groups` | Expert Groups | 专家分组数 | `int` | 高级 | 大于 0，且整除 `num_experts` |
| `model_overrides.topk_group` | Candidate Expert Groups | 候选专家组数 | `int` | 高级 | 单组时必须为 1；启用多组时必须小于 `num_expert_groups` |
| `model_overrides.routed_expert_hidden_size` | Latent MoE Hidden Size | Latent MoE 隐藏维度 | `int \| null` | 基础 | `None` 关闭 LatentMoE；否则大于 0 |
| `model_overrides.latent_moe_use_norm` | Normalize Latent MoE Output | LatentMoE 输出归一化 | `bool` | 高级 | 仅 LatentMoE 开启时生效 |
| `model_overrides.routed_scaling_factor` | Routed Output Scale | 路由输出缩放 | `float` | 高级 | 大于 0 |
| `model_overrides.renormalize` | Renormalize Top-K Weights | 归一化 Top-K 权重 | `bool` | 高级 | 控制选中专家权重是否重新归一化 |

多专家组还要求每组至少包含 2 个专家，并且被选中组内的专家总数不能小于
`router_top_k`。`debug_force_load_balance` 不属于模型
override；它由 `debug.moe_force_load_balance` 在运行时同步到每个 MoE 层。

## SiTU-GLU

| Web 字段 | English label | 中文名称 | 类型 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.situ_beta` | SiTU Beta | SiTU Beta | `float` | 高级 | 大于 0；Dense、共享专家和路由专家共用 |
| `model_overrides.situ_linear_beta` | SiTU Linear Clamp | SiTU 线性限幅 | `float \| null` | 高级 | `None` 关闭；否则大于 0 |

## 内部字段

| Web 字段 | English label | 中文名称 | 类型 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.param_init` | Parameter Initializers | 参数初始化器 | `map \| null` | 隐藏 | 通常包含 Python callable，不适合作为 Web JSON；普通 Web 请求应省略该字段，后端保持 preset 的 `null` 值 |

## Preset 默认值

页面应先加载所选 config 返回的完整 `model_overrides`，再用它初始化表单；
不要在前端复制默认值。当前三个模型 preset 的关键差异如下：

| Preset/config | Layers | Dim | Heads | KDA/MLA | Dense layers | Experts | Top-K | Expert dim | Latent dim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `debug` / `kimi_k3_smoketest` | 4 | 256 | 8 | 3 / 1 | 1 | 8 | 3 | 128 | 128 |
| `16layer_reduced` / `kimi_k3_16layer_reduced` | 16 | 7168 | 96 | 12 / 4 | 1 | 32 | 16 | 3072 | 3584 |
| `full` / `kimi_k3_baseline_bf16`、`kimi_k3_baseline_mxfp8` | 93 | 7168 | 96 | 69 / 24 | 1 | 896 | 16 | 3072 | 3584 |

公共默认值为：`param_init=null`、`num_shared_experts=2`、`router_score_func="sigmoid"`、
`num_expert_groups=1`、`topk_group=1`、`routed_scaling_factor=1.0`、
`renormalize=true`、`situ_beta=4.0`、`situ_linear_beta=25.0`、
`conv_kernel_size=4`、`gate_lower_bound=-5.0`、
`use_full_rank_gate=true`、`norm_eps=1e-5`、`attn_res_block_size=12`。

## 上游最低联动与提交规则

1. 先选择 preset，并从后端读取完整 `model_overrides`。
2. 上游可以提交完整对象，也可以转换成 CLI 后只覆盖变化字段。
3. 修改 `n_layers` 时检查 dense 前缀范围，并检查新增层的 `kda_layers` 归属。
4. 修改 QK/V 维度时满足 `v_head_dim <= qk_nope_head_dim + qk_rope_head_dim`。
5. 修改专家数时同时检查 Top-K、专家分组整除关系和每组专家数。
6. `routed_expert_hidden_size=None` 时建议隐藏 `latent_moe_use_norm`。
7. 后端始终执行最终类型与联动校验；前端校验不能替代后端校验。

逐层 `layers[].layer_id`、attention 类型、dense/MoE 类型、各子配置中的
`dim`/`norm_eps`/`attn_res_block_size`，以及 MoE 的
`debug_force_load_balance` 都是派生字段，不应由上游直接提交。
