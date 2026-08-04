# DeepSeek-V4 模型参数 Web 字段目录

本文档定义 Web 页面可配置的 DeepSeek-V4 **模型参数**。范围仅包括：

- `DeepSeekV4Model.Config`
- `DeepSeekV4Model.Config.moe_args` 对应的 `MoEArgs`

不包括训练、并行、重计算、优化器、通信、检查点和 simulator 参数。

当前模型配置包含 35 个一级字段，其中 `moe_args` 是嵌套对象；展开后共有
49 个叶子参数。

## 字段命名约定

Web 请求和内部数据结构推荐使用 snake_case：

```json
{
  "model_overrides": {
    "n_layers": 16,
    "dim": 4096,
    "moe_args": {
      "num_experts": 256,
      "top_k": 8
    }
  }
}
```

转换为 CLI 时保持相同层级，仅将下划线转换为连字符：

```text
model_overrides.n_layers
    -> --model-overrides.n-layers

model_overrides.moe_args.num_experts
    -> --model-overrides.moe-args.num-experts
```

布尔字段：

```text
true  -> --model-overrides.use-smla
false -> --model-overrides.no-use-smla
```

可空字段使用 `None`。列表字段使用一个参数名加多个值：

```text
--model-overrides.compress-ratios 1 4 128
```

## Web 展示级别

| 级别 | 建议 |
|---|---|
| 基础 | 默认展示，适合模型规模和显存估算页面 |
| 高级 | 放入“高级模型参数”区域 |
| 专家 | 默认折叠，需要理解具体算法后修改 |
| 隐藏 | 不建议在普通页面展示，保留给调试或内部接口 |
| 派生 | 字段属于模型 schema，但当前运行时会从其他状态重新计算；建议只读或隐藏 |

## 核心模型结构

| Web 字段 | English label | 中文名称 | 类型/控件 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.vocab_size` | Vocabulary Size | 词表大小 | `int` / 整数输入 | 基础 | 必须大于 0；修改后需要匹配 tokenizer 和 checkpoint |
| `model_overrides.dim` | Hidden Dimension | 隐藏层维度 | `int` / 整数输入 | 基础 | 必须大于 0；直接影响参数量、激活和算子 shape |
| `model_overrides.n_layers` | Transformer Layers | Transformer 层数 | `int` / 整数输入 | 基础 | 必须大于 0；必须同步调整 `compress_ratios` |
| `model_overrides.n_heads` | Attention Heads | 注意力头数 | `int` / 整数输入 | 基础 | 必须大于 0，且能被 `o_groups` 整除 |
| `model_overrides.head_dim` | Attention Head Dimension | 注意力头维度 | `int` / 整数输入 | 基础 | 必须大于 0，且不小于 `rope_head_dim` |
| `model_overrides.norm_eps` | Normalization Epsilon | 归一化 Epsilon | `float` / 小数输入 | 高级 | 必须大于 0 |
| `model_overrides.max_batch_size` | Maximum Batch Size | 最大批大小 | `int` / 整数输入 | 高级 | 必须大于 0；模型侧容量参数，不等同于训练 batch size |
| `model_overrides.max_seq_len` | Maximum Sequence Length | 最大序列长度 | `int` / 整数输入 | 派生 | 必须大于 0；当前 `update_from_config()` 会用实际序列长度更新该值 |

## 注意力与压缩结构

| Web 字段 | English label | 中文名称 | 类型/控件 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.rope_head_dim` | RoPE Head Dimension | RoPE 头维度 | `int` / 整数输入 | 高级 | 必须大于 0，且不大于 `head_dim` 和 `index_head_dim` |
| `model_overrides.q_lora_rank` | Query LoRA Rank | 查询 LoRA 秩 | `int` / 整数输入 | 基础 | 必须大于 0；影响 Query 投影参数量 |
| `model_overrides.o_lora_rank` | Output LoRA Rank | 输出 LoRA 秩 | `int` / 整数输入 | 基础 | 必须大于 0；影响输出投影参数量 |
| `model_overrides.o_groups` | Output Projection Groups | 输出投影分组数 | `int` / 整数输入 | 高级 | 必须大于 0；`n_heads` 必须能被该值整除 |
| `model_overrides.window_size` | Sliding Window Size | 滑动窗口大小 | `int` / 整数输入 | 高级 | 必须大于 0 |
| `model_overrides.compress_ratios` | Per-Layer Compression Ratios | 逐层压缩比 | `list[int]` / 整数列表 | 高级 | 元素数量不能少于 `n_layers`；允许保留不会被当前层数消费的尾部值；每个值必须大于等于 0 |
| `model_overrides.use_smla` | Enable SMLA | 启用 SMLA | `bool` / 开关 | 派生 | 当前运行时根据是否安装 `npu_smla` converter 更新，普通页面建议隐藏 |

## 索引器

| Web 字段 | English label | 中文名称 | 类型/控件 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.index_n_heads` | Indexer Heads | 索引器头数 | `int` / 整数输入 | 高级 | 必须大于 0 |
| `model_overrides.index_head_dim` | Indexer Head Dimension | 索引器头维度 | `int` / 整数输入 | 高级 | 必须大于 0，必须是 2 的幂，且不小于 `rope_head_dim` |
| `model_overrides.index_topk` | Indexer Top-K | 索引器 Top-K | `int` / 整数输入 | 高级 | 必须大于 0 |
| `model_overrides.enable_indexer_loss` | Enable Indexer Loss | 启用索引器损失 | `bool` / 开关 | 高级 | 控制是否计算索引器辅助损失 |

## Hyper-Connection

| Web 字段 | English label | 中文名称 | 类型/控件 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.hc_sinkhorn_iters` | Hyper-Connection Sinkhorn Iterations | Hyper-Connection Sinkhorn 迭代次数 | `int` / 整数输入 | 专家 | 必须大于 0 |
| `model_overrides.hc_mult` | Hyper-Connection Multiplicity | Hyper-Connection 分支数 | `int` / 整数输入 | 高级 | 必须大于 0；影响隐藏状态扩展倍率 |
| `model_overrides.hc_eps` | Hyper-Connection Epsilon | Hyper-Connection Epsilon | `float` / 小数输入 | 专家 | 必须大于 0 |

## MoE 结构与路由

### 模型级 MoE 参数

| Web 字段 | English label | 中文名称 | 类型/控件 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.moe_inter_dim` | Expert Intermediate Dimension | 专家中间层维度 | `int` / 整数输入 | 基础 | 必须大于 0；直接影响单个专家参数量 |
| `model_overrides.load_balance_coeff` | Expert Load-Balance Coefficient | 专家负载均衡系数 | `float` / 小数输入 | 高级 | 必须大于 0；当前会同步到 `moe_args.load_balance_coeff` |

### `moe_args` 全量字段

| Web 字段 | English label | 中文名称 | 类型/控件 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.moe_args.num_experts` | Routed Experts | 路由专家数 | `int` / 整数输入 | 基础 | 必须大于 0，且不小于 `top_k` |
| `model_overrides.moe_args.num_shared_experts` | Shared Experts | 共享专家数 | `int` / 整数输入 | 基础 | 必须大于等于 0 |
| `model_overrides.moe_args.score_func` | Router Score Function | 路由评分函数 | `enum` / 下拉框 | 高级 | 可选 `softmax`、`sigmoid`、`sqrtsoftplus` |
| `model_overrides.moe_args.route_norm` | Normalize Routing Weights | 归一化路由权重 | `bool` / 开关 | 高级 | 控制 Top-K 路由权重是否重新归一化 |
| `model_overrides.moe_args.route_scale` | Routing Score Scale | 路由分数缩放系数 | `float` / 小数输入 | 高级 | 必须大于 0 |
| `model_overrides.moe_args.gate_bias` | Router Gate Bias | 路由门控偏置 | `bool` / 开关 | 专家 | 控制 router gate 是否使用 bias |
| `model_overrides.moe_args.score_before_experts` | Apply Score Before Experts | 专家计算前应用路由分数 | `bool` / 开关 | 专家 | 控制路由权重应用在专家计算前还是合并阶段 |
| `model_overrides.moe_args.top_k` | Experts Per Token | 每 Token 激活专家数 | `int` / 整数输入 | 基础 | 必须大于 0，且不大于 `num_experts` |
| `model_overrides.moe_args.num_expert_groups` | Expert Groups | 专家分组数 | `int \| null` / 可空整数 | 高级 | 非空时必须大于 0，并整除 `num_experts` |
| `model_overrides.moe_args.num_limited_groups` | Routing Candidate Groups | 路由候选组数 | `int \| null` / 可空整数 | 高级 | 非空时必须大于 0；若设置专家分组，则不能超过 `num_expert_groups` |
| `model_overrides.moe_args.use_grouped_mm` | Use Grouped Matrix Multiplication | 使用分组矩阵乘 | `bool` / 开关 | 高级 | 关闭后使用逐专家计算路径，通常不建议关闭 |
| `model_overrides.moe_args.load_balance_coeff` | MoE Load-Balance Coefficient | MoE 负载均衡系数 | `float \| null` / 可空小数 | 隐藏 | 非空时必须大于 0；当前会被模型级 `load_balance_coeff` 覆盖 |
| `model_overrides.moe_args.debug_force_load_balance` | Force Balanced Routing | 强制均衡路由 | `bool` / 开关 | 派生 | 当前会从调试状态更新，仅用于调试 |
| `model_overrides.moe_args.n_hash_layers` | Hash-Routed Layers | 哈希路由层数 | `int` / 整数输入 | 高级 | 必须大于等于 0；前 N 层使用 hash routing |
| `model_overrides.moe_args.swiglu_limit` | SwiGLU Activation Limit | SwiGLU 激活上限 | `float` / 小数输入 | 专家 | 必须大于 0 |

## RoPE 与上下文扩展

| Web 字段 | English label | 中文名称 | 类型/控件 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.compress_rope_theta` | Compression RoPE Theta | 压缩 RoPE Theta | `float` / 小数输入 | 专家 | 必须大于 0 |
| `model_overrides.original_seq_len` | Original Context Length | 原始上下文长度 | `int` / 整数输入 | 高级 | 必须大于 0 |
| `model_overrides.rope_theta` | RoPE Base Theta | RoPE 基础 Theta | `int` / 整数输入 | 高级 | 必须大于 0 |
| `model_overrides.rope_factor` | RoPE Scaling Factor | RoPE 缩放系数 | `int` / 整数输入 | 高级 | 必须大于 0 |
| `model_overrides.beta_fast` | YaRN Fast Beta | YaRN 快速 Beta | `int` / 整数输入 | 专家 | 必须大于 0，并大于 `beta_slow` |
| `model_overrides.beta_slow` | YaRN Slow Beta | YaRN 慢速 Beta | `int` / 整数输入 | 专家 | 必须大于等于 0，并小于 `beta_fast` |

## MTP

| Web 字段 | English label | 中文名称 | 类型/控件 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.num_mtp_modules` | MTP Modules | MTP 模块数 | `int` / 整数输入 | 派生 | 必须大于等于 0；当前运行时由 MTP 状态更新，普通页面建议只读或隐藏 |
| `model_overrides.mtp_layer_compress_ratio` | MTP Layer Compression Ratio | MTP 层压缩比 | `int` / 整数输入 | 高级 | 必须大于等于 0 |

## 内部与调试字段

| Web 字段 | English label | 中文名称 | 类型/控件 | 展示 | 规则与说明 |
|---|---|---|---|---|---|
| `model_overrides.param_init` | Parameter Initializers | 参数初始化器 | `map \| null` | 隐藏 | 值通常包含 Python callable，不适合作为通用 Web JSON 参数；建议固定为 `null` |
| `model_overrides.debug_force_load_balance` | Force Model Load Balance | 强制模型负载均衡 | `bool` / 开关 | 隐藏 | 模型级调试字段；实际 MoE 路由调试状态由嵌套字段和运行时状态决定 |

## 默认值使用规则

Web 页面应以所选 preset 解析后的完整 `model_overrides` 作为表单默认值，
不应在前端代码中复制一份默认配置。

推荐流程：

1. 用户选择 preset。
2. 后端加载对应 config，并返回完整 `model_overrides`。
3. Web 使用返回值初始化全部模型参数。
4. 用户只提交相对该默认值发生变化的字段，或提交完整模型参数对象。
5. 后端再次执行类型和联动校验。

以下数值是当前 `model_registry` 中四个模型 preset 的原始默认值。派生字段
仍可能在模型构建阶段按前文说明更新。

### 核心规模对比

| Preset | Layers | Hidden Dim | Heads | Head Dim | Q LoRA Rank | O LoRA Rank | O Groups | Expert Dim | Experts | Top-K | Index Top-K | Compression Pattern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoketest` | 4 | 128 | 4 | 32 | 64 | 32 | 4 | 64 | 8 | 2 | 16 | `(1, 1, 4, 128)` |
| `v4_flash_baseline` | 43 | 4096 | 64 | 512 | 1024 | 1024 | 8 | 2048 | 256 | 6 | 512 | `(1, 1, 4) + (128, 4) * 20` |
| `v4_pro_baseline` | 61 | 7168 | 128 | 512 | 1536 | 1024 | 16 | 3072 | 384 | 6 | 1024 | `(128,) + (128, 4) * 30` |
| `v4_pro_20t_baseline` | 96 | 12288 | 192 | 512 | 2304 | 1536 | 16 | 2752 | 2048 | 23 | 1024 | `(128, 4) * 47 + (128, 0)` |

### 完整模型默认值

| Web 字段 | `smoketest` | `v4_flash_baseline` | `v4_pro_baseline` | `v4_pro_20t_baseline` |
|---|---:|---:|---:|---:|
| `model_overrides.param_init` | `None` | `None` | `None` | `None` |
| `model_overrides.norm_eps` | `1e-6` | `1e-6` | `1e-6` | `1e-6` |
| `model_overrides.vocab_size` | `129280` | `129280` | `129280` | `129280` |
| `model_overrides.dim` | `128` | `4096` | `7168` | `12288` |
| `model_overrides.n_layers` | `4` | `43` | `61` | `96` |
| `model_overrides.n_heads` | `4` | `64` | `128` | `192` |
| `model_overrides.head_dim` | `32` | `512` | `512` | `512` |
| `model_overrides.max_batch_size` | `2` | `4` | `4` | `4` |
| `model_overrides.max_seq_len` | `128` | `4096` | `4096` | `4096` |
| `model_overrides.rope_head_dim` | `16` | `64` | `64` | `64` |
| `model_overrides.q_lora_rank` | `64` | `1024` | `1536` | `2304` |
| `model_overrides.o_lora_rank` | `32` | `1024` | `1024` | `1536` |
| `model_overrides.o_groups` | `4` | `8` | `16` | `16` |
| `model_overrides.window_size` | `32` | `128` | `128` | `128` |
| `model_overrides.compress_ratios` | `(1, 1, 4, 128)` | `(1, 1, 4) + (128, 4) * 20` | `(128,) + (128, 4) * 30` | `(128, 4) * 47 + (128, 0)` |
| `model_overrides.index_n_heads` | `4` | `64` | `64` | `64` |
| `model_overrides.index_head_dim` | `16` | `128` | `128` | `128` |
| `model_overrides.index_topk` | `16` | `512` | `1024` | `1024` |
| `model_overrides.enable_indexer_loss` | `false` | `true` | `true` | `true` |
| `model_overrides.hc_sinkhorn_iters` | `4` | `20` | `20` | `24` |
| `model_overrides.hc_mult` | `4` | `4` | `4` | `4` |
| `model_overrides.hc_eps` | `1e-6` | `1e-6` | `1e-6` | `1e-6` |
| `model_overrides.moe_inter_dim` | `64` | `2048` | `3072` | `2752` |
| `model_overrides.load_balance_coeff` | `0.001` | `0.001` | `0.001` | `0.001` |
| `model_overrides.compress_rope_theta` | `40000` | `160000` | `160000` | `160000` |
| `model_overrides.original_seq_len` | `128` | `65536` | `65536` | `65536` |
| `model_overrides.rope_theta` | `10000` | `10000` | `10000` | `10000` |
| `model_overrides.rope_factor` | `4` | `16` | `16` | `16` |
| `model_overrides.beta_fast` | `32` | `32` | `32` | `32` |
| `model_overrides.beta_slow` | `1` | `1` | `1` | `1` |
| `model_overrides.use_smla` | `false` | `false` | `false` | `false` |
| `model_overrides.num_mtp_modules` | `0` | `0` | `0` | `0` |
| `model_overrides.mtp_layer_compress_ratio` | `1` | `1` | `1` | `1` |
| `model_overrides.debug_force_load_balance` | `false` | `false` | `false` | `false` |

### 完整 MoE 默认值

| Web 字段 | `smoketest` | `v4_flash_baseline` | `v4_pro_baseline` | `v4_pro_20t_baseline` |
|---|---:|---:|---:|---:|
| `model_overrides.moe_args.num_experts` | `8` | `256` | `384` | `2048` |
| `model_overrides.moe_args.num_shared_experts` | `1` | `1` | `1` | `1` |
| `model_overrides.moe_args.score_func` | `sqrtsoftplus` | `sqrtsoftplus` | `sqrtsoftplus` | `sqrtsoftplus` |
| `model_overrides.moe_args.route_norm` | `true` | `true` | `true` | `true` |
| `model_overrides.moe_args.route_scale` | `1.5` | `1.5` | `1.5` | `1.5` |
| `model_overrides.moe_args.gate_bias` | `false` | `false` | `false` | `false` |
| `model_overrides.moe_args.score_before_experts` | `false` | `false` | `false` | `false` |
| `model_overrides.moe_args.top_k` | `2` | `6` | `6` | `23` |
| `model_overrides.moe_args.num_expert_groups` | `None` | `None` | `None` | `None` |
| `model_overrides.moe_args.num_limited_groups` | `8` | `8` | `8` | `8` |
| `model_overrides.moe_args.use_grouped_mm` | `true` | `true` | `true` | `true` |
| `model_overrides.moe_args.load_balance_coeff` | `0.001` | `0.001` | `0.001` | `0.001` |
| `model_overrides.moe_args.debug_force_load_balance` | `false` | `false` | `false` | `false` |
| `model_overrides.moe_args.n_hash_layers` | `0` | `3` | `3` | `3` |
| `model_overrides.moe_args.swiglu_limit` | `10` | `10` | `10` | `10` |

## Web 端最低联动规则

Web 页面至少应实现以下即时校验，后端仍需执行最终校验：

1. `compress_ratios` 的元素数量必须大于等于 `n_layers`；减少层数时可以保留尾部值，增加层数时必须补足。
2. `n_heads % o_groups == 0`。
3. `rope_head_dim <= head_dim`。
4. `rope_head_dim <= index_head_dim`。
5. `index_head_dim` 必须是 2 的幂。
6. `moe_args.top_k <= moe_args.num_experts`。
7. `num_expert_groups` 非空时必须整除 `num_experts`。
8. `num_limited_groups` 和 `num_expert_groups` 都非空时，前者不能大于后者。
9. `beta_fast > beta_slow >= 0`。
10. 除明确允许 0 或 `null` 的字段外，整数和浮点规模参数必须大于 0。

## 推荐的 Web 分组

普通用户默认展示：

- 模型规模：`vocab_size`、`n_layers`、`dim`、`n_heads`、`head_dim`
- 注意力投影：`q_lora_rank`、`o_lora_rank`
- MoE 规模：`moe_inter_dim`、`num_experts`、`num_shared_experts`、`top_k`

高级区域展示：

- 注意力压缩、索引器、RoPE、Hyper-Connection、专家分组和 MTP 压缩参数

隐藏或只读：

- `param_init`
- 两个 `debug_force_load_balance`
- `moe_args.load_balance_coeff`
- `use_smla`
- `num_mtp_modules`
- `max_seq_len`
