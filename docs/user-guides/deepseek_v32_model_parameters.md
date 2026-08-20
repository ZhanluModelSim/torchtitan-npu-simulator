# DeepSeek-V3.2 模型参数 Web 字段目录

本文档定义 DeepSeek-V3.2 训练配置和 simulator 共同使用的稳定模型参数接口。
内部 `DeepSeekV32ModelNpu.Config` 保存展开后的逐层 `layers[]`；公共接口使用
`DeepSeekV32ModelOverrides`，修改任一字段后由后端统一重建 dense/MoE layer、
DSA/MLA attention、RoPE 和 MTP 配置。

## 请求格式与 CLI 映射

Web 请求使用 snake_case：

```json
{
  "model_overrides": {
    "n_layers": 16,
    "dim": 4096,
    "num_experts": 64,
    "router_top_k": 8,
    "index_topk": 2048
  }
}
```

CLI 将下划线转换为连字符：

```bash
python3 -m torchtitan_npu.entry \
  --module torchtitan_npu.simulator \
  --config deepseek_v32_smoketest \
  --model-overrides.n-layers 3 \
  --model-overrides.dim 192 \
  --model-overrides.n-heads 6 \
  --model-overrides.num-experts 16 \
  --model-overrides.router-top-k 4
```

布尔字段使用 `--model-overrides.<field>` 或
`--model-overrides.no-<field>`；可空字段传 `None`。训练模块
`torchtitan_npu.models.deepseek_v32` 与 simulator 使用同一 schema 和校验规则。

## 核心规模与 FFN

| 字段 | 类型 | 规则与说明 |
|---|---|---|
| `vocab_size` | `int` | 大于 0，需匹配 tokenizer/checkpoint |
| `dim` | `int` | 大于 0，所有主干层的隐藏维度 |
| `n_layers` | `int` | 大于 0，不含附加 MTP 层 |
| `n_dense_layers` | `int` | `0 <= value <= n_layers`，dense 必须构成前缀 |
| `dense_hidden_dim` | `int` | 大于 0，dense FFN 中间维度 |
| `moe_hidden_dim` | `int` | 大于 0，单个 routed/shared expert 中间维度 |
| `norm_eps` | `float` | 大于 0，统一作用于 RMSNorm |
| `num_mtp_modules` | `int` | 大于等于 0；同时同步到 training 配置 |

## MLA 与 DeepSeek Sparse Attention

| 字段 | 类型 | 规则与说明 |
|---|---|---|
| `n_heads` | `int` | 大于 0 |
| `q_lora_rank` | `int` | 大于 0 |
| `kv_lora_rank` | `int` | 大于 0 |
| `qk_nope_head_dim` | `int` | 大于 0 |
| `qk_rope_head_dim` | `int` | 大于 0，且不超过 `index_head_dim` |
| `v_head_dim` | `int` | 大于 0 |
| `mscale` | `float` | 大于 0，attention score 缩放 |
| `index_n_heads` | `int` | 大于 0，lightning indexer head 数 |
| `index_head_dim` | `int` | 大于 0 且为 2 的幂，满足 Hadamard 变换 |
| `index_topk` | `int` | 大于 0，稀疏 attention 每个 query 选择的 key 数 |
| `enable_mla_absorb` | `bool` | 是否使用 MLA absorb 投影路径 |
| `mask_type` | `str` | `causal` 使用 SDPA；`block_causal` 构造 FlexAttention |

## MoE 与路由

| 字段 | 类型 | 规则与说明 |
|---|---|---|
| `num_experts` | `int` | 大于 0，且不小于 `router_top_k` |
| `num_shared_experts` | `int` | 大于等于 0；0 关闭 shared expert |
| `router_top_k` | `int` | 大于 0，且不超过 `num_experts` |
| `router_score_func` | `enum` | `sigmoid` 或 `softmax` |
| `router_num_expert_groups` | `int \| null` | 非空时大于 0 且整除 `num_experts` |
| `router_num_limited_groups` | `int \| null` | 非空时大于 0，且不超过 expert group 数 |
| `router_route_scale` | `float` | 大于 0 |
| `router_route_norm` | `bool` | 是否归一化选中专家的 route score |
| `score_before_experts` | `bool` | 控制 route score 在 expert 前还是输出后应用 |

## YaRN RoPE

| 字段 | 类型 | 规则与说明 |
|---|---|---|
| `rope_max_seq_len` | `int` | 大于 0；运行时仍受 `training.seq_len` 约束 |
| `rope_theta` | `float` | 大于 0 |
| `rope_factor` | `float` | 大于 0 |
| `rope_beta_fast` | `float` | 大于 0，且必须大于 `rope_beta_slow` |
| `rope_beta_slow` | `float` | 大于等于 0 |
| `rope_original_seq_len` | `int` | 大于 0 |

## Preset 与运行约束

| Config | 主层数 | Dense/MoE | Dim | Experts | 用途 |
|---|---:|---:|---:|---:|---|
| `deepseek_v32_smoketest` | 2 | 1 / 1 | 128 | 8 | 单卡、FSDP、EP、CP、PP 和 AC 建模验证 |
| `deepseek_v32_tp_smoketest` | 2 | 1 / 1 | 128 | 8 | TP=2；使用 ATen MoE 路径以避开仅 TP 不支持的 NPU GMM |
| `deepseek_v32_671b_4layers_debug` | 4 | 3 / 1 | 7168 | 256 | 训练 debug 与结构放大验证 |
| `deepseek_v32_671b_61layers_4k_128die` | 61 | 3 / 58 | 7168 | 256 | 4K 正式规格 |
| `deepseek_v32_671b_61layers_32k_128die` | 61 | 3 / 58 | 7168 | 256 | 32K CP 正式规格 |

当前 V3.2 并行实现明确不支持 ETP，`expert_tensor_parallel_degree > 1` 会快速
失败。默认融合 MoE (`npu_moe_dispatch` + `npu_gmm`) 也不支持只开启 TP；纯 TP
验证应使用 `deepseek_v32_tp_smoketest`。这些约束不会静默回退。

上游应先读取所选 config 返回的完整 `model_overrides` 初始化表单，再提交完整
对象或仅转换变化字段为 CLI。逐层 `layers[]`、converter 生成的权重布局和
`debug.moe_force_load_balance` 都是派生/运行时字段，不应由 Web 直接提交。
