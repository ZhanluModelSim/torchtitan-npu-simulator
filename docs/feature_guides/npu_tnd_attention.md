# TND Attention (Per-Document Causal Attention)

SFT 训练中样本通常短于最大序列长度，BSND 逐样本训练有大量算力浪费在 padding 上。
TND Attention 将多条短样本打包到同一序列中，通过 block_causal mask 隔离各文档的注意力范围，
实现更高训练吞吐并防止跨文档信息泄露。

## 1. 实现原理

继承上游 `VarlenAttention`，通过 `inner_attention` 插件机制替换 attention 后端：

```
GQAttention.forward (上游, 不修改)
  ├─ QKV / QK-norm / RoPE    ← 复用上游
  └─ inner_attention ──────► NPUVarlenAttention.forward
                               └─ CANN FA v3 sparse_mode=7
```

### 1.1 文档边界检测

直接复用上游 EOS-based VarlenMetadata（`model.get_attention_masks()` 在
`post_dataloading_process` 中计算，`add_eos=True` 确保每文档尾部有 EOS）。

### 1.2 Context Parallel

使用 Ulysses all-to-all：pre-hook 交换 Q/K/V (scatter head → gather seq)，
post-hook 还原。forward 处理完整序列，与 CP=1 逻辑一致。

约束：`n_heads % CP == 0` 且 `n_kv_heads % CP == 0`。
Qwen3-30B-A3B (n_kv_heads=4): CP ≤ 4。

## 2. 配置方式

TND 通过替换模型 spec 中的 `inner_attention` 实现，核心调用链：

```
config_registry.py                       tnd_config.py
┌─────────────────────┐   调用    ┌──────────────────────────────┐
│ sft_xxx_tnd() ──────┼─────────►│ _enable_npu_varlen_attention │
│   config = sft_xxx()│           │   layer.attention             │
│   config.model_spec  │           │     .inner_attention          │
│     = _enable_npu_   │           │     = NPUVarlenAttention     │
│       varlen_attn(…)│           │     .mask_type                │
└─────────────────────┘           │     = "block_causal"         │
                                  └──────────────────────────────┘
```

`_enable_npu_varlen_attention` 遍历模型每一层，将 `inner_attention` 替换为
`NPUVarlenAttention.Config()` 并将 `mask_type` 设为 `"block_causal"`。
同时调用 `_patch_update_from_config` 绕过上游 CP 对 VarlenAttention 的限制。

以 Qwen3-30B-A3B 为例，BSND 与 TND 的配置仅后缀不同，其余参数完全一致：

| 模式 | 配置名 |
|------|--------|
| BSND | `sft_qwen3_30ba3b_gsm8k` |
| TND | `sft_qwen3_30ba3b_gsm8k_tnd` |

> **注意**：当前仅支持 Qwen3 系列模型。其他模型需完成 attention 层适配后接入。

## 3. 关键文件

| 文件 | 职责 |
|------|------|
| `torchtitan_npu/models/common/npu_varlen_attention.py` | NPUVarlenAttention 类 + FA v3 forward |
| `torchtitan_npu/models/qwen3/tnd_config.py` | `_enable_npu_varlen_attention` + CP-Varlen bypass |
| `torchtitan_npu/models/qwen3/__init__.py` | NPU model registry（`parallelize_fn` 注入） |
| `torchtitan_npu/distributed/context_parallel/npu_varlen_cp.py` | UlyssesVarlenCP（all-to-all CP 策略 + mask handler） |
| `torchtitan_npu/models/qwen3/config_registry.py` | TND 配置注册 |

## 4. 适配新模型（当前仅 Qwen3）

> **当前支持范围**：仅 Qwen3 系列模型已完成 TND 适配。其他模型架构需完成
> attention 层适配（确保 `inner_attention` 插件机制可用）后方可接入。

适配步骤：

1. 在模型的 `__init__.py` 中确保 `GQAttention` 作为默认 attention 层
2. 注册 TND config，调用 `_enable_npu_varlen_attention` 替换 `inner_attention`：

```python
from torchtitan_npu.models.qwen3.tnd_config import _enable_npu_varlen_attention
model_spec = _enable_npu_varlen_attention(model_registry("your_flavor"))
```
