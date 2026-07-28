# DeepSeek-V4 TND 适配

DeepSeek-V4（DSV4）仅在 DSA 注意力内部使用 TND。模型、Compressor、MoE 和训练器仍保持 `[B, S, ...]` 接口；进入 CANN 融合算子前才收集有效 token，算子输出随后还原到原排布。本文说明当前实现、启用方式和已知边界。

## 启用方式

融合路径要求完整的 Ascend NPU 运行环境，以及与 CANN 版本匹配的 `cann_ops_transformer`。推荐使用仓内脚本：

```bash
TORCHTITAN_DIR=/path/to/torchtitan \
ATTENTION=smla \
./scripts/run_train_dsv4.sh
```

手动配置时，以下两个 override 必须同时启用：

```text
torchtitan_npu.override.deepseek_v4.fused_dsa.npu_smla_tnd_override
torchtitan_npu.override.deepseek_v4.varlen_dsa.npu_dsv4_packed_mask_handler_override
```

前者替换 DSA 内核，后者把 varlen mask 转为压缩计划。缺少 packed mask handler 时，模型构建会报错。完整 recipe 及 Golden 路径见 [override 说明](../torchtitan_npu/override/readme.md)。

## 数据流

```text
模型张量 [B, S, ...]
        │
        ├─ positions + VarlenMetadata
        │      └─ DSV4PackedMetadata
        │
        ├─ Compressor：生成固定存储 [B, S//r, D]
        │
        └─ DSA 内部
             query       [B, S, N, D] → [T, N, D]
             original KV [B, S, D]    → [T, 1, D]
             compressed  [B, S//r, D] → [T_cmp, 1, D]
             LI / SMLA / SMLAG / SLIG
             output      [T, N, D]     → [B, S, N, D]
```

TND 只存在于 DSA 的局部计算中，不会成为模型公共布局。

| 符号 | 含义 |
| --- | --- |
| `B` | 模型侧固定容器的行数 |
| `S` | 每行 token 容量 |
| `T` | 收集后的原始 token 总数 |
| `T_cmp` | 某个压缩比下的完整压缩块总数 |
| `N` | 注意力头数 |
| `D` | 单头维度 |
| `r` | 压缩比；当前 DSV4 配置使用 1、4 和 128 |

## Packed 元数据

[`packed.py`](../torchtitan_npu/models/deepseek_v4/packed.py) 生成后端无关的 `DSV4PackedMetadata`，Golden 和 CANN 路径共用同一套序列及压缩边界。

主要信息包括：

- `varlen`、`lengths` 和 `sequence_ranges`：原始 token 流的序列边界；
- `token_indices`、`token_sequence_ids` 和 `token_positions`：`[B, S]` 容器与连续 token 流之间的映射；
- `compressed[ratio]`：对应压缩比的完整块边界、尾部余数和物理存储映射；
- `container_batch_size`、`container_seq_len`：模型侧容器形状；
- `cache_id`：区分不同 microbatch 的 CANN metadata 缓存。

容器行数不等于逻辑序列数。例如：

```text
positions     = [[0, 1, 2, 0, 1]]
B             = 1
num_sequences = 2
cu_seqlens    = [0, 3, 5]
```

每条序列必须从位置 0 开始并连续递增。该约束保证压缩、因果 mask 和稀疏索引不会跨越序列边界。

## 压缩与索引规则

[`compressor.py`](../torchtitan_npu/models/deepseek_v4/compressor.py) 只处理完整压缩块。对长度为 `L` 的序列：

```text
compressed_len = floor(L / r)
residual       = L - r * compressed_len
```

不足一个完整块的尾部不生成压缩 KV，而是通过 `residual` 传给 CANN metadata。

- `r=1`：不生成压缩 KV，仅执行原始 KV 的滑窗注意力。
- `r=4`：启用重叠压缩和 LightningIndexer；前驱块必须属于同一序列。
- `r=128`：生成连续压缩 KV，不执行 LightningIndexer TopK。

参考 `IndexSelection` 使用 FP32 计算 q/k 分数和权重归约。索引是当前序列压缩 KV 范围内的局部下标；候选不足时以 `-1` 和 `-inf` 补齐固定 TopK 宽度。

## CANN 融合路径

[`fused_dsa.py`](../torchtitan_npu/override/deepseek_v4/fused_dsa.py) 执行以下步骤：

1. 校验模型输入和 `DSV4PackedMetadata` 描述同一 `[B, S]` 容器。
2. 收集 query、原始 KV、压缩 KV，以及 `r=4` 时的 Indexer 输入。
3. 调用 LightningIndexer 生成序列内局部 TopK。
4. 调用 SparseFlashMLA；手写 Autograd 封装在反向中调用 SMLAG，并在需要时调用 SLIG 计算 Indexer auxiliary loss 梯度。
5. 将 `[T, N, D]` 输出写回 `[B, S, N, D]`。

关键算子参数如下：

| 参数 | 来源 |
| --- | --- |
| `cu_seqlens_q` | 原始 query 序列边界 |
| `cu_seqlens_ori_kv` | 原始 KV 序列边界 |
| `cu_seqlens_cmp_kv` | 压缩块序列边界 |
| `cmp_residual_kv` | 每条序列不足一个完整块的尾部长度 |
| `layout_q` / `layout_kv` | 固定为 `TND` |

`torch.ops.cann_ops_transformer.sparse_flash_mla` 和 `sparse_flash_mla_grad` 是独立算子，当前算子包未将两者注册为自动反向关系，因此 `_SparseFlashMLATND` 仍负责连接前向、SMLAG 和 SLIG。

## 当前边界

| 项目 | 当前状态 |
| --- | --- |
| `positions` | 必须提供 `[B, S]`，且每条序列从 0 连续递增 |
| Padding | 训练用 varlen 流当前必须覆盖完整本地容器；`valid_tokens` 只用于直接构造 metadata，尚未贯穿训练流程 |
| Context Parallel | DSV4 sparse attention 当前明确拒绝 CP |
| 其他并行方式 | 已声明部分 TP、FSDP、EP 和序列并行切分；不能据此推断所有多卡组合均已验证 |
| 空压缩流 | metadata 可以表达 `T_cmp=0`，真实 CANN 前反向仍需按设备和版本单独验证 |
| 运行环境 | 需要 `torch_npu`、CANN、HCCL、Ascend NPU，以及匹配的 `cann_ops_transformer` |

## 轻量检查

以下命令只验证 Python 语法和文档改动，不等价于 NPU 训练验证：

```bash
python -m compileall -q torchtitan_npu
git diff --check
```
