# DeepSeek-V4 TND 适配

DeepSeek-V4（DSV4）在模型层使用文档打包（varlen）的 `[B, S, ...]` 接口；压缩 KV 以容器网格 `[B, S//r, D]` 表达，仅在进入 AscendC 融合内核时转换为 TND。本文说明当前实现、启用方式和已知边界。

## 启用方式

融合路径要求完整的 Ascend NPU 运行环境，以及与 CANN 版本匹配的 `cann_ops_transformer`。推荐使用仓内脚本：

```bash
TORCHTITAN_DIR=/path/to/torchtitan \
./scripts/run_train.sh
```

DSA 内核有两种可切换的 recipe（`USE_GOLDEN` 选择；`--override.imports` 是
全控制逃生口）：

```text
USE_GOLDEN=1  # GOLDEN_OVERRIDES:
              #   override.common.rope.workaround
              #   override.deepseek_v4.sparse_attn.golden
              #   (the MoE is the normal clamped bf16 path on both sides)
USE_GOLDEN=0  # TEST_OVERRIDES (default):
              #   override.common.rms_norm.asc
              #   override.common.rope.asc_complex
              #   override.deepseek_v4.sparse_attn.asc_metadata=<geometry>
              #   override.deepseek_v4.sparse_attn.asc
```

融合路径必须搭配 AscendC metadata 扩展（`sparse_attn.asc_metadata`，
geometry 由脚本按 CONFIG 注入）。golden 参考路径使用模型默认的 metadata
构建（无扩展），无需额外 override。Golden DSA 参考
（eager 逐文档、FP32、与 `dsv4-infer-npu` 比特一致）用于双层数值校验：
golden 与 patched transformers 比特一致，AscendC 融合内核与 golden 在容差内
一致。`MODULE` 默认 `torchtitan_npu.models.deepseek_v4`，保证
`torchtitan_npu` 在构建早期被导入（其 `__init__` 激活 patches/torchtitan
backports）。

## 数据流

```text
模型张量 [B, S, ...]
        │
        ├─ positions + VarlenMetadata
        │      └─ CompressedVarlenMetadata（build_compressed_varlen_metadata 一次性构建公共契约；
        │         参考 tier 与 AscendC 内核张量经 metadata_extension 注入）
        │
        ├─ Compressor：按统一契约（gather_indices/block_positions/first_indices）压缩 → 池化键流；CP 下 Compressor 内部 borrow 原始行（borrow_x），调用侧按 plan 的 kept_indices/out_width 装入容器网格（pack_container）
        │
        ├─ 参考内核（golden/默认）
        │     kv = [swa_k | cmp_k 容器 | sink]，文档感知 BlockMask + dense mask top-k
        │
        └─ AscendC 融合内核（AscCompressedSparseInnerAttention）
             q/swa_k flatten → [T, N, D] / [T, 1, D]
             cmp_k 容器 → TND：container_flat[:n_blocks]（B=1 下等价于恒等映射）
             LI / SMLA / SMLAG / SLIG
             output [T, N, D] → [B, S, N, D]
```

TND 只存在于 NPU 融合内核的局部计算中，不会成为模型公共布局。

| 符号 | 含义 |
| --- | --- |
| `B` | 模型输入的 batch size |
| `S` | 模型输入的序列长度 |
| `T` | TND 下的 token 总数，等于 `B * S` |
| `T_cmp` | 某个压缩比下的完整压缩块总数 |
| `N` | 注意力头数 |
| `D` | 单头维度 |
| `r` | 压缩比；当前 DSV4 配置使用 1、4 和 128 |

## 压缩布局：CompressedVarlenMetadata

[`metadata.py`](../../torchtitan_npu/models/deepseek_v4/metadata.py) 定义唯一的注意力契约 `CompressedVarlenMetadata`，由同文件中的 `CompressedBlockMaskHandler`（模型目录默认 handler）每个 batch 构建一次，供所有 DSA 层复用：

- `varlen`：`VarlenMetadata`，`cu_seq_q` 是序列边界的唯一权威来源。模型目录的参考 tier 要求 `cu_seq_q == cu_seq_k`（连续文档，non-CP）；统一 kernel contract（`build_kernel_layout`）无此限制，CP 流由 AscendC 融合路径消费。
- `batch_size` / `seq_len`：容器网格形状。DSV4 打包场景使用 `local_batch_size == 1`，因此 `batch_size == 1`、`seq_len` 等于总 token 数（`cu_seq_q[-1]`），元数据完全由 varlen 流推导，不依赖 positions。
- `plans[ratio]`：每个模型实际使用的压缩比（1、4、128）对应一个 `CompressedBlockLayout`：
  - `cu_seqlens_cmp_k`、`block_remainder`：打包压缩块流边界与每文档尾部余数，直接喂给 AscendC 算子；
  - `gather_indices`、`block_positions`、`first_indices`：统一的 Compressor 契约（块 token gather、文档内块起点 RoPE、文档/段首块位置——重叠 borrow 掩码），CP 与非 CP 以同一字段名提供（CP 下 `plans[ratio]` 即 CP ratio plan——part 1 由 `build_kernel_layout` 在虚拟增强流上推导，`block_positions` 经 `+A` 文档锚定；`first_indices` 为 plan-block 序的段首位置；非 CP 下为 `cu_seqlens_cmp_k[:-1]`）；
  - `dense_mask` / `doc_of_block` / `block_local` / `static_blocks`：参考 tier（`reference.py`），`window_size`/`block_size` 为模型配置常量，经 `ReferenceMetadataExtension` 一次性预计算；
  - AscendC `*_metadata` 张量不进入模型目录元数据；AscendC override handler 返回携带 `asc_plans` 的 `AscCompressedVarlenMetadata` 包装。

一个 batch 里可以打包多条序列，所以 `B` 与序列条数无关。例如：

```text
positions  = [[0, 1, 2, 0, 1]]
B          = 1
cu_seqlens = [0, 3, 5]   # 一行里打包了 2 条序列
```

每条序列必须从位置 0 开始并连续递增。该约束保证压缩、因果 mask 和稀疏索引不会跨越序列边界。

## 压缩与索引规则

[`compressor.py`](../../torchtitan_npu/models/deepseek_v4/compressor.py) 只处理完整压缩块，且**绝不跨文档压缩**。对长度为 `L` 的序列：

```text
compressed_len = floor(L / r)
residual       = L - r * compressed_len
```

不足一个完整块的尾部不生成压缩 KV，而是通过 `residual` 传给 AscendC metadata。例如 `cu_seq_q = [0, 10, 27]`（两条序列 10 与 17），`r = 4`：

```text
cu_seqlens_cmp_k = [0, 2, 6]   # 压缩块：2 + 4
block_remainder = [2, 1]
gather_indices   = [A0..A3], [A4..A7], ..., [B12..B15]  # [6, 4]
block_positions  = [0, 4, 0, 4, 8, 12]
overlap_valid    = [F, T, F, T, T, T]  # 文档起始块没有前驱
```

- `r=1`：不生成压缩 KV，仅执行原始 KV 的滑窗注意力（plan 只含 AscendC metadata，`has_cmp_kv=False`）。
- `r=4`（CSA）：启用重叠压缩和 LightningIndexer；前驱块必须属于同一文档（`overlap_valid`）。
- `r=128`（HCA）：生成连续压缩 KV，不执行 LightningIndexer TopK。

## 参考路径（默认内核与 golden）

模型目录默认内核是 varlen 化的 `CompressedSparseInnerAttention`（上游 port）：在 `[swa_k | cmp_k | sink]` 拼接的容器 KV 上构建文档感知的两级 `BlockMask`（块列表超集 + token 级 `mask_mod`），CSA 的 top-k 由 `Indexer.select` 在 dense mask 上选择。块列表的静态部分（滑窗、sink、HCA 压缩区）由 handler 预计算进 `static_blocks`，每层只做 CSA top-k scatter 与 mask_mod 过滤。golden override 用 eager 逐文档 FP32 实现（gather-matmul + 逐文档 indexer top-k）作为比特级数值参考。

## AscendC 融合路径

[`sparse_attn/ascendc.py`](../../torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py) 中的 `AscCompressedSparseInnerAttention` 执行以下步骤：

1. 校验模型输入与 `CompressedVarlenMetadata` 一致，按 `self.compress_ratio` 取 plan。
2. 把 query、原始 KV 转成 TND；压缩 KV 与 Indexer key 取容器网格头部 `n_blocks` 槽位转成打包流。
3. 调用 LightningIndexer 生成序列内局部 TopK（使用 `asc_plans[ratio].li_metadata`）。
4. 调用 SparseFlashMLA；`_SparseFlashMLATND` 在反向中调用 SMLAG 与 SLIG（使用 `asc_plans[ratio].smla_grad_metadata`、`asc_plans[ratio].slig_metadata`）。SLIG 的 Indexer KL 梯度按 `indexer_loss_coeff`（NPU 内核配置，默认 1.0）缩放，LI 损失值累积到 `_indexer_loss_acc` 缓冲；Indexer auxiliary loss 已从模型目录移除，随上游机制落地后另行设计。

`cann_ops_transformer` 包负责内核的算子注册（schema、meta/fake 实现），本仓只在融合路径内做前反向桥接，不再自建 `torch.library` 封装。

| 参数 | 来源 |
| --- | --- |
| `cu_seqlens_q` / `cu_seqlens_ori_kv` | `varlen.cu_seq_q`（无 CP 时相同） |
| `cu_seqlens_cmp_kv` | `plan.cu_seqlens_cmp_k` |
| `cmp_residual_kv` | `plan.block_remainder` |
| `layout_q` / `layout_kv` | 固定为 `TND` |
| 四个 `*_metadata` 张量 | `plan`（NPU handler 预计算） |

## 当前边界

| 项目 | 当前状态 |
| --- | --- |
| `positions` | 必须提供 `[B, S]`，且每条序列从 0 连续递增 |
| Padding | varlen 流必须覆盖本 rank 的全部 `B * S` 个 token，metadata 不表达 padding token |
| Context Parallel | 支持（AscendC 融合路径，`CP_DEGREE=2 ./scripts/run_train.sh`）；CP 与非 CP 共用统一的 `plans[ratio]`（两部分：统一契约 + dispatcher 字段），CP 全部逻辑在 `token_dispatcher.py`（纯全局上下文 plan + `BaseEPTokenDispatcher` 形态的 dispatcher），`cp_plan` 只提供 `borrow_swa`/`borrow_x`/`pack_container` |
| Tensor Parallel | DSA 路径固定 TP=1（indexer score 对 head 维求和），TP 方案待 CP 之后另行设计 |
| 空压缩流 | doc-packing 场景要求每条序列至少产生一个完整压缩块；NPU handler 对 `T_cmp=0` 直接报错（AscendC CSA/HCA 不接受空 `cmp_kv`） |
| 运行环境 | 需要 `torch_npu`、CANN、HCCL、Ascend NPU，以及匹配的 `cann_ops_transformer` |
