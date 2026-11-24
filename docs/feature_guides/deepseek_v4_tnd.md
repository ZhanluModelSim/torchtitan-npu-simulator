# DeepSeek-V4 TND 适配

DeepSeek-V4（DSV4）在模型层使用文档打包（varlen）的 `[B, S, ...]` 接口；压缩 KV 在非 CP 下是 `[B, S//r, D]` 整行，CP 下是本地 chunk（内核边界 all-gather 成整行），仅在进入 AscendC 融合内核时转换为 TND。本文说明当前实现、启用方式和已知边界。

## 启用方式

融合路径要求完整的 Ascend NPU 运行环境，以及与 CANN 版本匹配的 `cann_ops_transformer`。推荐使用仓内脚本：

```bash
MODULE=torchtitan_npu_recipes.tests.models.deepseek_v4 CONFIG=deepseek_v4_flash_1p_2k NGPU=1 bash scripts/run_train.sh
```

DeepSeek-V4 配方通过 `config.override.imports` 显式选择算子，不读取 `USE_GOLDEN`。`scripts/run_train.sh` 只负责通用启动和原生参数透传；参考算子组合需在 Python 配方中定义。融合配置及 selector 选择以对应 recipe 为准，见[配方说明](../../torchtitan_npu_recipes/README.md)。

## 数据流

```text
模型张量 [B, S, ...]
        │
        ├─ positions
        │      └─ 文档边界。融合后端按压缩比构建 AscV4Metadata
        │
        ├─ Compressor：扁平 ratio 折叠 + softmax gate → 池化键流（上游 `Compressor`）
        │     按整行文档边界池化、不跨文档；CP 下由内核边界 all-gather 成整行（CP 当前 deferred）
        │
        ├─ CSA 的 indexer.inner_indexer → (topk_indices, topk_scores)
        │
        ├─ 参考 inner_attention（CompressedSparseAttention）
        │     按 topk_indices 调用 gather_attn，配置辅助损失时读取 topk_scores
        │
        └─ AscendC 融合内核（AscCompressedSparseInnerAttention）
             q/swa_k flatten → [T, N, D] / [T, 1, D]
             cmp_k 整行 / 本地 chunk（CP 由 S(1) -> R 声明式 all-gather）→ TND
             SMLA / SMLAG（SMLAG 反向把 teacher 写到 topk_scores 梯度）
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

## 压缩布局：DeepSeekV4Metadata

参考配方沿用 TorchTitan 的文档边界。融合配方的 `AscCompressedSparseInnerAttention.Config`
根据 `positions` 和 padding mask 构建 `AscV4Metadata`。其中 `documents` 保留投影需要的完整边界，
`q`、`swa_k`、`cmp_k` 是供 CANN 内核使用的 frame。相同压缩比的层共享每个 microbatch 的这份记录。
文档布局和 frame 定义见 [metadata.py](../../torchtitan_npu/models/deepseek_v4/metadata.py)。

- `ratio`：这个 entry 的压缩比（1、4、128 之一）；每个 consumer 都按它寻址。
- `q` / `swa_k` / `cmp_k`：V4.1 形态的三个 frame（`cu_seqlens` / `seqused` / `residual`）。CP 下 `q` 是本 rank 的文档 chunk，`swa_k` / `cmp_k` 是整行，`seqused` 是本 rank 的 reach；非 CP 下三者都退化为边界差。没有 `cu_seq_q` / `cu_seq_k` 别名：用哪个 frame 就读哪个网格——行网格 `swa_k.cu_seqlens`、查询网格 `q.cu_seqlens`、压缩网格 `cmp_k.cu_seqlens`，`ratio == 1` 的 key 网格就是 `swa_k.cu_seqlens`。
- `slots`：无选择器几何（ratio 128）的参考选择，构建 entry 时按整行一次算好；融合路径不读它。
- `position_k`：key 流自己的文档内位置（纯算术，零通信）。degree=1 时就是输入的 `positions`，CP 下是本 rank slab 的——消费者的 RoPE 读它，不自己推、也不需要知道流被搬过。
- `cp`（`CPDeepSeekMetadata`）：只有 CP 下存在，带 `restore`（`RestorePlan`：只有 `exchange`），即 KV 侧恢复回 slab 的移动；恢复后流自己的位置由 entry 的 `position_k` 承载。
- `batch_size`：查询 container 的真实 batch 维（构建 entry 时按 `positions.size(0)` 记一次，各 degree 不变；frame 只描述文档，carry 不了行数）。融合注意力在 CP=1 时支持 `local_batch_size > 1`，输入布局按批展平、输出保留 `[B, S]`；CP>1 暂不支持多 batch。参考选择器仍仅支持 MBS1。
- `aicpu_metadata`（`AscV4Metadata`）：只有融合 entry 才有，按 kernel 分组保存 `sparse_attn.fwd/bwd` 与 `lightning_indexer.fwd/bwd`；模型目录的 `DeepSeekV4Metadata` 不携带这些张量。
- `a3_workaround`（`AscV4Metadata`，`override/a3_workaround.py`）：arch22 的融合内核按 key frame 的**末端**对齐 mask 且不读 key 侧 `seqused`，所以 reach 只是文档前缀的 rank 要把两路 key 换成 reach 自己写成的帧并压行；两个流各带一个可微 `index_select`。arch35 不建这个字段。

压缩器没有自己的 metadata 字段：输入为 `[B, L, D]`，每行 token 数必须是 `ratio` 的整数倍。公共入口把整批展平成一条 `[1, B * L, D]` 打包流，完成压缩和 RoPE 后还原为 `[B, L // ratio, D]`。每行 positions 从 0 开始，`create_varlen_metadata_for_document` 据此把行首加入整批 `cu_seq_q`，使 ratio-4 的重叠不会跨行借 token。

V4 的 CP 当前尚未支持（见「当前边界」）。原 ratio-4 重叠 carry 的脚手架——`models/common/carry_exchange.py` 的 `CarryExchange` / `CarryStream`，以及本地 `Compressor` 内部的 restore + carry——已随 0.4.0 清理删除，CP 重做后按新契约补充。

一个 batch 里可以打包多条序列，所以 `B` 与序列条数无关。例如：

```text
positions  = [[0, 1, 2, 0, 1]]
B          = 1
cu_seqlens = [0, 3, 5]   # 一行里打包了 2 条序列
```

每条序列必须从位置 0 开始并连续递增。该约束保证压缩、因果 mask 和稀疏索引不会跨越序列边界。

## 压缩与索引规则

文档在进入行拼接前按 `lcm(compress_ratios)`（V4 为 `128`）对齐：dataloader 在 EOS 之后补 pad token
（label 为 `-100`，位置继续累加），因此每篇文档都是所有压缩比的整数倍，尾部余数恒为 0，压缩块不会跨文档。
对齐由 processor 的 `per_doc_alignment` 驱动：非 CP 时 registry 从 `compress_ratios` 推出 `lcm`（V4 为 `128`）；CP 时该值提升为 `lcm × cp_size`（V4 为 `128 × cp`，由 recipe/example 通过 `DSV4_PER_DOC_ALIGNMENT` 指定），与压缩器、内核看到的块网格同源。布局构建器把这条契约当作前提：`build_v4_metadata` 对非空且按
`lcm(compress_ratios)` 对齐的流，直接把这一行按 ratio 除出压缩网格（`cmp_k.cu_seqlens`），没有逐文档扫描、
也没有「尾部不足一块」的分支；破约的流（含空文档或长度不是对齐倍数）在构建处直接报错。压缩器本身同样是扁平的
`unflatten(1, (-1, ratio))` 折叠，只有在 ratio-4 重叠时才用 `cu_seqlens` 的段起点做掩码。CP 时文档按 `128 × cp` 对齐（每个 entry 再检查 `cp × max(ratio, 1)`）：查询按文档内部分 cp 片，切分后每段仍要是压缩块的整数倍。`doc_chunk_entry` 对未按该值对齐的流直接报错。CP=128 面向长文本（1M 序列），把短文档拼成长文本再切分不是目标场景，填充开销在该场景下可接受。

`max_context_length` 和每批 token 数必须是对齐值的倍数，processor 在构建时校验。
CPT 与 SFT 直接使用上游 `GrainDataLoader`，无需 loader override。
`SingleDatasetConfig` 连接 Hugging Face 数据源、processor 与过滤：CPT 使用 `AlignedTextProcessor`，
SFT 使用 `DeepSeekV4ChatProcessor`，两个 processor 共用 `align_text_sequence` 补齐文档。
聊天处理器保留官方 chat encoder 的 tool、developer、reminder 和 assistant-only 标签语义，
先做 next-token shift，再补无监督 pad。过长或无监督 token 的对话被过滤。
CPT 使用 `ConcatThenSplitPackingConfig`，SFT 使用 `FirstFitPackingConfig`。

SFT 入口选择 `deepseek_v4_flash_sft_4k_a3`；JSON/JSONL 路径直接修改配方中的 `data_files`，不读取资源环境变量。各节点配置启动地址和 rank 后运行：

```bash
MODULE=torchtitan_npu_recipes.models.deepseek_v4 CONFIG=deepseek_v4_flash_sft_4k_a3 NGPU=16 bash scripts/run_train_multinodes.sh
```

自定义 recipe 可按 registry 中的嵌套配置组合随机访问或流式 Hugging Face source、chat encoder 和
`num_packing_bins`（默认 `1`），并配置上游 loader 的 seed、shuffle/repeat、read options 和 prefetch。
数据采用 Grain 的排序、分片和打包状态，不能直接恢复旧
TorchData loader 的 checkpoint；迁移后需使用新的训练 checkpoint 目录，模型权重可单独初始化。

补出的 pad 只影响它自己的位置：压缩槽位只在所属块的**最后一个** token 上可见，而被 pad 补完的尾块最后一个位置必然是 pad，
因此真实 token 因果上看不到任何含 pad 的压缩键（ratio-4 的选择器同样只会从因果可达的整块里挑），pad 的 label 又是
`IGNORE_INDEX`，所以对齐不改变单篇文档真实 token 的前向与 loss。打包密度变化（pad 占掉 row 容量）才会改变同一 step
见到的文档集合——这是 ST 基线平移的原因，不是数值路径变化；`test_padding_a_document_tail_is_invisible_to_its_real_tokens`
把这条不变量钉在测试里（只改 pad 内容，真实位置的 pool、输出与 CE 逐位不变，pad 位置输出改变且 CE 贡献为 0）。

[`compressor.py`](../../torchtitan_npu/models/deepseek_v4/compressor.py) 只处理完整压缩块，且**绝不跨文档压缩**。文档先按 `lcm(compress_ratios)` 对齐（见上），因此对每篇文档 `compressed_len = L / r` 是整数、`residual = 0`；`cmp_k.residual`（内核的 `cmp_residual_k`）仍按内核契约传入，对齐后恒为 0。例如行边界 `[0, 128, 384]`（两条文档 128 与 256），`r = 4`：

```text
cmp_k.cu_seqlens = [0, 32, 96]  # 压缩块：32 + 64（= 行边界 // 4）
cmp_k.residual  = [0, 0]        # 对齐后恒 0；CP frame 的 residual 按 cp*ratio 对齐后也恒 0
块位置           = positions[..., ::4]  # 文档内相对位置，RoPE 直接用它
首块掩码         = cu_seq_seqlens[:-1] // 4  # 文档起始块没有前驱
```

CP 的 `cmp_k.residual = reach % ratio`；文档按 `cp × ratio` 对齐后 `reach` 是 ratio 的整数倍，因此 residual 恒 0，内核不会遇到半块前缀。

- `r=1`：不生成压缩 KV，仅执行原始 KV 的滑窗注意力。该几何没有压缩 frame（`cmp_k is None`），key 网格就是行网格 `swa_k.cu_seqlens`；融合 entry 仍生成对应的 `aicpu_metadata` 记录，融合算子使用 `has_cmp_kv=False`。
- `r=4`（CSA）：启用重叠压缩，由 `indexer.inner_indexer` 选择文档内的 TopK；前驱块必须属于同一文档（`overlap_valid`）。
- `r=128`（HCA）：生成连续压缩 KV，使用因果可见的压缩槽。

## 参考路径（默认内核）

CSA 层的 `Indexer` 投影出索引查询、键和权重，交给 `inner_indexer` 返回文档内的
`(topk_indices, topk_scores)`。`attention.inner_attention` 使用这些索引，调用上游
`CompressedSparseAttention` 同款 `gather_attn` 计算注意力。SWA 和 HCA 沿用各自的上游参考实现。

参考训练配方可以给 CSA 的 `inner_attention.aux_loss` 配置上游 `SparseIndexerLoss`。
此时 `inner_indexer` 同时返回可导的 student logits，辅助损失为 indexer 提供梯度。
参考 Muon 配方已启用这一配置。默认参考配方关闭辅助损失，TopK 索引本身只参与选槽。
参见 [Muon 配方](muon_optimizer.md)。

## AscendC 融合路径

[`sparse_attn/ascendc.py`](../../torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py) 中的 `AscCompressedSparseInnerAttention` 与 `indexer.inner_indexer` 按以下顺序协作。

1. 校验模型输入是 `AscV4Metadata`，读这一层的 `q` / `swa_k` / `cmp_k` frame。
2. query 转 TND；swa_k / cmp_k 是整行（CP 下由 wrapper 的 `S(1) -> R` 声明式 all-gather 得到），直接转 TND。
3. `indexer.inner_indexer` 的独立 override（`lightning_indexer.asc`）调用 `lightning_indexer` 生成序列内局部 TopK，返回 `topk_indices` 与 `topk_scores`。索引按文档内位置降序排列，真实分数按同一组槽位重排。索引是文档内位置，`-1` 表示无效槽，分数的梯度传递 teacher。
4. V4 和 V4.1 使用上游 `SparseIndexerLoss(mass_weighted=True)`。参考 attention 从前向的完整 softmax LSE 计算并累计 loss。融合 attention 在 SMLAG backward 取得 teacher，清掉 `-1` 槽位和 padding 行，调用上游 KL 计算并累加到 `AuxLoss.instance_acc`。它按 `coeff / aux_loss_denominator` 缩放 teacher，通过 `topk_scores` 的梯度直接传给 SLIKG。训练融合 indexer 时同时启用 `sparse_attn.asc` 和 `lightning_indexer.asc`。[融合 attention](../../torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py) · [融合 indexer](../../torchtitan_npu/override/deepseek_v4/lightning_indexer/ascendc.py)

`cann_ops_transformer` 包负责内核的算子注册（schema、meta/fake 实现），本仓只在融合路径内做前反向桥接，不再自建 `torch.library` 封装。

| 参数 | 来源 |
| --- | --- |
| `cu_seqlens_q` / `seqused_q` | `metadata.q`（CP 下是 chunk 前缀和；非 CP 是整行） |
| `cu_seqlens_ori_kv` / `seqused_ori_kv` | `metadata.swa_k`（CP 下整行，`seqused` 是 reach） |
| `cu_seqlens_cmp_kv` / `seqused_cmp_kv` / `cmp_residual_kv` | `metadata.cmp_k` |
| `layout_q` / `layout_kv` | 固定为 `TND` |
| 四个 `*_metadata` 张量 | metadata 的 `asc`（融合 backend 的 config 在 metadata 边界预计算；CP 下按 rank-local frame 重建） |

## Grain SFT 集成验证

`tests/integration_tests/deepseek_v4.py` 注册了 `dsv4_grain_sft`：单 NPU、debugmodel、参考算子、eager，检查两步训练完成。提供包含官方 `encoding_dsv4.py` 的 tokenizer 目录和至少一条编码后不超过 512 token、含 assistant 监督标签的 JSON/JSONL 样本：

```bash
DSV4_SFT_ASSETS=/path/to/DeepSeekV4_tokenizer \
DSV4_SFT_DATASET=/path/to/short_chat.jsonl \
python -m tests.integration_tests.run_tests /tmp/dsv4-grain-sft-results \
  --test_suite deepseek_v4 --test_name dsv4_grain_sft --ngpu 1 --no-parallel
```

输出目录必须为空。未提供两个环境变量时该 case 被禁用；不能把跳过视为验证通过。外部语料没有固定 golden loss，此 case 只检查完成性，不证明旧 loader 的数值等价或吞吐提升。CPU 单测使用固定编码结果验证 labels、padding、positions 和 Grain 状态恢复，不替代官方编码器与真实 NPU 训练验证。

## 当前边界

| 项目 | 当前状态 |
| --- | --- |
| `positions` | 必须提供 `[B, S]`，且每条序列从 0 连续递增 |
| Padding | varlen 流必须覆盖本 rank 的全部 `B * S` 个 token，metadata 不表达 padding token |
| Context Parallel | 当前 0.4.0 迁移尚未支持，不能通过修改启动参数启用 |
| Tensor Parallel | DSA 路径固定 TP=1（indexer score 对 head 维求和），TP 方案待 CP 之后另行设计 |
| 空压缩流 | doc-packing 场景要求每条序列至少产生一个完整压缩块；融合路径在 metadata 边界对 `ratio > 1` 且无完整块直接报错（AscendC CSA/HCA 不接受空 `cmp_kv`） |
| 运行环境 | 需要 `torch_npu`、CANN、HCCL、Ascend NPU，以及匹配的 `cann_ops_transformer` |
