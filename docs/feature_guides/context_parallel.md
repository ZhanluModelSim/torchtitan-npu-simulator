# DeepSeek-V4 系列上下文并行（CP）

> **0.4.0 配置迁移说明：** 下文保留历史实现说明与字段 CLI 示例，不能直接作为当前启动命令。训练工厂已迁至 `torchtitan_npu_recipes`；当前启动方式见[快速上手](../user-guides/quickstart.md)，专用配置与 golden 应按 [Python 配置组合指南](../developer_guides/python_recipes.md)定义工厂。功能支持以 0.4.0 升级计划为准。

DeepSeek-V4（DSV4）与 DeepSeek-V4.1（DSV4.1）共用同一套上下文并行语义：**查询按文档内部分成
`cp` 份，KV 与压缩 KV 保持整行**。融合内核在边界处把本地 slab 声明式 all-gather 成整行，再用
frame 的 `seqused = reach` 把可见范围限制在本 rank 的 chunk 之前。两栈共用
`torchtitan_npu/models/common/metadata.py` 的 `KernelFrame`，也共用同一份输入布局：
`torchtitan_npu/models/common/restore.py` 的静态 doc-chunk 布局由 **CP 切分本身**施加——`cp_shard`
在切之前按 `doc_chunk_permutation` 重排整行，所以每个 rank 拿到的输入就是本 rank 的 chunk，不需要
任何集合通信；每层只在需要"本 rank 窗口"的消费方内部做恢复（压缩器池化前、核心取 swa 键前，
同一个 `key_stream`），layer 里不再有 `x_k`。要说明次数：不池化层是一次（swa 键）；池化层是两次
（swa 键 + 主压缩器的 `x`）；池化且带 indexer 的几何是三次，因为 indexer 自己的压缩器对同一份 `x`
再恢复一次——这是「key 侧自己管移动」的代价，各压缩器不必知道别的模块已经恢复过它。ratio-4
重叠的 carry 由 `torchtitan_npu/models/common/carry_exchange.py` 的 `CarryExchange` 在调用点处理，
两栈的压缩器本身都不感知 CP。

本文是 V4 与 V4.1 的合并指南（原 V4 CP 指南的内容已并入）。

## 1. 共同契约

### 1.1 坐标

设一行有文档 `d`、长度 `L_d`，CP 度 `R = cp_size`，rank `r`，slab 宽度 `C = N / R`
（`N` 是整行 token 数），chunk 宽度 `Q_d = L_d / R`：

```text
rank r 的输入（doc-chunk）：   每篇文档的第 r 个 chunk，文档顺序（cp_shard 的置换结果）
key 侧 slab：              [r·C, (r+1)·C)，由消费方按需从上面的 chunk 行恢复（压缩器内部、核心之前的 swa 键）
查询帧 q：                 每篇文档的第 r 个 chunk（输入布局已放好，查询不移动）
滑窗 KV 帧 swa_k：         整行；seqused = reach_d = (r+1)·Q_d
压缩 KV 帧 cmp_k：         整行按 ratio 折叠；seqused = reach_d // ratio，residual = reach_d % ratio
positions：                 查询自己的文档内位置（输入自带，查询侧不再单列字段）
position_k：               key 侧 slab 自己的文档内位置（entry 顶层字段）
```

`q.cu_seqlens = 整行边界 // R`（每篇文档 chunk 的前缀和），`q.seqused = Q_d`；
`swa_k.cu_seqlens = 整行边界`，`cmp_k.cu_seqlens = 整行边界 // ratio`。因此
`seqused_ori_kv - seqused_q = r·Q_d` 正好是本 rank 在该文档内的 chunk 偏移，内核的滑窗与压缩
因果上限都锚在这个偏移上。

整行网格只由 frame 承载：行网格 = `swa_k.cu_seqlens`，查询网格 = `q.cu_seqlens`，压缩网格 =
`cmp_k.cu_seqlens`，ratio-1 的 key 网格 = `swa_k.cu_seqlens`。key 流的文档内位置是 entry 顶层的
`position_k`：degree=1 时就是输入的 `positions`，CP 下是本 rank slab 的。两栈都不再单列查询位置
字段——查询的行就是输入的行，它的位置由输入自带；V4 的 `DeepSeekV4Metadata` 没有
`cu_seq_q` / `cu_seq_k`，V4.1 的 `DeepSeekV41Metadata` 仍保留这两个，因为它的内核按这些名字调用。

### 1.2 plan 与 exchange

输入侧的移动就是 CP 切分的置换，只有一种：

- `doc_chunk_permutation(cu_seqlens, cp_size, seq_len)`（`seq_len` 是 slab 宽度 `C`）给出整行的
  全局 token 序：稳定地按「这一行属于哪篇文档的第几个 chunk」排序，于是置换后 rank `r` 落在
  `[r·C, (r+1)·C)` 的窗口正是「每篇文档的第 `r` 个 chunk，文档顺序」。
- 布局按框架自己的方式接进切分：`cp_shard` 认识 `doc_chunk` 这个名字，用 `DocChunkLoadBalancer`
  （`patches/torchtitan/distributed/context_parallel.py`）构造它，由 balancer 的
  `_generate_indices` 自己算置换并交给 `_context_parallel_shard`：先重排**复制在每 rank 的整行**，
  再均分切片，所以这一步零通信，反向是 `argsort` 的逆置换（autograd 自己处理）。调用方只把
  **整行的 positions** 交进去，不在外面预先算好任何下标——与 `_HeadTailLoadBalancer` 只差一个
  构造参数（它要长度，这里要文档边界）。
- `document_starts(positions)` 从位置的重启读出整行的文档边界（行号减位置，`unique_consecutive`，
  **全 device**）：切分先于任何 metadata 消费者发生，这是在那里拿到边界的唯一途径；它与 metadata
  的 `cu_seqlens` 是同一组数，由 `test_the_permutation_is_the_layout` 锁住。
- 布局里的"本 rank 的数"（`rank`、`cp_size`、slab 宽度）一律是 **python int**：rank 是运行的 host
  值，写成 device 标量只会让张量算术无法区分常量与本 rank，并把整条路径绑到单进程模拟上。
- `chunk_plan(cu_seqlens, cp_size, rank, seq_len)` 产出一份 `ExchangeMetadata`：
  - `send_index` / `send_split`：本 rank slab 里每一行要去的 peer（按 peer 连续、peer 内按槽位
    升序），前者是重排下标，后者是 `input_split_sizes`；
  - `recv_split`：本 rank 从每个 peer 收到的行数（`output_split_sizes`）。接收 buffer 已经是
    held-row 顺序，所以没有 scatter 下标、没有二次重排；每篇文档的 chunk 是一个全局连续区间、
    至多跨两个 slab，接收侧因此只需 split。
- `to_slab(tensor, plan, group=...)` 交换两边的 split 再按 `send_index` scatter，是 doc-chunk ->
  slab 的精确逆（置换的反向就是另一次 all_to_all）。
- `slab_positions(cu_seqlens, cp_size, rank, seq_len)` 给出本 rank slab 自己的文档内位置，纯 host
  计算；它填的就是 entry 的 `position_k`。
- `RestorePlan(exchange)` 只带「chunk -> slab 的交换」，`apply(x, group=...)` 就是 `to_slab`；
  恢复出来的流该在哪些位置做 RoPE，由 entry 的 `position_k` 说，不跟着 plan 走。

`to_slab` 在没有 plan（`cp = 1`）时是恒等，并且先返回、不碰进程组，所以同一条前向路径可以服务
两个 degree，调用点不需要分支。每个 geometry 的 rank-local entry 在 `doc_chunk_entry` /
`shard_asc_v41_metadata` 里各算一份相同的 plan，本层 key 侧直接读自己 entry 的 `restore`。

**host 同步的位置**：`ExchangeMetadata` 的 `send_split` / `recv_split` 是其中一处。集合通信按 python
list 接收 split，这是框架的签名；它们由 plan 推导，而 plan **每个 microbatch 每个 geometry 只建一次**
（不是每层），所以这部分同步是 O(microbatch × geometry)。同一频率上还有文档对齐校验：
`require_document_alignment` 在 entry 与 `chunk_index` 各读一次（V4.1 另有一次 `int(chunk.sum())` 的
一致性校验）。

每层内的路径上没有 D2H：ratio-4 的 carry ring 用 host rank 直接构造两条 split（`_ring_split`，每层
0 次同步），`document_starts`、`slab_positions` 与 carry 的边界裁剪都是 device 算术。arch22 的 key
压实每个 microbatch 同步一次（`nonzero` 的行数是 device 值，见 §4.2）。

### 1.3 frame 与 metadata

一行由三个 `KernelFrame` 描述：`q`、`swa_k`、`cmp_k`（不池化的几何没有 `cmp_k`）。CP 下由融合
entry 的 `prepare_cp_metadata` 用 rank-local chunk 重建这三个 frame 以及按它们算出的 AI_CPU
记录；参考 entry 的 hook 显式拒绝 CP（它只有一条连续文档流的契约）。frame 的归属两栈不同：

| 栈 | 参考 metadata | CP 重建 |
| --- | --- | --- |
| V4 | `DeepSeekV4Metadata` 自带 `q`/`swa_k`/`cmp_k`、`slots` 与 `cp: CPDeepSeekMetadata(restore)`（`build_v4_metadata` 用整行边界构造） | `AscV4Metadata.prepare_cp_metadata` → `doc_chunk_entry` → `build_aicpu_metadata` |
| V4.1 | `DeepSeekV41Metadata` 只有整行边界与位置；frame 由融合 entry `AscV41Metadata` 携带 | `AscV41Metadata.prepare_cp_metadata` → `shard_asc_v41_metadata` |

两栈的重建都从 `cp_shard_context()` 读 pre-shard `positions` 与 CP mesh，并要求
`--parallelism.context-parallel-load-balancer doc_chunk`：这些 frame 描述的就是 doc-chunk 布局，切分用
的布局与 frame 描述的不一致会静默错位，所以在 entry 里直接拒绝。V4 的 `doc_chunk_entry` 重算 frame
与 `restore`、`position_k`；V4.1 的 `shard_asc_v41_metadata` 同样重建这三样，只是入口不同。

## 2. 启用方式

```bash
# DeepSeek-V4.1，两个 rank；先设置 HF_ASSETS_PATH 指向 tokenizer。
MODULE=torchtitan_npu_recipes.models.deepseek_v4_1 CONFIG=deepseek_v4_1_stripped NGPU=2 \
  bash scripts/run_train.sh \
  --hf-assets-path "$HF_ASSETS_PATH" \
  --training.max-context-length 512 \
  --training.num-tokens-per-microbatch-per-dp-rank 512 \
  --training.num-tokens-per-train-step 512 \
  --training.steps 2 --training.disable-cuda-graphs --compile.no-enable \
  --dataloader.per-doc-alignment 4 \
  --parallelism.spmd-backend spmd_types \
  --parallelism.context-parallel-degree 2 \
  --parallelism.context-parallel-load-balancer doc_chunk \
  --parallelism.data-parallel-shard-degree 1 \
  --parallelism.data-parallel-replicate-degree 1 \
  --parallelism.expert-parallel-degree 1 \
  --parallelism.tensor-parallel-degree 1 \
  --parallelism.pipeline-parallel-degree 1 \
  --optimizer.name AdamW --checkpoint.no-enable \
  --override.imports torchtitan_npu.override.common.rope.workaround \
    torchtitan_npu.override.common.optimizer.virtual \
    torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc \
    torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc
```

DeepSeek-V4 的 Grain 配置将文档对齐值放在 dataset 的 `AlignedTextProcessor.Config.per_doc_alignment`；
启用 CP 时，需在 recipe 中将该值设为 `128 × CP` 的倍数。

V4.1 的 CP 配方是 `examples/deepseek_v4_1/deepseek_v4_1_flash_cpt_1m_a5.sh`：脚本默认
`CP=128`、`EP=128`、`--dataloader.per_doc_alignment 256`，启动时按 `EP` 校验拓扑（`WORLD_SIZE`
与 `384` 都要能被 `EP` 整除）；`CP` / `EP` 可用同名环境变量覆盖。

- `--parallelism.context-parallel-degree > 1` 启用 CP；必须配合
  `--parallelism.spmd-backend spmd_types`（声明式 `S(0) -> R` all-gather 的前提）。切分与 metadata
  重建之后，`preprocess_inputs` 统一施加输入 SPMD 标注，这一步只在 `spmd_types` 后端执行。
- 布局必须是 doc-chunk：`--parallelism.context-parallel-load-balancer doc_chunk`（上游默认 `headtail`
  会被融合 entry 的 `prepare_cp_metadata` 拒绝）。输入在切之前按该置换重排，零通信；示例脚本已按
  该值传入。两栈的 metadata 类都声明了 `cp_layout = DOC_CHUNK_LAYOUT`，所以留空（`None`，即"不做
  load balance"）时也解析成 doc-chunk——一个 rank-local entry 只对一种行排列成立，让 metadata 说出
  它是哪一种，比让用户猜更安全。
- 每篇文档按 `lcm(compress_ratios) × cp` 填充：V4 为 `128 × cp`（CP=2 即 256），V4.1 为其
  `compress_ratios` 的 lcm × cp（当前 lcm 为 2，1M 配方用 256 = 2 × 128）。这条约束是**刻意严格**
  的：CP 把每篇文档切成 `cp` 个等长 chunk，除了 `doc % (cp × ratio) == 0`，**每个 chunk 自身**也
  必须是 ratio 的整数倍，否则该 rank 的压缩前缀会停在组中间，而部分压缩块不在已验证的内核契约内
  ——因此不放松。例如 cp=2、ratio=128、doc=128 会得到 64-token chunk（不是 128 的倍数）并被明确
  拒绝。数据加载器配置的文档对齐值必须是对应 lcm × cp 的倍数：模型配置更新时由
  `require_dataloader_alignment` 检查，每个 geometry 的 entry 再由
  `require_document_alignment(cu_seqlens, cp_size=cp, ratio=r)` 检查（V4 在 `doc_chunk_entry`，V4.1
  在 `shard_asc_v41_metadata`），V4.1 使用 `--dataloader.per-doc-alignment`；V4 Grain 使用 dataset processor 的配置。
- 需要完整 NPU 环境（`torch_npu`、CANN、HCCL、匹配的 `cann_ops_transformer`）。只有 AscendC 融合
  路径支持 CP；模型目录的参考 metadata hook 显式拒绝 CP，V4.1 的 `update_from_config` 还会要求
  同时请求 `override.deepseek_v4_1.sparse_attn.asc` 与
  `override.deepseek_v4_1.lightning_indexer.asc`。
- 量化后端（V4 的 `QuantV4Metadata`、V4.1 的 `_V41QuantAttentionMetadata`）目前不重建 rank-local
  metadata，`prepare_cp_metadata` 直接拒绝：量化路径只能跑 `context-parallel-degree 1`。
- 模型输入使用 folded-token 布局：token 与位置为 `[T]`，隐藏状态为 `[T, D]`。
  不同文档通过 positions 重置及 metadata 边界区分；DSA 路径的 TP 限制以模型配置为准。

## 3. 两栈差异总览

| 维度 | DeepSeek-V4 | DeepSeek-V4.1 |
| --- | --- | --- |
| 输入布局 | 同一份：`cp_shard` 的 doc-chunk 置换，tokens / labels / positions / MTP 深度一次到位 | 同左 |
| 压缩器与 CP | ratio-4 重叠块需要前驱 rank 的 carry：`CarryExchange`（与压缩器同级的模块）做交换/prepend，调用点压缩后再裁剪 | 压缩器 CP-agnostic：文档按 `cp × ratio` 对齐后，本 rank 的 slab 起点就是组边界，本地池化即可；无 carry |
| 压缩流取得方式 | 本地流（含 carry）池化后，内核边界 `S(0) -> R` gather 成整行 | 本地 slab 池化后同样在核心边界 `S(0) -> R` gather |
| 内核锚定 | arch35 读 `seqused_q` / `seqused_ori_kv` / `seqused_cmp_kv`，窗口与压缩因果上限直接锚在 reach 上；arch22 的内核不读 key 侧 `seqused`、把 mask 右对齐到 frame 末尾，因此非末尾 rank 的两路 key 流要压成 rank-local 前缀 | arch22 同样对 key 前缀做整理；arch35 读取 `seqused` |
| rank-local key 压缩（仅 arch22） | `A3KeyWorkaround`（`override/a3_workaround.py`）：`swa_k_gather` / `cmp_k_gather` 两个可微 `index_select`，由 `build_aicpu_metadata` 在 `if not _IS_A5:` 里加在 `AscV4Metadata` 上 | 同样使用 `A3KeyWorkaround` |
| frame 归属 | 模型 metadata（`DeepSeekV4Metadata`，含 `cp: CPDeepSeekMetadata(restore)`） | 融合 entry（`AscV41Metadata`） |
| indexer teacher 归一化 | `_LightningIndexerTND.backward` 除以整行 token 数 `swa_k.cu_seqlens[-1]` | `num_global_queries`（本 step 全部 query 数，按 DP 求和） |
| 参考 metadata | `DeepSeekV4Metadata.prepare_cp_metadata` 拒绝 | `DeepSeekV41Metadata.prepare_cp_metadata` 拒绝 |
| 量化 metadata | `QuantV4Metadata` 拒绝 | `_V41QuantAttentionMetadata` 拒绝 |

## 4. DeepSeek-V4

### 4.1 数据流

```text
Decoder.preprocess_inputs（每个 microbatch 一次）
  整行 [N] --cp_shard(doc_chunk 置换)--> 本地 chunk 行 [C]
    tokens / labels / positions 随同一次切分到位（零通信）
    随后按 spmd_backend == spmd_types 施加 annotate_input_spmd_types
DeepSeekV4Model.preprocess_inputs 只额外准备 MTP 深度，并让它们走同一次 cp_shard

Attention.forward（每层）
  x（doc-chunk 行）
    ├─ wq_a / q_norm / wq_b / q_head_norm / rope(positions)   [查询不移动]
    ├─ swa：wkv / kv_norm / rope(positions)（chunk 行）      [CP=S(0)]
    │    └─ key_stream（RestorePlan.apply）-> 本 rank slab，供核心边界 gather
    ├─ compressor(x)（见 4.3：内部 restore + ratio-4 carry）  [CP=S(0)]
    └─ indexer：idx_q / idx_w 在 chunk 行上投影；idx_k 由它自己的压缩器产出
  wo_a / wo_b 输出留在 doc-chunk 行（不 unpermute）

inner_attention / lightning_indexer 的 ShardingConfig
  swa_k, cmp_k（核心）与 idx_k（selector）: cp S(0) -> R（内核边界 all-gather 整行）
  q, idx_q, idx_w:      保持本 rank 的 chunk（attention 直接调用这两个模块）
```

`preprocess_inputs` 的 CP 切分就是本布局：基类 seam 用 `doc_chunk_permutation` 生成的置换
（`--parallelism.context-parallel-load-balancer doc_chunk`）先重排整行再均分，tokens、labels 与
positions 一次到位；`DeepSeekV4Model.preprocess_inputs` 只把 MTP 的每个深度也用同一次 `cp_shard`
切好，保证预测与被预测的行一起换位。切完由 `annotate_input_spmd_types` 重新施加输入 SPMD 标注。
没有 CP（或 `cp_size == 1`）时整段跳过，batch 原样进入前向。

`Attention.forward` 收到的是 doc-chunk 行，查询侧直接投影、在随行搬来的 `positions` 上做 RoPE。key
侧同样在这批行上投影、旋转、池化——每一步都是逐 token 的，因此与移动可交换——需要"本 rank 窗口"的
消费方各自恢复：压缩器在池化前恢复（`attention_metadata.key_stream`，degree=1 即恒等），核心在取
swa 键前恢复（同一函数，窄张量）。layers 里没有 `x_k`：移动只发生在拥有它的模块内部。输出仍留在
doc-chunk 行：`wo_b` 是逐 token 投影，残差、MoE 与 loss 都按同一布局计算，loss 的 labels 也是同一份
plan 搬过的。

`AscV4Metadata` 的 `q`/`swa_k`/`cmp_k` frame 被 SMLA/SMLAG 与独立的 LI override（前向
`lightning_indexer`、反向 SLIKG）直接读取；V4 的 metadata 只有 `ratio`、三个 frame、`position_k`、
`slots` 与 `cp`，没有 `cu_seq_q`/`cu_seq_k`，也没有单独的重切函数（rank-local entry 由
`doc_chunk_entry` 产出）。不再有 `_assemble_tnd`、`cmp_k_global_gather_indices`、`compressed_rows`、
`first_indices` 或容器网格。

### 4.2 arch22 的 key 压缩

压缩流与滑窗流在 gather 后都是整行，其 `seqused` 是本 rank 的 reach。arch35 的内核直接读它；arch22
的内核（`SparseFlashMla` / `LightningIndexer` 系列）不读 key 侧 `seqused`，而是把 mask 右对齐到
**它拿到的 key frame 末尾**，SMLA/SMLAG 的压缩侧还要求 key 张量、网格与 `sparse_indices` 在同一
坐标 frame 里。因此 `build_aicpu_metadata`（`override/deepseek_v4/sparse_attn/ascendc.py`）在
`if not _IS_A5:` 里把两路 key 换成"reach 自己写成的帧"，并把压行下标存在 entry 的
`a3_workaround`（`override/a3_workaround.py`，与 `aicpu_metadata` 并列）上：

- `a3_key_frame(cu_seqlens, seqused, residual)`：把 reach 写成帧（`cu_seqlens = cumsum(reach)`，每篇
  文档的前缀即一段），`seqused` 保持 reach，ratio > 1 时 `residual` 为空则写全零（arch22 的 tiling
  不接受空残块）；
- `a3_key_gather(cu_seqlens, seqused)`：在 device 上算出把整行压到该帧的 `index_select` 下标
  （`repeat_interleave` + 比较 + `nonzero`；`nonzero` 的行数是 device 值，每个 microbatch 每条流
  一次同步，代价见 §7 的性能注）；
- 两个 gather 都可微：反向经 scatter 回整行，再由 `S(0) -> R` 的 reduce-scatter 回到本 rank 的
  chunk；`cp=1` 与最后一个 rank 的 reach 就是整篇文档，此时两个 gather 都是 `None`。

entry 的 `swa_k` / `cmp_k` 就是内核帧，forward / backward / selector 各取一个 frame 与一个 gather，
只有一处 `if not _IS_A5:` 用它压行；行自身的网格仍在 `q.cu_seqlens` / `restore` 里，与 arch35 一致。

**arch22 + CP 下不编译**：压实的行数是 device 值（`nonzero` 的行数），它进入
`cann_ops_transformer` 的 meta 函数后变成 unbacked 符号，图在 `Could not guard on data-dependent
expression` 处失败（CP=1 时没有压实，同一份代码正常编译）。这条路径也没有地方断图：torchtitan 每个
block 用 `fullgraph=True` 编译，而全量 activation checkpoint 会把 block 放进 `checkpoint` 这个
higher-order operator 里 trace，HOP 内部不允许图断点（`Skip calling torch.compiler.disable()d
function`）。

因此**栈自己在 `parallelize` 里做这个判断**（`models/common/compile_policy.py`：`_IS_A5 or cp <= 1`
才编译），判为不可编译时打一条告警并把 `compile_config.enable` 关掉，让训练照常跑 eager：
`--compile.enable` 因此可以留在同一份配方里——A5 上生效，arch22 + CP 上退化为 eager 而不是报错。
判断放在模型的 parallelize 而不是 patch 里，是因为它只取决于「哪个 arch × 多少度」，与上游
`apply_compile` 的接口无关。

**代价要写在这里**：eager 是逐算子直接跑，吞吐低于编译后的水平。原本在 A3 上开着 `--compile.enable`
跑 CP 的作业会因此变慢，日志里只有一条 warning；评估性能时按「arch22 + CP>1 不编译」这一条来读，
不要把 CP=1 的编译吞吐当成同一份配方在两个 degree 上的预期。

### 4.3 压缩器：restore / carry / prepend / trim

压缩器拿到的是**布局原样产出的流**（degree=1 是整行，CP 下是本 rank 的 chunk 行）、key 流自己的
`positions`（entry 的 `position_k`）以及**整行**的 `cu_seqlens`。池化本身仍是扁平的 `ratio` 折叠，
重叠的左半块在每个 segment 起点被 mask（score `-inf`、kv `0`）；CP 的两件额外工作在模块内部完成：

- **restore**：`attention_metadata.key_stream(x)`（degree=1 即恒等），把 chunk 行换回本 rank 的
  slab，使池化的块网格与整行一致；
- **carry**：ratio-4 的前驱块由 `CarryExchange`
  （`torchtitan_npu/models/common/carry_exchange.py`）交换、prepend、池化后由压缩器自己裁掉。

调用方只交 `x` 与 `position_k`，看不到这两步。CP 的 KV all-gather 仍由 `sharding.py` 在融合内核
边界声明式完成。

ratio-4 的**重叠前驱**由 `CarryExchange` 处理：

1. 每篇文档长度按 `cp × ratio` 对齐，slab 边界落在压缩网格上；但 ratio-4 的块首池化需要前一块的行。
2. rank `r > 0` 把本 rank 最后 `ratio` 个 token 通过一次 `all_to_all_single` 发给 rank `r+1`，
   同时接收 rank `r-1` 的。**位置不参与交换**：它是前驱 slab 末块的位置，由整行边界直接算出
   （`slab_positions`），而且 carry 块一定会被裁掉、overlap 读的是 RoPE 之前的投影态，数值上惰性。
3. 把收到的 block prepend 到本地 slab 前面；`CarryExchange` 用「整行边界裁剪到增强窗口
   `[slab_start-ratio, slab_end)` 再 rebase」构造增强 `cu_seqlens`：若 slab 起点在某文档内部，
   窗口里没有落在接缝上的边界，carry 自然与该文档的第一段合并；若起点正好是文档边界，整行边界
   被保留，carry 是前一篇文档的独立 segment。两种情况都只靠「mask 每个 segment 起点」这一条规则，
   carry 自己的输出块会被裁掉，因此不需要特判。
4. 用增强后的 `cu_seqlens` 调压缩器（融合实现 `coff=2`），输出比本地块多一块；丢掉第一块
   （carry block），保留本地 slab 的 `C/ratio` 块。
5. RoPE 使用 `positions[..., ::ratio]`（保留块的文档内位置）。
6. `CarryExchange` 只产出增强流（`CarryStream`：`x` / `positions` / `cu_seqlens` / `trim`）；
   融合实现的 `_state_inputs` 在内部按增强流的形状符号构造全 0 的 kernel tiling 表
   （`state_block_table` 不在模型层接口上）。ratio=128 无重叠，不做 carry。

rank 0 只发送、不接收；最后一个 rank 只接收、不发送；其余 rank 两者都做。所有 rank 都进入同两次
集合通信（空 split 也参与），避免死锁。

调用点在每个 degree 都传 `metadata.swa_k.cu_seqlens`（整行边界）；`cp = 1` 时 `CarryExchange`
直接返回同一份流，压缩器拿到的是整行边界。参考实现与融合实现共用同一接口
`compressor(x, positions, *, cu_seqlens=None)`（`cu_seqlens` 是 keyword-only：带 sharding
config 的模块走框架的位置参数契约，位置参数个数必须与签名一致）。
输入 `x` 为 folded-token 形状 `[T, D]`，positions 为 `[T]`，每篇文档的 token 数必须按
压缩比对齐。内部压缩实现使用单行容器，输出再恢复为 `[T / ratio, D]`；文档边界防止 ratio-4
的重叠跨文档。

### 4.4 参考路径

`DeepSeekV4Metadata.prepare_cp_metadata` 明确拒绝 CP：参考内核（Attention Gym 的 `gather_attn`）
只有一条连续文档流的契约。参考**压缩器**本身支持 CP（carry 在 `CarryExchange` 里，与压缩器
实现无关），所以 debug model（参考压缩器 + 融合注意力）也能跑 CP。V4 的数值验收在融合路径的
CP=1/CP=2 之间做。

## 5. DeepSeek-V4.1

### 5.1 数据流

```text
Attention.forward
  x (本 rank 的 chunk 行，输入布局已放好，查询不移动)
    ├─ q 投影 / q_norm / rope(positions)
    ├─ swa_k 投影 + rope(positions)（chunk 行）             [CP=S(0)]
    │    └─ key_stream（RestorePlan.apply）-> 本 rank slab，供核心边界 gather
    ├─ compressor(x, position_k)：内部 restore，source 层池化，无 carry   [CP=S(0)]
    └─ indexer：idx_q / weights 在 chunk 行上投影，idx_k 由压缩器的 slab latent 投影
  输出留在 chunk 行：rope(positions, inverse) -> wo_a -> wo_b

Sparse core / selector（sharding 声明）
  swa_k, cmp_k, idx_k: cp S(0) -> R（整行）
  q, idx_q, weights:   保持本 rank 的 chunk
  topk_indices:       按 token 维切分；sink: 不按 CP 切分
```

V4.1 的参考 metadata（`DeepSeekV41Metadata`）只带整行边界与 `position_k`（degree=1 时等于输入的
`positions`）；CP 的 frame 与记录由融合 entry 重建，消费者（`AscV41SparseAttention`、`AscSelector`）
直接读 `attention_metadata.q` / `.swa_k` / `.cmp_k` 与 `aicpu_metadata`。

### 5.2 rank-local frame 重建

`AscV41Metadata.prepare_cp_metadata`（`torchtitan_npu/override/deepseek_v4_1/sparse_attn/ascendc.py`）
从 `cp_shard_context()` 取 pre-shard `positions` 与 mesh，调用 `shard_asc_v41_metadata`：

```text
seqused = diff(cu_seq_q)
alignment = cp_size * max(ratio, 1)          # 每篇文档必须对齐
chunk = seqused // cp_size                   # Q_d
reach = (cp_rank + 1) * chunk                # K_d
exchange = chunk_plan(cu_seq_q, cp_size, cp_rank, slab_len)
q       = Frame(cu_seq_q // cp_size, chunk)
swa_k   = Frame(cu_seq_q, reach)
cmp_k   = Frame(cu_seq_k, reach // ratio, residual=reach % ratio)   # ratio > 0
position_k = slab_positions(cu_seq_q, cp_size, cp_rank, seq_len)
```

其中 `exchange` 的移动表来自 `restore.chunk_plan`（与 V4 是同一份实现），`position_k` 来自
`restore.slab_positions`（纯 device 算术，与本 rank slab 真实所在的行同一组数；degree=1 时它就是
输入的 `positions`）。

并检查 `sum(chunk) == slab_len`（边界描述的必须是调用方切出来的那条行）。frame 变
则记录必须重算：entry 用 `build_aicpu_metadata(local, geometry=...)`（读 entry 自己的
`q` / `swa_k` / `cmp_k`）重建 `sparse_attn` 与 `lightning_indexer` 两组 `*_metadata`；
`reach` 这类"本 rank 的数"由 python int 的 `cp_rank` 参与张量运算（rank 是 host 值，见 §1.2）。

V4.1 的 `compress_ratio` 可以是 0（window-only，没有任何压缩 frame）、1（整分辨率第二路 KV）
或 2（压缩共享 KV）；`cmp_k` 只在 `ratio > 0` 时存在，`ratio == 1` 的 `residual` 为 `None`
（轴没有折叠，内核对非零值报错）。

### 5.3 压缩器与 sharding

V4.1 的 `Compressor`（`torchtitan_npu/models/deepseek_v4_1/compressor.py`）没有 CP 分支：source
layer 在本地 slab 上做 `ratio` 折叠（ratio==1 是普通投影），因为文档已按 `cp × ratio` 对齐，
本 rank 的 slab 起点就是组边界，`C/ratio` 是整数块。压缩输出（`[C/ratio, D]`）在 sparse core
的边界被声明式 `cp: S(0) -> R` gather 成整行；`cmp_k.seqused = reach // ratio` 让内核只读本 rank
之前的块。ratio-4 那种跨 slab 的 carry 在 V4.1 不存在，因此没有 `CarryExchange`。

`torchtitan_npu/models/deepseek_v4_1/sharding.py` 声明 CP 的输入/输出放置：

- sparse core：`q` 的 CP token 维保持 `S(0)`，TP head 维为 `S(1)`（本 rank chunk），`swa_k` / `cmp_k` 在消费侧为整行
  （CP 输入为 `S(0)`，消费侧为 `R`），
  `topk_indices` 按 token 维切分，`attn_sink` 不按 CP 切分；
- selector：`idx_q` / `weights` replicated，`idx_k` 做同样的 `S(0) -> R`（选择器用整行 index keys
  给本 rank 的 chunk 打分）。

### 5.4 indexer teacher 的归一化

V4.1 的 selector 在反向把 teacher 除以 `Selector.num_global_queries`：本 step 全部 accumulation
microbatch 的 query token 数（在 CP 切分之前统计，DP 维再求和）。V4 的融合 selector 反向则除以
`metadata.swa_k.cu_seqlens[-1]`（整行 token 数，即所有 rank 的 chunk 之和），两者是同一个量的两种
来源，都是为了避免 CP/DP 下参数梯度被 rank-local 分母放大。

### 5.5 拒绝 CP 的路径

`DeepSeekV41Metadata.prepare_cp_metadata` 拒绝参考内核；`_V41QuantAttentionMetadata`
（`torchtitan_npu/extensions/adaptor/torchao/converter.py`）拒绝量化 attention。两者都在任何层
执行之前报错，不会静默地拿着全局 metadata 跑 rank-local 行。

Engram 不在这一列。它的 token history 要的是每行**在本文档内**的前驱，doc-chunk 下这些前驱不是
全局行的一段连续区间（`[lo - H, lo)` 是别的文档的尾巴），一个窗口也说不出每行各自的前驱。所以
`DeepSeekV41Model.preprocess_inputs` 在切分之前把整行的前驱算成 `[C, H]`（第 `d - 1` 项是这一行
往前第 `d` 个 token），`doc_chunk_permutation` 与 shard 用的是同一个置换，因此不需要任何跨 rank
通信；文档起点那几行没有前驱，仍然由 entry 的 `positions`（`positions >= distance`）掩掉。engram 的
`_shift_tokens` 因此有两种 history 契约：`[H]`（连续切分给的一个窗口）与 `[T, H]`（按 token）。
按行那一支直接取 `token_history[:, distance - 1]` 这一片，**不**先分配一个填满 pad 的缓冲再整片写回：
整片写回读起来像一次拷贝，但在分片的派发下写进了错的那一片（A5 单卡 cp=2 实测里 rank 0 有 970/1024
行拿到了 rank 1 的 token，step 1 的 `loss` 从 11.94468 变到 11.94288，而两个 degree 各自仍是可复现的）。
`deepseek_v4_1_stripped` 现在挂 Engram（`_stripped` 结尾走 `_attach_engram`，只挂一层
`layer_ids=(1,)`），所以这条取数约束由单卡对拍与 910 的用例共同覆盖。

## 6. 组件

| 文件 | 职责 |
| --- | --- |
| `torchtitan_npu/models/common/restore.py` | `ExchangeMetadata` / `RestorePlan` / `chunk_index` / `chunk_plan` / `doc_chunk_permutation` / `document_starts` / `to_slab` / `slab_positions`：两栈共用的静态 doc-chunk 布局 |
| `torchtitan_npu/patches/torchtitan/distributed/context_parallel.py` | `DocChunkLoadBalancer`：`cp_shard` 内部按整行边界构造它，`_generate_indices` 自己算置换（切之前重排，零通信） |
| `torchtitan_npu/models/common/compile_policy.py` | `skip_compile_under_cp`：arch22 + CP 时把编译关掉并告警（见 §4.2） |
| `torchtitan_npu/models/common/metadata.py` | `KernelFrame`（两栈共用的 frame 原语） |
| `torchtitan_npu/models/common/packing.py` | `require_document_alignment` / `require_per_doc_alignment`：文档长度与 `per_doc_alignment` 的严格对齐检查 |
| `torchtitan_npu/models/deepseek_v4/model.py` | `DeepSeekV4Model.preprocess_inputs`：MTP 深度与 batch 走同一次 `cp_shard`（布局由 seam 施加，模型不再搬 batch） |
| `torchtitan_npu/models/deepseek_v4/metadata.py` | `DeepSeekV4Metadata(ratio, q, swa_k, cmp_k, slots, cp)` / `CPDeepSeekMetadata` / `build_v4_metadata` / `doc_chunk_entry` |
| `torchtitan_npu/models/deepseek_v4/attention.py` | 查询与 key 都在 doc-chunk 行上投影；swa 键在核心前用 `key_stream` 恢复、压缩器内部恢复+carry、selector + 内层核心直调（无 wrapper）、输出不 unpermute；参考内核读 `q` / `cmp_k` frame 与预计算的 HCA 选择 |
| `torchtitan_npu/models/common/carry_exchange.py` | `CarryExchange` / `CarryStream`：ratio-4 carry 的 ring（只换 tokens；位置由 `slab_positions` 算出）、增强 `cu_seqlens` 与 `trim`；cp=1 为 no-op |
| `torchtitan_npu/models/deepseek_v4/compressor.py` | 压缩器（CP/非 CP 同一实例）：`forward` 用 `attention_metadata.key_stream` 恢复窗口、ratio-4 时内部 carry 并裁掉自己那块；`Indexer` 复用它，`idx_q` 在 chunk 行、`idx_k` 是行内的池化块 |
| `torchtitan_npu/models/deepseek_v4/sharding.py` | `swa_k` / `cmp_k` / `idx_k` 的 `cp: S(0) -> R` 声明 |
| `torchtitan_npu/override/deepseek_v4/metadata.py` | `AicpuMetadata` / `AscV4Metadata`（`aicpu_metadata` + `a3_workaround`）/ `prepare_cp_metadata`（`doc_chunk_entry` + `build_aicpu_metadata`） |
| `torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py` | `build_aicpu_metadata`（按 entry 的 frame 重建记录，arch22 上建 key 压缩）、SMLA/SMLAG 的 frame 参数 |
| `torchtitan_npu/override/deepseek_v4/lightning_indexer/ascendc.py` | LI 前向 / SLIKG 反向（选择器自己的 frame、arch22 的压缩池、teacher 边，分母为 `swa_k.cu_seqlens[-1]`） |
| `torchtitan_npu/override/deepseek_v4/lightning_indexer/pypto.py` | PyPTO 版 selector：独立实现同一契约（输入准备、排序、teacher 边），只替换 LI/SLIKG kernel 对 |
| `torchtitan_npu/override/deepseek_v4/compressor/ascendc.py` | CANN 融合压缩器（无 carry、无 CP 分支；`_state_inputs` 按 stream 形状内部构造零 tiling 表） |
| `torchtitan_npu/models/deepseek_v4_1/attention.py` | V4.1 查询保持 doc-chunk 布局，key 消费方按需恢复 slab |
| `torchtitan_npu/models/deepseek_v4_1/compressor.py` | V4.1 本地池化压缩器（CP-agnostic，无 carry） |
| `torchtitan_npu/models/deepseek_v4_1/sharding.py` | V4.1 sparse core / selector 的 `S(0) -> R` 声明 |
| `torchtitan_npu/models/deepseek_v4_1/model.py` | `DeepSeekV41Metadata` 与 `build_v41_frames`（整行边界与三帧）；`prepare_cp_metadata` 拒绝 CP；`preprocess_inputs` 按行构造 Engram 的 token history；teacher 成对启用检查 |
| `torchtitan_npu/override/deepseek_v4_1/sparse_attn/ascendc.py` | `AscV41Metadata` / `shard_asc_v41_metadata` / `build_fused_metadata` |
| `torchtitan_npu/override/deepseek_v4_1/lightning_indexer/ascendc.py` | V4.1 BF16 selector（整行 index keys × 本 rank chunk） |
| `torchtitan_npu/override/deepseek_v4_1/metadata.py` | V4.1 的 `AicpuGeometry` / `AicpuMetadata`：按 kernel 分组的 AI_CPU 记录结构 |

## 7. 验证

以下跨 CP degree 的数值记录来自原有实现的历史验证，不能直接作为当前 folded-token
迁移版本的数值结论。迁移版本需要重新按相同 checkpoint、输入和确定性条件验证。

### 7.1 层次与判据

CP 的验收分三层，越往下越严格：

1. **CPU 契约**：布局的纯函数（`chunk_plan` 的 send/recv split 在 peers 之间一致、移动复现
   doc-chunk 行集合、`to_slab` 与切分置换互为逆、跨两个 slab 的 chunk、`slab_positions`、
   未对齐文档被拒绝、`cp=1` 恒等）、`document_starts` 与 metadata 的 `cu_seqlens` 一致、frame 坐标
   与 `seqused` 锚点、rank-local 前缀池、metadata 重建与拒绝路径。这一层不看数值，只看"谁被放到了
   哪"。
2. **多 rank 集合通信**（`tests/unit_tests/models/common/test_restore_distributed.py`）：两个 gloo
   rank 上真实跑 `all_to_all_single`，验移动、逆向、`position_k` 以及 `to_slab` 的梯度等于反向
   交换。
3. **单机数值**（A5 单卡 + `--comm.mode local_tensor` 模拟两个 rank）：CP=1 与 CP=2 两个运行比
   logits、`loss` 与梯度。

判据是分层的，因为跨 rank 归约改变了加法树：

| 量 | 判据 | 原因 |
| --- | --- | --- |
| step 1 的输出 logits | **逐位** | 计算图逐算子相同，只是行顺序与按 rank 分段 |
| step 1 的 `loss` | **逐位** | 同上（`loss` 已按 token 加权，跨 rank 求和顺序不同时仍应逐位，因为每个 rank 的 `loss` 是全体 query 的确定性函数） |
| step 1 的梯度 | 相对误差 ~1e-5 | 每个参数分片的梯度是 `cp` 个 partial 之和："两份 fp32 partial 相加"与"单 rank 逐行累加"不是同一棵加法树，而 partial 的量级远大于抵消后的总和 |
| step 2 的 `loss` / `grad_norm` | 同上 | 由上面的梯度差传播而来 |

因此判断**一次改动是否中性**，看 step 1 的 logits 与 `loss`（逐位）；梯度与 step 2 只能给量级
结论，不能作为逐位判据。

### 7.2 取数要点（每一条都曾把一个对比做废）

- **权重必须对齐**：两侧加载同一份 checkpoint（`--checkpoint.enable` + `--checkpoint.load-only`
  + `--checkpoint.folder=<绝对目录>`，且日志里要看到 `Loading the checkpoint from ...`）。只给
  `load_only` 不给 `enable` 时 checkpoint 管理器整个不启用，日志里没有加载行——"看起来对齐"的对比
  是无效的。
- **同一 seed 重新初始化不算对齐**：模型初始化依赖 world size（分片张量的抽样位置不同），world=1
  与 world=2 拿到的是不同权重。
- **单卡模拟器还要手工镜像权重**：模拟器按 Philox 状态的*元素偏移*摆放每个 rank，而 NPU 生成器
  每次 kernel 启动只把 offset 推进 4（与张量大小无关），所以同一 seed 下 `Shard(0)` 的两个 rank
  不是单 rank 那次抽样的并集（`Replicate` 是）。做法是由一侧导出完整张量、按 shard 规则切分后写回。
- **`NGPU` 必须等于 `CP`**：CP=1 的对照必须 `NGPU=1`，否则 `cp=1` 会落到 `dp_shard=2`，两侧看的
  不是同一条样本、也不是同一份权重。
- **文档对齐值两侧相同**：按 `lcm × CP` 配置，跨 degree 对比时要显式
  写同一个值（否则文档边界不同，比的是两份数据）。
- **A5 上需要 `torch_npu.npu.set_deterministic_level(3)`，两侧同时打开**：不加时同一 degree 重复跑
  是逐位可复现的，但不同 degree 会落到不同的 kernel 实现上（形状相关的实现选择），logits 差异可达
  1e-1~1e0；打开后才有逐位结论。level 3 也会改变前向数值本身，所以只能做**同 level** 的对比。
  arch22（910）没有这一档，逐位判据只在 A5 上取。
- **logits 的取法**：把模型输出按 rank dump，用 entry 的 `restore.exchange` 把 doc-chunk 行还原成
  slab 行；CP=2 的各 rank 按 rank 顺序拼起来应与 CP=1 逐位相同。
- **一次改动是否中性，用同一棵树重跑作分母**：先确认"改动前后、同一配置"逐位相同，再谈跨 degree。

### 7.3 910（arch22）上的补充约定

- 不开 `set_deterministic_level`：arch22 上没有这一档。V4 在 arch22 上跨 degree 的前向**不**逐位
  （既有的 kernel/归约选择差异，改动前就存在）；V4.1 在同配置下实测跨 degree 仍逐位，但那是这一
  套内核在该形态下的结果，不能当成 arch22 的普遍结论。因此 910 上的主要判据仍是"同一 degree、
  改动前后逐位相同"，跨 degree 的逐位判据以 A5 上的结果为准。
- 长跑（50 step）用来确认不漂移：两个 degree 的 `loss` 曲线应保持常数级间距（该间距来自上面的
  实现选择差异），不随 step 放大。
- `--compile.enable` 在 CP=1 下照旧编译（开启后 step 1 的数值应与 eager 一致）；CP>1 下跳过编译并
  告警，跑 eager（§4.2）。

### 7.4 Engram 的取数方式

挂 Engram 的 `deepseek_v4_1_debugmodel_text` 过不了融合内核（`SparseFlashMla` 要求 q head dim 为
512、RoPE 表 512），所以这一路用 `deepseek_v4_1_stripped`：released 几何 + debug 的 Engram 表（只挂
一层，`layer_ids=(1,)`）。它同时是两个 degree 的精度基线，可按第 2 节使用 `NGPU=2`、`CP=2` 和融合算子。

拼好之后的判据与其它用例相同，两条都实测过（A5 单卡，level 3，同一份镜像权重）：

- **整模型**：CP=1 与 CP=2 的 logits 逐位相同（`0/10485760`），step 1 的 `loss` / `grad_norm` 两侧
  都是 `11.94468` / `18.9753`；各自重跑一次也逐位复现。
- **单模块**：把 Engram 模块单独拿出来，用整行前向的结果当基准，再按各 rank 的行 + 按行 history 跑
  一遍，输出同样逐位相同——文档长度不等、`cp=4`、`history` 2/3 都成立。换成"连续一个窗口"的 history
  就不逐位，这正是 §5.5 那条按行 history 存在的理由。

镜像权重、`set_deterministic_level(3)` 两侧同开这两条对 Engram 一样成立；Engram 的表是
`LocalTensor`（没有 `device_mesh`），脚手架在 `_piece` 里要把它当作未切分处理，否则 CP=2 加载镜像
时会崩在取 `device_mesh` 上。

## 8. 相关文档

- [DeepSeek-V4 TND 适配](deepseek_v4_tnd.md)
- [融合算子使用与开发](fused_ops.md)
- [数值对齐](../developer_guides/numerical_alignment.md)
- [MTP](mtp.md)
