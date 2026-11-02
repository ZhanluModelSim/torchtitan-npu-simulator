# Agent Note: 移除 `metadata_extension` —— attention metadata 的所有权回到 attention backend

Status: implemented（A/B/C 全部落地：三栈迁移、`metadata_extension` 删除、`.ci` 债务边清零）
Created: 2026-10-14
Last updated: 2026-10-28
Tracking: upstream PR #4897 / #4116 / #4617（回移层）；本仓 A/B/C 由 !1063 / !1064 / !1065 / !1066 交付

## 落地状态（2026-10-28）

A（接缝回移）、B（V3.2 / V4.1 / V4 三栈迁移）、C（删除 `metadata_extension` 与 `.ci` 债务边）均已交付：模型不再定义
`build_attention_masks`，每个 attention backend 由自己的 config 构建 entry，CP 半边由 metadata 的
`prepare_cp_metadata` 重建。CP 的 rank-local 行布局此后由 !1099 收敛成一套 doc-chunk（V4 与 V4.1 相同，query 不再
在层内搬动、由整行边界重建 metadata），见 `./2026-10-28-doc-chunk-layout.md`；head-tail 不再计划恢复，原先那句
「先迁到 V4.1 模式、再在它上面加 head-tail」的两步随之作废。

## Context

DeepSeek-V3.2 / V4 / V4.1 的 vendor kernel 需要 per-batch 预计算的 metadata：TND host 序列长度（V3.2），CANN
`lightning_indexer_metadata` / `sparse_flash_mla_metadata` / `sparse_flash_mla_grad_metadata` /
`sparse_lightning_indexer_kl_loss_grad_metadata`（V4 / V4.1）。记录只依赖本 microbatch 的形状与几何，同模型所有层可共享，
故需要「每 microbatch 构建一次」的接缝。原接缝 `models/common/metadata_extension.py` 有三个问题：接缝长在模型上（同一
模型只能有一种 metadata 产出方式）、vendor 类型与公共类型是替换而非包装关系、`build_attention_masks` 只是 pinned 0.3.0
的临时形态。

上游把 ownership 收到 backend：#4897 由 `InnerAttention.Config.build_attention_metadata` 构建本 backend 消费的
metadata、`Decoder` 每 microbatch 建一次并按 backend key 分发给各层；#4617 补 CP 半边（backend 自己把全局 metadata 变
rank-local）。0.3.0 三者皆无（无 `InnerAttention` / `ParallelismContext` / `preprocess_inputs`），故 A 以 patch 回移。

## 目标状态（「重构完成」的判据）

1. `metadata_extension.py` 与 `metadata_nodes.py` 都不存在；无模型定义
   `build_attention_masks`，也无 `metadata_extension` / `lightning_indexer_metadata` 字段：metadata 需要的几何与
   记录契约只存在于 port 自己的 config 上（`AicpuGeometry`），`models/` 与 `patches/` 不留任何 GPU 上无用的
   provider 节点；
2. per-batch metadata 由 backend config 构建（每 microbatch 每 backend 一次、同几何共享），CP 下由 **metadata 自己**的
   `prepare_cp_metadata` 产出 rank-local 副本（几何由 `cp_shard_context` 发布）（V4/V4.1 的记录在 CP 后**重建**）。
   宿主是 metadata 而不是 backend 类：被重建的就是它，key 因此退化成纯寻址（V4/V4.1 如此；V3.2 的 CP 已退场）；
3. `.ci/architecture_import_baseline.json` 不再有 metadata 相关的 `extensions → models` 债务边：metadata dataclass
   直接用 `torch.utils._pytree.register_dataclass` 注册（没有自定义 pytree shim，也没有 patch 目录里的新文件），
   activation order 不动；
4. torchao 量化 LI 与 cv_parallel 的分块 metadata 走同一接缝选中，不再有 `model._metadata_extension` 这类反手接线。

## 落地路径（A 接缝 → B 三栈迁移 → C 删除收尾）

### A. 回移接缝（纯 patch 层，本 PR）

- `attention_metadata.py`（新）：`build_attention_metadata` 默认实现、`attention_metadata_key`、`AttentionMetadataMap`、
  `prepare_cp_metadata_map`（逐 entry 调用 backend 类 hook）、`CPInnerAttention`、`cp_shard_context`、
  #4897 的 `VarlenAttentionMetadata` 别名（挂到 `torchtitan.models.common.attention`）。
- `decoder.py`（新）：`_get_attention_metadata`（每 microbatch 一次、按 key 去重）、`preprocess_inputs`（#4116 等价物：
  建 map → CP shard → SPMD 注解；`full_dtensor` 输入路径上游已删，此处显式拒绝）、`_cp_shard`（#4617 落点：切分后把 map
  变 rank-local，用**切分前**的 positions 与 permutation）。
- `trainer.py` / `components/validate.py`：`post_dataloading_process` 委托模型，两处重复实现合一（`getattr(model,
  "preprocess_inputs", None)` 守卫，非 `Decoder` 模型回落 pinned 实现）；`models/deepseek_v3/mtp.py`（新）：
  `MTPDecoder.forward` 逐层分发（V3/V3.2 继承它；V4 的 MTP 解码器自覆写 forward，不受影响）。
- **不改现有栈行为**：map 只在某 backend config 声明 builder 时非空（此时无栈声明）；有 `build_attention_masks` 的模型走
  过渡分支，其余走 0.3.0 单 mask 路径；`metadata_extension.py` 保留。验收：`tests/unit_tests/patches` 全绿；
  `patches/models/override/extensions` 四目录失败集合与改动前逐条一致；ruff + 架构边界脚本通过。

### B. 三个栈迁到接缝（每栈一个 commit，可按同序拆成三个 PR）

- **V3.2**：`SparseInnerAttention.Config.build_attention_metadata` 建 packed layout（`VarlenMetadata`）；
  `AscSparseInnerAttention.Config` 包成带 TND host 序列长度的 `AscVarlenMetadata`。删 `metadata_extension` /
  `_metadata_extension` / `build_attention_masks` 与 `sparse_attn.asc_metadata`。该栈的 varlen-CP 已整体退场：
  所有 attention config 都派生自 `VarlenAttention.Config`，pinned decoder 在 CP>1 时于配置阶段报错，不再需要
  rank-local 的 metadata 变换。
- **V4.1**：参考核只保留自己读的几何（`window_size` / `softmax_scale` / `compress_ratio`），metadata 是**整行的
  document 边界**（`create_varlen_metadata_for_document` 的 `cu_seq_q`）加上后端在同一处按自己的 ratio 折一次的
  池化轴 `cu_seq_k`（消费者不再除）；kernel 几何与 records
  （SMLA/SMLAG + indexer 的 LI/SLIKG 四条 AI_CPU 算子）属于端口：`AscV41SparseAttention.Config` 用 `map key =
  (端口类, compress_ratio)` 声明自己的 entry，按 `build_attention_metadata` 每 microbatch 建一次 frames + records
  （`AscV41Metadata` 的 `q` / `swa_k` / `cmp_k` 三个 `KernelFrame` 与 `aicpu_metadata`），消费者直接读字段而不是查表，
  `AscV41Metadata.prepare_cp_metadata` 用 rank-local frames 重建；五个几何数是该 config 的必填字段（无默认值），
  由 override 参数写入、默认值即 V4.1-Flash 发布形状，records 是内核 metadata 的唯一来源（无 inline 回退；
  拿着没经过端口构建的 metadata 前向会直接失败）。CP 的 rank-local 行布局此后由 !1099 收敛成 doc-chunk，Engram 的
  n-gram 前驱在切分前按行算好（见 `./2026-10-28-doc-chunk-layout.md`）。删 `AscV41MetadataExtension` 与
  `asc_metadata`。
- **V4**（已落地）：一条 entry 一个几何（`(端口类, compress_ratio)`）——`DeepSeekV4Metadata` 的 `ratio` + `plan`
  取代 `plans` 表，builder 只建本 ratio（`build_compressed_varlen_metadata(varlen, ratio)`），没有 `plan_for`、没有
  跨几何表、也没有 `cu_seq_q_host` / `seq_len_host` 这类宿主缓存（上游 `VarlenMetadata.cu_seq_q_host` 与 `max_k` 足够）。
  `AscCompressedSparseInnerAttention.Config` 用 `AicpuGeometry`（四个必填宽度 + 层自己的 `window_size` /
  `index_topk`）构建四算子 records；LI record 由 port 建，量化的 selector 是**另一个 backend**
  （`torchao_npu.quantized_modules.deepseek_v4.QuantizedV4SparseAttention`：自己的 config、自己的 entry、
  `quant_lightning_indexer_metadata`、CP 重切直接拒绝），所以 `override/` 与 `models/` 都不感知量化，
  也没有可替换的 provider 节点。`prepare_cp_metadata` 在 metadata 上、CP 后重建并 `_mark_dynamic`；MTP batch 在
  `preprocess_inputs` 里于全局行 roll 后单独切分。
  `MetadataExtension` 类在第 C 步已删：它的两个用途各自归位 —— 状态表改由融合 attention 的 metadata 构建器
  在 eager 边界写进 plan（不再有 `compressor.asc_metadata` 入口，也没往 `models/deepseek_v4/` 或公共路径
  加钩子）；CV 分块重切
  改由它自己的模型子类（`BatchChunkDeepSeekV4Model` / `BatchChunkGraphTrainerDeepSeekV4Model`）在
  `preprocess_inputs` 里做，`chunk_metadata` 现在替换的是模型 config。
- 验收（每栈）：相关 CPU UT 全绿且失败集合同改动前；NPU 侧同 checkpoint 的逐 step loss / grad_norm 一致（本仓无设备，
  待补）；`build_attention_masks` 调用点全改走 `preprocess_inputs`。

### C. 删除与收尾（已落地）

- 删 `metadata_extension.py` 与 `models/common/metadata_nodes.py`：`LightningIndexerMetadata` /
  `LightningIndexerKernelConfig` 没有替代物（LI record 由 port 按 `AicpuGeometry` 在同一 builder 里建，
  量化的 selector 是独立 backend），`register_pytree_node_for_dataclass` 改为各 metadata 模块
  直接调用 `torch.utils._pytree.register_dataclass`。`MetadataExtension` 与 `CompressorMetadata`
  两个类删除，模型不再有 `metadata_extension` / `lightning_indexer_metadata` 字段；
  `batch_chunk_metadata.chunk_metadata` 改为替换模型 config。
- `.ci` 的 metadata 债务边全部删除（`metadata_nodes` 两条 + `batch_chunk_metadata → models/deepseek_v4.model`）；`fused_ops.md`、`deepseek_v4_tnd.md`、`context_parallel.md`、`deepseek_v4_graph_trainer.md`、
  `graph_trainer_cv_parallel.md`、`override/README.md` 同步；本 Note 精简后移入 `implemented/architecture/`。
- 退出条件：pinned `torchtitan` 升到同时含 #4897 / #4116 / #4617 的 release 后，删除 A 引入的四个 patch 模块与 `.ci` 的
  activation 条目；B 叠在 A 的两个 commit 之上（A 被 squash 时 B 需一次 `git rebase --onto`，A 保留 commit 则无需）。

## Risks

- **map 键默认是 backend 类**（同上游；#4897 自述的限制）：一个类多套几何时由 config 用 `attention_metadata_key` 声明
  几何键。0.3.0 无 `InnerAttention.__init__`，键在 `Module.Config.build` 时**无条件**盖到实例上——SimpleFSDP /
  GraphTrainer 会改写模块的 `__class__`，条件盖章会让层按运行时类型查找而拿到 `None`。
- **不能切分的栈必须显式拒绝**：无 `prepare_cp_metadata` 的条目原样透传；V3.2 已没有 rank-local CP 变换，其所有
  attention config 都派生自 `VarlenAttention.Config`，pinned decoder 在 CP>1 时于配置阶段报错，层不会静默读到
  本 rank 不持有的全局 layout。
- **V4 的 LI provider 随 metadata 走**（provider config 是该 metadata 上的字段）：只带扁平几何时，量化 LI +
  CP/batch-chunk 的重新切分会静默产出 BF16 记录。
- **本仓无 NPU**：A/B 的等价性依据是逐字段比对（同 frames 输入、同算子参数与顺序、同报错文本）与 CPU UT，数值结论待设备
  侧补跑；torchao 相关改动只能靠代码审查。
- 迁移期两套路径并存：有 builder 的栈走 map（`extra_kwargs["attention_metadata"]`），其余仍单 mask；`Decoder.forward`
  两者都接，0.3.0 的 `get_attention_masks` 保留到最后一步。

## 遗留与后续

- **builder 按栈声明，暂不装到 `VarlenInnerAttention.Config` 基类**：qwen3.5 现在在 `override/qwen3_5/parallelize.py`
  的钩子里改写 `kwargs["attention_masks"]`（CP 下转成自己的 `QwenCPMetadata`）。基类一旦带 builder，它会自动切到 map
  路径而丢掉这个变换——所以等 qwen3.5 一起迁（或明确排除）之后再回移基类版本。
- **records 每 microbatch 每几何重建一次**：四个 CANN 算子是否 value-independent（只依赖形状与几何、不看数值）还没验证；
  验证过之后可以考虑跨 microbatch 缓存，前提是缓存键能覆盖几何与 CP 行数。
- **几何键回馈上游**：#4897 自己描述的「同一个 backend 类按不同方式配置」限制由本仓的 `attention_metadata_key` 绕开，
  建议提 issue/PR，把「一个 backend 可声明多个几何键」收进上游。

## CP 边：`patches/torchtitan/distributed/context_parallel.py` 的退出逻辑

CP 变换落在模型的 `Decoder._cp_shard`（与上游同名同职责；0.3.0 无 `ParallelismContext` / `shard_tensors`，内部用 pinned
的 `prepare_context_parallel_input`），在**切分之后**执行，拿到**切分前**的 positions 与 permutation——rank-local 副本
只能由全局几何推导。这个文件里只有一半与 metadata 接缝相关，另一半是 NPU 运行时改写，两者退出条件不同：

| 文件内容 | 性质 | 退出条件 |
| --- | --- | --- |
| `patched_cp_shard` 的 doc-chunk 布局 balancer | 仓库自有布局：rank `r` 的窗口是每篇文档的第 `r` 段；rank-local metadata 由各栈自己的 `prepare_cp_metadata` 或整行边界重建，shard 只切张量 | 上游 `cp_shard` 提供布局注册点，或 doc-chunk 布局并入上游后删除 |
| `apply()` 里的 `_compiled_create_block_mask` eager 改写 | **NPU 运行时约束**：torch CP helper 内部惰性 `torch.compile(create_block_mask)`（`torch/distributed/tensor/experimental/_context_parallel/_attention.py`）会起 worker 进程、无法再初始化 NPU，「CP + BlockMask」路径必须换成 eager | 搬到 **`patches/torch_npu/`**（它改的是 torch 内部符号，不属于 torchtitan 的 CP 语义补丁；`extensions/torchtitan/eager_flex.py` 里那份改写的是另一个同名符号 `torchtitan.models.common.attention._compiled_create_block_mask`，两者不重复），或等 torch/torch_npu 不再用 worker 进程建 mask |

DeepSeek-V3.2 的 varlen-CP backport（`patched_cp_shard` 的 varlen 分支、参考核与 AscendC entry 的
`prepare_cp_metadata`）已整体删除：该栈不再支持 CP，pinned decoder 在配置阶段就拒绝 CP>1，不需要第二道 rank-local
守卫。qwen3.5 的 rank-local 转换仍在其自己的 `override/qwen3_5/parallelize.py` 钩子里（`QwenCPMetadata`），不经过
这条分支。

qwen3.5 那一步的核对点：它的 inner attention 要声明 `build_attention_metadata`（+ 需要的 geometry key），并把 rank-local
转换从 forward 的掩码适配器挪进自己的 `prepare_cp_metadata`。这条路径**参与数值**，删前须做 NPU CP=1/CP=2 对齐。
