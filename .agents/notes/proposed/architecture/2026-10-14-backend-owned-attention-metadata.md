# Agent Note: 移除 `metadata_extension` —— attention metadata 的所有权回到 attention backend

Status: proposed（分三步实施，本 Note 随 A 合入；B/C 落地时更新本文状态）
Created: 2026-10-14
Last updated: 2026-10-14
Tracking: upstream PR #4897 / #4116 / #4617（回移层）；本仓 A（接缝）、B（三栈迁移）、C（删除收尾）

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

1. `metadata_extension.py` 不存在；无模型定义 `build_attention_masks`，也无 `metadata_extension` /
   `lightning_indexer_metadata` 字段；
2. per-batch metadata 由 backend config 构建（每 microbatch 每 backend 一次、同几何共享），CP 下由 **backend 类**自己的
   `prepare_cp_metadata` 产出 rank-local 副本（几何由 `cp_shard_context` 发布）（V4/V4.1 的记录在 CP 后**重建**）；
3. `.ci/architecture_import_baseline.json` 的两条 `extensions → models.common.metadata_extension` 债务边消失
   （activation order 不动）；
4. torchao 量化 LI 与 cv_parallel 的分块 metadata 走同一接缝选中，不再有 `model._metadata_extension` 这类反手接线。

## 重构流程（三步，各自独立 review / 合并）

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

- **V3.2**：`SparseInnerAttention.Config.build_attention_metadata` 建 packed layout（`ReferenceVarlenMetadata`；参考核
  不可 CP → `prepare_cp_metadata` 拒绝）；`AscSparseInnerAttention.Config` 包成带 TND host 序列长度的
  `AscVarlenMetadata`，CP 用 `prepare_cp_varlen_metadata` 得 rank-local 再重读 host 列表。删 `metadata_extension` /
  `_metadata_extension` / `build_attention_masks` 与 `sparse_attn.asc_metadata`。
- **V4.1**：参考核只保留自己读的几何（`window_size` / `softmax_scale` / `compress_ratio`），metadata 是**整行的
  document 边界**（`create_varlen_metadata_for_document` 的 `cu_seq_q`）加上后端在同一处按自己的 ratio 折一次的
  池化轴 `cu_seq_k`（消费者不再除）；kernel 几何与 records
  （SMLA/SMLAG + indexer 的 LI/SLIKG 四条 AI_CPU 算子）属于端口：`AscV41SparseAttention.Config` 用 `map key =
  (端口类, compress_ratio)` 声明自己的 entry，按 `build_attention_metadata` 每 microbatch 建一次 frames + records
  （`AscV41Metadata` 的 `q` / `swa_k` / `cmp_k` 三个 `KernelFrame` 与 `aicpu_metadata`），消费者直接读字段而不是查表，
  `AscV41Metadata.prepare_cp_metadata` 用 rank-local frames 重建；五个几何数是该 config 的必填字段（无默认值），
  由 override 参数写入、默认值即 V4.1-Flash 发布形状，records 是内核 metadata 的唯一来源（无 inline 回退；
  拿着没经过端口构建的 metadata 前向会直接失败）。输入的连续 slab 交回
  通用 CP 路径（recipe 本就 `load-balancer None`），模型只留 engram halo。删 `AscV41MetadataExtension` 与
  `asc_metadata`。
- **V4**：`CompressedVarlenMetadata` 的帧 + CP plan 搬到 backend；`AscCompressedSparseInnerAttention.Config` 构建四算子
  records，LI record 的 provider 成为该 config 的字段（torchao 量化 / cv_parallel 分块仍以 `@override` 替换该节点
  选中），`prepare_cp_metadata` 在 CP 后重建并 `_mark_dynamic`；MTP batch 在 `preprocess_inputs` 里于全局行 roll 后
  单独切分。删 `metadata_extension` / `lightning_indexer_metadata` 与 `asc_metadata` / `asc_li_metadata`。
- 验收（每栈）：相关 CPU UT 全绿且失败集合同改动前；NPU 侧同 checkpoint 的逐 step loss / grad_norm 一致（本仓无设备，
  待补）；`build_attention_masks` 调用点全改走 `preprocess_inputs`。

### C. 删除与收尾

- 删 `metadata_extension.py`：`register_pytree_node_for_dataclass` 与 `LightningIndexerKernelConfig` 搬到
  `models/common/` 下不以 extension 命名的模块（前者让 graph_trainer 的 `minimal_fx_tracer` 把 metadata 展开成
  tensor/primitive 叶子）；torchao 的 `_QuantizedLightningIndexerMetadataAdapter` 去基类化；cv_parallel 的
  `BatchChunkMetadataExtension` 重新接到新节点（B 之后它匹配不到节点，pipeline 会静默不注册）。
- `.ci` 删两条债务边；`fused_ops.md`、`deepseek_v4_tnd.md`、`deepseek_v4_cp.md`、`deepseek_v4_graph_trainer.md`、
  `graph_trainer_cv_parallel.md`、`override/README.md` 同步；本 Note 精简后移入 `implemented/architecture/`。
- 退出条件：pinned `torchtitan` 升到同时含 #4897 / #4116 / #4617 的 release 后，删除 A 引入的四个 patch 模块与 `.ci` 的
  activation 条目；B 叠在 A 的两个 commit 之上（A 被 squash 时 B 需一次 `git rebase --onto`，A 保留 commit 则无需）。

## Risks

- **map 键默认是 backend 类**（同上游；#4897 自述的限制）：一个类多套几何时由 config 用 `attention_metadata_key` 声明
  几何键。0.3.0 无 `InnerAttention.__init__`，键在 `Module.Config.build` 时**无条件**盖到实例上——SimpleFSDP /
  GraphTrainer 会改写模块的 `__class__`，条件盖章会让层按运行时类型查找而拿到 `None`。
- **不能切分的栈必须显式拒绝**：无 `prepare_cp_metadata` 的条目原样透传，V3.2 参考核因此在变换里 raise，而不是让层静默读
  到本 rank 不持有的全局 layout。
- **V4 的 LI provider 随 metadata 走**（provider config 是该 metadata 上的字段）：只带扁平几何时，量化 LI +
  CP/batch-chunk 的重新切分会静默产出 BF16 记录。
- **本仓无 NPU**：A/B 的等价性依据是逐字段比对（同 frames 输入、同算子参数与顺序、同报错文本）与 CPU UT，数值结论待设备
  侧补跑；torchao 相关改动只能靠代码审查。
- 迁移期两套路径并存：有 builder 的栈走 map（`extra_kwargs["attention_metadata"]`），其余仍单 mask；`Decoder.forward`
  两者都接，0.3.0 的 `get_attention_masks` 保留到最后一步。

## CP 边：`patches/torchtitan/distributed/context_parallel.py` 的退出逻辑

CP 变换落在模型的 `Decoder._cp_shard`（与上游同名同职责；0.3.0 无 `ParallelismContext` / `shard_tensors`，内部用 pinned
的 `prepare_context_parallel_input`），在**切分之后**执行，拿到**切分前**的 positions 与 permutation——rank-local 副本
只能由全局几何推导。这个文件里只有一半与 metadata 接缝相关，另一半是 NPU 运行时改写，两者退出条件不同：

| 文件内容 | 性质 | 退出条件 |
| --- | --- | --- |
| `_varlen_from_masks` + `patched_cp_shard` | pinned `cp_shard` 不处理 varlen mask | **qwen3.5 迁到接缝之后**。三个 DeepSeek 栈不再往 `extra_kwargs` 放 `attention_masks`（V3.2+asc+CP 下该函数已等价于 `original_cp_shard`，只做输入切分，rank-local metadata 由 `_cp_shard` → `prepare_cp_metadata_map` → 各栈 **backend 类**的 `prepare_cp_metadata` 产生），只剩 qwen3.5 的兜底 `get_attention_masks()` 产出 `VarlenMetadata`，由 varlen 分支改写成 `CPVarlenMetadata` 供它的掩码适配器读 `k_global_gather_indices` |
| `apply()` 里的 `_compiled_create_block_mask` eager 改写 | **NPU 运行时约束**：torch CP helper 内部惰性 `torch.compile(create_block_mask)`（`torch/distributed/tensor/experimental/_context_parallel/_attention.py`）会起 worker 进程、无法再初始化 NPU，「CP + BlockMask」路径必须换成 eager | 搬到 **`patches/torch_npu/`**（它改的是 torch 内部符号，不属于 torchtitan 的 CP 语义补丁；`extensions/torchtitan/eager_flex.py` 里那份改写的是另一个同名符号 `torchtitan.models.common.attention._compiled_create_block_mask`，两者不重复），或等 torch/torch_npu 不再用 worker 进程建 mask |

两个条件都满足后该文件整体删除。`varlen_cp.py` 的 `CPVarlenMetadata`/`from_global` **不随之删**：三栈的 map 路径仍用它
构造 rank-local varlen，它随上游 #5071（`HeadTailCPVarlenMetadata`）或「把 V4.1 的 CP 方案推广到 V3.2/V4」一起退场。

qwen3.5 那一步的核对点：它的 inner attention 要声明 `build_attention_metadata`（+ 需要的 geometry key），并把 rank-local
转换从 forward 的掩码适配器挪进自己的 `prepare_cp_metadata`。这条路径**参与数值**，删前须做 NPU CP=1/CP=2 对齐。
