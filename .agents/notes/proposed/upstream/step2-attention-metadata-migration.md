# Step 2：把 DeepSeek 栈搬到回移的 metadata 接缝

分支 `exp/metadata-seam-step2`（基于 `local/exp/v32-attn-gym`）。Step 1（回移 #4897 + #4116）与 CP 半边（map 的 `prepare_cp_metadata` 调用点）已完成并验证。

## 目标

1. 删除 `torchtitan_npu/models/common/metadata_extension.py`（`MetadataExtension` / `LightningIndexerMetadata`）及其 5 个实现、模型 Config 上的 `metadata_extension` 字段、`models` 里三处 `build_attention_masks`。
2. 每个栈的 per-batch 元数据由 **attention backend 自己的 config** 产出（`build_attention_metadata`），几何即 map 键，一次构建、全层共享。
3. 四个 CANN `*_metadata`（LI / SMLA / SMLAG / SLIG）进**同一张按几何索引的 records 表**（当前分散在 `asc_plans[ratio]`、`asc[(ratio, cmp_topk)]`，且 V4.1 的 LI/SLIG 仍是 per-layer inline）。
4. `.ci/architecture_import_baseline.json` 删掉 `extensions → models.common.metadata_extension` 两条债务边。

## 顺序与任务

### 2a. V3.2 先迁（最小、覆盖 CP + TND）— 已落地
- `models/deepseek_v3_2/model.py`
  - 新增 `ReferenceVarlenMetadata(VarlenMetadata)`：V3.2 参考核读的全局 packed layout，`prepare_cp_metadata` 直接 raise（参考核不可 CP：attn-gym 的 indexer 取的是「本查询在本文档内的前缀」，CP rank 只持有文档中段）。拒绝放在 metadata 变换里而不是 forward 里：否则 CP 下 map 条目会被原样透传，层会静默读到本 rank 不持有的全局 layout。
  - `build_reference_varlen_metadata(positions)`：两个 backend 的 builder 共用的唯一构建入口（`create_varlen_metadata_for_document` + 包装）。
  - `SparseInnerAttention.Config.build_attention_metadata` → 上面的函数（几何键默认即类键，该栈单几何）。
  - 删除 `Config.metadata_extension` 字段、`__init__` 的 `self._metadata_extension`、整个 `build_attention_masks`；模型 docstring 说明接缝已回到 backend。
- `override/deepseek_v3_2/sparse_attn/ascendc.py`：删 `AscVarlenMetadataExtension`；`AscVarlenMetadata` 增 `from_varlen()` 与 `prepare_cp_metadata()`（CP 下 `prepare_cp_varlen_metadata` 得到 rank-local varlen，再重读 host 列表）；`AscSparseInnerAttention.Config.build_attention_metadata` 直接构建 TND 条目。
  - **偏离原计划**：没有再开一个 `@override(target=SparseInnerAttention.Config, exact=True)` 的 metadata 入口——同一 node 上的第二个入口会与已有 `sparse_attn.asc` 冲突，而 `asc` 返回的 config 本来就能带 builder，metadata 就该长在这里。
  - 注意 `@dataclass(slots=True)` 会重建 config 类，零参 `super()` 会绑到装饰前的类（`TypeError: obj must be an instance...`），因此 Asc builder 调模块级共享函数而不是 `super()`。
- 消费侧无需改动：`AscVarlenMetadata` 的字段（`varlen_meta` / `actual_seq_qlen` / `actual_seq_klen`）与 forward 读取方式不变。
- CP 半边：`patches/torchtitan/distributed/varlen_cp.py` 抽 `prepare_cp_varlen_metadata()`，mask 路径（`cp_shard`）与 map 路径（`prepare_cp_metadata`）共用一份实现；`patched_prepare_context_parallel_input` 把**切分前**的 positions 交给变换（`from_global` 需要全局 batch/seq 才能重建 load balancer 的置换）。
- MTP：新增 `patches/torchtitan/models/deepseek_v3/mtp.py`（#4897 的 `MTPDecoder.forward` 回移）。`DeepSeekV3Model(MTPDecoder)`，V3.2 继承它，map 以 keyword-only `attention_metadata` 进入，逐层（含 MTP 深度）分发。
- **暂不在基类装 builder**：上游把通用 builder 装在 `VarlenInnerAttention.Config` 上，但本仓 `override/qwen3_5/parallelize.py` 在 parallelize 钩子里改写 `kwargs["attention_masks"]`（CP 下转 `QwenCPMetadata`）。若给基类 `VarlenAttention.Config` 加 builder，qwen3.5 会自动切到 map 路径而丢掉这个变换。因此本阶段 builder 按栈声明，等 qwen3.5 一起迁（或明确排除）后再回移基类版本。
- 验证：`tests/unit_tests/models/deepseek_v3_2/test_metadata_seam.py`（新增 5 例：一次构建/全层共享、`preprocess_inputs` 交 map、Asc 条目含 host 列表、CP 逐 rank 重建、参考核拒绝）、`tests/unit_tests/patches/.../test_context_parallel_metadata.py`（+2 例）、`mtp_test_utils` 改走 `preprocess_inputs`；`patches/models/override/extensions` 四目录失败集合与基线逐条一致（models 27、extensions 2、override 3 全是既有环境失败）。
- 顺带修掉随本步失效的引用：`tests/integration_tests/deepseek_v3_2.py` 的 override 列表、`torchtitan_npu/override/README.md` 的 V3.2 表、`docs/feature_guides/fused_ops.md`、`docs/feature_guides/deepseek_v4_cp.md`。

### 2b. V4.1
- `models/deepseek_v4_1/model.py`：`DeepSeekV41Metadata` 增 `cu_seq_k`（后端按自己的 `compress_ratio` 折好的池化轴，`build_row_attention_metadata` 随之下线、改由后端直接调 `create_varlen_metadata_for_document`）；`KernelFrame` 组装搬进 `AscV41SparseAttention.Config.build_attention_metadata`；`prepare_cp_metadata` 承担现在的 `_sharded_attention_masks`（含 engram halo、`exchange.positions_q`）。
- `override/deepseek_v4_1/metadata.py`（新）：`KernelFrame` / `AicpuGeometry` / `AicpuMetadata`，三者都注册为 pytree 节点。
- `override/deepseek_v4_1/sparse_attn/ascendc.py`：删 `AscV41MetadataExtension`；record 由该 core 的 config 构建（SMLA + SMLAG + **LI + SLIG**，一并纳入预计算），落在 `AscV41Metadata` 的 `aicpu_metadata` 字段上（`geometry` + 四个算子输出），消费者的 `_asc_v41_record` 查表改为直接读字段（`attention_metadata.aicpu_metadata.sparse_attn` 等）：一条 entry 就是一个几何，没有 miss 可 raise；几何是该 config 的**必填**字段（无默认值），默认值只在 override factory 上。
- `override/deepseek_v4_1/lightning_indexer/ascendc.py`：`_LightningIndexerTND` 的 fwd/bwd 改读 records（LI 与 SLIG 各一条），`AscV41IndexerRecords` 与 `build_indexer_records` 换成 `build_indexer_metadata` + `kernel_options`。
- 删除模型里构造期 warn-once 与 `reads_asc_metadata` ClassVar（几何来自同一份配置后不再需要）。
- 验证：`tests/unit_tests/override/deepseek_v4_1/test_asc_metadata_reuse.py`、`test_lightning_indexer_fused.py`、`tests/unit_tests/models/deepseek_v4_1/`。

### 2c. V4
- `models/deepseek_v4/metadata.py`：`CompressedVarlenMetadata` 增 records；`build_compressed_varlen_metadata` 保持为共享契约构建，算子记录由 `AscCompressedSparseInnerAttention.Config.build_attention_metadata` 产出（原 `_fill_asc_metadata` 的四个算子 + `_mark_dynamic`）。
- `override/deepseek_v4/sparse_attn/ascendc.py`：删 `AscMetadataExtension`、`AscLightningIndexerMetadata`、`AscCompressedVarlenMetadata`；消费侧 `metadata.asc_plans[ratio]` → records 查找。
- MTP batch 准备（`prepare_mtp_batch`）迁到模型的 `preprocess_inputs` override（上游 #4116 的模型侧钩子）。
- 验证：`tests/unit_tests/models/deepseek_v4/`、`tests/unit_tests/override/deepseek_v4/`、`tests/unit_tests/models/deepseek_v4/test_cp_dispatch.py`。

### 2d. 收尾
- `extensions/adaptor/torchao/converter.py`：量化 LI provider 改为**注册 record builder**，不再 subclass `LightningIndexerMetadata`。
- `extensions/experiment/cv_parallel/batch_chunk_metadata.py` + `batch_chunk.py`：删 `_chunk_li_metadata` 反手接线，改按 records 切分。
- `patches/torchtitan/components/validate.py` 与 trainer 的 `preprocess_inputs` 调用保持（Step 1 已成形）；确认无模型再需要 `build_attention_masks`。
- 删除 `models/common/metadata_extension.py` 与 `.ci` 两条债务边；更新 `docs/feature_guides/deepseek_v4_tnd.md`、`deepseek_v4_cp.md`（决策 1 表述）、`override/README.md` 的 override 表。

## 每步的完成标准

- `patches/` + 该模型的 `tests/unit_tests/models/…` + `tests/unit_tests/override/…` 全绿；
- `ruff check/format`、`scripts/check_architecture_boundaries.py` 通过；
- 失败集合与改动前基线逐条一致（用 `git stash` + 逐目录对照，方法同 Step 1）；
- NPU 侧数值验证另跑（融合路径与 golden 同 checkpoint 逐 step loss/grad_norm）。

## 待决问题

1. records 的键是否要包含 CP shard 形状，还是统一由 `prepare_cp_metadata` 重建（当前倾向后者）。
2. 四个算子是否 value-independent（只依赖形状/几何）→ 决定能否加跨 microbatch 缓存（Step 2 之后再做）。
3. geometry 键是否回馈上游（#4897 自述的 "same backend class configured differently" 限制）——建议提 issue/PR。
