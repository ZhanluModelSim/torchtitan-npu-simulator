# DeepSeek-V4 上下文并行（CP）

DeepSeek-V4（DSV4）在文档打包序列上训练时，压缩注意力（CSA/HCA）的 KV 块按压缩比 `r` 对齐文档前缀：块可能跨越相邻 CP 卡的边界，各 rank 可压缩的块数随文档分布不均，查询可见的滑窗行与压缩块是文档相对范围。本仓为 DSV4 的 AscendC 融合路径实现了上下文并行：每个 batch 由 `build_cp_plan` 从切分前的全局文档结构纯本地推导出「计划」，`CPTokenDispatcher` 按计划完成窗口行与压缩块的 gather，以及压缩容器的声明式 all-gather。context parallel 由 CLI 参数 `--parallelism.context-parallel-degree` 激活（配合 `--parallelism.spmd-backend=spmd_types`）。参考内核与 golden 数值基线不启用 CP。

本文是 CP 功能的**实施交接文档**，目标读者是接手 DSV4 CP 开发的后续开发者：先看「启用方式」「设计要点」「关键设计决策」掌握整体方案与取舍依据，再看「每层数据流」「压缩器在 CP 下的工作方式」（含图示）与「组件与文件」定位代码，最后按「验证方式」执行回归。已知限制与内核契约均记录在案。

## 启用方式

```bash
NGPU=2 \
  bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh \
  --parallelism.context-parallel-degree 2 \
  --parallelism.spmd-backend spmd_types \
  --training.steps 2
```

- `--parallelism.context-parallel-degree > 1` 时即启用 CP；必须配合 `--parallelism.spmd-backend spmd_types`（`spmd_types` 后端是压缩容器 `S(1) -> R` 声明式 all-gather 的前提）。
- `--parallelism.context-parallel-load-balancer` 支持 `headtail`（默认）与 `None`（plain 顺序切分）。headtail 要求 `seq_len` 整除 `2 * context_parallel_degree`，plain 要求整除 `context_parallel_degree`，不满足时 `_build_cp_metadata` 直接报错。
- 需要完整 NPU 环境：`torch_npu`、CANN、HCCL 与匹配的 `cann_ops_transformer`。CP 只在融合路径生效，参考 tier 与 golden 限定无 CP（对 `cu_seq_q != cu_seq_k` 报错）。
- `local_batch_size == 1` 是当前文档打包约定（增大批数据应提高 `seq_len`）；DSA 路径 TP 固定为 1。

## 设计要点

通用 CP（切分 + 全部 KV all-gather）对 DSV4 不适用的三个原因：

1. 压缩块跨 rank：`r` 个连续 token 成一块，块可能跨越相邻 rank 的边界，单个 rank 无法独立完成边界块。
2. 各 rank 压缩数量不均：文档独立压缩（每文档 `len // r` 块），各 rank 块数随文档分布变化，通信规模不固定。
3. 注意力可见区是文档前缀：查询可见的压缩块是所在文档的前缀块，滑窗行是文档相对范围，均需跨 rank 取行。

方案的整体原则是「**计划吸收全部不规则性，数据路径保持单一**」：

- **纯本地推导**：[`build_cp_plan`](../../torchtitan_npu/models/deepseek_v4/token_dispatcher.py) 从切分前的全局文档结构（`get_attention_masks` 的 varlen）加负载均衡置换，在 frame 内推导每个 rank 的 rank-local varlen（复用 `CPVarlenMetadata.from_global`）与全部计划，计划阶段零通信。
- **文档切片**：全部计划行都是 `docs[doc]` 的切片（文档按置换后坐标存储，键为置换后的文档起点）。负载均衡的置换被切片完全吸收，无需访问置换本身。
- **双计划**：ratio 无关的 `WindowPlan`（窗口交换路由、ori 流 `gather_indices`、`cu_seqlens_ori_kv`）与每 ratio 一份的块计划 `CompressedBlockLayout`（内核与压缩器契约 + 块交换路由 + 容器打包 + `cmp_k_global_gather_indices`），分别挂于 `CompressedVarlenMetadata` 的 `window` 与 `plans[ratio]`。非 CP 时块计划的 `gather_indices` 即文档序本地 gather，其余 dispatcher 字段为空，前向路径不区分 CP。
- **块区间**：每段的计划块区间 `[A, B)` 覆盖本段完整块加借入源（段首前驱块、段尾跨块、case-C 前驱头），`strip` 记借入源块数；段尾不足一块的文档余数不产生压缩条目。

## 关键设计决策

以下决策是当前方案的设计依据，接手开发时应保持，不做反向重构。

1. **单一 metadata 接缝**。模型以 `build_attention_masks(inputs, labels, extra_kwargs, *, cp_mesh, load_balancer_type)` 统一构建每 batch 的 metadata：CP 时在其中完成输入切分与计划推导，vendor 内核张量由配置解析的 `metadata_extension`（`AscMetadataExtension`）在最后一步填充。接缝只有一处，dsv3.2 走同一机制。

2. **纯本地计划**。`build_cp_plan` 输入 `(global_varlen, load_balancer, rank, cp_size, shard_len, window_size, ratios)`，用 `_RankMesh` 垫片逐 rank 调用 `CPVarlenMetadata.from_global`，返回 `(rank-local varlen, plans, window)`。全局上下文（切分前的文档结构 + 置换）在 `build_attention_masks` 帧内即可获得，因此计划阶段零通信，且推导出的 rank-local varlen 与 shard 路径的产物一致。

3. **文档切片表达**。`build_cp_plan` 构建一个 `docs: dict[doc, list[pos]]`（键为置换后的文档起点，值为该文档全部 token 的置换后坐标）。基于恒等式 `kg[k_start + p] == docs[doc][p]`，所有计划行都是文档切片：窗口行 `docs[doc][win_start : win_start + win_len]`，块行 `docs[doc][A : block_end]`。压缩器契约由块区间直接推导：`block_positions` = 每段 `arange(A, block_end, ratio)`，`first_indices` = 每段池化起点。

4. **统一 gather 接口**。窗口与块共用 `dispatcher.gather(x, plan)`：无交换时为本地 gather（`plan.gather_indices` 直接索引本地流），有交换时为远端 gather + 重排（同一 `gather_indices` 索引 `cat([x_local, recv])`），始终返回 `[1, N, D]`。重排结果本身就是池化流，数据路径上无多余拷贝。

5. **交换内容按需投影**。窗口交换 RoPE 后的 `swa_k`（`head_dim` 宽），块交换投影后的 kv/score（`coff·head_dim` 宽；`coff` 为 r4 重叠时 2、其余 1）。Compressor 内部以 `dispatcher.gather(self.wkv(x), plan)`、`dispatcher.gather(self.wgate(x), plan)` 各执行一次交换。`wkv`/`wgate`/RMSNorm/RoPE 都是逐 token 运算，与交换可交换，因此数值逐位一致，通信量远小于交换原始 `x`。

6. **压缩级 gather 声明式**。各 rank 用 `dispatcher.select` 将保留块装入等宽填充容器 `[1, out_width, D]`（`out_width` 为跨 rank 最大保留块数，使容器成为合法的 `S(1)` 分片）；`sharding.py` 把容器声明为 `cp: S(1) -> R`，框架在内核边界自动发出 all-gather；`cmp_k_global_gather_indices`（`slot = owner × out_width + 容器内偏移`）在 `ascendc.py::_assemble_tnd` 内组装每段前缀流。

7. **借入先压缩、借入源键剥离**。`_block_range` 的四种 case（段首前驱 / case-C 前驱头 / 首块跨块 / 空）给出每段区间 `[A, B)` 与 `strip`；借入源块用于补全本段块的投影与重叠链，其自身池化键经 `compressed_rows` 从输出中剥离；段首块的借入行经 `first_indices` 掩码（分数置 `-inf`，softmax 权重精确为 0）。打包流只包含重叠链完整的键。图示见「压缩器在 CP 下的工作方式」。

8. **每 Compressor 自带 dispatcher**。`Compressor.Config` 自带 `token_dispatcher` 字段，`Compressor.parallelize` 经框架的 `Module.parallelize` 递归自行接线；`Attention` 另有自己的 dispatcher（窗口）。每个压缩器（注意力与索引器各自的）独立执行块交换，组件自包含；ratio-4 层因此有 5 次 alltoallv。

9. **内核契约是方案的前提**。窗口打包依赖内核「窗口范围由属性决定而非区间长度」（端对齐坐标），CP 与参考的可见范围一致依赖内核的 p0 通道（`limit(pos) = floor((p0 + pos + 1) / r)`）。计划据此构造 `cu_seqlens_cmp_k` / `block_remainder`（文档相对块数与余数）与 `cu_seqlens_ori_kv`（窗口长度前缀和），修改计划必须保持这两个内核语义不变。

10. **参考实现与 golden 不启用 CP**。两者按连续文档（`cu_seq_q == cu_seq_k`）构建，对 CP 形状流显式报错；数值验收以融合路径自身的三配置一致性为准。

## 每层数据流

`Attention.forward` 在本地流上完成 Q 侧投影与 `swa_k` 的投影 + RoPE（按文档相对位置），随后：

1. **窗口 gather**：`dispatcher.gather(swa_k, metadata.window)` 交换 RoPE 后的 `swa_k` 行，组装为打包 ori 流（每段窗口行，`cu_seqlens_ori_kv` 给出边界）。
2. **块 gather**：每个 `Compressor` 内部对投影后的 kv/score 执行 `dispatcher.gather(·, plans[ratio])`，得到池化块流；`dispatcher.select` 将保留块装入等宽填充容器。ratio-4 层的索引器 Compressor 走同一机制。
3. **压缩级 all-gather**：容器按 `cp: S(1) -> R` 声明式 all-gather（`sharding.py`），AscendC 内核内用 `cmp_k_global_gather_indices` 组装每段前缀压缩流，再调用 LI/SMLA。

每层 alltoallv 次数：ratio-1 层 1 次（窗口）；ratio-128 层 3 次（窗口 + kv + score）；ratio-4 层 5 次（再加索引器 Compressor 的 kv/score）。反向梯度沿同一路由回传。

## 压缩器在 CP 下的工作方式

CP 方案中复杂度最高的是压缩器的借入与组装。以 `r = 4` 为例，按「为什么需要借入 → 交换与组装 → 数学与剥离 → 容器组装」展开；`r = 128` 除无重叠外机制相同。

### 为什么需要借入

一个文档被 CP 切成片段后，片段边界通常不会正好落在块的边界上：块可能跨两个 rank，r=4 的池化还需要前一块的行做重叠。以 16 token 的文档、r=4、rank k+1 的片段只有 8..9 为例：

```text
0               4               8               12              16
└───── b0 ─────┘└───── b1 ─────┘└───── b2 ─────┘└───── b3 ─────┘

┌──────────────────────────────┐  rank k 段：持有 0..7（块 b0、b1 完整）
└──────────────────────────────┘
                                ┌──────┐  rank k+1 段：持有 8..9（块 b2 的前一半）
                                └──────┘
                                ↑ 还差：
                                  · token 10..11 在片段外（跨块尾，凑不齐块 b2）
                                  · 前一块 b1 的行（r=4 池化的重叠前驱）

                计划区间 [A, B) = [4, 12)
                ┌──────────────────────────────┐
                └──────────────────────────────┘
                A = 4：借入源块 b1（strip = 1，池化键剥离）
                B = 12：跨块终点；无跨块时退化为片段尾
```

把 `b1`（重叠前驱）和 `10..11`（跨块尾）一起纳入计划区间后，本段就能独立完成块 b2 的池化；借入源块 b1 的池化键在 `select` 时被剥离，不会混入本段的压缩 KV。借入源块（A 处）与跨块尾（B 处）是同一套行交换的两类来源——段首块自身的重叠前驱正是借入源块。

### 交换与组装

投影是逐 token 运算，先投影后交换与先交换后投影逐位一致，因此交换的是窄的投影行：

```text
① 本地投影:  kv/score 行 [S_local, coff·hd]（逐 token，先算好再交换）
② 交换:      alltoallv，发送行按接收方分组
③ 组装:      cat([x_local, recv])，再按 gather_indices 重排为池化序
④ 池化行流:  [n_blocks, r, coff·hd]（借入源块在内，池化序）
```

`gather_indices` 把「本地行 + 借入行」重排为池化序；无 CP 时它退化为文档序本地索引。

### 数学与借入源剥离

```text
① 池化块行流:  [n_blocks, r, coff·hd]（每块 r 个 token 一行）
② 池化:        reshape → 重叠（r=4）→ softmax pool → norm → RoPE
                （重叠：块内拼接前一块的行；段首块的借入行被 -inf/0 掩码）
                → pooled keys [n_blocks, D]
③ 剥离:        select 按 compressed_rows 去掉借入源块的键
                （借入源块的重叠链不完整，不能当压缩 KV 用）
④ 容器:        保留块零填充为等宽容器 [1, out_width, D]
                （各 rank 等宽，才是合法的 S(1) 分片）
```

### 容器与每段前缀组装

```text
① 容器:      每 rank 一个 [1, out_width, D]（保留块靠前，尾部零填充）
② all-gather: 声明式——ShardingConfig 把容器标为 cp: S(1) -> R，
              框架在内核边界自动发出全量聚合
③ 网格:      [cp × out_width, D]（按 rank 顺序拼接的容器）
④ 组装:      cmp_k_global_gather_indices 逐段挑出前缀块：
              slot = owner × out_width + 容器内偏移
⑤ 前缀流:    每段 TND 流，块 0 .. seqlen_k // r − 1
⑥ 内核:      LI（r=4 选择）→ SMLA
```

**剥离不变式**：打包流只包含重叠链完整的键——每段前缀块取自 owner（块起点所在段）的计算结果；借入源块在本 rank 的池化键被剥离，只供相邻块的重叠投影使用。`r = 128` 无重叠时剥离是去重，`r = 4` 时剥离是正确性必需。

## 内核契约要点

CP 方案依赖 AscendC `sparse_flash_mla` 家族的以下语义（均从内核行为验证得出）：

- **TND 布局**：每段一个 batch；`cu_seqlens_q`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`、`cmp_residual_kv` 为前缀和/每段余数；`kvHeadNum = 1`；`ori_win_left = window − 1`、`ori_win_right = 0`；`ori_mask_mode` / `cmp_mask_mode` 固定为 4 / 3。
- **端对齐**：ori token `k` 被读作文档 token `(s1Size − oriLen) + k`，掩码以端对齐坐标计算，窗口范围由属性（`ori_win_left/right`）决定而非区间长度。因此 `cu_seqlens_ori_kv` 不必是因果前缀，窗口打包得以成立。
- **压缩因果上限**：`limit(pos) = floor((p0 + pos + 1) / r)`。计划按文档相对块数与余数构造 `cu_seqlens_cmp_k` / `block_remainder`，使 CP 形式与参考调用的可见范围一致（内核的 p0 通道）。
- **sink**：1-D fp32 `[N1]`，softmax 初值 `max = sink`、`sum = 1`；LSE 为 `log(sum) + max`。

## 组件与文件

| 文件 | 职责 |
| --- | --- |
| `torchtitan_npu/models/deepseek_v4/token_dispatcher.py` | CP 的全部逻辑：`build_cp_plan` 纯推导（文档切片、窗口与块区间、交换路由）、`WindowPlan` / `ExchangePlan`、`CPTokenDispatcher`（`gather` / `select`） |
| `torchtitan_npu/models/deepseek_v4/metadata.py` | 公共契约：`CompressedBlockLayout`（内核与压缩器契约 + dispatcher 字段）、`CompressedVarlenMetadata`（`varlen` / `plans` / `window`）、非 CP 的 `build_kernel_layout` |
| `torchtitan_npu/models/deepseek_v4/model.py` | `build_attention_masks` 单入口；`_build_cp_metadata` 切分输入并调用 `build_cp_plan` |
| `torchtitan_npu/models/deepseek_v4/attention.py`、`compressor.py` | 前向数据路径：窗口 gather、Compressor 内部块 gather、`select` 容器打包 |
| `torchtitan_npu/models/deepseek_v4/sharding.py` | 容器的 `cp: S(1) -> R` 声明（内核边界 all-gather 由此发出） |
| `torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py` | `AscMetadataExtension`（按窗口打包的 `cu_seqlens_ori_kv` 填充 `*_metadata` 内核）、`_assemble_tnd`（`cmp_k_global_gather_indices` 组装） |
| `torchtitan_npu/patches/torchtitan/distributed/varlen_cp.py` | `CPVarlenMetadata`（上游 PR 的本地回传） |

## 已知数值边界

CP 与参考实现之间已知的数值差异有三个来源，全部位于 AscendC 内核内部；CP 数据路径本身与参考逐位一致（由 CPU 单测固定）。

1. **SMLA 的批次布局敏感（几何来源）**。flash-softmax 的行求和按 fp32 部分和分组执行；packed 调用与参考调用的批次布局不同，同一查询窗口的值落入不同的部分和分组，fp32 非结合性带来 bf16 ulp 级的输出差异。该差异与值相关，与 CP 无关——任何改变调用几何的方式（批量大小、段结构）都会触发同类差异。

2. **LightningIndexer-V2 的并列 tie（选择来源）**。indexer 分数经 ReLU 后大量块得分为精确 0，top-k 边界常落在 0 分池内，此时入选块由调用的核切分几何决定。后果是结构不同的两次调用（CP 与参考、plain 与 headtail）可能为同一查询选中**完全不同的块集合**。需要注意：indexer 的相关性分数相近（甚至同为 0）并不意味着这些块在 SMLA 的 query/key 相关性也相近——在初始权重（随机初始化）阶段尤其如此，被 tie 换掉的块在 SMLA 中的权重可能差异明显，单块的替换会放大到整行输出与损失。这是 r4 层 CP 与参考发散的主导来源，修复依赖 AscendC 提供值无关的规范 tie-break；修复前，发散量级随 r4 层数与权重触发情况增长。

3. **SMLAG 的 kv 侧求和（梯度来源）**。一个 kv token 同时出现在多个段的 kv 流中，各内核调用各自产生 bf16 舍入的梯度部分，CP 梯度为这些部分的 fp32 求和，与参考的逐位结果存在部分舍入量级的差异；查询侧梯度（dq、l1、SLIG）逐位一致。

验收口径：以上差异在「三配置一致性」验收中表现为打印精度内的逐 step 损失差。出现超出预期量级的发散时，先按上述三条归类（几何、选择、梯度），再按「精度定位相关单测」的定位路径检查 CP 数据路径——其逐位性已由 CPU 单测固定。

## 验证方式与接手流程

CPU（无 NPU 依赖）：

```bash
pytest tests/unit_tests
```

包含 plan 对 oracle 的逐张量比对与真实双进程 gloo 用例；需要 `TORCHTITAN_DIR` 指向本地 torchtitan checkout。

### 精度定位相关单测

出现 CP 与参考的数值发散时，以下用例按层级排除问题：

| 用例 | 定位对象 |
| --- | --- |
| `test_cp_dispatch.py::test_plan_matches_experiment` | 计划张量（内核契约、保留块、窗口边界）对实验的独立推导逐张量比对；排除计划本身的推导错误 |
| `test_cp_dispatch.py::test_dispatcher_vs_oracle` | 窗口与块 gather 的往返结果对全局流 oracle 逐段比对；排除交换路由与组装（`gather_indices`）的错误 |
| `test_cp_dispatch.py::test_compressor_cp_branch` | 压缩器在 CP 计划（借入流 + 剥离）下的池化键对 oracle 比对；排除压缩数学在借入流上的错误 |
| `test_cp_dispatch.py::test_asc_extension_cp_metadata` | AscendC metadata 内核的填充参数（窗口打包的 `cu_seqlens_ori_kv`） |
| `test_cp_dispatch.py::test_cp_attention_flow` | 前向全流程的形状契约与容器内容（保留块 vs oracle） |
| `test_cp_dispatch.py::test_multiprocess_gloo` | 真实双进程集合通信（不均衡 alltoallv、压缩级 all-gather）的分组与偏移语义 |
| `test_cp_compressor.py`（矩阵） | 压缩数学本身：plain/headtail、cp 1~8、ratio 4/128，f32/f64 双精度矩阵；f64 预算在浮点噪声底（逐位量级），f32 在单精度舍入量级 |
| `test_cp_compressor.py::test_degeneracy` | 计划推导的无漂移护栏：cp=1 的 `build_cp_plan` 与非 CP `build_kernel_layout` 五个契约字段逐位一致 |
| `test_dsv4.py`（数值系列） | 非 CP 基线：压缩器对独立逐文档参考、重叠掩码逐位、索引器选择、参考内核对 golden 的舍入底比对 |

定位路径与「已知数值边界」对应：上述用例全部通过时，组装的数据与键对 oracle 一致，发散必然位于内核层（几何 / 选择 / 梯度三类来源）；任一失败则说明 CP 数据路径本身被改坏，按失败用例定位。

NPU 一致性验收（改动数据路径后建议执行）：加载同一固定初始权重 checkpoint，CP=1 / CP=2 plain / CP=2 headtail 各跑相同步数，期望三种配置的逐 step 损失与 grad_norm 在打印精度内一致；发散量级见「已知数值边界」。每次运行必须使用全新 `--checkpoint.folder`。

接手开发的默认路径：

1. 只改计划推导（`token_dispatcher.py`）时，`build_cp_plan` 必须保持纯函数与零通信，传入的 ratios 需去重（`sorted(set(ratios))`）；以 CPU 单测的 plan-vs-oracle 比对为逐张量护栏。
2. 改动数据路径（`attention.py` / `compressor.py` / `ascendc.py`）时，非计算性改动必须保持逐位一致（CPU 单测 + 三配置验收）；计算性改动需在验收之外记录数值证据。
3. 新增或调整配置（`config_registry.py` / `run_train.sh`）时，同步检查本指南的启用方式与相邻的 [TND 指南](deepseek_v4_tnd.md)。
4. 修改实现后，同步更新本指南中受影响的「设计要点」「关键设计决策」条目，避免文档与代码漂移。
