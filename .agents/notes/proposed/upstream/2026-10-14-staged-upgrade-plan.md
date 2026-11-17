# torchtitan-npu 分阶段升级与 patch 清理方案（2026-10-14）

> 本文是"torchtitan 升级 + `patches/torchtitan/` 清理"的**唯一入口文档**，整合了此前的三份材料：
> - 本文 §1–§5：分阶段升级阶梯（新增）；
> - 本文 §6（附录 A）：原 `2026-10-14-backport-plan.md` —— **不换 pin** 的波次与逐模块工单；
> - 本文 §7（附录 B）：原 `2026-10-14-patches-vs-mainline.md` —— 逐个 patch 的上游对照审计证据。
>
> 基线说明：附录 A/B 写于上游基线 `3386c6d6f`（2026-10-08），其后的 74 个提交差异记在附录 B 末尾的 delta 小节；本文 §1–§5 按最新 `origin/main` = `eb92f645c`（2026-10-13）重核。

---

## 1. 现状、机制与判据

### 1.1 pin 的实际位置

| 项 | 值 |
| --- | --- |
| 仓库 pin | `requirements.txt` → `torchtitan==0.3.0`（PyPI）+ `.ci/setup_torchtitan.sh` 推导 tag `v0.3.0` |
| pin 的提交 | `086bf6c16`（2026-09-02，"Prepare TorchTitan 0.3.0"），是发版分支提交 |
| 与 main 的分叉点 | `48cb7ae1f`（2026-08-18，`#4215`） |
| 当前 `origin/main` | `eb92f645c`（2026-10-13，`#5372`） |
| 待吸收量 | **分叉点 → main = 582 个提交**（≈8 周）。注意：此前记录的口径 "74" 是"上次审计基线 → HEAD"，不是 pin → HEAD |

### 1.2 阶梯的机制成本很低（关键前提）

`.ci/common.sh:43` 已经是：

```
TORCHTITAN_COMMIT="${TORCHTITAN_COMMIT:-v${TORCHTITAN_VERSION}}"
```

随后 `git fetch --depth 1 origin "${TORCHTITAN_COMMIT}"` + `checkout --detach` + `pip install -e`。

**但 L1 预检实测推翻了"零成本"这一半**：gitcode 镜像 `https://gitcode.com/GitHub_Trending/to/torchtitan.git` **不接受按 sha 取** —— `git fetch --depth 1 origin dc3985ad4` → `fatal: couldn't find remote ref`；而 `refs/tags/v0.3.0` 与上游同值（所以当前 pin 能用）。镜像**持有**这些对象：`--filter=blob:none --no-checkout` clone 后 `checkout --detach dc3985ad4 / 3f087cf15 / eb92f645c` 实测**成功**（blob 懒加载可用）。

**这是阶梯唯一的阻断项**，三种解法（按推荐次序）：

1. 改我们自己的 CI 脚本：`clone --filter=blob:none --no-checkout` + `checkout --detach <sha>`（`.ci/common.sh` 与 `.ci/setup_torchtitan.sh` 共用一个 helper），并在 `setup_torchtitan.sh` 补 `TORCHTITAN_COMMIT` 覆盖；
2. 在**我们可控的镜像**上给每一级 rung 打 tag（`.ci/common.sh` 零改动）；
3. 直接改用上游 GitHub（集群网络可达性未验证）。

另外记录两条环境差异：共享 venv 里 `torch_npu` 实际是 `dev20261001`，而 `requirements.txt` 写的是 `dev20260918`；镜像 `main` 已到 `58a3027d9`（`#5392`），所以 **L3 的 rung 在开工前必须重新冻结**。

### 1.3 两个度量与判定准则

- **度量 A（耦合度）**：我方代码引用"只在 pin 上存在的上游名字"的站点数。当前 **A = 513**（A1 = 239 上游 main 已无此名 + A2 = 274 本仓库自造 seam），目标 ≈190。
- **度量 B（可删度）**：`patches/torchtitan/` 下"删文件 + 删一行激活"即可移除的模块数。当前 **5/36**，目标 ≥22。
- **判定四分类**：① 上游已吸收（bump 即可删）／② 已吸收但形态不同（需按新形态重写）／③ 上游仍无 + 锚点已漂移（bump 时必须重写）／④ 冲突（host 文件被删/被重设计）。
- **今天能不能改的准则**：只依赖 `torchtitan==0.3.0` 已有符号 + 我们自己代码的改动 → 现在能做；需要新上游符号的 → 等换 pin。回移的目的是**催熟**，让换 pin 时的动作退化为"删文件 + 删一行激活"，而不是"重写"。
- **口径提示（L0 实测）**：新落地的 `scripts/measure_upstream_coupling.py` 是**全树粗计数**（同一棵树上 A1 = 590 / A2 = 321 / A = 911），**不复现**上面的一次性审计值（239 / 274 / 513）；差异集中在 `parallel_dims`（188 vs 5）、`VarlenMetadata`（114 vs 30）、`spmd_backend`（44 vs 12）。两者不可混用：**审计值用于设定目标，脚本值用于看同口径趋势**（L0 后 A1 −6、A2 −6）。

### 1.4 为什么"分阶段升级"在结构上成立

单个 patch 的处置方式只取决于两件事，而这两件事在上游都是**按 PR 逐周发生**的：

1. 上游是否已提供该能力（能删 → ①）；
2. 我们 patch 的 host 文件是否还在（能重锚 → ②③④）。

在 `origin/main` 上核过：**已有 10 个 host 文件消失/改名**，集中在两次重构里 ——

| 消失的 host | 变化 |
| --- | --- |
| `distributed/full_dtensor.py` | `#4217` 删除整个 full_dtensor 后端（连同 `get_spmd_backend()` 与 `spmd_backend` 配置字段） |
| `components/{validate,checkpoint,optimizer,ema,metrics}.py` | `#5277` 重组：`validate`→`validator.py`、`checkpoint`→`checkpointer/`、`optimizer`/`ema`→`components/optim/`、`metrics`→`observability/metrics.py` |
| `hf_datasets/*` | `#4088` grain + `#5277` 并入 `components/data/` |
| `protocols/` | `#5277` → `models/base/` |
| `experiments/graph_trainer/{ep_chunk_concretization,graph_trainer_runtime_context}.py` | `#4952` 删除图式 EP chunking；runtime context 重设计 |

这些就是"一次性跳过去会同时炸"的部分，也正是阶梯要隔离的东西。阶梯买到的是**可 bisect + 每段 diff 可读 + 依赖同步可跟着段走**；买不到的是总工作量（513 处耦合点 + 36 个 patch 的重锚总量不变）。

---

## 2. 分阶段升级阶梯（总览）

| 段 | 目标上游修订 | 上游窗口 | 该段消化的能力 | 该段删除/收缩的 patch | 同段必须完成的我们侧改动 | 指标目标（估算） |
| --- | --- | --- | --- | --- | --- | --- |
| **L0**（pin 不动，现在做） | `v0.3.0` | — | 无新能力，纯减法 + 形状对齐 | 提前退役 `distributed/full_dtensor.py`（patch，非后端） | 退役 full_dtensor patch + 把 CP+SDPA 窄守卫搬进我们自己的代码；`spmd_backend` 43 处**只做分类登记、不清零**；auto-load patch 加存在性防护；补 7 个 `Upstream-PR:` 头；aux_loss/EMA/moe/rope/CP 的"形状对齐"重写 | A 基本不变（早先 "A1 −104" 估算作废）；B 5→6/36 |
| **L1**（地雷段 + 机械段） | `dc3985ad4`（09-05） | 08-19 → 09-05 | `#4217` full_dtensor 删除、`#4088` grain、`#4239` mm_collator 一维契约、`#4122` DistMuon 存储分片、`#4186` checkpoint storage seam、`#4116` `Decoder.preprocess_inputs`、`#4265` spmd_types 0.2.5 去 `SpmdLayout`、`#3634` 上游 V4 + rope 契约、`#4474` optimizer state | `full_dtensor.py`（L0 已删）、`models/common/rope.py` 的 key/inverse 半、`hf_datasets/text_datasets.py`（重锚到 grain 或删除） | `spmd_backend` 残留确认为 0（L0 已清 43 处）；3 个 `sharding.py` 去 `SpmdLayout`/`protocols.sharding`；`models/common/decoder.py` 收缩（`preprocess_inputs` 回归上游）；DistMuon 锚点断言；collator 一维契约 | A ≈330；B ≈8/36 |
| **L2**（架构段） | `3f087cf15`（10-06） | 09-09 → 10-06 | `#4516` EP ready nodes、`#4598`+`#4664` torchft、`#4952` 删 EP chunking、`#4842` functionalize、`#4617` `prepare_cp_metadata`、`#4897` metadata 所有权、`#4639` attention 包化 + `AttentionMetadata(Map)` | `experiments/graph_trainer/{ep_ready_nodes_dedup,ep_chunk_concretization,ep_shape_live_out}.py`、`experiments/torchft/{checkpoint,manager}.py`、**整个 metadata-seam 回移** | CP 变换重锚到 `prepare_cp_metadata(metadata, *, permutation)` / `shard_inputs`；`distributed/context_parallel.py` 收缩 | A ≈230；B ≈14/36 |
| **L3**（重组段） | `eb92f645c`（10-13，HEAD） | 10-09 → 10-13 | `#5277` 目录重组、`#5280` 配置拆分、`#5279` DistMuon dtype、`#5282` EMA 主机内存、`#4772` `BatchWiseLoadBalanceLoss`、`#4835` `SparseIndexerLoss`、`#5307`/`#5296` Varlen 默认、`#5114` checkpoint states | `components/{ema,optimizer,metrics,validate}.py`、`models/common/{aux_loss,linear,moe}.py`、`hf_datasets/multimodal/mm_collator.py` 等最后一批 | 10 个消失/改名 host 的落位迁移；`trainer.py` patch 重锚（`patch_train_step` 的 `inspect.getsource`+`exec` 最脆） | A ≈190；B ≥28/36 |

> 阶梯的段边界选的是**上游能力成簇落地**的时间点，不是"我们想清理多少"。若资源紧张，可退化为 **2 段**（≈09-30 与 HEAD）；若 L3 预检发现硬阻断，可拆成"`#5277` 纯重命名段 + 之后语义段"变成 4 段。
>
> 规模参考（自分叉点 `48cb7ae1f` 起累计的上游提交数，已核对每个目标修订确实包含该段列出的全部 PR）：L1 = 157（+157）、L2 = 455（+298）、L3 = 582（+127）。

---

## 3. 每段详细步骤

### 3.0 L0 — 不换 pin 的减法与防护（当前正在做）

**内容**

1. **退役 `full_dtensor` 这个 patch（不是后端）**：
   - 事实：我们的 `patches/torchtitan/distributed/full_dtensor.py` 把上游 `validate_config` 整体替换为"只在 CP + SDPA 时报错"的窄版本（有意放行 Varlen，因为 `patches/torchtitan/distributed/varlen_cp.py` 支持 CP+Varlen），全仓唯一调用点是 `models/deepseek_v4_1/parallelize.py:170`（0.3.0 上游代码从不调用它，`components/validate.py` 只调 `parallelize_inputs`）；
   - **更正（L0 实测）**：0.3.0 上游**确实**调用 `validate_config` —— llama3 / qwen3 / qwen3_5 / deepseek_v3 / gpt_oss / kimi_k2_7 / muse_glimmer 共 7 个模型模块在自己的 `parallelize` 里调用，且上游版本**连 `VarlenAttention` 一起拦**。因此"删 patch + 只在本仓调用点加守卫"会破坏本仓的 Varlen-CP 支持（如 qwen3.5 CP）；
   - 处置（已落地）：patch 文件删除，但窄版谓词保持**全局生效** —— 搬进 `patches/torchtitan/distributed/context_parallel.py`（与 varlen CP 支持同处）为 `reject_cp_with_scaled_dot_product_attention`，并在该 patch 的 `apply()` 里装回 `full_dtensor.validate_config`（带 `require_attrs` 防护，pin 移动后自动降级为"不覆盖"）；V4.1 直接调用同一函数，不再依赖 import 期绑定。同时删除激活行与 `.ci/architecture_import_baseline.json` 条目；`resolve_fsdp_mesh` / `resolve_sparse_fsdp_mesh` 导入保留；
   - **不做**"给 full_dtensor 加全局 fail-fast"：`full_dtensor` 在 0.3.0 仍是可选 Literal，是否可达取决于各脚本里 `${SPMD_BACKEND}` 的取值，需要逐处登记而不是一刀切。
2. **`spmd_backend` 43 处引用：只做分类登记，不清零**（更正早先判断）：
   - `spmd_backend` 在 0.3.0 是**活的、配置驱动**的开关：`torchtitan/trainer.py:319` 会 `set_spmd_backend(config.parallelism.spmd_backend)`，`experiments/graph_trainer/precompile_main.py:54`、`experiments/rl/actors/trainer.py:139`、`experiments/rl/models/vllm_wrapper.py:291` 同样写入；
   - `partial_dtensor` 是 graph trainer 的默认值（`experiments/graph_trainer/configs.py:246`），`components/loss.py:43` 直接断言 `get_spmd_backend() == "partial_dtensor"`，本仓示例脚本也显式传 `partial_dtensor`；
   - 我们自己的 `models/deepseek_v4/parallelize.py` 有 `assert get_spmd_backend() != "spmd_types"` —— V4 GraphTrainer 目前**依赖**非 spmd_types 后端；
   - 因此清零会改变 graph trainer / partial_dtensor 路径的行为，必须与"GraphTrainer 迁到 spmd_types 语义"绑定，属 L1 的前置决策（见 §3.1）。L0 只产出"逐处分类清单"（真常量 vs 随 backend 变化）。
3. **auto-load patch 的存在性防护**：`experiments/graph_trainer/{ep_chunk_concretization,ep_shape_live_out,ep_overlap_shape_queries,ep_ready_nodes_dedup,chunked_loss}` 与 `graph_trainer_runtime_context` 在包导入期就会 apply；其中前两者模块级 `import ...ep_chunk_pass`（L2 会被上游删除），runtime context 在目标缺失时静默 no-op。要求：只捕获针对特定上游模块的 `ModuleNotFoundError`，warning 写明缺失模块 + patch 名，跳过激活，不半激活、不吞无关错误。
4. **元数据补齐**：7 个缺少 `Upstream-PR:` 家族标记的 patch 文件补头（`scripts/check_architecture_boundaries.py` 会检查）；把度量 A 的复现脚本落到 `scripts/`。
5. **形状对齐重写**（为 L1–L3 的"删文件"做准备）：EMA 移向 `components/optim/ema.py` 的形态；`moe.py` 改为"子类化上游 router + 只覆盖 `_select_experts`"；`rope` 改为调用侧切片。**预检更正**：`models/common/aux_loss.py` 在 0.3.0 与 L1 rung **都不存在**（上游 09-10 的 `#3864` 才加入），所以 aux_loss 的签名对齐（`inject(raw_sum, *, carrier, denominator)`）**不属 L0/L1，归 L2**。

**退出判据**：`full_dtensor.py` patch 已退役且 CP+SDPA 窄守卫仍生效；auto-load patch 在上游模块缺失时给出可读 warning 而非 ImportError；`spmd_backend` 43 处产出逐处分类清单（**不要求归零**）；CPU UT 新增失败 = 0。

**L0 结果（2026-10-14，分支 `exp/patch-cleanup-w0`，未 push）**

- commit `7b9902e7`：退役 `patches/torchtitan/distributed/full_dtensor.py`，窄守卫迁入 CP patch 并保持全局生效（激活表 + baseline 各 −1 行）；
- commit `6ecb73c1`：`patches/_guard.py`（`require_upstream` / `require_attrs` / `require_signature`）+ 6 个 patch 的锚点防护（4 个 auto-load graph_trainer EP patch + `chunked_loss` + `dist_muon`）+ 7 个缺失 `Upstream-PR:` 头 + `scripts/measure_upstream_coupling.py` + `tests/unit_tests/patches/test_guard.py`；
- 验证：改动前后按目录各跑一轮，`extensions` 2F/319P、`models` 27F/557P、`override` 3F/211P 的**失败用例名逐条相同**（预存在环境问题）；`patches` 50P → 57P；**新增失败 = 0**；ruff + 架构边界检查通过；
- 报告：`torchtitan-npu-w0/.agents/notes/proposed/upstream/2026-10-14-l0-patch-cleanup.md`（107 行，含逐处 `spmd_backend` 分类、防护设计、未决项）；
- 未决：`spmd_backend` 未清零（授权内，卡 GraphTrainer 迁移）；未加全局 fail-fast（授权内）。

### 3.1 L1 — 地雷段 + 机械段（目标 `dc3985ad4`，2026-09-05）

**吸收的上游能力**（按落地时间）

| 日期 | PR | 内容 | 对我们的影响 |
| --- | --- | --- | --- |
| 08-19 | `#4217` | 删除 full_dtensor 后端 → main 上 `partial_dtensor` 字符串也**完全消失**（`git grep partial_dtensor origin/main` 为空） | **本段最大新增项**：我们的 GraphTrainer/V4 路径依赖非 spmd_types 后端（`models/deepseek_v4/parallelize.py` 的 `assert get_spmd_backend() != "spmd_types"`、`components/loss.py:43` 的 `partial_dtensor` 断言、`examples/*graphtrainer*.sh` 的显式传参），必须先把 GraphTrainer 迁到 spmd_types 语义 |
| 08-19 | `#4088` | grain 取代 dataset/dataloader | `hf_datasets/text_datasets.py` 的 host 契约变化 |
| 08-20 | `#4239` | multimodal collator 一维契约 | `hf_datasets/multimodal/mm_collator.py` 必须重写/删除 |
| 08-20 | `#4122` | DistMuon 存储分片 | `flex_shard/dist_muon.py` 锚点核对（正是"静默失败"风险点） |
| 08-21 | `#4186` | `BaseCheckpointManager` storage seam | `components/checkpoint.py` patch 收缩 |
| 08-26 | `#4116` | `Decoder.preprocess_inputs` | metadata 接缝回归上游 → `models/common/decoder.py` 大幅收缩 |
| 08-26 | `#4265` | spmd_types 0.2.5，删 `SpmdLayout` | 3 个 `sharding.py` 必须迁移 |
| 09-03 | `#3634` | 上游 DeepSeek-V4 + rope 契约 | `models/common/rope.py` 的 key/inverse 半可删 |
| 09-05 | `#4474` | 部分初始化 optimizer state 的 checkpoint 修复 | `components/optimizer.py` Part A 可删 |

**同段我们侧改动**：`spmd_backend` 选择器整体退役（上游已删，含 `partial_dtensor`）→ GraphTrainer 及其断言/示例脚本一并迁移；`sharding.py` 去 `SpmdLayout` 并把 `torchtitan.protocols.sharding` 改成当时的路径；`decoder.py` 收缩为只做"能力补丁"；DistMuon/`mm_collator` 的契约对齐。**预检更正**：`components/{checkpointer/,optim/}` 与 `distributed/context_parallel/` 的重组在 L1 rung **就已经完成**（不是 `#5277`/L3），所以这三处 host 迁移的工作量记在 **L1**；L3 只剩 `validate→validator`、`hf_datasets→components/data`、`protocols→models/base`、`metrics→observability`。

**前置条件（L1 预检结论，2026-10-14）**

1. **按 sha 取上游必须先解决**（见 §1.2，唯一阻断项）；
2. 依赖：L1 rung 的声明里**没有任何 git pin**，全部可从 PyPI 取 —— `grain==0.2.18`、`spmd_types==0.2.5`、`torch_checkpointing@12b3c57`、`torch_remat 0.2.0`、`renderers 0.1.11`、`pyarrow 25.0.1`；`attn-gym` **保持我们的 0.0.16**（它是 rung 要求的 0.0.8 的严格超集，rung 用到的算子两版一致）。注意 `spmd_types 0.2.5` 删了 `SpmdLayout`，**必须与换 pin 同时落地**，不能在 0.3.0 上先升；
3. **GraphTrainer 后端迁移**（`partial_dtensor` → spmd_types 语义，需 NPU 数值验证）；
4. torch 2.14.0 兼容性：预检**没有发现 ① 类（API 缺失）失败**（该 rung 自带 CPU UT 7F/721P/47S/8E 全部归类为缺依赖或环境问题，`test_trainer.py` 单跑 11 passed）→ 残余风险集中在 **NPU 面**（torch_npu 2.14.0.dev vs 上游 CI 的 torch nightly）。

### 3.2 L2 — 架构段（目标 `3f087cf15`，2026-10-06）

**吸收的上游能力**：`#4516` EP ready nodes 去重、`#4598`/`#4664` TorchFT 修复、`#4952` 删除图式 EP chunking、`#4842` functionalize 变更保存、`#4617` CP 输入/元数据分片解耦（`prepare_cp_metadata` + `shard_inputs`）、`#4897` attention metadata 所有权下放到 backend、`#4639` attention 包化 + `AttentionMetadata`/`AttentionMetadataMap`。

**预检更正（两条时间线）**：`models/common/aux_loss.py` 上游到 **09-10 的 `#3864`** 才出现（0.3.0 与 L1 rung 都没有）→ aux_loss 的形状对齐/整体替换归**本段**；我们 patch 里的 `ep_chunk_concretization.py` / `graph_trainer_runtime_context.py` 是**我们自己的文件名**，其上游 target（`experiments/graph_trainer/{ep_chunk_pass,ep_pass_utils,passes,registry,common_utils,trainer}.py`）里 `ep_chunk_pass.py` **在 L1 rung 仍存在**，被 `#4952` 删除 → "导入期炸弹"属于**本段**，L0 的 `require_upstream` 防护方向与目标名已核对正确。

**删除/收缩**：`ep_ready_nodes_dedup.py`、`torchft/{checkpoint,manager}.py`、`ep_chunk_concretization.py`、`ep_shape_live_out.py`；**整个 metadata-seam 回移**（`patches/torchtitan/models/common/attention_metadata.py` + `decoder.py` 里自造的那部分）在本段消失，改为使用上游 `build_attention_metadata` / `attention_metadata_key` / `prepare_cp_metadata`。

**同段我们侧改动**：CP 变换从"读 metadata 对象"改为按上游签名调用 `prepare_cp_metadata(metadata, *, permutation)`；`distributed/context_parallel.py` patch 收缩为只保留 NPU 差异。

**与 Step 2 的关系**：当前的 metadata-seam step1/step2 工作正是"把形状改成上游形状"，本段做的是"删掉回移"。两件事必须一起看，但**不要**在 L0/L1 里动它。

### 3.3 L3 — 重组段（目标 `eb92f645c`，HEAD）

**吸收的上游能力**：`#5277` 目录/依赖/文档重组、`#5280` sharded model configs + RL generator 拆分、`#5279` DistMuon dtype（`ComputeLayout.compute_dtype`）、`#5282` CPU-offload EMA 重写、`#4772` `BatchWiseLoadBalanceLoss` + memory policy "Pass 1b"、`#4835` 上游 `SparseIndexerLoss(AuxLoss)`、`#5307`/`#5296` 默认 Varlen attention、`#5114` checkpoint `extra_states`。

**删除/收缩**：`components/{ema,optimizer,metrics,validate}.py`、`models/common/{aux_loss,linear,moe}.py`、`hf_datasets/multimodal/mm_collator.py` 等最后一批（详见附录 B 的逐 patch 工单）。

**同段我们侧改动**：host 落位迁移 —— **预检更正**：`checkpointer/`、`components/optim/`、`distributed/context_parallel/` 的重组在 **L1 rung 就已完成**，L3 只剩 `validate→validator`、`hf_datasets→components/data`、`protocols→models/base`、`metrics→observability/metrics.py`；`trainer.py` patch 重锚（`patch_train_step()` 的 `inspect.getsource` + `exec` 是最脆的一处，建议在 L2 就改成不依赖源码文本的形态）；`ComputeLayout.compute_dtype` 带来的 dtype 语义变化要在 flex_shard 里显式处理。**另：L3 的 rung 必须在开工前重新冻结** —— 镜像 main 在预检时已前进到 `58a3027d9`（`#5392`），`eb92f645c` 不再是 tip。

---

## 4. 每段的统一验证流程

1. **静态**：`ruff check --line-length 120`（改动的文件）+ `python scripts/check_architecture_boundaries.py`；激活顺序变化要同步 `.ci/architecture_import_baseline.json`。
2. **CPU UT 对照**：按目录跑（整包会 OOM），`TORCH_DEVICE_BACKEND_AUTOLOAD=0`；与**上一段**的失败集逐一对比，判据是"新增失败 = 0、环境性失败集完全一致"。预存在的环境问题（如 `torchvision::nms` 缺失导致的两个 collection error）单独列出。
3. **补丁自测**：每删一个 patch，先删/改它的单测，并加"该 patch 不再激活"的断言，避免删除后测试静默跳过。
4. **NPU 阶梯**：小配置 smoke → 短步数 ST → 与上一段做 loss/数值对比（bitwise 仅在随机性可控时要求，否则看曲线与关键中间量）；checkpoint 往返 + HF 适配器 + engram/多模态数据通路（L1 动过 collator）。
5. **冻结 sha**：每段的目标 sha 一旦开工即冻结，段内不追 HEAD；段末记录 A/B 指标与遗留项。

---

## 5. 硬门槛与风险

1. **依赖要能被内网安装（L3/main 的问题，不是 L1 的）**：**L1 rung（`dc3985ad4`）的依赖声明里没有任何 git pin**，全部可从 PyPI 取（`grain==0.2.18`、`spmd_types==0.2.5`、`torch_checkpointing@12b3c57`、`torch_remat 0.2.0`、`renderers 0.1.11`、`pyarrow 25.0.1`；`attn-gym` 保持我们的 0.0.16，它是 rung 要求的 0.0.8 的严格超集）。git pin（`spmd_types@a1b86deadc`、`torch_remat@624f19cb`、`torch_checkpointing@dcbb3a59`）是 **main/L3** 才引入的，届时需要镜像/OBS 或 vendored wheel。另：`torch_checkpointing` 要挑修订——PyPI 0.1.0 与 main 的 pin 都缺 `distributed_metadata.METADATA_FILE_NAME`（10-07 被搬走），`12b3c57`(09-03) 有，实测 `@12b3c57` + `spmd_types 0.2.5` 下 checkpoint 测试 84 passed。
2. **torch 版本双轨**：`requirements.txt` 是 `torch==2.14.0` + `torch_npu 2.14.0.dev20260918`，而 CI 镜像给的是 `torch 2.15.0.dev`。上游不 pin torch，因此真正的门槛是"上游代码是否用到 >2.14 的 API"——**每段开工前先做 torch API 兼容性预检**，这是整条阶梯最大的不可控项。
3. **段间重复 re-anchor**：靠"每段只碰该段窗口内 host 文件变过的 patch"控制；§2 的批次已按此原则切分。
4. **auto-load 炸弹**：`ep_chunk_concretization` / `ep_shape_live_out` / `graph_trainer_runtime_context` 是导入期生效的，没有 L0 的防护时 L2 会在 import 阶段炸且难以定位。
5. **静默失败点**（必须配锚点断言）：`flex_shard/dist_muon.py`（无版本校验，HEAD 用 `BlockShard(block_sizes=…)`，我们传 `block_size`）、`hf_datasets/multimodal/mm_collator.py`（上游一维契约）、`graph_trainer_runtime_context.py`（`apply()` 静默 no-op）、`ep_chunk_*`（auto-load + 模块级 import）。
6. **bump 期硬错误**：`full_dtensor.py` 模块被删（L1）、`ComputeLayout.compute_dtype`（L3）会让 `optimizer_reshard` 撞上硬 `AttributeError`、`set_sharding_` 被删（`#5280`）。
7. **未验证项**：所有数值结论（`ClampedSwiGLU` vs 我们的 `silu`+clamp、变长 `block_sizes` 的 Muon 更新、EMA 语义、aux_loss 的 `reduce_mesh` `"batch"`→`"dp"`）目前都只有静态证据，**尚未在 NPU 上跑过训练**；每段都要安排数值验证。
8. **已作废的估算（L0 实测更正）**：早先"清空 `spmd_backend` 分支 → A1 −104"的判断**不成立**。`spmd_backend` 是活开关（`torchtitan/trainer.py:319` 写入；graph trainer 默认 `partial_dtensor`；`components/loss.py:43` 断言它；我们的 `models/deepseek_v4/parallelize.py` 要求 `!= "spmd_types"`）。因此 A 的下降主要靠 L1–L3 的换 pin，而不是 L0 的减法；清零动作必须与 **GraphTrainer 后端迁移**绑定并做 NPU 数值验证。
9. **阶梯唯一的硬阻断：按 sha 取上游**（L1 预检实测）—— gitcode 镜像不接受 `git fetch --depth 1 origin <sha>`，但持有对象、`clone --filter=blob:none --no-checkout` + `checkout --detach <sha>` 可用。修法见 §1.2（推荐改我们自己的 `.ci` 脚本，兼顾 `setup_torchtitan.sh` 的 `TORCHTITAN_COMMIT` 覆盖）。
10. **torch 2.14.0 已通过静态/UT 预检，但 NPU 面未验证**：该 rung 的 CPU UT 里 **① 类（torch API 缺失）失败 = 0**，残余风险集中在 torch_npu 侧（`2.14.0.dev` vs 上游 nightly ≈ 2.15.0.dev 的算子/语义差异）；另外共享 venv 的 `torch_npu` 实际是 `dev20261001`，与 `requirements.txt` 写的 `dev20260918` 不一致，**需要对齐**（否则"我们验证的是哪个 torch_npu"说不清）。

---

## 6. 附录 A：不换 pin 的波次与逐模块工单（原 `2026-10-14-backport-plan.md` 全文）

## torchtitan pin 回移计划（补丁清理 / 刷新）

- 计划日期：2026-10-14
- 我方基线：worktree `/home/lrwei/cann/torchtitan-npu-metadata-seam`，分支 `exp/metadata-seam-step2`，HEAD `0110132d`
  （含 `84816a3a` = #4897+#4116 元数据 seam、`0110132d` = CP 路径 metadata map；这两个提交的后续迁移见 §0）
- pin：`torchtitan==0.3.0`（`requirements.txt:12`），上游 tag `v0.3.0` = `086bf6c16`（2026-09-02），源码 `/home/lrwei/pytorch/torchtitan-v030`，安装副本 `/home/lrwei/pytorch/venv/3.11/lib/python3.11/site-packages/torchtitan`（`torchtitan-0.3.0.dist-info`）
- 上游对照：`/home/lrwei/pytorch/torchtitan` @ **`origin/main` = `eb92f645c`**（`[CI] Preserve NumPy 1.x support by constraining PyArrow (#5372)`）
  - 审计基线 `3386c6d6f` → `eb92f645c` 共 **74 个提交**（`git rev-list --count 3386c6d6f..origin/main` = 74）
  - `/tmp/patch_audit_delta.md` **不存在**（已确认），74 个提交的增量由本计划自行核对，下文凡引用 `eb92f645c` 的行号均为本轮实测
- 只读声明：本计划未修改上游与我方任何仓库文件；唯一写入物是本文件
- 计数口径：所有 `file:line` 均为实测（`grep -n` / `git grep -n <pat> <commit> -- <path>` / `git show <commit>:<path> | sed -n`）；凡未能核实的条目显式标注 **未验证**

---

### 0. 本轮范围与非目标

**本轮只做一件事：把 `torchtitan_npu/patches/torchtitan/` 下的每个 patch 模块归类，并给出"现在删 / 现在刷新 / 加守卫 / 推迟"的具体动作与同 PR 必须一起改的我方调用点。**

明确排除（**不要在本计划里排期**）：

- **DeepSeek metadata seam Step 2**（V4 / V4.1 / V3.2 迁到 `build_attention_metadata` / `prepare_cp_metadata`，删除 `MetadataExtension` / `LightningIndexerMetadata` / `build_attention_masks`）→ 归独立的 Step 2 工作流。
  本计划中下列 3 个 patch 模块**整体归属 Step 2**，只在 §3.0 各留一行：
  `torchtitan_npu/patches/torchtitan/models/common/attention_metadata.py`、
  `torchtitan_npu/patches/torchtitan/models/common/decoder.py`、
  `torchtitan_npu/patches/torchtitan/components/validate.py`。
- 训练/数值本身的改动（模型结构、算子、并行策略）不在本轮范围；本轮的产物是**补丁面的收敛**。

非目标（本轮不做）：pin bump 本身、`requirements.txt` 版本号变更、`torchtitan_recipes` 迁移。

---

### 1. 判据与度量

#### 1.1 度量 A：我方代码的上游耦合度

**定义**：`torchtitan_npu/**/*.py` 中引用"因 0.3.0 pin 才成立"的名字或形状的 `(file, line)` 命中数。分两类：

- **A1 = pin-only 上游名字**：在 `v0.3.0` 存在，在 `eb92f645c` 被删除/改名/改路径。
- **A2 = 本仓自造的 0.3.0 形态 seam**：在 `v0.3.0` 与 `eb92f645c` **都不存在**，但它的契约按 0.3.0 的形状写死（例如 `BatchedLinear` 的 `[H*D_out, D_in]` 权重、`build_attention_masks` 的 `(inputs, labels, extra_kwargs, cp_mesh, load_balancer_type)` 签名）。

**计数规则（可复现）**：对固定正则做 `grep -n`，**含注释、docstring 与派生类名**，因此是**上界**（语义站点数更少）。脚本见本文件 §6 附录；同一脚本在每波结束时重跑，数字只用于**波次间比较**，不用于绝对承诺。

**实测基线（@`0110132d`，A = 513）**

| 度量 | 值 | 主要构成（命中数 / 代表性 `file:line`） |
|---|---|---|
| A1 pin-only | **239** | `inner_experts` 31、`GroupedExperts` 30、`get_spmd_backend` 30、`VarlenMetadata` 30、`torchtitan.components.optimizer` 28、`full_dtensor` 19、`spmd_backend` 12、`AttentionMasksType` 12、`checkpoint_utils` 8、`hf_datasets.text_datasets` 8、`post_dataloading_process` 7、`cp_shard` 6、`BlockShard(block_size=)` 6、`parallel_dims` 模块 5、`build_dist_muon` 5、`set_spmd_backend` 1、`components.validate` 1 |
| A2 自造 seam | **274** | `swiglu_limit` 38、`MetadataExtension` 37、`vision_enabled\|bias_vl` 27、`BatchedLinear` 24、`CPVarlenMetadata` 22、`build_attention_masks` 22、`HashRouter` 17、`Split*RoPEConfig` 13、`EMA_OPTIMIZER` 10、`prepare_context_parallel_input` 10、`_Clamp*` 9、`set_step_denominator` 7、`EMACheckpointManager` 7、`MetricsProcessor.log` 7、`RoPE.split` 6、`EMAOptimizersContainer` 6、`sqrtsoftplus` 6、`HashMoE` 3、`EMATrainer` 2、`RouterConfig.hash=` 1 |
| **A 合计** | **513** | A1 239 + A2 274 |

**存在性矩阵（本轮实测，`git cat-file -e <rev>:<path>`）**：

| 名字 / 路径 | `v0.3.0` `086bf6c16` | `3386c6d6f` | `eb92f645c` |
|---|---|---|---|
| `torchtitan/distributed/full_dtensor.py` | OK | **GONE** | **GONE** |
| `torchtitan/components/metrics.py` | OK | **GONE** | **GONE** |
| `torchtitan/components/optimizer/` | OK | **GONE** | **GONE** |
| `torchtitan/distributed/parallel_dims.py` | OK | **GONE** | **GONE** |
| `torchtitan/components/checkpoint_utils.py` | OK | **GONE** | **GONE** |
| `torchtitan/components/validate.py` | OK | OK | **GONE** → `components/validator.py` |
| `torchtitan/hf_datasets/text_datasets.py` | OK | OK | **GONE** → `components/data/text_datasets.py` |
| `GroupedExperts`（全树） | 91 命中 | — | **0** |
| `spmd_backend` / `get_spmd_backend` / `full_dtensor`（全树） | 194 / — / 77 | — | **0 / 0 / 0** |
| `build_attention_masks` / `MetadataExtension` / `set_step_denominator` | **0**（自造） | — | **0 / 0 / 0**（仅 `.claude/skills/local_compile_perf/references/harness.md:55` 过期文档提到 `set_step_denominator`） |
| `cp_shard`（公开函数） | `context_parallel/api.py:166` | — | **仅 `_cp_shard_dims`（`distributed/context_parallel.py:171`）与 `Decoder._cp_shard`（`models/common/decoder.py:334`）** |
| `AttentionMasksType` / `VarlenMetadata` | `models/common/attention.py:86` / `:71` | — | **0 / 0**（改为 `AttentionMetadataMap`（`models/common/attention/__init__.py:34`）/ `VarlenAttentionMetadata`（`models/common/attention/attention.py:76`）） |
| `post_dataloading_process` | `trainer.py:695`、`components/validate.py:139` | — | **0** |
| `BlockShard` 字段 | — | — | `block_sizes: tuple[int, ...]`（`distributed/flex_shard/optimizer_reshard.py:36,58`，`892ecfed0` #4840 引入） |

> **口径更正（重要）**：任务书把 `build_attention_masks`、`MetadataExtension`、`set_step_denominator`、`BatchedLinear`、`CPVarlenMetadata`、`HashMoE`、`EMATrainer` 列为"只存在于 0.3.0 的名字"。实测它们**在 0.3.0 里也不存在**——它们是**本仓自造**的 seam（A2），契约按 0.3.0 形状写死。本计划按 A1/A2 分开计数，因为两者的消除手段不同：A1 靠"改 import / 改调用点"，A2 靠"改契约 / 改注入方式"。

#### 1.2 度量 B：补丁的可删度

**定义**：对 `torchtitan_npu/patches/torchtitan/` 下的 **36 个 patch 模块**（不含 11 个 `__init__.py`）逐个给出 bump 时的动作：

| 档 | 含义 | bump 时动作 |
|---|---|---|
| **D0 现在删** | 能力已不需要或已被上游删除；**本波内即可删除文件 + 激活行** | 已删 |
| **D1 bump 即纯删** | 能力上游已等价；pin 上仍需，但 bump 时"删文件 + 删 1 行激活"即可 | 删文件 + 1 行 |
| **R 现在刷新** | 本轮把它改成 **bump 时纯删**：对齐调用点 / 换成公开扩展点 / 把契约改成上游形态 | 删文件 + 1 行 |
| **X 推迟** | 上游仍是移动目标，或宿主被上游删除，现在无合法形态可对齐 | 重写 |
| **S2** | 归属 Step 2 元数据 seam 工作流 | 由 Step 2 决定 |

**实测基线（B@`0110132d`）：D1 = 5、D0 = 0、R = 0（今天全是"重写"）、X = 9/36 里含静默失败风险、S2 = 3、其余 19 个在 bump 时都需要重写或改写锚点。**
即：**今天"删文件 + 删 1 行"就能收工的模块只有 5/36（13.9%）**，这就是度量 B 的起点。

#### 1.3 每波如何同时压低 A 与 B

（下表数字按 §1.1 的正则口径线性相减，是**上界估计**；每波结束后用 §6 脚本实测替换。）

| 波 | 对 A 的作用（上界） | 对 B 的作用 |
|---|---|---|
| W0 | A 不变（只加守卫与注记） | D1 5 → 5，但**消除静默失败**：36 个模块全部获得锚点断言 |
| W1 | A1 −**104**：`get_spmd_backend` 30 + `full_dtensor` 19 + `spmd_backend` 12 + `set_spmd_backend` 1 + `torchtitan.components.optimizer` 28（路径改名）+ `checkpoint_utils` 8 + `parallel_dims` 5 + `components.validate` 1 | D1 5 → 10（`full_dtensor` 变 D0；`metrics`/`optimizer(A)`/`dist_muon` 变纯删） |
| W2 | A2 −**39**：`EMA_OPTIMIZER` 10 + `EMACheckpointManager` 7 + `set_step_denominator` 7 + `EMAOptimizersContainer` 6 + `EMATrainer` 2 + `MetricsProcessor.log` 7（后者的 3–4 处迁往 `extensions/` 后仍计入 A2，但不再是 patch 面） | D1 10 → 17（`checkpoint`/`ema`/`trainer`/`metrics`/`optimizer(A)`/`functionalize`/`memory_policy` 变纯删） |
| W3 | A1 −**66**：`inner_experts` 31 + `GroupedExperts` 30 + `build_dist_muon` 5；A2 −**117**：`swiglu_limit` 38 + `BatchedLinear` 24 + `HashRouter` 17 + `Split*RoPEConfig` 13 + `_Clamp*` 9 + `sqrtsoftplus` 6 + `RoPE.split` 6 + `HashMoE` 3 + `RouterConfig.hash=` 1。**口径说明**：`swiglu_limit` 的 38 命中里只有与 `_Clamp*` 工厂/Config 注入绑定的部分算 A2 —— 改挂 `RoutedExperts.Config.activation_fn` 后该名字在上游也存在（`activation.py:179`），不再计入 A2 | D1 17 → 22（moe/linear/rope/dist_muon/mm_collator/vision_encoder 变纯删） |
| W4 | 不动（这些正是"暂不回合"的项） | 明确"为什么不删"，避免误删 |

目标终态：**A ≈ 190（513 − 104 − 39 − 183），剩余全部是 W4 明确推迟的项与 Step 2 的 metadata 项；D 档 ≥ 22/36。**

---

### 2. 波次表

#### W0 — 守卫与元数据补齐（不动契约，1 个 PR，可立刻做）

| 项 | 内容 |
|---|---|
| **目标** | 让 bump 时每个 patch 模块**要么生效、要么显式报错**，绝不静默误装；补齐 patch 元数据债务 |
| **我方需要改动的文件与调用点** | ① 3 个 "import 即崩" 的 graph_trainer 模块：`torchtitan_npu/patches/torchtitan/experiments/graph_trainer/ep_chunk_concretization.py:99`、`ep_shape_live_out.py:23`（模块级 `import ...ep_chunk_pass`）、`ep_forward_accumulation.py:85` 加 `importlib.util.find_spec("torchtitan.experiments.graph_trainer.ep_chunk_pass")` 守卫（`ep_chunk_pass.py` 已被 `43f73649f` #4952 删除）；<br>② **锚点断言**：`patches/torchtitan/distributed/flex_shard/dist_muon.py:565 apply()`、`patches/torchtitan/hf_datasets/multimodal/mm_collator.py:141 apply()`、`patches/torchtitan/experiments/graph_trainer/graph_trainer_runtime_context.py:150 apply()`、`patches/torchtitan/components/checkpoint.py:74 apply()`、`patches/torchtitan/components/optimizer.py:92 apply()`；<br>③ 补 `Upstream-PR:` 头（`.ci/architecture_import_baseline.json` 的 `patch_metadata.missing` 实测 7 项）：`flex_shard/{dist_muon,optimizer_reshard,_optimizer_reshard_schedule,replica_dedup}.py`、`experiments/graph_trainer/functionalize_recompute_mutations.py`、`hf_datasets/text_datasets.py`、`models/common/rope.py`；<br>④ 更新 `.ci/architecture_import_baseline.json` 的 `activation_order["torchtitan_npu/patches/torchtitan/__init__.py"]` 与 `patch_metadata.missing`；<br>⑤ 每个模块头补 "origin/main 锚点注记"（file:line + commit） |
| **上游目标形态（`eb92f645c`）** | 断言所引用的对象：`dist_muon.py` 的 `_LoweredComputeShader` 别名在 `distributed/flex_shard/dist_muon.py:885,889`；`mm_collator` 的 `_build_mrope_positions` 在 `components/data/multimodal/mm_collator.py:189`（**1-D 契约**：`tokens: (num_tokens,)`、`:230 tokens.unsqueeze(0)`、`:312` 区域注释）；`TagSAC` 的 `tag_sac_policy` 在 `experiments/graph_trainer/memory_policy.py:282` |
| **验证方式** | CPU UT：`tests/unit_tests/patches/torchtitan/experiments/graph_trainer/{test_ep_chunk_concretization,test_ep_forward_accumulation,test_graph_trainer_runtime_context}.py` 增加"守卫触发即 skip/明确报错"的断言；tooling UT：`tests/unit_tests/tooling/test_architecture_boundaries.py`（消费 `scripts/check_architecture_boundaries.py:95,98` 的元数据规则）；全量 `pytest tests/unit_tests -q` |
| **依赖顺序** | 无前置；W1–W3 的前置 |
| **预计下降量** | A：0（不变）；B：0 个模块变纯删，但 **36 个模块从"静默失败"变"显式失败"**——这是后续所有波次的安全网 |
| **风险与回退** | 断言过严会在合法的小版本差异上误报 → 断言只查"被替换符号存在 + 形参名/属性名集合"，不查函数体；回退 = 删除断言块（守卫模块本身不动） |
| **最高杠杆动作** | **在 `apply()` 里加"锚点断言"**：一条统一规则同时挡掉 `dist_muon` 的静默回退（会把 `eb92f645c` 的 4 个函数覆盖回 `6ee6b10b`）、`mm_collator` 的静默 2-D 误装（`_EXPECTED_PARAMETERS` 只比形参名，而 `eb92f645c` 形参名完全相同 → 会静默把 1-D 数据当 2-D 处理）、`graph_trainer_runtime_context` 的静默 no-op |

#### W1 — 命名 / 路径 / 开关对齐

| 项 | 内容 |
|---|---|
| **目标** | 把所有"只是改了名字或模块路径"的耦合一次性消掉；其中 **`full_dtensor` 后端退场**是本波的主战场 |
| **我方需要改动的文件与调用点** | **① `full_dtensor` + `spmd_backend` 退场（D0）**——我方只有一处 *设置* `spmd_backend`：`models/deepseek_v4/config_registry.py:411`（且值为 `"spmd_types"`），其余全是防御性分支，可整体删除：<br>　删除 `patches/torchtitan/distributed/full_dtensor.py` + `patches/torchtitan/__init__.py:9` 的 `full_dtensor`；<br>　`override/qwen3_5/parallelize.py:15`（import）、`:87-88`（`raise NotImplementedError`）、`:102,120,242,256`；<br>　`models/deepseek_v4_1/parallelize.py:26`（`from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh, validate_config`）、`:169,182`；<br>　`override/deepseek_v4/moe.py:21,52`；`override/common/swiglu_group/ascendc.py:18,71`；`override/common/token_dispatcher.py:51,207,217,220,332,347`；<br>　`models/deepseek_v4/{mtp.py:29,109, parallelize.py:15,146, config_registry.py:411, lora.py:26,227, token_dispatcher.py:68,603}`；`models/deepseek_v3_2/model.py:17,83,364,382`；`extensions/torchtitan/experiments/torchft/trainer.py:101`（`dist_utils.set_spmd_backend`）；<br>　`patches/torchtitan/models/common/moe.py:43,384`、`patches/torchtitan/models/common/token_dispatcher.py:27,197,206,209,318,326`；<br>　`patches/torchtitan/models/common/decoder.py:43,203-205`（**Step 2 文件，但与 full_dtensor 分支交叉，改动需与 Step 2 协调**）；<br>　文档 `docs/feature_guides/ep_process_group.md:26`；测试 `tests/unit_tests/models/deepseek_v4_1/test_model_split.py:183`、`tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py`、`tests/unit_tests/override/common/test_moe_score_absorption.py`、`tests/unit_tests/override/common/test_swiglu_group.py`、`tests/unit_tests/extensions/adaptor/torchao/test_converter.py`。**并在 `models/deepseek_v4_1/parallelize.py` 用本地校验函数承载原来由 full_dtensor patch 提供的 "CP + SDPA 不支持" 断言**（上游删了该模块，见 `601cf4d23` #4217 / `5dd944e62` #4218）。<br>**② 模块路径改名**：`patches/torchtitan/components/metrics.py:11` → 双路径解析（`torchtitan.observability.metrics` 优先）；**同一处改名也要落在 `extensions/torchtitan/components/metrics.py:19,22,35`**（易漏）；`patches/torchtitan/components/optimizer.py:18-19`（`torchtitan.components.checkpoint_utils` / `torchtitan.components.optimizer` → `torchtitan.components.optim.{utils,optimizer}`）；`patches/torchtitan/components/validate.py:28`（→ `torchtitan.components.validator`）；`models/{common/muon.py:12, deepseek_v4/config_registry.py:25, deepseek_v4/sharding.py:9, deepseek_v4_1/config_registry.py:23, deepseek_v4_1/sharding.py:15}`（`torchtitan.distributed.parallel_dims.MeshAxisName` → `parallelism_context`）；`torchtitan_npu/tools/checkpoint_conversion/convert_to_hf.py:30`（`checkpoint_utils`）。**注意**：`torchtitan.hf_datasets.text_datasets` 的 8 处命中（如 `override/deepseek_v4/chat_dataloader.py:11`）**不放在本波** —— 它不是改名而是整套 API 消失（grain 迁移），归 W4 |
| **上游目标形态（`eb92f645c`）** | `spmd_backend` / `get_spmd_backend` / `full_dtensor` 全树 **0 命中**；`annotate_input_spmd_types` 在 `distributed/spmd_types.py:180`（`49a07abcf` #5230 更新过）；`MetricsProcessor` 在 `observability/metrics.py:305`，属性是 `parallelism_context`（`:355,394`），`log(...)` 在 `:509`（`extra_metrics` 在 `:514` 附近）；`init_optim_state` 在 `components/optim/utils.py:29`；`OptimizersContainer.state_dict` / `load_state_dict` 在 `components/optim/optimizer.py:404,418`；`Validator` 在 `components/validator.py:54`；`MeshAxisName` 随 `parallel_dims.py` 一起消失（`5dc97a3e7` #4905）；`hf_datasets/` 整个目录在 `#5277`（`6f2225575`，2026-10-11）被重组 |
| **验证方式** | CPU UT：`tests/unit_tests/test_dsv4_muon.py`、`tests/unit_tests/models/deepseek_v4_1/test_model_split.py`、`tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py`；新增/改写 `tests/unit_tests/patches/torchtitan/components/test_optimizer_patch.py`（B 部分锚点）；全仓 `grep -rn "spmd_backend\|full_dtensor\|parallel_dims\b"` 应为 0（`torchtitan_npu/` 内）。NPU ST：`deepseek_v4_1` 与 `qwen3_5` 各一条最小 spmd_types 训练 smoke |
| **依赖顺序** | 依赖 W0 的守卫；`full_dtensor` 删除必须在 W2 之前（它是横切依赖） |
| **预计下降量** | A：**−104**（A1 239 → 135）。B：D1 5 → 10（`full_dtensor` 变 D0 删除，`metrics`/`optimizer(A)`/`dist_muon` 变纯删）。调用点删除：≈**60 处** |
| **风险与回退** | ① 若仍有用户在用 `full_dtensor`（pin 的 `configs.py:163` Literal 仍可选），删除后需要给出明确报错而不是静默走 `spmd_types` → 在 `models/deepseek_v4_1/parallelize.py` 保留一条 `if parallelism.spmd_backend == "full_dtensor": raise NotImplementedError`（qwen3_5 已有，`:87-88`）；② `MeshAxisName` 换了模块后，`tests/unit_tests/test_dsv4_muon.py:16`、`tests/unit_tests/models/deepseek_v4_1/test_muon_profile.py:17` 必须同步，否则测试收集期就 ImportError。回退 = 保留一层 `torchtitan_npu/models/common/_compat.py` 双路径 import 帮助函数 |
| **最高杠杆动作** | **`full_dtensor` / `spmd_backend` 整体退场**：一次消掉 A1 里 61 个命中（`get_spmd_backend` 30 + `full_dtensor` 19 + `spmd_backend` 12），删除 1 个 patch 模块，并把 bump 时必然 ImportError 的 `from torchtitan.distributed.full_dtensor import ...` 提前清干净 |

#### W2 — 契约 / seam 对齐（EMA、aux 分母、metrics、pp 桥、trainer/engine 拆分）

| 项 | 内容 |
|---|---|
| **目标** | 把"上游已吸收但机制不同"的模块改成 **bump 时纯删**；重点是拆掉 `patch_train_step()` 的 `inspect.getsource`+`exec` 注入 |
| **我方需要改动的文件与调用点** | **① 删 `patch_train_step()`**（`patches/torchtitan/trainer.py:117-143`，`apply()` 在 `:164`）：H2D 部分对应上游 `8792c3a24`（#4421）→ 落到 `components/data/types.py:38-44 to_input_dict(device, *, non_blocking=)`；aux 分母部分改为**显式 per-forward 契约**（见 ③）。<br>**② EMA 三件套**：`patches/torchtitan/components/ema.py:26,53`（`EMA_OPTIMIZER="ema_optimizer"` / `EMAOptimizersContainer`）、`patches/torchtitan/components/checkpoint.py:22,25,34,44,74-80`（`EMACheckpointManager`）、`patches/torchtitan/trainer.py:145,171`（`EMATrainer` + `torchtitan.trainer.Trainer = EMATrainer`）。调用点：`extensions/experiment/anticipatory_routing/config.py:74-76`、`engine.py:145-148`、`models/{deepseek_v4/config_registry.py:323, qwen3_5/config_registry.py:109}`（`config.checkpoint`）、`patches/torchtitan/trainer.py:150`（`Config.ema_weights`）；UT `tests/unit_tests/ema/test_ema_initial_load.py`（`EMA_OPTIMIZER` / `EMAOptimizersContainer` / `EMATrainer` / `post_dataloading_process` 四项都在这一个文件里）。<br>**③ aux 分母**：删除 `patches/torchtitan/trainer.py:135` 生成 `AuxLoss.set_step_denominator(global_valid_tokens)` 的语句注入，改为在模型侧通过 `extra_kwargs["aux_loss_denominators"]` 传入（上游在 `models/common/decoder.py:250,261,331`、`training_engine.py:566-569` 填全局值）；我方需同步 `override/deepseek_v3_2/sparse_attn/ascendc.py:148` 的断言（`"AuxLoss.set_step_denominator() must be called"`）与 `tests/unit_tests/models/deepseek_v3_2/test_indexer_loss.py`、`tests/unit_tests/models/test_deepseek_v32_mtp_model.py`。<br>**④ metrics**：`patches/torchtitan/components/metrics.py:21-45` 的 aux 注入上游已移到 `trainer.py:453-463`（`**collect_aux_loss_metrics(parallelism_context)` 进 `extra_metrics`），人读行 `aux_loss step:` 与 `sl.log_trace_scalar` 上游**没有**对应物 → 迁到 `extensions/torchtitan/components/metrics.py`（该文件已经在包 `MetricsProcessor.log`，`:22,35`），而不是继续 patch 上游方法（`docs/architecture.md:70` 明确"带副作用的 monkey-patch 收敛至 `apply()`"，且 extensions 是"永久扩展"的正式归属地）。<br>**⑤ pp 桥**：`patches/torchtitan/trainer.py:59-115`（`_run_presplit_pipeline_schedule` / `patched_pp_forward_backward_step`）→ 上游 `TrainingEngine._pp_forward_backward_microbatch_group`（`training_engine.py:737`）调 `pp_schedule.step(..., finalize_gradients=)`（`:755-762`）。<br>**⑥ graph_trainer 两个锚点改名**：`patches/torchtitan/experiments/graph_trainer/functionalize_recompute_mutations.py:100`（`parallel_dims=` → `parallelism_context=`）、`patches/torchtitan/experiments/graph_trainer/memory_policy.py:198-208`（删掉 CLI 注解 hack：Tyro 已被 `eeeb39b42` #4908 移除）；调用点 `models/deepseek_v4/config_registry.py:44`、`extensions/torchtitan/experiments/graph_trainer/__init__.py:24-30`、`models/deepseek_v4/memory_policy.py:32` |
| **上游目标形态（`eb92f645c`）** | `components/optim/ema.py:113 class EMA`（`decays` `:133` / `half_life_fractions` `:140` / `_EMA_STATE_KEY` `:36` / `FixedDecay` `FIXED_DECAY_KEY_PREFIX` `:37` / `HALF_LIFE_KEY_PREFIX` `:38` / `ema_key` `:41`）；`components/optim/optim.py:40,91-93,119-120`（`ema` build + `step`）；`components/checkpointer/base.py:41 EMA = "ema"`；`dcp.py:159 ema:` 与 `:405 Reseed EMA from the just-loaded weights`；`aux_loss.py:75 class AuxLoss` / `:181 inject(..., *, carrier, denominator)` / `:228 scale = 1.0/denominator.clamp_min(1)` / `:108 reduce_mesh: str = "dp"` / `:33 import torch_remat as remat` / `:205 remat.region(..., recompute=False)`；`trainer.py:332 train_step` / `:453-463 extra_metrics` + `metrics_processor.log`；`components/data/types.py:38-44`；`models/common/moe.py:609 class BatchWiseLoadBalanceLoss(AuxLoss)`（**新能力**，`8b4a1300e` #4772，2026-10-11） |
| **验证方式** | CPU UT：`tests/unit_tests/ema/test_ema_initial_load.py`（重写，覆盖冷启动 reseed + 无 EMA 的旧 ckpt）、`tests/unit_tests/patches/torchtitan/models/common/test_aux_loss.py`（改 `denominator=` 契约）、`tests/unit_tests/models/deepseek_v3_2/test_indexer_loss.py`、`tests/unit_tests/extensions/experiment/anticipatory_routing/test_anticipatory_routing.py`。数值：EMA 前后 `loss/global_avg_loss` 与 `grad_norm` 逐 step 对比（`numerics-debugging` skill 的逐 op 对比流程）—— **需要 NPU 设备**。 |
| **依赖顺序** | 依赖 W1（`components/optim` 路径已对齐）；③ 的 `aux_loss_denominators` 契约与 Step 2 的 `preprocess_inputs` 有交集 → **需要与 Step 2 约定 key 名后再动** |
| **预计下降量** | A：**−39**（A2 274 → 235）。B：D1 10 → 17（`checkpoint`/`ema`/`trainer`/`metrics`/`optimizer(A)`/`functionalize`/`memory_policy` 变纯删）。调用点：≈**22 处** |
| **风险与回退** | ① **checkpoint 状态键 `"ema_optimizer"` → 上游 `"ema"`**：现在改会让旧 ckpt 失效 → 决策点：本波一次性改名 + 提供转换脚本与兼容测试（推荐，避免 bump 时同时改键名与实现）；回退 = 保留旧键并加 `states_to_load` 别名。② `torch_remat`：**已安装**（`/home/lrwei/pytorch/venv/3.11/lib/python3.11/site-packages/torch_remat-0.2.0.dist-info`），pin `v0.3.0` 仅在 `pyproject.toml:47-48` 的 `remat` extra 里声明，`eb92f645c` 的 `pyproject.toml:58` 是必需依赖，而**我方 `requirements.txt` 未声明** → 若要用 `remat.region(..., recompute=False)` 替换 `_should_run_forward()`，需先在 `requirements.txt` 显式加依赖，否则 NPU 环境不可复现。③ aux 分母口径从 `labels.numel()` 变成上游 per-microbatch token counts → 数值必须重测 |
| **最高杠杆动作** | **删除 `patch_train_step()`**：它是全仓唯一的 `inspect.getsource` + 字符串改写 + `exec` 注入（`patches/torchtitan/trainer.py:117-143`），bump 时必然因源码文本漂移而断言失败（`:186` 引用的 `.to(self.device)` 在 `eb92f645c` 的 `trainer.py` 已无对应文本），是本仓最脆的一处机制 |

#### W3 — 模型层与 flex_shard / 数据层

| 项 | 内容 |
|---|---|
| **目标** | 把"上游已吸收但形态不同"的模型层 patch 从**类替换注入**降级为**我们自己的子类 + 公开扩展点**，让模块在 bump 时纯删 |
| **我方需要改动的文件与调用点** | **① `models/common/moe.py`（511 行，5 处类替换 + 2 个工厂包装）**：`moe.py:486-513 apply()` 替换 `TokenChoiceTopKRouter`/`MoE`/`GroupedExperts`/`RoutedExperts`/`FeedForward`。调用点：`models/deepseek_v4/__init__.py:337-351`（`hash=layer_id < n_hash_layers` @`:349`、`vocab_size=` @`:350`、`score_func="sqrtsoftplus"` @`:346`、`swiglu_limit=_SWIGLU_LIMIT` @`:360`）、`models/deepseek_v4_1/{__init__.py, attention.py}`、`models/deepseek_v3_2/*`、`models/qwen3_5/*`；`override/deepseek_v4/moe.py:23,28,32,107,111-112`、`override/common/swiglu_group/{__init__.py:10,29,34, asc.ascendc.py:20,46,48,51}`、`models/deepseek_v4/lora.py:29,65,551,579,589`、`extensions/adaptor/torchao/converter.py:40,117,162-166,275-276`；`extensions/experiment/anticipatory_routing/router.py:14`、`override/deepseek_v4/moe_router.py:15`（`HashRouter`）；状态字典/分片路径 `models/deepseek_v4/{config_registry.py:132,162,186,218, state_dict_adapter.py:72-74,170,236-239,269, memory_policy.py:50}`、`models/deepseek_v4_1/{config_registry.py:194,213, state_dict_adapter.py:66-68,274}`、`extensions/torchtitan/distributed/fsdp.py:168-170`、`override/common/swiglu_group/__init__.py:30`、`extensions/adaptor/torchao/converter.py:512`（`inner_experts` FQN）。<br>**② `models/common/linear.py`（`BatchedLinear`）**：调用点 `models/{deepseek_v4/attention.py:20,312,437, deepseek_v4/__init__.py:28,293, deepseek_v4/lora.py:35,61,117,135,545,583, deepseek_v4_1/attention.py:49,190, deepseek_v4_1/__init__.py:28,426, deepseek_v3_2/model.py:32,325-326, deepseek_v3_2/__init__.py:36,114,120}`、`extensions/adaptor/torchao/converter.py:54,116`；UT `tests/unit_tests/models/deepseek_v4/{test_lora,test_lora_layers,test_lora_training}.py`。<br>**③ `models/common/rope.py`（`split` 字段）**：调用点 `models/deepseek_v4/__init__.py:113,151,260,511`、`models/deepseek_v4_1/__init__.py:353-354`、`override/common/rope.py:29-31,126,205,254-256,274-286`；UT `tests/unit_tests/patches/torchtitan/models/common/test_cossin_split_rope.py`、`tests/unit_tests/override/common/test_rope.py`。<br>**④ `models/qwen3_5/vision_encoder.py`**：注入点从 `torchtitan.models.qwen3_5.vision_encoder.compiled_create_block_mask` 改为 `torchtitan.models.common.vision_encoder.compiled_create_block_mask`；调用点 `models/qwen3_5/__init__.py:16-21`；UT `tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py:53-58`。<br>**⑤ `hf_datasets/multimodal/mm_collator.py`**：调用点 `models/qwen3_5/__init__.py:13-15,20`；UT 同上 `:50-52,111`。<br>**⑥ `flex_shard/dist_muon.py` 的 `BlockShard(block_size=)`**：调用点 `models/deepseek_v4/config_registry.py:110,114,142`、`models/deepseek_v4_1/config_registry.py:173,177`（共 5 处）→ 引入 `torchtitan_npu/models/common/muon.py` 里的版本无关帮助函数（探测 `BlockShard` 的字段名后选 `block_size=` 或 `block_sizes=(...)`）；UT `tests/unit_tests/test_dsv4_muon.py:33,136-152`。<br>**⑦ `hf_datasets/text_datasets.py`**：调用点 `models/deepseek_v4_1/config_registry.py:269`、`models/deepseek_v4_1/vision/dataloader.py:26`、`override/deepseek_v4/chat_dataloader.py:11`；UT `tests/unit_tests/models/deepseek_v4_1/test_text_dataloader.py:16-20`。<br>**⑧ `components/optimizer.py` 的 B 部分（replica dedup）**：`optimizer.py:65-89` + `flex_shard/replica_dedup.py:44-45,490-520` 重锚；调用点 `config/configs.py:55,107-108`、`extensions/cpu_offload/cpu_offload_muon.py:735,739`、`override/common/optimizer.py:984`、`tests/unit_tests/extensions/cpu_offload/test_no_optimizer_state.py:27-92` |
| **上游目标形态（`eb92f645c`）** | `models/common/moe.py:171 class TokenChoiceTopKRouter`（`Config` 字段 `num_experts/gate: HiMidLoLinear.Config/score_func: UnaryActivationFn.Config/top_k/route_norm/route_scale/aux_loss/freeze_gate`，**无** `hash`/`vocab_size`）、`:219 _select_experts(scores_TE, expert_bias_E=None, **router_kwargs)`、`:61 class RoutedExperts`（`Config{w13: GroupedLinear.Config, w2: GroupedLinear.Config, token_dispatcher, activation_fn: BinaryActivationFn.Config = SwiGLU.Config, output_postprocess}`）、`:707 class MoE`、`:777 MoE.forward(x_TD, *, padding_mask_T=None, aux_loss_denominator=None, **router_kwargs)`、`:949 register_moe_load_balancing_hook`、`:609 BatchWiseLoadBalanceLoss`；`models/common/activation.py:140 SqrtSoftplus`、`:173 ClampedSwiGLU`（`Config{swiglu_alpha=1.702, swiglu_limit=7.0}`，`:197-201`）——`GroupedExperts` 全树 **0 命中**；`models/deepseek_v4/moe.py:37 class DeepSeekV4Router(TokenChoiceTopKRouter)`（`Config{vocab_size, n_hash_layers=3, layer_id=0}` `:41-44`，`self.hash = config.layer_id < config.n_hash_layers` `:51`，`_select_experts(..., *, input_ids_T=None, **router_kwargs)` `:77`）、`models/deepseek_v4/attention.py:298 wo_a: Linear.Config`、`:439-445 weight.view(...) + torch.einsum("tgd,grd->tgr", ...)`、`:377-385 torch.split` 部分 RoPE；`models/common/linear.py:54 num_linears`、`:348 class GroupedLinear`、`BatchedLinear` **0 命中**；`models/common/rope.py:138 apply_rotary_emb(query, key, rope_cache, *, inverse=False)`、`:161 forward(..., key=None, *, inverse=False)`、**无 `split`**（`git log -S split -- torchtitan/models/common/rope.py` 空）；`models/common/vision_encoder.py:38 compiled_create_block_mask`、`:106 create_block_diagonal_mask`、`:123` 内部调用；`models/qwen3_5/vision_encoder.py:29` import、`:437 create_block_diagonal_mask(...)`、`:451 return self.merger(x)`（**该模块不再暴露 `compiled_create_block_mask`**）；`components/data/multimodal/mm_collator.py:189 _build_mrope_positions`（1-D）；`components/data/text_datasets.py`（grain 化，`_read_text`/`TextProcessor`/`ChatProcessor`）；`flex_shard/optimizer_reshard.py:36 class BlockShard { dim:int, block_sizes: tuple[int,...] :58 }`；`components/optim/optimizer.py:186 class DistMuon(FlexShardDistMuon, BaseOptimizer)`（`build_dist_muon` **0 命中**，被 `17067f00b` #4904 删除） |
| **验证方式** | CPU UT：`tests/unit_tests/override/common/test_moe_score_absorption.py`、`tests/unit_tests/override/common/test_swiglu_group.py`、`tests/unit_tests/models/deepseek_v4/{test_lora,test_lora_layers,test_lora_training,test_lora_merge,test_lora_state_dict_adapter,test_peft}.py`、`tests/unit_tests/models/deepseek_v4_1/{test_baseline_contract,test_v41_rope,test_text_dataloader}.py`、`tests/unit_tests/models/deepseek_v3_2/*`、`tests/unit_tests/test_dsv4_muon.py`、`tests/unit_tests/extensions/adaptor/torchao/test_converter.py`。数值/NPU：**`ClampedSwiGLU` vs 我方 `silu+clamp`**（见 §5）、`block_sizes` 变长块下的 Muon 更新、RoPE 调用方切片的注意力数值 —— 均**需要 NPU 设备** |
| **依赖顺序** | ②③ 依赖 ①（`wo_a`/`w_uk` 的形态和 MoE 配置面一起动）；⑦ 依赖 W1 的 `components/data` 路径对齐；⑧ 与 `flex_shard/dist_muon.py` 同批 |
| **预计下降量** | A：**−183**（A1 −66：`inner_experts` 31 + `GroupedExperts` 30 + `build_dist_muon` 5；A2 −117）。B：D1 17 → 22（`moe`/`linear`/`rope`/`dist_muon`/`mm_collator`/`vision_encoder` 变纯删）。调用点：≈**70 处** |
| **风险与回退** | ① `ClampedSwiGLU` 与我方 clamp 数值语义不同（上游 `gate*sigmoid(alpha*gate)` + `addcmul`，我方 `F.silu(gate)*up`）→ 需先判定 DSV4 参考实现要哪一个，判定前**不要**把 `swiglu_limit` 迁到 `activation_fn`；② `BatchedLinear` 的 DSV3.2 `w_uk`/`w_uv` **上游没有对应模型**，只能自研（建议 `Linear.Config(num_linears=H)`）；③ `mm_collator` 的 1-D/2-D 契约差异是**静默数值错误**而非报错 → W0 的守卫必须先落地；④ `BlockShard` 改 `block_sizes` 会改变 DistMuon 的矩阵切分 → 数值必须重跑 |
| **最高杠杆动作** | **把 `moe.py` 的 5 处类替换降到 0**：hash 路由用上游 `DeepSeekV4Router` 子类（`layer_id`/`n_hash_layers` 取代 `hash=`/`vocab_size=`）、`sqrtsoftplus` 用 `SqrtSoftplus.Config`、clamp 用自写 `BinaryActivationFn` 挂在 `RoutedExperts.Config.activation_fn`、多模态 `bias_vl`/`sorted_topk` 只覆写 `_select_experts`（`eb92f645c` 的 `moe.py:219` 就是干净覆写缝）——一次性消掉 A1 的 `GroupedExperts`(30) 与 A2 的 `HashRouter`(17)+`sqrtsoftplus`(6)+`_Clamp*`(9)+`HashMoE`(3) |

#### W4 — 暂不回合（明确推迟，写清理由与跟踪项）

| 模块 | 为什么推迟 | 上游现状（`eb92f645c` 实测） | 跟踪 |
|---|---|---|---|
| `patches/torchtitan/models/common/token_dispatcher.py`（router score 吸收） | 能力上游仍无，且**锚点大面积漂移**：`AllToAllTokenDispatcher` 改继承 `BaseEPTokenDispatcher`（`:181,235`）、`_permute(self, routed_input_RD, num_global_tokens_per_local_expert_E)` 返回三元组（`:535-545`）、`combine(routed_output_RD, metadata, x_TD)` 去掉 legacy 参数（`:595-601`）、`_sync_token_count_exchange` 返回 tensor 三元组（`:283-287`）；我方 `:265-292` 的 5 元组覆写与 `:294-358` 的 legacy 参数**完全对不上**，但**现在重写没有上游回报**（PR 未合） | `absorb_router_scores` / `routed_scores` 全树 **0 命中**；PR #4095 未合并 | PR #4095 |
| `patches/torchtitan/distributed/varlen_cp.py` + `distributed/context_parallel.py` 的 varlen 分支 | 上游方向不同：只有 Ulysses 路线（`UlyssesCPVarlenInnerAttention`，`CPInnerAttention.prepare_cp_metadata` 直接 `del permutation`），我方的 K/V all-gather + headtail rank-local metadata 未上游化；`VarlenMetadata` 已改名 `VarlenAttentionMetadata`、`cp_shard` 已删除、依赖 torch 私有 `_HeadTailLoadBalancer._generate_indices()` | `CPVarlenMetadata`/`k_global_gather_indices` **0 命中**；`6ef6a6f6c`（#3430）不是 HEAD 祖先 | #3430 / #4617 / Ulysses 决策 |
| `patches/torchtitan/distributed/flex_shard/replica_dedup.py` + `components/optimizer.py` B 部分 | 能力上游仍无，**且挂钩点被删**：`build_dist_muon` 被 `17067f00b`（#4904）删除，替代路径是 `components/optim/optimizer.py:186 DistMuon(Config, params=)`；上游 `dist_muon.py:1488-1489` 明确 "replicated compute is not implemented"。重锚是 M 级工作，但**上游一旦合 #4411 就要再改一次** | `replica_dedup` **0 命中** | PR #4411 |
| `patches/torchtitan/experiments/graph_trainer/{ep_chunk_concretization,ep_shape_live_out,ep_forward_accumulation}.py` | 宿主 `ep_chunk_pass.py`（2455 行）被 `43f73649f`（#4952，2026-09-30）整体删除，改为 **eager chunking**（`ep_eager_chunk.py`）。是否要在 eager 路径上重建这三处修复，取决于"独立 DSV4 metadata 被 chunk shape 吞掉"的问题是否在新路径复现 —— 需要先做复现判定 | `populate_chunk_dim_metadata_pass` / `mark_chunk_dynamic_dims` / `_split_live_out_users` / `_materialize_live_out` 全树 **0 命中**；`ep_eager_chunk.py:409 maybe_apply_ep_overlap_eager_chunking` / `:445 populate_eager_chunk_metadata_pass` | #4952 + 复现判定 |
| `patches/torchtitan/experiments/graph_trainer/graph_trainer_runtime_context.py` | 注入点 `GraphTrainer._make_fx_forward_backward_step` 已不存在（改为 `GraphTrainingEngine` + `spmd_graph_builder.py:194/272` 的 trace 函数，`apply_graph_passes` 调用在 `spmd_graph_builder.py:257` / `graph_builder_utils.py:155,243,295,308`）；迁移目标与 auto-overlap 校准流程绑定，需与 NPU 图编译排期一起做 | `apply_graph_passes(gm, example_inputs, passes, *, compile_config=None, respect_disable_passes=True)`（`passes.py:495-503`，**无** `runtime_context`）；PR #4763 未合并 | PR #4763 |
| `patches/torchtitan/experiments/graph_trainer/memory_policy.py` 的 `save_input_every_n_layers` | 语义上游确实没有；但**主体已改写**（无 `save_ops`），继续 fork 整份 `tag_sac_policy` 会丢上游修复（cacheable-effect 保存） | `memory_policy.py:282 tag_sac_policy(..., force_save_nodes=...)`、`:123 _make_default_memory_policy(save_ops=...)`、`:335 _is_cacheable_effect`；PR #3949 未合并 | 本计划 W2 只做"薄封装"改造；语义扩展继续观察 #3949 |
| `patches/torchtitan/experiments/graph_trainer/functionalize_recompute_mutations.py` 的 pass 本体 | pass 上游没有，但上游用**另一条路线**解决同类问题且尚未写完（`[1/2]`） | `bc86435b7`（#4842）新增 `experiments/graph_trainer/mutation_utils.py`（`mutation_deps:195`、`is_mutation_node:30`、`base_tensor_for_mutation_target:146`），只接入 GraphPP；`functionalize_recompute_mutations_pass` 全树 **0 命中** | #4842 [2/2] |
| `patches/torchtitan/experiments/torchft/{optimizer,trainer}.py` 的**重写**（不是删除） | 4 个上游 PR 全部已合（`cd890fedc`/`91abd2301`/`277d4c930`/`59bbedfcd`），但我方 patch 的形态已失效：`ft_manager=` 关键字不再存在（改为 `configure_fault_tolerance`）、`_configure_chunked_loss` 已被 `ec953b360`（#4773）删除（能力移入 `training_engine.py:403-404`）、`FaultTolerantTrainer` 不再是 `Trainer` 子类。**跟上游走需要 NPU TorchFT 集成整体重挂**，与"补丁清理"不是同一件事 | `torchft/optimizer.py:35,55,68,84,96,102`；`torchtitan/training_engine.py:403-404` | 归独立的 TorchFT 工作流；本轮**只做 W0 守卫** |
| `patches/torchtitan/hf_datasets/text_datasets.py` | `torchtitan/hf_datasets/` 整个目录在 `#5277`（`6f2225575`，2026-10-11）重组，`text_datasets.py` 移到 `components/data/`；能力本身（`pad_segments_to_multiple`）上游**从来没有**（header 声称的"上游镜像"只在 fork 分支 `br_dpsk_v4_1` 的 `9cb8f2274`，非 `origin/main` 祖先），且 pin 的 `HuggingFaceTextDataset/Loader` 早已被 grain（`1b04fc1c3` #4088）删除。迁移到 grain 数据栈是 M–L 级、且与 DSV4.1 数据管线排期绑定 | `pad_segments_to_multiple` 全树 **0 命中**；`components/data/text_datasets.py` 已是 grain 版 | pin bump + grain 迁移 |

#### 明确指向 Step 2（不在本计划排期）

`patches/torchtitan/models/common/attention_metadata.py`、`patches/torchtitan/models/common/decoder.py`、`patches/torchtitan/components/validate.py` 三个模块 + `patches/torchtitan/trainer.py` 的 `post_dataloading_process` 部分（`:33-57`）——详见 DeepSeek metadata seam Step 2 工作流。

---

### 3. 逐模块工单

格式：**模块** → 现状（①-④）→ 本波要做的具体动作 → 涉及的我方调用点（`file:line`）→ 验收断言 → 工作量。

> 分类沿用审计口径：① 上游已吸收（可删）；② 上游已吸收但形态/机制不同（重写）；③ 上游仍无（保留）；④ 与上游方向冲突（重新设计）。

#### 3.0 归属 Step 2 的 3 个模块（各一行）

- `patches/torchtitan/models/common/attention_metadata.py`（②）→ Step 2：随模型迁移到 `build_attention_metadata` 后整体删除。激活点 `patches/torchtitan/__init__.py:20`。
- `patches/torchtitan/models/common/decoder.py`（②）→ Step 2：`build_attention_masks` 过渡分支（`:158-163`）删除后本模块失去存在理由。**注意**：`decoder.py:43,203-205` 的 `full_dtensor` 分支与 W1 交叉，删除 `full_dtensor` 时需与 Step 2 同步这一处。
- `patches/torchtitan/components/validate.py`（②）→ Step 2：`Validator.post_dataloading_process`（`:32-53`）与 `trainer.py:33-57` 是同一件事的两半。

#### W0 工单

**`patches/torchtitan/experiments/graph_trainer/ep_chunk_concretization.py`** → ④（宿主被 #4952 删除）
→ 本波：在模块顶部加 `find_spec("torchtitan.experiments.graph_trainer.ep_chunk_pass")` 守卫（当前 `:99 apply()` 内的 `required_helpers` 检查只在 main 上打 warning 后静默退出，bump 时"看起来生效其实无效"）；把 `apply()` 从 `patches/torchtitan/__init__.py:13` 摘到 opt-in 链。
→ 调用点：`patches/torchtitan/__init__.py:13`；UT `tests/unit_tests/patches/torchtitan/experiments/graph_trainer/test_ep_chunk_concretization.py`。
→ 验收：`import torchtitan_npu` 在"上游无 `ep_chunk_pass`"的模拟下不抛异常且日志明确说明跳过。
→ **S**

**`patches/torchtitan/experiments/graph_trainer/ep_shape_live_out.py`** → ④（同上）
→ 本波：模块级 `import ...ep_chunk_pass`（`:23`）改为守卫式导入（当前 bump 到 main 会**直接 import 崩**）。
→ 调用点：`patches/torchtitan/__init__.py:16`。
→ 验收：无 `ep_chunk_pass` 时 `import torchtitan_npu` 成功。
→ **S**

**`patches/torchtitan/experiments/graph_trainer/ep_forward_accumulation.py`** → ④（同上）
→ 本波：`apply()`（`:85`）加同样的守卫（它是 opt-in，仅由 `functionalize_recompute_mutations.py:95,99` 调用，当前无 `hasattr` 守卫）。
→ 调用点：`patches/torchtitan/experiments/graph_trainer/functionalize_recompute_mutations.py:95,99`。
→ 验收：`apply()` 在无锚点时抛带模块名的 `RuntimeError` 而不是 `AttributeError`。
→ **S**

**`patches/torchtitan/distributed/flex_shard/dist_muon.py`** → ②（上游已含 #4122，但 4/7 个函数被后续 4 个 PR 改写）
→ 本波：`apply()`（`:565-578`）加锚点断言：校验被注入的 7 个自由函数在 `dist_muon` 模块中存在，且**当前模块的函数体与 patch 携带的 `6ee6b10b` 版本同源**；不同源则 `RuntimeError`（否则 bump 时会静默把 `BlockShard(block_sizes)` 变长块与 #4811/#4871 的修复覆盖回旧实现）。补 `Upstream-PR:` 头（在 `patch_metadata.missing` 里）。
→ 调用点：`patches/torchtitan/__init__.py:10` → `patches/torchtitan/distributed/flex_shard/__init__.py:3-8`。
→ 验收：构造一个假的 `dist_muon` 模块（字段名改回 `block_size`）后 `apply()` 报错。
→ **S**

**`patches/torchtitan/hf_datasets/multimodal/mm_collator.py`** → ②（修复已在 main，`539aa6cdf` #4239）
→ 本波：把 `_EXPECTED_PARAMETERS`（`:25-33`）的"形参名 + kind"校验升级为**形状契约**校验（例如探针调用一次并断言返回维度与 2-D 输入一致），因为 `eb92f645c` 的形参名与列表**完全相同** → 只比名字会**静默装上 2-D 变体**，把 1-D 数据流当 2-D 处理（静默数值错误）。
→ 调用点：`models/qwen3_5/__init__.py:13-15,20`；UT `tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py:50-52,111`。
→ 验收：用一个 1-D 契约的 fake collator 触发 `RuntimeError("Unsupported upstream MultiModalCollator signature")`。
→ **S**

**`patches/torchtitan/experiments/graph_trainer/graph_trainer_runtime_context.py`** → ③（PR #4763 未合并，注入点已消失）
→ 本波：`apply()`（`:150-167`）在探测到 `GraphTrainer._make_fx_forward_backward_step is None` 时**抛错**而不是 `logger.warning` 后 `return`（现在是静默降级，auto-overlap 校准会静默失去 runtime context）。
→ 调用点：`extensions/torchtitan/experiments/graph_trainer/__init__.py:24-30`；UT `tests/unit_tests/patches/torchtitan/experiments/graph_trainer/test_graph_trainer_runtime_context.py`、`tests/unit_tests/extensions/torchtitan/experiments/graph_trainer/test_auto_overlap.py`。
→ 验收：模拟"上游无该方法"时 `apply()` 抛错；正常路径 UT 仍绿。
→ **S**

**`patches/torchtitan/experiments/graph_trainer/{ep_overlap_shape_queries,functionalize_recompute_mutations,memory_policy}.py`、`patches/torchtitan/distributed/flex_shard/{optimizer_reshard,_optimizer_reshard_schedule,replica_dedup}.py`、`patches/torchtitan/hf_datasets/text_datasets.py`、`patches/torchtitan/models/common/rope.py`**
→ 本波：仅补 `Upstream-PR:` 头 + 锚点漂移注记（`scripts/check_architecture_boundaries.py:95` 只接受 `Upstream-PR:` / `upstream PR:` / `Upstream PR:` 三种写法）。
→ 调用点：`.ci/architecture_import_baseline.json` 的 `patch_metadata.missing`（7 项，实测）。
→ 验收：`python scripts/check_architecture_boundaries.py` 报告 `patch_metadata.missing` 为空。
→ **S**（7 个模块合计）

**`patches/torchtitan/__init__.py` + `.ci/architecture_import_baseline.json`**
→ 本波：任何激活链变更后同步 `activation_order["torchtitan_npu/patches/torchtitan/__init__.py"]`（当前 19 项，实测）。
→ 验收：`tests/unit_tests/tooling/test_architecture_boundaries.py` 通过。
→ **S**

#### W1 工单

**`patches/torchtitan/distributed/full_dtensor.py`** → ④（上游 `601cf4d23` #4217 + `5dd944e62` #4218 已删除整个后端）→ **D0：本波删除**
→ 动作：① 删除模块 + `patches/torchtitan/__init__.py:9` 的 `full_dtensor`；② 把 `patched_validate_config` 的"CP + SDPA 不支持"断言迁到我方 `models/deepseek_v4_1/parallelize.py` 的本地校验（`eb92f645c` 的 `ScaledDotProductAttention` 已改名 `ScaledDotProductInnerAttention`，`models/common/attention/attention.py:505`）；③ 清掉全部 `spmd_backend == "full_dtensor"` 分支（清单见 W1 波次表）。
→ 调用点：`override/qwen3_5/parallelize.py:15,87-88`、`models/deepseek_v4_1/parallelize.py:26,169,182`、`override/deepseek_v4/mhc/tilelang.py:187`（注释）、`patches/torchtitan/models/common/decoder.py:43,203-204`（Step 2 交叉）。
→ 验收：`grep -rn "full_dtensor" torchtitan_npu/` 只剩 `deepseek_v4_1/parallelize.py` 里那条显式 `NotImplementedError`；`tests/unit_tests/models/deepseek_v4_1/test_model_split.py` 通过。
→ **M**

**`patches/torchtitan/components/metrics.py`** → ②（上游注入点移到 `trainer.py:456`）
→ 动作（W1 部分）：`from torchtitan.components.metrics import MetricsProcessor`（`:11`）改为双路径解析（`observability.metrics` 优先 + `components.metrics` 回退）；`collect_aux_loss_metrics(self.parallel_dims)`（`:27`）改为属性探测（`parallelism_context` 优先，回退 `parallel_dims`）。W2 再把整段迁到 `extensions/torchtitan/components/metrics.py` 并删除本模块。
→ 调用点：`patches/torchtitan/__init__.py:8`；`extensions/torchtitan/components/metrics.py:19,22,35`（**同一路径改名的第二个落点**）；`extensions/experiment/anticipatory_routing/schedule.py:450`（`MetricsProcessor.log`）；UT 无专用文件 → W2 新增。
→ 验收：在"只有 `torchtitan.observability.metrics`"的模拟 import 环境下模块可导入。
→ **S**

**`patches/torchtitan/components/optimizer.py`** → A 部分 ①（#4474 = `dc3985ad4`，上游 `components/optim/utils.py:29`）/ B 部分 ③（上游仍无）
→ 动作（W1 部分）：`import torchtitan.components.checkpoint_utils` / `torchtitan.components.optimizer`（`:18-19`）改为 `torchtitan.components.optim.{utils,optimizer}` 双路径；`apply()`（`:92-104`）加锚点断言。W3 处理 B 部分重锚。
→ 调用点：`patches/torchtitan/__init__.py:8`；UT `tests/unit_tests/patches/torchtitan/components/test_optimizer_patch.py`。
→ 验收：UT 在双路径下通过；A 部分（`patched_init_optim_state`）在"上游已有 `init_optim_state`"的模拟下不覆盖。
→ **S**

**`patches/torchtitan/components/validate.py`**（②，Step 2 归属）
→ 动作（W1 部分）：只把 `from torchtitan.components.validate import Validator`（`:28`）改为 `components.validator` 双路径解析；其余随 Step 2。
→ 验收：模块在新旧路径下都可导入。
→ **S**

**`patches/torchtitan/models/common/{moe,token_dispatcher}.py` 的 `get_spmd_backend` 分支** → ③/②
→ 动作：删除 `get_spmd_backend()` 判定（`moe.py:43,384`、`token_dispatcher.py:27,197,206,209,318,326`），改为在 `spmd_types` 的 `spmd.is_type_checking()` 上直接判定（上游 `debug.spmd_typechecking`，`distributed/...`；`eb92f645c` 的 `annotate_input_spmd_types` 在 `spmd_types.py:180`）。
→ 调用点：`override/common/token_dispatcher.py:51,207,217,220,332,347`、`override/common/swiglu_group/ascendc.py:18,71`、`override/deepseek_v4/moe.py:21,52`、`models/deepseek_v4/{mtp.py:29,109, parallelize.py:15,146, lora.py:26,227, token_dispatcher.py:68,603}`、`models/deepseek_v3_2/model.py:17,83,364,382`。
→ 验收：`grep -rn "get_spmd_backend" torchtitan_npu/` 为 0；`tests/unit_tests/override/common/test_swiglu_group.py`、`tests/unit_tests/override/common/test_moe_score_absorption.py` 通过。
→ **M**

**`patches/torchtitan/components/checkpoint.py`** → ②（#3985 已合）
→ 动作（W1 部分）：`apply()`（`:74-80`）加锚点断言 + 目标命名空间列表改为"存在才替换"（当前无条件写 `torchtitan.components.checkpoint.CheckpointManager`，bump 后该模块不存在会 `AttributeError`）。W2 处理 EMA 语义。
→ 验收：模拟"无 `components.checkpoint`"时 `apply()` 只改存在的命名空间。
→ **S**

#### W2 工单

**`patches/torchtitan/trainer.py`** → ②（含 ① 的 H2D 与 ③ 的 pp 桥）
→ 现状：`(i)` `post_dataloading_process`（`:33-57`，Step 2）；`(ii)` `pp_forward_backward_step` 桥（`:59-115`）；`(iii)` `patch_train_step()`（`:117-143`）；`(iv)` `EMATrainer`（`:145-162`）+ `torchtitan.trainer.Trainer = EMATrainer`（`:171`）。
→ 本波动作：**删除 `(iii)` 整个函数与 `apply()` 中 `:167` 的日志行**（H2D 交给 `components/data` 侧、aux 分母改为 `extra_kwargs["aux_loss_denominators"]`）；`(iv)` 删除（EMA 走 `optim.ema`）；`(ii)` 改为覆盖上游 `TrainingEngine._pp_forward_backward_microbatch_group`（`training_engine.py:737-762`）；`(i)` 留待 Step 2。
→ 调用点：`patches/torchtitan/__init__.py:7`；`extensions/experiment/anticipatory_routing/engine.py:127`（`self.trainer.post_dataloading_process(...)`）；`config/manager.py:27`（注释里说明 `patches.torchtitan.trainer` 会替换 `torchtitan.trainer.Trainer`）；UT `tests/unit_tests/ema/test_ema_initial_load.py`。
→ 验收：`grep -rn "inspect.getsource" torchtitan_npu/` 为 0；`tests/unit_tests/ema/test_ema_initial_load.py` 在 `optim.ema` 语义下通过（测试重写）。
→ **L**

**`patches/torchtitan/components/ema.py`** → ②（#3985 → 上游 `components/optim/ema.py:113 class EMA`）
→ 本波动作：把 `Config(enable, decay, half_life_fraction, ...)`（`:53-96`）改成上游的列表语义（`decays: list[float]` / `half_life_fractions: list[float]`，上游无 `enable`）；`EMA_OPTIMIZER = "ema_optimizer"`（`:26`）对齐上游 `"ema"`（`checkpointer/base.py:41`）；把 `checkpoint.py` 的 DCP 布局上下文（`:144-186 state_dict/load_state_dict`）改为复用上游 `get_flat_optim_state_dict` 布局。
→ 调用点：`patches/torchtitan/components/checkpoint.py:22,28,34,52,63,65`；`patches/torchtitan/trainer.py:23,150`；`extensions/experiment/anticipatory_routing/{config.py:73-76, engine.py:145-148}`；UT `tests/unit_tests/ema/test_ema_initial_load.py`。
→ 验收：旧 ckpt（`ema_optimizer.*` 前缀）能加载并转换成 `ema.*`；`EMA.step` 在 `Optim.step` 后由 engine 调用（对齐 `components/optim/optim.py:119-120`）。
→ **M**

**`patches/torchtitan/components/metrics.py`** → ②（能力已吸收到 `trainer.py:456`，人读行上游无）
→ 本波动作：**删除本模块**（`apply()` 在 `:43-45`），把人读行 `aux_loss step:` 与 `sl.log_trace_scalar`（`:37-40`）迁到 `extensions/torchtitan/components/metrics.py`（该文件已在包 `MetricsProcessor.log`），走 `docs/architecture.md:70` 的 extensions 归属。
→ 调用点：`patches/torchtitan/__init__.py:8`；`extensions/torchtitan/components/metrics.py:19-35`；`extensions/experiment/anticipatory_routing/schedule.py:450`。
→ 验收：`tests/unit_tests` 中 aux 指标日志断言迁移到 extensions 的测试；`grep -rn "MetricsProcessor.log" torchtitan_npu/patches/` 为空。
→ **S**

**`patches/torchtitan/models/common/aux_loss.py`** → ②（文件本体被 #3864 吸收，契约不同）
→ 本波动作：把 `set_step_denominator`（`:174`）与类级分母（`:201,209`）改为 per-forward `denominator=`；`reduce_mesh` 从 `"batch"`（`:140`）改为上游 `"dp"`（上游 `aux_loss.py:108`）；`collect_aux_loss_metrics` / `register_aux_loss_zero_hook` 签名从 `parallel_dims` 改为 `parallelism_context`（上游 `:256, :298`）；`_should_run_forward()`（`:62`）的存废取决于是否引入 `torch_remat`（**0.2.0 已安装**，但 `requirements.txt` 未声明，见 §5）。
→ 调用点：`patches/torchtitan/components/metrics.py:15-27`；`patches/torchtitan/trainer.py:135`；`models/deepseek_v3_2/model.py:31,191`、`models/deepseek_v3_2/__init__.py:33`；`override/deepseek_v3_2/sparse_attn/ascendc.py:29,147-148`；`override/deepseek_v4/sparse_attn/golden.py:32-34,344`；`extensions/experiment/anticipatory_routing/engine.py:91,214`；UT `tests/unit_tests/patches/torchtitan/models/common/test_aux_loss.py`、`tests/unit_tests/models/deepseek_v3_2/test_indexer_loss.py`、`tests/unit_tests/models/test_deepseek_v32_mtp_model.py`。
→ 验收：`grep -rn "set_step_denominator" torchtitan_npu/` 为 0；`tests/unit_tests/patches/torchtitan/models/common/test_aux_loss.py` 用 `denominator=` 契约通过。
→ **M**

**`patches/torchtitan/experiments/graph_trainer/functionalize_recompute_mutations.py`** → ③（pass 上游无，锚点已变）
→ 本波动作：`construct_default_graph_passes` 的 `parallel_dims=`（`:100`）改为 `parallelism_context=`；pipeline 签名对齐 `spmd_graph_builder.py:190` 的 `pipeline_fn(traced, trainer_config, parallelism_context=...)` 约定；保留 `anchor = names.index("tag_with_memory_policy_pass")`（上游 `memory_policy.py` 仍有该 pass）。
→ 调用点：`models/deepseek_v4/config_registry.py:44`、`extensions/torchtitan/experiments/graph_trainer/auto_overlap.py:30`；UT `tests/unit_tests/patches/torchtitan/experiments/graph_trainer/test_rope_recompute_integration.py`。
→ 验收：pipeline 注册后在 `pass_pipeline="mutation-functionalization"` 下可用；`grep -rn "parallel_dims=" patchtorchtitan_npu/patches/torchtitan/experiments/` 为空。
→ **M**

**`patches/torchtitan/experiments/graph_trainer/memory_policy.py`** → ③（PR #3949 未合并，主体已改写）
→ 本波动作：**不要 fork 整份 `tag_sac_policy`**（`:122-160`）→ 改为薄封装上游 `memory_policy.tag_sac_policy(...)`，只在 `save_input_every_n_layers > 1` 时追加 `_force_boundary_saves`（`:41-68`）；删除 CLI 注解 hack（`:198-208`，Tyro 已被 `eeeb39b42` #4908 移除）。
→ 调用点：`models/deepseek_v4/memory_policy.py:32,161,185,198,229`；UT `tests/unit_tests/models/deepseek_v4/memory_policy/`。
→ 验收：DSV4 `@register_memory_policy("dsv4-mhc")` 在薄封装下产出与现状一致的 `NodePolicyKey` 集合。
→ **M**

**`patches/torchtitan/experiments/graph_trainer/ep_overlap_shape_queries.py`** → ③（**锚点未变**，最便宜的一个）
→ 本波动作：更新头注为"锚点已验证于 `eb92f645c`"，并把 `@wraps` 幂等标记改为显式钩子探测 + W0 锚点断言（`ep_overlap_pass.py:196` / `:111` / `:123` / `common_utils.py:175`）。
→ 调用点：`patches/torchtitan/__init__.py:14`。
→ 验收：`apply()` 在锚点缺失时报错；EP overlap 整网 ST 不变。
→ **S**

#### W3 工单

**`patches/torchtitan/models/common/moe.py`** → ②（含 ①③④ 子项）
→ 现状：`apply()`（`:486-513`）替换 5 个上游类 + 2 个 config 工厂，依赖"先换类再 import `config_utils`"的顺序。
→ 本波动作：① hash 路由改用上游 `DeepSeekV4Router` 子类，`models/deepseek_v4/__init__.py:337-351` 的 `hash=`/`vocab_size=` 换成 `layer_id`/`n_hash_layers`；② `score_func="sqrtsoftplus"`（`:346`）换成 `SqrtSoftplus.Config`；③ 删除 `HashMoE`（`:177-290`）与 `MoE.forward` 的 `input_ids` 透传，改走 `**router_kwargs`；④ `_ClampGroupedExperts`（`:348`）/`_ClampFeedForward`（`:411`）改为一个 `BinaryActivationFn` 子类，挂 `RoutedExperts.Config.activation_fn`；⑤ `bias_vl`/`vision_enabled`/`sorted_topk` 改为覆写 `_select_experts`；⑥ 删除 `cfg.inner_experts` 访问（`:458-463`）与 `config_utils` 注入（`:504`）。
→ 调用点：见 W3 波次表 ①（≈40 处）。
→ 验收：`grep -rn "GroupedExperts\|inner_experts\|HashMoE\|_Clamp" torchtitan_npu/` 只剩 `deepseek_v4/{lora,state_dict_adapter,config_registry,memory_policy}.py` 的自有 FQN 拼接（这些在 `eb92f645c` 的 `RoutedExperts` 下需要同步改名）；`tests/unit_tests/override/common/test_moe_score_absorption.py`、`test_swiglu_group.py`、`tests/unit_tests/extensions/adaptor/torchao/test_converter.py` 通过。
→ **L**

**`patches/torchtitan/models/common/linear.py`** → ②（`BatchedLinear` 上游无）
→ 本波动作：DSV4/DSV4.1 的 `wo_a` 改成 `Linear.Config` + `weight.view(n_groups, o_lora_rank, -1)` + `torch.einsum("tgd,grd->tgr", ...)`（对齐上游 `deepseek_v4/attention.py:298,439-445`）；DSV3.2 的 `w_uk`/`w_uv` 改 `Linear.Config(num_linears=H)`（上游无该模型，自研）；`models/deepseek_v4/lora.py:117` 的 `issubclass(parent_cls, BatchedLinear)` 分支改写；`extensions/adaptor/torchao/converter.py:116` 的目标类型列表同步。
→ 调用点：`models/deepseek_v4/attention.py:20,312,437`、`models/deepseek_v4_1/attention.py:49,190`、`models/deepseek_v3_2/model.py:32,325-326`、`models/*/__init__.py`（见波次表）、`models/deepseek_v4/lora.py:35,61,135,545,583`。
→ 验收：`grep -rn "BatchedLinear" torchtitan_npu/` 为 0；LoRA 相关 4 个 UT 通过；`wo_a` 的 TP/FSDP 分片与上游 `Linear` 规则一致（`models/deepseek_v4/config_registry.py` 的 `BlockShard(dim=0, block_size=attention.wo_a.out_features)` 需随之改写）。
→ **M**

**`patches/torchtitan/models/common/rope.py`** → ②（`split` 上游从无；`key`/`inverse` 已吸收）
→ 本波动作：删除 `_complex_apply_rotary_emb` / `_cossin_apply_rotary_emb` 两个 monkey-patch（上游 `apply_rotary_emb(query, key, rope_cache, *, inverse=False)` 已同名同义，`rope.py:138`）；`split` 字段的取舍二选一：(a) 迁到 attention 调用方 `torch.split`（对齐 `deepseek_v4/attention.py:377-385`，改动 2 个模型配置 + 1 个 override），或 (b) 只在 `RoPE` 内部保留 `Split*RoPEConfig` 并把 `apply()` 缩小到 `RoPE.__init__/_split/_unsplit/forward`。
→ 调用点：`models/deepseek_v4/__init__.py:113,151,260,511`、`models/deepseek_v4_1/__init__.py:353-354`、`override/common/rope.py:29-31,126,205,254-256,274-286`。
→ 验收：`grep -rn "dataclasses.replace(rope, split=" torchtitan_npu/` 为 0（选 (a)）；`tests/unit_tests/patches/torchtitan/models/common/test_cossin_split_rope.py`、`tests/unit_tests/override/common/test_rope.py` 通过。
→ **M**

**`patches/torchtitan/distributed/flex_shard/dist_muon.py`（W3 部分）** → ②
→ 动作：① 引入版本无关的 `block_shard()` 帮助函数，把 `models/deepseek_v4/config_registry.py:110,114,142`、`models/deepseek_v4_1/config_registry.py:173,177` 的 `BlockShard(dim=0, block_size=X)` 收敛到一处；② `_normalize_compute_placement` 的 `placement.block_size`（`:509`）与 `:434` 的报错文案同步；③ 保留 NPU 准入（上游 `c215a12ee` #4530 已把 `_validate_parameter_storage` 放宽为"每进程一个设备"，我方原为"必须是 npu"，可只保留极小 delta）。
→ 验收：`tests/unit_tests/test_dsv4_muon.py:33,136-152` 在两种 `BlockShard` 字段形态下都能通过；`grep -rn "block_size=" torchtitan_npu/models/` 只剩非 BlockShard 用途。
→ **M**

**`patches/torchtitan/models/qwen3_5/vision_encoder.py`** → ③（锚点已移走）
→ 本波动作：注入点从 `torchtitan.models.qwen3_5.vision_encoder.compiled_create_block_mask` 改到 `torchtitan.models.common.vision_encoder.compiled_create_block_mask`（`eb92f645c` 的 `vision_encoder.py:38`；`qwen3_5/vision_encoder.py` 已改为 import `create_block_diagonal_mask`，`:29`，调用在 `:437`）；`.clone()` 包装（owning tensor）先在新 pin 上复现，若上游 forward 已直接返回 `merger` 输出（`:451`）则删除该修复以缩小注入面。
→ 调用点：`models/qwen3_5/__init__.py:16-18,21`；UT `tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py:53-58`。
→ 验收：`apply()` 在两种上游布局（`qwen3_5.vision_encoder` 暴露 / 不暴露 `compiled_create_block_mask`）下都指向 `common.vision_encoder`；NPU dense-SDPA 回退仍生效。
→ **S**

**`patches/torchtitan/hf_datasets/multimodal/mm_collator.py`（W3 部分）** → ②
→ 动作：`apply()` 的契约校验升级（见 W0）+ 头注修正（`Upstream PR: pending publication ...` → #4239 / `539aa6cdf`）；bump 时**直接删除**（不要为"保留 NPU 行为"重新套上）。
→ **S**

**`patches/torchtitan/components/optimizer.py`（W3 部分，B = replica dedup）** → ③/④
→ 动作：锚点从 `torchtitan.components.optimizer.optimizer.build_dist_muon` 迁到 `components/optim/optimizer.py:186 DistMuon(Config, params=)`（`build_dist_muon` 已被 `17067f00b` #4904 删除）；`flex_shard/replica_dedup.py:44-45,490-520` 的 `_ORIGINAL_BUILD_DIST_MUON` / `muon.step` 覆盖随之改写；DCP 布局上下文（`:65-89`）保留（依赖 `get_flat_optim_state_dict` 的平铺布局）。
→ 调用点：`config/configs.py:55,107-108`、`extensions/cpu_offload/cpu_offload_muon.py:735,739`、`override/common/optimizer.py:984`、`tests/unit_tests/extensions/cpu_offload/test_no_optimizer_state.py:27-92`。
→ 验收：`grep -rn "build_dist_muon" torchtitan_npu/` 只剩新入口；HSDP `dp_replicate>1` 下 replica 去重仍生效（NPU 验证）。
→ **M**

**`patches/torchtitan/experiments/graph_trainer/chunked_loss.py`** → ③（PR #4734 未合并，锚点已变）
→ 本波动作：`_gradient_backprop` 改为**实例方法**包装并处理 `*accumulated_grads` 元组（上游 `chunked_loss.py:31-37`，`f93fd4ccf` #4451 的 MTP 改动）；`_can_coalesce()`（`:109`）去掉 `parametrization.full_dtensor` 依赖（上游 `simple_fsdp.py:391 ReplicateComputation` 已删除该属性），改用 `param_sharding`/`grad_placements` 判定。
→ 调用点：`patches/torchtitan/__init__.py:12`；UT `tests/unit_tests/patches/torchtitan/experiments/graph_trainer/test_chunked_loss.py`。
→ 验收：MTP（多 prediction）下 chunk 梯度合并仍在 reduce-scatter 前完成；UT 通过。
→ **M**

**`patches/torchtitan/experiments/torchft/{checkpoint,manager}.py`** → ①（可删，bump 即删）
→ 动作：本轮**不动**（pin 上仍需要）；bump 时删文件 + `patches/torchtitan/experiments/torchft/__init__.py:8` 一行。注意 `torchft/checkpoint.py` 是**整方法替换**，保留会覆盖上游演进（丢 `ema=`、丢 `_is_checkpoint_owner` 重构）→ bump 时必须删而不是留着。
→ 调用点：`extensions/torchtitan/experiments/torchft/__init__.py:20`。
→ 验收：bump 后 `grep -rn "TorchFTCheckpointManager" torchtitan_npu/` 为 0。
→ **S**

#### W4 工单（推迟项与 D1 确认项）

**`patches/torchtitan/experiments/graph_trainer/ep_ready_nodes_dedup.py`** → ①（已被上游吸收，`c4bef0bb7` #4516）
→ 本波动作：**无代码改动**；本轮只做"可删性确认"：`eb92f645c` 的 `ep_overlap_pass.py:526 selected: set[fx.Node] = set()` / `:534 if node in selected: continue` / `:541 selected.add(node)` 与补丁逐行等价。bump 时删文件 + `patches/torchtitan/__init__.py:15` 一行。
→ 调用点：`patches/torchtitan/__init__.py:15`。
→ 验收：bump 后 `grep -rn "_ready_nodes" torchtitan_npu/` 为 0，且 EP overlap 整网 ST 不变。
→ **S**

**`patches/torchtitan/distributed/flex_shard/optimizer_reshard.py`** → ①（#4122 已合；`eb92f645c` 的 `ComputeLayout` 代码体与 `6ee6b10b` 一致，只有 docstring 差异）
→ 本波动作：只补 `Upstream-PR:` 头 + 可删性确认。bump 时删文件 + `patches/torchtitan/distributed/flex_shard/__init__.py:3-8` 一行。
→ 调用点：`patches/torchtitan/distributed/flex_shard/__init__.py:3-8`；消费方 `models/common/muon.py:11,15`、`models/deepseek_v4/config_registry.py:20,78,96,109-142`、`models/deepseek_v4_1/config_registry.py:19`、`tests/unit_tests/extensions/cpu_offload/test_no_optimizer_state.py:263`。
→ 验收：`scripts/check_architecture_boundaries.py` 的 `patch_metadata.missing` 不含本文件；`ComputeLayout` 字段集合与 `eb92f645c` 一致。
→ **S**

**`patches/torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py`** → ①（同上；`eb92f645c` 上这 6 个定义 AST 等价，仅 `_dtensor_storage_regions` 合并了两个同体 `elif`）
→ 本波动作：只补 `Upstream-PR:` 头 + 可删性确认。bump 时删文件 + `patches/torchtitan/distributed/flex_shard/__init__.py:4` 一行。注意该模块**额外向 `dist_muon` 注入** `_RedistributionGroup` / `_build_dim0_shard_redistribution_plan` / `_dtensor_storage_regions`（`:305-318`），因为 pin 的 `dist_muon` 是按值 import 的 —— 这三处注入是 pin 特有，bump 后自然消失。
→ 调用点：`patches/torchtitan/distributed/flex_shard/__init__.py:4`；UT `tests/unit_tests/test_dsv4_muon.py:12,138-146,149-152`。
→ 验收：同 `optimizer_reshard.py`；且 `grep -rn "_build_replicated_to_dim0_shard_plan" torchtitan_npu/` 为 0。
→ **S**

**`patches/torchtitan/distributed/context_parallel.py`** → ③（`cp_shard` 锚点整体消失）+ 部分归属 Step 2
→ 本波动作：① W0 加锚点断言（`context_parallel.api.cp_shard` 在 `eb92f645c` **不存在**；上游是新公开面 `__all__`，`shard_tensors` / `ContextParallelLoadBalancer` / `CPInnerAttention.prepare_cp_metadata`）；② W1 把 `models/deepseek_v4/model.py:16,274` 的直接 `cp_shard` 调用收敛到本模块的一个函数后面（bump 时只改一处）；③ varlen 分支的技术路线（自研 KV-allgather headtail vs 改走 Ulysses）留 W4 决策。
→ 调用点：`models/deepseek_v4/model.py:16,274`；`patches/torchtitan/__init__.py:9`；UT `tests/unit_tests/patches/torchtitan/distributed/test_context_parallel_metadata.py`、`tests/unit_tests/models/test_mtp_cp_chunk_loss.py`。
→ 验收：`grep -rn "cp_shard" torchtitan_npu/` 只剩本模块内的包装与 `models/deepseek_v4/model.py` 的一处调用。
→ **M**（W0/W1 部分）/ W4 决策另计

**`patches/torchtitan/distributed/varlen_cp.py`** → ③（#3430 未合，锚点改名/删除）
→ 本波动作：只做"anchor drift 注记"（`VarlenMetadata`→`VarlenAttentionMetadata`；`cp_shard` 删除；依赖 torch 私有 `_HeadTailLoadBalancer._generate_indices()`）。**不要现在重写**。
→ 调用点：`models/deepseek_v4/metadata.py:51`、`models/deepseek_v4/token_dispatcher.py:70`、`override/deepseek_v3_2/sparse_attn/ascendc.py:28`；UT `tests/unit_tests/models/deepseek_v3_2/test_cp_metadata.py`、`tests/unit_tests/models/deepseek_v4/{test_cp_dispatch.py,cp_dispatch_worker.py,test_cp_compressor.py}`、`tests/unit_tests/conftest.py:205`。
→ 验收：注记里写明 3 个锚点各自的 `eb92f645c` 位置与替代物。
→ **S**（注记）

**`patches/torchtitan/experiments/torchft/optimizer.py`** → ②（#4578 + #4598 已合，构造契约变了）
→ 本波动作：W0 加锚点断言（`ft_manager=` 关键字在 `eb92f645c` 的 `TorchFTOptimizersContainer.__init__`（`torchft/optimizer.py:35`）**不存在**，已改为 `configure_fault_tolerance`，`:55`）。重写跟 W4 的 TorchFT 工作流走。
→ 调用点：`extensions/torchtitan/experiments/torchft/optimizer.py:19-23`、`trainer.py:29-30`、`patches/torchtitan/experiments/torchft/__init__.py:8`。
→ 验收：`apply()` 在 `__init__` 没有 `ft_manager` 形参时报错。
→ **S**（W0 部分）/ M（W4 重写）

**`patches/torchtitan/experiments/torchft/trainer.py`** → ②（#4534/#4575/#4663 全合，但结构大改）
→ 本波动作：W0 加锚点断言（`FaultTolerantTrainer` 在 `eb92f645c` 不再是 `Trainer` 子类，`torchft/trainer.py:152`；`_configure_chunked_loss` 已被 `ec953b360` #4773 删除，能力在 `training_engine.py:403-404`）。400+ 行的 `__init__`/`train_step` 重建留 W4。
→ 调用点：`extensions/torchtitan/experiments/torchft/trainer.py:26`（`FaultTolerantTrainerEx`）、`ft_manager.py:59`、`patches/torchtitan/experiments/torchft/__init__.py:9`。
→ 验收：`apply()` 检测到 `torchtitan.trainer.Trainer` 不再是宿主时报错（不是静默替换）。
→ **S**（W0 部分）/ M（W4 重写）

**`patches/torchtitan/hf_datasets/text_datasets.py`** → ④（能力上游从无，锚点全部消失）
→ 本波动作：只补 `Upstream-PR:` 头并**修正误导性 header**（现 header 声称"Mirrors upstream `components/data/packing.py`"，实测 `pad_segments_to_multiple` 在 `eb92f645c` 全树 **0 命中**，该实现只存在于 fork 分支 `br_dpsk_v4_1` 的 `9cb8f2274`，非 `origin/main` 祖先）。迁移到 grain 数据栈留 W4。
→ 调用点：`models/deepseek_v4_1/config_registry.py:269`、`models/deepseek_v4_1/vision/dataloader.py:26`；UT `tests/unit_tests/models/deepseek_v4_1/test_text_dataloader.py:16-20`。
→ 验收：header 不再声称上游有对应物。
→ **S**（header）/ L（W4 迁移）

---

### 4. PR 切分建议

每个 PR 自包含：**代码 + UT + 文档/基线更新**。下列 9 个 PR 与 W0–W3 一一对应（W4 不出 PR，只在 `docs/` 记一条）。

| PR | 波次 | 内容 | 必须同 PR 更新的测试 | 预计规模 |
|---|---|---|---|---|
| **P0** | W0 | 守卫 + 锚点断言 + `Upstream-PR:` 头 + 锚点注记；`.ci/architecture_import_baseline.json` 同步 | `tests/unit_tests/tooling/test_architecture_boundaries.py`（元数据债务）；`tests/unit_tests/patches/torchtitan/experiments/graph_trainer/{test_ep_chunk_concretization,test_ep_forward_accumulation,test_graph_trainer_runtime_context}.py`（守卫断言） | +180/−40 |
| **P1a** | W1 | `full_dtensor` / `spmd_backend` 退场（删 1 模块 + ≈60 处调用点 + 本地校验函数） | `tests/unit_tests/test_dsv4_muon.py:16`、`tests/unit_tests/models/deepseek_v4_1/{test_model_split.py:183,test_muon_profile.py:17}`、`tests/unit_tests/override/common/{test_swiglu_group,test_moe_score_absorption}.py`、`tests/unit_tests/extensions/adaptor/torchao/test_converter.py`、`tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py`；`docs/feature_guides/ep_process_group.md:26` | +80/−260 |
| **P1b** | W1 | 模块路径改名（`observability.metrics` / `components.optim` / `components.validator` / `components.data` / `parallelism_context` / `checkpoint_utils`）+ 双路径解析 | `tests/unit_tests/patches/torchtitan/components/test_optimizer_patch.py`；**新增** `tests/unit_tests/patches/torchtitan/test_import_paths.py`（在"只有新路径"的 fake 环境下 import 每个 patch 模块） | +120/−90 |
| **P2a** | W2 | 删 `patch_train_step()` + pp 桥迁 `TrainingEngine` | `tests/unit_tests/ema/test_ema_initial_load.py`（拆分） | +60/−120 |
| **P2b** | W2 | EMA 三件套对齐上游（含 ckpt 键名迁移与转换） | `tests/unit_tests/ema/test_ema_initial_load.py`（重写，含旧 ckpt 兼容）；`tests/unit_tests/extensions/experiment/anticipatory_routing/test_anticipatory_routing.py`；`tests/smoke_tests/anticipatory_routing/test_anticipatory_routing_smoke.py` | +200/−260 |
| **P2c** | W2 | aux 分母契约 + metrics 迁 extensions | `tests/unit_tests/patches/torchtitan/models/common/test_aux_loss.py`、`tests/unit_tests/models/deepseek_v3_2/test_indexer_loss.py`、`tests/unit_tests/models/test_deepseek_v32_mtp_model.py`；**新增** `tests/unit_tests/extensions/torchtitan/components/test_metrics_aux_log.py` | +150/−150 |
| **P3a** | W3 | `moe.py` 重构（hash/score_func/clamp/`bias_vl`/`inner_experts`） | `tests/unit_tests/override/common/{test_moe_score_absorption,test_swiglu_group}.py`、`tests/unit_tests/extensions/adaptor/torchao/test_converter.py`、`tests/unit_tests/models/deepseek_v4_1/{test_baseline_contract,test_v41_rope}.py`、`tests/unit_tests/models/deepseek_v4/memory_policy/test_swiglu_memory_policy.py`、`tests/integration_tests/lora_config.py` | +400/−500 |
| **P3b** | W3 | `linear.py`（`BatchedLinear` 退场）+ `rope.py`（`split`）+ LoRA 分支 | `tests/unit_tests/models/deepseek_v4/{test_lora,test_lora_layers,test_lora_training,test_lora_merge,test_lora_state_dict_adapter,test_peft}.py`、`tests/unit_tests/models/deepseek_v3_2/*`、`tests/unit_tests/patches/torchtitan/models/common/test_cossin_split_rope.py`、`tests/unit_tests/override/common/test_rope.py` | +250/−300 |
| **P3c** | W3 | flex_shard（`block_sizes` 帮助函数 + `replica_dedup` 重锚 + `chunked_loss` 锚点）+ qwen3_5 vision/mm_collator | `tests/unit_tests/test_dsv4_muon.py`、`tests/unit_tests/patches/torchtitan/experiments/graph_trainer/test_chunked_loss.py`、`tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py`、`tests/unit_tests/extensions/cpu_offload/test_no_optimizer_state.py`、`tests/unit_tests/models/deepseek_v4_1/test_muon_profile.py` | +250/−280 |

**测试面清单（本轮必须触碰的 `tests/unit_tests/` 文件，实测存在）**：
`patches/torchtitan/components/test_optimizer_patch.py`、`patches/torchtitan/distributed/test_context_parallel_metadata.py`、`patches/torchtitan/models/common/{test_aux_loss,test_cossin_split_rope,test_attention_metadata,test_decoder_metadata_seam}.py`、`patches/torchtitan/experiments/graph_trainer/{test_chunked_loss,test_ep_chunk_concretization,test_ep_forward_accumulation,test_graph_trainer_runtime_context,test_rope_recompute_integration}.py`、`ema/test_ema_initial_load.py`、`test_dsv4_muon.py`、`override/common/{test_moe_score_absorption,test_rope,test_swiglu_group}.py`、`models/deepseek_v3_2/{test_cp_metadata,test_indexer_loss}.py`、`models/deepseek_v4/{test_lora,test_lora_layers,test_lora_training,test_cp_dispatch,test_dsv4,test_cp_compressor}.py`、`models/deepseek_v4_1/{test_text_dataloader,test_model_split,test_baseline_contract,test_v41_rope,test_muon_profile}.py`、`models/qwen3_5/test_qwen3_5_npu_adapter.py`、`models/test_deepseek_v32_mtp_model.py`、`extensions/adaptor/torchao/test_converter.py`、`extensions/cpu_offload/test_no_optimizer_state.py`、`extensions/torchtitan/experiments/graph_trainer/test_auto_overlap.py`、`tooling/test_architecture_boundaries.py`。

**文档/基线同 PR 更新**：`.ci/architecture_import_baseline.json`（每删/加 patch 模块都改 `activation_order` 与 `patch_metadata.missing`）、`docs/architecture.md`（§patch 清单口径）、`docs/feature_guides/{aux_loss,deepseek_v4_cp,graph_trainer_ep_overlap,ep_process_group,deepseek_v4_graph_trainer_recompute}.md`（受影响的 5 个）。

---

### 5. 风险登记

#### 5.1 需要 NPU 设备验证的数值项

| 项 | 分歧点（我方 vs `eb92f645c`） | 验证方式 |
|---|---|---|
| **`ClampedSwiGLU` vs 我方 silu+clamp** | 我方 `patches/torchtitan/models/common/moe.py:398-401` 是 `F.silu(gate) * up` 后 clamp；上游 `activation.py:173-201` 是 `gate.clamp(max=limit)` / `up.clamp(±limit)` / `gate*sigmoid(alpha*gate)` / `torch.addcmul(silu, silu, up)`，且 `swiglu_limit` 不接受 0/负数。**两者数学上不等价** | 同一 DSV4 小模型（固定 seed）在两条实现下逐 step 对比 `loss` / `grad_norm`，并做逐 op 激活对比（`numerics-debugging` skill）；**未验证**：DSV4 参考实现到底要哪一种 |
| **`spmd_backend` 迁移** | 删掉 `full_dtensor` 后，`validate_config` 的 CP+SDPA 断言需自建；上游 `shard_tensors` 会抛新的 shape/divisibility 错误 | NPU ST：`deepseek_v4_1` + CP 一条、`qwen3_5` 一条；对比迁移前后的 loss 与显存 |
| **flex_shard `block_sizes`** | `BlockShard(dim, block_size=int)` → `block_sizes: tuple[int, ...]`（`892ecfed0` #4840）改变 DistMuon 的矩阵切分（`(4,)` vs `(128,64)` 这类变长块）；`210fc60cb`（#5279）还改了 redispatch 输入的 dtype（BF16） | NPU ST：DSV4 Muon 训练一条，对比 `grad_norm` 与参数更新（`tests/unit_tests/test_dsv4_muon.py` 只覆盖结构，不覆盖数值） |
| **EMA 语义** | `enable/decay/half_life_fraction` 单副本 → `decays`/`half_life_fractions` 列表 + `buffer_patterns`；`cf8222804`（#5282）改了 CPU-offload EMA 的初始化；ckpt 键 `ema_optimizer` → `ema` | NPU ST：EMA 开关下的 eval 指标与 ckpt 往返；旧 ckpt 加载兼容测试 |
| **aux 分母口径** | `labels.numel()`（step 级，`set_step_denominator`）→ per-microbatch token counts（`aux_loss_denominators`，`training_engine.py:566-569`） | CPU UT 可覆盖契约；数值需 NPU：`aux_loss/*` 指标与主 loss 的一致性 |
| **`torch_remat` 依赖** | 0.2.0 **已安装**；pin `v0.3.0` 只在 `pyproject.toml:47-48` 的 `remat` extra 声明；`eb92f645c:pyproject.toml:58` 是必需依赖；**我方 `requirements.txt` 未声明** | 显式声明依赖后在 NPU 上跑一遍 aux loss AC 路径；否则不要用 `remat.region` 替换 `_should_run_forward()` |

> **未验证**：本轮所有数值类结论都只有静态证据（源码/行号），没有跑过任何 NPU 训练；表中"验证方式"是待执行项，不是已完成结论。

#### 5.2 上游仍是移动目标（不要现在重写）

| 项 | 上游状态（`eb92f645c` 实测） | 处置 |
|---|---|---|
| **token_dispatcher #4095**（router score 吸收） | `absorb_router_scores` / `routed_scores` 全树 0 命中；`_permute` 返回三元组（`:535`）、`combine` 去 legacy 参数（`:595`）、`AllToAllTokenDispatcher` 继承 `BaseEPTokenDispatcher`（`:235`） | W4 推迟；**W0 加锚点断言**，否则 bump 时 `:265-292` 的 5 元组覆写会以错误形状静默生效 |
| **CP varlen / Ulysses #4617 + #3430** | `CPVarlenMetadata` 0 命中；`VarlenMetadata`→`VarlenAttentionMetadata`（`attention/attention.py:76`）；varlen+CP 只有 Ulysses（`UlyssesCPVarlenInnerAttention`），其 `prepare_cp_metadata` 直接 `del permutation`（不做 rank-local 改写） | W4 推迟；**决策分叉**：是否改走 Ulysses（需 AscendC sparse-attention 内核支持 head 维切分）——建议单出设计评审 |
| **ep_chunk eager chunking #4952** | `ep_chunk_pass.py` 已删除；新路径 `ep_eager_chunk.py:409/445` | W4 推迟；先做"问题是否在 eager 路径复现"的判定 |
| **#4842 [2/2]**（mutation 依赖，SPMD/SAR 路径） | `mutation_utils.py` 已存在（`mutation_deps:195`），只接入 GraphPP | 跟踪；若 [2/2] 覆盖 SPMD，`functionalize_recompute_mutations` + `ep_forward_accumulation` 整体删除 |
| **#4763**（graph pass runtime context） | `apply_graph_passes` 无 `runtime_context`（`passes.py:495-503`） | W4 推迟；W0 把静默降级改为报错 |
| **#3949**（`save_input_every_n_layers`） | 全 ref 无该符号 | W4 保留语义扩展；W2 只做薄封装 |
| **#4772 / #5252**（batch-wise aux loss / 局部编译统计） | `BatchWiseLoadBalanceLoss`（`models/common/moe.py:609`）、`_microbatch_load_balance_local_stats`（`:498`）——**新增上游能力** | 跟踪：可能改变我方 aux loss 的实现选择（W2 的 aux 改造应先看这条） |
| **#4835**（DSV4 sparse indexer loss，`fc6ed2d32`） | 上游 `models/deepseek_v4/` 在 74 个提交内有 5 处改动 | 跟踪：与我方 `SparseIndexerLoss`（`models/deepseek_v3_2/model.py:191`）可能收敛 |

#### 5.3 静默失败风险（W0 的锚点断言就是为了这些）

| 模块 | 静默失败的后果 |
|---|---|
| `flex_shard/dist_muon.py` | 把 `eb92f645c` 的 4 个改写函数覆盖回 `6ee6b10b`，回退 `BlockShard(block_sizes)` 与 #4811/#4871 的修复 → 运行期报错或**静默算错** |
| `hf_datasets/multimodal/mm_collator.py` | 形参名完全相同 → 静默把 1-D 数据流当 2-D 处理（多出/错位 batch 维，`<` 与 `== 0` 的边界判定不同） |
| `experiments/graph_trainer/graph_trainer_runtime_context.py` | 打印 warning 后 `return` → auto-overlap 校准静默失去 runtime context（不报错，最难排查） |
| `experiments/graph_trainer/ep_chunk_concretization.py` | `required_helpers` 只 warning → 补丁"看起来生效其实无效" |
| `components/checkpoint.py` | 无条件写三个命名空间 → bump 后 `AttributeError`（`torchtitan.components.checkpoint` 已不存在） |
| `experiments/graph_trainer/ep_shape_live_out.py` | 模块级 `import ...ep_chunk_pass` → bump 后 **import 直接崩**（`import torchtitan_npu` 失败） |

---

### 6. 附录：度量 A 复现脚本

在 `torchtitan_npu/` 的父目录执行（只读）：

```python
import re, os
A1 = {  # pin-only：v0.3.0 有、eb92f645c 无
 "GroupedExperts": r"\bGroupedExperts\b", "inner_experts": r"inner_experts",
 "spmd_backend": r"\bspmd_backend\b", "get_spmd_backend": r"\bget_spmd_backend\b",
 "set_spmd_backend": r"\bset_spmd_backend\b", "full_dtensor": r"\bfull_dtensor\b",
 "AttentionMasksType": r"\bAttentionMasksType\b",
 "VarlenMetadata": r"(?<!Compressed)(?<!Reference)\bVarlenMetadata\b",
 "cp_shard": r"\bcp_shard\b", "post_dataloading_process": r"\bpost_dataloading_process\b",
 "torchtitan.components.optimizer": r"torchtitan\.components\.optimizer\b",
 "torchtitan.components.checkpoint_utils": r"torchtitan\.components\.checkpoint_utils\b",
 "torchtitan.distributed.parallel_dims": r"torchtitan\.distributed\.parallel_dims\b",
 "torchtitan.components.validate": r"torchtitan\.components\.validate\b",
 "torchtitan.hf_datasets.text_datasets": r"torchtitan\.hf_datasets\.text_datasets\b",
 "build_dist_muon": r"\bbuild_dist_muon\b",
 "BlockShard(block_size=)": r"BlockShard\(\s*dim=0,\s*block_size=",
}
A2 = {  # 自造 seam：两边都没有
 "BatchedLinear": r"\bBatchedLinear\b",
 "RouterConfig.hash=": r"^\s*hash=layer_id < n_hash_layers|^\s*hash=True",
 "RoPE.split": r"dataclasses\.replace\(rope, split=",
 "Split*RoPEConfig": r"\bSplit(Complex|CosSin)RoPEConfig\b",
 "MetadataExtension": r"(?<!Asc)(?<!V41)(?<!BatchChunk)(?<!Reference)\bMetadataExtension\b",
 "set_step_denominator": r"\bset_step_denominator\b", "EMATrainer": r"\bEMATrainer\b",
 "EMAOptimizersContainer": r"\bEMAOptimizersContainer\b", "EMA_OPTIMIZER": r"\bEMA_OPTIMIZER\b",
 "EMACheckpointManager": r"\bEMACheckpointManager\b", "HashMoE": r"\bHashMoE\b",
 "HashRouter": r"\bHashRouter\b", "sqrtsoftplus": r"sqrtsoftplus",
 "_Clamp*": r"\b_Clamp(GroupedExperts|FeedForward)\b", "swiglu_limit": r"\bswiglu_limit\b",
 "vision_enabled|bias_vl": r"\bvision_enabled\b|\bbias_vl\b",
 "MetricsProcessor.log": r"MetricsProcessor\.log",
 "prepare_context_parallel_input": r"\bprepare_context_parallel_input\b",
 "CPVarlenMetadata": r"\bCPVarlenMetadata\b", "build_attention_masks": r"\bbuild_attention_masks\b",
}
def count(pats):
    total = 0
    for pat in pats.values():
        rx = re.compile(pat)
        for root, dirs, files in os.walk("torchtitan_npu"):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(root, f)
                    for line in open(p, encoding="utf-8", errors="replace"):
                        total += bool(rx.search(line))
    return total
print("A1 =", count(A1), "A2 =", count(A2), "A =", count(A1) + count(A2))
## 2026-10-14 @0110132d: A1 = 239  A2 = 274  A = 513
```

**上游侧核对命令（只读）**：

```bash
U=/home/lrwei/pytorch/torchtitan; M=eb92f645c
git -C $U rev-list --count 3386c6d6f..origin/main        # 74
for p in GroupedExperts spmd_backend full_dtensor build_attention_masks MetadataExtension \
         AttentionMasksType VarlenMetadata post_dataloading_process EMATrainer \
         absorb_router_scores routed_scores CPVarlenMetadata replica_dedup build_dist_muon \
         pad_segments_to_multiple; do echo "$p: $(git -C $U grep -c "$p" $M 2>/dev/null | wc -l)"; done
for c in 086bf6c16 3386c6d6f eb92f645c; do
  for p in torchtitan/components/metrics.py torchtitan/components/validate.py \
           torchtitan/components/validator.py torchtitan/components/optim/optimizer.py \
           torchtitan/distributed/parallelism_context.py torchtitan/components/data/text_datasets.py \
           torchtitan/distributed/full_dtensor.py; do
    git -C $U cat-file -e $c:$p 2>/dev/null && echo "$c OK $p" || echo "$c GONE $p"
  done
done
```

---

## 7. 附录 B：patch × 上游 mainline 逐项对照审计（原 `2026-10-14-patches-vs-mainline.md` 全文）

> 本附录写于上游基线 `3386c6d6f`（2026-10-08），末尾的 delta 小节记录了到 `eb92f645c` 的 74 个提交差异。

## patches/torchtitan 对照上游 mainline 的审计（pin 仍为 torchtitan 0.3.0）

- 我方 patch 根：`torchtitan_npu/patches/torchtitan/`（44 个 .py，含 `__init__.py`；实际 patch 模块 35 个）
- 我方 pin：`torchtitan==0.3.0`（`086bf6c16`，2026-09-02，见 `requirements.txt` / `.ci/setup_torchtitan.sh`）
- 对照基线：上游 mainline `3386c6d6f`（2026-10-08，`~/pytorch/torchtitan`）
- 分类口径：① 已被上游吸收（升级 pin 即删）；② 上游已吸收但形态/名字/机制不同（按新形态重写）；③ 上游仍无此能力（保留，但锚点在 mainline 已漂移 → 升级时必须重写）；④ 与上游方向冲突（重新设计，例如宿主文件被删）
- 本文只做对照与分类，不改变任何代码；工作量 S/M/L 为审计估计


<!-- section g1 -->

## TorchTitan-NPU patch 审计 — Group 1（models/common + qwen3_5）

**审计对象**：`torchtitan_npu/patches/torchtitan/models/common/{aux_loss,linear,moe,rope,token_dispatcher}.py`、`torchtitan_npu/patches/torchtitan/models/qwen3_5/vision_encoder.py`
**我们的仓库**：`/home/lrwei/cann/torchtitan-npu`（`requirements.txt` pin `torchtitan==0.3.0`）
**pin 的上游源码**：`/home/lrwei/pytorch/torchtitan-v030` = tag `v0.3.0` = `086bf6c16`（2026-09-02）
**最新 mainline**：`/home/lrwei/pytorch/torchtitan` HEAD = `3386c6d6f7a8dd24a5da7766f7d310122c7ddf7f`（2026-10-08，`[kimi] Update and Make Kimi K3 B200 numerics reproducible (#5143)`，detached at `origin/main`）
**只读**：本次审计未修改任何文件。

---

### 汇总表

| # | 模块 | 类型 | `Upstream-PR` 标记（原文） | mainline 现状（路径:行） | 引入 commit/PR | 分类 | 工作量 |
|---|------|------|---------------------------|--------------------------|----------------|------|--------|
| 1 | `models/common/aux_loss.py` | 新版上游文件 backport（独立模块，非注入） | `# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/3864` | 已存在：`torchtitan/models/common/aux_loss.py:80 AuxLoss`，API 同名但**归一化机制不同**（`inject(..., denominator=)` 逐次传入，无 `set_step_denominator`；`torch_remat` 区域保护） | `183efed45`（PR #3864，2026-09-10） | **②** | M |
| 2 | `models/common/linear.py` | fork 分支文件 backport（独立模块，非注入） | `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634`（**标记已失效**） | main 中**不存在 `BatchedLinear`**（全历史 `git log -S BatchedLinear` 仅命中 `br_dpsk_v4_1` 分支）。同能力改由 `Linear`/`GroupedLinear`（`linear.py` 的 `num_linears`）+ DSV4 `wo_a: Linear.Config` 的 `view+einsum`（`deepseek_v4/attention.py:254,383-392`）实现 | PR #3634 合并为 `0ff246463`（**未含** BatchedLinear）；`eb83b90cb`（PR #4872）引入 `GroupedLinear/num_linears` | **②**（形态被取代，带 ④ 性质） | M |
| 3 | `models/common/moe.py` | precise-injection（import 时替换 5 个上游类 + 2 个 config 工厂） | `# Pending upstream PR: .../3634`、`# Pending upstream PR: .../4095` | 拆分核对见下：hash 路由→ `deepseek_v4/moe.py:37 DeepSeekV4Router`；`input_ids` 透传→ `MoE.forward(**router_kwargs)`；`sqrtsoftplus`→ `activation.py:140 SqrtSoftplus`；`swiglu_limit`→ `activation.py:173 ClampedSwiGLU`；`GroupedExperts` **已删除**（`RoutedExperts` 改为 `w13/w2/activation_fn`） | `0ff246463`(3634) / `183efed45`(3864) / `e07084202`(4609) / `eb83b90cb`(4872) / `286578760`(5147) | **②**（整体；含 ①③④ 子项） | L |
| 4 | `models/common/rope.py` | precise-injection（monkey-patch `RoPE.__init__/_split/_unsplit/forward`、两个 `apply_rotary_emb`、替换两个 `Config`） | Backport 头：fork 提交 `24dd1a1`（"rope: make the rotary split a config field"）+ `https://github.com/pytorch/torchtitan/pull/3634`（merged as `0ff2464`） | 可选 `key` + `inverse` **已上游**（`rope.py:138,161`，同一 PR `0ff246463`）；**`split` 字段 main 全历史不存在**（`git log -S split -- rope.py` 空）；上游改由调用方切片：`deepseek_v4/attention.py:326-336,377-379` | `0ff246463`（PR #3634） | **②**（split 部分；key/inverse 部分为 ①，可删） | S–M |
| 5 | `models/common/token_dispatcher.py` | precise-injection（子类化 + 替换 3 个上游类与 3 个 `config_utils` 绑定） | `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4095` | 能力**仍无**：main 全树无 `absorb_router_scores` / `routed_scores*`；`combine` 仍后置乘分数（`token_dispatcher.py:160 _score_and_scatter_add`）。但锚点大量漂移（`_permute` 改签名、`combine` 去掉 legacy kwargs、`AllToAll` 改继承 `BaseEPTokenDispatcher`、`_sync_token_count_exchange` 返回 tensor） | PR #4095 **未合并**；`4a93ee4e4`(3970)、`27eb87115`(4958)、`73aed7f6c`(4121) 改变锚点 | **③**（保留，但"升级时必须重写"） | M–L |
| 6 | `models/qwen3_5/vision_encoder.py` | precise-injection（替换上游模块的 `compiled_create_block_mask` 与 `Qwen35VisionEncoder.forward`） | `Upstream PR: pending publication of the Qwen3.5 vision-mask fix.` / `Upstream base: b175497ea3502388bb9513391810a86ca46dbfa8`（= `Skip GraphTrainer H100 EP-overlap MoE failures (#4053)`，2026-08-03，在 v0.3.0 内） | 能力**仍无**（上游仍 `torch.compile(create_block_mask)`），但**锚点已移走**：`qwen3_5/vision_encoder.py` 不再暴露 `compiled_create_block_mask`，改调 `common/vision_encoder.py:106 create_block_diagonal_mask`（内部 `:38 compiled_create_block_mask`） | 锚点迁移见 `73aed7f6c`(4121) / `a8bf4ba83`(4960) | **③**（保留，但"升级时必须重写"） | S |

> **分类口径**：能力在上游存在（同名或异形）→ ①/②；上游完全不存在 → ③（保留，核对锚点）/④（与上游方向冲突）。混合模块取"主导能力"分类，并在逐模块小节列出每个子能力的分类。

#### 共同背景：4 个上游 PR 一次性吸收了本组的大部分能力

| upstream commit | PR | 日期 | 内容（与本组相关） | 是否在 pin `v0.3.0` |
|---|---|---|---|---|
| `0ff246463` | #3634 | 2026-09-03 | Add DeepSeek V4 model support：RoPE 可选 `key`+`inverse`、router `sqrtsoftplus`、`deepseek_v4/moe.py:37 DeepSeekV4Router`（hash 路由）、`MoE.forward(**router_kwargs)` | ❌ `NOT-in-v0.3.0` |
| `183efed45` | #3864 | 2026-09-10 | `models/common/aux_loss.py`（`AuxLoss` / `collect_aux_loss_metrics` / `register_aux_loss_zero_hook`） | ❌ |
| `e07084202` | #4609 | 2026-09-14 | `TokenChoiceTopKRouter` 可覆写钩子；`score_func` 由 `Literal[...]` 改为 `UnaryActivationFn.Config` | ❌ |
| `eb83b90cb` | #4872 | 2026-09-24 | 删除 `GroupedExperts`；`RoutedExperts` 改为一等 `w13/w2` `GroupedLinear` + `activation_fn` | ❌ |
| `286578760` | #5147 | 2026-10-08 | `ClampedSwiGLU`（`swiglu_alpha/swiglu_limit`）+ dist_moe 语义保留 | ❌ |

（`git merge-base --is-ancestor <c> v0.3.0` 对以上全部返回 false；`git tag --contains` 只列出 `ciflow/*` CI tag，无 release tag。）

---

### 1. `torchtitan_npu/patches/torchtitan/models/common/aux_loss.py`

#### 目标
提供 `AuxLoss` 基类、`collect_aux_loss_metrics`、`register_aux_loss_zero_hook`：把辅助 loss 的梯度在 backward 注入 carrier，并按 step 的**全局有效 token 数**归一化后聚合为 `{metric}/mean` 日志指标（`aux_loss.py:101` `class AuxLoss(Module)`、`:184` `inject`、`:174` `set_step_denominator`、`:245` `collect_aux_loss_metrics`、`:282` `register_aux_loss_zero_hook`）。

#### 上游标记
`# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/3864`（`aux_loss.py:1`）。

#### 注入方式
**新版上游文件的 backport（独立模块，非 monkey-patch）**：pin 的 0.3.0 里 `torchtitan/models/common/aux_loss.py` 不存在（已核对 `torchtitan-v030/.../common/` 与 site-packages），所以本文件是"把上游新文件搬过来"；它不替换任何上游符号，而是被我方代码直接 import：
- `torchtitan_npu/patches/torchtitan/components/metrics.py:15-27` → `collect_aux_loss_metrics(self.parallel_dims)`
- `torchtitan_npu/patches/torchtitan/trainer.py:153` → 生成 `AuxLoss.set_step_denominator(global_valid_tokens)` 代码
- `torchtitan_npu/models/deepseek_v3_2/model.py:191 SparseIndexerLoss(AuxLoss)`、`override/deepseek_v3_2/sparse_attn/ascendc.py:147-148`、`override/deepseek_v4/sparse_attn/golden.py:32-34,344`（`_should_run_forward`）

#### 上游 main 现状
符号**全部存在且同名**，但机制不同：

- `/home/lrwei/pytorch/torchtitan/torchtitan/models/common/aux_loss.py:80` `class AuxLoss(Module)` — 同 PR #3864 (`183efed45`, 2026-09-10)。
- `:150-156` `def inject(self, raw_sum, *, carrier, denominator) -> torch.Tensor:` — **denominator 由 owning model block 每次 forward 传入**（"The owning model block passes the applicable routed-token denominator to `inject` on every forward."，`:87-88`），**没有** `set_step_denominator`（main 全文件无此符号）。
- `:176-182` `remat.region(self._accumulate_and_inject, self.remat_region_name("aux_loss"), recompute=False)` + `import torch_remat as remat`（`:39`）——用 `torch_remat` 保留区域抑制 AC 重放；我们因为 pin 版本无 `torch_remat` 依赖改用 `_should_run_forward()` graph-task 判定（我们的 `aux_loss.py:62`）。
- `:199` `scale = 1.0 / denominator.clamp_min(1)` — 多了 `clamp_min(1)`。
- `:115` `reduce_mesh: str = "dp"` —— **我们的 `:140` 是 `"batch"`**（pin 的 `ParallelDims` mesh 名）。
- `:43` `from torchtitan.components.optim import OptimizersContainer`（#4904 重构后的路径）；`ParallelDims` → `ParallelismContext`（#4905），故 `:228-229 collect_aux_loss_metrics(parallelism_context)`、`:272 register_aux_loss_zero_hook(..., parallelism_context)`。
- `:55-72 _AuxLossInjection` 的 SPMD 类型声明改为 `spmd_typecheck(result, *, carrier, aux_loss)`（`:68-72`），不再走我们 `:86 typecheck_forward` + `spmd.assert_type`。
- 现有使用方：`torchtitan/models/common/moe.py:483 MicrobatchWiseLoadBalanceLoss(AuxLoss)`、`models/kimi_k3/model.py:291`、`models/deepseek_v4/model.py:80-116` 等（均按 `aux_loss_denominator` 逐层传参）。

#### 结论分类：**② 上游已吸收但形态/机制不同（需要重新实现）**
子项：① 文件本体与 API 名（`AuxLoss`/`collect_*`/`register_*`）已被吸收 → 可直接删除本文件；② 归一化契约不同（step 级 `set_step_denominator` ↔ 逐层 `denominator=`）→ 必须重写调用方。

#### 重写要点与工作量
1. 删除本 patch，改从 `torchtitan.models.common.aux_loss` import。
2. `SparseIndexerLoss`（我方 `models/deepseek_v3_2/model.py:191-260`）与 `override/deepseek_v3_2/sparse_attn/ascendc.py`、`override/deepseek_v4/sparse_attn/golden.py` 改为在 forward 处拿到每层 routed-token/valid-token 计数并传 `denominator=`；`trainer.py:153` 的 `set_step_denominator` 注入删除；`metrics.py` 改传 `ParallelismContext`（并注意 mesh 名 `batch`→`dp`）。
3. `_should_run_forward()` 是否需要保留取决于我们是否引入 `torch_remat`：若上游 pin 带 `torch_remat`，直接用 `remat.region(..., recompute=False)` 删除该辅助函数与 `golden.py:344` 的使用点。
4. **工作量 M**（1 个文件删除 + 4-6 处调用点/派生类改写 + UT `tests/unit_tests/patches/torchtitan/models/common/test_aux_loss.py` 重写）。上游对应文件/函数：`torchtitan/models/common/aux_loss.py` 的 `AuxLoss.inject`、`AuxLoss._accumulate_and_inject`、`collect_aux_loss_metrics`、`register_aux_loss_zero_hook`。

---

### 2. `torchtitan_npu/patches/torchtitan/models/common/linear.py`

#### 目标
新增 `BatchedLinear(Module)`：按 head 分组的线性层，权重展平为 `[H * D_out, D_in]`，forward 用 `reshape/transpose + torch.bmm` 实现每 head 独立投影（`linear.py:16` 类定义，`:39-44` forward）。用于 DSV4/DSV4.1 的 `wo_a` 与 DSV3.2 的 `w_uk`/`w_uv`（我方 `models/deepseek_v4/__init__.py:291`、`models/deepseek_v4_1/__init__.py:426`、`models/deepseek_v3_2/__init__.py:114`、`models/deepseek_v3_2/model.py:325-326`）。

#### 上游标记
`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634`（`linear.py:1`）——**标记已失效**：PR #3634 已合并为 `0ff246463`，`git grep -n BatchedLinear 0ff246463 -- '*.py'` **无命中**；`git log -S BatchedLinear --all` 只命中 fork 分支 `br_dpsk_v4_1` 的提交（`4080cb909`/`f8077d7d7`/`40cf15752`，`linear: add BatchedLinear for the grouped output projection`），均 `NOT-in-HEAD`。

#### 注入方式
**fork 分支文件/类的 backport（独立模块，非 monkey-patch）**：无 `apply()`，由我方模型直接 `from torchtitan_npu.patches...linear import BatchedLinear`。因为它只是一个新增类、不替换上游符号，升级成本取决于上游是否有等价形态。

#### 上游 main 现状
- `BatchedLinear` **在 main 中不存在**（`grep -rn "class BatchedLinear\|BatchedLinear" torchtitan/` 无命中；`n_batches` 同样无命中）。
- 同能力被上游用**别的机制**实现：
  - DSV4 的 grouped output projection：`wo_a: Linear.Config`（`torchtitan/models/deepseek_v4/attention.py:254`），forward 里 `wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)`（`:386`）+ `o = torch.einsum("tgd,grd->tgr", o, wo_a)`（`:392`）；config 在 `deepseek_v4/flavors.py:291`。
  - 通用"每个 group 一组权重"：`Linear.Config.num_linears`（`torchtitan/models/common/linear.py:47-51`，`num_linears` 在 `:50`，权重形状 `[num_linears, out, in]`）与 `GroupedLinear`（同文件 `:306`，`torch._grouped_mm`），引入于 `eb83b90cb`（PR #4872）。
  - LoRA 侧上游已支持这两种形态：`torchtitan/models/common/lora.py:51 _LoRALinearMixin`（含 `num_linears`）、`:199 _LoRAGroupedLinearMixin`、`:322 get_lora_linear` / `:328 get_lora_grouped_linear`——我方 `models/deepseek_v4/lora.py:117` 的 `issubclass(parent_cls, BatchedLinear)` 分支需要随之改写。
- DSV3.2（`w_uk`/`w_uv`）在 mainline **没有对应模型**（`torchtitan/models/` 无 `deepseek_v3_2`；v0.3.0 也没有，我方自研），这部分没有上游参照。

#### 结论分类：**② 上游已吸收但形态/名字/机制不同（需要重新实现）**
（带 ④ 性质：DSV4 的同一投影上游已用 `Linear + einsum` 落地，`BatchedLinear` 这种 `bmm` 形态在上游没有未来。）

#### 重写要点与工作量
1. `wo_a`（DSV4/DSV4.1）：改为 `Linear.Config(in_features=..., out_features=n_groups*o_lora_rank)`，在 attention forward 内 `weight.view(n_groups, o_lora_rank, -1)` + `einsum`，与 `deepseek_v4/attention.py:383-392` 对齐；sharding 按上游 `deepseek_v4/sharding.py:132-135` 的 `wo_a` 规则（普通 `Linear` 的 TP/FSDP 分片）。
2. `w_uk`/`w_uv`（DSV3.2，无上游模型）：建议改 `Linear.Config(num_linears=H)` 或 `GroupedLinear`，避免继续维护 fork 型类；若保留，也必须改为 `Module.Config.param_init` 语义（main 的 `Module.Config.param_init` 仍在，`protocols/module.py:96`，所以该类本身可继续 import，不会 import 崩，但会与上游形态分叉）。
3. LoRA：`models/deepseek_v4/lora.py`、`models/deepseek_v3_2/lora` 相关分支改用 `get_lora_linear`/`get_lora_grouped_linear`。
4. **工作量 M**（3 个模型的 attention + LoRA 集成；DSV3.2 需要自行设计）。

---

### 3. `torchtitan_npu/patches/torchtitan/models/common/moe.py`

#### 目标（多个能力，逐项）
| 子能力 | 我方位置 |
|---|---|
| DSV4 hash 路由（`hash`/`vocab_size`/`tid2eid`，hash 层不做 top-k、不施加 `expert_bias_E`） | `moe.py:65 HashRouter`、`:55 _build_hash_routing_table`、`:94-101` |
| `MoE.forward` 增加 `input_ids` 透传 | `moe.py:177 HashMoE`、`:194-200` |
| `"sqrtsoftplus"` score function | `moe.py:82-84,140-143` |
| `sorted_topk`（降序选择） | `moe.py:79-81,122-123` |
| 多模态 `vision_enabled` → `bias_vl` 图像掩码 choice bias | `moe.py:77-78,91-93,156-157` |
| `swiglu_limit`（DSV4 clamp：`up∈[-l,l]`、`gate≤l`）用于 routed experts 与 shared expert | `moe.py:348 _ClampGroupedExperts`、`:411 _ClampFeedForward`、`:439/:466` 两个 config 工厂包装 |
| 派发器对齐的 router score 吸收（配合 token_dispatcher patch） | `moe.py:293 _RouterScoreAbsorbingRoutedExperts` |

#### 上游标记
`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634`、`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4095`（`moe.py:1-2`）。

#### 注入方式
**precise-injection（import 时 monkey-patch 上游符号）**：`apply()`（`moe.py:486-511`，模块末尾直接调用）替换
`torchtitan.models.common.moe.{TokenChoiceTopKRouter, MoE, GroupedExperts, RoutedExperts}`、`torchtitan.models.common.feed_forward.FeedForward`，并包装 `config_utils.{make_routed_experts_config, make_ffn_config}`；依赖"先换类再 import `config_utils`"的顺序（`moe.py:490-505`）。

#### 上游 main 现状（逐子能力）

1. **hash 路由 → ② 已吸收，位置/名字不同。**
   上游把它做成模型私有 router：`/home/lrwei/pytorch/torchtitan/torchtitan/models/deepseek_v4/moe.py:37 class DeepSeekV4Router(TokenChoiceTopKRouter)`，`Config{vocab_size, n_hash_layers=3, layer_id=0}`（`:41-44`），`self.hash = config.layer_id < config.n_hash_layers`（`:51`），`tid2eid` buffer（`:54-62`），`_select_experts(self, scores_TE, expert_bias_E=None, *, input_ids_T=None, **router_kwargs)`（`:77-101`，hash 分支在 TP 下先 redistribute `input_ids` 再查表）。由 `0ff246463`（PR #3634）引入。config 装配见 `deepseek_v4/flavors.py:327,348-349`。
   我们的 `TokenChoiceTopKRouter.Config(hash=..., vocab_size=...)` 是**类替换**注入（我方 `models/deepseek_v4/__init__.py:333-345` 直接传 `hash=`/`vocab_size=`），main 的 `TokenChoiceTopKRouter.Config`（`moe.py:168-176`）**没有**这两个字段 → 升级即报 `unexpected keyword argument`。

2. **`input_ids` 透传 → ② 已吸收为 `**router_kwargs`。**
   `moe.py:679-687 def forward(self, x_TD, *, padding_mask_T=None, aux_loss_denominator=None, **router_kwargs)`（`:707-713` 把 kwargs 透给 `self.router(...)`）；`TokenChoiceTopKRouter.forward(..., **router_kwargs)`（`:218-233`）再透给 `remat.region(self._select_experts, "routing_decision", recompute=False)(scores_TE, expert_bias_E, padding_mask_T=..., **router_kwargs)`（`:259-267`）。所以 `HashMoE` 整个类（`:177-290`，还镜像了 pin 版 `(B,L,D)`/`sp_size`/seq padding 旧语义）**不需要存在**。

3. **`"sqrtsoftplus"` → ① 已吸收（可删）。**
   上游先在同一 PR 把该字符串加入 `score_func` Literal，随后 `e07084202`（PR #4609）把 `score_func` 改为 `UnaryActivationFn.Config`（`moe.py:171`），实现为 `activation.py:140 class SqrtSoftplus(UnaryActivationFn)`（`F.softplus(x).sqrt()`，`:156`）。我们的 `Literal["softmax","sigmoid","sqrtsoftplus"]`（`moe.py:82-84`）在新 pin 下是**类型/字段冲突**，应直接删除。

4. **`sorted_topk` → ③ 上游仍无。** main 默认仍是 `torch.topk(..., sorted=False)`（`moe.py:205-217`，该调用在 `:214-216`；main 全树无 `sorted_topk`）。但 PR #4609 之后有了干净的覆写缝：**只需覆写 `_select_experts`**（`moe.py:205`）。

5. **`swiglu_limit` clamp → ② 已被异形吸收（并有数值语义分歧）。**
   `GroupedExperts` **已被删除**（`git log -S "class GroupedExperts"` 的删除方为 `eb83b90cb`，PR #4872；main 全树 `git grep GroupedExperts HEAD` 无命中），`RoutedExperts` 改为 `w13`/`w2`（`GroupedLinear.Config`，`moe.py:66-67`）+ `activation_fn: BinaryActivationFn.Config = field(default_factory=SwiGLU.Config)`（`:69`）+ `output_postprocess`；`RoutedExperts.forward` 调 `self.activation_fn(gate_up_R2F, offsets=offsets_E)`（`:105-160` 区间）。
   上游的 clamp 落在 activation 上：`activation.py:173 class ClampedSwiGLU(BinaryActivationFn)`，`Config{swiglu_alpha=1.702, swiglu_limit=7.0}`（`:177-190`），实现为 `gate.clamp(max=limit)` / `up.clamp(min=-limit,max=limit)` / `silu = gate*sigmoid(alpha*gate)` / `torch.addcmul(silu, silu, up)`（`:192-201`），由 `286578760`（PR #5147）引入；dist_moe 侧对应 `common/dist_moe/routed_experts.py:70 swiglu_limit` 与 `config/transform/dist_moe.py:47-58`。
   ⚠️ **语义分歧**：我们 `moe.py:398-401` 是 `F.silu(gate)*up` + clamp；上游 `ClampedSwiGLU` 是 `alpha`-sigmoid + `addcmul` 形态（且 `swiglu_limit` 不允许 0/负数）。若 DSV4 参考实现要的是 `silu` 版 clamp → 需要新写一个 `BinaryActivationFn`（或向上游提 PR），不能直接复用 `ClampedSwiGLU` 的数值。
   我们的 `_ClampFeedForward(FeedForward)`（`moe.py:411-428`）也失去落脚点：main `feed_forward.py:38 class FeedForward` 仍是 `w1/w3/w2`，但上游 MoE 的 shared expert 通过 `shared_experts: FeedForward.Config`（`moe.py:638`）配置，没有 clamp 概念。

6. **`vision_enabled`/`bias_vl` → ③ 上游仍无。** main 全树无 `bias_vl`/`vision_enabled`/`image_mask`（`grep -rn` 无命中），且上游**没有 deepseek_v4_1 模型**（`torchtitan/models/deepseek_v4_1/` 只有未跟踪的 `__pycache__`，`git ls-files` 为空）。锚点 `TokenChoiceTopKRouter` 仍存在（`moe.py:162`），但配置面已变成 `UnaryActivationFn`/`HiMidLoLinear`（`:170-171`），**升级时必须重写**（建议继续做成自己的 router 子类，新增 `bias_vl` 参数 + 覆写 `_select_experts`，并在 `sharding`/`state_dict_adapter` 侧同步，我方已有 `models/deepseek_v4_1/sharding.py:269-270`、`vision/state_dict_adapter.py:40`）。

7. **`_RouterScoreAbsorbingRoutedExperts` → 依附第 5 号模块的 ③**：main `RoutedExperts.forward`（`moe.py:105`）签名仍为 `(x_TD/topk_scores_TK/topk_expert_ids_TK/num_local_tokens_per_expert_E)`，但内部没有 `inner_experts`（我方 `moe.py:332` 调 `self.inner_experts(..., routed_scores_R=...)` 会 AttributeError），也没有 metadata 里的分数。

8. **附带失效的其它锚点**：`config_utils.make_routed_experts_config`（`config_utils.py:371`）现在返回 `RoutedExperts.Config(w13=..., w2=..., token_dispatcher=AllToAllTokenDispatcher.Config(...))`，**没有 `inner_experts` 字段**（我方 `moe.py:458-462` 直接访问 `cfg.inner_experts` → AttributeError），`config_utils` 的 import（`:36-40`）里也**没有 `GroupedExperts`**（我方 `moe.py:504` 会给模块凭空加属性）；`register_moe_load_balancing_hook` 已从 `torchtitan.components.optimizer` 迁到 `torchtitan/models/common/moe.py:849`。

#### 结论分类：**② 上游已吸收但形态/名字/机制不同（需要重新实现）**
子项汇总：hash ②、`input_ids` ②、`sqrtsoftplus` ①、`sorted_topk` ③、clamp ②（数值语义需重新设计）、`bias_vl` ③、score 吸收 ③（见模块 5）。**不能"升级 pin 后直接删"**，因为 hash 配置字段、`HashMoE`、clamp 类、`RoutedExperts`/`GroupedExperts` 注入全部与新上游不兼容。

#### 重写要点与工作量
1. 删除 5 处类替换与 2 个 config 工厂包装；`HashRouter` 改为两层：
   - hash 层直接用上游 `torchtitan.models.deepseek_v4.moe.DeepSeekV4Router`（`Config(vocab_size, n_hash_layers, layer_id)`）——我方 `models/deepseek_v4/__init__.py:333-345` 的 `hash=/vocab_size=` 改为 `layer_id/n_hash_layers`；
   - 多模态扩展做**我们自己的子类**：覆写 `_select_experts(scores_TE, expert_bias_E=None, *, input_ids_T=None, image_mask=None, **router_kwargs)` 实现 `bias_vl` 替换 + `sorted_topk`（`torch.topk(..., sorted=True)`），不 patch 上游 forward。
2. `score_func` 从字符串改成 `SqrtSoftplus.Config()`（或直接 `activation.SqrtSoftplus.Config`）。
3. clamp：在 `RoutedExperts.Config.activation_fn` 上提供一个 `silu` 版 clamped 激活（新类），shared expert 若要 clamp 则在 `FeedForward` 之后自行处理；如果接受 `ClampedSwiGLU` 的数值语义，则零成本（改一行 config）。
4. `MoE.forward` 调用改成 `(T, D)` + `**router_kwargs`（`input_ids=`/`image_mask=` 作为 router kwargs 传入），删除 `HashMoE`；注意 main 的 `padding_mask_T` 与 `_maybe_shard_routed_branch_inputs_across_tp` 语义。
5. `_RouterScoreAbsorbingRoutedExperts` 待模块 5 定案后再定（若保留 pre-W2 吸收，需要在新的 `RoutedExperts.forward` 里把 `metadata.routed_scores_R` 传进 `activation_fn`/`w2` 之间）。
6. **工作量 L**（500 行 patch 基本重写；影响 `models/deepseek_v4`、`models/deepseek_v4_1`、`models/deepseek_v3_2`、`models/qwen3_5` 的 MoE 配置 + sharding + state_dict_adapter + UT）。
   上游对应：`common/moe.py`（`RoutedExperts`、`TokenChoiceTopKRouter._select_experts`、`MoE.forward`、`MicrobatchWiseLoadBalanceLoss`）、`common/activation.py`（`SqrtSoftplus`、`ClampedSwiGLU`）、`common/config_utils.py`（`make_routed_experts_config`）、`deepseek_v4/moe.py`（`DeepSeekV4Router`）。

---

### 4. `torchtitan_npu/patches/torchtitan/models/common/rope.py`

#### 目标
把"部分 RoPE（partial RoPE）"做进 `RoPE` 本身：新增 `split` 配置字段与 `SplitComplexRoPEConfig`/`SplitCosSinRoPEConfig`（`rope.py:30-46`），forward 只旋转尾部 `dim` 通道并拼回前缀（`rope.py:77-97`），同时 backport 可选 `key` 与 `inverse` 语义（`rope.py:100-140`）。

#### 上游标记
Backport 头（`rope.py:1-10`）：fork 分支 `sdmyzlp/torchtitan` `br_dpsk_v4_1` 的 commit `24dd1a1`（"rope: make the rotary split a config field"）+ `https://github.com/pytorch/torchtitan/pull/3634`（merged as `0ff2464`）。

#### 注入方式
**precise-injection**：`apply()`（`rope.py:143-170`，import 时执行）替换 `RoPE.__init__/_split/_unsplit/forward`、`ComplexRoPE.apply_rotary_emb`、`CosSinRoPE.apply_rotary_emb`，并把 `ComplexRoPE.Config`/`CosSinRoPE.Config` **整体换成** `Split*RoPEConfig`（`:161-166`）。由 `torchtitan_npu/patches/torchtitan/__init__.py:18` 随包自动 import 生效。

#### 上游 main 现状
- `24dd1a1` **不在 mainline**（`git log -1 24dd1a1` 命中本地 fork 分支提交，`git merge-base --is-ancestor 24dd1a1 HEAD` = false）；`split` 在 `torchtitan/models/common/rope.py` 的**全部历史**中都没出现过（`git log -S split -- torchtitan/models/common/rope.py` 为空输出）。
- 可选 `key` + `inverse` **已被 `0ff246463`（PR #3634）吸收，且名字完全一致**：
  - `/home/lrwei/pytorch/torchtitan/torchtitan/models/common/rope.py:138-144` `def apply_rotary_emb(query, key: torch.Tensor | None, rope_cache, *, inverse: bool = False)`
  - `:161-171` `def forward(self, query, key=None, positions=None, *, inverse=False)`
  - `:257-276` `ComplexRoPE.apply_rotary_emb`（含 `if inverse: rope_cache = rope_cache.conj()`）
  - `:348-357` `CosSinRoPE.apply_rotary_emb`（`inverse` → `NotImplementedError("CosSinRoPE does not support inverse rotation.")`，与我们 `rope.py:129-130` 一致）
  - 新增 `:284 CosSinRoPE.forward` 覆写（`:284-292` 调 `super().forward(...)`）——monkey-patch `RoPE.forward` 仍会被走到，但语义已由上游接管。
- **上游把 partial RoPE 放在调用方**：`torchtitan/models/deepseek_v4/attention.py:326` `q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)`、`:332` 对 kv 同样切片、`:334` `q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(1), positions)`、`:335-336` 拼回；反旋转路径 `:377-379` `o_rope = self.rope(o_rope, positions=positions, inverse=True)`（只传 query，正是我们 backport 的"可选 key"表面）。`deepseek_v4/compressor.py:174,234` 同模式。

#### 结论分类：**② 上游已吸收但形态/机制不同（需要重新实现）**
子项：**①** 可选 `key` + `inverse`（同名同义，可直接删掉 `rope.py:100-140` 的 backport 部分）；**②** partial RoPE 能力上游存在，但落在 attention 调用方而不是 RoPE 配置（`split` 字段方向未被上游采纳，"升级时必须按调用方切片重写"）；**③** `Split*RoPEConfig` 这类我们自用的 config 若仍想保留，需要在 main 的 `rope.py` 上重新锚定（`RoPE.__init__(config)`/`forward`/静态 `apply_rotary_emb` 都还在，改动量小）。

#### 重写要点与工作量
1. 删除 `_complex_apply_rotary_emb`/`_cossin_apply_rotary_emb`/`_rope_forward` 的 monkey-patch（上游已具备可选 key + inverse + `CosSinRoPE` inverse 报错）。
2. 我方模型配置（`models/deepseek_v4/__init__.py:111,149,258,508`、`models/deepseek_v4_1/__init__.py:353-354` 的 `dataclasses.replace(rope, split=...)`）改为在 attention 层 `torch.split` 后调 rope（对齐 `deepseek_v4/attention.py:326-336,377-379`）；`torchtitan_npu/override/common/rope.py:30-31,126,205,254,309-313` 的 Ascend 专用 rope（`AscPartialComplexRoPE` 等）随之改造。
3. 若短期要保留 `split` 走 RoPE 内部：仅保留 `Split*RoPEConfig` + `_rope_init/_split/_unsplit/_rope_forward`，删除两个 `apply_rotary_emb` 替换（上游已含 inverse/None-key 分支），并核对 `CosSinRoPE.forward` 覆写路径。
4. **工作量 S–M**（上游对齐路线：S–M，牵动 2 个模型配置 + 1 个 override；保留 split 路线：S）。

---

### 5. `torchtitan_npu/patches/torchtitan/models/common/token_dispatcher.py`

#### 目标
给标准 EP 派发路径增加"**pre-W2 router score 吸收**"：dispatch 时把 expert-sorted 分数随 token 一起（含 EP all-to-all、rank-major→expert-major 重排）搬运到 metadata 的 `routed_scores_R`，让 grouped experts 在下投影前乘分数，combine 侧跳过乘分（`token_dispatcher.py:47 LocalDispatchMetadata.routed_scores_R`、`:54 AllToAllDispatchMetadata.routed_scores_R`、`:60-119 Local`、`:122-358 AllToAll`、`:361-376 DeepEP 配置旋钮`）。同时修掉 `_sync_token_count_exchange` 的 async D2H 竞争（`:125-160`）。

#### 上游标记
`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4095`（`token_dispatcher.py:1`）。

#### 注入方式
**precise-injection**：`apply()`（`:379-403`，import 时执行）替换 `torchtitan.models.common.token_dispatcher.{LocalTokenDispatcher, AllToAllTokenDispatcher, DeepEPTokenDispatcher}` 与 `config_utils` 里对应 3 个绑定；消费方是 `torchtitan_npu/override/common/token_dispatcher.py`（NPU 融合算子派发器，`:184,315,381` 读 `absorb_router_scores`/`routed_scores_R`）。

#### 上游 main 现状
- **能力仍不存在**：`grep -rn "absorb_router_scores\|routed_scores" torchtitan/` **零命中**；PR #4095 **未合并**（`git log --all --grep="(#4095)"` 无命中）。main 的 `LocalDispatchMetadata`（`:22-27`）只有 `token_indices_experts_sorted_N` / `topk_scores_experts_sorted_N`；分数仍在 combine 后置乘：`LocalTokenDispatcher.combine`（`:136-158`）→ `_score_and_scatter_add`（`:160-179`，`routed_output_ND * topk_scores_N`），`AllToAllTokenDispatcher.combine`（`:595-668`）同。
- **但锚点漂移严重（"升级时必须重写"）**：
  1. `AllToAllTokenDispatcher` 现在继承 `BaseEPTokenDispatcher`（`:181`、`:235`），不再是 `LocalTokenDispatcher` 的直系子类 → 我们 `:179 LocalTokenDispatcher.dispatch(self, ...)` 的"EP=1 回退"调用需改为 `BaseEPTokenDispatcher`/新的 EP=1 路径。
  2. `_permute(self, routed_input_RD, num_global_tokens_per_local_expert_E)`（`:535-587`）**返回 3 元组** `(routed_input_RD[permuted_indices,:], permuted_indices, num_global_tokens_per_local_expert_e)`，`input_shape` 由 `dispatch` 自行构造（我方 `:265-292` 的 5 元组覆写 `_permute(..., routed_scores_rank_major_R)` 完全对不上）。
  3. `combine(self, routed_output_RD, metadata, x_TD)`（`:595`）**没有** `num_local_tokens_after_padding` / `local_seq_len_after_padding`（我方 `:294-358` 的 legacy 参数与 `num_local_tokens_after_seq_dim_padding` 分支已过时）。
  4. `_sync_token_count_exchange`（`:283-313`）返回 **tensor** `(num_global_tokens_per_local_expert_E, input_splits, output_splits)`，而 pin 版返回 list；竞争点仍在（`input_splits` 仍 `non_blocking=True`，`:306-309`），我们 `:141-154` 的同步拷贝修复**思路仍然有效但签名/返回类型要重写**。
  5. main 无 `sp_size` / `_sp_global_token_indices`（我方 `:334,339`）：SP 由 `#4121`（`73aed7f6c`，fold batch dim）与 spmd region 处理。
  6. `config_utils` 现在只 import `AllToAllTokenDispatcher`（`config_utils.py:44`）→ 我方 `:392-400` 的三个绑定要减为 1 个（`LocalTokenDispatcher` 由上游 `AllToAllTokenDispatcher` 内部构造，`config_utils.py:400-403`）。
  7. `DeepEPTokenDispatcher.Config`（`:841-855`）字段完全不同（`cuda_graph_compatible`/`num_max_tokens_per_rank`/`hidden_dim`，且 `__init__` 强制校验），我们 `:370-372` 只加一个布尔字段的形态已无意义；`EPDispatchMetadata`（`:814-818`）是 DeepEP/HybridEP 的新 metadata 契约。
- 仍存在的锚点：`deterministic_scatter_add`（`torchtitan/ops/scatter_add.py:11`）、`_local_reorder`（`:59`）、`_dispatch_token_exchange`（`:317`）、`_combine_token_exchange`（`:343`）、`maybe_set_sparse_mesh`/`current_spmd_mesh`/`get_spmd_backend` 等。

#### 结论分类：**③ 上游仍无（保留）——但锚点改名/移动/语义变化，标为"升级时必须重写"**

#### 重写要点与工作量
1. 保留 `absorb_router_scores` 语义，但按新契约重写：`LocalDispatchMetadata.routed_scores_R` 扩展仍可行（frozen dataclass 继承注 `kw_only=True` 即可）；`AllToAll._permute` 改为新 3 元组并在 `dispatch` 内自行对 `routed_scores_rank_major_R[permuted_indices]` 做同样索引；`combine` 去掉 legacy 参数。
2. 重新锚定 `_sync_token_count_exchange` 覆写（返回 tensor，保持 `non_blocking=False` 同步拷贝；这个 compile 竞争修复值得顺带提到上游）。
3. `config_utils` 绑定只剩 `AllToAllTokenDispatcher`；我方 override（`torchtitan_npu/override/common/token_dispatcher.py`，含 `npu_moe_re_routing`/`npu_moe_token_unpermute` 融合路径）需要同步改造（`:184,315,381`）。
4. 与模块 3 的 `_RouterScoreAbsorbingRoutedExperts` 联动：新的 `RoutedExperts.forward` 里要把 `metadata.routed_scores_R` 接到 activation 之前/`w2` 之前。
5. **工作量 M–L**（patch 主体 M；连带 NPU fused override 与 MoE 侧接线为 L）。

---

### 6. `torchtitan_npu/patches/torchtitan/models/qwen3_5/vision_encoder.py`

#### 目标
两个窄修复（`vision_encoder.py:39-49`）：
1. `eager_create_block_mask`：绕开 `torch.compile(create_block_mask)`，改用 eager 构建（并 `mark_dense_sdpa_mask_mod` 打标走 NPU dense-SDPA 回退）；
2. `owning_vision_encoder_forward`：对 `Qwen35VisionEncoder.forward` 输出 `.clone()`，返回 owning tensor 以保住 FSDP pre-backward hook。

#### 上游标记
`Upstream PR: pending publication of the Qwen3.5 vision-mask fix.` + `Upstream base: b175497ea3502388bb9513391810a86ca46dbfa8`（`vision_encoder.py:8-9`）。该 base commit 实为 `Skip GraphTrainer H100 EP-overlap MoE failures (#4053)`（2026-08-03），`git merge-base --is-ancestor b175497 v0.3.0` = true（在 pin 内）。

#### 注入方式
**precise-injection（显式 apply）**：`apply()`（`:52-67`）在 `torchtitan.models.qwen3_5.vision_encoder` 命名空间上替换 `compiled_create_block_mask` 与 `Qwen35VisionEncoder.forward`，并做了 API 守卫（`:56-62` 遇到未知实现直接 `RuntimeError`）。由我方 `torchtitan_npu/models/qwen3_5/__init__.py:16-22` 显式调用（不在 `patches/torchtitan/__init__.py` 的自动链上）；UT 断言见 `tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py:53-58`。

#### 上游 main 现状
- 能力**仍无**：上游依旧把 mask 构建编译掉——`/home/lrwei/pytorch/torchtitan/torchtitan/models/common/vision_encoder.py:38` `compiled_create_block_mask = torch.compile(create_block_mask)`，`:106 def create_block_diagonal_mask(...)`，`:123 return compiled_create_block_mask(mask_mod, 1, None, total_tokens, total_tokens, device=device)`（注意力侧还有 `common/attention/attention.py:707-712 _compiled_create_block_mask`）。CANn 下编译失败的规避仍未上游化。
- **锚点 1 已经移走**：`torchtitan/models/qwen3_5/vision_encoder.py` 不再 import/暴露 `compiled_create_block_mask`（`grep -n compiled_create_block_mask` 在该文件**无命中**）；`Qwen35VisionEncoder.forward`（`:392-447`）现在调 `create_block_diagonal_mask(segment_lengths, total_tokens, x.device)`（`:433`，import 于 `:27`）。
  → 我方 `apply()` 第一行 `mask_builder = upstream_vision.compiled_create_block_mask`（`:54`）在 main 上会直接 **AttributeError**。
- **锚点 2 语义变化**：main 的 forward（`:392-447`）输入改为 `(pixel_values (T, patch_dim), grid_thw)`（`:4121` fold batch dim），内部 `create_block_diagonal_mask`；返回仍是 `self.merger(x)`（`:447`），`PatchMerger.forward`（`:285-298`）返回 `linear_fc2(...)`。owning-tensor 问题是否仍存在取决于 FSDP/NPU 行为，需要在新 pin 上重测，不能照搬。
  锚点迁移可归因于 `73aed7f6c`（#4121，fold batch dim）与 `a8bf4ba83`（#4960，local_spmd regions）。

#### 结论分类：**③ 上游仍无（保留）——但锚点已移动、语义已改，标为"升级时必须重写"**

#### 重写要点与工作量
1. 掩码注入点改为 `torchtitan/models/common/vision_encoder.py` 的 `compiled_create_block_mask`（或更稳：包 `create_block_diagonal_mask`，在内部把 `_compile=False` 传给未编译的 `create_block_mask`），保留 `mark_dense_sdpa_mask_mod` 打标；`apply()` 的 API 守卫按新模块/新符号重写（守卫本身是资产，建议保留并更新断言对象）。
2. `Qwen35VisionEncoder.forward` 的 `.clone()` 包装：在新 pin 上先复现 FSDP pre-backward hook 丢失问题（若 upstream `forward` 现在直接返回 `merger` 输出、不再是 view，可考虑删除该修复，减少注入面）。
3. UT `tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py:53-58` 的断言路径同步更新。
4. **工作量 S**（2 个注入点、1 个 UT；风险主要在"新 pin 下 compiled mask 是否仍失败"需要 NPU 复测）。

---

### 附：升级 pin 时的建议顺序（本组视角）

1. **可直接删**：`aux_loss.py` 的文件本体（改 import + API 调用）、`linear.py` 的 `BatchedLinear`（若接受 `Linear+einsum`/`GroupedLinear` 形态）、`rope.py` 的 key/inverse backport 段、`moe.py` 的 `sqrtsoftplus` 字符串分支。
2. **必须重写（先做，风险最高）**：`moe.py`（hash 字段/`HashMoE`/clamp 类/`GroupedExperts` 全部失效，`config_utils.make_routed_experts_config` 不再有 `inner_experts`）、`token_dispatcher.py`（`_permute`/`combine`/`_sync_token_count_exchange`/父类/`config_utils` 绑定）、`rope.py` 的 split（移到 attention 调用方）。
3. **改写注入点**：`qwen3_5/vision_encoder.py`（`compiled_create_block_mask` 换模块）。
4. 验证面：`tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py`、`tests/unit_tests/patches/torchtitan/models/common/test_aux_loss.py`、`tests/unit_tests/override/common/test_moe_score_absorption.py`、`tests/unit_tests/models/deepseek_v3_2/test_indexer_loss.py`、`tests/unit_tests/models/deepseek_v4/*`（LoRA 相关）都会因本组重构而需要更新。

<!-- section g3 -->

## Patch Audit — Group 3: `experiments/graph_trainer` + `experiments/torchft`

Read-only audit. Our repo: `/home/lrwei/cann/torchtitan-npu` (pinned torchtitan 0.3.0, tag `v0.3.0` = `086bf6c166`, 2026-09-02).
Latest upstream mainline: `/home/lrwei/pytorch/torchtitan` @ `3386c6d6f` (2026-10-08).
Patch modules: `torchtitan_npu/patches/torchtitan/experiments/{graph_trainer,torchft}/`.

### Headline

* Upstream **kept** both experiments (`torchtitan/experiments/graph_trainer/`, `torchtitan/experiments/torchft/` both still exist), but **both were restructured**:
  * `ep_chunk_pass.py` (2455 lines) was **deleted** by `43f73649f [graph_trainer] Remove graph-based EP chunking (#4952)` (2026-09-30); graph-based EP chunking was replaced by **eager** chunking in `ep_eager_chunk.py`. Three of our nine graph_trainer patches target that dead file.
  * `GraphTrainer` gained `GraphTrainingEngine(TrainingEngine)`; `GraphTrainer._make_fx_forward_backward_step` no longer exists; `construct_default_graph_passes(..., parallel_dims=)` was renamed to `parallelism_context=`.
  * `FaultTolerantTrainer(Trainer)` → `FaultTolerantTrainer(Configurable)` + `FaultTolerantTrainingEngine(TrainingEngine)`; the entire `train_step`/`__init__` we backport no longer has that shape.
* **All 6 TorchFT-related upstream PRs** referenced by our torchft patches **are merged** in main (`cd890fedc`, `91abd2301`, `277d4c930`, `59bbedfcd`, `d771abb06`, `f5c97dc0e`) → 4 modules in that group are dead weight.
* Of the 9 graph_trainer modules, only **`ep_ready_nodes_dedup.py` (PR #4516)** was absorbed; `ep_chunk_concretization.py` (#4529) was absorbed then its host file deleted; the remaining 6 PRs (#4734, #4650, #4651, #4708, #4763, #3949) have **no commit on any ref** in the upstream checkout.

### Summary table

| # | Module | 目标 symbol(s) | 上游标记 | main 现状 | 结论 | 工作量 | 形态 | 加载路径 |
|---|--------|----------------|----------|-----------|------|--------|------|----------|
| 1 | `graph_trainer/chunked_loss.py` | wraps `chunked_loss.ChunkedLossWrapperWithParamGrads.__call__` / `._gradient_backprop` | `Pending upstream PR: …/pull/4734` (L1) | **PR #4734 not merged.** Anchor rewritten by `f93fd4ccf` (#4451, MTP, post-0.3.0): `_gradient_backprop` now an **instance method** with tuple args; `ReplicateComputation.full_dtensor` **deleted** from `simple_fsdp.py` | **③ 上游仍无（保留；锚点已变 → 必须重写）** | M | 精准 monkey-patch（2 个方法） | **auto** (`patches/torchtitan/__init__.py:12`) |
| 2 | `graph_trainer/ep_chunk_concretization.py` | wraps `ep_chunk_pass.populate_chunk_dim_metadata_pass`, `mark_chunk_dynamic_dims`, `prepare_ep_overlap_trace_inputs`, `_fresh_unbacked_symbol_for_copy`, `ep_pass_utils.{chunk_symbol_hints_for_mode,_placeholder_symbol_hints,_concretize_value}` + `registry.TRACE_INPUT_PREPARERS` | `Pending upstream PR: …/pull/4529` (L1) | PR #4529 **merged** (`79008449f`, in HEAD) **but host `ep_chunk_pass.py` deleted** by `43f73649f` (#4952); `ep_pass_utils` lost `_record_symbols_from_extent`/`tensor_meta` | **④ 与上游方向冲突（重新设计）** | L | 7 个 wrap 点 + registry 注入 | **auto** (L13) |
| 3 | `graph_trainer/ep_forward_accumulation.py` | wraps `ep_chunk_pass._materialize_live_out` | `Pending upstream PR: …/pull/4708` (L1) | **Host file gone** (`#4952`); PR #4708 unmerged | **④ 与上游方向冲突（重新设计）** | L | 精准 monkey-patch（1 个函数） | **opt-in** (`functionalize_recompute_mutations.py:95,99`) |
| 4 | `graph_trainer/ep_overlap_shape_queries.py` | wraps `ep_overlap_pass._collect_token_exchanges` | `Pending upstream PR: …/pull/4650` (L1) | **PR #4650 not merged.** Anchor **byte-identical** to v0.3.0: `_collect_token_exchanges` main `ep_overlap_pass.py:196`; helpers `_custom_meta:111`, `_is_token_exchange_launch:123`; `_EP_TOKEN_EXCHANGE` `common_utils.py:211` | **③ 上游仍无（保留；锚点未变）** | S | 精准 monkey-patch（1 个函数） | **auto** (L14) |
| 5 | `graph_trainer/ep_ready_nodes_dedup.py` | replaces `ep_overlap_pass._ready_nodes` | `Pending upstream PR: …/pull/4516` (L1) | **已吸收** `c4bef0bb7` (#4516): main `ep_overlap_pass.py:514-541` 含 `selected: set[fx.Node]` (`:526`) + `if node in selected: continue` — 与补丁逐行等价 | **① 已被上游吸收（可删）** | S（删） | 精准 monkey-patch | **auto** (L15) |
| 6 | `graph_trainer/ep_shape_live_out.py` | wraps `ep_chunk_pass._split_live_out_users` | `Pending upstream PR: …/pull/4651` (L1) | **PR #4651 not merged** and `_split_live_out_users` **不存在于 main**（随 `ep_chunk_pass.py` 一起删除） | **④ 与上游方向冲突（重新设计）** | L | 精准 monkey-patch（1 个函数） | **auto** (L16) |
| 7 | `graph_trainer/functionalize_recompute_mutations.py` | registers pass pipeline `"mutation-functionalization"`; wraps `passes.construct_default_graph_passes` + adds `functionalize_recompute_mutations_pass` | 无 `Upstream-PR:` 头；docstring `Backport of … upstream PR …/pull/4708 (head a64792d2, fixing issue 4688)` (L6-12) | **Pass 不存在于 main**（`git log -S functionalize_recompute_mutations_pass` 全 ref 无结果）。但上游用**另一条路线**解决同类问题：`bc86435b7 [1/2] … Save forward dependencies hidden by mutations (#4842)` 新增 `mutation_utils.py`（`mutation_deps:195`），仅覆盖 **GraphPP** 路径 | **③ 上游仍无（保留；锚点已变 → 必须重写）**，并关注 #4842 [2/2] 可能取代之 | M | backport 新上游文件（pass + pipeline 注册） | **opt-in**（`models/deepseek_v4/config_registry.py:44`） |
| 8 | `graph_trainer/graph_trainer_runtime_context.py` | wraps `passes.apply_graph_passes` + `GraphTrainer._make_fx_forward_backward_step` | `Pending upstream PR: …/pull/4763` (L1) | **PR #4763 not merged.** `apply_graph_passes` 仍在 main `passes.py:506`（签名无 `runtime_context`）；`construct_default_graph_passes` 无 `runtime_context`；`GraphTrainer._make_fx_forward_backward_step` **已删除**（`GraphTrainingEngine` + `spmd_graph_builder`） | **③ 上游仍无（保留；锚点已变 → 必须重写）** | M | monkey-patch 注入（模块属性 + 类方法 + ContextVar） | **opt-in** (`extensions/.../graph_trainer/__init__.py:24-30`) |
| 9 | `graph_trainer/memory_policy.py` | 并行实现 `memory_policy.tag_sac_policy`；新增 `NodePolicyKey`/`make_node_override_memory_policy`；注入 `GraphTrainerCompileConfig.memory_policy` 注解 | `Pending upstream PR: …/pull/3949` (L1) | **PR #3949 not merged**；`save_input_every_n_layers` 全 ref 无出现。main `tag_sac_policy` 在 `memory_policy.py:282`（有 `force_save_nodes`，**无** `save_input_every_n_layers`），且主体已改写（新增 cacheable-effect 保存、边界规则改为 layer-id 不等）；CLI 注解 hack 已过时（main `configs.py:201` 仍 `Literal[...]`，Tyro 已被 `eeeb39b42` 移除） | **③ 上游仍无（保留；锚点已变 → 必须重写）** | S–M | **并行实现 + 1 处注入**（非整文件替换） | **model-gated** (`models/deepseek_v4/memory_policy.py:32`) |
| 10 | `torchft/checkpoint.py` | replaces `TorchFTCheckpointManager.__init__` | `Pending upstream PR: …/pull/4598` (L1) | **已吸收** `d771abb06` (#4598)：main `torchft/checkpoint.py:123` `optimizers.init_cache_state_dict()`、`:127` `optimizers._refresh_cached_state_dict()` | **① 已被上游吸收（可删）** | S（删 + 迁移） | 整方法替换（backport `__init__`） | **opt-in** (`extensions/.../torchft/__init__.py:20`) |
| 11 | `torchft/manager.py` | replaces `TorchFTManager.maybe_set_all_reduce_hook` | `Pending upstream PR: …/pull/4664` (L1) | **已吸收** `f5c97dc0e` (#4664)：main `torchft/manager.py:159-172` 与补丁**逐行相同** | **① 已被上游吸收（可删）** | S（删） | 整方法替换，与 PR 逐字一致 | **opt-in** |
| 12 | `torchft/optimizer.py` | replaces `TorchFTOptimizersContainer.__init__` / `.step` / `.zero_grad`; adds `_refresh_cached_state_dict` | `Pending upstream PR: …/pull/4578` + `…/pull/4598` (L1-2) | **已吸收** `59bbedfcd` (#4578) + `d771abb06` (#4598)：main `torchft/optimizer.py:46-51` (init_optim_state/cache)、`:55-60` `configure_fault_tolerance`、`:68-74` `_refresh_cached_state_dict`、`:84-94` `step`、`:96-100` `zero_grad`。**构造契约变了**：`ft_manager=` 关键字不再存在 | **② 上游已吸收但形态不同（按新形态重新实现）** | S–M | 整方法替换 ×3 + 新增方法 | **opt-in** |
| 13 | `torchft/trainer.py` | replaces `FaultTolerantTrainer.__init__` / `.train_step`; adds `._configure_chunked_loss` | `Pending upstream PR: …/pull/4534` + `…/pull/4575` + `…/pull/4663` (L1-3) | 三个 PR **全部已吸收**（`cd890fedc`/`91abd2301`/`277d4c930`），**但结构大改**：`FaultTolerantTrainer(Configurable)` + `FaultTolerantTrainingEngine(TrainingEngine)`；`_configure_chunked_loss` **已删除**（能力移入 `torchtitan/training_engine.py:403-404`）；`train_step` 重写为 `TrainingEngine` + `TrainingMicrobatch` 形态（main `torchft/trainer.py:330-455`），PP/override 在 `:158`，replica 平均在 `:412-413` | **② 上游已吸收但形态不同（按新形态重新实现）** | M | 大型手写 backport（494 行重建 `__init__`+`train_step`+新方法） | **opt-in** |

### 加载路径总览（成本评估）

`torchtitan_npu/patches/torchtitan/__init__.py:11-17` 在 `import torchtitan_npu` 时**无条件**导入 5 个 graph_trainer 补丁（模块导入即 `apply()`，但只做方法替换，代价低）：

```
chunked_loss, ep_chunk_concretization, ep_overlap_shape_queries,
ep_ready_nodes_dedup, ep_shape_live_out          # → 模块 1,2,4,5,6
```

其余为**按需加载**（实验/模型启用时才导入）——升级时更便宜、更容易漏：

| 模块 | 加载点 |
|------|--------|
| 3 `ep_forward_accumulation` | 仅由 `functionalize_recompute_mutations.py:95` 在 `"mutation-functionalization"` pipeline 构造时 `apply()` |
| 7 `functionalize_recompute_mutations` | `torchtitan_npu/models/deepseek_v4/config_registry.py:44`（注册 `"mutation-functionalization"`）、`extensions/.../graph_trainer/auto_overlap.py:30` |
| 8 `graph_trainer_runtime_context` | 仅当 `auto_overlap.is_npu_auto_overlap_pipeline(compile.pass_pipeline)` 为真时，由 `GraphTrainerEx.__init__` 调用 `apply()`（`extensions/torchtitan/experiments/graph_trainer/__init__.py:24-30`） |
| 9 `memory_policy` | `torchtitan_npu/models/deepseek_v4/memory_policy.py:32`（DSV4 模型模块导入时；`@register_memory_policy("dsv4-mhc")` 消费其 API） |
| 10-13 `torchft/*` | `torchtitan_npu/extensions/torchtitan/experiments/torchft/__init__.py:20`（仅 opt-in TorchFT 集成） |

---

### 1. `graph_trainer/chunked_loss.py` — ③ 上游仍无（保留，必须重写）

* **目标**: 包装 `torchtitan.experiments.graph_trainer.chunked_loss.ChunkedLossWrapperWithParamGrads.__call__` 与 `._gradient_backprop`（our `chunked_loss.py:27-29`, apply 在 `:155-163`），在 1-D SimpleFSDP（`Shard(0)`、fp32 reduce）下把 chunk 的 lm_head 梯度在 **reduce-scatter 之前合并**，用 `_LocalGradientLMHead` + `_ChunkGradsBackprop` 保留通信元数据。自述为临时实现，等上游 SimpleFSDP 支持后删除。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4734`（our `:1`）。
* **main 现状**: PR #4734 未合并（`git log --all --grep="4734"` 无匹配）。但**锚点已被重写**：
  * commit `f93fd4ccf`（**2026-09-09，v0.3.0 之后**）`support MTP with chunked loss (#4451)`：main `chunked_loss.py:15` 的 `_gradient_backprop` 变成**实例方法**，签名 `(self, hidden_states: tuple[torch.Tensor, ...], accumulated_grads: tuple[torch.Tensor, ...], total_loss)`（main `:31-37`），不再接收 `lm_head`/`fsdp_enabled`；`_ChunkedLossWrapperWithParamGrads.forward` 改为 `(ctx, num_predictions: int, *args: Any)`（main `:74`）。
  * main `simple_fsdp.py` 的 `ReplicateComputation`（`:274`）**已删除 `full_dtensor` 属性**（v0.3.0 有 7 处引用，main 0 处），新增 `grad_placements`；补丁的 `_can_coalesce()` 通过闭包内省读 `parametrization.full_dtensor`（our `:121`）→ 在 main 上直接失效或误判。
* **结论**: **③**——能力上游仍无，保留；但两处锚点（`_gradient_backprop` 签名、`ReplicateComputation.full_dtensor`）都已改变，**升级时必须重写**，不是简单改 import。
* **重写要点**: ① `_gradient_backprop` 改为实例方法包装，并处理 `*accumulated_grads` 元组（MTP 多 prediction）；② `_can_coalesce` 去掉 `full_dtensor` 依赖，改用 main 的 `param_sharding`/`grad_placements` 与 `mode` 判断；③ 若上游最终在 loss/model 层修（PR 描述里 reviewer 的要求），应直接把合并逻辑上移到 `ChunkedLossWrapper` 而不是继续 wrap 私有方法。
* **工作量**: **M**。**形态**: 精准 monkey-patch 注入（2 个方法，`apply()` 在 import 时执行）。

### 2. `graph_trainer/ep_chunk_concretization.py` — ④ 与上游方向冲突（重新设计）

* **目标**: 重写 EP chunk 的符号具体化：只从带 `torchtitan_chunk_{batch,seq}` shape ID 的维度收集 symbol hints，保留独立的 packed/varlen 符号，并对 copied unbacked symbol 使用 chunk-local range（our `ep_chunk_concretization.py:99-253`）。wrap 了 7 个下游符号 + `registry.TRACE_INPUT_PREPARERS["ep_overlap"]`。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4529`（our `:1`）。
* **main 现状**: PR #4529 **已合并**（`79008449f Fix EP chunk concretization by limiting hints to selected symbols (#4529)`，in HEAD，改动 `ep_pass_utils.py` +5 行），但随后 `43f73649f [graph_trainer] Remove graph-based EP chunking (#4952)`（2026-09-30）**整体删除 `torchtitan/experiments/graph_trainer/ep_chunk_pass.py`（-2455 行）**并瘦身 `ep_pass_utils.py`（-666 行）。现在：
  * `populate_chunk_dim_metadata_pass` / `mark_chunk_dynamic_dims` / `prepare_ep_overlap_trace_inputs` / `_fresh_unbacked_symbol_for_copy` / `_split_live_out_users` / `_materialize_live_out` **均不存在**（`grep -rn` 全树仅测试残留 `chunked_region_role`）。
  * `ep_pass_utils.py` 现在只有 `is_c10d_functional_node:32`、`ordered_nodes:41`、`ChunkOwner:47`、`ChunkBody:56`、`ChunkedRegion:67`、`collect_chunked_regions:119`；`tensor_meta` / `_record_symbols_from_extent` / `_concretize_value` **全部消失**。
  * 新路径是**eager chunking**：`ep_eager_chunk.py:409 maybe_apply_ep_overlap_eager_chunking()` + `:445 populate_eager_chunk_metadata_pass()`（把 eager traceback 的 `chunk_id` 提升为 pass metadata），不再有 unbacked symbol 具体化阶段。
* **结论**: **④**。补丁的宿主子系统已被上游删除，方向从「graph chunk + 符号具体化」改为「eager chunk + metadata 提升」。补丁的 `apply()` 里虽有 `required_helpers` 守卫（`our :109-119`），在 main 上只会打印 warning 后静默退出——保留无害但完全无效。
* **重写要点**: ① 先确认 PR #4529 修的**问题**（独立 DSV4 metadata 被 chunk shape 处理吞掉）在 eager 路径是否复现；若复现，应针对 `ep_eager_chunk._EagerChunkedForward`（`:136`）的 shape 契约重写；② 若不复现，连同 `ep_shape_live_out`/`ep_forward_accumulation` 一起删除，并改测 `ep_overlap_pass` + `ep_eager_chunk` 组合；③ 移出 `patches/torchtitan/__init__.py` 的无条件导入。
* **工作量**: **L**（先做复现判定，再决定是重写还是删除）。**形态**: 多点 monkey-patch 注入（7 个 wrap + registry 注入），非整文件 backport。

### 3. `graph_trainer/ep_forward_accumulation.py` — ④ 与上游方向冲突（重新设计）

* **目标**: 包装 `ep_chunk_pass._materialize_live_out`（our `ep_forward_accumulation.py:85-146`），为 functionalization 产生的「`updated = buffer + delta` + `copy_`」epilogue 提供 forward 累加证明，把跨 chunk 的 buffer 更新重建成 `(buffer + delta0) + delta1`。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4708`（our `:1`）。
* **main 现状**: PR #4708 未合并；宿主 `ep_chunk_pass._materialize_live_out` **不存在**（`grep -rn "_materialize_live_out" torchtitan/` 无结果，随 `#4952` 删除）。上游对 mutation 问题的处理走的是另一条路线：`bc86435b7 [1/2] … Save forward dependencies hidden by mutations (#4842)`（2026-10-02）新增 `mutation_utils.py`（`mutation_deps:195`、`is_mutation_node:30`、`base_tensor_for_mutation_target:146`），并只接入 **GraphPP** 分区器（`graph_pp/partition.py:95-98`、`fsdp_patterns.py:31,310`）；SPMD/`selective_activation_remat` 路径**未**包含任何 mutation 处理（`grep -n mutation selective_activation_remat.py` 为空）。
* **结论**: **④**。同一个补丁还依赖模块 7 的 functionalization 管线（`functionalize_recompute_mutations.py:95,99` 调用 `ep_forward_accumulation.apply()`），而 functionalization 本身也未上游；且其唯一锚点已随 graph EP chunking 一起消失。
* **重写要点**: ① 与模块 7 一起重新评估：若上游 #4842 的 [2/2] 覆盖 SPMD 路径，本补丁整体删除；② 若仍需保留，锚点要从 `ep_chunk_pass._materialize_live_out` 迁到 eager chunking 的 `_cat_chunked_outputs`（`ep_eager_chunk.py:44`）或 `ep_overlap_pass` 的 live-out 处理；③ 目前 `ep_forward_accumulation.apply()` 没有 `hasattr` 守卫，在 main 上 import 即 `AttributeError`（只在被模块 7 调用时才触发）。
* **工作量**: **L**。**形态**: 精准 monkey-patch（1 个函数，`@wraps` + 标记属性）。

### 4. `graph_trainer/ep_overlap_shape_queries.py` — ③ 上游仍无（保留，锚点未变）

* **目标**: 包装 `ep_overlap_pass._collect_token_exchanges`（our `ep_overlap_shape_queries.py:37-61`）：把 in-body token-exchange 的 `aten.sym_size.int` 查询上误继承的 `EP_token_exchange` 注解删掉，再交回上游做正常校验/等待归一化。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4650`（our `:1`）。
* **main 现状**: PR #4650 未合并。**锚点未变**：
  * `_collect_token_exchanges` 仍在 `torchtitan/experiments/graph_trainer/ep_overlap_pass.py:196`，函数体与 v0.3.0 **逐字相同**（无任何 sym_size/shape-query 处理，`grep -n "sym_size" ep_overlap_pass.py` 无结果）。
  * 依赖的 `ep_overlap_pass._custom_meta`（main `:111`）、`_is_token_exchange_launch`（main `:123`）、`common_utils._EP_TOKEN_EXCHANGE`（main `common_utils.py:211`）均存在且语义未变。
  * 注意 main 把 `ep_pass_utils._is_c10d_functional_node` 改名为 `is_c10d_functional_node`（无下划线），但本模块不使用该符号，不受影响。
* **结论**: **③ 上游仍无（保留）**，锚点仍在 mainline 且未变 → 升级成本最低的一个 graph_trainer 补丁。
* **重写要点**: 只需在升级后重跑 `tests/unit_tests/patches/...` 与 EP overlap 整网；建议把 `@wraps` 标记改成显式钩子以便上游合并后自动失效；顺带把「上游未合并」的说明更新为「锚点已验证于 `<commit>`」。
* **工作量**: **S**。**形态**: 精准 monkey-patch 注入（1 个函数 + 幂等标记）。

### 5. `graph_trainer/ep_ready_nodes_dedup.py` — ① 已被上游吸收（可删）

* **目标**: 替换 `ep_overlap_pass._ready_nodes`（our `ep_ready_nodes_dedup.py:30-70`），去掉同一节点在多个 chunk candidate 集合里重复入队造成的循环 phase-order 边，保留首次出现顺序。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4516`（our `:1`）。
* **main 现状**: **已合并**：`c4bef0bb7 Fix duplicate EP ready nodes by deduplicating shared candidates (#4516)`（2026-09-09，in HEAD）。commit 内容正是给 `_ready_nodes` 加 `selected: set[fx.Node]` + `if node in selected: continue` + `selected.add(node)`。main `ep_overlap_pass.py:514-541` 与补丁实现**逐行等价**（补丁的 `selected` 去重、`include_waits` 过滤、`_body_deps` 检查顺序完全一致）。
* **结论**: **①**。直接删除模块与 `patches/torchtitan/__init__.py:15` 的导入即可。注意若未来仍需替换 `_ready_nodes`，main 里该符号已被 `ep_pass_utils.is_c10d_functional_node`（无下划线）改名影响 `include_waits` 分支。
* **工作量**: **S**（删除 + 更新 `__init__.py` + 删除/改写对应 UT）。**形态**: 精准 monkey-patch（1 个函数整体替换）——已被上游原生化。

### 6. `graph_trainer/ep_shape_live_out.py` — ④ 与上游方向冲突（重新设计）

* **目标**: 包装 `ep_chunk_pass._split_live_out_users`（our `ep_shape_live_out.py:41-66`），把同 root 的 `sym_size.int(all_to_all, 0)` 用户改判为 chunk-local，避免 AllToAll 尺寸查询被当作 full-value live-out 物化。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4651`（our `:1`）。
* **main 现状**: PR #4651 未合并，且 `_split_live_out_users` **在 main 中不存在**——随 `43f73649f`（#4952）与整个 `ep_chunk_pass.py` 一起删除；`is_module_fqn_inside_root` / `_get_module_fqn` / `_set_synthetic_meta` / `_validate_materialized` 等依赖符号同样消失。上游现以 eager chunking（`ep_eager_chunk.py:409/445`）替代 graph chunk，不再有 full-value live-out 物化与校验阶段。
* **结论**: **④**。同模块 2、3：方向冲突，不能按原形态重写。补丁 import 时即 `AttributeError`（`our :23` 直接 `import ...ep_chunk_pass`），且它是 **auto-load**（`patches/torchtitan/__init__.py:16`），升级到 main 会**直接炸 import**。
* **重写要点**: ① **首要动作**：升级前把模块 2/3/6 从 `patches/torchtitan/__init__.py` 摘出或加 `importlib.util.find_spec` 守卫，否则 `import torchtitan_npu` 直接失败；② 判定 eager 路径是否出现同款「AllToAll 尺寸查询被跨 chunk 物化」问题（看 `ep_overlap_pass._build_region_phases` 的 live-out 分类，main `ep_overlap_pass.py` 仍有该阶段）；③ 若问题消失则删除，并在 EP overlap 文档中记录上游方向变更。
* **工作量**: **L**（含 import 守卫的紧急修复）。**形态**: 精准 monkey-patch（1 个函数 + 依赖 3 个私有符号）。

### 7. `graph_trainer/functionalize_recompute_mutations.py` — ③ 上游仍无（保留，必须重写）

* **目标**: 注册新 pass pipeline `"mutation-functionalization"`（our `:34`, `:81`），在 `tag_with_memory_policy_pass` 之后插入 `functionalize_recompute_mutations_pass`（our `:37-78`），把 forward 的原地写（`buffer.add_`）先 functionalize 成纯 dataflow，避免原生 SAR 重算时丢失 mutation 语义；同时按需装载 `ep_forward_accumulation`（our `:95-99`）。
* **上游标记**: 无 `Upstream-PR:` 头；docstring 写 `Backport of the mutation functionalization pass from upstream TorchTitan PR https://github.com/pytorch/torchtitan/pull/4708 (head a64792d2, fixing https://github.com/pytorch/torchtitan/issues/4688)`（our `:6-12`）。注意 `a64792d2` 在本 checkout 中**不可达**（`fatal: ambiguous argument`）。
* **main 现状**: **Pass 不存在**：`git log --all -S "functionalize_recompute_mutations_pass"` 无结果；`grep -rn "dispatch_functionalize\|FunctionalTensorMode\|propagate_input_mutations" torchtitan/` 无结果（唯一命中是 SAR 的 `has_recomputable_ops`，`selective_activation_remat.py:16,127`）。但**锚点已变、且上游另有路线**：
  * `passes.construct_default_graph_passes` 的 `parallel_dims=` 形参**已改名** `parallelism_context=`（main `passes.py:439-444`；重命名来自 `5dc97a3e7 rename ParallelDims to ParallelismContext (#4905)`）。补丁 ours `:100` 仍传 `parallel_dims=parallel_dims` → main 上 `TypeError`。
  * pipeline 调用约定也变了：main `spmd_graph_builder.py:188-193` 以 `pipeline_fn(traced, trainer_config, parallelism_context=parallelism_context)` 调用 `PASS_PIPELINE_REGISTRY`（`registry.py:42-45` 仍在，`register_pass_pipeline` 仍是有效扩展点）。
  * 同类问题的上游解法：`bc86435b7 [1/2] [graph_trainer] Save forward dependencies hidden by mutations (#4842)` 用 `mutation_utils.mutation_deps` 显式追踪 mutation 依赖，仅接入 GraphPP（`graph_pp/partition.py:95`、`fsdp_patterns.py:31`）；SPMD/SAR 路径尚无处理，标题 `[1/2]` 提示还有后半段未合并。
* **结论**: **③ 上游仍无（保留）**。锚点变化 → **升级时必须重写**；同时把 #4842 的 [2/2] 列为跟踪项（可能直接取代本补丁）。
* **重写要点**: ① 把 `parallel_dims=` 改为 `parallelism_context=`，并让 pipeline 签名接受 `parallelism_context` kwarg（对齐 `spmd_graph_builder.py:190`）；② `anchor = names.index("tag_with_memory_policy_pass")` 仍有效（main `memory_policy.py:641` 定义、`passes.py:83` 导入），但需确认它在 pipeline 里的实际位置；③ `from torchtitan.experiments.graph_trainer.passes import _get_pass_name, construct_default_graph_passes` 与 `_is_backward_node`（`common_utils`）在 main 中仍在，导入可保留；④ 与模块 3 一起决定是否继续需要 `ep_forward_accumulation`。
* **工作量**: **M**。**形态**: backport 新上游文件（一个完整 pass + 一个 pipeline 注册，111 行）——不是 monkey-patch 注入，而是通过公开扩展点挂载。

### 8. `graph_trainer/graph_trainer_runtime_context.py` — ③ 上游仍无（保留，必须重写）

* **目标**: monkey-patch `passes.apply_graph_passes`（our `:67-102`）与 `GraphTrainer._make_fx_forward_backward_step`（our `:105-147`），用 `ContextVar`（our `:45-48`）把 `traced_result`/`module`/`args`/`train_context` 传给显式声明 `_requires_runtime_context` 的 NPU pass（whole-graph benchmarking 用）。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4763`（our `:1`）。
* **main 现状**: PR #4763 未合并：
  * `apply_graph_passes` 仍在 main `passes.py:506`，签名 `(gm, example_inputs, passes, *, compile_config=None, respect_disable_passes=True)`——**没有** `runtime_context`。
  * `construct_default_graph_passes` 仍无 `runtime_context` 形参（main `passes.py:439`）。
  * `GraphTrainer._make_fx_forward_backward_step` **已被删除**：main `graph_trainer/trainer.py` 现在是 `GraphTrainingEngine(TrainingEngine)`（`:41`）+ `GraphTrainer(Trainer)`（`:332`），trace 入口移到 `spmd_graph_builder.py`（`_trace_joint_stage_graph:210`、`_trace_spmd_stage_graph:289`）与 `graph_builder_utils.py`（`apply_graph_passes` 调用在 `:153`、`:240`，**不传** runtime context）。
  * 后果：补丁的 `apply()` 会走到 `:151` 的判断（`_UPSTREAM_PIPELINE_ACCEPTS_RUNTIME_CONTEXT` 为 False）→ `original_make_fx_forward_backward_step is None` → **打印 warning 后静默 return**（our `:153-159`）；`torchtitan.experiments.graph_trainer.trainer.apply_graph_passes = ...`（our `:167`）在 main 只是新建一个未使用的模块属性（`grep -n apply_graph_passes trainer.py` 无结果）。
* **结论**: **③ 上游仍无（保留）**。锚点变化 → **升级时必须重写**，否则 `GraphTrainerEx` 的 auto-overlap 校准会静默失去 runtime context（不报错，最难排查的一类）。
* **重写要点**: ① 注入点从 `GraphTrainer._make_fx_forward_backward_step` 迁到 `spmd_graph_builder` 的 trace 函数（`:210`/`:289`）或 `graph_builder_utils` 的 `apply_graph_passes` 调用点（`:153`/`:240`）；② `ContextVar` 工厂改为由 `GraphTrainingEngine._forward_backward_body`（main `trainer.py:245`）设置；③ 保留 `_UPSTREAM_PIPELINE_ACCEPTS_RUNTIME_CONTEXT` 探测，把探测目标改成 `spmd_graph_builder` 的 pipeline 构造；④ 给 `whole_graph_benchmark.make_calibration_runner` 的 legacy-dict 分支加断言，避免再次静默降级。
* **工作量**: **M**。**形态**: monkey-patch 注入（模块属性 + 类方法 + ContextVar），非 backport。

### 9. `graph_trainer/memory_policy.py` — ③ 上游仍无（保留，必须重写）

* **目标**: 提供 NPU 侧 memory-policy API：`tag_sac_policy(..., save_input_every_n_layers=...)`（our `:122-160`）、`NodePolicyKey`（`:34-38`）、`make_node_override_memory_policy`（`:173-195`），并在 import 时把 `GraphTrainerCompileConfig.memory_policy` 注解改成 `str` 以放行自定义策略名（`:198-208`）。被 DSV4 的 `@register_memory_policy("dsv4-mhc")` 消费（`torchtitan_npu/models/deepseek_v4/memory_policy.py:161,185,198,229`）。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3949`（our `:1`）。
* **main 现状**: PR #3949 未合并；`save_input_every_n_layers` / `NodePolicyKey` / `make_node_override_memory_policy` / `_force_boundary_saves` 在 **所有 ref** 中都查不到（`git log --all -S` 无结果）。锚点仍在但**主体已改写**：
  * `tag_sac_policy` 仍在 main `memory_policy.py:282`，签名保留 `policy_fn` + `force_save_nodes`，**没有** `save_input_every_n_layers`。
  * 上游边界规则与我们的不同：main 的 Pass 2 用 `_get_layer_id(user) != node_layer_id`（main `memory_policy.py:~385-405`）；我们的 `_force_boundary_saves`（our `:41-68`）用 `(produce_layer_id + 1) % save_input_every_n_layers == 0` 且只在 user 层号更大时保存。
  * main 新增了我们的副本**没有**的处理：`_is_cacheable_effect` → MUST_SAVE（来自 `9486f94d3 [graph_trainer] Preserve cacheable effects during rematerialization (#4876)`）。继续用我们的副本会静默丢掉该修复。
  * CLI hack 已过时/有害：main `configs.py:201` 仍是 `memory_policy: Literal[...]`，但 Tyro 已被 `eeeb39b42 Replace Tyro CLI with Python config loading (#4908)` 移除（`grep -rn tyro torchtitan/` 无结果），把注解改成 `str` 只会削弱配置校验。
  * `_make_default_memory_policy`（main `:123`，新增 `save_ops` 形参）与 `matches_module_fqn_pattern`、`_MODULE_FQN`、`_get_layer_id`、`_is_backward_node`、`_NOT_IN_LAYERS`（`common_utils.py`）均仍在——补丁的依赖面完好。
* **结论**: **③ 上游仍无（保留）**。`save_input_every_n_layers` 语义上游确实没有；但锚点实现已变化 → 升级时必须重写（属"锚点变化"情形）。
* **重写要点**: ① 不要再 fork 整份 `tag_sac_policy`：改为**薄封装**上游 `memory_policy.tag_sac_policy(...)`，先调用上游（拿到 cacheable-effect + 新边界规则），再仅对 `save_input_every_n_layers > 1` 施加额外的 `_force_boundary_saves`；② 删除 `_patch_graph_trainer_memory_policy_cli_choices`（`:198-208`），改为在 `register_memory_policy` 时校验名字，或确认 Python config 加载路径不需要注解 hack；③ 确认 DSV4 的 `_DSV4_MHC_SAVE_OVERRIDES`（`NodePolicyKey` 用法）在新边界规则下数值仍一致（`tests/unit_tests/models/deepseek_v4/memory_policy/` 已有覆盖）。
* **工作量**: **S–M**。**形态**: **并行实现**（非整文件替换：仅 `_patch_graph_trainer_memory_policy_cli_choices()` 一处注入 + 一套独立 API），比 monkey-patch 更容易重写。

### 10. `torchft/checkpoint.py` — ① 已被上游吸收（可删）

* **目标**: 整方法替换 `torchtitan.experiments.torchft.checkpoint.TorchFTCheckpointManager.__init__`（our `checkpoint.py:32-105`），在 FT 导出恢复状态前调用 `optimizers._refresh_cached_state_dict()`（our `:77`），并补 FT dataloader checkpoint 所需的 staging / gloo PG。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4598`（our `:1`）。
* **main 现状**: **已合并** `d771abb06 Fix restored optimizer learning rates in TorchFT (#4598)`（in HEAD）。main `torchtitan/experiments/torchft/checkpoint.py`：
  * `:123` `optimizers.init_cache_state_dict()`
  * `:127` `optimizers._refresh_cached_state_dict()`（补丁的核心修复）
  * 另有上游演进：`:130` 的 state 集合加入 `EMA`，构造签名新增 `ema: EMAContainer | None`（main `:89`），FT dataloader 逻辑改用 `self._is_checkpoint_owner()`。
* **结论**: **① 可删**。补丁是**整方法替换**，且目标方法签名已变（多出 `ema`），保留会**覆盖掉上游演进**（丢 EMA、丢 `_is_checkpoint_owner` 重构）→ 必须删除而不是保留。
* **重写要点**: 删除 `patches/.../torchft/checkpoint.py`；把 `TorchFTCheckpointManager.Config` 的构造调用点（`torchtitan_npu/extensions/torchtitan/experiments/torchft/*`）对齐 main 的 `ema=` 形参。
* **工作量**: **S**。**形态**: 整方法替换（backport `__init__`），非精准注入。

### 11. `torchft/manager.py` — ① 已被上游吸收（可删）

* **目标**: 整方法替换 `TorchFTManager.maybe_set_all_reduce_hook`（our `manager.py:20-37`），让 FSDP+EP 下每个 FSDP 参数组都装上 replica 平均 hook（多参数组场景）。
* **上游标记**: `# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4664`（our `:1`）。
* **main 现状**: **已合并** `f5c97dc0e Fix TorchFT hooks for multi-group FSDP modules (#4664)`（in HEAD）。main `torchtitan/experiments/torchft/manager.py:159-172` 与补丁体**逐行相同**（`_get_fsdp_state()._fsdp_param_groups` 遍历 + `param_group._all_reduce_hook = all_reduce_hook`）。main 另有 `loss_sync_pg` property（`:174-179`）。
* **结论**: **① 可删**（最干净的一个：逐字被吸收）。
* **重写要点**: 删除模块与 `patches/.../torchft/__init__.py:8` 的导入。
* **工作量**: **S**。**形态**: 整方法替换，与上游 PR 逐字一致。

### 12. `torchft/optimizer.py` — ② 上游已吸收但形态不同

* **目标**: 整方法替换 `TorchFTOptimizersContainer.__init__`（改用 `init_optim_state` 预物化状态，绕开 FT quorum 路径，our `optimizer.py:30-43`）、`.step`（非递归派发，`:46-53`）、`.zero_grad`（`:56-59`），并新增 `_refresh_cached_state_dict`（`:62-67`）。
* **上游标记**（两行）:
  ```
  # Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4578
  # Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4598
  ```
  （our `:1-2`）
* **main 现状**: **两个 PR 都已合并**（`59bbedfcd Fix recursive TorchFT optimizer dispatch (#4578)`、`d771abb06 … (#4598)`），但**装配方式变了**：
  * main `torchft/optimizer.py:35-51` `__init__(self, config, *, model_parts, enable_cuda_graph=False)`——**没有** `ft_manager=` 关键字；内部 `init_optim_state(optim)`（`:46-48`）、`cache_state_dict = {}`（`:50`）、`_quorum_manager = None`（`:51`）。
  * 新增独立入口 `configure_fault_tolerance(self, ft_manager)`（main `:55-60`）负责 quorum 接线，由 engine 调用（main `torchft/trainer.py:117,119`）。
  * `_refresh_cached_state_dict`（main `:68-74`）与补丁逐行一致；`step`（`:84-94`）与 `zero_grad`（`:96-100`）语义一致（`super().zero_grad` 替代 `OptimizersContainer.zero_grad(self, …)`）。
  * main 还新增了 `TorchFTOptim`（`:102`，非有限步投票，`714579846`）。
* **结论**: **②**。能力已被上游吸收，但补丁的**替换形态已失效**：`patched_init(..., ft_manager=...)` 在 main 上会 `TypeError`。正确做法是删除该 3 处方法替换，并在 build 调用点改用 `configure_fault_tolerance(ft_manager)`。
* **重写要点**: ① 删除 `__init__`/`step`/`zero_grad` 替换，只保留（如需）`_refresh_cached_state_dict` 的断言；② 更新 `torchtitan_npu/extensions/torchtitan/experiments/torchft/optimizer.py:19-23`（`TorchFTOptimizersContainerEx`）与 `trainer.py:29-30` 的 Config/build 路径，把 `ft_manager=` 改成显式 `configure_fault_tolerance()` 调用；③ 核对 `enable_cuda_graph=` 新形参在 NPU 上是否需要传 False。
* **工作量**: **S–M**。**形态**: 整方法替换 ×3 + 新增方法（backport），非精准注入。

### 13. `torchft/trainer.py` — ② 上游已吸收但形态不同

* **目标**: 重建 `FaultTolerantTrainer.__init__`（our `trainer.py:39-345`，把 `apply_overrides` 提前到模型构建之前、加 `ft_manager.get_dp_info`、`maybe_set_all_reduce_hook`、`ft_manager=` 传给 optimizer/checkpointer）、新增 `_configure_chunked_loss`（`:346-364`）、重建 `train_step`（`:367-486`，含跨 grad-accumulation 的 loss 累加与 `ft_pg` replica 平均）。
* **上游标记**（三行）:
  ```
  # Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4534
  # Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4575
  # Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4663
  ```
  （our `:1-3`）
* **main 现状**: **三个 PR 全部已合并**，但宿主类被重构：
  * `cd890fedc Fix TorchFT config overrides before model construction (#4534)` → main `torchft/trainer.py:158` `apply_overrides(config.override, config)`（模型构建前）。
  * `91abd2301 Fix chunked loss initialization in TorchFT trainer (#4575)` → 该能力**已从 torchft 移出**：`_configure_chunked_loss` 在 main 的 `torchft/trainer.py` 与 `torchtitan/trainer.py` 中都**不存在**（`git log -S _configure_chunked_loss` 最后一条是 `ec953b360 [RL] Share the core training engine (#4773)`，即该 commit 删除了它）；现由共享引擎完成：`torchtitan/training_engine.py:403-404` `self.loss_fn.set_lm_head(lm_head)` + `model_with_lm_head._skip_lm_head = True`。
  * `277d4c930 Fix TorchFT loss averaging across replicas (#4663)` → main `torchft/trainer.py:412-413` `ft_pg = engine.ft_manager.loss_sync_pg` + `global_avg_loss /= ft_pg.size()`。
  * **结构变化**：main `torchft/trainer.py:44 FaultTolerantTrainingEngine(TrainingEngine)`、`:143 class FaultTolerantTrainer(Configurable)`（不再是 `Trainer` 子类）、`:152 @record __init__`、`:330 train_step`（基于 `TrainingMicrobatch` / `engine.forward_backward` / `engine.optim_step`，含 `routing_token_counts`、EMA、aux-loss metrics，与补丁的 `train_step` 形态完全不同）。
* **结论**: **②**。能力全在上游，但补丁重建的 `__init__`/`train_step` 已与 main 的 engine 架构不兼容（`_configure_chunked_loss` 甚至是死属性）。删除补丁后，NPU 侧若仍需定制（`FaultTolerantTrainerEx` 在 `torchtitan_npu/extensions/torchtitan/experiments/torchft/trainer.py:26` 同时继承 `TrainerEx` 与 `FaultTolerantTrainer`），必须**按新形态**改挂 `FaultTolerantTrainingEngine`。
* **重写要点**: ① 删除全部三个整方法替换；② `_configure_chunked_loss` 不要重加（main 在 `training_engine.py:403` 已做，重复设置 `_skip_lm_head` 会与 PP 分支冲突）；③ `FaultTolerantTrainerEx` 从「Trainer 多继承」改为「`Configurable` + `engine: FaultTolerantTrainingEngine` 子类」，NPU 定制下沉到 engine 的 `_initialize_distributed_runtime`/`_initialize_model`/`_initialize_optim`/`_initialize_checkpointer`（main `:67/:102/:114/:121`）；④ `FTManagerEx`/`ManagerEx`（`torchtitan_npu/extensions/.../torchft/ft_manager.py:59`、`manager.py`）的 Config 注入点需同步改造。
* **工作量**: **M**（补丁删除本身 S，NPU 集成重挂 M）。**形态**: 大型手写 backport（494 行重建两个方法 + 新增一个方法），不是精准注入。

---

### 建议动作顺序

1. **紧急（防 import 崩溃）**：模块 2/6 是 auto-load 且 import 已删除的 `ep_chunk_pass`；升级到 main 前必须从 `patches/torchtitan/__init__.py:11-17` 摘除 2/3/6，或加 `find_spec` 守卫。
2. **零风险删除**：模块 5（#4516 逐行被吸收）、11（#4664 逐字被吸收）、10（#4598 已吸收，且保留反而覆盖上游 EMA/checkpoint-owner 演进）。
3. **删除 + 调用点迁移**：模块 12、13（能力都在 main，改挂 `configure_fault_tolerance` / `FaultTolerantTrainingEngine`）。
4. **保留但必须重写**：模块 1（M，两处锚点已变）、4（S，锚点未变，最便宜）、7（M，`parallel_dims`→`parallelism_context`，跟踪 #4842 [2/2]）、8（M，注入点迁到 `spmd_graph_builder`）、9（S–M，改为薄封装上游 `tag_sac_policy`，删掉 CLI hack）。
5. **方向冲突需重新设计**：模块 2、3、6——先判定问题在 eager chunking 路径是否复现，再决定重写（L）或整体删除。

<!-- section g4 -->

## 补丁审计 G4：flex_shard（4）+ hf_datasets（2）

审计对象：`/home/lrwei/cann/torchtitan-npu`（pin = torchtitan **0.3.0**）
对照：`/home/lrwei/pytorch/torchtitan`（origin/main，HEAD = `3386c6d6f`，2026-10-08）、`/home/lrwei/pytorch/torchtitan-v030` + 已安装副本。
只读审计，未修改任何仓库文件。

### 0. 基线与方法（先说结论性的两个事实）

1. **pin 的 flex_shard 就是上游 `858b9e248`（PR #4078）**，逐字节相同（sha256 全等）：

   ```
   $ git -C /home/lrwei/pytorch/torchtitan show 858b9e248:torchtitan/distributed/flex_shard/<f> | sha256sum
     vs sha256sum /home/lrwei/pytorch/torchtitan-v030/torchtitan/distributed/flex_shard/<f>
     dist_muon.py / optimizer_reshard.py / _optimizer_reshard_schedule.py
     / _optimizer_reshard_runtime.py / __init__.py / README.md   → 全部 IDENTICAL
   $ git log -1 --format='%h %ad %s' 858b9e248
     858b9e248 Tue Aug 18 17:31:15 2026 -0700 [DistMuon] Support oversharded QKV heads (#4078)
   ```
   （整棵树并不完全等于 `858b9e248`，有 33 个文件不同；**flex_shard 目录是逐字节相同**，这是本次 backport 的真实基线。）
   即：flex_shard 组三个 patch 是「把 PR #4122（`6ee6b10b`，2026-08-20）从 `858b9e248`→`6ee6b10b` 的增量，用 monkeypatch 方式补到 0.3.0 上」。

2. **分类口径**（因为 4 个类别里 ①②的「pin」有两读，这里显式声明）：
   - ① = 上游 main 已含该实现的**等价内容**，升级 pin 后本补丁成为 no-op → 删除；
   - ② = 上游 main 已含，但同一文件/同名函数已被后续提交**改写** → 需要按新 pin 重新同步（多数情况是「删除 + 迁移我们自己的旁路改动」）；
   - ③ = 上游仍无此能力且 patch 锚点仍在 → 保留；
   - ④ = 上游仍无此能力**或**已改方向，且 patch 锚点已消失 → 必须重新设计。

---

### 1. 摘要表

| # | 模块（行数） | 目标：替换/回填的上游符号 | 上游标记（文件头） | 上游 main（HEAD 3386c6d6f）现状 | 复制性质 | 结论 | 风险 |
|---|---|---|---|---|---|---|---|
| 1 | `flex_shard/dist_muon.py` (581) | monkeypatch `torchtitan.distributed.flex_shard.dist_muon` 的 7 个函数 + 2 个类型别名；并覆盖 `DistMuon._validate_parameter_storage`（NPU 准入） | `PR #4122` / `6ee6b10b934730df00cbcd0cab5c93d04ae2e38a`；`NPU admission issue #4405` | 同名文件、同路径仍在；PR #4122 已在 main；但 7 个函数中 **4 个已被后续 4 个 PR 改写**；NPU 准入本身已被 `c215a12ee`(#4530) 上游修掉 | 6ee6b10b 的**逐函数逐字副本**（AST 等价）+ 1 个本地 NPU 变体 | **②** | 删除 **S**；原样保留到新 pin **H**（会静默回退 BlockShard v2） |
| 2 | `flex_shard/optimizer_reshard.py` (210) | 回填 `optimizer_reshard.ComputeLayout`（新增 `shard_order_by_tensor_dim` + `_FrozenConfigMapping`），并重绑到 `optimizer_reshard` / `flex_shard` / `dist_muon` 三处 | 同上（PR #4122 / 6ee6b10b） | 同路径仍在；HEAD 的 `ComputeLayout` **代码与 6ee6b10b 完全一致**，只有 docstring 更新（`block_sizes`、`edp_shard`） | 逐字副本（6ee6b10b） | **①** | 删除 **S** |
| 3 | `flex_shard/_optimizer_reshard_schedule.py` (321) | 回填 `_RedistributionGroup`、`_tensor_region_intersection(_numel)`、`_build_dim0_shard_redistribution_plan`、`_redistribution_group`、`_dtensor_storage_regions`，并注入 `dist_muon` | 同上（PR #4122 / 6ee6b10b） | 同路径仍在；HEAD 上这 6 个定义的 **AST 与本地等价**（仅 `_dtensor_storage_regions` 合并了两个同体 elif，语义相同） | 逐字副本（6ee6b10b） | **①** | 删除 **S** |
| 4 | `flex_shard/replica_dedup.py` (520) | 包裹上游工厂 `build_dist_muon`（并在 3 个模块里重绑）；替换 `DistMuon.step`；DCP state-dict/layout 上下文 | `TODO: Delete this patch once .../pull/4411 ([DistMuon] Deduplicate HSDP replica compute) is merged upstream` | 能力**仍无**（PR #4411 未合并：`git grep -i "replica.*dedup" HEAD` 空、`git log --all --grep=4411` 空）；但**锚点已消失**：`build_dist_muon` 被 `17067f00b`(#4904) 删除，`components/optimizer/` 被移到 `components/optim/` | 本地自研实现（无上游对应物） | **④** | 重新设计 **M**（内部私有钩子 `_initialize_plan`/`_apply_update` 等在 HEAD 仍在） |
| 5 | `hf_datasets/text_datasets.py` (201) | 不是替换上游文件，而是**子类化** pin 的 `HuggingFaceTextDataset`/`HuggingFaceTextDataLoader`，新增 `pad_segments_to_multiple` / `AlignedTextDataset` / `AlignedHuggingfaceDataloader` | `Mirrors upstream torchtitan/components/data/packing.py (_pad_segments_to_multiple) and .../types.py (DatasetBuildContext.pad_segments_to_multiple), which the 0.3.0 release predates` | 能力**仍无**（`git grep pad_segments_to_multiple HEAD` 全树空；HEAD 的 `DatasetBuildContext` 无该字段）；**锚点全消失**：`HuggingFaceTextDataset/Loader` 被 `1b04fc1c3`(#4088 Grain) 删除，`torchtitan/components/dataloader.py` 同commit 删除；header 声称的「上游镜像」只存在于 fork 分支 `br_dpsk_v4_1`（`9cb8f2274`，非 origin/main 祖先） | 本地自研（子类，非拷贝） | **④** | 重新设计 **M**（能力只需小函数，但要迁到 grain 数据栈） |
| 6 | `hf_datasets/multimodal/mm_collator.py` (151) | 替换 `MultiModalCollator._build_mrope_positions`（连续视频 MRoPE，不把 video grid 摊成逐帧） | `Upstream PR: pending publication of the Qwen3.5 contiguous video-MRoPE fix.` / `Upstream base: b175497ea3502388bb9513391810a86ca46dbfa8` | 修复**已在上游**：`539aa6cdf`（#4239，2026-08-20）是 origin/main 祖先；HEAD 的 `mm_collator.py:188` 保留原始 `[T,H,W]` 且无 `repeat_interleave` —— 但契约不同（HEAD 是 1-D `tokens`、`resets = positions[:,1:]==0`、返回 squeeze(0)） | 本地变体（取自 pin 的函数体，删掉视频展开） | **②** | 删除 **S**；若原样重套 **M**（`apply()` 契约校验只查形参名，会静默装上 2-D 变体） |

---

### 2. 逐模块详细

#### 2.1 `torchtitan_npu/patches/torchtitan/distributed/flex_shard/dist_muon.py`

**目标 / 能力**
- 回填 PR #4122「[DistMuon] support storage sharding of EP Shard(0) EFSDP Shard(1)」：`shard_order_by_tensor_dim` 的 `_StridedShard` 降级、dim0 shard 重分布计划、正交 dim0 shard 重分布、存储→计算转换校验。
- 额外本地能力：**NPU 准入**——`apply()` 把 `DistMuon._validate_parameter_storage` 换成要求 `local_device.type == "npu"` 的版本（`dist_muon.py:550-562, 578`）。
- 机制：`_rebind()`（`:536-547`）+ `apply()`（`:565-578`）把本地自由函数塞进 `torchtitan.distributed.flex_shard.dist_muon.__dict__`，并设置 `_LoweredComputeSharding` / `_AxisComputeSharding`：
  ```python
  dist_muon._LoweredComputeSharding = Owned | Replicate | Shard | _StridedShard | BlockShard
  dist_muon._AxisComputeSharding  = Replicate | Shard | _StridedShard | BlockShard
  ```
- 我们仓库的消费点：
  - 生效入口：`torchtitan_npu/patches/torchtitan/__init__.py:10` → `patches/torchtitan/distributed/flex_shard/__init__.py:3-8`（导入即 `apply()`）；
  - 使用被 patch 出的行为：`torchtitan_npu/models/deepseek_v4/config_registry.py:19,109-142`（`BlockShard` + shard order）、`torchtitan_npu/models/deepseek_v4_1/config_registry.py:18,173-177`；
  - 结构断言测试：`tests/unit_tests/test_dsv4_muon.py:33,136-152`；调用内部函数：`tests/unit_tests/models/deepseek_v4/test_lora_training.py:100`。

**上游标记（逐字）**
```
## Backport of upstream TorchTitan PR #4122:
## https://github.com/pytorch/torchtitan/pull/4122
## Upstream commit: 6ee6b10b934730df00cbcd0cab5c93d04ae2e38a
## NPU admission issue: https://github.com/pytorch/torchtitan/issues/4405
## Remove this patch when the pinned TorchTitan dependency includes PR #4122.
```

**上游 main 现状（content 比对）**
- 路径未变：`torchtitan/distributed/flex_shard/dist_muon.py` 在 HEAD 仍在（1957 行；6ee6b10b 为 1960 行）。
- 6ee6b10b 之后触碰 flex_shard 的全部提交（`git log --oneline --reverse 6ee6b10b..HEAD -- torchtitan/distributed/flex_shard/`）：
  `10cc9ffbb`(#4250，仅 `_optimizer_reshard_runtime.py`)、`c215a12ee`(#4530，`dist_muon.py` 4 行)、`da2f82670`(#4587)、`9c5302543`(#4750，仅 runtime)、`63c4e9fef`(#4808)、`c104b2947`(#4811)、`892ecfed0`(#4840)、`7031b7f1b`(#4871)、`17067f00b`(#4904)、`5dc97a3e7`(#4905，`optimizer_reshard.py`)、`b09a88b9b`(#4913，仅目录重组)。
- 逐定义比对（AST 归一化，脚本见 §4）——**本 patch 携带的 7 个函数里，4 个已被改写**：

  | 本 patch 的函数 | @6ee6b10b | @HEAD | 上游改写提交 |
  |---|---|---|---|
  | `_build_parameter_redistribution_plan` | 等价 | **改写** | `892ecfed0`：`assert type(...) is Shard` → 支持 `BlockShard`，`_build_batched_matrix_redistribution_plan(..., shard_participants=…, block_shard=…)` |
  | `_is_supported_orthogonal_dim0_shard_redistribution` | 等价 | **改写** | `892ecfed0`/`7031b7f1b`：`compute_view` → `block_shard`；`ndim != 3` → `ndim < 3`；`storage_dim == 1` → `== param.ndim - 2` |
  | `_resolve_storage_to_compute_transition` | 等价¹ | **改写** | `63c4e9fef`/`c104b2947`/`892ecfed0`/`7031b7f1b`：签名去掉 `global_compute_shape`/`compute_view`，改 `block_shard`+`block_sizes`，删掉「cannot yet change tensor sharding」分支 |
  | `_normalize_compute_placement` | 等价 | **改写** | `892ecfed0`：`BlockShard(dim, placement.block_size)` → `placement.block_sizes` |
  | `_lower_shard_order_to_strided_shards` | 等价 | 等价 | — |
  | `_validate_shard_order_compute_targets` | 等价 | 等价 | — |
  | `_normalize_storage_placement` | 等价 | 等价 | — |

  ¹ 本地版与 6ee6b10b 只差「把两个同体 `elif` 合并成 `if type(placement) is Shard or type(placement) is _StridedShard`」，语义相同（`ast.unparse` diff 只有这两处）。
- NPU 准入：本地 `_validate_parameter_storage` 是上游同名方法的**改写版**（要求 `type == "npu"`）；上游 HEAD 的版本已被 `c215a12ee`（#4530，修 #4405，2026-09-09）改为
  `if len(local_devices) != 1: raise ValueError("DistMuon requires one device per process")` —— **上游已不再限制 CUDA，NPU 天然可用且更宽**。
- 反面确认：PR #4122 中本 patch **未携带**的部分是 `_build_batched_matrix_redistribution_plan`、`_matrix_batch_view_from_compute_layout`、`_build_owned_redistribution_plan`、`_build_whole_tensor_redistribution_plan`，逐一比对后**只改了类型别名/注释（annotation-only）**，无运行时差异 → 这个 backport 在 pin 上是**行为完整**的。

**复制性质**：**逐字副本（6ee6b10b 的函数体）+ 本地 NPU 变体**。所以「实现方式」是 re-copy + 手写 monkeypatch 断言，会随着新 pin 的改写而全部失配。

**结论分类：②（上游已含但内容已更新）**
- 升级 pin 后该 patch **应整体删除**；绝不可原样保留：`apply()` 无任何版本/实现校验，`_rebind`+`setattr` 会把 HEAD 的 4 个新实现**静默覆盖**成 6ee6b10b 版本，回退 `BlockShard(block_sizes)` 变长块与 #4811/#4871 的覆盖修复 → 运行期报错或静默算错。
- 同步要点：
  1. 删除本 patch；两个类型别名锚点在 HEAD 仍在且**定义与 patch 注入的完全相同**（`dist_muon.py:885 _LoweredComputeSharding = Owned | Replicate | Shard | _StridedShard | BlockShard`、`:889 _AxisComputeSharding = Replicate | Shard | _StridedShard | BlockShard`），即别名部分在新 pin 上已自动成为 no-op；需要重新处理的是那 7 个函数。
  2. NPU 准入 delta 已被 `c215a12ee` 上游吸收；若仍要「只允许 npu」，保留一个只替代 `DistMuon._validate_parameter_storage` 的极小 patch（并同步 `tests/unit_tests/test_dsv4_muon.py:33-42` 的断言）。
  3. 本 repo 自身的调用点用的是**旧 BlockShard API**：`models/deepseek_v4/config_registry.py:110,114,142`、`models/deepseek_v4_1/config_registry.py:173,177` 仍写 `BlockShard(dim=0, block_size=...)`，而 HEAD 的参数名是 `block_sizes: tuple[int, ...]`（`892ecfed0`）→ pin 升级时必须一起迁移（`block_size=X` → `block_sizes=(X,)`），否则 TypeError。
- 冲突风险：**删除 S；误留 H**。

#### 2.2 `.../flex_shard/optimizer_reshard.py`

**目标 / 能力**：回填 PR #4122 版 `ComputeLayout`（`shardings_by_mesh_axis` 冻结为 `_FrozenConfigMapping`、新增 `shard_order_by_tensor_dim` 与 `_DEFAULT_SHARD_ORDER`），`apply()` 把同一对象绑到三处（`:204-207`）：
```python
optimizer_reshard.ComputeLayout = ComputeLayout
flex_shard.ComputeLayout = ComputeLayout
dist_muon.ComputeLayout = ComputeLayout
```
（`flex_shard/__init__.py` 与 `dist_muon` 在 patch 生效前已按值导入 `ComputeLayout`，所以必须三处重绑。）

**上游标记**：同 §2.1 的 PR #4122 / 6ee6b10b 四行头。

**上游 main 现状**
```
$ git log --oneline -5 -- torchtitan/distributed/flex_shard/optimizer_reshard.py
5dc97a3e7 rename ParallelDims to ParallelismContext (#4905)
892ecfed0 [DistMuon] variable-sized BlockShard to shard inside a q/kv head (#4840)
6ee6b10b9 [DistMuon] support storage sharding of EP Shard(0) EFSDP Shard(1)  (#4122)
```
- 逐定义 AST 比对：本地 `ComputeLayout` 与 `6ee6b10b` **完全一致**；与 HEAD 的差异**只在 docstring**（`BlockShard(dim=0, block_sizes=(4,))`、示例轴名 `edp_shard`，见 `892ecfed0`/`5dc97a3e7`），代码体一致。
- 消费点：`torchtitan_npu/models/common/muon.py:11,15`、`models/deepseek_v4/config_registry.py:19-23,78,96,109-142`、`models/deepseek_v4_1/config_registry.py:18-22`、`tests/unit_tests/extensions/cpu_offload/test_no_optimizer_state.py:263`。

**复制性质**：逐字副本（6ee6b10b）+ monkeypatch 绑定。

**结论分类：①（上游已含且内容一致 → 删除）**。新 pin 上此 patch 是 no-op，唯一副作用是把 HEAD 的 docstring 换回旧的（提到 `block_size`）。风险 **S**。

#### 2.3 `.../flex_shard/_optimizer_reshard_schedule.py`

**目标 / 能力**：回填 PR #4122 排程层——`_RedistributionGroup` 增加 `mesh_axis_participants`、新增 `_tensor_region_intersection` 与 `_build_dim0_shard_redistribution_plan`（取代 pin 的 `_build_replicated_to_dim0_shard_plan`）、重写 `_redistribution_group`/`_dtensor_storage_regions`。`apply()`（`:305-318`）除重绑 schedule 模块外，还**显式向 `dist_muon` 注入三个符号**：
```python
dist_muon._RedistributionGroup = _RedistributionGroup
dist_muon._build_dim0_shard_redistribution_plan = schedule._build_dim0_shard_redistribution_plan
dist_muon._dtensor_storage_regions = schedule._dtensor_storage_regions
```
—— 因为 pin 的 `dist_muon` 是以 `from ... import _build_replicated_to_dim0_shard_plan` 形式**按值**导入的，这是 backport 必须手工补的一处。
- 消费点：`patches/torchtitan/distributed/flex_shard/__init__.py:4`；断言测试 `tests/unit_tests/test_dsv4_muon.py:12,138-146,149-152`；运行期所有 DistMuon 走 `_build_parameter_redistribution_plan → _build_dim0_shard_redistribution_plan/_dtensor_storage_regions`。

**上游 main 现状**
- 文件仍在同路径（HEAD 1285 行）。6ee6b10b 之后只有 `da2f82670`(#4587) 触碰它，且只改 `_validate_disjoint_regions_in_bounds`（`_tensor_region_intersection_numel` 改动在 6ee6b10b 本身）。
- 逐定义 AST 比对（本地 vs 6ee6b10b / HEAD）：`_RedistributionGroup`、`_tensor_region_intersection_numel`、`_tensor_region_intersection`、`_build_dim0_shard_redistribution_plan`、`_redistribution_group` 在 HEAD 上**完全一致**；`_dtensor_storage_regions` 仅差「两个同体 `elif` 合并为一个 `or`」，语义相同。
- 锚点仍在：`_device_mesh_ranks`（HEAD `_optimizer_reshard_schedule.py:893`）、`_validate_tensor_region_partition`（:371）、`_dtensor_storage_region_for_participant`（:913）、`_build_dim0_shard_redistribution_plan`（:585，签名与 6ee6b10b 一致）。

**复制性质**：逐字副本（6ee6b10b）+ monkeypatch 绑定（含跨模块注入）。

**结论分类：①（上游已含且等价 → 删除）**。风险 **S**。新 pin 上 `_build_dim0_shard_redistribution_plan` 等已原生存在，patch 只会重绑同样的实现。

#### 2.4 `.../flex_shard/replica_dedup.py`

**目标 / 能力**：HSDP `dp_replicate` **副本计算去重**——按 FQN 哈希选 owner 坐标，owner 只跑自己那份原生 bucket 计划，随后用**规范的异步 broadcast**（`_ReplicaBroadcastPipeline`，`:218-376`）把更新后的本地参数分片广播给同 `dp_replicate` 组；并提供 DCP 所需的 `standard_state_dict_layout`/`standard_load_layout` 上下文（`:424, 449`）与 owner 态校验（`:388-401`）。
- 机制：`_ORIGINAL_BUILD_DIST_MUON = dist_muon_module.build_dist_muon`（`:44`）、`_ORIGINAL_STEP = dist_muon_module.DistMuon.step`（`:45`）、`build_dist_muon(..., enable_hsdp_replica_dedup=False)`（`:490-510`）、`muon.step = _replica_dedup_step.__get__(...)`（`:509`）、`apply()` 重绑三处工厂（`:513-517`）。
- 我们仓库的消费点：`patches/torchtitan/components/optimizer.py:67,73,81,88`（DCP state-dict 包装）、`torchtitan_npu/config/configs.py:55,107-108`（`muon_enable_hsdp_replica_dedup` → 工厂 kwargs）、`extensions/cpu_offload/cpu_offload_muon.py:735,739`（组合时显式拒绝）、`override/common/optimizer.py:984`（按 `_replica_dedup_all_layouts` 分派）、`tests/unit_tests/extensions/cpu_offload/test_no_optimizer_state.py:27-92`。

**上游标记（逐字）**
```
## TODO: Delete this patch once https://github.com/pytorch/torchtitan/pull/4411
## ([DistMuon] Deduplicate HSDP replica compute) is merged upstream.
```

**上游 main 现状**
- 能力**仍无**：
  ```
  $ git grep -n -i "replica_dedup\|deduplicate.*replica\|replica.*deduplicat" HEAD   → 空
  $ git log --all --oneline --grep="4411"                                            → 空
  $ git grep -n "build_dist_muon" HEAD                                               → 空
  ```
  （本机镜像无网络，PR #4411 的状态以 HEAD=3386c6d6f 的树与引用为准：未合并。）
- **但 patch 的挂钩点已被上游删除/搬迁**：
  ```
  $ git log --oneline -S "def build_dist_muon" -- torchtitan/
  17067f00b Refactor optimizer configuration (#4904)      ← 删除
  5b7089939 [Kimi 2.7] rename DistributedMuon to DistMuon (#4155)
  ```
  替代构造路径（HEAD）：`torchtitan/components/optim/optimizer.py:186  class DistMuon(FlexShardDistMuon, BaseOptimizer)`，用 `@dataclass Config(compute_sharding_by_fqn, bucket_configs, …)` + `__init__(config, *, params)`；`torchtitan/distributed/flex_shard/__init__.py` 的 `__all__` 只剩 `DistMuon/BlockShard/BucketConfig/ComputeLayout/Owned`（不再导出工厂）。
- **模块路径也搬了**：`torchtitan/components/optimizer/optimizer.py` → `torchtitan/components/optim/optimizer.py`（`2164881c4`，2026-09-29，#4660；`git log --oneline --diff-filter=A -- torchtitan/components/optim/optimizer.py`）。patch 第 31 行的 `import torchtitan.components.optimizer.optimizer as optimizer_module` 在 HEAD 直接 ImportError。
- patch 依赖的**私有**内部在 HEAD 仍在（所以主体逻辑不必重写，只需换钩子）：`_initialize_plan`、`_validate_plan_across_ranks`、`_redistribution_runtime`、`_bucket_plans`、`_local_tensor_spec`、`_parameter_compute_layouts`、`_apply_update`、`reserve_buffers` 均在 `dist_muon.py`/`_optimizer_reshard_runtime.py` 中命中。

**复制性质**：**本地自研实现**（上游无对应文件/PR），不是任何 upstream commit 的拷贝 → 重新实现 = 真合并/重写。

**结论分类：④（与上游方向冲突，需重新设计）**
- 判据：能力本身上游仍无（形似③），但③要求「核对锚点仍存在」，而锚点（模块级工厂 `build_dist_muon` + `components/optimizer` 模块路径）已被 #4904/#4660 的**配置化重构**移除 → 按规则判④。
- 重新设计要点：
  1. 钩子改为 `torchtitan/components/optim/optimizer.py` 的 `DistMuon(Config, params=...)`（或在 `build_optimizer`/`OptimizerConfig` 层做 factory hook），并新增 `enable_hsdp_replica_dedup` 到该 `Config`；
  2. `step` 覆盖、`_apply_update` 包裹、`_initialize_plan`/`reserve_buffers` 等可直接沿用（HEAD 内部签名未变）；
  3. 我们仓库的 plumb 点（`config/configs.py:107-108`、`extensions/cpu_offload/cpu_offload_muon.py:735`、`patches/torchtitan/components/optimizer.py:67`）需同步到新入口；
  4. 持续跟踪 PR #4411；若上游合入，按上游方案替换本 patch。
- 冲突风险：**M**（钩子面重写，主体算法可留）。

#### 2.5 `torchtitan_npu/patches/torchtitan/hf_datasets/text_datasets.py`

**目标 / 能力**：为 pinned 文本数据管线补上**按文档的 pooling 对齐**：`pad_segments_to_multiple()`（`:58-84`，把每个 packed 文档段补齐到 `multiple` 的整数倍，pad 的 label 为 `IGNORE_INDEX`）、`AlignedTextDataset`（`:87-147`，在入 buffer 前补齐，使压缩 KV pool / indexer 的分组不跨文档）、`AlignedHuggingfaceDataloader`（`:150-193`，绕过 `HuggingFaceTextDataLoader.__init__` 直接调 `ParallelAwareDataloader.__init__`）。
- 它**不是 monkeypatch**：靠子类化 pin 的 `torchtitan.hf_datasets.text_datasets.HuggingFaceTextDataset/HuggingFaceTextDataLoader`（`:47-50`）与 `torchtitan.components.dataloader.ParallelAwareDataloader`（`:44`）、`torchtitan.components.loss.IGNORE_INDEX`（`:45`）。
- 消费点：`torchtitan_npu/models/deepseek_v4_1/config_registry.py:269`（`AlignedHuggingfaceDataloader.Config(per_doc_alignment=…)`）、`torchtitan_npu/models/deepseek_v4_1/vision/dataloader.py:26`（`pad_segments_to_multiple`）、`tests/unit_tests/models/deepseek_v4_1/test_text_dataloader.py:16-20`。

**上游标记（逐字，节选）**
```
## Backports the per-document pooling alignment API onto the pinned upstream text
## dataset: ``pad_segments_to_multiple`` as a data-pipeline field and its
## implementation.  Mirrors upstream ``torchtitan/components/data/packing.py``
## (``_pad_segments_to_multiple``) and ``torchtitan/components/data/types.py``
## (``DatasetBuildContext.pad_segments_to_multiple``), which the 0.3.0 release
## predates.  Remove this module once the pinned TorchTitan dependency carries the
## equivalent field and the compatibility tests pass.
```

**上游 main 现状**（header 说法需要更正）
- 能力**不在上游**：
  ```
  $ git grep -n "pad_segments_to_multiple" HEAD            → 空（全树）
  $ git grep -n "_pad_segments_to_multiple" HEAD           → 空
  HEAD:torchtitan/components/data/types.py:98  class DatasetBuildContext:
       tokenizer / max_context_length / num_tokens_per_microbatch / read_options
       / max_num_documents / num_mtp_layers     ← 无 pad_segments_to_multiple
  HEAD:torchtitan/components/data/packing.py                ← 无 _pad_segments_to_multiple
  ```
- header 声称的「上游镜像」实际只存在于 **fork 分支**：`git log --all -S pad_segments_to_multiple` 命中的是 `9cb8f2274`（"packing: pad document segments to a multiple in the build context"，author sdmyzlp，2026-09-14），
  ```
  $ git branch -a --contains 9cb8f2274   → br_dpsk_v4_1, remotes/local/br_dpsk_v4_1
  $ git merge-base --is-ancestor 9cb8f2274 origin/main && echo YES || echo NO   → NO
  ```
  即：**pytorch/torchtitan mainline 从未有过该字段**（审计建议同步修正 header，避免误导后续 pin 升级）。
- 锚点**全部消失**：
  ```
  $ git grep -n "HuggingFaceText" HEAD -- torchtitan/        → 空
  $ git log --oneline -S "class HuggingFaceTextDataset"      → 1b04fc1c3 [Grain] Replace dataset/dataloader with grain (#4088)  (2026-08-19)
  $ git cat-file -e HEAD:torchtitan/components/dataloader.py → 不存在（同 commit 删除）
  ```
  HEAD 的 `torchtitan/hf_datasets/text_datasets.py` 已被改写成 grain 版：只定义 `_read_text` / `TextProcessor` / `ChatProcessor`，并从 `torchtitan.components.data.{dataset,sources,types}` 导入（`SampleProcessor`、`HuggingFaceRandomAccessSource`、`HuggingFaceStreamingSource`、`DatasetBuildContext`）。`hf_datasets/` 目录本身仍在（未迁到 `torchtitan_recipes/`）。
- 顺带（同族、影响同一次 pin 升级）：我们仓库还引用了 `torchtitan.hf_datasets.multimodal.utils.packing.MMSamplePacker`（`models/deepseek_v4_1/vision/dataloader.py:24`），该文件在 HEAD **已不存在**。

**复制性质**：本地自研（子类 + 独立函数），无可重拷贝的上游物。

**结论分类：④（与上游方向冲突，需重新设计）**
- 判据：能力上游仍无（像③），但③要求锚点仍在——`HuggingFaceTextDataset/Loader` 与 `ParallelAwareDataloader` 均被 grain 迁移（#4088）删除，patch 连 import 都无法完成（显式失败，不会静默）→ 判④。
- 重新设计要点：
  1. 能力本身很小（`:58-84` 的 `pad_segments_to_multiple` 逻辑可直接移植），迁移到新栈：`torchtitan/components/data/packing.py`（对齐放置在 packer 之后/`DatasetBuildContext` 新增字段）或 `components/data/dataset.py` 的 `SampleProcessor` 链路；
  2. 我们仓库的 `per_doc_alignment` 配置面（`models/deepseek_v4_1/config_registry.py:269`、`vision/dataloader.py:26`）与 `AlignedHuggingfaceDataloader.Config` 需要重写成 grain 版 dataloader config；
  3. 保留的测试契约（`tests/unit_tests/models/deepseek_v4_1/test_text_dataloader.py`）需按新栈重写。
- 冲突风险：**M**。

#### 2.6 `torchtitan_npu/patches/torchtitan/hf_datasets/multimodal/mm_collator.py`

**目标 / 能力**：把 `MultiModalCollator._build_mrope_positions`（`:24` 取原函数）替换为 `patched_build_mrope_positions`（`:40-138`）：**不再把视频 `[T,H,W]` 摊成 T 个 `[1,H,W]`**，而是一个连续占位 run 消费一个原始 3D grid（本地版删掉的正是 `torch.repeat_interleave(grid_thw_videos, grid_thw_videos[:,0], dim=0); grid_thw_videos[:,0] = 1`），并保留 pin 的 **2-D (batch, seq)** 契约、用 `resets = positions[:,1:] < positions[:,:-1]` 判文档边界。
- `apply()`（`:141-148`）先做**形参契约校验**（`_EXPECTED_PARAMETERS`，`:25-33`），契约不符则抛 `RuntimeError("Unsupported upstream MultiModalCollator signature")`，否则 `MultiModalCollator._build_mrope_positions = patched_build_mrope_positions`。
- 消费点：`torchtitan_npu/models/qwen3_5/__init__.py:13-20`（导入 patch 并 `_mm_collator_patch.apply()`）；`tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py:50-52,111`。

**上游标记（逐字）**
```
"""Backport contiguous-video MRoPE positions to the pinned upstream.

Upstream PR: pending publication of the Qwen3.5 contiguous video-MRoPE fix.
Upstream base: ``b175497ea3502388bb9513391810a86ca46dbfa8``.

The collator builds positions before the model Config exists, so a late Config
override cannot repair this behavior. Remove when the pinned upstream revision
contains the contiguous-video MRoPE fix.
"""
```

**上游 main 现状（该能力已在上游，但契约不同）**
- 修复已合入 main：`539aa6cdf`「[Bug][Qwen3.5 Video] Fix MRoPE temporal patch positions (#4239)」(2026-08-20)，`git merge-base --is-ancestor 539aa6cdf origin/main` → 是祖先。该 commit 的说明与我们 patch 的意图一致（"stop expanding each video grid [T,H,W] into T independent [1,H,W] entries"），`git log -S repeat_interleave -- .../mm_collator.py` 命中 `539aa6cdf`（删除该展开）。
- HEAD 的实现（`mm_collator.py:188-…`）注释明说 *"Our data pipeline emits one contiguous placeholder run per video instead, so each run must consume the original [T, H, W] grid as a single 3D region."*，且全文件再无 `repeat_interleave`。
- **内容差异（关键）**：HEAD 的契约已变为 **1-D**：`tokens.unsqueeze(0)`、`positions = positions.unsqueeze(0)`、`resets = positions[:, 1:] == 0`、`return mrope_positions.squeeze(0)`（docstring：`tokens: (num_tokens,)`）。这是 `73aed7f6c`「fold batch dim (#4121)」(2026-08-20，origin/main 祖先) 带来的；我们 patch 声明的 base `b175497ea`（2026-08-03）与其后 pin 是 2-D 契约，本地实现也是 2-D。
- 其它上游演进：`0b7536202`(#5028)、`304fd882e`(#4285)、`d1e46be33`(#4398)、`d398a8fb9`(#4572)、`62b423a5a`(#4347)、`ec953b360`(#4773)、`ed35bdaed`(#5055) 等仍在改这个文件（HEAD 与 6ee6b10b 相比该文件 307 行变化），但那些改动不在本 patch 的覆盖面内。

**复制性质**：**本地变体**（pin 版函数体删掉视频展开 + 少量防御性写作），非任何上游 commit 的逐字拷贝。

**结论分类：②（上游已含但版本/内容更新）**
- 同步要点：升级 pin 后**删除本 patch**（`models/qwen3_5/__init__.py:16-20` 的 `apply()` 调用一并去掉），因为上游已自带该修复；**不要为了“保留 NPU 行为”重新套上**。
- 明确的坑：`apply()` 的契约校验只比较**形参名与 kind**，而 HEAD 的签名与 `_EXPECTED_PARAMETERS` **完全一致** → 若 pin 升级后忘记删除，patch 会**静默替换** HEAD 的实现，把 1-D 数据流当成 2-D 处理（多出/错位的 batch 维、`<` 与 `== 0` 的边界判定差异），属于静默数值错误而非显式报错。
- 同时修正 header：`Upstream PR: pending publication …` 已过时（应为 #4239 / `539aa6cdf`）。
- 冲突风险：**删除 S；误留 M**。

---

### 3. 路径与搬迁核对（HEAD = 3386c6d6f）

| 路径 | HEAD 状态 | 证据 |
|---|---|---|
| `torchtitan/distributed/flex_shard/` | **仍在**（`dist_muon.py`、`optimizer_reshard.py`、`_optimizer_reshard_schedule.py`、`_optimizer_reshard_runtime.py`、`README.md`、`__init__.py`） | `git ls-tree -r --name-only HEAD -- torchtitan/distributed/flex_shard/` |
| `torchtitan/hf_datasets/` | **仍在**（`text_datasets.py` + `multimodal/**`），但 `text_datasets.py` 已 grain 化，`HuggingFaceTextDataset/Loader` 被删 | `1b04fc1c3` (#4088)；`git grep HuggingFaceText HEAD` 空 |
| `torchtitan/components/optimizer/` | **已不存在** → `torchtitan/components/optim/` | 新路径由 `2164881c4` (#4660) 引入 |
| `torchtitan/components/dataloader.py` | **已删除**（`ParallelAwareDataloader` 消失） | `1b04fc1c3` (#4088) |
| `torchtitan/distributed/parallel_dims.py` | **已删除** → `torchtitan/distributed/parallelism_context.py`；`MeshAxisName` 不再由该模块导出 | `5dc97a3e7` (#4905) |
| `torchtitan/hf_datasets/multimodal/utils/packing.py` | **已删除** | `git cat-file -e HEAD:<path>` 失败 |
| `torchtitan_recipes/` | **新增**顶层包（`b09a88b9b` #4913 重组 recipes）；`torchtitan/models/kimi_k2_7/config_registry.py` 在 HEAD 已不存在（`torchtitan_recipes/tests/models/kimi_k2_7.py` 取而代之）；`torchtitan/experiments/` 仍在 | `git ls-tree --name-only HEAD torchtitan_recipes/` |
| 未搬迁的 flex_shard patch 锚点 | `_build_dim0_shard_redistribution_plan`(HEAD:585)、`_RedistributionGroup`、`_device_mesh_ranks`(:893)、`_validate_tensor_region_partition`(:371)、`_dtensor_storage_region_for_participant`(:913) 均在 | `git grep -n` |

**与本次审计无关但会影响同一次 pin 升级的路径断裂**（供 parent 汇总）：
- `torchtitan.distributed.parallel_dims.MeshAxisName`（`models/common/muon.py:12`、`models/deepseek_v4/config_registry.py:29`、`models/deepseek_v4_1/config_registry.py:23`）；
- `torchtitan.components.optimizer`（多处）→ `torchtitan.components.optim`；
- `torchtitan.distributed.flex_shard.dist_muon._ParameterComputeLayout`（`extensions/cpu_offload/cpu_offload_muon.py:44`）在 HEAD 已被删除；
- `BlockShard(block_size=...)` → `block_sizes=(...)`（§2.1 同步要点 3）。

---

### 4. 复现命令（审计可重跑）

```bash
U=/home/lrwei/pytorch/torchtitan; P=/home/lrwei/pytorch/torchtitan-v030; O=/home/lrwei/cann/torchtitan-npu

## 基线：pin 的 flex_shard == 858b9e248
for f in dist_muon.py optimizer_reshard.py _optimizer_reshard_schedule.py _optimizer_reshard_runtime.py __init__.py README.md; do
  diff <(git -C $U show 858b9e248:torchtitan/distributed/flex_shard/$f) $P/torchtitan/distributed/flex_shard/$f >/dev/null && echo "$f IDENTICAL"
done

## backport base 之后的提交
git -C $U log --oneline --reverse 6ee6b10b..HEAD -- torchtitan/distributed/flex_shard/
git -C $U show --stat --format='%h %s' 6ee6b10b          # PR #4122 文件清单

## 逐定义 AST 比对（本地 patch vs 6ee6b10b / 858b9e248 / HEAD）
python3 /tmp/g4/astcmp.py  $O/torchtitan_npu/patches/torchtitan/distributed/flex_shard/dist_muon.py \
        torchtitan/distributed/flex_shard/dist_muon.py 6ee6b10b 858b9e248 HEAD
python3 /tmp/g4/astcmp.py  $O/.../optimizer_reshard.py        torchtitan/distributed/flex_shard/optimizer_reshard.py 6ee6b10b 858b9e248 HEAD
python3 /tmp/g4/astcmp.py  $O/.../_optimizer_reshard_schedule.py torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py 6ee6b10b 858b9e248 HEAD
python3 /tmp/g4/percommit.py 10cc9ffbb,c215a12ee,da2f82670,9c5302543,63c4e9fef,c104b2947,892ecfed0,7031b7f1b,17067f00b,5dc97a3e7,b09a88b9b

## 锚点消失证据
git -C $U log --oneline -S "def build_dist_muon" -- torchtitan/
git -C $U log --oneline -S "class HuggingFaceTextDataset"
git -C $U grep -n "pad_segments_to_multiple" HEAD
git -C $U merge-base --is-ancestor 9cb8f2274 origin/main || echo "9cb8f2274 不在 mainline"
git -C $U merge-base --is-ancestor 539aa6cdf origin/main && echo "MRoPE 连续视频修复已在 mainline"
```

（辅助脚本 `/tmp/g4/{astcmp,loc_vs_up,unparse_diff,fnset,percommit,mmcmp,vps2}.py` 为本次审计临时生成，未写入任何仓库。）

<!-- section g2 -->

## TorchTitan-NPU 上游补丁审计 — Group 2（components / trainer / distributed）

审计对象：`torchtitan_npu/patches/torchtitan/{components/checkpoint.py, components/ema.py, components/metrics.py,
components/optimizer.py, components/validate.py, trainer.py, distributed/context_parallel.py,
distributed/full_dtensor.py, distributed/varlen_cp.py}`

- 我们的 pin：`torchtitan==0.3.0`，tag `v0.3.0` = `086bf6c16` (2026-09-02)。
- 最新 mainline：`/home/lrwei/pytorch/torchtitan` HEAD = `3386c6d6f` (2026-10-08)。
- 所有结论以 `git log -S` / `git grep` / `git merge-base --is-ancestor` 为准，只读、未修改任何文件。

### 0. 汇总表

| # | 模块 | 目标（上游符号） | 上游标记 | main 现状 | 结论 | 补丁形式 | 工作量 |
|---|------|------------------|----------|-----------|------|----------|--------|
| 1 | `components/checkpoint.py` | `components.checkpointer.CheckpointManager`（构造/加载） | `Pending upstream PR: #3985` | **已合入**（`1b9eef3bd`），机制不同：`CheckpointManager(ema=…)` 构造参数 + `dcp.py:410-415` reseed；`components/checkpoint.py`、`checkpoint_utils.py` 已删 | **②** | backport 子类 + 3 处命名空间注入 | S |
| 2 | `components/ema.py` | `OptimizersContainer`（EMA 伪优化器） | `Pending upstream PR: #3985` | **已合入**（`1b9eef3bd`），形态不同：`components/optim/ema.py:96 class EMA`，`decays/half_life_fractions` 列表 + `Optim.Config.ema` | **②** | backport（新上游文件下沉副本，无 `apply()`） | S（代码）/ M（含配置与测试迁移） |
| 3 | `components/metrics.py` | `MetricsProcessor.log` | `Pending upstream PR: #3864` | 注入部分**已吸收**到 `trainer.py:453`（`collect_aux_loss_metrics` 内联）；`MetricsProcessor` 仍在 `observability/metrics.py:508`，但属性 `parallel_dims`→`parallelism_context`；结构化/人读 aux 行上游无对应 | **②** | monkey-patch（函数包装） | S |
| 4 | `components/optimizer.py` | `checkpoint_utils/optimizer.init_optim_state` + `OptimizersContainer.{state_dict,load_state_dict}` | `Pending upstream PR: #4474` | A 部分（#4474）**已合入** `components/optim/utils.py:29`；B 部分（#4411 replica 去重）上游仍无，且 `dist_muon.py:1488` 明确 "replicated compute is not implemented"；路径 `components/optimizer/`→`components/optim/` | **③**（A 可删、B 必须重写） | monkey-patch + A 的函数体 backport | M |
| 5 | `components/validate.py` | `Validator.post_dataloading_process` | `Pending upstream PR: #3634` | **符号已删除**（#4116 重构），新 seam 是 `Model.preprocess_inputs`（`protocols/model.py:52`），`Validator.validate` 在 `components/validate.py:224` 调用 | **②** | monkey-patch（函数包装） | M |
| 6 | `trainer.py` | `Trainer.{post_dataloading_process, train_step, pp_forward_backward_step}`、`torchtitan.trainer.Trainer` | `Pending #3634` + `Pending #3985` + `Backport #4421` | **Trainer 已拆分**：`trainer.py:39 class Trainer` 只剩 input/validate/report，`training_engine.py:71 class TrainingEngine` 持有 model/optim/checkpointer；`post_dataloading_process`/`pp_forward_backward_step` 全库无命中；`#4421` 已合入（`8792c3a24`）；`set_step_denominator` 上游不存在 | **②**（其中 H2D=①、NPU 调度兼容层=③ 且必须重写） | 属性替换 + `inspect.getsource` 源码重写并 `exec` + 子类注入 | L |
| 7 | `distributed/context_parallel.py` | `context_parallel.api.cp_shard` | `Pending upstream PR: #3430` | **锚点整体删除**：`context_parallel/` 包被 #4617(`f35966713`) 换成单文件，`cp_shard` 无命中；新 API 为 `shard_tensors`(:317) + `ContextParallelLoadBalancer`(:50) + `CPInnerAttention.prepare_cp_metadata`(:70)；varlen+CP 上游只有 Ulysses(#4459) | **③**（能力未被吸收，注入点全部消失 → 升级时必须重写） | monkey-patch（包装 + 替换 torch 私有符号） | L |
| 8 | `distributed/full_dtensor.py` | `distributed.full_dtensor.validate_config` | `Pending upstream PR: #3430` | **整个模块被删**：`601cf4d23` "Remove the full_dtensor SPMD backend (#4217)"，`#4218` 再删 partial_dtensor；`parallelism.spmd_backend` 字段不复存在；`ScaledDotProductAttention`→`ScaledDotProductInnerAttention` | **④** | monkey-patch（整函数替换） | M（外围迁移+回归） |
| 9 | `distributed/varlen_cp.py` | `torchtitan/distributed/varlen_cp.py::CPVarlenMetadata` | `Pending upstream PR: #3430` | **上游仍无**：`CPVarlenMetadata`/`k_global_gather_indices` 在 HEAD 零命中；PR #3430 的 commit `6ef6a6f6c` **不是** HEAD 祖先；上游 varlen+CP 走 Ulysses（`cp_attention.py:339`），metadata 类型改名 `VarlenAttentionMetadata` | **③**（锚点改名/移动 → 升级时必须重写） | backport（纯定义，无 `apply()`） | M–L |

结论分布：②×5（1/2/3/5/6）、③×3（4/7/9）、④×1（8）、①×0；另有 3 个"能力已吸收"的子块（optimizer A 部分、trainer H2D、metrics 注入）。

### 0.1 上游针对本组的重点变化（对应任务里的 a–d）

**(a) Trainer 是否仍是一个类 —— 已拆分。**
- `torchtitan/trainer.py:39 class Trainer(Configurable)`：只负责 dataloader/tokenizer/validator/metrics/训练循环；字段 `engine: TrainingEngine`(:137)、`engine_cls: type[TrainingEngine] = TrainingEngine`(:138)，`self.engine = self.engine_cls(...)`(:166)。
- `torchtitan/training_engine.py:71 class TrainingEngine(Configurable, torch.distributed.checkpoint.stateful.Stateful)`：拥有 `_initialize_model`(:312)、`_initialize_optim`(:422)、`_initialize_checkpointer`(:436)、`forward_backward`(:541)、`optim_step`(:776)。
- `Trainer` 上**没有** `self.model_parts` / `self.optimizers` / `self.checkpointer`（`grep -n "self.model_parts" torchtitan/trainer.py` 无命中；它们通过 `self.engine.*`，metrics 侧靠 `self.metrics_processor.model_parts = engine.model_parts`）。
- `Trainer.post_dataloading_process`、`Trainer.pp_forward_backward_step` 在 mainline **全库无命中**；`Trainer.train_step(self, data_iterator: Iterator[TrainingMicrobatch])`(:329) 里也没有 `.to(self.device)` 文本（H2D 已下沉）。

**(b) CP 输入/metadata sharding —— 全部重写。**
- `cp_shard` 消失；`torchtitan/distributed/context_parallel.py` 现在的公开面是 `__all__ = [... "get_cp_input_seq_len", "get_token_fragments", "shard_tensors", "supports_cp_inner_attention"]`(:39-47)。
- `shard_tensors(input_dict, *, input_shardings: dict[str, SpmdType], permutation)`(:317) 按 `SpmdType` 里 CP 轴是 `spmd.Shard(dim)` 的名字做 permute + shard（`_cp_shard_dims` :171）。
- load balancer 从字符串变成配置对象：`ContextParallelLoadBalancer`(:50) / `HeadTailCPLoadBalancer`(:59) / `PTRRFlexAttentionCPLoadBalancer`(:97)，`generate_permutation()` 产出 `[1, seq_len]` 或 `[batch, seq_len]`；配置字段 `context_parallel_load_balancer: ContextParallelLoadBalancer.Config | None`（`config/parallelism.py:167`）。
- 调用点在**模型**里：`models/common/decoder.py:289 def preprocess_inputs(...)` → `:334 def _cp_shard(...)`，内部先 `get_cp_input_seq_len` → `load_balancer_config.build(...)` → `permutation` → 每个 CP inner attention 的 `prepare_cp_metadata(metadata, permutation=...)` → `context_parallel.shard_tensors(...)`；metadata 容器是 `AttentionMetadataMap = Mapping[type[InnerAttention], AttentionMetadata]`（`models/common/attention/__init__.py`）。
- `prepare_context_parallel_input` / `prepare_cp_metadata`(自由函数) 都不存在；`prepare_cp_metadata` 现在是 `CPInnerAttention` 的 staticmethod（`cp_attention.py:70`、`:96`、`:275`）。
- varlen+CP：只有 `UlyssesCPVarlenInnerAttention`（`cp_attention.py:339`，`prepare_cp_metadata` 直接 `del permutation; return attention_metadata`），集成用例 `cp_ulysses_varlen`。**没有** K/V all-gather + headtail 的 rank-local varlen metadata。

**(c) full_dtensor / spmd backend —— 后端选择器被整体删除。**
- `601cf4d23 Remove the full_dtensor SPMD backend (#4217)`（body：*"Now that spmd_types is the default backend … drop `parallelism.spmd_backend == "full_dtensor"` and everything that only existed to support it"*）；随后 `5dd944e62 Remove apply_cp_to_forward, the partial_dtensor Context Parallel path (#4218)`。
- `git grep spmd_backend torchtitan/` → 无命中（`get_spmd_backend`/`set_spmd_backend` 一并消失）；只剩 `torchtitan/distributed/spmd_types.py`，入口 `annotate_input_spmd_types(parallelism_context, input_dict, input_sharding)`(:180)，typecheck 开关是 `debug.spmd_typechecking`（`config/validation.py:157`）。
- `torchtitan/distributed/full_dtensor.py` 已不存在；`ScaledDotProductAttention` 改名 `ScaledDotProductInnerAttention`（`models/common/attention/attention.py:505`）。

**(d) checkpoint / EMA / metrics / optimizer 的 mainline 挂点。**
| 能力 | mainline 位置 |
|------|---------------|
| EMA 容器 | `components/optim/ema.py:96 class EMA(OptimizersContainer)`；构建/步进在 `components/optim/optim.py:90-94`、`:111-112`，配置 `optim.ema`(`:40`) |
| EMA 入 checkpoint | `components/checkpointer/base.py:41 EMA = "ema"`；`dcp.py:164 ema: EMAContainer | None`、`:183-184 self.states[EMA] = ema`；冷启动 `dcp.py:410-415`；装配 `training_engine.py:451 ema=self.optim.ema` |
| 部分初始化 optimizer state | `components/optim/utils.py:29 init_optim_state`（#4474 已合入，docstring/实现与我们的 patch 等价） |
| metrics | `observability/metrics.py:304 class MetricsProcessor`、`:508 def log(..., extra_metrics)`；aux 注入在 `trainer.py:453` |
| optimizer state dict | `components/optim/optimizer.py:404 state_dict` / `:418 load_state_dict`（无 hook 点） |
| 目录迁移 | `components/{checkpoint.py,checkpoint_utils.py,metrics.py}` 删除；`components/optimizer/` → `components/optim/`；`models/common/attention.py` → `attention/` 包 |

---

### 1. `torchtitan_npu/patches/torchtitan/components/checkpoint.py`

**目标**：用 `EMACheckpointManager` 替换 `CheckpointManager`：在内嵌 `Config` 上增加 `ema_weights`，构造 EMA 容器并注册为第 4 类状态（`self.states["ema_optimizer"]`），并在 `dcp_load` 时探测 checkpoint 是否真的含 EMA 前缀 —— 没有就剔除该状态并让 EMA 从刚加载的权重冷启动（`components/checkpoint.py:25-71`，注入在 `:74-80`）。

**上游标记**：`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3985`（`:1`；docstring 声明 *"Backport EMA-aware checkpoint construction and loading from TorchTitan PR #3985 / Remove this module after …"*）。

**上游 main 现状：已吸收，但机制不同。**
- PR #3985 已合入：`1b9eef3bd add dummy-ema-optimizer for model averaing (#3985)`（2026-09-22），该 commit 同时改了 checkpointer/`training_engine.py`。
- 状态键名变了：`torchtitan/components/checkpointer/base.py:41` → `EMA = "ema"`（我们用的是 `"ema_optimizer"`）。
- 不再子类化 + 改 Config，而是构造参数：
  - `torchtitan/components/checkpointer/dcp.py:164` → `ema: EMAContainer | None,`
  - `torchtitan/components/checkpointer/dcp.py:183-184` → `if ema is not None:` / `self.states[EMA] = ema`
  - 装配点 `torchtitan/training_engine.py:451` → `ema=self.optim.ema,`
- 冷启动 reseed 内联在 `_load_checkpoint`，用 states 选择而不是探测 DCP metadata：
  - `torchtitan/components/checkpointer/dcp.py:410-415` → `# Reseed EMA from the just-loaded weights if it wasn't itself restored` / `if MODEL in states and EMA in self.states and EMA not in states:` / `self.states[EMA].load_state_dict({})`
  - 选择逻辑在 `components/checkpointer/base.py:392 _states_to_load(model_only)`（`exclude_from_loading` / model_only）。
- 我们的 `dcp_load` 重写目标已不存在：mainline 只有 `_load_checkpoint(states, checkpoint_id, *, from_hf, from_quantized)`（`dcp.py:359`），无 `dcp_load`。
- 模块路径消失：`git ls-files torchtitan/components` 只有 `checkpointer/` 包，没有 `checkpoint.py` / `checkpoint_utils.py`；`torchtitan.components.checkpoint.CheckpointManager` 与 `torchtitan.components.checkpoint_utils` 的 import 会直接 ImportError。
- 另有第二个后端 `components/checkpointer/torch_checkpointing.py:279-280` 同样接收 `ema=`。

**结论：②（上游已吸收，形态/机制不同，需按上游新形态重新实现）。**补丁形式：**backport（子类复刻 #3985）+ monkey-patch 注入**（同时改写 `checkpointer.dcp`、`checkpointer`、`components.checkpoint` 三个命名空间的名字）。

**重写要点（S）**
1. 删除本模块与 `patches/torchtitan/__init__.py:6` 的导入；不需要再改 `Config`、不需要 `_checkpoint_has_prefix` 探测（上游用 `EMA not in states` + `load_state_dict({})`）。
2. EMA 的开关注解移到 `optim.ema`（见模块 2），`TrainingEngine._initialize_checkpointer` 会自动把 `self.optim.ema` 传下去。
3. 若下游还有依赖 `"ema_optimizer"` 键名的工具（旧 checkpoint 转换/对比脚本），需要改成 `"ema"` 并处理旧 ckpt 的向前兼容。

### 2. `torchtitan_npu/patches/torchtitan/components/ema.py`

**目标**：backport `EMAOptimizersContainer`（`OptimizersContainer` 子类、伪优化器，每个参数在 `optimizer.state[p]["ema_params"]` 存一份 EMA；`Config(enable, decay, half_life_fraction, start_step, step_bias, update_every_n_steps, offload_to_cpu)`；复用 `checkpoint_utils` 的 FQN flatten/resharding 机制做 DCP；`load_state_dict({})` 时从模型冷启动）（`components/ema.py:26-95`、`:144-191`）。

**上游标记**：`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3985`（`:1`）。

**上游 main 现状：已吸收，形态不同。**
- `torchtitan/components/optim/ema.py:96` → `class EMA(OptimizersContainer):`（类名从 `EMAOptimizersContainer` 改为 `EMA`）。
- 配置从"单副本 + enable 开关"变成"多副本列表"：`ema.py:116` `decays: list[float]` / `ema.py:123` `half_life_fractions: list[float]`，另有上游新增的 `buffer_patterns`（对 `named_buffers()` 里的 FQN 做 EMA）；**没有** `enable` 字段（不配置 `optim.ema` 即关闭），`start_step` / `step_bias` / `update_every_n_steps` / `offload_to_cpu` 保留同名。
- 每副本一个 checkpoint key：`ema.py:35-48` `_EMA_STATE_KEY = "ema_params"`、`FIXED_DECAY_KEY_PREFIX = "decay_"`、`HALF_LIFE_KEY_PREFIX = "half_life_"`、`def ema_key(...)`/`parse_ema_key(...)`；导出脚本 `scripts/checkpoint_conversion/convert_ema_to_hf.py`。
- 构建与步进不再由 Trainer hook 驱动：`components/optim/optim.py:40 ema: EMA.Config | None = None`、`:90-94 self.ema = config.ema.build(model_parts=model_parts)`、`:111-112 if self.ema is not None: self.ema.step(current_step)`；`OptimizersContainer` 侧无 hook。
- 冷启动 reseed 在 checkpointer 里（见模块 1），EMA 自身 `load_state_dict({})`（`ema.py:550`）。

**结论：②（上游已吸收，形态/机制不同，需按上游新形态重新实现）。**补丁形式：**backport（上游新文件的本地副本）**，无 `apply()`，只提供被 `trainer.py` patch 引用的类。

**重写要点（S 代码 / M 含迁移）**
1. 删除本模块，改用 `torchtitan.components.optim.EMA`；配置映射：`ema_weights.enable=true, decay=d` → `optim.ema.decays=[d]`；`half_life_fraction=f` → `optim.ema.half_life_fractions=[f]`；`enable=false` → 不设置 `optim.ema`。
2. 我们仓库里对 `EMAOptimizersContainer` / `EMA_OPTIMIZER` 的引用需改：`tests/unit_tests/ema/test_ema_initial_load.py`（523 行，直接 `from torchtitan_npu.patches…import EMA_OPTIMIZER, EMAOptimizersContainer`）、`torchtitan_npu/extensions/torchtitan/trainer.py`(TrainerEx) 相关 wiring。
3. 上游 `buffer_patterns` 是新增能力，NPU 若对 MoE `expert_bias` 类 buffer 有平均需求可直接用。
4. `offload_to_cpu` 的 pinned-CPU 异步管线上游同样有（`ema.py:402/489`），可整段替换，不需要保留我们的实现。

### 3. `torchtitan_npu/patches/torchtitan/components/metrics.py`

**目标**：包装 `MetricsProcessor.log`，把 `collect_aux_loss_metrics(self.parallel_dims)` 合进 `extra_metrics`，并额外写 structured trace（`sl.log_trace_scalar`）与人读日志 `aux_loss step: {step}  name: value`（`components/metrics.py:21-45`）。

**上游标记**：`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3864`（`:6`）。

**上游 main 现状：主能力已吸收到另一个 seam，属性名已变。**
- aux 指标注入已上游化，且不在 `MetricsProcessor.log` 里，而是在 `Trainer.train_step`：
  - `torchtitan/trainer.py:453` → `**collect_aux_loss_metrics(parallelism_context),`（同一 dict 作为 `extra_metrics=` 传给 `self.metrics_processor.log(...)`，`:455-461`）
  - `torchtitan/models/common/aux_loss.py:227 def collect_aux_loss_metrics(parallelism_context: ParallelismContext) -> dict[str, float]`（签名是 `parallelism_context`，不是 `parallel_dims`）
- `MetricsProcessor` 仍在，签名未变：`torchtitan/observability/metrics.py:304 class MetricsProcessor`、`:508 def log(`、`:514 extra_metrics: dict[str, Any] | None = None`。
- 但我们的锚点属性不存在：该类现在以 `parallelism_context: ParallelismContext`（`observability/metrics.py:354`、`:393`）为准，没有 `parallel_dims`；`MetricsProcessor` 也从 `torchtitan.components.metrics` 搬到了 `torchtitan.observability.metrics`（`ad1461f8f improve logging folder and module organization (#4628)`，`torchtitan/components/metrics.py` 已删除）。
- patch 的"结构化日志/人读行"上游**没有对应实现**：`MetricsProcessor.log` 只写 `self.logger.log(metrics, step)`（`:571-575`），structured trace 在 `train_step` 里只记 `global_valid_tokens` 等（`trainer.py:410-411`）。

**结论：②（能力已吸收但形态/机制不同：注入点从 `MetricsProcessor.log` 变成 `Trainer.train_step`，且属性改名；结构化/人读 aux 行仍需按新 seam 重新实现）。**若产品上接受丢掉 `aux_loss step:` 行与 trace scalar，则退化为①（直接删除）。补丁形式：**monkey-patch（函数包装）**。

**重写要点（S）**
1. 删除 `MetricsProcessor.log` 包装（升级后它既重复又会因 `self.parallel_dims` 缺属性报错）。
2. 需要保留人读行时，改在 `Trainer.train_step` 的 `extra_metrics` 组装处（`trainer.py:447-453`）追加 `titan_logger.info` / `sl.log_trace_scalar`；或写一个 `MetricsProcessor` 子类覆盖 `log`（配置级 `metrics` 是 Configurable，可走 `override.imports`，比改上游符号稳）。
3. `collect_aux_loss_metrics` 调用要改用 `parallelism_context`（我们的 aux_loss patch 也要同步改签名，属于另一个 group）。

### 4. `torchtitan_npu/patches/torchtitan/components/optimizer.py`

**目标**：两个独立能力。(A) 替换 `init_optim_state`：只为"尚无 state 的 trainable 参数"用零梯度 + lr=0 走一步来物化 optimizer state，保留既有梯度与 state，并把 Adam 的 step/exp_avg 清零 —— 使部分初始化的 optimizer state 能被完整 checkpoint 保存/恢复（`:28-62`，`#4474`）。(B) 包装 `OptimizersContainer.state_dict` / `load_state_dict`，在读写前后用 `flex_shard.replica_dedup` 的 `standard_state_dict_layout` / `standard_load_layout` 上下文，让 DistMuon 的 HSDP replica 去重状态能走 TorchTitan 标准 DCP 布局（`:65-89`，`#4411`）。

**上游标记**：`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4474`（`:5`；docstring *"Patch partial optimizer-state materialization for checkpointing. Remove this module after …"*）。

**上游 main 现状：A 已吸收（路径改名），B 仍无。**
- A：#4474 已合入 `dc3985ad4 Fix checkpointing with partially initialized optimizer state (#4474)`（2026-09-05）。mainline `torchtitan/components/optim/utils.py:29 def init_optim_state(optim)`，docstring 即 *"…This runs a step with zero gradients and ``lr=0`` … No-op if every trainable parameter already has state. Existing gradients and optimizer state are preserved."*，实现与我们的 patched 版本逐行等价（`:41-85`）。调用点 `components/optim/optimizer.py:414/422`。
- A 的锚点路径全部消失：`components/optimizer/` 包不存在了（`components/optim/`），`torchtitan.components.checkpoint_utils.init_optim_state` 不存在（`checkpoint_utils.py` 已删）。
- B：上游仍无 replica 去重。mainline `components/optim/optimizer.py:404 def state_dict` / `:418 def load_state_dict` 没有任何 hook 点，且 DistMuon 明确拒绝 replicated compute：
  - `torchtitan/distributed/flex_shard/dist_muon.py:1488-1489` → `f"Muon parameter {fqn!r} requests explicit replicated compute on "` / `f"mesh axes {replicated_axes}; replicated compute is not implemented"`
  - `git log --grep="4411"` 无命中，`git grep replica_dedup` 在 mainline flex_shard 无命中。

**结论：③（B 部分上游仍无，必须保留；A 部分升级后可删）。**锚点改名/移动 → **升级时必须重写**。补丁形式：**monkey-patch（函数替换，4 个命名空间）+ A 的函数体 backport**。

**重写要点（M）**
1. 删除 `patched_init_optim_state` 与 `checkpoint_utils` 注入（A 交给上游 `components/optim/utils.py:29`）；同时删掉 `import torchtitan.components.checkpoint_utils` 与 `torchtitan.components.optimizer`（升级后会 ImportError）。
2. B 重新指向 `torchtitan.components.optim.optimizer.OptimizersContainer`（`components/optim/__init__.py` 也 re-export），并核对上游 `state_dict` 仍走 `get_flat_optim_state_dict`（`optimizer.py:404-431`）—— 我们的 `replica_dedup.standard_*_layout` 依赖该平铺布局，若上游改了布局键名/顺序就必须同步改下游 `replica_dedup`。
3. 评估是否把 B 收敛为 `override`（配置级替换 `optim.optimizer` 容器）而不是 monkey-patch 上游 `OptimizersContainer`；上游现在把"参数更新及其状态"整体收进 `Optim`（`components/optim/optim.py:29`），patch 上游容器类会与 `Optim._update`/CUDA-graph 包装相互作用，风险高于替换容器实现。

### 5. `torchtitan_npu/patches/torchtitan/components/validate.py`

**目标**：包装 `Validator.post_dataloading_process`：如果 `model_parts[0]` 有 `build_attention_masks`（DeepSeek-V4 系）就由模型自己造 mask/metadata 与 CP sharding，并在 `spmd_backend == "full_dtensor"` 时补 `full_dtensor.parallelize_inputs`；否则回落到上游流程（`components/validate.py:20-41`）。

**上游标记**：`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634`（`:6`）。

**上游 main 现状：目标符号已删除，能力被新 seam 取代。**
- `git grep -n "post_dataloading_process" -- '*.py'` → **无命中**（`e8e39abc6 Refactor post dataloading process (#4116)` 起逐步拆掉，PR #3634 `0ff246463` 只是 DeepSeek-V4 模型接入）。
- 新 seam：`torchtitan/protocols/model.py:52 def preprocess_inputs(self, input_dict, *, parallelism_context, parallelism, max_num_documents=None, max_context_length=None, **kwargs) -> tuple[inputs, labels, extra_kwargs]`；`Decoder` 的标准实现 `models/common/decoder.py:289`（内部完成 mask 构造 + `_cp_shard` + `annotate_input_spmd_types`）。
- Validator 侧直接调用模型：`torchtitan/components/validate.py:224`（PP 分支）与 `:256`（非 PP 分支）→ `model_parts[0].preprocess_inputs(input_dict, parallelism_context=self.parallelism_context, parallelism=self.parallelism)`，在 `self.parallelism_context.activate_spmd()` 上下文里。
- `Validator.__init__` 已改签名（`components/validate.py:101`，`parallelism_context=` 等），`Validator` 仍存在（`:54`）。
- `torchtitan.distributed.full_dtensor` 已删（#4217），`parallelism.spmd_backend` 字段不存在 → patch 的两条判断都要重写。

**结论：②（上游已吸收但形态/机制不同：hook 从 Validator 挪到模型 `preprocess_inputs`，需按新 seam 重新实现）。**补丁形式：**monkey-patch（函数包装）**。

**重写要点（M）**
1. 删除本 patch；把三个 NPU 模型的 `build_attention_masks` 适配为 `preprocess_inputs`：`torchtitan_npu/models/deepseek_v4/model.py:191`、`deepseek_v4_1/model.py:639`、`deepseek_v3_2/model.py:465`（入参从 `(inputs, labels, extra_kwargs, cp_mesh=…, load_balancer_type=…)` 改为 `input_dict + parallelism_context/parallelism`，返回 `(inputs, labels, extra_kwargs)`，内部 `input_dict.pop("input")/pop("labels")`）。
2. `if self.parallelism.spmd_backend == "full_dtensor": full_dtensor.parallelize_inputs(...)` 删除；SPMD 标注改由模型内 `annotate_input_spmd_types(parallelism_context, input_dict, input_sharding)`（`distributed/spmd_types.py:180`）完成，并要求每个 tensor 输入都有 layout 条目（否则上游会 raise）。
3. token 统计语义变化：上游在 engine 侧 `self.ntokens_seen += num_tokens_per_microbatch_per_dp_rank // parallelism_context.cp`（`training_engine.py:613-616`），Validator 侧用 `microbatch.loss_token_counts`（`validate.py:190-192`）；我们 patch 里的 `labels.numel()` 记账要废弃。

### 6. `torchtitan_npu/patches/torchtitan/trainer.py`

**目标**：四件事。(i) `Trainer.post_dataloading_process` → 模型 `build_attention_masks` seam + `full_dtensor.parallelize_inputs` / `spmd_types.annotate_input_spmd_types` 分派 + CP 后 token 记账（`:34-74`，`#3634`）。(ii) `Trainer.pp_forward_backward_step` → 给只有 `_step_microbatches` 私有 API 的老 NPU pipeline schedule 打桥（`:77-130`）。(iii) `patch_train_step()`：用 `inspect.getsource` 取 `Trainer.train_step` 源码字符串，改 H2D 为 `non_blocking=True`（`#4421` backport）并插入 `AuxLoss.set_step_denominator(global_valid_tokens)`（`#3864`），再 `exec` 回去（`:135-160`）。(iv) `EMATrainer(Trainer)` + `torchtitan.trainer.Trainer = EMATrainer`：把 `ema_weights` 塞进 `Config`/`checkpoint`，注册 optimizer step pre/post hook 驱动 EMA（`:163-189`，`#3985`）。

**上游标记**：三行头（`:1-3`）
```
## Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634
## Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3985
## Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4421
```

**上游 main 现状（逐子能力）**

| 子能力 | mainline 现状 | 判据 |
|--------|---------------|------|
| (i) `post_dataloading_process` + `build_attention_masks` | **符号删除**，改为 `Model.preprocess_inputs` | `git grep post_dataloading_process` / `build_attention_masks` 均无命中；`protocols/model.py:52`；`training_engine.py:590 _preprocess_microbatch_groups`（`:600-608` 调 `preprocess_inputs`，`:606 microbatch.to_input_dict(self.device, non_blocking=True)`） |
| (ii) `pp_forward_backward_step` | **符号删除**；新实现直接调 schedule 的新签名 | `training_engine.py:740 _pp_forward_backward_microbatch_group` → `:757-765 self.pp_schedule.step(arg_mbs=…, kwarg_mbs=…, target_mbs=…, losses=…, loss_kwargs=…, return_outputs=False, finalize_gradients=…)` |
| (iii-a) H2D non_blocking | **已合入** | `8792c3a24 [data] Page-lock text batches so the H2D copy is async (#4421)`；`components/data/types.py:39 to_input_dict(device, *, non_blocking=False)`、`:44 value.to(device, non_blocking=non_blocking)`；`training_engine.py:606`；`trainer.py` 里已无 `.to(self.device)` 文本 → `patch_train_step()` 的 `assert src.count(...) == 1` 会失败 |
| (iii-b) aux 分母 | **能力已吸收，机制不同**：从 dataloader 的 microbatch token counts 来，经 `aux_loss_denominators` 传入模型 | `trainer.py:329-410`（`microbatch.loss_token_counts` / `routing_token_counts`，DP all-reduce），`training_engine.py:566-569 if "aux_loss_denominators" in model_kwargs: model_kwargs["aux_loss_denominators"] = global_routing_token_counts`；`decoder.py:331 input_dict["aux_loss_denominators"] = None`；`AuxLoss.set_step_denominator` / `set_step_denominator` 全库无命中 |
| (iv) EMATrainer | **能力已吸收到 Optim/engine**（见模块 1/2），且 Trainer 已无 `optimizers`/`checkpointer` 属性 | `components/optim/optim.py:40/90-94/111-112`；`training_engine.py:451 ema=self.optim.ema`；`grep self.model_parts torchtitan/trainer.py` 无命中 |

补充：Trainer 仍是类（`trainer.py:39`），但只持有 dataloader/tokenizer/validator/metrics/engine（`engine: TrainingEngine` :137、`engine_cls` :138、`self.engine = self.engine_cls(...)` :166）；`train.py:12 from torchtitan.trainer import Trainer` 后 `config.build()` 走 `Config._owner`，所以"替换 `torchtitan.trainer.Trainer` 名字"这套注入在升级后既无必要（EMA 归 Optim）也不可靠（recipe 直接构造 `Trainer.Config` 时 `_owner` 仍是原 Trainer）。

**结论：②（模块整体必须按上游新形态重写；其中 (iii-a) 已是①可直接删除，(ii) 属③且锚点消失必须重写，(i)(iv) 属②）。**补丁形式：**混合** —— 类属性替换（`Trainer.post_dataloading_process` / `pp_forward_backward_step`）+ **源码字符串重写后 `exec`**（`patch_train_step`）+ 子类注入（`torchtitan.trainer.Trainer = EMATrainer`）。

**重写要点（L）**
1. **(i) 迁到模型**：三个 NPU 模型实现/适配 `preprocess_inputs`（与模块 5 的重写是同一件事，只做一次）；删掉 Trainer 侧 hook 与 `self.ntokens_seen += labels.numel()`。
2. **(ii) 迁到 engine**：NPU pipeline schedule 兼容层改为覆盖 `TrainingEngine._pp_forward_backward_microbatch_group`（或在 `pp_schedule` 工厂层提供带 `arg_mbs/kwarg_mbs/target_mbs/finalize_gradients` 的 `step`），注意新签名多了 `finalize_gradients` 与 `_stages` 语义；老 schedule 若没有 `_step_microbatches` 仍需保留桥，但桥的"载荷"必须换成新签名。
3. **(iii-a) 删除** `patch_train_step` 的 H2D 改写（`#4421` 已进上游）；若 NPU 需要页锁定，改在 `components/data/collators.py` 侧对齐。
4. **(iii-b) 换机制**：aux 分母改走 `extra_kwargs["aux_loss_denominators"]`（在模型 `preprocess_inputs` 里放 key，engine 会填全局值），删除 `set_step_denominator` 语法树级插入；`global_valid_tokens` 口径从 `labels.numel()` 改为上游的 per-microbatch token counts。
5. **(iv) 删除 EMATrainer**：EMA 由 `optim.ema` + `Optim.step` 驱动；Trainer 子类只在确需改训练循环时才保留（此时用 `engine_cls`/`Trainer` 子类显式传入，而不是替换模块属性）。
6. 注意 `torchtitan_npu/extensions/torchtitan/trainer.py`(TrainerEx) 与 `models/*/config_registry.py`（`from torchtitan.trainer import Trainer`，返回 `Trainer.Config`）需要一起调整，否则 `EMATrainer.Config` 永远不会被 build。

### 7. `torchtitan_npu/patches/torchtitan/distributed/context_parallel.py`

**目标**：包装 `torchtitan.distributed.context_parallel.api.cp_shard`：当 attention mask 是 varlen（`VarlenMetadata` 或"全 varlen 的 dict"）时，先让原 `cp_shard` 只切输入，再用 `CPVarlenMetadata.from_global` 造 rank-local varlen metadata（只允许 headtail 均衡），并保持 hybrid attention 的 dict 形状；另外把 torch CP 私有的 `_compiled_create_block_mask` 换成 eager 版本（NPU 下 compile 会起 worker 进程，无法重新初始化）（`:24-87`、`apply()` `:90-120`）。

**上游标记**：`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430`（`:6`）。

**上游 main 现状：目标符号整体删除，能力未被吸收（只有不等价的 Ulysses 变体）。**
- `torchtitan/distributed/context_parallel/` **包已被删除**（工作区残留 `__pycache__`），改成单文件 `torchtitan/distributed/context_parallel.py`，由 `f35966713 [CP] Decouple input and metadata sharding with configurable CP load balancer (#4617)` 完成。
- `cp_shard` 不存在：`git grep -n "cp_shard"` 只命中 `context_parallel.py:171 _cp_shard_dims` 与 `models/common/decoder.py:334 def _cp_shard`。
- 新公开面：`context_parallel.py:39-47 __all__`（`ContextParallelLoadBalancer` / `HeadTailCPLoadBalancer` / `PTRRFlexAttentionCPLoadBalancer` / `get_cp_input_seq_len` / `get_token_fragments` / `shard_tensors` / `supports_cp_inner_attention`）：
  - `:50 class ContextParallelLoadBalancer(Configurable, ABC)`、`:59 HeadTailCPLoadBalancer`（`generate_permutation()` 用 `_HeadTailLoadBalancer(...)._generate_indices()`）、`:97 PTRRFlexAttentionCPLoadBalancer`
  - `:185 def get_cp_input_seq_len(input_dict, *, input_shardings)`、`:207 get_token_fragments`、`:317 def shard_tensors(input_dict, *, input_shardings, permutation)`
  - 配置字段变成对象：`torchtitan/config/parallelism.py:167 context_parallel_load_balancer: ContextParallelLoadBalancer.Config | None = None`（不再是 `"headtail"`/`"ptrr"` 字符串）
- metadata 制备搬进 attention 后端：`torchtitan/models/common/attention/cp_attention.py:53 class CPInnerAttention`、`:70 @staticmethod @abstractmethod def prepare_cp_metadata(attention_metadata, *, permutation)`；`Decoder._cp_shard`（`models/common/decoder.py:334-401`）串联 `get_cp_input_seq_len` → load balancer build → `prepare_cp_metadata` → `shard_tensors`。
- varlen+CP 上游只有 Ulysses：`cp_attention.py:339 class UlyssesCPVarlenInnerAttention`，`UlyssesCPInnerAttention.prepare_cp_metadata`（`:275-281`）就是 `del permutation; return attention_metadata`；集成用例 `tests/integration_tests/features.py:234 cp_ulysses_varlen`。**没有** K/V all-gather + head-tail permutation 的 varlen 路线（PR #3430 的 commit `6ef6a6f6c` 用 `git merge-base --is-ancestor` 判定 **not in HEAD**）。
- 类型改名/删除：`VarlenMetadata` → `VarlenAttentionMetadata`（`models/common/attention/attention.py:79`）；`AttentionMasksType` 全库无命中，改为 `AttentionMetadataMap = Mapping[type[InnerAttention], AttentionMetadata]`（`models/common/attention/__init__.py`）。
- `torchtitan.distributed.full_dtensor` 删除，`spmd_backend` 字段删除 → patch 里的 `original_cp_shard` 与 full_dtensor 分支都无法续用。
- eager block-mask workaround 的锚点也换了：mainline 走 titan 自己的 `models/common/attention/attention.py:707 _compiled_create_block_mask = torch.compile(create_block_mask)` 与 `:710 def create_attention_mask`，而不是 torch 私有的 `torch.distributed.tensor.experimental._context_parallel._attention._compiled_create_block_mask`。

**结论：③（能力上游仍无：KV-allgather + headtail 的 varlen CP metadata 没有被吸收，Ulysses 路线不等价；且锚点/符号全部改名或删除 → 升级时必须重写）。**若产品愿意把 varlen CP 整体改为 Ulysses（heads 切分 + metadata 全局），那这条能力可由上游直接提供，就退化为②。补丁形式：**monkey-patch（包装上游函数 + 替换 torch 私有符号）**。

**重写要点（L）**
1. 注入点从 `cp_shard` 换成 `ContextParallelLoadBalancer` 子类 + `CPInnerAttention.prepare_cp_metadata` 钩子：NPU 的 rank-local varlen metadata 应当在 `prepare_cp_metadata(metadata, *, permutation)` 里产出（返回值即模型 forward 用的 metadata），而不是在 shard 之后旁路改写 mask dict。
2. `permutation` 由上游 load balancer（`generate_permutation()`，shape `[1, seq_len]` 或 `[batch, seq_len]`）提供，`shard_tensors` 只负责 permute + `spmd.shard`；我们不再自己构造、也不再传 `"headtail"` 字符串。
3. mask 容器改成 `AttentionMetadataMap`（key 是 InnerAttention 类型），dict 形状"保 key"逻辑要按类型化 key 重写（`Decoder._cp_shard` 已按 `CPInnerAttention` 子类遍历并逐个 `prepare_cp_metadata`）。
4. eager `create_block_mask` workaround 改挂 `torchtitan.models.common.attention.attention._compiled_create_block_mask`（或直接 monkey-patch `create_attention_mask`），并确认新 CP 路径不再经过 torch 私有模块。
5. 与模块 9 是同一套重写：先定 metadata 契约，再改注入点。

### 8. `torchtitan_npu/patches/torchtitan/distributed/full_dtensor.py`

**目标**：整体替换 `torchtitan.distributed.full_dtensor.validate_config`，在 `full_dtensor` 后端 + CP + `ScaledDotProductAttention` 组合下直接 `raise NotImplementedError`（`:19-33`）。

**上游标记**：`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430`（`:6`）。

**上游 main 现状：目标模块被上游删除。**
- `601cf4d23 Remove the full_dtensor SPMD backend (#4217)` 删除 `torchtitan/distributed/full_dtensor.py`（commit message：*"Now that spmd_types is the default backend … and partial_dtensor is the fallback, drop `parallelism.spmd_backend == "full_dtensor"` and everything that only existed to support it."*），`git grep full_dtensor torchtitan/` 零命中。
- 紧接着 `5dd944e62 Remove apply_cp_to_forward, the partial_dtensor Context Parallel path (#4218)`；`parallelism.spmd_backend` 字段与 `distributed/utils.get_spmd_backend/set_spmd_backend` 一并消失（`git grep spmd_backend torchtitan/` 零命中），只剩 spmd_types 一套：`distributed/spmd_types.py:180 annotate_input_spmd_types`、`set_spmd_meshes`/`current_spmd_mesh`/`spmd_mesh_group`，typecheck 开关为 `debug.spmd_typechecking`（`config/validation.py:157`）。
- 我们 patch 里 `from torchtitan.models.common.attention import ScaledDotProductAttention` 也已改名：`ScaledDotProductInnerAttention`（`models/common/attention/attention.py:505`），且 `torchtitan.models.common.attention` 由文件变成包。

**结论：④（与上游方向冲突：整个 backend 被上游删除，目标符号不存在；继续维护 full_dtensor 等于永久分叉）。**补丁形式：**monkey-patch（整函数替换）**。

**重写要点（M，主要是外围迁移与回归）**
1. 删除本 patch 及 `patches/torchtitan/__init__.py:9 from .distributed import … full_dtensor`；`distributed/full_dtensor.py` 在新 pin 下 import 即失败。
2. 全仓清理 `spmd_backend` / `get_spmd_backend` / `from torchtitan.distributed.full_dtensor import …`：至少 `torchtitan_npu/override/qwen3_5/parallelize.py:15/87/102/120/242/256`、`override/deepseek_v4/moe.py:21`、`override/common/swiglu_group/ascendc.py:71`、`override/common/token_dispatcher.py:51/207/217`、`tests/unit_tests/override/common/test_moe_score_absorption.py`、`tests/unit_tests/models/qwen3_5/test_qwen3_5_npu_adapter.py`、`tests/unit_tests/models/deepseek_v4_1/test_model_split.py:183`、`docs/feature_guides/ep_process_group.md:26`。
3. 若确实需要保留 full_dtensor 后端（NPU 上某些算子只在 DTensor 下通），必须**自带一份后端实现**（下游模块 + 自己的模型 parallelize 校验），不再 monkey-patch 上游；同时把相应校验（CP + SDPA 不支持）搬到自己的 `preprocess_inputs`/parallelize 校验里。
4. 迁移后需要重新验证 CP + spmd_types 在 NPU 上的数值与 shape 约束（上游 `shard_tensors` 会 raise "sequence length … must be divisible by CP size" 等新错误）。

### 9. `torchtitan_npu/patches/torchtitan/distributed/varlen_cp.py`

**目标**：backport 上游未合入 PR #3430 的 `torchtitan/distributed/varlen_cp.py`：`CPVarlenMetadata`（`cu_seq_q/cu_seq_k/max_q/max_k/k_global_gather_indices`）+ `from_global(...)`，从全局 `VarlenMetadata` 推导每个 CP rank 的 Q 分段与 K/V 压缩收集索引（支持 head-tail load balancer 的置换与逆置换）（`:24-171`）。

**上游标记**：`# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430`（`:1`；docstring *"This mirrors ``torchtitan/distributed/varlen_cp.py`` from upstream PR #3430."*）。

**上游 main 现状：上游仍无该能力/符号；锚点已改名或删除。**
- `git grep -n "CPVarlenMetadata\|k_global_gather_indices\|k_local_indices" HEAD` → **零命中**。
- 该 PR 未合入：`6ef6a6f6c [Varlen CP] Enable Varlen attention under Context Parallel + Full DTensor`（message 内 `Pull-Request: …/pull/3430`）经 `git merge-base --is-ancestor 6ef6a6f6c HEAD` 判定 **NO**；同分支的 `c17c7ede4`（PTRR varlen）、`4325fc713`/`176f557cd`、`3d8fd83e4` 同样都不在 HEAD。
- 上游对 varlen+CP 的答案是 Ulysses：`models/common/attention/cp_attention.py:339 class UlyssesCPVarlenInnerAttention`，其 `prepare_cp_metadata`（`UlyssesCPInnerAttention.prepare_cp_metadata`，`:275-281`）对 metadata 不做任何 rank-local 改写；用例 `cp_ulysses_varlen`。
- 锚点变动：`torchtitan.models.common.attention.VarlenMetadata` → `VarlenAttentionMetadata`（`models/common/attention/attention.py:79`）；`AttentionMasksType` 删除；`cp_shard` 删除；torch 私有 `_HeadTailLoadBalancer._generate_indices()`（本文件 `:91` 依赖）在新上游流程中被包装为 `HeadTailCPLoadBalancer.generate_permutation()`（`distributed/context_parallel.py:89-94`）。

**结论：③（上游仍无 → 保留；但锚点 `VarlenMetadata`、`_LoadBalancer` 私有 API、`cp_shard` 调用点全部改名/移动/删除 → 升级时必须重写）。**补丁形式：**backport（纯定义，无 `apply()`）**，由 `context_parallel.py` patch 与 NPU 模型共同消费。

**重写要点（M–L）**
1. 数据类保留，但对外契约改成 `CPInnerAttention.prepare_cp_metadata(metadata, *, permutation)` 的返回类型（rank-local metadata），permutation 由上游 load balancer 提供；`from_global` 里"自己造 load balancer/自己算 indices"的部分改为接收上游 `permutation`（`[1, seq_len]` 或 `[batch, seq_len]`）。
2. 类型名跟随上游：`VarlenMetadata` → `VarlenAttentionMetadata`；不要再 import `AttentionMasksType`。
3. 消费方同步适配：`torchtitan_npu/override/deepseek_v3_2/sparse_attn/ascendc.py:28/258`（`isinstance(varlen_meta, CPVarlenMetadata)` 的分支要挂到新 metadata 类型上）、`torchtitan_npu/models/deepseek_v4/…`（compressor 依赖 `k_global_gather_indices` 语义）、`tests/unit_tests/models/deepseek_v3_2/test_cp_metadata.py`、`tests/unit_tests/models/deepseek_v4/{test_cp_dispatch.py,cp_dispatch_worker.py,test_cp_compressor.py}`、`tests/unit_tests/conftest.py:205`。
4. 依赖 torch 私有 `torch.distributed.tensor.experimental._context_parallel._load_balancer._LoadBalancer` 的部分要么随上游改用公开/自研 permutation（推荐，与 #4617 对齐），要么在 NPU torch 版本上继续 pin 并加版本护栏。
5. 若最终改走 Ulysses 路线（CP 切 head、metadata 全局），本模块可整体删除，但需评估 AscendC sparse-attention 内核是否支持 head 维切分——这是决策分叉点，建议单独出设计评审。

---

### 附：给升级排期的建议顺序

1. **先做无争议的删除**：模块 8（full_dtensor，必须先于其它迁移，因为 `spmd_backend` 是横切依赖）、模块 4A（init_optim_state）、模块 6(iii-a)（H2D）、模块 1/2/6(iv)（EMA 整体切换到 `optim.ema`）。
2. **再做 seam 迁移（一次到位）**：模块 5 + 模块 6(i)（三个 NPU 模型实现 `preprocess_inputs`）、模块 3（aux 指标/日志搬家）、模块 6(iii-b)（`aux_loss_denominators`）。
3. **最后做 CP 重写（风险最高、工作量最大）**：模块 7 + 模块 9，二选一——(a) 保留 KV-allgather + headtail 的 varlen CP，按 #4617 新形态重写（L）；(b) 改用上游 Ulysses varlen CP，删除自研 metadata（需 AscendC 内核支持 head 切分）。
4. 模块 4B（DistMuon replica 去重）在上游仍无对应实现，升级后必须重新锚定到 `components/optim/` 并复核 `get_flat_optim_state_dict` 布局；上游 `dist_muon.py:1488` 明确不支持 replicated compute，短期不要指望上游吸收。

<!-- delta audit: 3386c6d6f..eb92f645c (74 commits) -->

## TorchTitan-NPU patch 审计 — Delta（上游 3386c6d6f → eb92f645c，74 commits）

**任务**：把 G1–G4 四份审计（`/tmp/patch_audit_g{1..4}.md`，基线 `3386c6d6f`，2026-10-08）更新到新的 `origin/main` = `eb92f645c`（`[CI] Preserve NumPy 1.x support by constraining PyArrow (#5372)`）。
**方法**：只读。所有上游结论用 `git log 3386c6d6f..origin/main --grep=#NNNN`、`git grep <symbol> origin/main -- torchtitan/`、`git ls-tree -r origin/main`、`git diff 3386c6d6f origin/main -- <path>` 复核；引用一律写成 `origin/main:path:line`。
**范围**：上游 74 个 commit（`git rev-list --count 3386c6d6f..origin/main` = 74）；我方仓库 `/home/lrwei/cann/torchtitan-npu` 未改动任何文件。

---

### 0. 三条前置结论

1. **任务列出的 24 个 pending PR 一个都没有落地。** 对 `#4095 #4411 #4474 #4734 #4650 #4763 #3949 #3430 #4529 #4598 #4664 #4534 #4575 #4663 #3634 #4122 #4217 #4218 #4516 #4708 #4651 #3864 #3985 #4421` 逐个执行 `git log --oneline 3386c6d6f..origin/main --grep="#NNNN"` **全部为空**；74 个 commit 里出现过的 PR 号只有 `#4772 #4835 #4942 #5016 #5018 #5023 #5033 #5046 #5072 #5101 #5107 #5114 #5119 #5137 #5151 #5154 #5155 #5170 #5171 #5172 #5182 #5184 #5186 #5188 #5191 #5204 #5206 #5207 #5214 #5215 #5217 #5218 #5219 #5225 #5230 #5231 #5232 #5233 #5246 #5247 #5249 #5250 #5252 #5266 #5267 #5268 #5269 #5270 #5271 #5273 #5274 #5277 #5279 #5280 #5282 #5288 #5289 #5290 #5294 #5296 #5298 #5307 #5311 #5332 #5340 #5341 #5348 #5349 #5350 #5359 #5368 #5369 #5372 #5374`（与本任务集合无交集）。
   → 四份报告里所有「#NNNN 未合并」「能力仍无」的判断**全部继续成立**（唯一例外见 §2 的 G1-1：`#4772` 用另一条路线补上了 batch/step 级归一化，但不叫 `set_step_denominator`）。
2. **74 个 commit 里有 3 个是"地基级"改动，会连带改写大量锚点**：`6f2225575 Reorganize repo structure… (#5277)`（目录搬迁）与 `d2a55eeba [RL] Checkpoint prompt data state (#5114)` + `b81852f00 Build sharded model configs (#5280)`（checkpoint/trainer 契约）。§4 单独列。
3. G3/G4 中多数"死模块"判定（④）完全没变：`ep_chunk_pass.py` 仍不存在、`ep_eager_chunk.py` 仍在、`build_dist_muon` 仍不存在、`replica_dedup` 仍无、`HuggingFaceTextDataset`/`ParallelAwareDataloader`/`pad_segments_to_multiple` 仍无、`spmd_backend`/`full_dtensor.py` 仍未回来。

---

### 1. 变更总览（74 commits 按领域分组，只列有影响的）

#### 1.1 仓库结构重组（影响面最大）
| commit | PR | 一行说明 |
|---|---|---|
| `6f2225575` | #5277 | `protocols/` → `models/base/`；`hf_datasets/` → `components/data/`；`components/validate.py` → `validator.py`；`models/utils.py` 拆为 `models/common/{moe_state_dict_adapter,flops}.py`；`models/common/attention/kda.py` → `linear_attention.py`（`KDAAttentionMetadata` → `LinearAttentionMetadata`）；依赖改由 requirements 文件驱动，`torch_checkpointing` 改为 git commit pin（PyPI 0.1.0 无 `default_resharder`）；测试目录重排 |
| `49a07abcf` | #5230 | `spmd_types` 升到 commit `2ae49fce`（`.ci/docker/requirements.txt` 现为 `a1b86dea…`，仍在 git pin 上） |
| `913413cca` | #5231 | 新增 `models/common/attn_gym.py`，模型 import 不再强依赖 attn_gym |

#### 1.2 MoE / aux loss / 模型
| commit | PR | 一行说明 |
|---|---|---|
| `8b4a1300e` | #4772 | **新增 `BatchWiseLoadBalanceLoss(AuxLoss)`**（`moe.py:609`，rolling batch-wise/step 级 expert 计数）+ `AuxLoss._reset_accumulators/_reduce_token_partials/_norm_sum_scores_over_tokens`；`inject` 的 `denominator` 文档明确为 "Step-global routed-token count"；`memory_policy.tag_sac_policy` 新增 **Pass 1b：in-place mutation 永不重算（MUST_SAVE）**；`utils.py`/`config_utils.py` 同步 |
| `fc6ed2d32` | #4835 | **上游新增 `SparseIndexerLoss(AuxLoss)`**（`deepseek_v4/compressor.py:327`），经 FlexAttention `out_transform` 注入，并在 DSV4 optimizer post-build hook 里 `register_aux_loss_zero_hook`；`attention.py:137` "SparseIndexerLoss requires aux_loss_denominator." |
| `c9b10b5ec` | #5252 | microbatch aux-loss 统计用 `local_compile` 局部编译 |
| `4a4176e97` | #5271 | `TokenChoiceTopKRouter.Config.freeze_gate`（新字段）+ 新配置 transform `config/transform/freeze_moe_routing.py` |
| `8a75afd90` / `90c7fa4b5` | #5307 / #5296 | **Varlen 成为兼容模型的默认 attention，`ScaledDotProduct*` 全树删除**；Flux 也改用 varlen |
| `3210be829` | #5266 | vision encoder 改为 CP-invariant（只 dp_shard），`qwen3_5/vision_encoder.py` 增 CP 的 `spmd.mutate_type` |
| `01b1b69e3` / `69cbe7783` | #5184 / #5225 | 新增 `models/common/attention/{cp_linear_attention,cp_gdn}.py`：linear attention 的**rank-local CP metadata**（`ContextParallelLinearAttentionMetadata` + `prepare_linear_attention_cp_metadata`） |
| `826d018f5` | #5046 | `TrainingEngine.Config.dist_moe` → **`dist_moe_runtime`** 重命名；新增 DistMoE vLLM 推理路径 |
| `2a00be295` / `7fe5f405f` | #5267 / #5033 | offset_rmsnorm recompile limit 16；KimiK3 attention residual 局部编译 + fla kernel override |
| `49c4cff1f` | #5204 | kimi 多模态 TP SPMD 边界类型修复 |

#### 1.3 Linear / 权重梯度 / DistMuon
| commit | PR | 一行说明 |
|---|---|---|
| `1b0680885` | #5249 | `Linear.forward` "参数按存储形态传递、只读一次"：`_flatten_weight_and_bias` 变 **staticmethod**、`_unflatten_output(output, weight)`、`_linear(input, weight, bias)` |
| `06dd3abe2` | #5250 | `Linear.Config.inplace_wgrad_accum: bool = True`（新字段）+ `models/common/inplace_wgrad_accum.py`；graph tracing 下打开会 raise |
| `de222da3f` | #5137 | HiMidLoLinear 的 LM-head WGRAD 就地累加进 fp32 grad |
| `83113b9cc` | #5119 | graph_trainer 新增 `wgrad_accumulation.py`（+373）跨 add 链融合 WGrad |
| `210fc60cb` | #5279 | DistMuon 重分布输入/方向走 BF16：**`ComputeLayout.compute_dtype` 新字段**、`_ParameterComputeLayout.compute_dtype`、`_RedistributionBucketPlan.compute_dtype`、`_build_bucket_plans(get_compute_dtype=…)` |
| `31b503c89` | #5191 | Kimi K3 在 CP 下的 DistMuon layout 修复 |
| `e8fedd8e5` / `15f5f1a91` | #5341 / #5340 | MXFP8 Dist-MoE / HF 加载的 Dist-MoE 权重修复 |

#### 1.4 graph_trainer
| commit | PR | 一行说明 |
|---|---|---|
| `bed2b9ea8` | #5023 | **默认 pass 不再含 `cuda_graph_pass`**（改由外层 trainer 捕获）；`apply_graph_passes` 行号 506→495；`construct_default_graph_passes` 439→442；`paged_stash_memory_policy`/`graph_builder`/`spmd_graph_builder` 同步 |
| `448b9c8be` | #5101 | **FullAC 迁到 torch_remat**（`FullAC(RegionAC)`，`save_regions=[]`）：删除 `torch.utils.checkpoint` 路径与 dynamo LRU workaround；aux-loss 指标/专家计数不再重复计数，删除 `aux_loss.py` 里 "FullAC over-counts" 的已知限制段 |
| `c29918394` | #5171 | SimpleFSDP 梯度按 reduce_dtype 规约（`simple_fsdp.py` +201 行） |
| `1e95ae9a6` / `9ab0f591e` | #5206 / #5207 | MTP block 按 parent scope 分桶；UNSHARD 分桶与 eager 对齐 |
| `6165676ac` | #5151 | `GraphTrainingEngine` 显式拒绝 `parallelism.fsdp_symm_mem_scope is not None` |
| `8b1a0e82e` | #5155 | 新增 `local_compile_perf` skill 文档 |

#### 1.5 checkpoint / EMA / optimizer / metric
| commit | PR | 一行说明 |
|---|---|---|
| `d2a55eeba` | #5114 | **`CheckpointManager`/`TorchCheckpointingManager`/`TorchFTCheckpointManager.__init__` 的 `dataloader=` + `states=` 改为 `extra_states=`**（新增 `_validate_extra_state_keys`，禁止占用 model/optimizer/lr_scheduler/ema 键）；`TrainingEngine.initialize(extra_states=)`、`_initialize_checkpointer(extra_states=)` 同步 |
| `b1a9c594c` | #5170 | `torch_checkpointing` barrier：`TCPStoreBarrierConfig(…port 43001…)` → **`DefaultStoreBarrierConfig`**（复用默认 PG store，需新版依赖） |
| `cf8222804` | #5282 | CPU-offload EMA 直接在 pinned host storage 初始化：`_EMAParamOptimizer(..., *, offload_to_cpu=)`、删 `_init_cpu_offload/_pin_local/_local_view`、统一走 `get_local_tensor` |
| `99c10fbd3` | #5350 | `Optim.step` 首次调用先 `torch.cuda.synchronize()/empty_cache()`；`_run_update` 去掉 `ep_enabled=` |
| `b81852f00` | #5280 | **`Trainer.__init__` 不再调 `model_config.set_sharding_(config.parallelism)`；上游 `set_sharding_` 现已全树不存在**；`Trainer.Config.override`（从 TrainingEngine.Config 上移）+ `config.model = copy.deepcopy(config.model)` 提前到 `apply_overrides` 之前 |
| `caed64e8f` / `ba758b867` / `22e225055` | #5288 / #5172 / #5348 | loss_compare / checkpoint_compat 脚本与 TensorBoard 标量保留（工具侧） |

#### 1.6 torchft
| commit | PR | 一行说明 |
|---|---|---|
| `ff741a973` | #5016 | `TorchFTManager.Config.timeout_ms`（新字段，可由 `TORCHFT_TIMEOUT_SEC` 覆盖）+ engine 里的超时告警 |
| `34814f59a` | #5154 | 每 replica dataloader 只在 load 包含它时才加载：`_load_checkpoint` 内条件剥离 `DATALOADER`，删除子类的 `_states_to_load` 覆写；`enable_ft_dataloader_checkpoints` 时强制 `extra_states[DATALOADER]` |
| `d2a55eeba` | #5114 | 同上 `extra_states=`（checkpoint 契约统一） |

#### 1.7 RL / CI / docs（与本审计无关，仅说明 74 个 commit 的其余部分）
`#4942 #5018 #5046 #5072 #5107 #5154 #5182 #5186 #5188 #5214 #5215 #5217 #5218 #5219 #5232 #5233 #5268 #5269 #5270 #5271 #5273 #5274 #5280 #5289 #5290 #5294 #5298 #5311 #5332 #5340 #5341 #5348 #5349 #5350 #5359 #5368 #5369 #5372 #5374` 等：RL 训练/generator、Terminal-Bench、vLLM、CI 基线刷新、PyArrow/NumPy 约束、README 修复。对四组 patch 无直接影响（RL 侧的 `extra_states`/`rollout` 例外已在上表列出）。

---

### 2. 结论变化

> 只列**结论改变**或**必须换锚点**的行。①=上游已含（可删）；②=上游已含但形态不同（重写）；③=上游仍无（保留，核对锚点）；④=与上游方向冲突（重新设计）。

| 模块 | 原结论 | 新结论 | 依据（commit/PR + `origin/main` file:line） |
|---|---|---|---|
| G1-1 `models/common/aux_loss.py` | ② | **②（机制仍未对齐；但迁移路线变了）**：`set_step_denominator` 仍不存在（0 命中），`inject(raw_sum, *, carrier, denominator)` 仍是逐 forward 传参；**但 `#4772` 提供了 step 级累积的 `BatchWiseLoadBalanceLoss`，且 `#4835` 上游已自带 `SparseIndexerLoss(AuxLoss)`** → 我方 `SparseIndexerLoss` 不必再自研，直接继承上游写法即可 | `#4772` `8b4a1300e`：`models/common/aux_loss.py:137 _reset_accumulators`、`:141 _reduce_token_partials`、`:165 _norm_sum_scores_over_tokens`、`:181 def inject`、`:200` "Step-global routed-token count"、`:256 collect_aux_loss_metrics`、`:298 register_aux_loss_zero_hook`；`models/common/moe.py:609 BatchWiseLoadBalanceLoss`（`:511` 旧的 MicrobatchWise）；`#4835` `fc6ed2d32`：`models/deepseek_v4/compressor.py:327 class SparseIndexerLoss(AuxLoss)`、`models/deepseek_v4/attention.py:137`。锚点迁移：`Module` 现 import 自 `models/base/module.py`（`#5277`） |
| G1-2 `models/common/linear.py`（`BatchedLinear`） | ② | **②（锚点全变）**：`BatchedLinear` 仍 0 命中；`GroupedLinear` 从 `:306` 移到 `:348`，`Linear` 移到 `:34`，`num_linears` `:54`；**新增 `inplace_wgrad_accum`（`:56`，默认 True）**——若按上游形态改写成 `Linear/GroupedLinear`，在 GraphTrainer/compile 下必须置 False | `#5249` `1b0680885`、`#5250` `06dd3abe2`：`models/common/linear.py:34/54/56/348`、`models/common/inplace_wgrad_accum.py`；`models/common/lora.py:52 _LoRALinearMixin`、`:203 _LoRAGroupedLinearMixin`、`:326 get_lora_linear`、`:332 get_lora_grouped_linear`；`Module.Config.param_init` 现于 `models/base/module.py:96` |
| G1-3 `models/common/moe.py` | ② | **②（锚点全部平移，无结论翻转）**：`GroupedExperts` 仍 0 命中；router `Config` 仍**没有** `hash`/`vocab_size`（我方传参会 `unexpected keyword argument`）；`**router_kwargs`、`aux_loss_denominator` 语义不变；新增 `freeze_gate` 字段与 router 侧 `aux_loss` 字段需在重写时避让 | `#5271` `4a4176e97`：`moe.py:192 freeze_gate`；`moe.py:61 RoutedExperts`、`:66-67 w13/w2`、`:69 activation_fn`、`:105 forward`、`:171 TokenChoiceTopKRouter`、`:185 aux_loss`、`:219 _select_experts`、`:232 router.forward`、`:707 MoE`、`:777 MoE.forward`、`:949 register_moe_load_balancing_hook`（原 `:849`）；`config_utils.py:463 make_routed_experts_config`（仍无 `inner_experts`）、`:68` 只 import `AllToAllTokenDispatcher`；`activation.py:140 SqrtSoftplus`、`:173 ClampedSwiGLU`、`deepseek_v4/moe.py:37 DeepSeekV4Router` |
| G1-6 `models/qwen3_5/vision_encoder.py` | ③ | **③（锚点再次平移 + 新风险）**：`compiled_create_block_mask` 在该文件仍不存在，`create_block_diagonal_mask` 调用点从 `:433` 移到 `:437`，`forward` 从 `:392` 移到 `:396`；**`#5307/#5296` 删除全部 `ScaledDotProduct*`，我方的 dense-SDPA 回退目标在上游已无对应物** | `3210be829`(#5266)、`8a75afd90`(#5307)、`90c7fa4b5`(#5296)：`models/qwen3_5/vision_encoder.py:304/396/437`；`models/common/vision_encoder.py:38 compiled_create_block_mask`、`:106 create_block_diagonal_mask` |
| G2-1 `components/checkpoint.py`（EMA checkpoint） | ② | **②（契约变更，重写面扩大）**：`EMA = "ema"` 不变，但 `CheckpointManager.__init__` 现在收 **`extra_states=`**（不再有 `dataloader=`/`states=`），并新增 `_validate_extra_state_keys`；`TorchCheckpointingManager` 同；我方 `Config` 子类 + `dcp_load` 探测那套要按新签名重做 | `#5114` `d2a55eeba`：`components/checkpointer/dcp.py:159 ema:`、`:160 extra_states:`、`:169-170`、`:179 self.states[EMA]`、`:354 _load_checkpoint`、`:405-410` reseed；`components/checkpointer/base.py:41 EMA`、`:229 _validate_extra_state_keys`、`:392 _load_checkpoint`、`:402 _states_to_load`；`training_engine.py:448 ema=self.optim.ema` |
| G2-2 `components/ema.py` | ② | **②（形态再变）**：`class EMA` 从 `:96` 移到 `:113`；`decays :133`、`half_life_fractions :140`、`offload_to_cpu :164`、`buffer_patterns :168`；CPU-offload 改为**构造时直接分配 pinned host**，`_EMAParamOptimizer` 新增 `offload_to_cpu=` 形参，`_init_cpu_offload/_pin_local/_local_view` 被删除（引用旧私有名的 NPU 代码/测试会 AttributeError） | `#5282` `cf8222804`：`components/optim/ema.py:82 offload_to_cpu`、`:113 class EMA`、`:133/:140/:164/:168`、`:36 _EMA_STATE_KEY`、`:41 ema_key`、`:49 parse_ema_key` |
| G2-3 `components/metrics.py` | ② | **②（锚点平移）**：注入点仍是 `Trainer.train_step`（`:332`，`collect_aux_loss_metrics` 在 `:456`）；`MetricsProcessor` `:305`、`log` `:509`、`extra_metrics` `:515`；`parallel_dims` 仍不存在。**`NoColor/Color` 已从 `tools.utils` 移到 `observability.logging`** | `#5277` `6f2225575`：`observability/metrics.py:28`、`:305`、`:355 parallelism_context`、`:509`、`:515`；`trainer.py:332/456` |
| G2-4 `components/optimizer.py` | ③（A ①/B ③） | **③（不变，证据锚点移动；B 方案新增必做项）**：A 部分 `init_optim_state` 仍在 `components/optim/utils.py:29` 且本 74 commits **未改动** `components/optim/{optimizer,utils}.py`；B 的"上游拒绝 replicated compute"证据行从 `dist_muon.py:1488` 移到 `:1517`；`#5279` 新增 `ComputeLayout.compute_dtype`，B 重写时必须转发 | `git diff 3386c6d6f origin/main -- torchtitan/components/optim/` 仅 `ema.py`/`optim.py`；`components/optim/utils.py:29`、`components/optim/optimizer.py:186/225/404/418`；`distributed/flex_shard/dist_muon.py:1517`；`#5279` `210fc60cb` |
| G2-5 `components/validate.py` | ② | **②（文件被搬迁，锚点路径更换）**：`components/validate.py` → **`components/validator.py`**（`#5277`）；`Validator :54`、`validate :146`、`preprocess_inputs` 调用 `:224/:259`；`preprocess_inputs` 声明在 **`models/base/model.py:52`**（原 `protocols/model.py`）；同一 seam 上 `#5247` 给 PP 校验加了 `return_outputs=False` | `#5277` `6f2225575`、`#5247` `aed1d4ee4`：`components/validator.py:54/146/224/241/259`、`models/base/model.py:52` |
| G2-6 `trainer.py` | ② | **②（锚点平移 + 新增两处行为变化）**：`class Trainer :40`、`engine: TrainingEngine :142`、`train_step :332`；`post_dataloading_process`/`pp_forward_backward_step`/`set_step_denominator` 仍 0 命中。**新变化**：`Trainer.__init__` 不再 `set_sharding_`（该符号上游全树消失），`Config.override` 上移到 `Trainer.Config`，`copy.deepcopy(config.model)` 先于 `apply_overrides` | `#5280` `b81852f00`、`#5277` `6f2225575`：`trainer.py:40/76/142/156-158/332/456`；`git grep set_sharding_ origin/main` 0 命中；`components/validator` 新 import |
| G2-7 `distributed/context_parallel.py` | ③ | **③（结论不变；锚点未变，但出现新的可借鉴/可替代路径）**：`context_parallel.py` 与 `cp_attention.py` 在本 74 commits 中**逐字未改**（`git diff` 空），`cp_shard` 仍无、Ulysses 仍是唯一 varlen-CP 路线；**新增 `cp_linear_attention.py`（rank-local CP metadata for linear attention）** 是与我方 `CPVarlenMetadata.from_global` 最接近的上游模式。eager block-mask workaround 锚点行号 `attention.py:707→645` | `git diff 3386c6d6f origin/main -- torchtitan/distributed/context_parallel.py torchtitan/models/common/attention/cp_attention.py` 空；`#5184` `01b1b69e3`：`models/common/attention/cp_linear_attention.py`、`cp_gdn.py`；`attention.py:645 _compiled_create_block_mask`、`:648 create_attention_mask`；`models/common/decoder.py:289/334/393/397` |
| G2-8 `distributed/full_dtensor.py` | ④ | **④（更彻底的删除）**：`full_dtensor.py` 仍 MISSING、`spmd_backend` 仍 0 命中；**patch 依赖的 `ScaledDotProductAttention/ScaledDotProductInnerAttention` 现已被上游整体删除**（`#5307`），替代锚点是 `VarlenInnerAttention`/`FlexInnerAttention` | `#5307` `8a75afd90`、`#5296` `90c7fa4b5`：`git grep ScaledDotProduct origin/main -- torchtitan/` 0 命中；`models/common/attention/attention.py:166 VarlenInnerAttention`、`:284 FlexInnerAttention`；我方 `patches/.../full_dtensor.py:21-26` 仍 import 该符号 |
| G2-9 `distributed/varlen_cp.py` | ③ | **③（不变；可选路线增加）**：`CPVarlenMetadata`/`k_global_gather_indices` 仍 0 命中，`VarlenMetadata`→`VarlenAttentionMetadata`（`attention.py:76`），`AttentionMasksType` 仍无；`_HeadTailLoadBalancer` 私有 API 仍被上游包成 `HeadTailCPLoadBalancer.generate_permutation()`。新增参照：`cp_linear_attention.py` 的 `prepare_linear_attention_cp_metadata(metadata, permutation=…)` 与我方 `from_global(permutation)` 设计同构 | 同上；`distributed/context_parallel.py:50/59/97/317`；`models/common/attention/cp_attention.py:70/96/275/339` |
| G3-1 `graph_trainer/chunked_loss.py` | ③ | **③（不变）**：`chunked_loss.py` 在本 74 commits **逐字未改**，`_gradient_backprop` 仍是实例方法（`:15/:31-37`）、`_ChunkedLossWrapperWithParamGrads.forward` `:74`；但 patch 的另一处锚点 `ReplicateComputation` 从 `simple_fsdp.py:274` 移到 `:391`（`full_dtensor` 仍 0 命中） | `git diff --stat 3386c6d6f origin/main -- torchtitan/experiments/graph_trainer/chunked_loss.py` 空；`#5171` `c29918394`/`#5277`：`simple_fsdp.py:391 class ReplicateComputation`、`:396 param_sharding`、`:535 get_param_sharding` |
| G3-4 `graph_trainer/ep_overlap_shape_queries.py` | ③ | **③（不变，最便宜）**：`ep_overlap_pass.py` 逐字未改，`_collect_token_exchanges :196`、`_custom_meta :111`、`_is_token_exchange_launch :123` 均原位；唯一锚点漂移是 `common_utils._EP_TOKEN_EXCHANGE` 由 `:211` 移到 `:175` | `#4772`/`#5250` 改了 `common_utils.py`：`common_utils.py:175 _EP_TOKEN_EXCHANGE`、`:418/:421`；`ep_overlap_pass.py:111/123/196` |
| G3-7 `graph_trainer/functionalize_recompute_mutations.py` | ③ | **③（不变；上游出现部分替代路线）**：pass 仍不存在（0 命中）；`construct_default_graph_passes` 仍是 `parallelism_context=`（`:442`）→ 我方 `parallel_dims=` 仍 TypeError；`tag_with_memory_policy_pass` 仍在默认 pipeline（`passes.py:82/:235`），锚点可用。**新增**：`#4772` 在 `memory_policy` 里加了 "Pass 1b：in-place mutation MUST_SAVE"，正好覆盖本 pass 的部分动机；`#5023` 把 `cuda_graph_pass` 移出默认 pass | `passes.py:442/495`、`:82/:235`；`registry.py:42/45`；`memory_policy.py:369-385`（Pass 1b，来自 `8b4a1300e`）；`#5023` `bed2b9ea8`；`mutation_utils.py:195` 仍在（#4842 [2/2] 未落地） |
| G3-8 `graph_trainer/graph_trainer_runtime_context.py` | ③ | **③（不变，注入点全部再平移）**：`runtime_context` 在 graph_trainer 全目录 0 命中、`_make_fx_forward_backward_step` 仍不存在、`apply_graph_passes` 仍无 `runtime_context`（`:495`）→ 仍是"静默 warning 后失效"。新注入点：`spmd_graph_builder.py:194 _trace_joint_stage_graph`（原 `:210`）、`:272 _trace_spmd_stage_graph`（原 `:289`）、`:257 apply_graph_passes`；`graph_builder_utils.py` 的调用点 `:155/:243/:295/:308`（原 `:153/:240`） | `bed2b9ea8`(#5023) 改了 `spmd_graph_builder.py`/`graph_builder_utils.py`；`git grep -l runtime_context origin/main -- torchtitan/experiments/graph_trainer/` 空 |
| G3-9 `graph_trainer/memory_policy.py` | ③ | **③（不变，但 fork 代价上升）**：`tag_sac_policy :282`、`force_save_nodes :287`、`_make_default_memory_policy(save_ops=) :123`、`_is_cacheable_effect`（`:35` 导入）均在；`save_input_every_n_layers`/`NodePolicyKey`/`make_node_override_memory_policy` 仍 0 命中；`configs.py` 的 `memory_policy: Literal[...]` 仍在（`:204`），tyro 仍不存在。**新增 Pass 1b（mutation MUST_SAVE）→ 继续 fork 整份 `tag_sac_policy` 会静默丢掉该修复** | `8b4a1300e`(#4772)：`memory_policy.py:369-385`；`memory_policy.py:35/123/282/287`；`configs.py:204`；`git grep tyro origin/main -- torchtitan/` 空 |
| G3-10 `torchft/checkpoint.py` | ① | **①（更应删除）**：`init_cache_state_dict :120`、`_refresh_cached_state_dict :124` 仍在；**但 `__init__` 已改 `extra_states=`（`:88`），且 FT dataloader 改为"load 里含 DATALOADER 才加载"并删除了子类 `_states_to_load` 覆写**——我方 backport 的 `dataloader=`/`states=` 版本现在构造即 TypeError | `#5154` `34814f59a`、`#5114` `d2a55eeba`：`experiments/torchft/checkpoint.py:80/88/100/108-114/120/124/142-149/179+` |
| G3-11 `torchft/manager.py` | ① | **①（不变）**：`maybe_set_all_reduce_hook` 方法体仍与补丁逐行相同，仅行号 `:159-172 → :170-183`；新增 `Config.timeout_ms`（`:62`）与 `process_group_timeout_ms`（`:56`） | `#5016` `ff741a973`：`experiments/torchft/manager.py:56/62/170/186` |
| G3-13 `torchft/trainer.py` | ② | **②（锚点平移 + checkpoint 契约再变）**：`FaultTolerantTrainingEngine :45`、`FaultTolerantTrainer :152`、`apply_overrides :165`、`train_step :337`、`loss_sync_pg :420`、`ft_pg.size() :453`；`_configure_chunked_loss` 仍不存在；`_initialize_checkpointer` 现在收 `extra_states=`（`:131-134`）、`engine.initialize(extra_states={DATALOADER:…})`（`:251-253`）、`config.model = copy.deepcopy(config.model)` 不再 `set_sharding_` | `#5114`/`#5154`/`#5280`/`#5016`：`experiments/torchft/trainer.py:45/88-98/131-134/152/160-165/251-253/337/420/453` |
| G4-1 `flex_shard/dist_muon.py` | ② | **②（不变；注入的 7 个函数未被 #5279 触碰）**：本 74 commits 只有 `#5279` 动 flex_shard，其 hunk 落在 `DistMuon`/`_ParameterComputeLayout`/`_compute_muon_direction`；我方注入的 7 个函数仍在原位（`_build_parameter_redistribution_plan :1087`、`_lower_shard_order_to_strided_shards :1283`、`_validate_shard_order_compute_targets :1318`、`_is_supported_orthogonal_dim0_shard_redistribution :1380`、`_resolve_storage_to_compute_transition :1403`、`_normalize_compute_placement :1762`、`_normalize_storage_placement :1792`）→ 仍是"4/7 被旧 PR 改写 + 误留会静默回退"。别名行号 `_LoweredComputeSharding :913`、`_AxisComputeSharding :917`（原 `:885/:889`）；`_validate_parameter_storage :257` | `git diff 3386c6d6f origin/main -- torchtitan/distributed/flex_shard/`（仅 `#5279` `210fc60cb`，+125/-10）；`dist_muon.py:257/866/877/913/917/1087/1283/1318/1380/1403/1762/1792` |
| G4-2 `flex_shard/optimizer_reshard.py` | ①（可删，S） | **①但必须删（误留会硬报错，不再是 no-op）**：`#5279` 给 `ComputeLayout` 加了 `compute_dtype`（`:196`），`dist_muon.py:294` 会读 `compute_layout.compute_dtype`；我方 patch 用 6ee6b10b 版 `ComputeLayout` 重绑三处命名空间 → 新 pin 上 AttributeError / 静默丢 BF16 计算 | `#5279` `210fc60cb`：`optimizer_reshard.py:139 class ComputeLayout`、`:196 compute_dtype`、`:197 shard_order_by_tensor_dim`；`dist_muon.py:294-300/319/339`；我方 patch `optimizer_reshard.py:204-207` 三处重绑 |
| G4-4 `flex_shard/replica_dedup.py` | ④ | **④（不变；重写清单加一项）**：`#4411` 仍未合并（0 命中），`replica_dedup` 仍 0 命中，`build_dist_muon` 仍不存在；私有钩子 `_initialize_plan :324`、`_apply_update :592`、`reserve_buffers`（`_optimizer_reshard_runtime.py:325`）仍在。**新增必做**：钩子必须转发 `ComputeLayout.compute_dtype`（`#5279`），否则 BF16 重分布路径失效 | `git log 3386c6d6f..origin/main --grep=4411` 空；`dist_muon.py:324/592/1517`、`_optimizer_reshard_runtime.py:325`；`#5279` |
| G4-5 `hf_datasets/text_datasets.py` | ④ | **④（不变；路径再次断裂）**：能力仍无（`pad_segments_to_multiple`/`_pad_segments_to_multiple` 0 命中，`HuggingFaceTextDataset/Loader`、`ParallelAwareDataloader` 仍 0 命中）；**新 pin 上目标模块从 `torchtitan/hf_datasets/text_datasets.py` 搬到 `torchtitan/components/data/text_datasets.py`**，我方 patch 的 3 条 import 里两条路径消失 | `#5277` `6f2225575`：`R100 torchtitan/hf_datasets/text_datasets.py → torchtitan/components/data/text_datasets.py`；`torchtitan/hf_datasets/` 现不存在（`git cat-file -e origin/main:torchtitan/hf_datasets` 失败） |
| G4-6 `hf_datasets/multimodal/mm_collator.py` | ② | **②（不变；路径搬迁 + "静默错装"变为"先 ImportError"）**：文件搬到 `components/data/multimodal/mm_collator.py`；其 `_build_mrope_positions` 形参表（`:189-198`）与 patch 的 `_EXPECTED_PARAMETERS` **完全一致**，所以一旦把 import 路径改对而忘了删 patch，仍会静默装上 2-D 变体；上游契约仍是 1-D（`tokens:(num_tokens,)`、`resets = positions[:,1:] == 0`） | `#5277` `6f2225575`：`R099 hf_datasets/multimodal/mm_collator.py → components/data/multimodal/mm_collator.py`（仅多一个空行）；`components/data/multimodal/mm_collator.py:189-198` |
| **勘误** G4 §3 备注 | 「`dist_muon._ParameterComputeLayout` 在 HEAD 已被删除」 | **该说法不成立**：`_ParameterComputeLayout` 在基线 `3386c6d6f:dist_muon.py:839` 与新 pin `origin/main:dist_muon.py:866` **都存在**（`#5279` 还给它加了 `compute_dtype` 字段）。我方 `extensions/cpu_offload/cpu_offload_muon.py:44` 的 `TYPE_CHECKING` import 不会断 | `git show 3386c6d6f:torchtitan/distributed/flex_shard/dist_muon.py | grep -n "class _ParameterComputeLayout"` → `839`；`git grep -n "class _ParameterComputeLayout" origin/main -- torchtitan/distributed/flex_shard/dist_muon.py` → `866` |

---

### 3. 未变化（结论与锚点都不需要动）

- **G1-4 `models/common/rope.py`** — ③/①/② 全部不变：`split` 在 rope.py 全历史仍 0 命中；`key`/`inverse` 仍在（`apply_rotary_emb:138`、`RoPE.forward:161`、`ComplexRoPE.apply_rotary_emb:257`、`CosSinRoPE.forward:284`、`CosSinRoPE.apply_rotary_emb:348`）；本 74 commits 对该文件只改了 `Module` 的 import 路径（见 §4.1）。partial RoPE 仍由调用方切片（`deepseek_v4/attention.py`）。
- **G1-5 `models/common/token_dispatcher.py`** — ③ 不变，且锚点**逐行未移**：`LocalDispatchMetadata:22`、`AllToAllDispatchMetadata:30`、`BaseEPTokenDispatcher:181`、`AllToAllTokenDispatcher:235`、`_sync_token_count_exchange:283`、`_permute:535`、`combine:595`、`EPDispatchMetadata:814`、`DeepEPTokenDispatcher:820`；`absorb_router_scores`/`routed_scores*` 仍 0 命中；`config_utils.py:68` 仍只 import `AllToAllTokenDispatcher`。本 74 commits 只改了一句 docstring（`#5101`：FullAC 不再保存 DeepEP 通信）。
- **G2-9 之外的 CP 相关**：`distributed/context_parallel.py`、`models/common/attention/cp_attention.py`、`models/common/decoder.py`（除 import 路径）在本 74 commits 中逐字未改。
- **G3-2 `ep_chunk_concretization.py`（④）**、**G3-3 `ep_forward_accumulation.py`（④）**、**G3-6 `ep_shape_live_out.py`（④）** — `ep_chunk_pass.py` 仍 MISSING，`ep_eager_chunk.py` 仍在，`_split_live_out_users`/`_materialize_live_out`/`populate_chunk_dim_metadata_pass`/`mark_chunk_dynamic_dims` 仍 0 命中；`ep_pass_utils.py`/`ep_eager_chunk.py` 在本 74 commits 中逐字未改。
- **G3-5 `ep_ready_nodes_dedup.py`（①）** — `ep_overlap_pass.py:514 _ready_nodes`、`:526 selected: set[fx.Node]` 与补丁逐行等价，本 74 commits 未改该文件。
- **G3-12 `torchft/optimizer.py`（②）** — `__init__:35`（无 `ft_manager=`）、`configure_fault_tolerance:55`、`_refresh_cached_state_dict:68`、`step:84`、`zero_grad:96`、`TorchFTOpt:102` 全部与基线报告一致（本 74 commits 只删了 1 行）。
- **G4-3 `flex_shard/_optimizer_reshard_schedule.py`（①）** — 我方注入的 5 个定义未变（`_build_dim0_shard_redistribution_plan:585`、`_RedistributionGroup:741`、`_redistribution_group:900`、`_dtensor_storage_regions:950`、`_tensor_region_intersection:414`；`_device_mesh_ranks:894`，仅 `#5279` 的 `compute_dtype` 落在未注入的 `_RedistributionBucketPlan`/`_build_bucket_plans` 上）；仍可整体删除。
- **G4-5/G4-6 的"能力仍无"部分** — `pad_segments_to_multiple`、`HuggingFaceTextDataset/Loader`、`ParallelAwareDataloader`、`MMSamplePacker`、`replica_dedup`、`ep_chunk_pass` 在新 pin 上仍是 0 命中。

---

### 4. 新增风险（74 commits 带来、上一轮审计不可能知道）

#### 4.1 目录搬迁（#5277 `6f2225575`）——影响面最大
| 上游变化 | origin/main 证据 | 我方受影响面（`grep -rn --include=*.py` 计数） |
|---|---|---|
| `protocols/` → `models/base/` | `models/base/{model,module,sharding,state_dict_adapter}.py`（`protocols/` 已不存在） | `torchtitan.protocols` **37 处**（其中 `torchtitan.protocols.model` 10 处）：`models/deepseek_v4/{mhc,compressor,mtp,attention,__init__,config_registry}.py`、`deepseek_v4_1/*`、`deepseek_v3_2/__init__.py`、`extensions/adaptor/torchao/converter.py`、多处 tests。所有 patch 里对 `protocols.module.Module` / `protocols.model.BaseModel` / `protocols.state_dict_adapter` 的引用在 bump 时全部 ImportError |
| `hf_datasets/` → `components/data/` | `components/data/{text_datasets,multimodal/mm_collator,multimodal/utils/*}.py` | `torchtitan.hf_datasets` **17 处**：`override/deepseek_v4/chat_dataloader.py`、`models/deepseek_v4/{chat_dataset,config_registry}.py`、`models/qwen3_5/config_registry.py`、tests |
| `components/validate.py` → `validator.py` | `components/validator.py`（119 行同构重命名） | 我方 patch 自身 `patches/torchtitan/components/validate.py` + `patches/torchtitan/__init__.py:7` 的 import |
| `models/utils.py` 拆分 | `models/common/moe_state_dict_adapter.py`（R074 from `models/utils.py`）、`models/common/flops.py` | `torchtitan.models.utils` **6 处**：`models/deepseek_v4/{model,__init__}.py`、`deepseek_v4_1/{model,__init__}.py`、`deepseek_v3_2/__init__.py`、`extensions/adaptor/torchao/converter.py` |
| `models/common/attention/kda.py` → `linear_attention.py`；`KDAAttentionMetadata` → `LinearAttentionMetadata` | `attention/__init__.py:29/32/34/40`；`lib` 侧新增 `linear_attention.py`、`cp_linear_attention.py` | 我方无直接 import（0 处），但 `AttentionMetadata` 联合类型变了，任何按类型分派的 NPU metadata 代码需同步 |
| `tools.utils.Color/NoColor` → `observability.logging` | `observability/metrics.py:28` | G2-3 metrics patch 若引用 `utils.NoColor` 会 AttributeError（需按 patch 实际内容核对） |

#### 4.2 契约/签名变化（patch 形态直接失效）
1. **Checkpoint `extra_states=`**（`#5114` `d2a55eeba`）：`CheckpointManager`/`TorchCheckpointingManager`/`TorchFTCheckpointManager` 全部改为 `extra_states=`，且**新增键校验** `_validate_extra_state_keys`（禁 model/optimizer/lr_scheduler/ema 键）；`TrainingEngine.initialize(extra_states=)`、`_initialize_checkpointer(extra_states=)`。→ 直接击中 G2-1、G3-10、G3-13 三个 patch 的 backport 主体。
2. **`torch_checkpointing` barrier API**（`#5170` `b1a9c594c`）：`TCPStoreBarrierConfig` → **`DefaultStoreBarrierConfig`**；同时 `#5277` 把 `torch_checkpointing` 从 PyPI 改为 git commit pin（`.ci/docker/requirements.txt:15-16`，"PyPI 0.1.0 predates `default_resharder`"）。→ NPU 环境若仍装 PyPI 0.1.0，`torch_checkpointing.py` 直接 ImportError。
3. **`set_sharding_` 消失 + `override` 上移**（`#5280` `b81852f00`）：上游全树 `set_sharding_` 0 命中；`Trainer.Config.override` 新增，`config.model = copy.deepcopy(config.model)` 前移到 `apply_overrides` 之前。→ 任何依赖"trainer 侧先 set_sharding_ 再 override"的 NPU 逻辑（含 `TrainerEx`/`config_registry` 的 `Trainer.Config` 投影）都要重新核对顺序；G3-13 里 torchft 的 `apply_overrides` 时序描述已过时。
4. **`TrainingEngine.Config.dist_moe` → `dist_moe_runtime`**（`#5046` `826d018f5`）：配置字段改名；`GraphTrainingEngine` 同步使用。
5. **`ScaledDotProduct*` 全删除**（`#5307`/`#5296`）：除 G2-8 的 patch import 外，我方任何"dense SDPA 回退"假设（`vision_encoder` mask 打标等）在新 pin 上无上游承接；上游默认走 `VarlenInnerAttention`（`:166`），NPU 需确认 varlen kernel 能力。
6. **`Linear.Config.inplace_wgrad_accum=True` 默认值**（`#5249/#5250`）：trace/compile 下打开会在 forward 直接 raise；上游 GraphTrainer 走 `common_utils.disable_inplace_wgrad_accum(graph_model)`（`common_utils.py:65`，由 `configs.py:357` 的 `to_graph_trainer_config` 调用）。→ 我方 `models/deepseek_v4/config_registry.py:491` 包装的 `to_graph_trainer_config` 会自动继承这一步，但 `tests/unit_tests/test_dsv4_muon.py:187` 对上游函数**源码**的断言需要复核；NPU 自建 Linear 子类若拒绝 in-place 累加也必须显式置 False。
7. **`ComputeLayout.compute_dtype`**（`#5279` `210fc60cb`）：G4-1/2/4 三个 patch 的公共风险；`optimizer_reshard.py` 的重绑从"no-op"升级为"会 AttributeError 的硬错误"。
8. **EMA CPU-offload 重构**（`#5282` `cf8222804`）：`_EMAParamOptimizer(..., offload_to_cpu=)`；`_init_cpu_offload`/`_pin_local`/`_local_view` 全部删除，改 `get_local_tensor` + 构造期 pinned 分配。→ 引用旧私有名的 NPU 测试/扩展（G2-2 重写要点里提到的 `TrainerEx`、EMA UT）会断。
9. **`Optim.step` 首次调用 `cuda.synchronize/empty_cache`**（`#5350` `99c10fbd3`）：`torch.cuda.is_initialized()` 有守卫，NPU 上应无害，但 `_run_update` 去掉 `ep_enabled=` 后，若 NPU 侧有 `override/common/optimizer.py` 覆写同一函数需同步签名。
10. **torchft `timeout_ms` 新字段 + FT dataloader 条件加载**（`#5016`/`#5154`）：`FTManagerEx`/`TorchFTOptimizersContainerEx` 的 Config 注入与 `extra_states[DATALOADER]` 装配方式都要改。

#### 4.3 语义/行为变化（不会 import 失败，但会静默改变数值或让 patch 失效）
1. **`memory_policy` Pass 1b**（`#4772` `8b4a1300e`）：`tag_sac_policy` 新增"in-place mutation 与其消费者一律 MUST_SAVE"。→ G3-9 的 fork 副本会**静默丢掉该修复**（重算 mutation 会重复施加副作用）；同时这正是 G3-7 `functionalize_recompute_mutations` 想解决的问题的一半，建议把 G3-7 的定位从"补语义"改为"补 SPMD/非 GraphPP 路径中仍缺的部分"，并跟踪 `#4842 [2/2]`。
2. **aux-loss 归一化语义**（`#4772`）：`AuxLoss.inject` 的 `denominator` 文档明确为 "Step-global routed-token count"，并新增 rolling 计数实现。→ G1-1/G1-3 的重写不必再自造 step 级分母，直接用上游 `BatchWiseLoadBalanceLoss`/`aux_loss_denominators` 通道即可（`trainer.py:349-401`、`training_engine.py:565-566` 未变）。
3. **上游已自带 `SparseIndexerLoss(AuxLoss)`**（`#4835` `fc6ed2d32`）：DSV4/CSA 路径的 indexer 蒸馏 loss 已上游化（`compressor.py:327`、`attention.py:137`、`register_aux_loss_zero_hook` 挂在 DSV4 optimizer post-build hook）。→ 我方 `models/deepseek_v3_2/model.py:191 SparseIndexerLoss(AuxLoss)` 应改为对齐上游实现（这是 G1-1 从"自研 + backport"降级为"配置即可"的关键依据）。
4. **FullAC 语义**（`#5101` `448b9c8be`）：FullAC 现在是 `RegionAC(save_regions=[])`，重放不再重复执行 aux-loss 副作用，`aux_loss.py` 里 "FullAC over-counts 2x" 的限制段被删除。→ 我方 `_should_run_forward()`（aux_loss patch）与 `override/deepseek_v3_2/sparse_attn/ascendc.py` 的计数假设在新 pin 上需要重测；`FullAC` 要求 `preserve_rng_state=False` 且禁用 `debug=True`（`activation_checkpoint.py`）。
5. **graph_trainer 默认 pass 变化**（`#5023` `bed2b9ea8`）：`cuda_graph_pass` 移出默认列表，`use_cuda_graph` 改由 `config.training.disable_cuda_graphs` 推导；`final_inductor_compile_passes` 语义随之改变。→ G3-7 里"插在 `tag_with_memory_policy_pass` 之后"的 pipeline 位置仍成立，但任何依赖 `disable_passes=["cuda_graph_pass"]` 或 `use_cuda_graph=` 形参的 NPU 调用点会失效（`compile_time_passes` 已删除 `use_cuda_graph` 形参）。
6. **`GraphTrainingEngine` 拒绝 `fsdp_symm_mem_scope`**（`#5151` `6165676ac`）：`parallelism.fsdp_symm_mem_scope is not None` 直接 ValueError。→ NPU 的 GraphTrainer recipe 若开了该字段会启动失败（我方仓库当前 0 命中，属预防性提示）。
7. **Vision encoder CP-invariant**（`#5266` `3210be829`）：新增 `spmd.mutate_type(..., "cp", R→I)` 断言，vision encoder 只 dp_shard。→ G1-6 patch 的 mask/owning-tensor 逻辑需在 CP>1 下重测。
8. **`attn_gym` 变为可选**（`#5231` `913413cca`）：`models/common/attn_gym.py` 提供可选 import；`#5225` 的 GDN/KDA cuDNN chunk kernel 依赖 Blackwell，NPU 侧不要照搬。

#### 4.4 依赖 pin
- `spmd_types` → commit `2ae49fce`（`#5230` `49a07abcf`，改 `.ci/docker/requirements.txt` 与 `pyproject.toml`）；现文件里 pin 为 `a1b86dea…`。上游新代码使用 `spmd_mesh_size`/`assert_type_like`/`convert`/`local_compile` 等新 API。→ 我方 6 处 `spmd.assert_type(` 调用在当前上游仍有效（`components/loss.py` 仍在用），但 **NPU 侧 pin 的 spmd_types commit 必须与上游一致**，否则 patch 里的 spmd 辅助函数会缺符号。
- `torch_checkpointing` 改 git pin（见 4.2-2）；`torch_remat` 为必需依赖（`pyproject.toml:57-58`）。

---

### 5. 升级排序建议（相对上一轮的增量）
1. **先做 4.1 的机械搬迁**（`protocols→models/base`、`hf_datasets→components/data`、`validate→validator`、`models/utils` 拆分）——它同时是 37+17+6 处 import 的前提，先做可以让后面所有 patch 的报错变干净。
2. **再改 checkpoint 契约**（`extra_states=`、`_validate_extra_state_keys`、`DefaultStoreBarrierConfig`）——G2-1/G3-10/G3-13 一次改完，避免三次返工。
3. **flex_shard 组先删 G4-2（`optimizer_reshard.py`）**，否则 `#5279` 的 `compute_dtype` 路径必崩；G4-1/G4-3 随后整体删除，G4-4 按 `components/optim/optimizer.py:186 DistMuon(Config, params=…)` 重挂钩子并转发 `compute_dtype`。
4. **G3 组先删 10/11（已 ①）**，再决定 2/3/6（④）是否整体删除，最后处理 7/8/9（③，锚点已更新）；9 建议改为薄封装上游 `tag_sac_policy`（否则丢 Pass 1b）。
5. **G1 组**：`moe.py`/`token_dispatcher.py` 按新行号重写（锚点本身没动，工作量集中在形态）；`aux_loss.py` 改为直接用上游 `AuxLoss` + `BatchWiseLoadBalanceLoss`，`SparseIndexerLoss` 对齐 `deepseek_v4/compressor.py:327`。
6. **G2-7/G2-9（varlen CP）**：除了原有的"自研 vs Ulysses"二选一，本轮新增第三条参照——`models/common/attention/cp_linear_attention.py` 的 `prepare_linear_attention_cp_metadata(metadata, permutation=…)`，与 `CPVarlenMetadata.from_global` 同构，建议在重写时对齐它的函数签名（返回 rank-local metadata）。
