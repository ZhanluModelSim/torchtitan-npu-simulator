# TorchTitan v0.4.0 升级规划：DSV4 / DSV4.1 eager 优先

- Status: proposed
- Created: 2026-11-04
- Last updated: 2026-11-05
- Related Issue/PR: 尚未创建

## 当前执行状态（2026-11-05）

**最新验收范围变更：2026-11-05 用户要求两模型 8 卡长程改为 50 步，随后明确「续训测完 Muon 就够了」。当前必需范围为 8 卡 50 步训练及两模型 Muon 双卡完整续训；不再追加 8 卡 AdamW 恢复。此前已完成的 100-step、AdamW 续训与跨版本诊断保留为历史证据。模型规模、eager、FullAC 与精确判定不变。**

本轮指定的「两模型 8 卡 50 步训练 + Muon 双卡完整续训」已完成，训练证据固定为 C16；其后的 runtime 带宽接线修复仅有 CPU 与静态检查证据。整轮升级的历史 golden 和全仓 Lint 仍有未通过项。下文保留原规划与逐阶段证据，当前结论以本节和最新验证记录为准。

| 项目 | 当前证据与未完成项 |
| --- | --- |
| 软件栈 | 固定目标上游；Torch 2.15 / 固定源码 Torch-NPU wheel 已在两台隔离环境完成正式 requirements 安装、版本来源检查及 pip check。原 2.14 配对因缺少目标上游必需 API 未继续采用 |
| 文本 eager 主路径 | C16 同 head 核心 NPU 矩阵 14/14 通过，三项 HCCL 边界通过；覆盖两模型单卡、EP2/FSDP2、GA2/padding、无 AC/FullAC、V4 MTP 和 V4.1 Host Engram；融合 V4.1 路径保留 reference Sinkhorn 边界 |
| 新版续训 | C16 两模型 Muon EP2/FSDP2 GA2 连续 4 步 vs 2→4 恢复，指标与完整状态精确比较通过；包括 momentum、Engram/SparseAdam 及真实 Gloo metadata / HCCL WORLD 核验。按最新用户指令不再追加 AdamW 续训，已完成的历史对照保留 |
| V4 裁剪 Flash | C16 同 head 50 步已完成，raw TB 的 loss/grad_norm 覆盖 1..50 且有限，step50 indexer audit 通过（21 pairs / 84 projections）；按用户范围取消最终保存/AdamW 恢复，runner exit143 为有记录的主动取消。跨 head 18,520-key 恢复对照作为历史诊断单列 |
| V4.1 裁剪 Flash | C16 的 50 步 raw TB 指标完整有限、step50 indexer audit 通过；按最新用户范围有意取消多余最终保存/恢复，原 runner exit143 单列，不记作完整两阶段 suite 返回0。使用 reference Sinkhorn、融合 attention/indexer、65,536 行 Engram；历史 C12 100 步完整续训通过及 C15 保存失败均留证 |
| 严格数值门槛 | 三个历史 30-step golden 仍失败且未刷新；旧适配基线诊断也未通过历史 golden。V4 单卡 GA2 反事实已复现旧侧 30 步 loss / grad_norm；原默认路径、其他 golden 与最终全状态仍不能据此称为比特一致 |
| 全仓检查 | 完整 Lint 未通过；目标 Pyrefly 最新剩余 30 项后置能力与其他模型接口错误。补充公共入口 CPU 检查通过，不等于其他模型训练兼容 |

## Context

本次升级涉及公共模型接口、损失归一化、重计算和插件接入点，不能仅修改依赖版本。分析依据为本地固定版本的 Git 差异与源码，不代表运行验证通过。

| 项目 | 固定版本 |
| --- | --- |
| 起始 torchtitan-npu | `c422c623bdc15b967e93b8b46972b8d876e785ef` |
| 当前适配上游 | `ebdfc8c2a24d89d89c99c0ef96ed8ac149864175` |
| 目标上游 v0.4.0 | `45e33cd413b64cb37a7840cfd3c9765d3c639ae1` |
| 上游差异 | 317 个提交、990 个文件；基线为目标的祖先 |

起始静态扫描发现 NPU 生产代码的 60 处 import 引用了目标版本已删除或搬迁的 15 个模块，不包含同模块内符号、参数及行为变化。上一轮升级见 [dc3985 → ebdfc8 计划](2026-10-31-upgrade-torchtitan-plan.md)。

## Proposal

### 首阶段边界

- 验收 DSV4、DSV4.1 文本 CPT eager 训练与新版续训，按单卡小模型、EP2/FSDP2、8 卡裁剪 Flash 递进。
- 裁剪 Flash 保留完整层结构、16 experts；V4 为 43 层，V4.1 为 40 层。完整 Flash 规模验收后置。
- DSV4 包含已有 MTP；V4.1 文本实现没有 MTP，本轮不新增。
- V4.1 包含 Host Engram、稀疏参数更新、Full/Reindex/Reuse 链和状态恢复。
- TP、CP、PP 均固定为 1。不安排 PP、LoRA 适配；CP、多模态、SFT 后置。
- 保留 HF 权重加载，只要求新版 DCP 保存与续训，不兼容升级前完整 DCP 或 optimizer 状态。
- CLI 迁移已有独立计划，只作为配置入口前置依赖，不重复拆解或实现旧 CLI 兼容层。
- 只修改 NPU 插件仓，不修改上游 checkout，不维护旧上游兼容分支。
- eager 指模型和 loss 不进入 Dynamo/AOT 编译执行；关闭 local compile、GraphTrainer 和 CUDA graphs，`aot_eager` 不计作纯 eager。

### 上游变化与接口迁移

| 领域 | 上游变化 | NPU 适配决定 |
| --- | --- | --- |
| 依赖与目录 | 基类迁到 `models.base`；attention、linear、MoE 拆为子包；数据和 recipe 重组 | 同步 import、patch 目标、注册入口与依赖声明 |
| 模型构造 | 统一 `feed_forward`；构造期完成更多分片配置 | 同批迁移字段、工厂、sharding、FSDP、Muon 筛选、HF 映射和 override FQN |
| Linear / Router | `RouterGateLinear` 改为 `HiMidLoLinear`；新增原位 WGRAD | 首阶段关闭原位 WGRAD，HiMidLo 使用非 CUDA FP32 路径，优化另行验证 |
| 激活 | 输入改为 `gate_up[...,2,F]` | 同步小算子、TileLang 和 CANN 路径，保留 DeepSeek clamp 公式，不直接替换为上游同名但公式不同的激活 |
| 输入和 loss | preprocess 返回四项；loss / routing 分离逐 objective 计数 | 传递 `loss_kwargs`、`global_loss_token_counts` 与 routing denominators，覆盖 padding 和梯度累积 |
| MTP | 各深度分别归一化；chunked loss 累积 dtype 改变 | 统一 predictions / labels tuple，保留 mHC MTP 和本地 chunked loss；计算变化独立验收 |
| Attention | backend-owned metadata、DSV4 packed documents | 复用新接口，保留 NPU 多压缩几何 key、文档边界与 V4.1 复用语义 |
| MoE / EP | 新 `padding_mask_T`、dispatcher 返回值及 batch-wise aux loss | 迁移输入与统计协议，保留 pre-W2 score absorption，避免重复乘 score |
| AC | FullAC / SelectiveAC 迁到 `torch_remat` region | 先无 AC 建立基线，再接 FullAC，核验 router、aux loss 和 Engram 重算副作用 |
| 运行时与优化 | 新 GradientClipper、分布式初始化、GC / accelerator API | 接入 sparse norm 扩展点，移除失效调用，收缩整段上游复制 |
| Checkpoint | `extra_states`、HF 量化布局 keyword、多副本 EMA 状态 | 首阶段 native DCP；更新 adapter 签名；不新增旧格式转换层 |

关键上游变更包括 `#5345`（feed-forward）、`#5335/#4897`（metadata）、`#5055`（MTP）、`#4893/#5101`（remat）、`#5394`（clipper）、`#5063`（HF 布局）及 `#5417`（mHC）。上游源码以目标 SHA 为准，插件调用点见 [V4 模型](../../../../torchtitan_npu/models/deepseek_v4/model.py)、[V4.1 模型](../../../../torchtitan_npu/models/deepseek_v4_1/model.py)、[V4 MTP](../../../../torchtitan_npu/models/deepseek_v4/mtp.py) 和 [训练扩展](../../../../torchtitan_npu/extensions/torchtitan/trainer.py)。

### 必须保护的语义

- 保留 pre-W2 路由分数乘法及 BF16 舍入、DeepSeek clamp、partial RoPE、BatchedLinear 和 Engram ownership。
- `_grouped_mm` 接口仍匹配，复用现有 kernel，验证空专家、零 token 和反向边界。
- 两模型通过共享的插件侧 `_get_attention_metadata` 保留 geometry-keyed 构造；复用上游 preprocess，移除被替代的全局 stamping 和整段 Decoder 覆盖。
- eager mask patch 必须绑定拆包后的真实 attention 子模块，不能只修改包对象。
- 本地 mHC 已等价实现上游修正的混合轴，不重复改动计算。
- Lightning Indexer carrier 仅用于 autograd teacher 通道，不能读取其值作为上游新 SparseIndexerLoss 的 logits。保留 teacher pair、排序与既有缩放，单独核验 GA / EP 下梯度尺度。
- MTP tuple 要求在起始上游已存在，本地 list 输出是既有定制差异；本次新增的是逐 objective 计数与归一化，不能混淆归因。

## Migration plan

| 工作包 | 实施内容 | 完成条件 |
| --- | --- | --- |
| T0 版本与依赖 | requirements、pyproject、CI 固定目标 SHA；同步 attn-gym 0.0.17、spmd_types 0.3.1、torch_remat 0.2.2、torch_checkpointing 0.1.1、renderers 0.1.12 | 依赖解析无冲突、导入来源为目标 SHA；保留现有 torch / torch_npu 配对并实际核验 |
| T1 公共接口 | import、hook、patch 接线；feed_forward、新工厂、激活与 selector；新版 parallelize 参数 | 两个文本小模型与 HF adapter 可构造，关键 override 命中；未启用图模式不阻断导入 |
| T2 metadata 与 loss | 四项 preprocess、loss/routing counts、共享 geometry 扩展、V4 MTP | packed、padding、多压缩比和各 MTP 深度局部前反向通过 |
| T3 单卡 eager | 参考与融合算子分别验证，V4.1 小 Engram table；修复 mask 接线、关闭 local compile | 前向、反向、参数更新完成，各路径预期梯度产生，无误入编译执行 |
| T4 EP/FSDP 与 remat | dispatcher 协议、dense/expert mesh、Host ignored 参数、clipper、FullAC | EP2/FSDP2 无死锁，score 顺序、ownership、replica reduction 正确；无重复稀疏更新 |
| T5 HF 与新版续训 | adapter keyword / FQN、native DCP 完整状态、Engram SparseAdam | HF 加载 / 回环正确；恢复后的 loss、grad_norm 和最终训练状态一致 |
| T6 8 卡 Flash | 43/40 层、16 experts、EP8/FSDP8、4K 文本；V4.1 独立文本配方与 Engram 链 | 短冒烟后完成 50-step 训练与 indexer 审计；续训以 Muon 双卡完整对照验收（依最新用户指令），仅声明裁剪规模通过 |

依赖顺序为 `T0 → T1 → T2 → T3 → T4/T5 → T6`。HF adapter 静态迁移随 T1 完成，真实分布式续训在 T4 后验收。

先使用 AdamW；Engram 使用现有虚拟优化器和 SparseAdam。融合 attention/indexer 梯度完整后再验证 Muon/NovaSwap，不用缺少 indexer 梯度的参考路径验证 Muon。

### 后续新特性

| 特性 | 后续适配边界 |
| --- | --- |
| Sparse indexer loss | normalized KL、mass weighting 与现有 fused teacher 通道独立比较，不在接口迁移中改目标 |
| MoE balancing / Dist-MoE | aux loss 接协议但保持既有启用策略；Dist-MoE 单独适配 NPU 通信、容量和反向 |
| local compile / Linear 优化 | 逐 region 验证 RoPE、SwiGLU、loss、原位 WGRAD、HiMidLo 加速 |
| CP | 新 hook、backend 名称、metadata 分片、压缩文档、carry、复用语义 |
| GraphTrainer / overlap | 新 engine/runtime、GA、图 pass、paged stash、auto-overlap 与 CV parallel |
| 量化 | 审计旧 Float8 删除与 MXFP8 / NVFP4 扩展对 QAT、量化通信、权重和 remat 的影响 |
| EMA / checkpoint backend | 多副本状态、offload 初始化与 TorchCheckpointingManager 原生 HF 加载 |
| 数据 / SFT | streaming 恢复、packing/padding、子数据集 seed、generation tags、工具调用和 tokenizer |
| 新模型 / RL | Muse-Spark-Open、多模态和 RL 调度 / 生成扩展不自动纳入支持 |

其他已有模型做共享接口影响审计和导入回归，不扩大为全部模型训练验收。PP、LoRA 不列入本轮任务。

## Acceptance criteria

- CPU：配置、override 命中、geometry 隔离、激活公式、HF 映射、MTP counts 和 checkpoint 状态归属。
- 单卡：两个模型至少 3 步纯 eager；V4 覆盖 MTP；V4.1 覆盖 Engram 关闭与开启。
- 两卡：EP2/FSDP2，覆盖空专家、零 token、GA、padding、无 AC / FullAC 和稀疏更新。
- 续训：连续 4 步，对比第 2 步完整保存后恢复至第 4 步；逐值精确比较后续 loss、grad_norm 和最终状态，禁用最终 model-only 保存。
- Golden：保留 V4 单卡、V4 EP2/FSDP2、V4.1 EP2/FSDP2 的 30-step 精确比较，不以冒烟替代。
- 8 卡：裁剪 Flash 50-step 训练与 indexer 审计；不再追加 AdamW 续训。

数值规则遵循 [精度对齐开发指南](../../../../docs/developer_guides/numerical_alignment.md)。纯接口、metadata、布局迁移要求比特一致；MTP 分母、chunked loss 累积 dtype 等计算变化分别记录原因、参考实现和结果，采用既有近似阈值，不放宽或直接刷新 golden。LI 梯度独立检查，不能以 CE loss 正常下降替代。

完整 Lint 使用会话指定的构建环境与既定全量 pre-commit 命令。共享 Pyrefly checkout 仍为旧版，另用隔离配置明确指向目标 SHA；不修改共享 symlink，也不把旧源码检查计为目标版本通过。

服务器、容器、CANN、tokenizer 和临时产物位置由本轮执行记录管理，不写入此 Note。仅清理本轮可归属的临时产物；原始数据、其他任务和复核所需证据不纳入清理。

## Consequences

- 当前为实施中、未验收状态，保留 `proposed`，不晋升为已完成升级；各项证据和阻塞见下节。
- 新 DCP 使用新 FQN / 训练状态；旧完整 DCP 不提供兼容保证，外部 HF 加载继续保留。
- 首阶段关闭新编译和 WGRAD 优化，避免将性能特性引入基础迁移；后续启用需独立验证。
- 上游接口迁移与 NPU 数学定制分别管理；已上游化代码仅在证明等价后删除。
- CLI 为外部前置依赖；接口尚未就绪时报告具体阻断，不绕过模型、loss 或训练状态验证。

## Implementation record — 2026-11-04

### 依赖与运行环境修正

目标上游发布验证 workflow 固定 `torch==2.15.0` 和 `torchvision==0.29.2`。原发布文档示例不足以证明原 NPU 配对仍兼容。实际检查发现，原 Torch 2.14 环境缺少上游入口导入的 `analyze_pipeline_activation_liveness` 和 `PipelineStageInfo`；即使本轮不使用 PP，也会经过该导入链。

插件暂时保留原 Torch / Torch-NPU 配对，不能据此宣称安装可用。已查询的公开索引未发现 Torch-NPU 2.15 配套包；不通过伪造缺失符号、复制上游 pipeline 实现或修改目标 checkout 绕过问题。T0 仍需在可用 NPU 软件栈上确认兼容方案。

本地隔离环境使用实际 Torch 2.15 CPU 和目标上游 SHA，可正常导入上游 Trainer。该环境不代表 requirements 中的 NPU 组合。完整 NPU 包导入缺少 CANN `libhccl.so`；两台授权服务器及 tunnel 别名均未完成 SSH 握手，未进入容器、安装依赖或启动训练，也未干预现有设备进程。

### 已实施与待验收

| 工作包 | 已实施 | 剩余验收 |
| --- | --- | --- |
| T0 | 上游 SHA 与五项附加依赖统一固定，保留原 NPU 配对 | 解决 Torch API / Torch-NPU 配套，实际安装与完整包导入 |
| T1 | 目标文本路径 import、feed_forward、工厂、HF adapter、单输入激活、ConfigLoader；关闭 eager 的编译入口；增加最终 override 后的模型准备协议 | CANN 下完整配方与真实 override 命中，其他模型导入回归 |
| T2 | 共享 geometry metadata、四项 preprocess、MTP tuple 与逐 objective counts | packed/padding/GA 的设备数值、融合 teacher 通道及 chunked loss |
| T3 | 新增参考和融合 Python 配方，明确 paired sparse attention / LightningIndexer | 两模型单卡训练；Engram 开关；LI 梯度有限性、非零性和参数更新 |
| T4 | dispatcher 返回值、score 顺序、FSDP 参数与 optimizer-owned sparse clipper；FullAC 配置入口 | HCCL EP2/FSDP2、空 token、GA、FullAC 重算副作用和 Engram ownership |
| T5 | HF FQN/签名、新版 DCP extra states 与完整状态比较器 | 真实分布式续训、数据游标、Engram SparseAdam 与全状态一致性 |
| T6 | 两个 8 卡裁剪 Flash 配方：融合算子、FullAC、100 步；第 98 步恢复到第 100 步 | 设备冒烟、100-step 数值与续训，全部未运行 |

新版集成入口见 [测试说明](../../../../tests/integration_tests/README.md)、[用例矩阵](../../../../tests/integration_tests/v04_eager.py) 和 [Python 配方](../../../../tests/integration_tests/v04_eager_config.py)。保留三组原 30-step golden 与精确比较；没有刷新基线或放宽阈值。新增续训 runner 先保留连续训练的完整最终 DCP，再要求恢复流程重新生成最终状态，比较全部 tensor、dtype 和保存值。

EMA、旧量化转换和模型编译尚未迁移，入口显式拒绝不支持配置，避免静默忽略；GraphTrainer 和旧 CP backport 不再作为 eager 自动加载的前置条件。后置功能不能据此视为已适配。

### 本地验证边界

使用目标上游、Torch 2.15 CPU 与当前未提交代码复跑，下列 53 项通过：

- 模型契约 7 项：小模型 V4 MTP、V4.1 前反向，metadata 构造与选择，逐 objective CE 梯度 oracle，以及两模型 HF 权重回环。
- V4.1 attention 跨层复用 8 项。
- 运行时组件 15 项：ConfigLoader / Trainer 准备时序 5 项、sparse clipper 3 项、FSDP 参数接线 6 项、单进程 Gloo native DCP 恢复 1 项。
- MoE 与工具 23 项：单输入激活、pre-W2 score 与 BF16 舍入、dispatch 排列、命令构造、checkpoint 指标和完整状态比较的正反例。

模型和运行时测试隔离了 NPU 包自动初始化；运行时测试还隔离了 `torch_npu` 导入，未模拟目标上游 pipeline API。部分算子测试使用 CPU 参考函数检查接口。V4.1 前反向测试显式移除了 Engram；FSDP 测试记录分片调用，不执行 HCCL。这些结果只证明相应组件契约，不能宣称 NPU 包导入、Host Engram、融合内核、完整 metadata 数值、FullAC、分布式训练或设备续训通过。

额外配方探针隔离 CANN profiler 后构造了 10 个真实 Python 配方；其中三个无 NPU override 的配方完成模型准备，V4 EP2 专家权重声明为 `EP: S(0)`。其余配方的 Engram virtual / fused override 导入仍在 `libhccl.so` 处失败。Flash 工厂结构确认 43/40 层、MTP 1/0 和 EP8 + FullAC；这不是实际算子命中或分片训练验证。

完整 Lint 已按指定构建环境和命令执行并复跑。格式、Ruff、拼写、结构检查与 architecture boundary 均通过；OAT hook 因没有 staged 文件而跳过实际扫描。标准 Pyrefly 仍读取共享旧版 checkout，失败，不能算作 v0.4.0 结果。

另行指定目标 checkout 和 Torch 2.15 CPU 解释器的全仓 Pyrefly 有 75 项错误：V3.2 35 项、Qwen3.5 21 项、V4 LoRA 4 项、V4.1 vision 11 项、CP / GraphTrainer / runtime estimation 4 项。该检查未增加 suppress 或 exclude，也未修改共享 symlink；本轮两个文本模型和公共 eager 迁移范围没有报告类型错误，但全仓类型检查尚未通过。其他模型的共享接口导入回归据此判定为未通过，不纳入已支持范围。

`git diff --check` 通过；上游 checkout 保持目标 SHA 且无修改。代码仍为本地未提交变更，没有创建 PR 或晋升验证基线。本轮临时 CPU 环境与探针在结果归档后清理；没有产生远程训练中间文件。

### 继续执行顺序

1. 恢复授权服务器连接，核实容器内 Torch / Torch-NPU 与 CANN 1021 实际组合，优先解决完整包导入和 T0 依赖阻断。
2. 在真实 CANN 环境构造配方、应用 override、检查最终分片声明，然后运行 `v04_eager`；先无 AC，再启 FullAC。LI 梯度和 Engram sparse 更新需独立观测。
3. 跑三组 `v04_eager_golden`，分析 MTP 归一化与 dtype 变化；保持既有精度阈值和基线。
4. 完成 EP2 完整状态续训后，再运行 `v04_eager_flash` 的 8 卡 100-step 与续训。只有这些设备门槛通过后才能宣称本轮 eager 升级完成。

## Follow-up implementation — FullAC / Host Engram

继续核验当前工作树后，在实际 Torch 2.15 CPU、目标上游和 `torch_remat==0.2.2` 上复现并修复两项生产阻断：

1. V4 的 mHC 在重计算时通过裸 Tensor 操作消费 retained region 的输出，触发 remat placeholder 错误。在 V4、V4 MTP、V4.1 的 attention / MoE 输出进入 HcPost 前声明 `recompute_needs_tensor`。标记位于模型公共消费边界，避免被 ASC、Triton 或 TileLang HcPost override 绕过。
2. V4.1 block 返回的可选 carry 含 `None`，不满足新版 remat checkpoint 的 Tensor 输出协议。仅在 remat 启用时将缺省 carry 编码为标量 bool Tensor，在模型层间循环立即解码；真实 carry 不复制或 detach，无 AC 时保持原接口。

迁移期新增的 remat 测试（后续清理中已删除）使用真实小模型 registry、上游 FullActivationRematConfig 和 Host Engram，覆盖 V4 MTP 输出、V4.1 Full/Reindex/Reuse、两次微批累积。无 AC 与 FullAC 的 loss、参数梯度、grad norm、router / aux 统计、Host 稀疏索引和值、SparseAdam 更新均按零容差比较。另有四组模型 optimizer-hook 测试验证 eager / FullAC 下 expert bias 单次更新、路由计数清零、辅助损失汇总与清零，以及 V4 MTP 的 hook 遍历。没有放宽阈值。

[新增 HostSparse DCP 测试](../../../../tests/unit_tests/extensions/torchtitan/components/test_checkpoint_engram_v04.py) 在真实单 rank Gloo 下对比独立连续 4 步与第 2 步保存、全新组件恢复后继续到第 4 步。使用真实 Host lookup、混合 AdamW / SparseAdam、scheduler 和额外 step / cursor 状态；显式检查 DCP 中 Host weight 与 SparseAdam step / moments，覆盖保存前尚未使用的 dense 参数状态。恢复后的 loss、grad norm、模型、optimizer 和计数严格一致。这里的 cursor 是确定性测试序列的游标，不代表生产 dataloader 恢复已验证。

V4.1 的 EP2 与 Flash 续训用例新增必需 checkpoint key 检查：第 1、14 层 Engram 权重和各自 SparseAdam 的 step、exp_avg、exp_avg_sq 必须同时存在于两份 DCP，避免两侧共同漏存时被判为相等。

本轮统一复跑：上述测试及相关模型回归 **22 项 CPU 组件测试通过**；续训 runner / 比较器 **9 项 tooling 测试通过**，后者不计作模型训练覆盖。完整 pre-commit 的格式、Ruff 和架构检查通过，OAT 实际扫描仍因未暂存跳过；标准旧 checkout Pyrefly 失败，另行指定目标 checkout 的 Pyrefly 仍为既有 75 项错误，本轮三个模型文件没有新增类型错误。`git diff --check` 通过。

证据边界仍需保留：CPU 测试隔离 NPU 包初始化；参考 attention 不提供 LightningIndexer distillation teacher，因此未验证该梯度。上游 FullActivationRematConfig 只包装 `model.layers`，测试包含 MTP 输出与梯度，不等于 MTP block 自身重算已验证。真实 HCCL、NPU 融合算子、EP 分片归属和设备级续训仍待验证。服务器连接重试仍在握手阶段关闭，未进入容器。本轮为实际代码修复与验证进展，整体升级目标继续保持未完成状态。

## Follow-up implementation — objective counts / fused indexer acceptance

对照插件基线与两个上游固定点，确认 V4 的 LI teacher 使用当前 microbatch 行查询数归一化，V4.1 使用当前 step 的全部 microbatch 查询数并在 DP mesh 求和。这是已有差异，本轮不改变训练目标。两者的 LI 查询统计包含结构 padding，不能用 CE 有效标签数或 routing 有效输入数替代。MTP 每深度独立 CE 分母是此次明确跟随上游的计算变化，继续单独验收。

[packed MTP 覆盖](../../../../tests/unit_tests/models/deepseek_v4/test_deepseek_v4_mtp_model.py) 使用真实 TextCollator、小模型 preprocess 和前向；原先独立的 `test_v04_mtp_counts.py` 已并入该文件（逐深度 labels/padding mask、loss/routing counts）。

[V4.1 normalization 测试](../../../../tests/unit_tests/models/deepseek_v4_1/test_indexer_normalization.py) 已迁移新的 microbatch / engine 参数，使用真实 ASC selector 配置，检查全查询计数和两个 counts 参数原样传递。父类 forward/backward 与 DP 归约由 spy 隔离，该测试不是融合 kernel 或分布式求和的执行证据。

[融合 indexer 验收观测器](../../../../tests/integration_tests/v04_indexer_audit.py) 已接入两个模型的 fused 单卡、EP2 和裁剪 Flash 配方，保留 NPU sparse clipper。验收要求实际融合 selector / attention 执行、相关 projection 梯度每步有限，并在正学习率步骤累计观察到非零梯度和真实权重更新。DTensor 的空本地分片通过全局归约参与判定；权重快照放 CPU，参数已有更新证据后不再复制。每个训练阶段均须新生成对应模型、最终步数和 projection 列表的记录，runner 删除旧记录并拒绝缺失或错误阶段的记录，防止配置转换丢失观测器后误判成功。

本轮联合复跑 **46 项通过**：24 项 CPU 组件测试（模型契约 7、FullAC 6、packed counts 1、attention threading 8、LI normalization 1、Host DCP 1）与 22 项 tooling 测试。工具测试覆盖真实双 rank Gloo / DTensor 空分片、库存不一致、非有限梯度、无更新，以及 runner 证据检查；不计作 NPU 模型训练。随后扩展真实 ConfigLoader → Optim.build → step 的同一 testcase，并复跑 runner 的 12 项测试通过，确认动态 optimizer owner 没有被转换丢失。测试仍使用目标上游、Torch 2.15 CPU 和隔离的 NPU 初始化入口；没有模拟目标上游 API 或执行 NPU kernel。

联合执行的临时 CPU bootstrap 最初缺少 multiprocessing main guard，导致 spawn 子进程重新进入 pytest 并触发 CANN 导入失败；修正临时启动脚本后上述 46 项通过。该启动问题没有通过修改生产代码或放宽断言绕过。更新后的全量 pre-commit 中格式、Ruff、拼写和架构检查通过；标准 Pyrefly 仍因共享旧 checkout 失败，OAT 实际扫描仍因未暂存跳过。NPU 单卡、EP、golden 和 8 卡设备结果尚未获得，升级验收仍未完成。

三个 `v04_eager*` suite 目前均须显式运行，没有进入默认 `models` 或现有 CI 自动执行入口。执行方式见测试 README。恢复阶段新进程会重新收集 LI 更新证据，不能借用连续训练阶段的记录；跨 rank 汇总证明全局投影至少一个非空分片有梯度与更新，不证明每个 DDP replica 均更新，也不替代 HCCL 数值对照。

### 其他模型公共接口审计

额外修正能够定位到目标定义的符号搬迁：V3 初始化常量与构建器的 `flavors` 入口、MLA attention 与 metadata 类型、数据模块；Qwen GDN / GDNKernel 和 Ulysses attention 类型的新位置。没有新增类型 suppress / exclude，也没有修改目标上游。V3.2 的小模型工厂仍因旧 `moe_comm_backend` 参数不被新 builder 接受而失败，后续还有 `wkv_b` 布局、aux loss 分母和训练配置协议；其旧 FullAC 入口也尚未迁移，不能将模型包导入成功视为可训练。

Qwen GDN 的真实目标依赖导入被 `attn-gym==0.0.17` 的 CuTeDSL / `cutlass` 缺失阻断；相关测试出现 6 项失败、2 项跳过，未安装 CUDA 专属依赖或伪造上游模块绕过。另有两项目标接口测试通过：V3.2 导入真实 MLA / metadata 类型，以及 Qwen CP 配置 owner 与显式拒绝执行行为；后者不代表 CP 已适配。

此轮最终目标 checkout 全仓 Pyrefly 从前述 75 项减少到 **38 项错误**：V3.2 7、Qwen3.5 12、V4 LoRA 4、V4.1 vision 11、CP / graphs / runtime 4。历史检查记录保留；最终状态仍为类型检查未通过、其他模型训练未支持。两个首阶段文本模型没有新增类型报错。最终全量 pre-commit 的格式、Ruff、拼写、架构检查通过，标准 Pyrefly 失败；`git diff --check` 通过，上游保持干净。远端 SSH 重试仍未完成握手，未进入容器或产生训练中间文件。

## Follow-up implementation — existing regression / data continuation

沿已有测试继续检查首阶段路径，复现并修复两项生产遗漏：

- V4 与 V4.1 的 Muon pattern 仍使用 `moe.*`，专家和 router 权重会静默落入 AdamW 后备组。已改为 `feed_forward.*`；迁移后的 [Muon 分组测试](../../../../tests/unit_tests/models/deepseek_v4_1/test_muon_profile.py) 使用真实 registry 的 meta 模型和生产 `_build_param_group`，精确比较实际 Muon 参数集合、compute layouts 与 buckets，覆盖 V4 MTP 和 V4.1 Engram 的 SparseAdam / AdamW 归属。该测试证明配置选择，不代表 Muon 设备更新已验证。
- `MTPChunkedLossWrapper` 原先先调用主目标 chunked loss，再检查隔离 head；前者已在内部反向，拒绝不兼容 head 时留下梯度。现在先验证 head，在 MTP 分支才进入 detached-weight context，合法计算路径不变。已有拒绝测试同时要求 head 与 hidden 无梯度，正常 loss 继续与独立 CE oracle 比较。

[V4 MTP 既有测试](../../../../tests/unit_tests/models/deepseek_v4/test_deepseek_v4_mtp_model.py) 的非 CP 用例已迁移四项 preprocess、真实 SPMD 标注、tuple predictions / labels 和逐 objective counts，保留双文档边界、模型级 chunked loss、梯度隔离和 MTP 权重恢复检查。旧 CP 用例及 V3.2 辅助入口仍未迁移，不称为全部 MTP 测试已通过。

[Host Engram 既有测试](../../../../tests/unit_tests/models/deepseek_v4_1/test_engram_host.py) 已改用新 optimizer 配置与 optimizer-owned clipper。真实 Host token forward、多微批稀疏累积、norm、更新和 SparseAdam moments 与独立 Embedding / SparseAdam 按零容差比较；保留 shard checkpoint 元数据及 dense / sparse 非有限梯度拒绝更新检查。旧 trainer 包装层测试由对应 Engine / Optim 接线测试接替；旧 checkpoint 重算用例由真实 FullActivationRematConfig 测试接替；anticipatory wrapper 不计入本轮支持组合。

[Host DCP 续训测试](../../../../tests/unit_tests/extensions/torchtitan/components/test_checkpoint_engram_v04.py) 在前述 fixture cursor 的基础上进一步接入真实 GrainDataLoader。数据状态通过 native DCP `extra_states` 保存，恢复后直接从 loader 取下批，输入不再由测试计数器选择。每批 tokens 有独立手算预期；恢复后第 3、4 步的 tokens、loss、grad norm、最终模型、optimizer、scheduler、loader 状态及下一批均与连续运行精确一致。证据限于单 rank Gloo、random-access 文本 loader 和 Host 组件，不代表 streaming、完整 Trainer 或 NPU 分布式续训。

V4 的真实 aligned dataloader 回归也已迁移到 loss / routing counts：两个文档的 padding mask 与逐深度 `[5, 3]` counts 均检查，不再读取删除的 `num_valid_tokens`。

本轮统一执行相关 CPU 回归 **56 项通过、1 项 CP 用例 deselected**；测试环境仍为固定目标上游、Torch 2.15 CPU 和隔离的 NPU 启动入口。全量 pre-commit 的格式、Ruff、拼写与架构检查通过；标准旧 checkout Pyrefly 失败，目标版 Pyrefly 复跑仍为前述 38 项错误，无本轮新增类型错误；OAT 实际扫描仍因未暂存跳过。SSH 直连和 tunnel 再次在握手前关闭，无设备验证进展；T0 软件栈、单卡、EP2、golden 和 8 卡要求均保留，整体升级尚未完成验收。

## Follow-up validation — eager recipes / EP transport / HF loading

### 单卡与两卡验收配方

常规 `v04_eager` suite 现有 14 个用例。三个参考单卡基线使用无 AC、unchunked loss：V4 包含 MTP，V4.1 分别关闭和启用 Engram；另列 V4 chunked loss 用例。四个参考 EP2 无 AC / FullAC 用例和两个融合 EP2 用例均使用 GA2。续训、三组 30-step golden、8 卡 100-step 与第 98 步恢复的步数要求保持不变。

四个参考 EP2 用例使用仓内 [短文档 fixture](../../../../tests/assets/v04_short_text.jsonl)，经真实 Grain 数据入口、AlignedTextProcessor 和 packing 产生 padding。[验收 collator](../../../../tests/integration_tests/v04_padded_text.py) 逐批要求同时存在有效输入和 padding，检查 padding 标签为 `-100`，并通过独立枚举核对每个 objective 的 loss / routing counts 及有效目标。这里的单行短文档 fixture 不等于多文档边界覆盖，也不保证产生空专家或单 rank 零 token。

[runner / 配方测试](../../../../tests/unit_tests/tooling/test_v04_eager_runner.py) **28 项通过**，包含真实 ConfigLoader、HF tokenizer 和 Grain 对两个逻辑 DP rank 各读取 6 个 microbatch 的检查，以及 padding / counts 错误的拒绝用例。这些是 tooling 测试，不计作 NPU 模型训练。用例顺序和 [运行说明](../../../../tests/integration_tests/README.md) 明确先串行执行单卡，再执行两卡；三个 suite 均需显式选择。

### 实际双 rank EP transport

[Gloo EP 测试](../../../../tests/unit_tests/patches/torchtitan/models/common/test_v04_ep_alltoall_gloo.py) 使用真实 `ParallelismContext`（DP shard 2、EP2、TP1）与 `AllToAllTokenDispatcher`，执行 dispatch、combine、collective 和反向。两个用例覆盖均等分片中的空专家，以及全部 rank 零 token。token / score 顺序和 counts 使用手算预期；普通 matmul 专家消费与独立逐 token oracle 对照输出、输入梯度、score 梯度和专家权重梯度。传输检查为精确比较，FP64 专家计算采用 `1e-12` 容差，不作为 BF16 数值证明。两个新增用例及六个已有 MoE 契约测试合计 **8 项通过**。

单 rank 空输入、另一 rank 非空的探针在目标 `spmd_types` 的 uneven all-to-all 经 Gloo 执行时失败，报 `invalid tensor size`。没有修改生产通信实现或用编译执行替代 eager。该边界仍须在真实 NPU / HCCL 上验证；上述通过结果不包含不均等分片、FSDP 权重更新、生产 grouped-mm 或融合专家内核。

### 真实 HF 文件加载

[HF 加载测试](../../../../tests/unit_tests/extensions/torchtitan/components/test_checkpoint_hf_v04.py) 使用两模型的真实 registry 缩小尺寸，写入 safetensors 及索引文件，通过真实 `CheckpointManager.load()`、HF storage reader 和 `dcp.load()` 恢复全部模型状态。V4 包含 MTP 与 hash-router buffer；V4.1 包含 Full/Reindex/Reuse 和 Host Engram。两模型全部 state key、dtype 和 tensor 值均按零容差比较；各有缺少 head 权重时必须报错的负例。四个新用例和相关 native DCP / Host 续训 / adapter 回归合计 **13 项通过**。

fixture 由本仓 adapter 导出，实际加载路径未 mock，但对称的映射错误仍可能被回环掩盖，因此不能视为独立官方 HF 格式、大权重或量化加载验证。Engram 的 HF 逻辑行数有显式断言；源、目标物理 padding 预先均为零，仅证明零值保持，不证明 reader 会清除脏 padding。

### 当前验收状态

本次新增内容为验收配方与测试，没有进一步改变生产实现。最终全量 pre-commit 的格式、Ruff、拼写与架构检查通过；标准 Pyrefly 仍因共享旧 checkout 失败，OAT 因未暂存而未实际扫描。目标 checkout 类型检查沿用上一轮的 38 项错误结果，本轮未重复执行或声明全仓类型检查通过。上游 checkout 保持固定 SHA 且无修改。

授权服务器的直连和 tunnel 再次在 SSH 握手阶段断开，仍未进入容器。该访问阻断已连续多轮存在；独立可执行的本地迁移与相关组件验证已完成到当前证据边界，后续关键步骤需要恢复服务器访问并确认实际 Torch / Torch-NPU / CANN 配套。升级目标处于阻塞、未验收状态，计划仍保留 `Status: proposed`；T0、单卡纯 eager、HCCL EP2/FSDP2、设备续训、golden 和 8 卡验证均不计为通过。临时 Gloo 探针已清理，保留失败日志和隔离 CPU 环境用于继续验证；未产生远程中间文件。

## Resumed investigation — 2026-11-05

按会话指令恢复执行，远程访问限定为两个授权 tunnel 别名。进一步诊断发现本机 SSH 配置中的两个临时 tunnel 域名在公共 DNS 查询中均返回 `NXDOMAIN`；本地代理连接正常，失败发生于 SSH 认证之前。已请求当前有效地址，现有配置未擅自替换，远端仍未进入容器。

重新核对 T0：仓内 wheel 索引仍最高为 `torch_npu==2.14.0.dev20260918`，PyPI 未列出 2.15 版本。但 Torch-NPU 主线固定点 `848b8e7271069d5a9947bf75fd02a162092b41a6` 的 [version.txt](https://github.com/Ascend/pytorch/blob/848b8e7271069d5a9947bf75fd02a162092b41a6/version.txt) 已列出 `2.15.0` 构建目标，[ci/build.sh](https://github.com/Ascend/pytorch/blob/848b8e7271069d5a9947bf75fd02a162092b41a6/ci/build.sh) 支持 `--torch=2.15.0` 并检查已安装 PyTorch 的 major/minor。恢复访问后先检查容器现成包；缺少配套包时可在隔离目录评估该固定源码的构建。这里只确认源码构建入口，不代表已构建成功、兼容指定 CANN 或设备测试通过，requirements 暂未改成未经验证的新配对。

额外只读复核了 14 个常规配方的纯 eager 调用链。`TrainingEngine → BaseModel.parallelize` 实际执行 `apply_local_compile([])`，覆盖 loss、HiMidLo split 和激活函数；V4 委托该生命周期，V4.1 继承该生命周期。`torch_remat==0.2.2` 的 FullAC 普通路径使用非重入 checkpoint，只有已经处于编译时才进入 compiled 分支；chunked loss 使用 Python 分块及普通 backward，grouped-mm 直接进入注册 NPU 算子。未发现该矩阵中绕过 local compile 配置的实际编译触发。该结论为静态复核，本轮没有新增设备测试结果。

随后会话更新了 tunnel 地址，两台授权服务器均已连通并进入指定容器。宿主机进程核验确认，一台的设备由占位 worker 使用，另一台部分设备运行现有 vLLM 作业；未干预现有训练或推理任务。原容器的 Torch 2.14 / Torch-NPU 2.14 在 CANN 1021 下可导入，但仍缺少上述两个上游必需 API。

已在空间充足的服务器建立独立环境，安装 Torch 2.15.0+cpu、torchvision 0.29.2+cpu 和固定目标 TorchTitan；实际检查 `direct_url.json` 确认为 `45e33cd413b64cb37a7840cfd3c9765d3c639ae1`，上游 Trainer 导入通过。插件送测快照为 `65a01fc731ee64c963aab09b15f8eee13e760eb6`，通过独立 index 创建，不改变本地工作分支或暂存区；远端恢复该 Git 身份并核验工作树干净。此时使用临时验证 requirements 将 Torch 配对替换为候选版本，尚不代表正式 requirements 完整安装通过。

Torch-NPU 固定源码构建已启动。首次缺少 ACL 头文件，补齐固定 gitlink 的 runtime、ge、graph-autofusion 后进入 C++ 编译；未修改第三方源码。后续正式依赖方案采用仓内构建入口和本地 wheelhouse，避免依赖尚未发布的 2.15 wheel 或继续绕过旧包依赖约束。完整插件导入、NPU 运算及训练仍待新 wheel 产出后验证；原服务器访问阻断已解除，任务继续执行。

### Torch-NPU 构建与首轮设备验证

固定源码 `848b8e7271069d5a9947bf75fd02a162092b41a6` 的官方构建已完成，产物为 `torch_npu-2.15.0+git848b8e7-cp312-cp312-linux_aarch64.whl`，SHA256 为 `19addafa14d7734b57385357f16288ca24d39f820122a95e01ad570a34cf7313`。构建关闭 TorchAir、FXRT 和 RPC，使用 CANN 1021，未修改第三方源码。仓内 helper 的 wheel 校验通过；helper 自身完整构建分支另行验证中。

正式 requirements 已在不继承 system-site-packages 的干净 Python 3.12 环境中通过本地 wheelhouse 完整安装，`pip check` 返回 `No broken requirements found`。固定目标上游、插件入口和两个 debug 配方可导入。单设备矩阵乘、反向和 AdamW 更新通过，loss 为 1220，grad norm 为 708.6136474609375；该基础运算不计为模型训练通过。

首个真实 `v04_v4_mtp_1p` 用例在 C2 `e5337712c71dd5e2aea1b523b8efe42a94b69153` 的启动阶段失败，尚未执行训练步。启动脚本默认选择 AscendC backend，device-copy patch 导入可选模块时触发父包初始化，而当前 wheel 不含该 backend。修复采用不执行父包的逐级模块存在性检查，缺失时跳过 patch；存在时仍传播内部导入错误。相关 CPU 单测 9 项通过。C3 `e9bdf42f762d8ce9ade499964f21fa2b64ea98bd` 已开始重新验证默认启动路径，结果待回收；尚不声明单卡、分布式或完整升级通过。

C3 越过入口导入后，暴露 `UnitScaleRMSNorm.forward(input)` 与 `q_head_norm` 分片声明键 `x` 不一致。修正为 `input`，主层与 MTP 共用 setter。实际 CPU/Gloo 模型 parallelize、norm 前反向逐位比较，以及两模型 EP1/EP2 声明审计共 5 项通过。C4 `c3c944795d9b118c09dead3f7c3b86084a8b0d73` 完成真实 NPU 模型与优化器初始化，进入第一步；随后因参考 ComplexRoPE 的 cache 位置索引触发 `aclnnIndex` 不支持 `DT_COMPLEX64` 而失败。同步诊断明确定位上游 `_reshape_for_broadcast` 的 `rope_cache[positions]`，不是异步栈表面显示的 FSDP stream hash 问题。正在修复参考算子接入；尚无完整模型训练步通过。

本轮重新执行目标 checkout Pyrefly，仍为 38 项错误，无 device-copy 修改引入的新错误。错误分布仍集中在已记录的其他模型、LoRA、多模态、CP 与图模式接口；不将此结果描述为全仓类型检查通过。

### C5 参考 eager 训练进展

C5 `826a84cd988eaa2b6b51c1090114a17011c87388` 为新增参考配方补齐既有 `rope.workaround`，融合配方仍显式选择 `asc_complex`。该 workaround 使用纯 PyTorch 实数运算，不进入编译或融合 RoPE kernel。CPU 对照独立实数公式的前反向、实际 override 命中与分片回归共 28 项通过。FP32 与原 complex 算术存在约 `2.38e-7` 量级的前向舍入差异，不声明二者比特一致；原 golden 保持既有 workaround 路径和判定门槛。

以下真实 NPU 用例均完成 3 步，runner 返回 0，并读取到完整的 TensorBoard loss 步数：

| 用例 | 结果与边界 |
| --- | --- |
| V4 MTP 单卡，unchunked | 通过，loss 16.00956 → 15.93213 → 15.84342 |
| V4 MTP 单卡，chunked | 通过，第三步 loss 15.84309；与 unchunked 的差异尚待专项计算变化分析，不算数值一致通过 |
| V4.1 单卡，无 Engram | 通过，loss 12.21235 → 12.39323 → 12.23818 |
| V4.1 单卡，Host Engram | 通过，实际创建 CPU table 和 SparseAdam；完整稀疏状态续训尚待验证 |
| V4 MTP EP2/FSDP2，无 AC | 通过，GA2 与 padding counts fixture 生效 |
| V4 MTP EP2/FSDP2，FullAC | 通过，GA2 与 padding；与无 AC 的完整数值逐项比较仍待回收 |
| V4.1 Host Engram EP2/FSDP2，无 AC | 通过，CPU table 按 EP2 分片 |

以上均为非融合参考路径，不包含融合 Lightning Indexer 梯度证明，不替代 30-step golden、完整状态恢复或 8 卡裁剪 Flash 验收。融合路径首次扩展编译已切到本任务独立缓存；共享缓存中的旧锁和其他作业产物不作清理。

V4.1 EP2 FullAC 随后也通过 3 步。直接读取四个 EP2 用例的 TensorBoard scalar，两个模型各自的无 AC / FullAC 三步 `global_avg_loss` 和 `grad_norm` 均精确相等。该比较不包含最终模型或优化器状态的 AC 对照。

V4 的 native DCP 两卡续训用例通过：连续 4 步与第 2 步恢复至第 4 步的 loss、grad_norm 和最终完整 DCP 状态比较均通过。V4.1 续训在首次第 2 步保存时两 rank 段错误，栈位于 DCP `_SerialCpuLoader.values` 的 `tensor.cpu()`，正在定位具体张量及复制路径，不预先归因于 Host table。

原 V4 单卡 30-step golden 的首次执行完成训练但精确比较失败。已发现配方迁移遗漏：旧 golden 显式使用仓内 `tests/assets/deepseek_v3` mini tokenizer，新配方却继承了 smoke 环境提供的 DSV4 tokenizer。当前先恢复原 golden 输入条件并重测，不能将这次结果归为上游数值变化，也不更新 golden 文件。两卡 golden 同轮使用了相同错误输入条件，须一并重测。

### C6 与融合 / 续训诊断

C6 `b3da56c1a305d6d0e6fabdde02aa3db148c78411` 恢复 golden 的 mini tokenizer 和固定数据输入条件；三个 golden 均完成 30 步训练，但精确 loss 比较仍失败。保留原基线与失败日志，继续核对固定原插件 / 上游基线，不将差异直接视作允许的计算变化。已排除一种错误归因：指定上游基线 `ebdfc8` 与目标 `45e33cd` 均为 `w13 → w2` 初始化顺序，不能用更老版本的 `w1 → w2 → w3` 历史变化解释本次升级。

C5 的 V4 融合单卡完成 3 步，runner 返回 0；第 3 步 indexer 审计记录 `1 pairs, 4 projections` 通过，覆盖实际融合选择、梯度和参数更新。V4.1 融合单卡在 C6 失败：CANN 1021 的 `aclnnMhcSinkhorn_1019_MhcSinkhornAiCore` 缺少动态 kernel 配置，正在核对 SDK 中同版本可用源码与入口，未切换 CANN 或用 reference 算子冒充融合验收。

V4.1 DCP 段错误已通过独立 NPU 对照缩小到交换内存 tensor 的直接主机复制：普通 NPU tensor D2H 通过，`empty_with_swapped_memory` 的直接 D2H 和直接 CPU→交换 tensor 写入均段错误。经普通 NPU 临时张量中转的两个方向均通过。原日志中的失败对象是 dense optimizer embedding moment，并非 Host Engram CPU table。正在以逐项 CPU checkpoint 视图和 NPU 中转恢复适配 virtual optimizer，额外 CPU snapshot 内存开销须记录。

仓内 Torch-NPU 构建 helper 完整执行成功，固定源及七个 gitlink、参数、Python / Torch 版本和 SHA256 均记录在 manifest；helper 产物 SHA256 为 `7f6017137d0afb1af581fd69e88200965c5daa527b5a73deca674081cc8138fb`，未替换本轮训练使用的手工构建 wheel。两产物校验和不同，不声明二进制逐字节可复现。helper 临时 checkout 已自动清理，手工构建的 533 MiB build 目录也已清理，保留 wheel、固定源与证据日志。

### C7–C9 验证进展

C7 `75041c258f92f570e7e942a7c726c43c30e93d9c` 的 V4.1 EP2 native DCP 续训已通过。第 3、4 步恢复后的 loss 分别为 `12.35831069946289`、`12.252204895019531`，grad norm 分别为 `2.0679025650024414`、`2.112722635269165`，均与连续运行精确一致；最终完整状态比较及 Engram 参数 / SparseAdam 必需键检查通过。

新增三个独立 HCCL EP2 边界用例在 C7 全部通过：空专家、两 rank 全空、rank0 空而 rank1 非空。传输 token、score、counts 和 splits 精确比较；输入、路由 score、专家权重梯度与 CPU FP64 逐 token oracle 按测试声明的 FP32 小容差比较。单 rank 空的 Gloo 失败未作为跳过理由，真实 HCCL 路径已验证。

C8 `fbf9aab0b48c4af71423c4cb2a236b80a1ada5dd` 的 V4 融合 EP2 三步通过。额外的 `v04_eager_attention` suite 验证了 V4.1 融合 attention / indexer、参考 Sinkhorn、融合 HcPost 的单卡与 EP2 三步，并通过 indexer 梯度和更新审计。它不替代原有完整融合 MHC 用例；CANN 1021 独立 Sinkhorn kernel 缺失仍未解决。

为 8 卡完整状态比较调整了验证工具：DCP 按字段直接加载并精确比较，避免生成两份全量 `.pt`；连续训练最终检查点使用移动而非同盘复制，并支持单独的临时基准存储，成功后清理外置副本。覆盖真实 DCP 数值改变拒绝、阶段数据保存和外置副本生命周期的 tooling 回归共 37 项通过。该工具调整不放宽比较门槛。

C9 `0c399aba9408578ad2cf1fd21058370270a92b85` 已启动 V4 43 层、16 experts、4K、MTP、EP8/FSDP8、FullAC 的三步融合冒烟。该配置约 251 亿参数，须为完整状态保留足够磁盘和主存空间；100-step 和续训尚未完成。

补充 golden 溯源：V4 / V4.1 golden 文件最后写入提交分别为 `27898b9cdca93bf904e4028cff48206615325520` / `e057e9110ce728d5f699a459478b0be059ed435d`，两处 requirements 当时均固定 `torchtitan==0.3.0`。指定当前适配基线已是 `ebdfc8`。这说明基线跨度需要进一步核验，但不能单凭历史 pin 宣称已解释全部数值差异或通过 golden。隔离原 A/B checkout 已固定，原容器包版本与原 requirements 并不完全相同，当前只作明确标记的兼容性诊断。

### C9 长程验证与基线对照

V4 裁剪 Flash 三步融合冒烟已通过：43 层、16 experts、4K、MTP、EP8/FSDP8、FullAC；第 3 步 loss 为 `14.25827`，grad norm 为 `8.6036`，峰值显存约 59.28 GiB，indexer 梯度审计通过。该结果仅证明上述裁剪配置的短冒烟。相同 C9 已启动 100-step 与第 98 步恢复验证，结果尚待回收；完整状态基准副本放在本任务独立主存文件系统目录，成功比较后清理，以避免同盘三份检查点耗尽空间。

固定原插件 / 原上游与 C9 的 debug 模型 CPU 初始化对照：固定 seed 42，meta 构造后转 CPU 初始化，将旧 `.moe.` 字段规范化为 `.feed_forward.` 后，110 项状态的 shape、dtype 和 SHA256 全部一致。该证据不包含 NPU/FSDP 初始化，不证明训练 golden 一致；原基线诊断环境仍与历史 requirements 存在版本差异。

C9 再次执行完整 pre-commit：格式、Ruff、codespell、架构检查通过；OAT 因无暂存文件实际跳过。标准 Pyrefly 仍受共享旧上游路径影响而失败；独立指定目标 checkout 的检查仍为 38 项错误，未声明完整 Lint 通过。

原适配基线与 C9 的真实 Grain 前四个单卡批次中，input、labels、positions、padding mask 的 shape、dtype、内容哈希全部相等；有效 token 数依次为 2048、2028、1880、2039，新版 loss/routing counts 与之相符。此对照仅覆盖前四批，不扩展为全部数据轨迹一致。

MTP loss 局部 CPU 诊断采用同一输入 / 线性 head 权重、两个 objective、有效计数 `[128, 127]`、8 个 chunk。FP32 的 hidden 梯度完全相等，head 梯度最大绝对差为 `1.1175870895385742e-08`；BF16 的 hidden 梯度也相等，head 梯度最大绝对差为 `0.00048828125`。loss 的差异均约 `4.77e-7`。该结果定位了分块 head 梯度累加的舍入差异，不将它当成三步 NPU 训练差异的完整解释或数值门槛验收通过。

### 原基线设备诊断与 C10 附加冒烟

原插件 `c422c623` / 原上游 `ebdfc8` 的 V4 单卡 30-step 诊断已完成，也未通过仓内历史 golden。原 runner 的 `--lr-scheduler.total-steps` 已与其上游 CLI 不匹配，诊断启动器仅将它转换为 `--optim.lr-scheduler.total-steps`；原仓源码、模型和 golden 未修改。该诊断使用 Torch 2.14 / Torch-NPU 2.14.dev20261019，与历史 pin 并非完全相同。

读取完整 TensorBoard 标量，与 C6 的目标 v0.4 单卡 30-step 运行比较：前 8 步 loss 完全相等，首个 loss 差异出现在第 9 步（`1.71661376953125e-5`），30 步最大绝对差为 `0.000152587890625`；grad norm 从第 1 步出现 `4.76837158203125e-7` 的差异，最大绝对差为 `0.0002346038818359375`。因此历史 golden 不匹配与新旧运行间的小差异必须分开排查。两侧软件栈和 AC 配置不同，尚不归因于单一迁移点，也不声明比特一致通过。

新增 V4.1 40 层、16 experts、4K、Engram、FullAC 的三步附加诊断配方，保留融合 attention / indexer 与参考 Sinkhorn，接入 `v04_eager_flash_smoke`。原完整融合和 100-step suite 不变。配方与 runner tooling 共 38 项通过，相关 Ruff 检查通过；真实设备执行尚待完成。

C10 `8ab65ec366e53b07bf0ddcaf91b2210cdd4b33c3` 的 V4.1 8 卡首次附加冒烟由本任务主动停止，未计为失败算子定位或通过：默认 Flash Engram 每 N-gram 目标 16M 行，连同 SparseAdam 稠密 moments 约需 2.4 TB 主存，超过单机容量。停止的是经核实的本任务进程，设备和主存已回收；未发生已确认的主存 OOM。后续附加诊断显式采用现有 `TORCHTITAN_ENGRAM_TABLE_ROWS=65536` 开关，保留两层 Engram、宽度、heads、token-map 要求和 40 层拓扑，结果仅代表裁剪行数规模。

新增显式 `v04_eager_flash_attention` 诊断 suite，100 步后从第 98 步恢复，保留完整 DCP 比较、两层 Engram / SparseAdam 八个必需状态键和 indexer 审计；只移除独立 Sinkhorn 融合 override。原完整融合 100-step suite 保持不变。补强工具测试后共 41 项通过，相关 Ruff 检查通过；长程诊断尚未执行。

原基线首步梯度上的范数诊断显示，旧完整列表算法与新版按 mesh 分组算法均返回 `5.2936625480651855`，只有一组 98 项梯度。它排除了该梯度集合的分组顺序影响，尚未排除跨 Torch / Torch-NPU runtime 的 reduction 或 backward 差异。关闭原基线 AC 的单卡对照另行执行中。

C11 `f552015ace16431b1849a3fe3f003cf1e78743f2` 已在两台授权服务器独立检出并验证干净，目标上游保持 `45e33cd`。第二台服务器清除继承的 PYTHONPATH 后，正式 requirements 完整安装、`pip check` 和 runtime 版本检查通过；此前一次 pip check 读到了继承路径下的 CANN profiler 包，不能算干净环境检查结果。Torch-NPU wheel 与首台训练产物 SHA256 一致。修正后的 V4.1 8 卡三步诊断已启动，使用显式 65,536 行配置，结果待回收。

原基线关闭 AC 的 30 步已完成，TensorBoard loss 和 grad_norm 全部与旧 FullAC 运行精确相等。因此新旧轨迹差异不能归因于本次对照中的 AC 开关；继续收集首步按参数的裁剪前梯度，区分 backward 差异与 runtime reduction 差异。历史 golden 仍未更新。

C11 的 V4.1 8 卡附加三步冒烟已通过（runner 0）：40 层、16 experts、4K、FullAC、每 N-gram 目标 65,536 行 Engram、融合 attention / indexer 与参考 Sinkhorn。第 3 步 loss 为 `10.74944`，grad norm 为 `9.0211`，峰值显存约 47.7 GiB；审计明确记录 `8 pairs, 20 projections` 通过。模型总参数量为 `31,681,333,360`。同一 C11 已启动 100-step 与 98→100 完整状态续训；尚不将长程或完整融合 MHC 标记为通过。

原基线与新版本首步裁剪前状态诊断：104 个参数初值逐项完全一致，98 个非空梯度中 19 个 FP32 RMSNorm weight 梯度不同，其他梯度（含输出 head）完全相等。最终 norm.weight 最大绝对差约 `5.50e-5`。固定 synthetic BF16 输入 / 输出梯度的局部 RMSNorm 重放，在两套 runtime 中前向、输入梯度、权重梯度均逐位一致；尚不能归因于 RMSNorm 算子公式，继续核对 GA2 / FSDP 混精累加路径。

V4 C9 长程运行完成了第 98 步训练（loss `10.66829`、grad norm `1.9967`），随后首次 DCP 保存失败；未完成 100 步或恢复，不计为 T6 通过。DCP 尚在 plan 元数据 scatter 阶段，仅留下空目录。更早设备日志报 `libhccl_aicpu_kernel.so` / `HcclLaunchAicpuKernelA3` 加载失败，AICPU errcode 11002（open so failed），后续同步报 507018。独立 8 rank 小对象 gather/scatter 及小型 DCP save 可通过，正在核对规模相关通信路径；不能仅凭当前证据归因于 CPU tensor 复制、磁盘容量或显存 OOM。V4.1 C11 长程仍在独立设备组运行。

首步梯度 GA1 对照亦完成：新旧首步 loss 均为 `12.303995132446289`，104 个参数初值相同，仍只有同样 19 个 RMSNorm weight 梯度不同。因此 GA2 跨微批累加不是该差异的必要条件。继续以真实 norm 输入、权重、输出梯度核验，未修改训练实现或 golden。

### 首步舍入差异与 EP8 保存故障的进一步定位

GA1 真实训练中全部 19 个差异 norm 梯度逐元素满足 `old == new.to(torch.bfloat16).float()`，其余 79 个非空梯度相等。该关系将差异缩小到 BF16 舍入，而非笼统的 RMSNorm 公式变化。旧 / 新 runtime 的 CPU 最小实验均表明 BF16 叶参数默认保留 BF16 梯度，显式 `grad_dtype=None` 可保留 FP32 梯度；新版 FSDP 在同步 backward 的 deferred upcast 路径设置该属性。真实 NPU 输入重放与实现归因仍待完成，不修改新版梯度语义或刷新 golden。

EP8 DCP 故障对照中，8 rank 每 rank 10 MB 的对象 gather/scatter、500 个真实 NPU 分片 DTensor 的保存后清零再加载校验，在默认 HCCL 组和独立 Gloo 组均通过；开启确定性仍通过。成功探针与原故障均执行 `ScatterRingFor91093Executor` / `AICPU_TS`，故不能将设备日志解释为 CANN 文件恒定缺失。原训练各卡约 97% HBM 占用，继续做受控显存压力对照；尚未据此改用 Gloo 或宣称根因已确认。

8 rank HBM 占位至每卡剩余约 1.59 GB 后，10 MB/rank 的 HCCL 对象 gather/scatter 仍通过，通信后余量约 1.25 GB；低显存余量单独不足以复现。进一步设备 RUN 日志明确记录 `libhccl_aicpu_kernel.so` 加载失败的底层原因是 `undefined symbol: HcommIsSupportHcommAicpuTsTaskCacheClear`。正在比较训练与成功探针的设备动态库加载次序及来源；此前的 `open so failed` 应解释为动态链接失败，不能等同于文件不存在。尚未通过更换通信组绕过问题。

CANN 1021 的独立 MhcSinkhorn 限制经 SDK 文件核对确认：A3 `ascend910_93` 目录含 `mhc_pre_sinkhorn` 及 backward，但没有独立 `mhc_sinkhorn` / backward 的 kernel 与 JSON；现有独立算子 C++ stub 引用缺失的 `arch35` 实现头文件。Python fake 注册或重建 ACLNN wrapper 均不能补齐设备实现。已有 Triton V4 Sinkhorn 初次 row normalization 与 V4.1 reference 的 eps 语义不同，不能直接替换；整体 MhcPreSinkhorn 也会改变 V4.1 跨子层 pre_mix 语义。本轮保留原生融合失败状态与显式 reference Sinkhorn 诊断边界，不静默替换算子。

目标 Pyrefly 的 38 项错误经入口静态审计分类：未发现直接阻断已验证 DSV4 / V4.1 文本 eager 的项；25 项属于 LoRA、CP、多模态、Graph/CV 后置能力，12 项属于 V3.2 / Qwen 公共入口及训练接口，另 1 项是自动加载但已无目标 consumer 的 runtime-estimation 属性赋值。V3.2 recipe 与 Qwen 公共 factory 的导入回归仍需补齐，不能将全部错误归为后置能力；默认 models suite 也仍包含后置模型。继续修复公共导入与构造，不将其扩展为其他模型训练通过。

真实最终 norm 的 GA1 样本重放已获得精确证据：新旧训练采集的 x、weight、y、dy 全部逐元素相同（BF16，x/dy 为 `[2048, 256]`，eps `1e-6`）。在旧 runtime 的 NPU 上重放同一输入，默认 BF16 叶梯度精确等于旧训练梯度；仅将叶 `grad_dtype` 显式设为 FP32，所得梯度即精确等于新训练梯度。结合新版 FSDP 的 deferred-upcast 语义，这证明梯度 dtype 的变化足以解释该最终 norm 的 GA1 权重梯度差异，不需要假设 RMSNorm 算子公式变化。该证据不自动等同于全部 30 步差异均已解释；生产配置与 golden 未修改。

V4.1 长程运行期间，共享磁盘可用空间从约 796 GiB 降至 698 GiB。首次保存前，将本任务尚不存在的 `checkpoint/step-98` 定向独立 tmpfs 目录；step-100 保持磁盘位置，连续训练的最终状态仍由 runner 移至既有外置基准目录。预计峰值约为 tmpfs 708 GiB、磁盘 354 GiB；未改源码、配置值或检查点内容。验证后清理本任务临时状态，保留日志和对比证据。V4 早期保存诊断收缩为两步、仅一份完整状态，启动前需同时预留两任务所需空间。

V4.1 C11 第 98 步完成训练（loss `8.02717`、grad norm `2.5216`）后，DCP 已成功越过元数据通信并写入约 355 GiB 的完整状态。设备日志显示相同 CANN 的 `HcclLaunchAicpuKernelA3` 加载成功；此前同样执行 legacy `libccl_kernel.so` 的 EP all-to-all。该对照排除了「先执行 EP / legacy 库」单独必然导致 V4 故障的解释，尚不等同于最终 100-step 及恢复比较通过。

公共入口补充迁移：V3.2 recipe 已改为 `FullActivationRematConfig`，导入测试通过；真实构造仍受上游 MLA 类型与权重归属变化阻塞，未绕过校验。Qwen 公共模型工厂接入 `build_model_config`，保留 flex 默认和显式 varlen debug 入口，CPU 构造检查通过；未迁移正式尺寸 recipe 保留符号并明确拒绝调用，不用 debug 默认值冒充正式配置。相关隔离 CPU 检查 5 项和定向 Ruff 通过。Qwen debug 使用上游新版 DefaultActivationRematConfig，不能称为与原 SelectiveAC 等价或 NPU 训练兼容。该补充尚未纳入 C11 远端验证快照。

V4 C11 两步 Flash 保存诊断已越过原失败的元数据通信阶段并开始写入完整状态，尚待保存完成。C9→C11 生产目录没有改动。第 1 步 loss `15.898028373718262`、grad norm `9.220611572265625` 与原 C9 完全相等；第 2 步不相等，因为短诊断总步数使上游将 warmup 从 20 截断为 2，首次更新的学习率倍率从 0.05 变为 0.5。该诊断只验证训练后保存机制，不作为长程数值等价或问题已修复的证据。

V4.1 C11 恢复到第 100 步后的 TensorBoard 精确比较已通过：第 99、100 步 loss 分别为 `7.80725622177124`、`8.043328285217285`，grad norm 分别为 `2.733940839767456`、`4.370377540588379`，均与连续训练按原始标量逐位一致。恢复后的最终完整 checkpoint 仍在保存，全部状态与 Engram / SparseAdam 必需键的比较尚待 runner 完成，不提前标记整项通过。

公共入口补充修改后，按约定环境和完整命令复跑 pre-commit：格式、Ruff、codespell、架构检查通过；OAT 实际因无暂存文件跳过；标准 Pyrefly 仍受共享旧上游路径影响而失败。通过 CLI 明确指定固定目标 checkout 和 Torch 2.15 CPU 解释器的全仓 Pyrefly 从 38 项降至 31 项，未增加 suppression 或 exclude；余项包括 V3.2 训练接口及后置能力，完整 Lint 仍不通过。

### 完整状态比较器的正确性与规模修正

真实 DCP 反例发现 C9–C11 的逐项比较路径存在非 Tensor 状态遗漏：`DefaultLoadPlanner(flatten_state_dict=False)` 仍默认启用 `flatten_sharded_tensors`，该过程复制 state 字典，BYTES 项的加载结果未写回原字典，step 等值可能始终为 None。修复显式关闭这项不需要的 flatten；新增真实 scalar、dtype / value、key 和必需键差异拒绝检查，相关 tooling 共 44 项通过。C9–C11 旧比较器不能证明非 Tensor 状态精确相等；Tensor 和独立 loss / grad norm 证据不因此失效。

C5 / C7 / C8 的比较器 blob 相同，采用 `dcp_to_torch_save` 全量转换，未使用上述路径。以 C7 原始源码和真实 step=4 / step=5 检查点重放，正确拒绝 `state.step` 差异，故不撤销此前 C5 / C7 / C8 的完整状态对照结论。

V4.1 元数据实测为 13,371 个 key、7,781,209 字节，单次反序列化约 0.29–0.39 秒；逐 key 重新读取会带来数小时的重复开销。新比较器在只读检查点范围复用 metadata，逐项加载与精确比较不变；真实 DCP 测试确认两份检查点仅物理读取两次 metadata。C11 V4.1 的训练产物保留，将由修复后固定验证器检查全部状态，不使用旧比较器的完整通过结论。

修复后的工具已固定为 C12 `37267639337ba65bfc23b8a06900f66721b4644f`（tree `bdbacbda0b06baf68217b22b7bf0c2748b71a94b`），两台验证环境均独立检出且干净。C11 V4.1 两阶段训练均返回 0，indexer 审计和恢复指标通过后，仅停止旧 runner 的状态比较进程，保留 checkpoint；其 wrapper 终止不代表训练失败，也不能计为完整 suite 通过。C12 验证器正在对同一 C11 产物重新执行全部状态和八个 Engram 必需键检查，训练代码版本与验证器版本分别记录。

C11 V4.1 的保留产物已通过 C12 修复后完整验证器（返回 0，422.18 秒）：13,371 个保存值的类型、Tensor dtype / shape / 元素及其他值精确相等，八个 Engram / SparseAdam 必需键齐全，后两步 loss / grad norm 再次通过精确比较。三个完整 checkpoint 每份 `.distcp` 合计 `380,221,693,825` 字节；已保留 metadata、SHA256、配置与版本 manifest、TensorBoard、indexer 审计和日志，随后仅清理本任务三份大状态。原 C11 wrapper 在旧比较阶段被停止，不能记作原 suite 整体返回 0；此结论明确为 C11 训练加 C12 验证器。下一轮在 C12 同一固定快照执行原始训练与修复后的完整检查。

C12 同版本端到端复验已启动：V4 使用原 100-step / resume98 配方，前两步 TensorBoard loss、grad norm 与 C9 原长程精确一致；V4.1 使用原显式 reference Sinkhorn 附加配方。两模型分别占用独立八卡组，step98 指向各自独立 RAM 目录，最终状态留磁盘并由 runner 转存基准。恢复到100且原始指标精确通过后，归档并清理已消费的 step98 输入状态，保留最终两份供完整比较，以约束并行存储峰值。另在可用的独立两卡组逐例补跑 C12 小模型矩阵。所有结果须待各 runner 和完整比较结束回收。

C12 本地核心产品回归 33 项通过，覆盖模型契约、MTP counts、remat、sharding 签名、sparse clipper、FSDP 接线、Host Engram native DCP 及真实 tiny-model HF 加载。使用实际 Torch 2.15 CPU 和目标上游，仅隔离不可用的 NPU 包启动入口；不作为 NPU 算子验收。C12 V4 正式长程前 23 步 TensorBoard loss / grad norm 已与 C9 原长程精确相等。

C12 主 eager 路径完成独立只读复审，未发现可定位的新语义缺陷；核对范围包括四项 preprocess / MTP counts、geometry / packed / Reuse、pre-W2 score / EP、FullAC 副作用、Host Engram clip / swap 恢复、HF FQN 与编译边界。复审不替代正在运行的同 head NPU 验收，不扩展为任意拓扑、MTP 深度或外部 HF 权重集已验证。

为覆盖普通 `pre-commit --all-files` 不会读取的未跟踪新文件，使用独立临时 Git index 纳入当前全部改动，仍执行约定的完整命令；实际 index 哈希保持不变。格式、Ruff、codespell、架构检查通过；OAT 此次实际检查 136 个改动文件并通过，不再仅记作无暂存文件跳过。标准 Pyrefly 仍失败，独立目标上游检查保留前述 31 项遗留。

C12 同一快照的核心 NPU 矩阵 14/14 返回 0，覆盖两模型参考单卡、V4 chunked / MTP、V4.1 Engram 开关、EP2/FSDP2 GA2/padding 无 AC 与 FullAC、既有融合 attention/indexer 路径及两模型完整续训。V4.1 原生 Sinkhorn 仍不包含在附加融合诊断中。两模型连续 4 步与 2→4 恢复的原始指标、完整 DCP 状态及必需 Engram 键均通过 C12 修复后比较器；V4.1 恢复第 3、4 步 loss 为 `12.35831069946289` / `12.252204895019531`，grad norm 为 `2.0679025650024414` / `2.112722635269165`，与连续训练精确相等。另运行原 HCCL runner 的 empty_expert、all_empty、one_empty 三个边界，均返回 0。小模型临时 checkpoint 已在比较通过后清理；8 卡长程仍在执行，不能提前标为 T6 通过。

### Muon 新计算布局协议

在 C12 两模型的融合 attention/indexer 小模型上选择实际 DistMuon，均在构造阶段因 `ComputeLayout.compute_dtype=None` 解析为 FP32 参数 dtype 而失败，尚未开始训练。已确认 optimizer materialize 执行正常；该问题不是临时诊断入口漏转换。旧、新上游的 Newton–Schulz 均显式使用 BF16，新版新增布局 dtype 约束。本地修复在共享 owned/expert helper 和两模型直接构造的 BlockShard layout 显式声明 BF16，MTP 复用相同策略；未改参数、梯度、momentum 存储 dtype 或 SparseAdam 分组。真实模型 profile 与专家布局 CPU 检查共 6 项通过，独立审查未发现漏配或 AdamW 路径影响。新版重分片 buffer 和 BlockShard 缩放仍可能产生 optimizer 整步差异，不能仅据 NS dtype 一致宣称旧新数值等价；实际 Muon 更新及完整续训待新固定快照验证。此前 C12 的 AdamW 验证仍按其原始代码版本记录。

Muon 布局修复固定为 C13 `d676a64daf725da1fbe9d2029aaf1374a9f9e24a`，相对 C12 仅三个生产文件的 Muon layout 字段改变，以及相关测试和 Note；正式工作区未提交、上游 checkout 未修改。C13 完整 pre-commit 的格式、Ruff、codespell、架构与 OAT 通过；标准 Pyrefly 仍使用旧共享源码而失败。以独立 CLI 搜索路径固定目标上游的全仓 Pyrefly 仍为 31 项，未新增错误、未扩大 suppression。实际 Muon 三步及完整状态续训另用独立 checkout 验证，C12 的 AdamW 长程任务不受部署影响。

C13 两模型纯 eager 融合 attention/indexer 的 Muon 三步实际 NPU 诊断均返回 0：V4 DistMuon 管理 75 个参数，indexer audit 覆盖 1 个 pair / 4 个 projection；V4.1 DistMuon 管理 40 个参数，audit 覆盖 3 个 pair / 8 个 projection，保留 reference Sinkhorn。外部诊断 factory 仅选择现有 Muon profile，未修改固定源码或默认 AdamW recipe；此结果不代表旧 CLI 已迁移、旧新 Muon 数值逐位相等或 NovaSwap 已验证。两模型 4 步与 2→4 完整续训仍在执行。

C12 V4 原 100-step 重跑在第 98 步越过此前失败的元数据通信并开始写入完整状态；未修改生产 HCCL/DCP 路径。与 C9 原长程的 98 步 TensorBoard 对照显示前 44 步 loss / grad norm 精确相等，第 45 步起共 54 步存在差异；loss 最大绝对差 `0.00017452239990234375`，grad norm 最大绝对差 `0.005328178405761719`。两次 V4 生产主路径、配置、数据文件哈希、设备组及软件环境的初步核对未见差别，尚未解释首次分歧原因。该对照不作全程一致或保存故障已修复的结论，原始指标已归档；当前运行自身的完整续训验收继续。

C12 V4 第 98 步保存最终失败，runner 返回 1。此次首因不同于 C9 的设备动态库缺符号：rank 0 完成约 37.7 GB 分片后，在 DCP `all_reduce("write", write_data, finish_checkpoint)` 的 `gather_object` 内等待其他 rank，HCCL ALLGATHER（64→512 元素）超过 `train_timeout_seconds=100`。退出时 rank 1/3 仅写约 22 GB，部分 rank 尚未完成，未生成 `.metadata`，不能用于恢复。同期 V4.1 写约 355 GiB 状态是资源竞争的可能因素，但时间重合不能单独证明 I/O 根因。分片尺寸、失败栈、原始日志保留后清理不可恢复的本任务 partial 状态。下一轮仅调整裁剪 Flash 测试配方的通信等待预算至 600 秒；不修改生产默认、训练步数、精度门槛或 suite 总 timeout。

C13 Muon 补充诊断 4/4 完成：两模型三步实际训练和各自连续 4 步 vs 2→4 完整续训均返回 0；wq_b / wo_a 的 BlockShard momentum 必需键存在并精确比较，V4.1 额外覆盖 layer1 Engram weight 与 SparseAdam step / exp_avg / exp_avg_sq。V4 恢复 step3/4 loss 为 `15.90595531463623` / `15.969268798828125`，grad norm 为 `1.0772656202316284` / `1.189597487449646`；V4.1 分别为 `12.280075073242188` / `12.233474731445312` 和 `4.410612106323242` / `4.5374298095703125`，均与连续训练逐位相同。两阶段融合 indexer 审计通过，已保留 metadata、TensorBoard 与日志并清理本任务 RAM 状态。该结果限单卡、小模型和显式诊断配置，不扩展为 Muon EP/Flash、NovaSwap 或旧新优化器数值一致。

Flash 等待预算固定为 C14 `3ec0226e45502481cf779c9e5b6fc31d186dd8b7`，相对 C13 仅变更 `flash and resume` 测试配方的通信超时、对应测试及 Note；生产代码与 C13 相同。相关 runner/config 检查 47 项通过。下一轮 V4 使用独立目录从第 1 步运行原 100-step / resume98；C12 V4.1 同 head 的两阶段训练与恢复指标已经通过，正在做完整状态比较，保留原版本归属。

### 用户调整长程步数为 50

根据本轮明确指令，两模型正式 8 卡长程从 100 步改为 50 步，checkpoint interval / resume step 从 98 改为 48；恢复后精确比较第 49、50 步及完整状态。保持 43/40 层、16 experts、4K、FullAC、纯 eager、600 秒通信等待预算，原 30-step golden、3-step 冒烟和小模型续训步数不变。已完成的 C12 V4.1 100-step / resume98 原 runner 返回 0，完整 DCP 与 indexer 审计通过，保留其历史证据；正在运行的 C14 V4 100-step 将按用户新指令仅停止本任务后重跑。新 50-step 配方的调度器依既有规则响应总步数，因此不把旧 100-step 轨迹当作新配方数值基线。

C14 V4 在完整第 7 步后按用户 50-step 新指令停止：核验本任务进程树后仅向 torchrun 发送 SIGTERM，所有子 rank 与 runner 均退出；日志保留并标为用户要求中止，不归为训练故障。独有且为空的 step98 链接与 RAM 目录已清理。C12 V4.1 同 head 端到端验收的 runner 返回 0，完整 DCP 比较成功；三份 metadata、分片尺寸、哈希、原始 TensorBoard、两阶段 indexer audit、日志及 manifest 均已归档，仅删除通过后的大 checkpoint。

C15 `1932bb7ef9a607ee7980f774e617a818c93f7795` 已在两台验证环境独立检出并核验干净，正式两组 8 卡 50-step / resume48 均已启动；相对 C13 无生产代码变化。完整 pre-commit 再次运行，格式、Ruff、codespell、架构及 OAT 通过，标准 Pyrefly 仍因旧共享搜索路径与既有接口项失败；目标源码下 31 项遗留状态不变。补充双卡复验因设备被其他真实训练占用而等待资源，不干预其他作业。

C15 相关核心产品 CPU 回归 39 项通过（5.16 秒），涵盖此前 33 项模型/MTP/remat/sharding/clipper/FSDP/HF/DCP 检查和 6 项 Muon profile / 专家布局检查。显式指定固定目标上游及插件目录的 PYTHONPATH，使用实际 Torch 2.15 CPU，隔离不可用的 NPU 启动导入；不作为 NPU kernel 覆盖。两组 8 卡状态转存采用固定顺序：V4.1 优先，V4 只在训练子进程全部结束后等待 V4.1 的恢复指标通过且已消费 step48 清理，再继续 CPU 基准转存；未暂停训练 rank 或修改 checkpoint 内容。

C15 V4.1 在第 48 步 native DCP 保存计划分发阶段失败：`reduce_scatter → scatter_object_list` 触发 `libhccl_aicpu_kernel.so / HcclLaunchAicpuKernelA3` 的 AICPU `open so failed` 11002，随后 ACL 507018；目录为空、无 metadata，非容量耗尽或 600 秒等待超时。当前可见设备日志不足以确认本次是否也是此前具体的 `HcommIsSupportHcommAicpuTsTaskCacheClear` 缺符号，故仅记录已证实加载失败。原始 runner、rank0 error.json、TensorBoard、structured logs 已归档；仅清理空 partial，未生成通过标记。V4 尚在训练，其转存等待已解除，未暂停训练进程。

针对重复发生的 HCCL 对象通信加载故障，开始最小同步 native DCP 适配：默认组为 HCCL 时使用独立 Gloo 组承载保存/加载元数据通信，保持训练/DTensor mesh 为 HCCL、native planner/storage 和状态格式不变；HF 与 async 路径不扩大。进程组必须由所有参与 rank 一致创建，仅销毁扩展自身拥有的组，关闭流程需等待既有后台操作并幂等处理。该方案正在实施及验证，尚不能记为修复通过。

同步 native DCP 的独立 CPU 通信组实现已完成并通过独立只读审查。相关 checkpoint CPU 测试共 25 项通过，包含真实两 rank Gloo 的模型、分片 DTensor、AdamW moment、scheduler 和 Python step 回环；确认 save/load 实际传入专属组，重复 close 仅销毁一次且 WORLD 仍可 all-reduce。默认 Gloo、async、创建失败及 WORLD 先销毁边界均覆盖；两模型 HF 与 Host Engram 回归通过。该测试模拟 HCCL 选择条件，实际 NPU/HCCL 与 Gloo 共存仍待验证。专属组采用 Gloo 默认超时，不继承训练 HCCL 的 600 秒；HF、async 和可选 verify_hash_manifest 通信不在本次改造范围，不能称全部 checkpoint 通信均绕过 HCCL。

DCP 修复固定为 C16 `b5ca3542ea251fd4156944732632c0ed6875ff75`（tree `2377f287766d0c872a3881c2c0ab2a08d3e17df5`），两环境独立 checkout 已核验。完整 pre-commit 的格式、Ruff、codespell、架构及 OAT 通过；明确目标上游的全仓 Pyrefly 仍为 31 项，未新增错误。真实 NPU 验证先对两模型 EP2 native DCP 执行连续训练/恢复完整对照，并显式核验 metadata 组为 Gloo、WORLD 为 HCCL，再放行 8 卡 50-step 复验。

C16 V4 EP2 AdamW 连续 4 步 vs 2→4 恢复真实 NPU 验证返回 0；完整 DCP 比较通过。每个 rank 的保存与加载均经外部断言确认 `owned=True backend=gloo world=hccl`。恢复 step3/4 loss 为 `15.979705810546875` / `16.055057525634766`，grad norm 为 `2.5548622608184814` / `2.5621678829193115`，与连续训练逐位相等。metadata、TensorBoard、日志留证后清理大状态；V4 8 卡 C16 长程据此放行。V4.1 的对应小模型完整续训继续验证。

C16 V4.1 EP2 AdamW 连续 4 步 vs 2→4 恢复返回 0，完整 DCP 与第 1/14 层 Engram weight / SparseAdam 必需状态比较通过。两 rank 的保存/加载均确认 owned Gloo、WORLD HCCL。恢复 loss `12.35831069946289` / `12.252204895019531`、grad norm `2.0679025650024414` / `2.112722635269165` 与连续训练逐位相同。C16 V4.1 正式 8 卡 50 / 48→50 据此启动。

C15 V4 连续 50 步及 step48、最终 baseline 保存均成功，但恢复加载后首次反向失败：rank2 设备 MatMulV2 程序申请 `3,327,428` 字节时 `halMemAlloc` 返回 6，后续 BinaryLoad / rtsFuncGetByEntry 207001。未生成恢复 49/50 指标，不计完整续训通过；与此前 AICPU 加载和写后汇总超时分别记录。目前只能确认设备程序分配失败，碎片、allocator 保留及并发因素未完成因果定位。保留两份有效状态，使用 C16 执行 48→50 短恢复诊断并记录加载前后内存，不提前宣称 Gloo 修复该内存问题。该对照为 C15 baseline 加 C16 resume，不冒充同 head 全程 50 步。

C16 核心 NPU 矩阵已完成 14/14，包含两模型单卡、EP2/FSDP2 GA2、无 AC/FullAC、参考及既有融合 attention/indexer 路径和两项完整续训。V4.1 原生全融合 Sinkhorn 的 SDK 限制仍单列，未纳入通过项。HCCL 边界与 C16 Muon EP2 续训继续执行。

C16 从 C15 V4 step48 加载后成功完成恢复第 49、50 步，原始 TensorBoard 精确比较通过：loss `11.601776123046875` / `11.441768646240234`，grad norm `1.5599669218063354` / `2.176398515701294`。实际 metadata 组为 Gloo，未添加 empty_cache 或修改训练计算；尚不能仅据此断言此前设备程序分配失败的唯一原因。消费的 step48 已在元数据/哈希/尺寸归档后释放，两份最终状态仍保留供完整 DCP 比较。该结果继续按跨 head 短恢复诊断记录。

C16 三项 HCCL 边界（empty_expert / all_empty / one_empty）均返回 0。V4 跨 head 短恢复训练返回 0，融合 indexer 审计覆盖 21 pairs / 84 projections 并通过；18,520 个 key 的完整状态比较仍在 CPU 执行。C16 V4 同 head 正式 50 / 48→50 已在独立目录启动，与 C16 V4.1 正式长程并行，原跨 head 证据和正式用例分别记录。

### 用户收缩续训验收为 Muon

用户明确「续训测完 Muon 就够了」。C16 两模型 Muon EP2/FSDP2 GA2 的连续 4 步 vs 2→4 完整续训均已通过，涵盖逐位 loss / grad norm、完整 DCP、BlockShard momentum、融合 indexer 审计，V4.1 额外覆盖 Engram/SparseAdam；实际保存/加载组均为 owned Gloo，训练 WORLD 为 HCCL。因此不再追加任何 AdamW 恢复阶段。正在运行的 8 卡任务只要求完成 50 个更新步、原始指标完整有限及 indexer audit step50 通过；之后按用户缩小范围取消多余的最终 checkpoint 写入/恢复，单独记录 user-scope cancellation，不把取消归为训练故障，也不将原两阶段 suite 的非零退出写作整项返回 0。

C16 V4.1 已独立确认原始 loss 和 grad_norm 均唯一、有限且覆盖 1..50，最后一步分别为 `8.632946014404297`、`3.038093328475952`；step50 indexer 审计通过。取消后续 AdamW 恢复并归档清理。V4 的同 head 50 步训练继续执行。

C16 V4.1 本轮缩减范围已完成收尾：step48 完整 metadata/分片尺寸、raw TensorBoard、indexer audit、日志及 scope-cancellation manifest 均已归档；只清理本任务 step48 的 `380,229,475,034` 字节和 partial step50 的 `289,181,477,723` 字节。未执行 phase1，不产生本轮 50/48 续训结论。当前只剩 C16 V4 的 50 步训练收尾，不追加新的续训用例。

C16 V4 同 head 50 步于本轮完成：原始 TensorBoard loss / grad_norm 均唯一、有限且覆盖 1..50，step50 indexer audit 通过，包含 21 pairs / 84 projections。服务端 scope-stop 在验证这些证据后，仅终止本任务的最终 checkpoint 保存和后续恢复，runner exit143 标为用户范围收缩取消，未执行新的 AdamW phase1。两模型 8 卡 50 步与两模型 Muon EP2 完整续训已满足最新执行范围；原历史 golden 与全仓 Lint 未通过状态不因此改变。

### 本轮指定范围收尾

C16 两模型的 8 卡 50 步训练均完成，原始 loss / grad_norm 覆盖 1..50 且有限，step50 融合 indexer 审计通过。按用户收缩范围有意取消最终保存及后续 AdamW 恢复，两个 runner 的 143 退出均有独立取消记录，不称完整两阶段 suite 返回 0。V4 与此前 C15 的连续训练 50 步原始 loss / grad_norm 全部逐位一致；C15 step48 → C16 的跨 head 短恢复最终 18,520 个 key 精确匹配，作为补充诊断保留。V4 step48 的 metadata SHA256 为 `08cdb0dffac623f5c909db31a84e70774059420735b61e53b1258876116e282e`，归档后释放 `301,867,762,188` 字节 RAM 状态。所有本任务训练进程退出，设备已释放；大 checkpoint 已清理，保留日志、TensorBoard、audit、metadata、哈希及 manifest。

最终 NPU 清单为核心 14/14、HCCL 边界 3/3、两模型 Muon EP2 完整续训 2/2，以及按新范围完成的两个 8 卡 50-step 训练；不追加其他续训。V4.1 reference Sinkhorn 与 65,536 行 Engram 裁剪边界、三组历史 golden 失败和目标 Pyrefly 31 项错误均继续明确保留；本轮范围完成不等于全模型/全特性升级发布验收完成。

### 收尾检查：类型错误归属与 runtime 接线

重新逐项核对 C16 的目标 Pyrefly 输出，31 项错误分布为：V3.2 5 项、V4 LoRA 4 项、V4.1 vision 11 项、Qwen3.5 的 LoRA / CP / 量化配置 7 项、Graph / CV parallel 2 项、CP patch 1 项，以及自动加载的 runtime 带宽 patch 1 项。前 30 项仍属于其他模型或后置能力；不能通过扩大 exclude 或增加 suppression 将全仓 Lint 标为通过，也不将这些错误改称 DSV4 / DSV4.1 文本 eager 训练失败。

Torch 2.15 的 `torch.utils._runtime_estimation` 实际捕获 `get_device_dram_gbps(device)`，已不读取旧 `get_gpu_dram_gbps` 属性。现有 NPU patch 对旧 runtime 属性赋值不会接入真实带宽估算调用，需迁移到 device-aware 接口，并保留非 NPU 设备的原生行为。该问题涉及图调度估算，不能用已完成的 eager 训练作为修复验证；修复后应单独验证调用绑定、设备分流与重复激活，不重跑已完成的 50 步和 Muon 续训。

上述 runtime 接线已在 C16 后的工作区修复：新版 device-aware helper 和 runtime 已捕获的绑定均更新，显式 NPU 或当前 NPU 设备使用既有 1200 GB/s 估算值，其他设备委托原函数。Inductor scheduler / pad_mm 仍使用 legacy helper，因此保留该入口并按当前设备分流；已提前导入的 scheduler 旧捕获仍是既有时序边界，本次未扩展 GraphTrainer 适配。`standalone_compile` 的默认参数语义未变。

隔离子进程 CPU 测试 1 项通过，覆盖真实 `get_transfer_time`、NPU 不调用原 helper、CPU / CUDA / 默认设备委托、重复激活，以及 `donate_graph_module` 默认值和显式 False。完整 pre-commit 的格式、Ruff、codespell、架构与 OAT 通过；标准 Pyrefly 仍受共享旧源码影响失败。独立目标源码检查降为 30 项（173 suppressed），与 C16 逐行比较仅移除该 runtime 错误、没有新增错误或 suppression。记录位于本地验证目录的 `lint-runtime-bandwidth.log`、`pyrefly-runtime-bandwidth.txt`；未修改共享源码 symlink。

该追加修改仅有 CPU 行为与静态验证，尚未在 NPU 上验证图估算；不将 C16 的训练日志改记为追加修改后的 NPU 验证，也不据此扩大 Graph 模式支持。两模型 50 步和 Muon 续训不重跑，历史 golden 的失败状态保持不变。

### 既有 30 步混合精度近似诊断

只读回收旧 `baseline-a/results/v4-original-noac-1p-01` 与新 C6 `results/v4-golden-1p-02` 的原始 TensorBoard。旧、新 event 文件分别为 29,128 字节，SHA256 分别为 `dd28d68a8a0ae712cdb6298e6270d2aa6928a2b53de0f003f3e3084e54f801f7`、`4689770a5a39277e3ab46adc0a842d720c95be7d3a9236b208804398aaeea9e5`。两侧 loss / grad_norm 均覆盖唯一的 step 1..30 且全部有限。使用仓内 `scripts/numerical_alignment/loss_compare.py::compare_metric`，未修改其默认近似阈值 `MEAN_AE <= 0.02`、`MAX_RE <= 0.005`、`MEAN_RE <= 0.005`。

| 指标 | MEAN_AE | MAX_RE | MEAN_RE | 数值近似 | 比特一致 |
| --- | --- | --- | --- | --- | --- |
| loss | 3.236134847005208e-5 | 1.3926711031365217e-5 | 2.821034424404157e-6 | PASS | FAIL，首差 step9 |
| grad_norm | 5.7697296142578125e-5 | 4.8449545608255485e-5 | 1.1334314495977805e-5 | PASS | FAIL，首差 step1 |

独立源码审查确认新版 FSDP 在每个同步 backward 的 `_set_unsharded_grad_dtypes(True)` 中改变真实 unsharded leaf 的梯度 dtype，post-backward 取走梯度后再恢复。GA2 的已累积 preclip 梯度统一转 BF16 无法模拟旧路径，因此未执行这种无效对照。若进一步证明完整轨迹因果，必须先对每个 microbatch 的真实 leaf 做精确限定的外部诊断，验证单步全部梯度和更新参数，再决定是否延长；本次没有实施私有 FSDP 干预。

该结果只量化已经完成的 V4 单卡旧、新 30 步诊断，不是 C16 或追加 runtime 修复后的训练验证，不覆盖另两个 golden，也不替代用户要求的 50 步完成性证据或完整旧、新训练精度验收。三组历史 golden 保持 FAIL，未改阈值、未刷新 golden。完整 raw scalar、比较结果、脚本与哈希保存在本地验证目录 `numeric-30step-audit/`；本次没有启动 NPU 训练或续训。

### 剩余类型错误的适配边界

源码复核将当前 30 项明确区分为 25 项后置能力错误和 5 项 V3.2 训练接口缺口。后置项为 V4 LoRA 4 项、V4.1 vision / SFT 11 项、Qwen LoRA / 量化 4 项、Qwen CP 3 项、CP patch 1 项、Graph / CV 2 项；未发现这些路径进入本轮两模型文本 eager 的自动激活链。V3.2 的 5 项分别涉及 `AuxLoss.inject` 缺少 block-owned routed-token denominator（2 项）、已移除的 `_step_denominator`、已移除的 `MTPDecoder.Config.update_from_config` 和 metadata override 契约。不能补常数或借用 CE token count 来消除这些报错。

原规划对其他模型的要求是共享接口影响审计和导入回归。V3.2 当前导入测试只证明类型、recipe callable 与 FullRemat 绑定，不证明配置构造或训练；上述缺口是恢复其训练支持的前置事项。全仓 Lint 仍未通过，不通过增加排除项或扩展本轮 LoRA / CP / 多模态 / Graph 适配来更改这一事实。

### GA2 单步叶梯度 dtype 反事实

沿用旧 `c422c623` / `ebdfc8` 样本及新 C11 `f552015ace16431b1849a3fe3f003cf1e78743f2` 的单步配方，在独立外部启动器中运行一次诊断，退出 0。配置保持 seed42、纯 eager、无 AC、2048 token / microbatch、4096 token / step、GA2、8-chunk loss、scheduler total100；recipe 只将训练步数设为 1。训练输出和临时目录全部位于 `/data`，未修改插件、上游源码、正常梯度策略或 golden。

干预仅发生在新版 FSDP 每个同步 pre-backward 的 `_set_unsharded_grad_dtypes(True)` 原函数执行后：对先前已定位的 19 个实际 RMSNorm BF16 leaf，将 `grad_dtype` 从 None 设为 BF16。真实参数对象、模块身份、leaf 无梯度、同步 reduction 等断言均成立；False 分支完全保留原行为，不改 reduce dtype，不在累积完成后统一舍入。日志记录 19 个 FQN × 2 次调用，共 38 次命中；GA2 由实际配方和运行日志确认。白名单来自既有差异定位，属于针对假设的诊断，不是独立选择的通用测试。

三方完整比较显示：

- 三方 104 个更新前参数共 80,762,013 个元素全部相等，梯度 key / None 分布、shape、dtype 一致，无非有限值。
- 旧样本 vs 未干预新样本：98 个非空梯度中 19 个张量不同，共 5,887 个元素不同。
- 旧样本 vs 反事实：98 个非空梯度共 80,496,669 个元素全部相等，差异数为 0。
- 三方 loss 均为 `12.271688461303711`；反事实 raw TensorBoard 的 step1 grad_norm 为 `5.2936625480651855`。

旧样本与反事实 `preclip.pt` 的 SHA256 均为 `5ff3854bbd2307c96d4555b615cd6217c09569df524442bde25b8a9a6439db80`，未干预新样本为 `8aff8bb8abb2a98b4cfb3cad023c9432029b06427a5aa5f5dc143542eaae9df2`。本地验证目录 `norm-ga2-counterfactual-01/` 保留脚本、recipe、版本、raw scalar、比较结果、日志和 manifest；原始样本保留在远端 `/data`，原 old / new 样本仅只读。

该结果证明：在这个固定 GA2 首步，改变上述真实叶梯度精度足以消除全部已保存 preclip 梯度差异。没有采集更新后参数或 optimizer 状态，不宣称完整更新、30 步轨迹、EP2 或另外两组 golden 已对齐；也不把 C11 诊断写成 C16 后修改的验证。独立静态审查未发现本次干预范围或时序阻断，确认这些证据边界。本次不延长为 30 步或续训。

### 单步更新与 30 步指标轨迹诊断

在前述首步梯度比较通过后，旧侧与新反事实各单独运行 1 步，在真实 `optimizer.step()` 完成后、scheduler 推进前采集更新后参数及已实例化 optimizer 状态。按参数对象身份映射 FQN，仅规范化 `.moe.` → `.feed_forward.`；更新后 104 个参数共 80,762,013 个元素、294 个 AdamW 状态张量共 160,993,436 个元素全部相等。98 组状态均含 step / exp_avg / exp_avg_sq，另外 6 个无梯度 indexer 参数的空状态位置一致。loss、grad_norm 与参数组超参数均相同。原始 metadata 的 20 处 param_names 迁移差异单独保留；规范化仅用于该字段，双侧唯一性与实际组成员均校验。

独立审查发现初版外部比较器的放行条件遗漏 metadata / loss / grad_norm，已在延长训练前修正为五类结果全部通过，并检查 Tensor 和 Python float 有限值。使用已有快照重新比较通过；内存负例分别注入无穷 lr、不同 loss、不同 eps，均正确返回失败，未修改真实快照。采样与比较脚本、本地结果及负例在验证目录 `norm-ga2-update-01/`，不将这组诊断混入仓内产品 UT。

随后只执行一次新侧 30 步外部反事实，不续训、不重跑旧 30 步，退出 0。沿用相同 GA2 / no-AC / seed42 / scheduler total100；每步严格检查 19 个 FQN 各命中 2 次，30 步共 1,140 次记录均通过，首步更新结果再次与已验证的单步快照精确相等。对比已经保存的旧侧 no-AC TensorBoard：loss 与 grad_norm 两侧分别具有唯一且有限的 step1..30，全部逐步精确相等，最大绝对差均为 0。

旧 event SHA256 为 `dd28d68a8a0ae712cdb6298e6270d2aa6928a2b53de0f003f3e3084e54f801f7`，与前次离线诊断相同；反事实 event 为 `e0b9603b76036bd4f8366333514c289f8320ea6d238fe17eaeef889325b281fd`。本地验证目录 `norm-ga2-30step-01/` 保留三方 raw scalar、比较结果、脚本、日志、版本和哈希。作为第三方历史对照的未干预 C6 仍有 loss 22 步不同、grad_norm 30 步不同，最大绝对差分别为 `0.000152587890625` 和 `0.0002346038818359375`；其源码/runtime/AC 与其他侧的差异继续保留，不将三方对照写成只有一个变量的同版本实验。

结论限于该 V4 单卡配方：恢复旧叶梯度精度足以复现旧侧的已测 30 步标量轨迹。未采集第 30 步的全部参数和 optimizer，不推断最终训练全状态、其他模型、EP2 或原历史 golden 已对齐。生产默认策略继续采用新版 FSDP，原严格 golden 状态不改，近似门槛不放宽。两模型 50 步和 Muon 续训未重复执行。

当前源码已另存本地 C17 快照 `17afdf2a28f9a3645f7668a5211099b5e32953ca`，相对 C16 仅 runtime 带宽接线、其 CPU 测试和证据 Note；正常 Git index / 分支未改变，未推送。C17 没有新增 NPU 训练通过声明，本次因果诊断仍明确使用 C11。

### 最终范围与完成性审计（2026-11-05）

再次核对真实文件与日志：上游 checkout 为固定 `45e33cd` 且干净；当前生产源码及测试相对 C17 无新增差异，仅 Note 更新。C16 回归 manifest 保留核心 14/14、HCCL 边界 3/3 和 Muon EP2 2/2 的命令、退出码及产物路径。两模型 50 步按用户收缩范围完成，主动取消的多余保存/恢复不改写为原两阶段 suite 通过。原三组 golden 的远端退出文件均为 1，runner 明确报 `Loss assertion failed!`；这些属于此前 C6 的历史比较，不是重新执行的 C17 验证。目标 Pyrefly 输出仍为 30 项，其他完整 Lint hook 已通过。

本次目标不能标记为「整轮升级全部验收」：

1. 历史精确 golden 与新版混合精度计算路径的验收边界尚未统一。V4 单卡的旧、新差异已有近似指标与精度干预证据，但反事实不改变默认路径、不替代其余两组 golden，也不授权刷新历史结果。
2. 全仓类型检查与本轮不适配 LoRA / CP / 多模态 / Graph 的范围存在尚未解决的验收边界；V3.2 已完成影响审计，不等于恢复其训练支持。不能通过新增 exclude / suppression 或未经要求扩展训练适配来宣称全仓通过。

同一阻塞已在连续多轮核验中保留；在允许范围内的修复、两模型训练/续训、数值差异诊断及清理均已完成。本次不再重复这些用例。后续需明确历史精确基线的处理规则与后置接口适配范围，或提供能满足原门槛的外部基线/环境；在此之前，整体目标按验收阻塞记录，而非完成。源码、脚本、证据和可恢复的工作区全部保留，未推送或创建 PR。
