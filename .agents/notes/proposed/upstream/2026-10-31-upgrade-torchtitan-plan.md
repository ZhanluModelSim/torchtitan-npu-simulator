# TorchTitan-NPU 升级计划：`dc3985` → `ebdfc8`

## 1. 结论与目标

**这是一次模型接口、训练生命周期和参数布局的整体迁移，不能通过修改依赖 SHA、批量替换 import 完成。**

已确认的分析基线：

| 项目 | 版本 |
|---|---|
| 当前 NPU 插件 | `180c09245c1599802635ccc630175a9b7462406e` |
| 当前适配上游 | `dc3985ad4912f67f71d5a125f2fdf40144ec8354` |
| 目标上游 | `ebdfc8c2a24d89d89c99c0ef96ed8ac149864175` |
| 上游差异 | 215 个提交，764 个文件，新增 66,782 行、删除 36,396 行 |

静态扫描已发现：插件生产代码有 **98 处 import 引用了目标版本已删除或搬迁的 16 个模块**。这个数字还不包含同模块内的符号删除、参数变化和运行语义变化。

本计划采用已确认的边界：

- 仅支持 `ebdfc8`，不保留旧上游兼容分支。
- 现有 NPU 模型、训练功能和扩展全部纳入迁移；新增上游功能不自动扩大支持范围。
- 按新标准保存和恢复 checkpoint，**不兼容旧原生 DCP、旧 optimizer/EMA 状态或旧内部 LoRA 格式，不提供历史转换工具**。
- 保留外部 HF 权重加载、PEFT 导出，以及新格式的保存与续训能力。
- 仅修改插件仓，不修改上游 checkout。
- 阶段一至三已完成；阶段四、五的本轮范围与验证边界见第 7 节。

## 2. 上游变化及 NPU 影响

以下按最终行为归并这 215 个提交，不把文件搬迁和修复提交分别当作独立适配任务。

| 领域 | 上游主要变化 | NPU 处置 |
|---|---|---|
| 模型配置与生命周期 | `#4810` 删除 `ModelSpec`，registry 直接返回模型 Config；模型自身负责 parallelize、pipeline、optimizer hooks、state-dict adapter | **必须迁移**所有本地模型 registry、recipe、模型方法和 override FQN |
| 训练引擎 | `#4773` 抽出 `TrainingEngine`；Trainer 保留数据、验证和报告职责 | **必须重接** TrainerEx、SDC、anticipatory routing、CPU offload、梯度裁剪、GraphTrainer、TorchFT |
| 配置转换 | `#4322/#4648/#4684` 引入统一 `ModelConfigTransform`，CP、LoRA、量化转换调整归属 | 将 NPU 转换接入新模型配置树，明确转换顺序和冲突，不再操作 `model_spec.model` |
| 分布式/SPMD | `#4419/#4905/#4960` 删除旧 activation DTensor 路线，改用 `ParallelismContext`、`local_spmd`；mesh 轴与函数签名改变 | **必须重写语义接入**，不能只把 `ParallelDims` 改名 |
| Linear/FFN/MoE | `#4535/#4808/#4871/#4872` 默认融合 gate/up；删除 `GroupedExperts`，采用 `Linear/GroupedLinear` 的 `w13/w2` | 影响 NPU GMM、SwiGLU、sharding、Muon、量化、LoRA 和 HF 权重映射，是核心工作量 |
| TP 与专家通信 | `#4703/#4704/#4878/#4794` 线性投影显式管理通信；MoE 显式处理 token 布局，废弃纯 TP routed experts | NPU kernel 只替换本地计算，保留外层通信；拒绝不满足新约束的旧配置 |
| Router 与辅助损失 | `#4484/#4631/#4594/#3864` 引入 FP32 router backward、迁移分组路由、支持 padding mask、合入 AuxLoss | 迁移 router 配置和继承关系，统一 AuxLoss；计算变化单独做数值验收 |
| Attention、CP、MTP | InnerAttention 命名统一；`#4617` 分离输入和 metadata 分片；`#4544/#4451` 支持 MTP+CP/chunked loss | 保留本地文档级 metadata、CP carry、压缩注意力和 LI loss，接入新协议 |
| DSV4 上游实现 | `#4855/#4856/#4858` 使用 Attention Gym gather/indexer；`#4879` 拒绝 packed documents；`#4932` 修正 compressed KV 位置 | **不能整体替换本地 DSV4/V4.1 实现**：目标上游尚不等价覆盖本地 packed/CP/indexer 训练能力 |
| 优化器、EMA、DistMuon | `#4904` 引入类型化 optimizer 配置；`#3985` 合入 EMA；`#4811/#4840` 调整 Muon 布局；`#4615` 修复 BF16 状态恢复 | 迁移 HostSparse、Virtual/Swap/CPU-offload optimizer、Muon dedup 和新 EMA 接入 |
| 数据、SFT、验证 | batch 改为 `TrainingMicrobatch`；padding、长文档拆分和恢复状态变化；新增多轮 SFT；验证增加空数据保护 | 迁移本地 loader、视觉数据、调度缓存；单独检查数据语义变化 |
| Checkpoint | `#4190/#4191/#4528` 统一 load 后端、完善 HF consolidation；新 EMA 状态结构 | 迁移 DCP load hook、hash、PEFT、量化读取和导出；默认继续使用 DCP |
| AC 与编译 | `#4497` 引入 torch_remat/RegionAC；FullAC 默认 early-stop；`#4895` 删除常规逐 block compile | 保全既有 NPU FullAC/SelectiveAC 和模型编译功能；RegionAC 不纳入默认支持 |
| GraphTrainer | `#4666/#4884/#4952` 统一 GraphRuntime，收敛到 AOT tracing，删除 graph-based EP chunking | **必须迁移** NPU auto-overlap、runtime context、全图校准和 CV parallel；使用 eager chunking |
| CUDA graphs | 新增梯度累积 capture、graph-owned gradients、独立 optimizer capture、looped PP | 属于 CUDA 能力；NPU 不自动启用，验证不会进入 CUDA 专属执行分支 |
| 模型扩展与工程设施 | Qwen3.5 tying/多模态修复；新增 Qwen3.6 registry；Kimi K3 TP/PP/packing；RL 搬迁与扩展；日志和 CI 重组 | Qwen 现有路径回归；公共修复继承；新模型/RL 专属功能另列后续项，不复制 GPU CI/golden |

关键接口依据见 [模型生命周期](/home/wdy/project/train/tt-lite/adapt_tt_ebdfc8/torchtitan/torchtitan/protocols/model.py:105)、[训练引擎配置](/home/wdy/project/train/tt-lite/adapt_tt_ebdfc8/torchtitan/torchtitan/training_engine.py:110)、[新专家结构](/home/wdy/project/train/tt-lite/adapt_tt_ebdfc8/torchtitan/torchtitan/models/common/moe.py:60)。

## 3. 实施顺序与具体方案

### 阶段一：固定依赖与新配置契约

**目标：建立唯一的新版本配置体系，清除启动阻塞。**

- 同步更新 `requirements.txt`、`pyproject.toml` 中的完整上游 SHA；沿用现有 CI revision helper，不再增加一套版本来源。
- 同步日志、profiler、optimizer、LoRA、quantization、CP、graph 等模块路径；清理对已删除方法和模块的自动导入。
- 配置整体迁移：

| 旧接口 | 新接口 |
|---|---|
| `model_spec.model` | `model` |
| `model_spec.parallelize_fn/pipelining_fn` | 模型方法 |
| `model_spec.state_dict_adapter` | `state_dict_adapter_cls` |
| `optimizer`、`lr_scheduler` | `optim.optimizer`、`optim.lr_scheduler` |
| `training.max_norm` | `optim.max_norm` |
| 旧 EMA 配置 | `optim.ema` |
| `checkpoint` | `checkpointer`，以 `None` 禁用 |
| `parallelism.spmd_backend` | 删除，统一新 SPMD 路线 |
| `comm.mode` | `comm.backend` |
| `module_fqns_per_model_part` | `pipeline_parallel_module_fqns_per_model_part` |
| `compile.enable`、`compile.components=model` | 新的 loss 编译配置与独立 NPU 模型编译配置 |

- 更新配置 converter，使其能转换嵌套的 `optim` 等组件；保留 recipe 的专用类型和配置值，避免字段筛选时静默丢失。
- NPU 配置转换必须在模型构建前完成；所有 override 从 `model_spec.model...` 迁到 `model...`，并验证实际命中对象。
- 固定 `torch_remat` 为目标上游指定的 `d302699b1c58f83fa2c7b03bc2593967e9530335`，补齐 `renderers==0.1.11` 等依赖。

**依赖特例：Attention Gym。** 上游精确要求 `0.0.14`，插件要求 `0.0.16`，这是已有的依赖冲突。计划保留插件当前需要的 `0.0.16`，沿用受控的 `--no-deps` 安装方式，显式安装运行依赖并验证相关 API；同步修正安装文档，不能继续声称普通 pip 解析无冲突。`0.0.16` 与新上游调用的运行兼容性作为第一阶段验收门槛。

**阶段出口：** 包可导入，全部现有 recipe 和显式 override 可构造；旧配置报出明确错误，不建立旧接口别名层。

#### 阶段一实施小结（2026-10-31）

已固定目标上游 `ebdfc8` 并完成新配置契约；按确认后的范围，提前迁移了整包导入、recipe 构造和启动脚本所需的模型接口、递归 `_parallelize`、HF/PEFT 加载及 override。NPU 模型编译配置最终采用 `compile.model`；可选 `compile`、`checkpointer` 保留 `None` 语义，模型单独编译通过显式空 `components` 保持原行为。

- **配置验收：** 47 个 recipe、1 个模板共 **147/147** 项通过；**35/35** 组实际启动参数通过 CLI、token 预算、模型更新和真实 override 检查。
- **局部回归：** 集中测试 **180 项通过，0 失败、0 跳过**；另有真实 HF/PEFT 加载恢复 **15 项**、SwiGLU 与 dispatcher 组合 **14 项**通过。验证在指定容器和 1015 CANN 环境进行，结果不代表 NPU 训练通过。
- **Lint 状态：** 完整 pre-commit 尚未通过，共享旧上游路径下有 **278 条类型错误**；显式指向 `ebdfc8` 后有 **25 条错误、1 条警告**，其中环境/API 版本问题 9 条、运行迁移问题 9 条、其他静态解析与类型问题 7 条。其他适用检查通过，OAT 因无暂存文件未实际扫描。
- **剩余范围：** CV 配置与注册通过，但目标上游已删除其依赖的 graph chunking，运行入口明确拒绝尚未迁移的路径。Graph、anticipatory routing、TorchFT 恢复、Qwen CP 等运行适配仍需后续完成；本轮未验证 NPU 训练、精度、收敛或性能，不能据此宣布整体升级完成。
- **交付状态：** 上游源码未修改，插件改动未提交或推送；临时验证目录已清理，无残留任务进程，持久依赖版本及入口文件哈希未改变。

完整结果见[阶段一报告](../../../../test_reports/stage1-ebdfc8/REPORT.md)，最新证据以[结果索引](../../../../test_reports/stage1-ebdfc8/LATEST.json)为准。上述证据保存在本地，未纳入版本控制。

### 阶段二：迁移模型结构与 NPU 计算接口

**目标：采用新参数标准，同时保持 NPU 算子的计算含义。**

- Dense FFN 采用 `w13:[2,F,D]`；专家采用 `w13:[E,2,F,D]`、`w2:[E,D,F]`。
- NPU Linear、LoRA 和量化实现接入 `_linear`；NPU GMM 接入 `_grouped_mm(input_RI, weight_EOI, offsets_E)`。TP/SP 通信由上游外层负责，避免漏通信或重复通信。
- SwiGLU/clamp 接入 activation 配置；移除 `GroupedExperts`、`inner_experts` 和旧参数名依赖。
- 保留本地 DSV 路由分数在 **W2 前相乘并回到 BF16** 的语义；dispatcher combine 不得再次乘分数。上游 W2 后的 `output_postprocess` 不能替代此行为。
- Router 改用类型化 activation 配置；DSV3.2 继承新的分组路由实现，V4/V4.1 删除对通用 router 已移除字段的访问。
- 全部 transformer/MTP 路径传递真实 `padding_mask`；padding 不计入负载统计和辅助损失，但不能破坏 dispatch map。
- 统一使用上游 AuxLoss，迁移 DSV3.2 indexer loss、metrics 和 anticipatory 快照，避免两套分母与累加器并存。
- 更新 HF adapter、Muon 参数筛选、初始化、量化白名单和 LoRA target，确保它们共同使用新 FQN。

新分组线性层的存储与计算视图见 [GroupedLinear](/home/wdy/project/train/tt-lite/adapt_tt_ebdfc8/torchtitan/torchtitan/models/common/linear.py:232)。

**阶段出口：** Dense/MoE 局部前反向、参数映射、override 命中及新 HF roundtrip 通过。

### 阶段三：迁移训练引擎与数据协议

**目标：让既有 NPU 功能在新的实际执行位置生效。**

- 通过 `Trainer.engine_cls` 接入 NPU TrainingEngine；Trainer 保留数据调度和报告，Engine 接管模型执行、优化、checkpoint 和清理。
- 将 HF32、CPU-offload 初始化、pinned pool、SDC、anticipatory 微批次回调迁到对应生命周期。
- 删除对旧 `Trainer.train_step` 的源码字符串注入和函数 globals 替换；梯度裁剪放到 NPU `Optim` 的更新边界，包含 Engram sparse gradient 和 offload 参数。
- 本地 loader、视觉 loader 和缓存改为返回 `TrainingMicrobatch/TokenizedTrainingMicrobatch`，维护 labels、positions、padding mask、有效 token 数和 loss 参数。
- SFT 的 prompt loss mask 与 padding mask 分开处理，不能把 `labels == IGNORE_INDEX` 当作 padding。
- CANNProfiler、metrics、SDC 和 close 路径跟随新状态 owner；正常结束、异常结束和重复构造均释放自身资源。

当前旧 patch 在导入时直接读取被移除的方法，见 [旧 Trainer 注入](/home/wdy/project/train/tt-lite/adapt_tt_ebdfc8/torchtitan-npu/torchtitan_npu/patches/torchtitan/trainer.py:28)。仅增加属性转发无法修复这类生命周期变化。

**阶段出口：** 单卡 eager 训练、梯度累积、数据恢复、裁剪与资源清理通过。

### 阶段四：迁移 SPMD、FSDP、CP 与 PP

**目标：保留现有受支持并行组合，采用新上游分片规则。**

- 使用 `ParallelismContext`、`DistributedTopology`、`activate_spmd()`；专用 HCCL EP group wrapper 接受新的 `dense_sp_enabled` 等参数。
- 数据与指标使用 `dp` mesh；dense/expert FSDP 分别经上游 mesh resolver 获取。不能机械地把旧 `fsdp` 字符串替换为 `dp_shard`。
- `LocalMapConfig` 改为 `local_spmd` 和明确的输入、输出 sharding；验证梯度 placement。
- 重建 NPU FSDP 扩展，只保留 Engram `ignored_params`、CPU offload、FP32 policy 等必要差异，避免旧完整实现覆盖新 stacked-weight 和 multimodal 修复。
- 纳入目标提交 `#4949`：专家数量足够时保留专家维 `Shard(0)`，其他情况使用正确的矩阵行分片；额外覆盖 EP=1 的 eager/graph 一致性。
- CP 改用新 load-balancer 配置、permutation 和 `shard_tensors`；保留 NPU geometry-keyed metadata、doc-chunk、不同压缩比和 carry exchange。
- tokens、labels、positions、padding mask、MTP 各深度使用同一文档布局；metadata 只转换一次。
- PP 接入模型方法及新 first/last-stage module 配置，保留现有多模态分段和微批次语义。

**不能直接删除 metadata patch 组。** 插件已回移部分晚于目标提交的 metadata 协议，`ebdfc8` 尚未完整覆盖。此次只删除旧 `cp_shard` 接口包装，保留目标上游缺少的实际能力。

**阶段出口：** 现有 DP/HSDP、TP/SP、EP、CP、PP 组合通过真实 NPU 测试；原先明确不支持的组合继续明确拒绝。

### 阶段五：迁移优化器、checkpoint、LoRA 和低精度

**优化器与 EMA**

- 用类型化 optimizer configs 替代 `ParamGroupConfig` 和字符串 factory；保证所有可训练参数按 first-match 规则恰好归属一个 optimizer。
- 迁移 HostSparse、Virtual、Swap、CPU-offload container，并接受新的构造参数。
- DistMuon 使用新的矩阵视图和 `lr_reference_shape`；replica dedup 接入新构造入口，保留 owner-only state 与广播顺序。
- 复用上游 EMA，删除整套 EMA Trainer/checkpointer backport；仅保留 NPU stream/event/offload 适配。目标上游 EMA offload 仍直接使用 CUDA，见 [EMA offload](/home/wdy/project/train/tt-lite/adapt_tt_ebdfc8/torchtitan/torchtitan/components/optim/ema.py:326)。

**Checkpoint**

- 按新 `_load_checkpoint(states, checkpoint_id, *, from_hf, from_quantized)` 接入 hash、量化 reader 和 PEFT 分层加载，区分 live state 与 flattened state。
- 保留默认 DCP、同步写入的 NPU 参数及异步完成后的 hash 发布顺序；导出工具改读新 EMA 状态。
- 使用新标准验证 model-only、full resume、EMA、adapter-only 和量化状态；旧内部格式不静默 partial-load。
- 不默认切换到 `TorchCheckpointingManager`：目标版本支持 HF 最终导出，但其 HF load 仍有限制。

**LoRA 与量化**

- 使用新 `LoRATransform` 和 Linear/GroupedLinear handler，采用新 `w13` 参数化。
- 保留插件的 HF base 初始化、adapter-only save/resume 和 PEFT export；上游新 LoRA 仍未完整提供这些功能。
- TorchAO-NPU 改接新 Linear/GroupedLinear，逐一保全 MXFP8、HiF8、block-FP8、MXFP4-QAT、FSDP prequantize、KV/LI 量化。
- 不将 stack axis 当作矩阵行，也不把 LoRA 参数误纳入原基础权重量化规则。

**阶段出口：** 新格式保存后重启，下一步权重、optimizer state、LR、EMA 和步数正确；HF/PEFT 导出能够实际重新读取。

### 阶段六：迁移 AC、模型编译与 GraphTrainer

**常规训练**

- 保留 FullAC/SelectiveAC，继承上游 early-stop 和 effect 保护。
- 检查重算期间 router counter、AuxLoss、RoPE cache、Engram 状态只更新一次，DeepEP handle 正常释放。
- 上游 `CompileConfig` 现在只允许 loss component，仍保留 backend 字段。既有 NPU 模型编译已在阶段一迁到显式 `compile.model` 配方配置，保留现有编译参数和 A3 CP eager 限制，不复活旧上游 `apply_compile`。

**GraphTrainer**

- 迁到 `GraphTrainingEngine` 与 GraphRuntime，重新接入 trace/runtime context、全图校准、NPU pass registry。
- 删除 graph-based EP chunking，统一 eager chunking；更新 launcher 中旧 strategy 和执行模式配置。
- auto-overlap 放在 eager chunk metadata 已产生、FSDP bucketing 完成后的调度位置，不再依赖已删除的 concretization pass。
- CV parallel 单独迁到新的 graph builder/runtime 接口；不能只修默认 GraphTrainer 就宣称该能力已保全。
- 校准不得改变模型参数、grad、buffer、随机状态或训练进度。
- NPU 默认关闭 CUDA graph capture；新增 WGrad fusion、RegionAC 等优化不混入基础兼容性验收。

**阶段出口：** eager 与 graph 的现有训练组合、梯度累积、EP overlap、校准和 CV parallel 均有独立结果。

### 阶段七：迁移 TorchFT、多模态并清理补丁

- TorchFT 接入新的 FaultTolerantTrainingEngine，重建明确的 NPU engine 构造入口；不继续依赖旧 Trainer 多继承关系。
- 保留 HCCL 恢复、snapshot/rollback 和事务顺序：跨副本指标 collective 在投票前完成，失败尝试不推进 LR、step 或数据位置。
- 保持现有 TorchFT 支持边界，不扩展 TP/CP/PP。
- Qwen3.5 继承小模型 embedding tying，验证共享参数身份；检查不同 rank 混合 text/image/video 时的 FSDP collective 顺序。
- GDN 覆盖末尾重复 offsets、空 padding segment 和文档边界；不把上游 Ulysses 新增当作 GDN CP 已支持。
- 当前名为 Qwen3.6 的 launcher 实际调用 Qwen3.5 recipe：此次保持其实际模型行为并修正文档，不静默切换到新上游 Qwen3.6 registry。

补丁按以下结果处置：

| 处置 | 内容 |
|---|---|
| 删除已吸收实现 | AuxLoss 重复实现、EMA 整类 backport、DistMuon 非 CUDA storage 校验补丁、EP ready-node 去重，以及已吸收的 TorchFT 修复 |
| 随旧功能删除 | graph EP chunk concretization、shape-live-out、旧 forward-accumulation 补丁及专属入口 |
| 保留并重接 | metadata/decoder/MTP、router score 前置乘法、Muon replica dedup、EP shape-query、recompute functionalization、chunked-loss NPU 差异 |
| 不随此次升级删除 | torch_npu runtime patch、针对 PyTorch 的 filesystem mmap patch |
| 作为既有债务处理 | Qwen 旧 2D MRoPE patch；目标上游已具有 folded/video 行为，回归通过后退役 |

## 4. 验证计划

### 分层验证矩阵

| 层级 | 必须覆盖 | 通过条件 |
|---|---|---|
| 安装与配置 | 固定 SHA、运行依赖、默认 import、所有现有 recipe/override、可选组件 | 无过期模块引用；配置字段不丢失；可选组件不阻断默认启动 |
| CPU 行为 | 配置转换、参数归属、FQN、HF roundtrip、LoRA trainable 集合、tying、microbatch、metadata permutation | 验证实际行为，不以符号存在或 mock 导入替代 |
| NPU 算子 | dense/grouped W13、SwiGLU clamp、pre-W2 score、router、packed attention、GDN | 前向与关键梯度满足相应数值标准 |
| 模型训练 | DSV3.2、V4、V4.1 文本/视觉、Qwen3.5 dense/MoE、现有 Qwen3.6 launcher 的实际配方 | 每条既有能力有明确结果；基线未验证项单独标记 |
| 并行 | 当前支持的 DP/HSDP、TP/SP、EP、CP、PP 组合，CPU offload、Engram | 无 collective 次序错误；参数与梯度 placement 正确 |
| Graph/AC | eager、FullAC、SelectiveAC、GraphTrainer；累积 1/>1、overlap、CV、校准 | 状态更新次数正确，校准无训练副作用，无 handle/显存持续增长 |
| 保存与恢复 | 新 DCP 同步/异步、EMA、AdamW/Muon、HostSparse、LoRA、量化、hash | 重启后的下一步与连续运行一致；坏 hash 明确拒绝 |
| HF/PEFT 导出 | text-only、多模态、MTP 开关、EP、多 rank、未加载的可选 key | safetensors 实际重开成功，index 与张量集合正确，逐张量比较 |
| TorchFT | 成功提交、失败回滚、重试、rank 重入、数据与 checkpoint 步数 | 失败不推进状态，恢复后事务与 collective 顺序正确 |

### 数值验收规则

- 使用相同外部 HF 权重、固定随机性和同一保存的 microbatch 输入比较；无需恢复旧 DCP。
- 纯接口、metadata 或存储重构：要求未截断的逐步 `loss`、`grad_norm` 比特一致。
- fused GEMM、router FP32 backward、embedding tying 等计算变化：与对应参考实现比较，并进行不少于 **100 step** 的代表性训练；不能把变化混入「纯重构」后放宽判据。
- 数值近似遵循仓内标准：logits 的 `MEAN_AE/MEAN_RE ≤ 0.02`、`COS ≥ 0.9999`；loss/grad_norm 的 `MEAN_AE ≤ 0.02`、`MAX_RE/MEAN_RE ≤ 0.005`，且无 NaN/Inf。
- 不使用 `deterministic_warn_only`；stdout 四舍五入结果不作为比特一致证据。
- 数据 packing 行为变化另验，避免把不同输入造成的 loss 差异归因于模型迁移。
- 不复制上游 GPU golden 作为 NPU 正确性证据，不把 FakePG 或单卡导出结果外推为多卡训练通过。

### 本机与服务器执行约束

- 本机完整 lint 使用约定的 `torchnpu-x86-build` 环境和完整 pre-commit 命令。
- 保持约定的 `/tmp/torchtitan` 指向；另对当前目录的 `ebdfc8` 源码执行明确 search-path 的类型检查，报告实际解析的上游 SHA，避免检查到另一份 checkout。
- CPU 产品 UT 与 tooling 自测分别报告；完整 lint 只有全部 hook 通过才能标为通过。
- 真正执行远程验证前再确认服务器，默认容器按现有约定；先检查 NPU 占用，记录具体命令、环境与代码版本，测试后清理临时资源。

## 5. 交付与完成标准

实施按上述七阶段形成可独立审查的提交组，全部完成后再将升级视为可交付。最终交付包括：

- 统一固定到 `ebdfc8` 的插件代码、依赖、CI、recipe 和 launcher。
- 一份上游变化处置表：每项标明「已迁移、直接继承、补丁退役、新增功能暂不支持」。
- 新配置和 checkpoint 的升级说明，明确旧内部格式废弃。
- 各模型与扩展的实际支持矩阵，区分已验证能力、原有迁移债务和本次新增失败。
- CPU、NPU、数值、保存恢复和 TorchFT/Graph 专项记录，注明未执行项与限制。

**最终验收重点是：按新上游标准恢复现有 NPU 功能，并证明它们在新执行路径上确实生效。仅包能导入、单个 debug 模型跑通或局部测试通过，都不足以宣布升级完成。**

## 6. 审查后待完成项（2026-11-02）

以下两项基于插件提交 `4671d46`、`de312d5` 的审查，状态均为「待完成」。本轮仅修正 decoder FSDP 分片规则，不修改量化或 TorchFT 实现；后续适配 v0.4.0 时继续处理。

### 6.1 Shared-expert W13 量化白名单

- **现状与影响：** `extensions/adaptor/torchao/configs/deepseek_v4.py` 的 `_DENSE` 仍使用 `shared_experts.w1/w3`，模型已经采用融合的 `shared_experts.w13`。V4.1 复用该白名单，导致两种模型的 shared gate/up 投影被量化 recipe 和默认 FSDP prequantize 白名单遗漏。
- **处理方向：** 将目标迁到 `w13/w2`，同时核对实际模型配置树、量化转换器和 prequantize 过滤结果。不得仅更新字符串后就宣称量化路径已验证。
- **完成条件：** `all_mxfp8`、`all_hif8`、`mix`、`all_block_fp8` 的转换结果实际包含预期 shared-expert 投影；默认 prequantize 目标与 recipe 一致；融合 W13 的 shape、dtype、前向和反向经过相应 NPU 验证。
- **代码依据：** [V4 量化配置](../../../../torchtitan_npu/extensions/adaptor/torchao/configs/deepseek_v4.py)、[V4.1 量化配置](../../../../torchtitan_npu/extensions/adaptor/torchao/configs/deepseek_v4_1.py)。

### 6.2 TorchFT 训练生命周期迁移

- **现状与影响：** `FaultTolerantTrainerEx` 仍通过旧多继承结构使用普通 NPU engine，并访问旧 Trainer 的 `ft_manager`、`model_parts`、`lr_schedulers`。本地训练循环仍消费 tuple batch 和旧配置字段，尚未接入 `FaultTolerantTrainingEngine`。配置可构造不代表该路径可运行。
- **处理方向：** 在恢复运行支持前增加明确的未迁移入口检查；随后围绕目标版本的 FT engine 接入 NPU/HCCL 恢复，迁移状态 owner 与 `TrainingMicrobatch`，删除旧训练循环。保持跨副本指标 collective 先于提交投票、失败尝试不推进训练状态的原有约束。
- **完成条件：** 实际创建 FT engine；正常 step、失败回滚、重试和 rank 重入均进入新路径；验证失败尝试不推进 optimizer、LR、训练步数或数据位置，并验证 checkpoint 恢复。恢复被延期的 TorchFT 测试收集后，执行真实 NPU/HCCL 验证。
- **代码依据：** [TorchFT trainer](../../../../torchtitan_npu/extensions/torchtitan/experiments/torchft/trainer.py)、[恢复逻辑](../../../../torchtitan_npu/extensions/torchtitan/experiments/torchft/recovery/step_recovery.py)、[测试延期清单](../../../../tests/unit_tests/conftest.py)。


## 7. 阶段四、五实施范围（2026-11-02）

本轮继续固定 `ebdfc8`，不切换到 v0.4.0。Checkpoint、LoRA 和低精度由后续专项处理。按本轮确认，NPU 验证只保留 DSV4/V4.1 的基础 eager；EMA、CPU offload、Muon、TP/CP、Host Engram 和 PP 不列入本轮运行验收。阶段四、五原有完整出口仍未达成，不能将基础 eager 冒烟扩大为全部并行和优化器组合已通过。

### 7.1 已实施的适配

- **MoE 与 FSDP：** 保留上游分片配置、mesh resolver 和 FSDP 生命周期。`ebdfc8` 的 EP=1 routed experts 未声明权重 SPMD 状态，新的 FSDP `dp_mesh_dims` 入口无法消费普通权重；公共模型配置层仅补充 `w13/w2` 的 dense replicated 声明。上游补齐后可删除该包装。
- **多模态 FSDP：** 删除本地视觉 encoder 的重复 helper，调用上游 `apply_fsdp_to_multimodal_encoder`，传递 `dp_mesh_dims`。本轮未验证视觉训练。
- **分布式初始化：** 当前 torch_npu 配套 PyTorch 缺少 `pipeline_per_edge_p2p`。由 `extensions/torchtitan/distributed/utils.py` 单独拥有 `init_distributed` 适配，先于 trainer 激活；具备能力的 runtime 直接委托上游，旧 runtime 仅保留 PP=1 初始化并明确拒绝 PP。适配没有创建虚假的 runtime 配置字段。更新配套 runtime 后可删除降级分支。
- **类型化优化器：** Virtual/Swap/CPU-offload container 接受原生构造参数；CPU-offload 通过 typed optimizer config 的 `build` 接口接入上游 first-match 参数归属循环，保留 HostSparse 独立 owner。
- **DistMuon：** CPU-offload 更新改用 `lr_reference_shape`；两个模型复用公共 expert layout，显式声明 EP 与 EDP 同维分片次序。
- **EMA：** 复用上游调度、参数/缓冲区管理和状态结构，仅替换设备 stream/event 与 CPU offload 更新；配置转换按最近注册的 Trainer 基类选择适配器，保留具体 recipe owner。后者修复 NPU recipe 中嵌入原生 EMA.Config 时漏转、落入 CUDA stream 的问题；NPU EMA 不在本轮最终验收范围。

### 7.2 基础 eager 验证边界

入口为 `tests/integration_tests/run_tests.py --test_suite deepseek_eager_parallel`。固定 2 个 NPU、3 个 step、每 rank microbatch 256 tokens、每 step 512 tokens，关闭编译、activation checkpoint 和 checkpoint，使用 AdamW 和 reference 算子。DSV4.1 显式关闭 Engram。

- DSV4 EP1/FSDP2：已完成 3 个 step。
- DSV4 EP2/FSDP2：已完成 3 个 step。
- DSV4.1 EP1/FSDP2、无 Engram：已完成 3 个 step。

三个 case 的 runner 均返回 0，并从 TensorBoard 校验恰好包含 step 1、2、3，loss 有限且 step 不重复；另行读取的 grad_norm 也均为有限值。最终执行日志为 `base-eager-final.log`。

上述冒烟仅证明前向、反向、optimizer step 和训练结束流程可执行，不作为精度或性能结论。详细日志保留在本地 `test_reports/stage45-ebdfc8/`。

后续对比了插件 `180c092` / 上游 `dc3985` 与当前工作区 / 上游 `ebdfc8`：DSV4 EP1、EP2 和关闭 Engram 的 DSV4.1 EP1 均使用相同初始权重、输入和确定性设置运行 100 步。逐步 `loss`、`grad_norm` 满足仓库数值近似阈值，但不满足比特一致。DSV4 和 DSV4.1 EP1 首步最后一层的分支消融确认：上游 shared/routed expert `w13` 融合改变反向求和顺序；DSV4 四层临时拆分后，两步 loss 与 grad_norm 逐位一致。该定位尚未扩展到 EP2、DSV4.1 全层或完整 100 步。该结果仅覆盖双卡 debug model、序列长度 256、AdamW 和 reference 算子，不代表预训练权重、Flash 全规模或其他功能已通过。数据、权重验证方法及逐步指标见本地 `test_reports/stage45-ebdfc8/precision-report.md`。

CPU 回归覆盖配置 owner 保留、类型化 optimizer 构造、EMA 更新调度、FSDP expert storage、Muon layout 和分布式初始化：77 项通过，1 项因当前 runtime 不具备 per-edge PP 能力而跳过。SPMD/FSDP 存储测试实际调用模型 `parallelize`，fake process group 仅提供 CPU mesh，不作为 collective 验证。

完整本地 pre-commit 的格式、架构和其他 hook 通过，Pyrefly 未通过。默认 `/tmp/torchtitan` 按仓库约定指向共享旧基线；显式使用本工作区 `ebdfc8` 后仍有 21 项类型错误，均位于本轮未修改的文件，涉及 GraphTrainer、CP、LoRA、量化和旧 runtime 符号。本轮不将完整 Lint 标为通过。

### 7.3 后续验收

恢复扩展验证时，分别检查 HSDP owner-only Muon state 与广播、CPU-offload 的 optimizer 更新和资源释放、EMA 混合 CPU/NPU 参数更新、TP/CP 文档布局与梯度，以及 Host Engram。不能使用 CPU mesh 元数据测试替代真实 NPU collectives。

旧 integration case 的 `--optimizer.*`、`--checkpoint.*`、旧 dataloader 字段与 `model_spec` 仍需由对应专项迁移。现有 PP runtime 边界和 V4.1 的 TP/PP 显式限制保持可见。Checkpoint、LoRA、低精度及第 6 节的 TorchFT 债务继续延期。
