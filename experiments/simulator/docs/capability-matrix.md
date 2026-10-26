# Simulator 迁移能力矩阵

Kimi K3 新增范围及当前源码验收另见 [Kimi 接入与验证](kimi-k3-migration.md)。以下历史全规模矩阵仍专指 DSV4。

2026-10-23 已补齐 FSDP stream/event 控制依赖，四组配置复验通过。当前图包含零耗时 `stream_wait` 节点，上游必须保留其前驱约束。详见 [修复与复验](fsdp-dependency-audit.md#修复与复验) 和 [接入合同](stream-control-dependencies.md)。

当前实现已取消图折叠与 mem 以外的导出，FSDP 模板改写和恢复补偿也已删除，见 [精简记录](simulator-cleanup.md)；精度边界见 [mem 与精度审计](mem-only-precision.md)。以下 21 项全规模结果属于此前源码快照，不是本次源码的全规模重跑。

原生 dispatch 优先收敛阶段的复验与精确结果见 [收敛记录](dispatch-first-remediation.md)。本轮七个 preset × 三种 AC 共 21 项全规模 Meta 复验及独立导出／分阶段内存审计全部通过。表中「Meta 已验证」仅指固定软件栈下可执行的仿真路径；不代表 NPU 数值或性能已验证。
该阶段精确结果见 [本轮结果](dispatch-first-results.json)，复现方式见 [收敛记录](dispatch-first-remediation.md#镜像与复现)；本文末尾与 `s5-results.json` 保留历史对照。

| 能力 | 当前验证范围 | 状态／边界 | 证据入口 |
| --- | --- | --- | --- |
| 原生配置、override、显式激活与恢复 | native registry；正常／异常退出；普通导入隔离 | Meta 已验证 | [test_config.py](../tests/test_config.py)、[test_runtime.py](../tests/test_runtime.py)、[test_patch_scope.py](../tests/test_patch_scope.py) |
| attention/compressor/LI/indexer loss/mHC/MoE/shared expert | 七个 preset 的完整模块 F/B/O | 全规模 Meta 已验证 | [test_dsv4_capture.py](../tests/test_dsv4_capture.py)、[全规模结果](dispatch-first-results.json) |
| MTP 与主 loss、clip、AdamW、LR scheduler | MTP 0/1；完整梯度清单和更新集合 | Meta 已验证；不含真实数值 | [test_dsv4_capture.py](../tests/test_dsv4_capture.py)、[test_optimizer_contract.py](../tests/test_optimizer_contract.py) |
| tensor/storage/generation/alias/mutation | CPU、Meta、DTensor、原生保存值 | 已验证；不伪造同形保存张量 | [capture/test_saved_tensor_identity.py](../tests/capture/test_saved_tensor_identity.py)、[test_s4_mxfp8.py](../tests/test_s4_mxfp8.py) |
| DP/FSDP/HSDP/EFSDP、EP、TP/SP、CP | 11 个轴／组合；rank 1/8/15 | Meta 已验证；通信由 fake process group 承载 | [test_dsv4_capture.py](../tests/test_dsv4_capture.py) |
| AG/RS/AR/A2A/wait 与优化器依赖 | 原生 functional/c10d/DTensor transport；全部 chunk、实际 process group；归约→clip→optimizer | Meta 已验证；不替换通信 API/梯度；Gloo 两 rank 契约另验，无真实 HCCL 性能 | [test_dsv4_capture.py](../tests/test_dsv4_capture.py)、[capture/test_fsdp_residency.py](../tests/capture/test_fsdp_residency.py) |
| FSDP stream/event 等待与同流顺序 | FSDP／HSDP／组合并行／SAC+MXFP8；事件快照、延后传播、独立流重叠 | Meta 已验证；host synchronize、轮询和真实设备时序不在范围 | [流控制依赖](stream-control-dependencies.md)、[原生回归](../tests/test_fsdp_stream_capture.py) |
| Full/Selective AC 与保存生命周期 | none/full/selective；early stop、重算、save/unpack/release | Meta/CPU 已验证 | [test_s3_native_lifetimes.py](../tests/test_s3_native_lifetimes.py)、[memory/test_s3_lifetime_oracles.py](../tests/memory/test_s3_lifetime_oracles.py) |
| Selective save-set | default/gmm/quant-mm/comm/none 等已公开选择 | 有限支持；未命中 collective 时 warning；特殊 fake A2A cache 后置 | [test_s3_native_lifetimes.py](../tests/test_s3_native_lifetimes.py)、[test_s4_mxfp8.py](../tests/test_s4_mxfp8.py) |
| MXFP8 | 原生 TorchAO-NPU recipe；MM/BMM/grouped MM 前后向、真实 code/scale | Meta 已验证；不额外量化原生 recipe 未选中的 MTP 投影 | [test_s4_mxfp8.py](../tests/test_s4_mxfp8.py) |
| shard dtype 与保存激活 offload | BF16/MXFP8、三种 AC 的内存投影 | 有限支持；不执行 offload／传输建模 | [test_s3_native_lifetimes.py](../tests/test_s3_native_lifetimes.py)、[test_memory_delivery.py](../tests/test_memory_delivery.py) |
| GBS/GA | 批配置和 token 分母保留 | 单 microbatch 模板；无完整多 MB 峰值回放 | [test_s3_native_lifetimes.py](../tests/test_s3_native_lifetimes.py)、[test_memory_delivery.py](../tests/test_memory_delivery.py) |
| typed model overrides | 原生配置树重建，构建和量化 F/B/O | 已验证；非支持字段拒绝 | [test_model_overrides.py](../tests/test_model_overrides.py) |
| L0→L1→L2→L3 | 稳定模板 ID、全局 DAG、双向边、DataSlot/实例引用、通信归属 | 全规模 Meta 已验证 | [validation.py](../torchtitan_simulator/validation.py)、[test_memory_delivery.py](../tests/test_memory_delivery.py) |
| mem 导出与完整原始图 | CLI／simulate() 自动写 summary JSON；mem 增加明细；view/alias 不折叠 | Meta 已验证范围见本轮审计；其余图导出移除 | [test_memory_delivery.py](../tests/test_memory_delivery.py)、[本轮审计](mem-only-precision.md) |
| 原生精度配置 | base FP32，param BF16/FP32，reduce FP32 | DSV4 base BF16 完整捕获拒绝；不支持 FP16/低精度 reduce | [test_precision.py](../tests/test_precision.py)、[精度边界](mem-only-precision.md) |
| 稳态优化器状态驻留 | 实际本地 state storage 去重；前向／反向／优化器及总峰值 | Meta 投影；六项预算加逐层 offload payload 列表；未知 owner 不猜测 | [稳态内存](steady-state-memory.md)、[状态测试](../tests/memory/test_optimizer_state.py) |
| 逻辑内存估算 | 实际 storage/保存槽/FSDP 区间、独立峰值求和与去重 | Meta/CPU 已验证；不含 allocator/workspace | [内存测试](../tests/memory/)、validation.py |
| storage 释放检测 | 弱引用候选队列＋原生过期确认；每个 dispatch 边界；不支持弱引用时退回原生逐边界检查 | CPU／Meta、四并行／AC 组合及完整 Flash 已验证；不改变内存语义 | [生命周期优化](storage-lifetime-optimization.md)、[逐边界测试](../tests/test_storage_lifetime_capture.py) |
| PP | 原有结构单测保留 | 低优先级；训练入口拒绝 | [test_config.py](../tests/test_config.py) |
| FP8 EP dispatch／FSDP AG | 与 MXFP8 计算分开 | 低优先级；入口拒绝 | [test_config.py](../tests/test_config.py) |
| A5、真实训练数值、allocator 峰值 | 当前无真实 NPU | 未验证；A5 入口拒绝 | [test_config.py](../tests/test_config.py)、migration-status.md |
| Kimi K3 | 原生模型、MoE、FSDP/TP/EP/CP、AC、MXFP8、显式虚拟融合 | Meta/CPU 验证范围见专项记录；无真实 KDA 硬件 provider | [Kimi 验收](kimi-k3-migration.md) |
| raw_model/首层 replica／其他优化器 | 不属于当前标准路径 | 未迁移；不得据旧版支持推断新版可用 | legacy-patch-inventory.md |
| CANN metadata 普通 dispatch | 原生 `torch.ops`；按能力探针修正 Meta 分配设备；注册可恢复 | Meta 已验证；准备阶段成本范围仍独立 | [test_metadata.py](../tests/test_metadata.py)、open-questions.md |

CANN metadata 的泛型 Python 包装签名另有 [兼容修正与专项验证](cann-metadata-signature-compat.md)；用户已反馈其 CANN 9.2.0 环境轻量 CLI 运行成功；该反馈不等于完整矩阵复验，原 21 项结果仍按各自源码哈希解释。

## DSV4 组件与原生 override

以下实现均通过原生模型配置和默认 `override.imports` 进入；逐配置完成状态以全规模矩阵为准。
表中 `override.*` 是 `torchtitan_npu.override.*` 的缩写。
普通 CANN/NPU 前后向计算由 dispatch 自动捕获，Meta autograd 缺口才安装可撤销桥接。

| 组件 | 选中实现／捕获依据 | 状态与边界 |
| --- | --- | --- |
| Sparse attention | `override.deepseek_v4.sparse_attn.asc`；`sparse_flash_mla`／`sparse_flash_mla_grad` | Meta 前后向已验证 |
| Lightning Indexer 与辅助 loss | `asc_li`；`lightning_indexer`／`sparse_lightning_indexer_kl_loss_grad` | Meta 已验证；辅助 loss 参与反向 |
| attention metadata | `asc_metadata`／`asc_li_metadata`，普通 dispatcher 消费原生对象 | Meta 分配修正、真实 consumer 依赖及注册恢复已验证；准备阶段成本独立 |
| Compressor、共享 KV、输出分组 BMM | 原生模型组件与 TorchAO-NPU 包装后的实际 MM/BMM | BF16/MXFP8 Meta 前后向、保存对象来源已验证 |
| mHC pre/post | `override.deepseek_v4.mhc.asc_hc_pre`／`asc_hc_post`；对应 CANN forward/backward | Meta 前后向已验证 |
| mHC head | 原生 head／chunked loss 路径 | Meta 已验证；前向块中交错的 head backward 保留真实 phase |
| RoPE | `override.common.rope.asc_complex`；`npu_rotary_mul`／backward | Meta 已验证；原生梯度探针失败时安装桥接 |
| RMSNorm | `override.common.rms_norm.asc`；`npu_rms_norm`／backward | Meta 已验证；buffer 设备与实际保存值保留 |
| Token dispatcher | `override.common.token_dispatcher.asc`；count/data A2A、permute/unpermute 及梯度算子 | Meta 已验证；均衡路由是假设，不是真实 token 分布 |
| Routed/shared experts、SwiGLU | 原生 grouped MM；MXFP8 recipe；厂商 SwiGLU 原生前后向 | Meta 已验证；已知 backward arity 问题用局部桥接修复 |
| MTP | 原生 MTP builder、attention/MoE 和辅助 loss | MTP=0/1 已验证；不声称所有 MTP 层数均已验收 |
| MXFP8 选择范围 | 原生 `all_mxfp8`；实际 quantized_parameters 清单 | 不恢复旧 FQN 复写；原生未选择的 e/h projection 保持其原生精度 |

## 内存插件与诊断路径

| 插件／能力 | 启用条件 | 验证与状态 |
| --- | --- | --- |
| `FSDPFullParamResidencyPlugin` | 标准估算路径 | full/shard、AG staging、wait/reshard/finalize 释放；模型级与独立字节先验已验证 |
| `ActivationCheckpointPlugin` | 标准估算路径，AC none/full/selective | 使用实际 checkpoint 边界和保存值；原始 forward/recompute/backward 区分已验证 |
| saved-retention 与 storage ownership | 标准估算路径 | pack/unpack/release、实际所有权与 generation；最后读取不冒充释放 |
| `ActivationOffloadPlugin` | `memory_offload_ac_saved_tensors` | 三种 AC、BF16/MXFP8 投影已验证；不执行真实传输 |
| 参数 shard dtype 投影 | `memory_parameter_storage_dtype` | 常驻 shard 字节与计算／梯度／状态 dtype 分开；不会改动训练图 |
| 缺梯度诊断 | 原生梯度清单 | 标准入口缺梯度即失败；人工梯度生命周期插件已删除 |
| FP8 AG transport 投影 | 已删除当前禁用的低层模型 | 公开 FP8 通信入口继续拒绝 |
| native SAC cache 观察 | 原生 checkpoint/cache | 第二套 synthetic SAC cache 已删除；保留实际保存、消费与释放观察 |

这些状态不会扩大为真实 NPU 数值、allocator/workspace、异步 offload 性能或多 microbatch 累积峰值的验证结论。

## dtype 与通信口径

BF16/MXFP8 指计算路径，不能据名称推断所有常驻或通信张量都采用同一 dtype。
本次配置沿用目标原生精度策略；参数布局、实际梯度 dtype、量化 code/scale 和通信记录分别导出。
参数 shard 存储 dtype 选项仅改变估算投影，不改 native master parameter、梯度、优化器状态或训练通信。
旧版本的任意 dtype／FQN 覆盖不自动迁移为新版本支持，未公开的 CLI 字段会报错。

捕获 worker、logical rank、PP stage 和逻辑 world 分开记录；默认单 worker 不能被解释为逻辑 world=1。
最终版本统一计划层坐标标注。早期验收资料中计划层旧字段的兼容说明见 [输出契约](s5-delivery.md#输出契约与旧命令迁移)。

## S5 历史全规模配置结果

全部用例保留完整原生层数与几何，B=1、S=512、MTP=1、world=16、logical rank=15，组合 HSDP+EP+TP+CP。
每个单元格均执行完整 F/B/clip/optimizer，并通过图与内存独立复核。GiB 为捕获 logical rank 的逻辑活跃 storage；RSS 为捕获进程，不含 JIT 子进程。

| Preset | 层数 / dim / 专家 | none | full | selective | 参数梯度数 | 三模式最大 RSS GiB |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek_v4_flash_baseline_bf16 | 43 / 4096 / 256 | 通过；32071 节点；543.577 GiB | 通过；41900 节点；543.577 GiB | 通过；41562 节点；543.577 GiB | 1316 | 1.470 |
| deepseek_v4_flash_baseline_mxfp8 | 43 / 4096 / 256 | 通过；34024 节点；543.577 GiB | 通过；44562 节点；543.577 GiB | 通过；44439 节点；543.577 GiB | 1316 | 1.415 |
| deepseek_v4_pro_20t_baseline_bf16 | 96 / 12288 / 2048 | 通过；69803 节点；37720.882 GiB | 通过；91922 节点；37720.882 GiB | 通过；91113 节点；37720.882 GiB | 2907 | 1.903 |
| deepseek_v4_pro_20t_baseline_mxfp8 | 96 / 12288 / 2048 | 通过；74112 节点；37720.882 GiB | 通过；97814 节点；37720.882 GiB | 通过；97485 节点；37720.882 GiB | 2907 | 2.175 |
| deepseek_v4_pro_baseline_bf16 | 61 / 7168 / 384 | 通过；45131 节点；2982.633 GiB | 通过；59228 节点；2982.633 GiB | 通过；58714 节点；2982.633 GiB | 1864 | 1.613 |
| deepseek_v4_pro_baseline_mxfp8 | 61 / 7168 / 384 | 通过；47885 节点；2982.633 GiB | 通过；62988 节点；2982.633 GiB | 通过；62779 节点；2982.633 GiB | 1864 | 1.782 |
| deepseek_v4_smoketest | 4 / 256 / 4 | 通过；4108 节点；0.340 GiB | 通过；4818 节点；0.275 GiB | 通过；4791 节点；0.280 GiB | 143 | 1.255 |

上述历史数据的精确字节与哈希见 [S5 结果](s5-results.json)。当前修改后的全规模复验与对照结果见 [收敛记录](dispatch-first-remediation.md)。
另有独立安装镜像的完整 20T MXFP8 selective，以及 world=2048/rank=2047 的 BF16/CSV 限制验收。
更长序列、所有并行度笛卡尔积、多 microbatch 完整累积峰值和真实 NPU 数值均不由这组结果证明。
