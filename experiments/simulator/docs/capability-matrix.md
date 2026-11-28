# 当前能力与验证入口

本表描述当前标准 `simulate()` 路径及对应回归入口，不将历史全规模验收、主包设备能力或低层结构单测视为所有组合均已支持。首次使用先读[整体方案](architecture.md)；新增模型按[模型接入工作流](model-integration.md)提供新的证据。

## 模型与训练配置

| 能力 | 当前范围和边界 | 实现／验证入口 |
| --- | --- | --- |
| DeepSeek-V4 | smoke、flash、pro、pro_20t 结构 preset；原生 attention/compressor/indexer loss/mHC/MoE/shared expert；MTP 0/1 有回归 | `config.py`、`presets.py`；[test_dsv4_capture.py](../tests/test_dsv4_capture.py) |
| Kimi K3 | 原生模型与 loss、MoE、共享组件；KDA/attention/SiTU 采用显式 Meta 虚拟融合；不提供真实 KDA 硬件 kernel | `compat/kimi.py`；[test_kimi_capture.py](../tests/test_kimi_capture.py)、[test_kimi_parallel.py](../tests/test_kimi_parallel.py) |
| Kimi K3.1 Pro | 复用 K3，24 个 MLA 层增加冻结 DSA indexer；长序列 Meta 捕获采用选取／稀疏注意力正反向融合契约，普通设备保持 ATen 参考路径，短序列复用 dense 边界；不含索引器蒸馏、量化或融合 NPU 性能验证 | [接入说明](kimi-k3-1.md)、[test_kimi_k3_1_capture.py](../tests/test_kimi_k3_1_capture.py) |
| Qwen3-32B | dense：上游 torchtitan qwen3 几何（64 层、GQA 64:8、QK-norm、SwiGLU、不 tie 头）；attention 走真实融合算子 `npu_fusion_attention_v3`/`_grad_v3`（TND varlen，Meta 上由 [compat/attention.py](../torchtitan_simulator/compat/attention.py) 桥接 autograd）；`qwen3_32b_smoketest`（4 层切片）与 `qwen3_32b`（完整几何）有捕获回归；并行（FSDP/HSDP/DDP/TP×AC、首末与中间 rank）、双卸载与 CPU 数值参考已验证，CP/EP 显式拒绝；量化与真机数值未覆盖 | `override/common/varlen_attention.py`、[test_qwen3_capture.py](../tests/test_qwen3_capture.py)、[test_qwen3_parallel.py](../tests/test_qwen3_parallel.py)、[test_qwen3_offload.py](../tests/test_qwen3_offload.py) |
| system3 | 参照 external_models/system3 参考实现移植：[ISA, GDN2 x 3] 递归块（loop 2x）、Flash KV 共享（kv/reuse/reindex）、参考口径 ISA indexer、LatentMoE（dispatcher+grouped GEMM 执行，owner 批准留档）、官方范式 Engram（host 表 + SparseAdam 稀疏更新，EP 下 all-to-all 派发 + backward 行交换 + EP 组 row-square 归约）、mHC Sinkhorn、多模态编码器端到端联合训练；ISA/GDN2/编码器 FA 走显式 Meta 虚拟融合（isa_indexer、isa_sparse_attention、gdn2_chunked 等），稀疏生命周期走 `system3_sim::sparse_adam_update` 虚拟算子 + 稠密梯度豁免/稀疏证据协议（`sparse_optimizer_validation`，命名随 checkpoint 分片后缀）；GDN2 key/value 状态维度独立（owner 批准偏差，参考存在同款潜在缺陷）、ISA 掩码先于 clamp（owner 批准偏差：无效键零权重、全 pad 行零输出，参考先填充后 clamp 会复活掩码键）；调试/完整/单元 preset（debugmodel_text/multimodal、full_text/multimodal、unit_text）；indexer 打分与参考逐位一致（CPU 探针）；成本覆盖不含编码器 FA 反向（opm 家族合约未接受原生 BSND stats 布局前为诚实 unknown） | `torchtitan_npu/models/system3/`、[test_system3_capture.py](../tests/test_system3_capture.py)、[test_system3_cost.py](../tests/test_system3_cost.py) |
| Qwen3.5 | 上游 TorchTitan 0.3.0 原生模型、MoE；GDN 与 FlexAttention 采用显式 Meta 虚拟融合；文本捕获（vision detach）、无 MTP；CP／量化未接入 | `compat/qwen3_5.py`；[test_qwen3_5_capture.py](../tests/test_qwen3_5_capture.py)、[接入记录](qwen3-5-integration.md) |
| Multi-Head-MoE（SLA2） | 主包 `torchtitan_npu/models/multi_head_moe`（外部 multi-head-moe 参考实现的修正移植：front-2/back-2 dense + 中间 per-head gate MoE + shared expert）；SLA2 稀疏/线性两个分支核心采用显式 Meta 虚拟融合（mask 梯度仅线性分支返回，稀疏分支布尔门控与 eager 一致路由 None），**默认关，`--simulation.enable-sla2-fusion` 显式开启**，块池化/gumbel 软 top-k mask/alpha 融合为原生 dispatch；debug preset 捕获（单卡、FSDP/HSDP、EP2 all-to-all、full/selective AC）；TP/CP/MTP 拒绝；不提供真实 SLA2 硬件 kernel；SLA2 成本建模（4 族 roofline 与 DV120 硬件规格解析）在 opm-dev 仓 operator-performance-models 交付，不在本仓 | 主包 `models/multi_head_moe`、`compat/multi_head_moe.py`；[test_multi_head_moe_capture.py](../tests/test_multi_head_moe_capture.py) |
| Multi-Head-MoE 头并行（HP） | 同上模型的可选执行变体：flavor `debugmodel-head-parallel`／preset `multi_head_moe_debugmodel_head_parallel`（`Config.head_parallel`）按 MAGI-2 官方静态头 a2a 切换 `HeadParallelMultiHeadMoE`——每层两次定长 `all_to_all_single`（头所有权转置，不随 top-k 增长）、专家切片与 EP 相同、router 参数 replicated + 运行时按 rank 切片；捕获断言 16 次静态 a2a、无 token-count 交换；约束：`moe_heads` 必须被 ep 整除、token 均分（EP2 真机冒烟 loss 与默认 EP 档轨迹一致） | `models/multi_head_moe/head_parallel.py`；[test_multi_head_moe_capture.py](../tests/test_multi_head_moe_capture.py) |
| DeepSeek-V4.1 | 主包原生模型、MoE、mHC 与 host-offload Engram 表（Meta 初始化桥，CPU 稀疏表豁免 dense 梯度契约；host 参数与稀疏优化器状态单独报告 `host_param_bytes`/`host_optimizer_state_bytes`，不计入设备预算）；attention 元数据经 `build_attention_masks` 适配到逐后端 map；文本 debugmodel 捕获（multimodal flavor 的合成输入未实现，不在捕获 preset 表），MTP／CP／量化未接入 | `compat/deepseek_v4_1.py`；[test_deepseek_v4_1_capture.py](../tests/test_deepseek_v4_1_capture.py) |
| Sys2Mot（VLA） | 主包原生模型；`build_attention_masks` 合成确定性多模态输入（3 视图图像 + state/action/t），联合 loss（flow MSE + 文本 CE + indexer aux）+ 离线蒸馏 seam（`distill_offline`：runner 提供确定性 teacher logits/hidden 输入，复算 KD/hidden 五项）；routed experts 走共享 dispatcher + grouped GEMM（bf16 归一化执行是登记的共享偏差）；**FSDP2+EP**（student 与 teacher 双树，ep=1 回落）；三档几何：debugmodel（捕获契约测试）/ mid 8 层（真机对账口径）/ **student30b / student300b 全量 40 层**（mid 对比校验通过后的全量端到端预测；8 卡 e2e 皆 93674 节点 / 1 miss / 0 fail，30b 5.37 s/步、300b 29.21 s/步，后者需要 sim-platform 大图数值容差修复）；fp32 参数；TP／CP／MTP 拒绝 | `torchtitan_npu.models.sys2mot`；[test_sys2mot_capture.py](../tests/test_sys2mot_capture.py) |
| 原生配置与 override | `prepare_config()` 创建新配置，默认或显式 `override_imports`；`module.factory` 不绕过模型白名单或运行接口要求 | [test_config.py](../tests/test_config.py)、[test_model_overrides.py](../tests/test_model_overrides.py) |
| 模型几何覆盖 | typed `model_overrides` 只适用于受支持的 DSV4 preset；自定义工厂负责自己的原生配置树 | `model_overrides.py`、[接入接口](upstream-integration.md) |
| 训练精度 | 当前完整 DSV4/Kimi/Qwen3.5 捕获要求原始参数 FP32；计算参数 BF16/FP32，梯度归约 FP32；build-only 接受不等于完整捕获可用 | [test_precision.py](../tests/test_precision.py)、[test_kimi_capture.py](../tests/test_kimi_capture.py) |
| MXFP8 | 显式 `enable_quantized_training` 和原生 `all_mxfp8` recipe；需要配套 TorchAO-NPU，不能从 preset 名称推断；保留实际 code/scale 与原生参数选择 | [test_s4_mxfp8.py](../tests/test_s4_mxfp8.py)、[test_kimi_quantization.py](../tests/test_kimi_quantization.py) |
| AC 与 selective save-set | none/full/selective；选择项以 `selective_ac.py` 为准；未命中实际 comm cache 时 warning，不伪造命中 | [test_s3_native_lifetimes.py](../tests/test_s3_native_lifetimes.py)、[test_s4_mxfp8.py](../tests/test_s4_mxfp8.py) |
| GBS／GA 与稳态捕获 | 原生配置准备按有效 DP 和本地微批校验 GBS、推导 GA。预热复用该 GA，再捕获单 microbatch F/B 与一次 O；scheduler 不参与预热；GA 时间展开由下游处理 | [批次与草稿预检](upstream-integration.md#按草稿查询一个字段)、[结构预热](steady-state-capture.md) |
| Muon 混合优化器 | DSV4，TP=CP=PP=1；原生 DistMuon＋AdamW、DP/HSDP/EP、矩阵计算和双向 all-to-all；EP=1 支持三维专家权重在单个分片轴上的 Shard(1)→Shard(0) 转换；不执行真实 peer 计划哈希比较；拒绝量化、replica dedup、native swap/virtual；支持整段聚合状态卸载 | [Muon 捕获](muon.md)、[捕获测试](../tests/test_muon_capture.py)、[预取依赖测试](../tests/test_muon_dependencies.py) |

## 并行、捕获与交付

| 能力 | 当前范围和边界 | 实现／验证入口 |
| --- | --- | --- |
| DP/FSDP/HSDP、EP、TP/SP、CP | 按原生 parallelize 与合法 mesh 执行；单进程捕获一个 logical rank；各模型组合以对应测试为准，不代表全部并行度笛卡尔积 | `compat/distributed.py`；[test_dsv4_capture.py](../tests/test_dsv4_capture.py)、[test_kimi_parallel.py](../tests/test_kimi_parallel.py) |
| 原生通信与 autograd | functional/c10d/DTensor transport、实际 group/ranks、归约到 clip/update 的依赖；fake process group 不证明真实 HCCL 性能 | [test_native_collectives.py](../tests/capture/test_native_collectives.py)、[test_fsdp_group_capture.py](../tests/test_fsdp_group_capture.py) |
| 通信语义事实 v2 | 四个精确 AG／RS／AR／A2A 原语的组、归约和缓冲事实。记录 RS 原生标量乘数以及已证明的 EP 全组分流与反向转置，缺字段时标记 unresolved | [通信事实契约](capture-runtime.md#通信语义事实-v2)、[test_collective_facts.py](../tests/capture/test_collective_facts.py) |
| EP single exchange | DSV4/Kimi 共用原生 functional single 适配；count/data/score 顺序和异常恢复；均衡路由是声明假设 | [EP 交换说明](ep-single-transport.md)、[test_routing.py](../tests/test_routing.py) |
| 流与事件等待 | 同流前沿、event record 快照、wait_event/wait_stream、零耗时控制节点；无等待的流保持独立 | [流控制依赖](stream-control-dependencies.md)、[test_fsdp_stream_capture.py](../tests/test_fsdp_stream_capture.py) |
| FSDP 通信视图 | 默认隐藏 FSDP scope 非通信准备工作，保留通信/模型计算/控制依赖；源图保留作审计，内存不跟随过滤 | [捕获契约](capture-runtime.md#fsdp-通信视图)、[test_fsdp_communication_view.py](../tests/capture/test_fsdp_communication_view.py) |
| 原生参数与自定义语义 | 原生 schema 参数绑定、标量及 AdamW 列表身份；6 个 Kimi、5 个 MoE 包装、4 个 Qwen3.5 和 4 个 Multi-Head-MoE 虚拟融合 overload 的角色/轴声明 | [参数测试](../tests/capture/test_operator_arguments.py)、[自定义语义](custom-tensor-semantics.md)、[真实交付测试](../tests/test_custom_tensor_semantics_delivery.py) |
| 会话隔离 | 显式激活，正常／异常退出恢复；同进程串行捕获 | [test_runtime.py](../tests/test_runtime.py)、[test_patch_scope.py](../tests/test_patch_scope.py)、[test_chunked_loss_policy.py](../tests/test_chunked_loss_policy.py) |
| Python／文件交付 | 默认返回图与 MemoryPlan，不写文件；仅显式 mem 导出摘要／明细；无 JSON/HTML/DOT 图导出 | [test_memory_delivery.py](../tests/test_memory_delivery.py)、[接入接口](upstream-integration.md) |

## 静态显存与卸载

| 能力 | 当前范围和边界 | 实现／验证入口 |
| --- | --- | --- |
| storage 生命周期 | tensor/storage/generation 分开，alias/mutation、真实保存值、原生所有权；候选通知不替代释放确认 | [test_saved_tensor_identity.py](../tests/capture/test_saved_tensor_identity.py)、[test_storage_lifetime_capture.py](../tests/test_storage_lifetime_capture.py) |
| FSDP full/shard 驻留 | 实际 AG staging、full allocation、wait/reshard/finalize 区间；不按 Python view 是否存活推断设备驻留 | [test_fsdp_residency.py](../tests/capture/test_fsdp_residency.py)、[内存测试](../tests/memory/) |
| 稳态状态预算 | 实际本地 optimizer state 去重；四个峰值、权重与状态独立大小；预热后初始状态与 Muon runtime 常驻工作缓冲均计入 | [稳态内存](steady-state-memory.md)、[test_optimizer_state.py](../tests/memory/test_optimizer_state.py) |
| 激活卸载 | 独立开关；逐层 payload、H2D/D2H/等待和派生静态驻留；head/层外值保留，晚 producer 保守处理；不执行真实 copy | [激活卸载](activation-offload-design.md)、[test_activation_transfers.py](../tests/capture/test_activation_transfers.py) |
| 优化器卸载 | AdamW 保留更新窗口；DSV4 Muon 覆盖整个混合 optimizer step，H2D/D2H 与原生 DAG 并行；按本卡真实状态量传输，忽略状态 staging buffer；Muon 常驻工作缓冲保留 | [优化器卸载](optimizer-offload.md)、[test_optimizer_transfers.py](../tests/capture/test_optimizer_transfers.py)、[Muon 卸载测试](../tests/test_muon_offload.py) |
| shard dtype 投影 | `memory_parameter_storage_dtype` 只改常驻权重字节投影，不修改计算、梯度、状态和通信 dtype | [test_memory_delivery.py](../tests/test_memory_delivery.py)、[test_precision.py](../tests/test_precision.py) |

## 未支持或不由本仓证明的范围

- 标准训练入口拒绝 PP、FP8 FSDP AG／EP dispatch、A5 目标；底层结构或主包存在相关实现不改变该边界。
- 不支持宿主已有 process group、嵌套或同进程并发仿真；后续多任务应隔离 worker 生命周期。
- 当前静态显存不是完整 GA／PP 回放，也不包含 allocator cache/reserved、碎片或未建模 workspace。
- 不提供硬件算子成本、通信带宽、DES 或端到端性能；新算子捕获成功不表示已有 costmodel。
- Meta 和 CPU 测试不证明真实 NPU 数值、loss 收敛、HCCL 时延或真实 allocator 峰值。
- 自定义 Triton/TileLang/DeepEP 路径、额外优化器和未列出的模型需独立接入，不能按旧版本经验声称可用。

测试文件是复验入口，不是本次执行日志。代码或环境变化后按[开发指南](developer-guide.md)重跑对应范围，保留实际版本、配置、命令与结果；完整几何执行成本由调用者评估，不复用历史时间常数作为验收阈值。
