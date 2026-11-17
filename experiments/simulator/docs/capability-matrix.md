# 当前能力与验证入口

本表描述当前标准 `simulate()` 路径及对应回归入口，不将历史全规模验收、主包设备能力或低层结构单测视为所有组合均已支持。首次使用先读[整体方案](architecture.md)；新增模型按[模型接入工作流](model-integration.md)提供新的证据。

## 模型与训练配置

| 能力 | 当前范围和边界 | 实现／验证入口 |
| --- | --- | --- |
| DeepSeek-V4 | smoke、flash、pro、pro_20t 结构 preset；原生 attention/compressor/indexer loss/mHC/MoE/shared expert；MTP 0/1 有回归 | `config.py`、`presets.py`；[test_dsv4_capture.py](../tests/test_dsv4_capture.py) |
| Kimi K3 | 原生模型与 loss、MoE、共享组件；KDA/attention/SiTU 采用显式 Meta 虚拟融合；不提供真实 KDA 硬件 kernel | `compat/kimi.py`；[test_kimi_capture.py](../tests/test_kimi_capture.py)、[test_kimi_parallel.py](../tests/test_kimi_parallel.py) |
| Qwen3.5 | 上游 TorchTitan 0.3.0 原生模型、MoE；GDN 与 FlexAttention 采用显式 Meta 虚拟融合；文本捕获（vision detach）、无 MTP；CP／量化未接入 | `compat/qwen3_5.py`；[test_qwen3_5_capture.py](../tests/test_qwen3_5_capture.py)、[接入记录](qwen3-5-integration.md) |
| Multi-Head-MoE（SLA2） | 主包 `torchtitan_npu/models/multi_head_moe`（外部 multi-head-moe 参考实现的修正移植：front-2/back-2 dense + 中间 per-head gate MoE + shared expert）；SLA2 稀疏/线性两个分支核心采用显式 Meta 虚拟融合（含 mask 梯度），块池化/gumbel 软 top-k mask/alpha 融合为原生 dispatch；debug preset 捕获（单卡、FSDP/HSDP、EP2 all-to-all、full/selective AC）；TP/CP/MTP 拒绝；不提供真实 SLA2 硬件 kernel；SLA2 成本建模在 opm-dev 仓 operator-performance-models（4 族 roofline，需硬件与显式 vec 配置），硬件用 `cost/hardware.py::dv120_hardware` 解析的 DV120 真实规格 | 主包 `models/multi_head_moe`、`compat/multi_head_moe.py`；[test_multi_head_moe_capture.py](../tests/test_multi_head_moe_capture.py) |
| DeepSeek-V4.1 | 主包原生模型、MoE、mHC 与 host-offload Engram 表（Meta 初始化桥，CPU 稀疏表豁免 dense 梯度契约）；attention 元数据经 `build_attention_masks` 适配到逐后端 map；文本 debugmodel 捕获，MTP／CP／量化未接入 | `compat/deepseek_v4_1.py`；[test_deepseek_v4_1_capture.py](../tests/test_deepseek_v4_1_capture.py) |
| 原生配置与 override | `prepare_config()` 创建新配置，默认或显式 `override_imports`；`module.factory` 不绕过模型白名单或运行接口要求 | [test_config.py](../tests/test_config.py)、[test_model_overrides.py](../tests/test_model_overrides.py) |
| 模型几何覆盖 | typed `model_overrides` 只适用于受支持的 DSV4 preset；自定义工厂负责自己的原生配置树 | `model_overrides.py`、[接入接口](upstream-integration.md) |
| 训练精度 | 当前完整 DSV4/Kimi/Qwen3.5 捕获要求原始参数 FP32；计算参数 BF16/FP32，梯度归约 FP32；build-only 接受不等于完整捕获可用 | [test_precision.py](../tests/test_precision.py)、[test_kimi_capture.py](../tests/test_kimi_capture.py) |
| MXFP8 | 显式 `enable_quantized_training` 和原生 `all_mxfp8` recipe；需要配套 TorchAO-NPU，不能从 preset 名称推断；保留实际 code/scale 与原生参数选择 | [test_s4_mxfp8.py](../tests/test_s4_mxfp8.py)、[test_kimi_quantization.py](../tests/test_kimi_quantization.py) |
| AC 与 selective save-set | none/full/selective；选择项以 `selective_ac.py` 为准；未命中实际 comm cache 时 warning，不伪造命中 | [test_s3_native_lifetimes.py](../tests/test_s3_native_lifetimes.py)、[test_s4_mxfp8.py](../tests/test_s4_mxfp8.py) |
| GBS／GA 与稳态捕获 | 默认先按 GA 预热一轮，再捕获单 microbatch F/B 与一次 O；scheduler 不参与预热；GA 时间展开由下游处理 | [test_warmup.py](../tests/test_warmup.py)、[结构预热](steady-state-capture.md) |

## 并行、捕获与交付

| 能力 | 当前范围和边界 | 实现／验证入口 |
| --- | --- | --- |
| DP/FSDP/HSDP、EP、TP/SP、CP | 按原生 parallelize 与合法 mesh 执行；单进程捕获一个 logical rank；各模型组合以对应测试为准，不代表全部并行度笛卡尔积 | `compat/distributed.py`；[test_dsv4_capture.py](../tests/test_dsv4_capture.py)、[test_kimi_parallel.py](../tests/test_kimi_parallel.py) |
| 原生通信与 autograd | functional/c10d/DTensor transport、实际 group/ranks、归约到 clip/update 的依赖；fake process group 不证明真实 HCCL 性能 | [test_native_collectives.py](../tests/capture/test_native_collectives.py)、[test_fsdp_group_capture.py](../tests/test_fsdp_group_capture.py) |
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
| 稳态状态预算 | 实际本地 optimizer state 去重；四个峰值、权重与状态独立大小；预热后初始状态仍计入 | [稳态内存](steady-state-memory.md)、[test_optimizer_state.py](../tests/memory/test_optimizer_state.py) |
| 激活卸载 | 独立开关；逐层 payload、H2D/D2H/等待和派生静态驻留；head/层外值保留，晚 producer 保守处理；不执行真实 copy | [激活卸载](activation-offload-design.md)、[test_activation_transfers.py](../tests/capture/test_activation_transfers.py) |
| 优化器卸载 | 独立开关；按本卡真实状态量生成双向窗口，不改 AdamW 耗时；峰值按约定忽略逐 tensor 搬运 buffer | [优化器卸载](optimizer-offload.md)、[test_optimizer_transfers.py](../tests/capture/test_optimizer_transfers.py) |
| shard dtype 投影 | `memory_parameter_storage_dtype` 只改常驻权重字节投影，不修改计算、梯度、状态和通信 dtype | [test_memory_delivery.py](../tests/test_memory_delivery.py)、[test_precision.py](../tests/test_precision.py) |

## 未支持或不由本仓证明的范围

- 标准训练入口拒绝 PP、FP8 FSDP AG／EP dispatch、A5 目标；底层结构或主包存在相关实现不改变该边界。
- 不支持宿主已有 process group、嵌套或同进程并发仿真；后续多任务应隔离 worker 生命周期。
- 当前静态显存不是完整 GA／PP 回放，也不包含 allocator cache/reserved、碎片或未建模 workspace。
- 不提供硬件算子成本、通信带宽、DES 或端到端性能；新算子捕获成功不表示已有 costmodel。
- Meta 和 CPU 测试不证明真实 NPU 数值、loss 收敛、HCCL 时延或真实 allocator 峰值。
- 自定义 Triton/TileLang/DeepEP 路径、额外优化器和未列出的模型需独立接入，不能按旧版本经验声称可用。

测试文件是复验入口，不是本次执行日志。代码或环境变化后按[开发指南](developer-guide.md)重跑对应范围，保留实际版本、配置、命令与结果；完整几何执行成本由调用者评估，不复用历史时间常数作为验收阈值。
