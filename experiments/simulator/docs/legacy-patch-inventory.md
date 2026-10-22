# Patch 与 shim 逐项处置清单

状态：迁移前静态审计。38 是相关 helper 定义数，实际安装依赖 fake_backend / multi_proc_meta、PP 与模型模式，并非每次启动全部激活。保留表示“保留需求并重新验证”，不表示可以把旧函数原样复制到目标。

## meta_env.py 的 38 个安装函数

| 旧函数 | 结论 | 目标证据 / 必须验证的行为 |
|---|---|---|
| `_patch_tensor_npu_method_to_meta` | 保留候选 | 新 override/算子仍可能调用 Tensor.npu；重审设备转换，限制在 simulator 生命周期 |
| `_patch_torch_full_npu_device_literal` | 缩小/删除候选 | 原针对旧 SparseAttention 的硬编码设备；新 attention 已改写，按目标实际执行路径判断是否仍需要全局 torch.full wrapper |
| `_patch_grouped_mm_offsets_dtype` | 重定位 | 新 ops/ascendc/grouped_mm 承接 PrivateUse1；通用 Meta offset dtype 与 NPU int64 要分别验证 |
| `_neutralize_torch_npu_optimizer_device_probe` | 保留候选 | 新 torch_npu/utils/_optim.py 仍参与设备支持判定；确保 optimizer 第一次 step 不初始化硬件 |
| `_patch_swap_optimizer_get_device_info` | 旧目标退役 | patches.optimizer.swap_optimizer 不存在；新的 NovaSwap/CPU offload 走另一套配置和类 |
| `_patch_moe_dispatch_to_avoid_meta_tensor_value_reads` | 重写 | 新 AllToAllTokenDispatcher `_sync_token_count_exchange` 仍 `.cpu().tolist()`；用确定路由计算 split，保留实际 collective 的采集 |
| `_patch_parameter_dtensor_for_meta` | 需要动态复现 | nn.Parameter(DTensor) 的 Meta/local-storage 行为不可仅凭 Python 源码判定；测试 shape/spec/local device 与参数 identity |
| `_patch_li_loss_to_skip_buggy_einsum` | 旧目标退役、保留 fail-closed 原则 | 旧 LiLoss 消失；新 LightningIndexer/SparseAttention/AuxLoss 必须有完整反向和依赖，无法建模时拒绝而非零值替代 |
| `_neutralize_fsdp_meta_param_validation` | 保留 | torch 2.14 FSDPParamGroup._validate_no_meta_params 仍存在并拒绝 meta 参数 |
| `_patch_fsdp_chunk_with_empty_for_meta` | 保留性能需求并复核绑定 | `_chunk_with_empty` 移到 _fsdp_common，但 _fsdp_param 仍按值导入；只 patch 一个名字未必覆盖全部调用，需追所有 alias |
| `_patch_fsdp_mesh_caches_for_meta` | 重审 | _init_shard_mesh/_init_sharding_spec_tp 仍存在；full mesh、HSDP 和不同 placement 不得错误共用缓存 |
| `_patch_pipeline_schedule_warmup_for_meta` | 保留并验证 | _warmup_p2p 签名仍存在；STATIC/DYNAMIC metadata vote 不能读取 meta 值，禁止绕过必要阶段元数据 |
| `_patch_dtensor_meta_to_dtensor_for_meta` | 保留候选 | _DTensorMeta 在 pipelining/_utils.py，to_dtensor 签名相同；验证 TP/CP 的 local/global shape |
| `_patch_from_local_backward_for_meta` | 需要动态复现 | 不应无条件保留对私有 _FromTorchTensor.backward 的覆盖；普通 gradient 与 DTensor boundary 逐 placement 复现 |
| `_patch_rowwise_parallel_output_for_meta` | 需要动态复现 | 只有验证目标仍存在 local/global embedding shape 问题才保留；防止重复重分布 |
| `_patch_torch_split_for_meta_dtensor` | 需要动态复现 | 对 SplitWithSizesBackward 的旧兼容不是稳定 API；核对 slice/split 别名和梯度 placement |
| `_patch_window_exchange_for_fake_pg` | 重写 | 旧 compressor_attention_cp 路径不存在；新模型 metadata/CP frame 拥有通信，必须追实际调用点 |
| `_patch_redistribute_local_tensor_for_meta` | 保留候选 | torch 2.14 签名仍含 async_op/use_graph_based_transform/is_explicit；保持 placement planner 与 simulator transport 分工 |
| `_patch_object_collectives_for_fake_pg` | 保留候选 | PP metadata object 传输仍存在；fake/gloo capture rank 与 logical rank 必须分清 |
| `_patch_pipeline_stage_meta_exchange_for_fake_pg` | 必须改签名和语义 | 2.14 _prepare_backward_infra 含 loss_kwargs，旧 _sim_prepare_backward_infra 不接收；核对 _StageForwardMeta/_StageBackwardMeta |
| `_patch_torch_equal_for_meta` | 高风险保留候选 | Meta 缺 kernel 的 shape-only workaround；“恒真”仅可用于不影响模拟控制路径的受控判断，需验证目标使用者 |
| `_patch_pipeline_stage_for_pp_context` | 重审保留 | F/B/I/W stage hook 仍存在；新 schedule/metadata 额外 forward 不得算入模板，MB 代表选择一致 |
| `_patch_metadata_inference_skip` | 重审保留 | _compute_outputs 仍存在；排除 metadata forward 的同时不能跳过真实第一次 forward |
| `_patch_pipeline_action_context` | 重审保留 | _get_profiler_function_name 仍存在；需检查所有 action/composite action 编码，不能用 profiler 名称稳定性代替契约 |
| `_patch_pipeline_p2p_context` | 重审保留 | P2POp 创建和执行仍分离；保留 action/MB/stage 来源，不把混合 send/recv 按当前可变上下文归类 |
| `_patch_get_stage_indices_for_fake_pg` | 重审 | 旧单进程 all-stage 与新版 pipeline split/Module.Config、MTP 层结构重对齐；多进程路径不应重复托管全部 stage |
| `_patch_build_pipeline_schedule_for_fake_pg` | 重审 | 原 private builder 覆盖与新的 pre-split microbatch/loss_kwargs 组合验证 |
| `_patch_device_mesh_world_size_check` | 重审保留 | 用少量 capture workers 承载大逻辑 mesh；真实 gloo world 与逻辑 world 的校验范围不可混淆 |
| `_patch_parallel_dims_for_multi_proc` | 重审保留 | _global_meshes 仍生成，但新增 spmd_dense/sparse mesh；_unflatten/_flatten、mesh rank、坐标与 group cache 全部验证 |
| `_patch_new_group_for_fake_backend` | 重审保留 | 原生 fake PG 与二维 fake HSDP group 已通过最小探针；gloo world + 大逻辑 fake subgroup、group_src/group_dst/tag 仍须验证 |
| `_patch_fsdp_get_device_from_mesh` | 重审保留 | fake mesh 不能转成 torch.device('fake')；新 _get_mesh_info 支持 named dims，不能破坏它 |
| `_patch_dtensor_random_for_fake_mesh` | 保留候选 | fake mesh 的 RNG device resolution；不扩展成对真实训练 RNG 的改动 |
| `_patch_comm_layer_context` | 部分重写 | FSDP + PP 路径继续观测；旧 CompressorAttentionCP._allgather_seq 目标消失，重接新模型通信 |
| `_patch_mxfp8_for_meta` | 重写 | 原 patches.torchao_npu 消失；改对新独立 torchao_npu 参数 wrapper/算子注册建立模拟适配 |
| `_patch_init_distributed_for_multi_proc_meta` | 重写配置契约、保留需求 | 新 init_distributed 接口仍在；comm.mode、逻辑 world/capture world、init_device_mesh 路径核对 |
| `_patch_fused_adamw_for_meta` | 保留建模需求 | torch._fused_adamw_ 与多 OptimizersContainer 路径；逻辑参数全局体积与 local grads/states 存储分开 |
| `_patch_llama4_hsdp_ep_mesh_info` | 正常目标路径退役候选 | llama4.parallelize 不存在；目标 FSDP 调 _get_mesh_info；原生二维 fake HSDP 探针确认 shard=1/replicate=0，仍需模型级 DP/EDP+EP 验证 |
| `_patch_maybe_enable_amp_for_meta` | 退役旧接口 | maybe_enable_amp 已删除；新 get_spmd_context 需按实际 dtype/设备路径评估 |

## 其他 simulator 接入点

| 当前位置 | 结论 |
|---|---|
| hardware_shims/{mhc,smla,rms_norm}_converter.py | 全部从“替换旧 singleton converter.model_converter”改为配置级 override 或局部替换；MHC/attention/LI 的新类及参数由新模型持有 |
| hardware_shims/kda_converter.py、kimi_k3_fusion_shim.py | 依赖本地 Kimi K3；同模型移植，保留 TP/CP/FSDP module hook 与 parameter identity |
| hardware_shims/grouped_experts_shim.py | 保留“实际前向值→saved tensor→激活生命周期”的需求；对新 GroupedExperts/融合边界判断原生 autograd 是否足够 |
| capture/checkpoint_execution.py | CheckpointWrapper 仍是潜在观测点；验证新 AC early_stop/preserve_rng/context_fn；不能认为旧 policy 字段仍可用 |
| capture/fsdp_residency.py | FSDPParamGroup unshard/wait/reshard/post_backward、FSDPState hook 名称在 torch2.14 仍存在；叠加 torch_npu lazy wrapper 与 NPU policy wrapper 后验收 |
| capture/comm_events.py | c10d/funcol/fake P2P interception 保留；审计所有按值绑定、autograd collective、wait 与异步输出路径 |
| selective_ac.py | _get_save_ops 模块级目标消失；使用 SelectiveAC.get_save_ops；synthetic save 的策略与真实 save 集一起验收 |
| synthetic_ac.py | 通用思想保留；新的 native/synthetic op 不能双重缓存、重复计数 |
| rank_context.py / utils.py / run_simulator_spawn.py | 保留最少 worker 校验；配置字段、NGPU/WORLD_SIZE、PP capture worker 与逻辑 rank 映射需提前于模型初始化 |
| ir/、memory/、viz/、cost/ | 优先保留既有实现；对新 op namespace、dtype、memory annotation、alias 与 schedule 事件重新绑定，不假设输出逐行不变 |

## 继承自旧 NPU 基线的公共补丁

旧工作区 patches 目录有 33 个非 __init__.py 文件；目标仅 determinism.py 仍保持同一相对路径。这个统计是路径变化，**不是要求搬迁 32 个旧补丁**。以目标 master 的补丁和扩展作为新权威实现，只迁移 simulator 独有需求。

| 旧文件/组 | 处理原则 |
|---|---|
| torchao_npu/{mx_capability_check,mxfp8_grouped_mm,mx_linear}.py | 由独立 torchao_npu + 新 quant converter 接管，旧包不恢复 |
| tools/metrics.py | 新 patches/torchtitan/components/metrics 与 extensions/torchtitan/components/metrics；确认 simulator 的 device monitor stub 接口 |
| optimizer/{swap_optimizer,optimizer_selector,virtual_optimizer,muon_optimizer}.py | 新 OptimizersContainer、NovaSwap 与 override/common/optimizer；避免两套 optimizer 选择器叠加 |
| distributed/{utils,mtp_context_parallel,cp_shard_mask}.py | 新 model preprocessing、metadata、MTP 与 CP；从目标实际路径核验，不整文件复制 |
| torch/checkpoint.py | 新 AC 与 checkpoint manager 已重构；保留 simulator 边界观测需求，不带旧全局修改 |
| torch/micro_pipeline_tp.py | 仅在恢复现有 async TP 支持时评估新上游路径，不因 import 消失而恢复旧 patch |
| torch/pipelining.py | 本地 simulator 有额外改动，必须专门迁移；见主报告 |
| torch/functional_collectives_p2p.py | 新 c10d/funcol/PP 自身注册与 Meta kernels 逐项实测，保留缺口最小集 |
| torch/clip_grad.py | 新分布式 clip 路径检查；旧 simulator foreach=False 需求与新 SPMD 分别验证 |
| torch/_inductor/graph.py、torch_npu/_inductor/lowering.py | 当前 simulator 禁用 compile；不列为初期移植项，不把它们作为 eager 采集依赖恢复 |
| torch/fsdp/fsdp_dtype_fix.py | 官方 torch2.14 含对应修复；候选删除，明确与新 NPU per-parameter dtype policy 不同 |
| torch/testing/_internal/distributed/fake_pg.py | 本地修改过；新原生 fake PG/二维 fake HSDP 已通过最小复现，旧 Options 补丁是退役候选，混合 gloo/fake 组合仍要验收 |
| encoders/{dsv4,base}.py | 属于模型输入/权重工具；对新 model preprocessing 和新工具路径确认是否还被调用 |
| torch_npu/_meta_registrations.py | 新 torch_npu wheel 已有新的 meta 注册；逐个算子 schema/shape 验证，不覆盖整个旧文件 |
| torch_npu/determinism.py | 路径相同不代表行为相同；使用新 master 版本，simulator seed 仍须避免真实设备读取 |
| torchtitan/pp_loss_normalize.py | 新 loss 自带 global_valid_tokens，旧补丁退役候选；覆盖当前未提交 meta token 改动的需求验收 |
| torchtitan/qwen3_hf_export.py | 新 state_dict/checkpoint 模块路径；按新主线实现，非 simulator 主线计算依赖 |
| torchtitan/trainer_init_distributed.py | 旧 TrainerConfig stash 与初始化顺序已变；新 TrainerEx/config-manager 边界重接 |
| torchtitan/chat_dataset.py、hf_datasets.py | 新模型 dataloader/preprocess 及新 HF patches；明确模拟输入假设，无需恢复全部旧数据补丁 |
| torchtitan/loss.py | 新 loss 返回结构/分母由上游组件拥有；不要继续旧 monkey-patch |
| torchtitan/_trainer_config_stash.py | 首层复制实验的旧依赖；移到 simulator 显式配置传递，避免全局 stash |
| torchtitan/optimizer.py | 新 optimizer package + NPU extensions 接管；simulator 从 public param_groups 观测 |
| torchtitan/expert_parallel.py | 本地修改过；新 token_dispatcher/GroupedExperts 分离，重接而非复制 |

上表对应全量路径与按 basename 提示的候选位置见 evidence/legacy-patch-files.tsv。候选仅用于导航，不代表语义等价。

## S0–S2 实际处置（2026-10-21）

上表保留迁移前判断；下表记录实际实现，不把后续阶段或其他模型标为已支持。

| 旧 helper | 本阶段结果 |
|---|---|
| `_patch_tensor_npu_method_to_meta` | 未安装全局 Tensor.npu patch；本阶段原生 BF16 路径不需要。 |
| `_patch_torch_full_npu_device_literal` | 不覆盖全局 torch.full；仅局部 metadata Meta kernel 重定向 NPU factory。 |
| `_patch_grouped_mm_offsets_dtype` | 未安装旧覆盖；目标 native GroupedExperts 前反向、保存值和梯度通过。 |
| `_neutralize_torch_npu_optimizer_device_probe` | compat/optimizer.py 仅为 Meta 放行硬件检查，非 Meta 走原函数。 |
| `_patch_swap_optimizer_get_device_info` | 旧目标删除；新 offload optimizer 不属于 S0–S2。 |
| `_patch_moe_dispatch_to_avoid_meta_tensor_value_reads` | 重写 routing adapter；CPU split oracle + 实际 count/data A2A 及控制依赖。 |
| `_patch_parameter_dtensor_for_meta` | 不覆盖 nn.Parameter；原生 DTensor/SPMD/FSDP，验证 shape、placement 与 storage 来源。 |
| `_patch_li_loss_to_skip_buggy_einsum` | 旧目标退役；保留 native indexer/aux loss 反向，不用零值替代。 |
| `_neutralize_fsdp_meta_param_validation` | scoped Meta-only validation adapter，保留其余原生验证。 |
| `_patch_fsdp_chunk_with_empty_for_meta` | 未安装；当前模型通过。大规模主机性能仍需 S5 验收。 |
| `_patch_fsdp_mesh_caches_for_meta` | 不搬旧缓存覆盖；隔离 native DTensor/SPMD/RoPE 跨 capture 状态，保留 mesh 校验。 |
| `_patch_pipeline_schedule_warmup_for_meta` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_dtensor_meta_to_dtensor_for_meta` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_from_local_backward_for_meta` | 不覆盖原生 backward；目标 DTensor/SPMD 梯度路径通过。 |
| `_patch_rowwise_parallel_output_for_meta` | 未安装；原生 TP/SP 与 embedding/lm_head 路径通过。 |
| `_patch_torch_split_for_meta_dtensor` | 不覆盖 torch.split；原生 view/split 保留身份与 mutation 来源。 |
| `_patch_window_exchange_for_fake_pg` | 旧类退役；原生 CP planner/compressor/collective + host metadata 设备提升。 |
| `_patch_redistribute_local_tensor_for_meta` | 不覆盖旧私有实现；native redistribution 负责 placement，fake collective 负责运输。 |
| `_patch_object_collectives_for_fake_pg` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_pipeline_stage_meta_exchange_for_fake_pg` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_torch_equal_for_meta` | 不安装恒真 workaround；本阶段原生路径通过。 |
| `_patch_pipeline_stage_for_pp_context` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_metadata_inference_skip` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_pipeline_action_context` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_pipeline_p2p_context` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_get_stage_indices_for_fake_pg` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_build_pipeline_schedule_for_fake_pg` | PP/多 capture worker 后置；S0–S2 不安装，PP CLI 明确拒绝。 |
| `_patch_device_mesh_world_size_check` | 不安装旧 world-size 绕过；native fake PG 承载完整 logical world，单独记录 capture worker。 |
| `_patch_parallel_dims_for_multi_proc` | 不搬旧多进程覆盖；原生 ParallelDims + Meta 设备选择 seam，多 worker 后置。 |
| `_patch_new_group_for_fake_backend` | 未安装旧 Options workaround；native FakePG/HashStore/new_group 通过模型级 DDP/HSDP/EP 验收。 |
| `_patch_fsdp_get_device_from_mesh` | scoped Meta device adapter，刷新实际按值绑定，不改 mesh_info/placement。 |
| `_patch_dtensor_random_for_fake_mesh` | 未安装；当前原生 Meta 初始化不需要旧 fake RNG 覆盖。 |
| `_patch_comm_layer_context` | scoped FSDP observer + native collective 记录；CP 不再依赖旧 compressor 类。 |
| `_patch_mxfp8_for_meta` | S4 待迁移；BF16 验收不证明量化 wrapper 或 scale 可用。 |
| `_patch_init_distributed_for_multi_proc_meta` | 不覆盖生产 init；runner 自己拥有/销毁 fake world，多 worker 后置。 |
| `_patch_fused_adamw_for_meta` | native fused 配置 + Meta shim，分开逻辑全局工作量与 local grad/state 内存，保留全部依赖。 |
| `_patch_llama4_hsdp_ep_mesh_info` | 旧目标退役；不安装轴修正，原生 dense/expert mesh 经 HSDP+EP 与非零 rank oracle 验证。 |
| `_patch_maybe_enable_amp_for_meta` | 旧接口退役；Meta float32 projection policy 保留 native router cast/autograd 路径。 |

证据包括 native DSV4 模型矩阵、组件梯度/保存身份、collective group、FSDP 与 memory 测试。
完整日志与验证边界见 [migration-status.md](migration-status.md)。

## S3 实际处置（2026-10-21）

- `selective_ac.py` 的旧模块级 `_get_save_ops` 覆盖退役；改用原生 SelectiveAC 的 Configurable/get_save_ops 扩展点。
- `capture/checkpoint_execution.py` 保留原生 wrapper/context_fn，增加 native SAC 真实缓存观测；canonical functional A2A/RS 经过原生 Meta dispatcher。
  默认 eager SPMD 的通信重算遵循目标原生实现，特殊 synthetic A2A cache 仍后置，不能把策略声明当成实际保存。
- `capture/fsdp_residency.py` 区分 full parameter 与 staging，观测 copy-out、shared context 及 finalize 清理；
  分配成员 storage 与追溯别名分开，旧 shape-only 释放推断只用于历史缺少 provenance 的输入。
- `memory/` 不再仅凭最后一个消费者或 unpack 推断释放；结合 actual saved-slot release 与弱 storage ownership，
  初始化 buffer 与 CPU/设备内存单独计量。参数 dtype/offload 仍为显式投影假设，不改变训练图。
- 以上均留在独立 sidecar，未搬回旧公共 HSDP/dtype/FakePG workaround，未编辑上游安装源码。

## S5 交付补充

DSV4 的 BF16/MXFP8、none/full/selective、MTP=1 已完成七个原始 preset 的完整几何执行，
逐项结果见 [能力矩阵](capability-matrix.md) 与 [S5 结果](s5-results.json)。以上迁移前及 S0–S2 表格保留历史判断，
不能将其中「后续阶段」误读为当前仍缺少 MXFP8 或 AC。

S5 未恢复旧 HSDP 轴修正、全局 NPU 设备重定向、旧 TorchAO-NPU patch 包或 FQN CLI 复写。
20T 的旧压缩比 0 改用原生不压缩标记 1；图标识、跨模板出边和导出坐标的修正仅在 sidecar 内进行。
PP、FP8 通信、其他模型及特殊 fake A2A 缓存仍按能力矩阵明确后置；CANN metadata 直调 Meta kernel 的专项问题单列保留。

## 原生 dispatch 复核后的当前处置

前文保留迁移前判断与阶段证据。2026-10-22 的复核将通信 API 替换、自定义 collective autograd、
AdamW 手工节点、synthetic SAC、缺梯度人工生命周期和禁用的 FP8 AG transport 投影删除。
原生 c10d/funcol/DTensor 调用由 dispatcher 记录，FSDP 的转换、累加和 packing 不再整段屏蔽。
CANN metadata 顶层恢复普通 dispatcher，仅按能力探针修正厂商 Meta 分配设备。
必要的原生 autograd 桥接、均衡 split、实际保存值与驻留观察保留，详见 [收敛与复验](dispatch-first-remediation.md)。
