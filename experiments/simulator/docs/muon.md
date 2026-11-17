# DeepSeek-V4 Muon 捕获

Simulator 可显式选择 DeepSeek-V4 的原生 DistMuon＋AdamW 混合优化器，保留矩阵计算、双向 all-to-all、打包与搬移、状态更新以及 stream/event 依赖。默认仍使用 recipe 的 AdamW。模型参数匹配和 FlexShard compute layout 来自主包的 DSV4 Muon profile，sim 不重新定义算法或参数分组。

## 使用与范围

按 [README](../README.md) 准备并安装训练软件栈、加载 CANN 环境后运行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torchtitan_simulator \
  --config deepseek_v4_smoketest \
  --optimizer.name Muon \
  --simulation.world-size 8 \
  --parallelism.expert-parallel-degree 2
```

Python 入口为 `simulate(SimulationConfig(optimizer_name="Muon", world_size=8, expert_parallel_degree=2))`。
`--optimizer.muon-ns-steps`／`muon_ns_steps` 可覆盖 Newton–Schulz 迭代次数，必须为正整数；不指定时保留原生 recipe 值，当前 DSV4 为 10。其余超参继续来自原生配置工厂。显式 AdamW 不重建原生分组；自定义工厂若已 materialize Muon，不能只改名称伪装成 AdamW。

当前支持 DSV4、TP=CP=PP=1 下的单卡、DP shard、HSDP、EP 及合法组合。小模型回归包含非零 logical rank、MTP、full AC 和激活卸载。Kimi、Muon 量化训练、HSDP replica dedup、原生 swap/virtual optimizer override 均明确拒绝。优化器状态卸载可用独立开关启用，采用下述整段聚合近似。

Muon 在 EP=1 时若需要将实际参数存储 `Shard(1)` 转为 compute layout 的 `Shard(0)`，在模型并行化之后、构建优化器之前抛出 `ValueError`。报错包含参数 FQN、全局 shape、mesh 轴与大小，并建议使用 AdamW 或受支持的 EP/FSDP 配置；不实现该布局转换。检查依据实际 DTensor placement、原生参数分组和 compute layout，不按卡数一概拒绝。debug 模型有 4 个专家，EP=1、FSDP=2/4 仍通过，FSDP=8/16 提前拒绝；将专家数改为 8 后，EP=1、FSDP=8 的 `Shard(0)` 路径也通过。AdamW 和 EP>1 不受该规则影响。

设置 `memory_offload_optimizer_states=True` 或添加 `--simulation.memory-offload-optimizer-states` 可启用整段 H2D／D2H 卸载近似。共同入口覆盖整个混合 optimizer step，包括 momentum 准备、原生 all-to-all、矩阵计算和 AdamW fallback；不包含梯度裁剪与 scheduler。两方向传输与原生工作 DAG 并行，出口共同等待，内部预取与缓冲复用依赖保持。该窗口不能只包围最终参数 `add_`。

这是理想细粒度状态流水的聚合表达，不是精确逐桶依赖。在上下行资源独立、无其他争用时，窗口耗时为 `max(T_H2D, T_native_DAG, T_D2H)`；实际资源约束仍由平台处理。状态 staging buffer 暂按约定忽略，Muon 原生常驻工作缓冲和临时计算量照常保留。完整依赖与显存口径见[优化器卸载](optimizer-offload.md)。

## 原生执行与 Meta 适配

`compat/muon.py` 只在含 DistMuon 的仿真会话中安装可恢复适配：

- 允许全部本地参数为 Meta 的原生 DTensor；其他设备仍调用原校验。
- 保留原生 layout、bucket 构建和梯度／状态布局检查。跨 rank 的计划哈希比较需要真实 peers 和张量数值，单 logical-rank Meta 会话不执行该比较；`runtime_capabilities.muon_peer_plan_validation` 明确标记 `not_performed_single_logical_rank`。这不证明多进程计划一致性。
- 观察原生 `_apply_update`，将实际写入参数 storage 的最终 `add_(direction)` 关联到参数 FQN，不替换更新计算。

同一模型与优化器先完成结构预热，再捕获代表性 F/B/O。Muon momentum 和 AdamW moments 均来自真实 optimizer state，按实际 dtype、设备和本地 storage 容量去重计入静态预算，不按固定参数倍数或全局参数量除以 world 推断。Muon momentum 使用原生梯度 dtype。

预热后，runtime 的 local slot 与两个通信 slot 已经持有工作缓冲。正式捕获将其 `storage_exchange`、`compute_exchange`、`compute_scratch` 和 `storage_scratch` 登记为初始驻留；它们跨 F/B/O 及迭代复用，不能到 O 首次使用才开始计量。按实际 allocation 去重，而非累计各 bucket 的逻辑容量。这些工作缓冲不属于 `optimizer_state_bytes`，状态卸载后仍计入设备峰值；Newton–Schulz 临时张量按原生生命周期处理。

GMM 对应的 routed-expert `w1_EFD / w2_EDF / w3_EFD` 归 DistMuon；更新时批量矩阵 Newton–Schulz 展开为 `bmm/baddbmm`，前后向仍保留原生 GMM。

交付中的 `optimizer_parameter_groups` 记录实际容器的 algorithm 与参数 FQN。更新节点通过 `optimizer_updated_parameter_names` 和 `optimizer_algorithm` 标明真实更新对象；校验要求全部梯度参数恰好更新一次且算法与 owner 相符。梯度裁剪由 `gradient_clip` scope 识别，不把 Muon 的权重衰减 `mul_` 当成裁剪。AdamW 的六组原生输入绑定与原有更新注记保持。

## All-to-all 与计算重叠

FlexShard 对需要重分布的 bucket 执行：

1. transfer stream 准备输入，执行 storage→compute all-to-all，记录 `compute_input_ready`。
2. caller stream 等待该 event，执行需要重分布的矩阵计算，记录 `compute_done`。同 bucket 无需重分布的本地计算可在等待之前执行。
3. transfer stream 等待 `compute_done`，执行 compute→storage all-to-all 和参数更新，记录 `done`。
4. caller stream 在原生 release 点等待 `done`；复用 buffer slot 必须等待前一轮工作完成。

Python 提交顺序不等于跨流执行依赖。下一 bucket 的 inbound all-to-all 与上一 bucket 的矩阵计算，以及上一 bucket 的 outbound all-to-all 与当前 bucket 的矩阵计算，可无相互依赖。捕获按各流前沿和 event record 时的快照保留这些关系，不按全局记录顺序串行化。无需为 Muon 新建调度器或手工补通信节点。

每次 logical-world 会话将显式 Meta 的 `torch.get_device_module` 查询绑定到本轮设备模块，其他设备沿用原查询，退出后恢复。PyTorch 缓存旧 Meta handle 会导致后续会话的原生流上下文与捕获器看到的上下文不同，错误地把通信归入计算流。该适配避免跨会话缓存造成的虚假串行；依赖测试先执行一个真实 Muon 会话，再检查下一会话的重叠关系。

`tests/test_muon_dependencies.py` 对最终交付图验证必要等待、双向不可达的独立工作及 buffer 复用保护。DSV4 debug、world=8、EP=2、rank=0 的测试覆盖 9 个重分布 bucket 和 18 次原生 all-to-all；这些数量是该配置的预期，不是全模型常数。通信保留有序 ranks、split sizes、shape/dtype、流及正式前驱。

## 验证入口与边界

从 `experiments/simulator` 运行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/test_muon_config.py tests/test_muon_capture.py tests/test_muon_dependencies.py \
  tests/test_muon_offload.py tests/memory/test_optimizer_state.py \
  tests/test_optimizer_contract.py tests/test_dsv4_capture.py tests/test_kimi_capture.py
```

配置测试覆盖显式选择、原生默认值和拒绝组合；捕获测试覆盖混合参数所有权、完整更新、状态字节、异常恢复和并行组合；原有 DSV4/Kimi 与 AdamW 测试保护已有路径。[聚合卸载测试](../tests/test_muon_offload.py) 以单卡和 world=8、EP=2 检查独立 workspace 容量预期、F/B/O 常驻、状态传输量与原生前驱保持。[内存字节测试](../tests/memory/test_optimizer_state.py) 检查 alias 去重和 workspace 卸载后保留。

依赖测试在卸载开关两种状态下证明图允许预取重叠，并不提供真实重叠时长。平台三路汇合与 Trace 诊断的 synthetic 验证边界见[优化器卸载](optimizer-offload.md#验证)。真实设备计算耗时、HCCL 传输及资源竞争由平台和算子后端处理；Meta 捕获不证明 NPU 数值、性能精度或真实跨 rank 计划一致性。

## 原生规划效率与宿主计时

主包扩展在兼容 backport 之后启用等价的 FlexShard 规划优化，正常训练和 simulator 共用同一实现，不额外改变 recipe 或 optimizer 开关。实现位于 `torchtitan_npu/extensions/torchtitan/distributed/flex_shard/`，与上游兼容补丁分开维护。

- 区域先执行完整边界校验；能沿任一轴证明互不相交时使用排序后的区间证明，否则回退通用两两校验。空区域、覆盖关系、重复参与者和非法布局仍须校验。
- 按源／目标 rank 建立 routes 索引，避免每个 rank 重扫所有 routes。
- 单次计划校验内复用相同不可变输入已经通过的区域／分区／覆盖检查；全部校验不变量保留，失败不缓存。成功事实表最多 1,024 条，另有 16,384 个几何元素的保留预算，按 FIFO 淘汰。计划入口先确认全部几何为原生不可变类型，避免每次命中重复扫描资格；这一步不证明覆盖、边界等校验不变量。资格记录只保留当前计划内的强引用，另设 4,096 条／65,536 个几何元素的预算，超限重新检查。大输入、非标准或可变输入仍执行完整检查；正常和异常退出均清空，不跨参数或会话保留计划。
- 单次几何生成内提取 shape、placements 和 mesh 尺寸，逐 rank 沿用原生分片公式。通信子组、正交 EP 分片和 domain offset 的原生检查与转换保持。
- All2All 打包按 peer 建立索引，并按原参与者、参数、route 顺序输出 splits 和 spans；保留零 split、自复制、replicated source 选择和全组 `has_remote_transfers`。
- 矩阵分块 route 按有序行区间二分定位可能相交的 compute 分片，再裁剪局部矩阵索引范围，避免逐个扫描空分片和不相交矩阵。保留原 source／participant／matrix 顺序、offset 公式、正反向 route、全部空分区、零 split 和完整校验。退化或越界几何回退原生构造；索引仅在单次参数规划内存在，不跨参数缓存计划。
- mesh 的 rank→坐标映射仅在一次 `build_bucket_plans` 调用内复用；结束或异常时清理，不跨模拟会话缓存 tensor、slot、图事件或 Owned 分配结果。

优化不改变分桶、compute owner、矩阵计算、通信组与有序 ranks、All2All／事件依赖、打包 offset 或 workspace 容量。不通过删除校验、跳过空参与者的 collective 或改变 Newton–Schulz 次数缩短捕获时间。

`simulate()` 成功返回的 `iteration.schedule.annotations["host_timings"]` 记录 `clock`、`scope`、`status` 和 `stages`；每阶段包含 `wall_seconds` 与 `calls`。配置、model build、parallelize、optimizer build、warmup、正式 F/B 捕获、内存规划、FSDP 投影及卸载插入分别记录。`optimizer_planning.*` 包含 build、geometry、routes、validation、packing；这些是**含子调用的墙钟时间，不可直接求和**。`capture_step.*` 兼容保留原 `capture_timings` 的细项。

日志以 INFO 输出单次宿主计时摘要；失败时记录 `status="failed"` 及已执行阶段，不返回伪成功负载。计时用 `time.perf_counter()`，不做设备同步，不作为算子或通信的设备执行耗时，不改变静态显存预算；旧结果缺少此注记时不补造。

规划等价性与 CPU Gloo 双向传输测试位于仓库根目录的 `tests/unit_tests/extensions/torchtitan/distributed/flex_shard/test_planning.py`。矩阵分块测试覆盖原生差分、pro 实际几何、异常／空分片、复制持有者与乱序 ranks；双卡和四卡 Gloo 用独立张量切片预期验证搬入、计算及搬回，包括跨矩阵边界和空 compute 分片。上述检查不替代真实 NPU 数值及性能验证。

Meta FSDP 初始化优化独立位于 `compat/meta_initialization.py`，只在 `parallelize` 上下文启用。两个已审计初始化方法使用私有函数命名空间中的按需空分片 helper，保留原生方法主体及装饰器；通用 `_chunk_with_empty` 的 list 契约保持不变。只创建真正读取的补齐空 Tensor，同一索引保持身份，不同索引不共享空 Tensor；真实 CPU／设备张量仍走原生 helper。上下文退出或异常时恢复方法，预热和正式捕获前已恢复，不改变 F/B/O 计算、通信或内存登记。

`tests/test_meta_initialization.py` 覆盖惰性容量、真实 helper 回退、嵌套／异常恢复和连续 Muon／AdamW 会话。主机内存验收应分别记录生成过程峰值 RSS、持有交付图时的 RSS、释放图并回收后的 RSS，使用相同 pro 配置与旧版普通 wheel 对照。校验预算限制的是临时 Python 元数据，不是设备显存；静态显存、单进程 RSS 与设备动态峰值应分别报告。

### 计划校验、几何生成与 Meta 初始化的根因及实现

route 区间裁剪后，剩余宿主开销不只来自 route 数量。单次计划中相同区域会反复接受分区、覆盖和 tensor shape 检查；同一参数的几何生成会为每个 participant 重读相同的 shape、placements 和 mesh 元数据；Meta FSDP 初始化还会补齐当前 rank 根本不读取的空分片对象。pro 4096 基线的完整校验约 40.71 秒、几何生成约 20.54 秒、并行化约 19.93 秒。前两项包含于优化器规划，不能作为独立阶段再次相加。

**计划校验：复用已经成立的谓词，而非省略不变量。** [planning.py](../../../torchtitan_npu/extensions/torchtitan/distributed/flex_shard/planning.py) 的 `_validate_plan` 仍检查参与者唯一性、分区顺序、区域边界及互不相交、区域体积与本地 tensor shape 一致、route 端点非空及唯一、参与者归属、端点 numel 相等，以及逻辑／本地分区的完整覆盖。成功证明的 key 包含检查种类、实际 regions、expected regions、bounds 和 tensor shape；只缓存成功，不缓存异常。direction 只影响失败诊断，因此不进入成功证明 key，失败时仍在当前调用点产生原有诊断。

资格判断与几何正确性是两件事。`_native_plan_geometry` 先检查原生 plan、partition、route、endpoint 的精确类型，以及所有后续会读取的区域和 shape 是否由原生不可变 tuple/int 构成；只有这种计划才能在查询成功证明前省略重复资格扫描。资格预检不会证明边界或覆盖正确，未命中证明仍执行完整检查。可变字段、类型子类等非标准输入退回逐次资格判断和原检查，不因相等比较或自定义 hash 复用不适用的证明。

这里没有通过扩容换时间。初版在每次命中前扫描资格，反而增加 Python 开销；2048 rank 隔离微基准中，盲目扩大成功证明表也没有收益。最终将原生计划的资格检查前移，并保留两组硬预算：成功证明 1,024 条／16,384 权重，资格记录 4,096 条／65,536 权重。权重按几何 tuple 和元素数量计数，不是字节数；单条证明超过 64 个区域不缓存。资格身份记录持有原对象强引用，命中还须满足 `is`，避免对象 ID 重用；预算满后重新检查，不无限增长。每个 plan 的 `finally` 同时清空证明、资格引用及 trusted 标志，再恢复外层 ContextVar。它不是跨参数计划缓存。

**几何生成：将参数内不变读取移出逐 rank 循环。** `_storage_region_reader` 在单次参数几何生成中提取 shape、有效分片轴和 mesh 尺寸，并使用已有、仅限当前 `build_bucket_plans` 调用的 rank→坐标映射。每个 participant 仍调用原生 `Shard.local_shard_size_and_offset` 公式。`_storage_regions` 保留当前兼容 backport 的通信子组匹配、placement 检查、正交 EP 分片、transport-local domain shape 和 offset 转换；非零 rank 与乱序 ranks 不按默认顺序推断。无法使用已有坐标映射时走原读取路径，不共享跨参数的几何或 route 结果。

**Meta 初始化：只减少未读取的空对象。** [meta_initialization.py](../torchtitan_simulator/compat/meta_initialization.py) 独立于主包规划扩展。上游 `_chunk_with_empty` 先执行 `torch.chunk`，再补齐到分片数；在大量分片为空时，每个参数都可能创建很多无用空 Tensor，而当前初始化 consumer 只读取自己的分片。`_InitializationChunks` 保留原生 chunk，只在访问某个补齐索引时创建对应的空 Tensor；同一索引复用身份，不同索引保持独立。4096 分片测试中，初始只有原生 chunk，读取两个空索引只额外创建两个空 Tensor。

接入仅作用于 `FSDPParam._init_sharded_param` 和 `_init_sharded_post_forward_param_metadata` 的私有函数命名空间。原生方法主体、默认参数、闭包、`no_grad` 及已有规范 wrapper 均保留；不替换通用 `_chunk_with_empty`，避免破坏其他调用者的 list 契约。非 Meta tensor 使用原 helper。`runner.py` 只在 `parallelize` 内启用 `PatchScope`，正常或异常退出都恢复方法，之后的预热与 F/B/O 捕获不在该作用域中。参数、优化器状态、Muon 常驻工作缓冲及传输节点的登记方式不变。

### 等价性与恢复验证

本次共 170 项测试通过，分为 50 项规划／Gloo 测试及最终普通 wheel 下的 120 项侧车回归；后者包含 18 项 Meta 初始化测试，不能重复相加为 188 项。

| 验证层 | 证据与范围 |
| --- | --- |
| 规划差分 | `tests/unit_tests/extensions/torchtitan/distributed/flex_shard/test_planning.py`：与保存的原生实现比较计划和 packed schedules；覆盖空分片、复制持有者、乱序 ranks、非法边界和异常诊断。单独改变 tensor shape 或 expected partition 必须仍被拒绝。 |
| 内存与作用域 | 同一测试文件验证证明淘汰、资格预算、身份误匹配、可变几何回退、成功／异常／嵌套退出后的引用清空。无跨参数或跨会话计划保留。 |
| CPU 多卡传输 | 实际 2、4 rank Gloo 双向传输，以独立 tensor 切片预期核对搬入、计算和搬回；4 rank 覆盖正交分片、非零 rank 的 domain shape／offset。 |
| Meta 恢复 | `tests/test_meta_initialization.py`：原 helper 回退、惰性分片身份、装饰器和默认参数、弱引用释放；真实 runner 在 parallelize 中抛异常后再运行 Muon／AdamW，与基线的布局、算子几何、依赖及显存一致。 |
| 侧车回归 | 配置、捕获、All2All 依赖、优化器／激活卸载、AdamW 契约、DSV4、内存交付和 runtime；最终普通 wheel 从源码目录外运行，使用 `-o pythonpath=` 关闭 pytest 的源码路径注入。120 项通过，198.55 秒。 |

对应测试入口可分别执行以下命令；运行环境须先安装本仓主包和 sidecar，加载实际 CANN 环境并设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。命令中的 Python 应指向待验证版本的环境，不用业务 `PYTHONPATH` 代替安装。

```bash
# 仓库根目录：50 项规划与 Gloo 测试
python -m pytest -q tests/unit_tests/extensions/torchtitan/distributed/flex_shard/test_planning.py

# experiments/simulator 目录：102 项相关回归 + 18 项 Meta 测试
python -m pytest -o pythonpath= -q \
  tests/test_host_timing.py tests/test_muon_config.py tests/test_muon_capture.py \
  tests/test_muon_dependencies.py tests/test_muon_offload.py tests/test_optimizer_contract.py \
  tests/test_dsv4_capture.py tests/test_memory_delivery.py tests/test_runtime.py \
  tests/memory/test_optimizer_state.py tests/capture/test_optimizer_transfers.py \
  tests/test_meta_initialization.py
```

最终 wheel 中四个生产文件与验收源码 SHA256 一致。修改文件的 Ruff、format 和 diff 空白检查通过；全仓检查未全绿：未改动的 `test_fsdp_residency.py` 有 3 项 Ruff 问题，主包有 38 项既有类型问题，侧车类型检查另有未改动量化导入的 `torchao_npu.wrapper_tensors.mx_wrapper_tensor` 解析错误。这里验证的是规划等价、CPU Gloo 传输和 Meta 结构，没有执行真实 NPU loss／grad_norm 或设备性能验证。

### Pro 4096 宿主验收

2026-11-05，以 route 区间裁剪后的 `b9859a65` 普通 wheel 为基线，同机串行对照当前计划校验、几何和 Meta 初始化优化。配置为 DeepSeek-V4 pro、world=4096、FSDP=2048、DP_R=2、EP=128、rank=0、LBS=1、GBS=4096、seq=4096，TP/CP/PP=1、无 AC、FP32 参数／BF16 mixed／FP32 reduction、Muon NS=10。仅测 Titan，未调用平台或修改捕获。

| 场景 | 基线端到端秒 | 优化后秒 | 基线峰值 RSS GiB | 优化后峰值 RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| Muon | 165.41 | 140.78 | 4.55 | 4.52 |
| Muon 状态卸载 | 161.44 | 141.03 | 4.66 | 4.63 |
| AdamW | 66.02 | 50.41 | 1.98 | 2.03 |

时间为新进程中脚本计时起点至 `simulate()` 返回的宿主墙钟，扣除额外计划指纹审计；图指纹在计时和 RSS 取样后计算。每场景单次冷进程，不代表统计性能分布，更不是设备训练耗时。三组的配置（除输出路径）、完整捕获／交付图节点与依赖、计划、packed schedules、静态显存和 workspace 容量／寿命均精确相等。峰值 RSS 使用 Linux `RUSAGE_SELF.ru_maxrss`，没有数量级增长。

Muon 的 `parallelize` 为 19.93→3.98 秒，geometry 为 20.54→12.82 秒；包含资格检查的完整 validation 为 40.71→41.22 秒，本轮未显示该分项的确定加速。主要收益来自 Meta 初始化和几何提取；capture 未改，其耗时波动不归因为捕获优化。上述阶段包含子调用，不可相加。

| Muon 阶段 | 基线秒 | 优化后秒 |
| --- | ---: | ---: |
| parallelize | 19.934 | 3.978 |
| optimizer_build | 88.037 | 83.188 |
| optimizer_planning.geometry | 20.542 | 12.816 |
| optimizer_planning.validation | 40.712 | 41.220 |
| optimizer_planning.routes | 76.974 | 72.584 |
| optimizer_planning.packing | 0.993 | 0.966 |
| warmup | 5.768 | 6.009 |
| capture_step | 42.181 | 37.551 |
| workload_validation | 3.912 | 4.114 |
| fsdp_projection | 5.077 | 5.067 |

`optimizer_build` 包含约 6 秒的计划指纹审计，端到端表已扣除这部分；`routes` 包含内部 geometry 和 validation。表中捕获耗时变化是本轮观测，不作为本次优化修改了捕获的证据。

另在同一进程连续生成两次 pro，每次删除交付对象并 `gc.collect()`，不调用 `malloc_trim`。基线两轮峰值 RSS 为 4.53／4.55 GiB，释放后为 2.33／2.76 GiB；候选两轮峰值均为 4.52 GiB，释放后为 2.19／2.61 GiB。第二轮保留增量没有高于基线，未出现数量级增长；两轮静态显存一致，规划上下文均已清空。释放后 RSS 包含分配器保留空间，这两轮实验不等同于无限会话内存稳定性证明。

| 实现／会话 | simulate 秒 | 峰值 RSS GiB | 持图 RSS GiB | 释放后 RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| 基线／1 | 161.010 | 4.528 | 3.268 | 2.329 |
| 基线／2 | 154.162 | 4.554 | 3.668 | 2.758 |
| 优化后／1 | 134.474 | 4.519 | 3.514 | 2.194 |
| 优化后／2 | 134.143 | 4.519 | 3.884 | 2.614 |

连续会话表仅计 `simulate()`，不包含导入和额外指纹审计，不能直接与前面的冷进程净耗时相减归因。候选持图 RSS 略高于基线，但峰值及释放后 RSS 未上升，故不宣称所有时刻的 RSS 都下降。验收环境为 Python 3.12、Torch 2.14.0 CPU、TorchTitan 0.3.0，使用 CANN Meta 适配；world=4096 是逻辑并行配置，不代表启动了 4096 个设备进程。
