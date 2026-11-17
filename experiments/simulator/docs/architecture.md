# 训练负载生成的整体方案

Simulator 在 Meta 张量上执行原生模型、并行化、loss 和优化器，观察实际算子、通信、依赖和存储生命周期，交付训练负载与静态显存预算。它不计算硬件时延，也不实现平台 DES。本文先解释方案及不可破坏的边界；新增模型按[模型接入工作流](model-integration.md)推进，捕获细节查[运行时契约](capture-runtime.md)。

## 1. 为什么复用原生训练路径

模型源码决定 attention、MoE、MTP、loss 和更新行为；原生 parallelize 决定本卡 shard、通信和 reshard；autograd 决定真实反向与保存值。直接复用这些路径，才能让训练开关真实影响负载，避免维护一套根据模型名称手工拼接的图。

Meta 没有张量数据，能够执行 shape、dtype、layout 和 autograd 结构逻辑，但不能提供数据相关路由或数值结果。输入准备在 CPU 计算必要的静态元数据，设备张量转到 Meta；MoE token 路由使用明确的均衡假设。没有数据时不能用 `.item()`、随机占位值或经验猜测伪装成真实事实。

当前标准入口支持 DeepSeek-V4、Kimi K3 与 Qwen3-32B（dense）。`module.factory` 是配置来源扩展点，不代表任意模型已满足运行接口。新增模型必须接通并验证原生构建、forward/loss、parallelize、optimizer 和保存值链路，不能只增加名称白名单。

## 2. 组件边界

以下源码相对于 `experiments/simulator/torchtitan_simulator/`。

| 组件 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| `config.py`、`presets.py`、`model_overrides.py` | 将显式请求转换为新建的原生配置树，应用受支持的 override、AC、并行和量化配置 | 不静默接受未消费参数，不凭 preset 名称选择计算精度 |
| `runtime.py`、`compat/` | 有作用域的 Meta、autograd、分布式兼容；模型特有虚拟融合 | 不修改已安装上游源码，不永久替换健康的原生实现 |
| `runner.py`、`warmup.py` | 构建和并行化模型、准备输入、结构预热、捕获与交付 | 不执行真实设备训练、性能预测或 GA 时间展开 |
| `capture/` | 观察 dispatch、参数身份、通信组、流事件、FQN、重算和保存值；组织图与卸载计划 | 不把捕获序号当作全局执行顺序，不按算子名删除必要依赖 |
| `ir/` | 保存节点、阶段模板、调度引用及外层工作负载 | 不等于平台公共图或 DES 内部对象 |
| `memory/` | 根据本地 storage、生命周期、保存槽和 FSDP 驻留生成静态预算 | 不估计 allocator reserved、碎片、未声明 workspace 或 DES 驱动的动态峰值 |
| 主包 `torchtitan_npu/models/`、`ops/`、`override/` | 原生模型结构、参数、并行化、真实算子和配置级组件替换 | 不反向依赖 simulator |

预测、硬件配置、资源分类、L1 DES、平台 L2 的阶段次数和指标归下游平台／算子仓。训练侧可以生成 H2D/D2H 及等待节点，但不填硬件传输耗时。源码中的 `iteration_time_est` 等遗留占位字段不构成已实现的性能预测接口。

## 3. 一次 simulate 的真实执行顺序

实现入口为 [`runner.simulate`](../torchtitan_simulator/runner.py)，不是替换 `Trainer.train`。

1. **配置解析。** `prepare_config()` 校验请求，创建原生 trainer 配置，重建受支持的模型几何，应用显式 override；关闭本捕获路径不支持的 compile/CUDA graphs，拒绝 PP、FP8 通信等未支持配置。
2. **建立会话。** `meta_runtime()` 安装可恢复兼容；`logical_world()` 创建指定 world/rank 的 fake process group 和 mesh。单进程代表一个 logical rank，不等于 world=1。
3. **构建与并行化。** `build_meta_model()` 构建并初始化 Meta 模型，再调用原生 `model_spec.parallelize_fn`，保留实际 local shape、DTensor placement 与 mesh。量化由原生 recipe 选择参数，不用模拟器 FQN 名单重写。
4. **准备训练组件和输入。** 构造原生 loss、optimizer、post-optimizer hook 与 scheduler；loss 绑定 lm_head。CPU 生成合成单文档 next-token 输入及有效 token 数，调用模型的 attention mask／CP planner，再转为 Meta。
5. **结构预热。** 同一模型与 optimizer 先运行实际 GA 次 F/B、一次 clip/update，清梯度；不安装正式观测器，不推进 scheduler，不修改 `training.steps`。目的是完成优化器状态等惰性初始化，不代表数值意义上的第二训练步。
6. **正式捕获。** 预热后安装 FSDP 驻留观察器，登记初始输入、已有 optimizer state 及 Muon runtime 常驻工作缓冲。`run_simulation_step()` 执行一份代表性 microbatch 的 F/B，再执行 clip、optimizer 和 scheduler；观察 dispatch、流事件、保存槽与生命周期，组装模板和静态内存。
7. **校验与负载变换。** 核对所有应有参数梯度的 shape/dtype/placement/local shape，拒绝捕获期间未建模的 buffer storage/layout 变化。先校验完整捕获，再生成 FSDP 通信视图；按独立开关插入激活／优化器传输与等待，修正派生静态显存，重新校验引用与 DAG。
8. **交付。** 返回 `WorkloadGraph`，默认不创建输出目录。只有显式 `output_formats=["mem"]` 才打印预算并导出 memory summary 与诊断明细；会话结束恢复临时 patch、逻辑分布式和捕获上下文。

低层 `run_simulation_step()` 不自动预热。调用低层入口不能假定获得默认 `simulate()` 的稳态口径。详见[结构预热](steady-state-capture.md)。

### F、B、O 与 GA

`comp_type` 表示所属语义 action，`phase`／`execution_kind` 表示实际 autograd 执行。chunked loss 可能在 F action 内执行 head backward；不能把所有 `phase=backward` 的算子移到 B 模板。重算也必须保留其真实执行身份与 FQN。

正式交付的 F/B 只有一个 microbatch；O 为一次更新，含其实际 clip 等路径。配置 GA 与 `captured_microbatches=1` 都保留。平台可按当前无 PP 方案组织 `(F → B) × GA + O`，但训练仓的静态内存不因此成为多 microbatch 累积回放。

训练 IR 中的 L0／L1／L2／L3 是节点、模板、调度引用、工作负载的组织层次，不代表已经实现平台 L1/L2 DES。`SchedulePlan` 和 `ScheduleGraph` 保留捕获行为和数据传递，不能把它们的存在理解为 PP 或所有跨迭代调度均已支持。

## 4. 三种信息必须分开消费

| 信息 | 读取位置 | 含义 |
| --- | --- | --- |
| 对外交付图 | `workload.step_templates` | FSDP 通信视图，加启用的卸载传输／等待；平台正常消费此图 |
| 完整原始图 | `workload.captured_workload` | 投影前的源节点、原始锚点和框架准备操作，供事实审计 |
| 静态显存与事实 | `workload.iteration.schedule.annotations` | `memory_plan`、`memory_summary`、实际 batch/capture/model/precision 等事实；不依赖文件 |

FSDP 通信视图有意隐藏 FSDP scope 内的非通信准备／搬移算子，保留模型计算、通信和流等待；按原图可达路径传递可见依赖。不能改为按根模块名筛选，也不能因希望获得更多节点就默认切换到源图预测。

内存仍观察完整捕获的 storage，包括被隐藏的分配、staging、cast 和释放。源内存事件的 op_id 应到 `captured_workload` 查找，不要求每个事件在交付图都有可见节点。卸载随后创建派生预算，但原始事件、lifetime 和审计来源继续保留。

## 5. 决定正确性的几个取舍

### 先观察原生行为，再补兼容

健康的 `torch.ops`／CANN dispatch 与 autograd 直接复用。先用真实 schema、Meta 调用和梯度探针定位缺口；只有确认必要时才桥接 Meta 分配或 backward。必须自行定义融合边界时，注册明确的 custom op、fake 实现、autograd 和真实保存输入，随节点声明角色与语义轴。不能用同 shape 的新 tensor 冒充实际保存值，不能把融合内部拆开后又重复保留融合节点。流程见[模型接入](model-integration.md)和[自定义张量语义](custom-tensor-semantics.md)。

### 数据依赖与流依赖共同约束执行

producer/consumer、alias/mutation、同流前沿和 event 等待表达不同事实。通信完成不等于 Python launch 返回；零耗时控制节点仍会阻塞后继。`record_stream` 管生命周期，不是执行屏障。不能将整图按捕获顺序串行化，也不能按零耗时删除节点。

### 保存值不等于仍占设备内存，更不等于可卸载

保存槽、Tensor wrapper、底层 storage 及 allocation generation 分开观察。最后使用、第一次 unpack 或 Python wrapper 销毁都不能单独证明释放。层归属、后续 B/recompute 使用及共享 storage 决定 offload 候选；head／层外值不能伪装成 decoder 层，也不能为报表整洁而丢弃。

### 卸载同时改变执行图与静态预算

激活卸载使用逐层保存量和 F/B 边界生成传输／等待；优化器卸载按实际本卡 state 生成独立双向传输窗口。平台负责耗时与 DES 的并发／掩盖，不再额外重复加拖尾。静态预算由训练侧按同一计划修正，不能由平台再猜一次。两种开关独立；优化器充分流水近似忽略逐 tensor 搬运 buffer，激活晚产生 payload 则可能保守驻留到 F 末。详见[激活卸载](activation-offload-design.md)、[优化器卸载](optimizer-offload.md)。

## 6. 从方案进入开发

新增模型先完成[模型接入工作流](model-integration.md)的小模型、真实 F/B/O、梯度与独立字节先验，再逐步增加并行、AC、量化和卸载组合。修改捕获基础设施先核对[运行时契约](capture-runtime.md)；当前支持范围查[能力矩阵](capability-matrix.md)。

Meta 测试证明结构、元数据和逻辑内存路径，不证明 NPU loss、梯度数值、HCCL 时延或 allocator 峰值。旧版本完整模型的通过次数和时间常数不能替代当前源码的验收。
