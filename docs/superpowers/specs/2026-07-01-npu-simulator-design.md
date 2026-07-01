# torchtitan_npu Simulator — 架构与方案设计

- Status: Draft for review
- Author: Copilot CLI (with @Pastens)
- Date: 2026-07-01
- Acceptance target: `deepseek_v4_pro_debug_61_layers_4k_384die`（本仓
  `torchtitan_npu/models/deepseek_v4/config_registry.py`）可以通过 simulator
  方式跑起来，产出正确的四层 IR 与可视化结果。

## 1. 背景与目标

在原始 torchtitan-npu 训练 workflow（`torchtitan_npu.entry:main` →
`torchtitan.trainer.Trainer`）之上，新增一个**侧载（side-loaded）** 的
`simulator` 子包：

1. **零硬件 / 零真实显存**：不需要真实 NPU 卡、不需要 `torch_npu` 的物理设备
   （`torch.npu.is_available()` 可以为 `False`），不为参数/激活值分配真实内存。
2. **图捕获完整、正确**：捕获训练一个 step 中，*所有卡*（即完整并行拓扑下的
   全部 rank）的计算图，覆盖 forward / backward / optimizer。
3. **通信域与通信量正确**：并行策略（DP/TP/PP/CP/EP 等）产生的通信域
   （类似 RankTable）和通信量（bytes）要能从捕获中正确构建出来，而不是编造。
4. **MoE 等动态路由强制确定性**：路由 / 专家分发等依赖运行时数据的逻辑，默认
   通过 patch 方式强制负载均衡，从而在 meta/shape-only 张量下也能产出确定的
   token→expert 映射和正确的 shape。
5. **四层 IR 可观测性**：单个 TrainStep 输出 L0 OpNode → L1 StepGraph → L2
   ScheduleGraph → L3 WorkloadGraph 四层信息，并可视化。四层 IR 定义见
   `https://github.com/ZhanluModelSim/workload-model-platform/tree/master/spec`
   （L0-OpNode.md / L1-StepGraph.md / L2-ScheduleGraph.md / L3-WorkloadGraph.md）。
6. **验收标准**：本仓 `deepseek_v4_pro_debug_61_layers_4k_384die`
   （61 层、384 专家、`expert_parallel_degree=192`、384 die）能以 simulator
   方式跑起来，产出完整、结构正确的四层 IR 和可视化文件。

### 非目标（Non-goals）

- **不做真实性能/时延仿真**（不是 SimuMax / DES 那类 roofline 仿真器）。
  L0 的 `flops` / `comm_bytes` / `peak_mem` 字段按 best-effort 静态公式估算，
  用于观测和后续研究，不承诺与真实上板时间对应。
- **不做真实数值正确性验证**（meta 张量没有数值，`torch_npu` 部分自定义算子
  在 meta 下也没有注册 autograd kernel，反传数值不可信——这在设计中是已知且
  接受的限制，我们只关心 shape/结构/依赖关系）。
- **不引入对同组织其他仓库（workload-model-platform / torchtitan-simulator /
  virtual-npu 等）的依赖**——按仓库约定，本仓保持 clean，不新增外部仓库依赖
  （用户明确指示：“不要依赖其他仓库，我们是一个clean的项目”）。四层 IR 的
  dataclass 按 spec 文档在本仓原创实现。

## 2. 已验证的关键可行性结论（Spike 结果）

在正式设计/实现前，按用户要求做了小规模验证。本仓开发沙盒本身没有 CANN/
`torch_npu`（`import torch_npu` 会因缺少 `libhccl.so`/`libascend_hal.so`
失败），因此使用宿主机已有的 CANN 容器镜像
`quay.m.daocloud.io/ascend/cann:9.1.0-beta.1-950-ubuntu22.04-py3.12` 新启动了
一个干净容器（`titan-npu-sim-validate`），安装 `torch==2.10.0+cpu` +
`torch_npu==2.10.0`（PyPI 公网可得的最接近版本；生产环境请用本仓
`requirements.txt` 锁定的 `torch_npu==2.12.0rc1` 复验），验证结论：

| # | 验证项 | 结果 | 说明 |
|---|--------|------|------|
| 1 | `import torch_npu`（容器内 source `set_env.sh` + 补 `LD_LIBRARY_PATH`） | ✅ 成功，`torch.npu.is_available()==False` | 证明"有 CANN 工具链、无物理硬件"这一目标场景是成立的 |
| 2 | `torch_npu.npu_rms_norm` / `npu_rotary_mul` / `npu_swiglu` / `npu_moe_token_permute` 在 `device="meta"` 上直接调用 | ✅ 全部成功返回正确 shape | 证明 NPU 自定义算子已有 Meta 核注册（大概率是为了支持 `torch.compile`/Dynamo，本仓 `deepseek_v4_pro_*` 配置已启用 `compile.enable=True` 且 DeepSeek-V4 全流程编译已验证，这是有力的先验证据） |
| 3 | `dist.init_process_group("fake", rank=0, world_size=384)`（单进程） | ✅ 成功，`get_world_size()==384` | 证明"单进程模拟全部 384 卡"的既有机制（`comm.mode=fake_backend`）可行 |
| 4 | `dist.all_reduce(meta_tensor)`（在 fake PG 下） | ❌ `NotImplementedError: c10d::allreduce_` 无 Meta 核 | **关键发现**：FakeProcessGroup 只解决"不需要多进程"，并不解决"张量在 meta device 上"的问题。c10d 集合通信算子本身没有 Meta 核注册 |
| 5 | `TorchDispatchMode` 包裹一个含 `npu_rms_norm` + `npu_moe_token_permute` 的 tiny MoE 层，在 meta 上做 forward+backward | ✅ 捕获到 21 个 op（含两个 NPU 自定义算子），forward/backward 都跑通 | 证明本设计的核心捕获机制（dispatch 级拦截）可行；有 `autograd kernel not registered` 的 UserWarning（数值不可信但不影响捕获，符合“不做数值验证”的非目标） |

**结论**：核心矛盾点只有一个——集合通信算子在 `meta` 张量 + `fake`
ProcessGroup 下不能直接跑（#4）。这与本仓已有的
`torchtitan_npu/distributed/process_group.py::is_fake_process_group()` +
`NpuExpertParallel`（`converters/kernels/moe_dispatch.py`）在 `is_fake` 分支
跳过真实 `all_to_all_single` 的做法，是**同一个问题的同一类解法**：
在“是 fake/模拟场景”时，绕过真实集合通信调用，直接在 Python 层构造正确
shape 的输出并记录一次“通信事件”。本设计把这一模式，从 MoE dispatch 一处，
推广成通用拦截层，覆盖 FSDP2 all-gather/reduce-scatter、TP all-reduce、
DP grad all-reduce、PP send/recv 等所有集合通信入口。

## 3. 架构总览

```
                          ┌───────────────────────────────────────────┐
                          │      torchtitan_npu.simulator (新增)        │
                          │                                             │
 现有训练入口 (不变)         │  SimulationTrainer(Trainer)                 │
 torchtitan_npu.entry      │    · 复用现有 ModelSpec / parallelize_fn /   │
   --module torchtitan_npu.│      pipelining_fn（DeepSeek-V4 一行代码不改）│
     simulator             │    · 强制 comm.mode=fake_backend            │
   --config deepseek_v4_   │    · 强制 debug.moe_force_load_balance=True │
     pro_simulate_61_layers├──▶ · 强制 compile.enable=False（捕获用不到）  │
                          │    · meta_env: 模型全程留在 meta，不 to_empty │
                          │    · fake_collectives: 集合通信拦截+录制      │
                          │                                             │
                          │  capture/ (L0 op 级 dispatch 捕获,           │
                          │            L1 step 边界, L2 rank/调度编排)    │
                          │  ir/     (L0-L3 dataclass，按 spec 原创实现)  │
                          │  cost/   (NPU 算子 FLOPs/mem/comm_bytes 估算)│
                          │  rank_table.py (通信域 = ParallelDims 展开)  │
                          │  viz/    (HTML + JSON 导出)                  │
                          └───────────────────────────────────────────┘
```

关键设计原则：**SimulationTrainer 只改变"怎么跑"和"跑的时候录什么"，不改变
"跑什么"**——模型结构、`parallelize_fn`、`pipelining_fn`、MoE dispatch
converter 全部原样复用 `torchtitan_npu.models.deepseek_v4`，不 fork 出一份
模型代码。这是“侧载包”的核心含义：新增文件，不修改现有文件。

## 4. 包结构

```
torchtitan_npu/simulator/
├── __init__.py
├── config_registry.py        # SimulationTrainer.Config 工厂函数（每个想模拟的模型配置一个）
├── trainer.py                 # SimulationTrainer(Trainer)：一步训练 + 捕获 + 导出
├── meta_env.py                 # 把 device_type/device_module 强制指向 "meta"；模型全程不 to_empty
├── fake_collectives.py         # 拦截 torch.distributed.* / funcol.* 集合通信，meta 下不做真实通信，只记录事件
├── moe_force_balance.py        # 校验/强制 debug.moe_force_load_balance=True 的兜底 patch
├── rank_table.py                # 从 ParallelDims/DeviceMesh 展开通信域（RankTable）
├── ir/
│   ├── __init__.py
│   ├── tensor_meta.py          # TensorMeta
│   ├── op_node.py              # OpNode                         (L0)
│   ├── step_graph.py           # StepGraph                      (L1)
│   ├── schedule_graph.py       # StepInstance/TensorSlot/DataPass/ScheduleGraph (L2)
│   └── workload_graph.py       # DataFlow/IterationSpec/WorkloadGraph          (L3)
├── capture/
│   ├── __init__.py
│   ├── dispatch_capture.py     # TorchDispatchMode: 捕获每个 aten/npu op -> OpNode 流
│   ├── step_boundary.py        # backward()/optimizer.step() 打点 -> forward/backward/optimizer 归类 (L1)
│   ├── comm_events.py          # fake_collectives 记录的通信事件 -> DataPass + comm_bytes (L2)
│   └── schedule_builder.py     # 组装 ScheduleGraph(L2) 与 WorkloadGraph(L3)
├── cost/
│   ├── __init__.py
│   └── op_cost_model.py        # op_type -> (flops, peak_mem, comm_bytes) 估算，可扩展注册表
└── viz/
    ├── __init__.py
    ├── json_export.py          # 四层 IR 结构化 JSON
    └── html_export.py          # 自包含 HTML：RankTable 网格 + 调度泳道 + 算子 DAG + 统计卡片
```

`pyproject.toml` 的 `[tool.setuptools.packages.find] include = ["torchtitan_npu*"]`
已经是通配符，新增子包无需改动打包配置。测试放在
`tests/unit_tests/simulator/`（不依赖真实 NPU 的部分）和
`tests/smoke_tests/simulator/`（需要真实 `torch_npu` 的部分，用
`npu_available`-类似的 fixture 跳过）。

## 5. 核心组件设计

### 5.1 Meta 执行替身层（`meta_env.py`）

沿用 `Trainer.__init__` 现有流程（`with torch.device("meta"): model =
model_config.build()`），但**跳过**其后的 `to_empty(device=init_device)` +
`init_weights()` 物化步骤——模型参数、激活值全程停留在 `meta` device。
`SimulationTrainer` 通过覆盖 `Trainer` 中触发物化的那一小段（而不是整个
`__init__`）来做到这一点，其余 tokenizer/dataloader/parallelize/optimizer
构建逻辑完全复用父类实现。

`config.comm.mode` 强制设为 `"fake_backend"`（复用 torchtitan 已有的
`init_fake_mode`，见 §2 验证 #3），因此 `ParallelDims.build_mesh()` 不需要
任何改动：384 的 world_size 在单进程下即可完整建立 mesh、划分
dp/tp/pp/cp/ep 各维度子组。

**补充验证（对照本仓 `requirements.txt` 锁定的 torchtitan commit
`ac13e536c84e7f6647b14fa9375c3c8a8a2b8578` 源码逐行核实）**：

1. `Trainer.__init__` 里 `self.device = torch.device(f"{device_type}:{LOCAL_RANK}")`
   （`trainer.py:207`）。经验证 `torch.device("meta:0")` 与
   `torch.device("meta")` 行为完全一致（张量 `.device` 归一化为
   `meta`），因此**只需要在 `Trainer.__init__` 执行前把
   `torchtitan.tools.utils.device_type`/`device_module` 猴子补丁为
   `"meta"` + 一个安全桩**，`Trainer.__init__` 本身完全不用改——包括
   `model.to_empty(device=init_device)` + `init_weights()`（`trainer.py:
   407-411`，已验证 `nn.init.trunc_normal_/zeros_/normal_` 和
   `Module.to_empty` 在 meta 张量上都能正常跑，不需要单独绕过）。这比最初
   设想的"覆盖 Trainer 内触发物化的那一小段"更简单、更不依赖具体版本的行号。
2. `torchtitan.components.metrics` / `torchtitan.distributed.parallel_dims` /
   `torchtitan.distributed.utils` 三个模块都在 import 时**按值**引入了
   `device_type`/`device_module`（`from torchtitan.tools.utils import
   device_type, device_module`），必须在这三处也做同名重绑定，否则它们
   仍然持有 patch 前的旧引用。
3. **新发现的两个 meta 崩溃点**（均已用 pinned 源码 + 本地脚本复现确认）：
   - `torchtitan.distributed.utils.set_determinism()`：当 `world_size>1`
     且 `debug_config.seed is None` 时，会执行
     `seed_tensor.to("cpu").view(torch.uint64).item()`
     （`distributed/utils.py:159`）。已验证 `meta_tensor.to("cpu")` 直接
     抛出 `NotImplementedError: Cannot copy out of meta tensor; no data!`。
     **修复**：`SimulationTrainer` 强制 `config.debug.seed` 为一个固定整数
     （如 `42`），这样 `if seed is None:` 分支（含那次广播+`.item()`）整个被
     跳过，不需要碰 `set_determinism` 本身。
   - `Trainer.train_step()`（`trainer.py:719-`）在两处不安全：
     ① `dist_utils.dist_sum(local_valid_tokens, batch_mesh)` 内部
     `_dist_reduce()` 无条件调用 `.item()`（`distributed/utils.py:55,58`）；
     ② 日志部分 `float(loss.detach().item())` / `float(grad_norm.item())`
     （`trainer.py:807,818`）。**修复**：`SimulationTrainer` 不调用
     `Trainer.train_step()`，而是直接调用 `forward_backward_step()`
     （`trainer.py:653`，其 `global_valid_tokens` 参数在真实调用点其实传的
     是 `dist_sum` 返回的 Python `float`，类型注解 `torch.Tensor`
     只是历史遗留，与我们直接传入一个由静态配置算出的 Python
     `float`——如 `local_batch_size * seq_len`——完全兼容），并跳过
     `clip_grad_norm_`/日志，直接调用 `self.optimizers.step()` +
     `self.lr_schedulers.step()`。这样一次 forward+backward+optimizer
     step 全程不触发任何 `.item()`。

### 5.2 集合通信拦截层（`fake_collectives.py`）

这是解决 §2 发现的核心矛盾（`c10d` 算子无 Meta 核）的关键组件。设计：

- 在 `SimulationTrainer` 的捕获上下文进入时，monkeypatch 一组入口：
  `torch.distributed.{all_reduce, all_gather, all_gather_into_tensor,
  reduce_scatter, reduce_scatter_tensor, all_to_all, all_to_all_single,
  broadcast, barrier}`，以及 DTensor 依赖的
  `torch.distributed._functional_collectives.{all_reduce, all_gather_tensor,
  reduce_scatter_tensor, all_to_all_single}`。
- 每个 patch 后的函数：
  1. 判断当前 group 是否 fake（复用已有
     `torchtitan_npu.distributed.process_group.is_fake_process_group`）；
     不是 fake 就调用原始实现（保证非模拟场景行为不变）。
  2. 是 fake 时，**不调用底层 C10D 方法**，而是在 Python 层直接构造正确
     shape/dtype 的输出 tensor（例如 all_gather 按 group size 复制/拼接
     shape，reduce_scatter 按 group size 切分 shape，all_to_all_single 按
     `input_split_sizes`/`output_split_sizes` 构造），保证下游代码拿到的
     shape 与真实语义完全一致。
  3. 记录一条通信事件：`(op_type, group 的 mesh 维度名, 全局 rank 列表,
     tensor shape/dtype -> bytes, 是否 async)`，供 §5.5 消费生成 L2
     `DataPass`。
- 与 `NpuExpertParallel`（MoE dispatch）里已经存在的 `is_fake` 分支是同一模式
  的推广，两者不冲突：MoE dispatch 的 fake 分支本来就不落到
  `all_to_all_single`，因此不会被本层拦截影响；本层主要覆盖 FSDP2 的
  all-gather/reduce-scatter、TP 的 all-reduce、DP 的梯度 all-reduce、PP 的
  send/recv。

### 5.3 强制负载均衡（`moe_force_balance.py`）

`torchtitan_npu.models.deepseek_v4.moe.TokenChoiceTopKRouter` 已经内置
`debug_force_load_balance` 开关（`config_registry.py` 里
`deepseek_v4_pro_debug_61_layers_4k_384die` 已经把
`debug.moe_force_load_balance=True`），验证过其
`_debug_force_load_balance_routing` 走的是与输入数值无关的 round-robin 逻辑
（不依赖 `.item()`/数据相关控制流），在 meta 张量下安全。

`SimulationTrainer` 在构建 `TrainerConfig` 时**无条件**将
`config.debug.moe_force_load_balance = True`（如果用户传入的配置忘记开启，
仍然强制打开并打印一条 warning），这是对用户需求“默认采用打patch的方式进行
强制负载均衡”的字面落实：不管传入什么配置，模拟器路径下都保证强制负载均衡
生效，而不是依赖使用者记得设置。若未来出现其他模型（非 hash 路由、非
round-robin）的动态路由逻辑无法直接复用这个开关，视为该模型接入模拟器时的
增量工作，在对应 model 的 `simulator` 适配层单独加一个等价的强制路由 patch，
不影响本设计的通用结构。

**关键正确性依据（不只是"数据无关"，而是"跨 rank 完全一致"）**：查看
`torchtitan.models.common.moe.TokenChoiceTopKRouter._debug_force_load_balance_routing`
的实现：

```python
selected_experts_indices = (
    torch.arange(n_tokens * self.top_k, device=scores.device, dtype=torch.int64)
    .reshape(n_tokens, self.top_k) % self.num_experts
)
```

`torch.arange(n_tokens * top_k)` 不依赖 rank、也不依赖输入数值，只依赖
`n_tokens`（由固定的 `local_batch_size`/`seq_len` 决定）。也就是说，在相同
配置下，**所有 384 个 rank 会算出完全相同的 `selected_experts_indices` 和
`num_tokens_per_expert`**。这是 §5.5 "L2 只需捕获一份模板，其余 rank 复用"
设计成立的数学依据，而不仅仅是工程上的近似简化：EP all-to-all 的
`input_splits`/`output_splits`（由 `num_tokens_per_expert` 推出）在所有
EP rank 上也完全一致，通信模式天然对称。

### 5.4 算子级捕获（L0，`capture/dispatch_capture.py`）

用 `torch.utils._python_dispatch.TorchDispatchMode` 包裹一次训练 step（§2
验证 #5 已证明可行）。对每次 `__torch_dispatch__` 回调：

- 记录 `raw_op_type = str(func)`（如 `aten.addmm.default` /
  `npu.npu_moe_token_permute.default`）。
- 用 `id(tensor)` 做张量级 producer 追踪（meta tensor 没有 storage
  data_ptr 可用于别名分析，L0 spec 里也明确"Meta tensor 环境下关闭存储级
  追踪，退化到纯 id(tensor) 级"——与我们的场景完全吻合）。
- 输入/输出转换为 `TensorMeta`（name/shape/dtype/device/is_parameter）。
- **去重（dedup）**：61 层模型结构高度重复，直接展开会产生数万级节点。采用
  与 spec 描述一致的做法：对相邻重复的 "op_type + shape 签名" 序列做
  `repeat_count` 折叠（例如 61 个 `TransformerBlock` 中除第 0/1 层外的其余
  59 层，如果 op 序列与前一层完全同构，则合并为一条带 `repeat_count=59` 的
  记录，同时保留完整的第一层结构用于展开查看）。这保证了 61 层/384 专家
  规模下 L0 图仍然可读、可视化不卡死。
- 每个 op 通过 `OP_MAPPING`（规范算子名映射表，覆盖 L0 spec 的规范算子集 +
  NPU 专有算子，见 §5.8）转换为 `OpNode.op_type`，并调用 cost model 填充
  `flops/peak_mem/param_mem/comm_bytes`。

### 5.5 Step 边界与调度编排（L1/L2，`step_boundary.py` +
`schedule_builder.py`）

- **L1（StepGraph）**：复用 spec 描述的自动识别边界方式——patch
  `torch.Tensor.backward` 和当前使用的 `Optimizer.step`（`AdamW`/`Muon`/
  `swap_optimizer` 等，本仓 `OptimizerConfig` 已支持多种），在边界切换时把
  当前 buffer 归入 forward/backward/optimizer 三个 `StepGraph`。Kahn 拓扑
  排序做 DAG 校验（对应 spec"必须是 DAG"约束）。
- **L2（ScheduleGraph）**：
  - `StepInstance`：因为整张 384-rank 拓扑在单进程内即可从 `ParallelDims`
    完整得到，我们对**每个逻辑 rank**（0..383）生成一个 `StepInstance`
    （`device_ids=[rank]`，`pipeline_stage` 从 `parallel_dims.pp`
    坐标算出，`dp_group` 从 `dp_shard`/`dp_replicate` 坐标算出）。
    **模板粒度是"每个 pipeline stage 一份"，不是"整个世界一份"**：
    DP/FSDP-shard/TP/CP/EP 这些维度下，同一 pipeline stage 内的所有 rank
    运行的是*同一段模型代码、同一组 shape*（只是参数/token 内容不同），
    op 结构完全同构，因此同一 stage 内的所有 rank 复用同一个 `step_ref`
    模板；而不同 pipeline stage 持有的是模型的不同层子集
    （`_pipeline_module_split` 切分的结果），op 结构不同，需要各自独立
    捕获一份模板。验收目标 `deepseek_v4_pro_debug_61_layers_4k_384die`
    的 `pipeline_parallel_degree=1`，即只有 1 个 stage、384 个 rank 全部
    复用同一份模板——这是本设计**首先要跑通**的场景。多 stage
    （`pp>1`）场景下，需要"以第 i 个 stage 的身份"分别构建/捕获 `pp`
    份模板（即多次以不同 `pp` 坐标运行 `pipelining_fn` 切分逻辑），列为
    §9 风险中的后续扩展项，不阻塞本次验收。
    这正是 spec 里"`StepGraph` 是模板，`StepInstance` 是具体的一次
    执行"的含义。
  - `DataPass`：从 §5.2 拦截层记录的通信事件转换而来。每条通信事件展开为
    该 mesh 维度下所有参与 rank 对之间的 `DataPass`（如 all-to-all 展开为
    EP 组内两两 rank 的数据传递语义），`slots` 记录张量 shape/dtype/
    `volume_bytes`，`requires_communication=True`，`comm_primitive` 取
    `"all_to_all" | "allgather" | "reduce_scatter" | "allreduce" |
    "p2p_send_recv"`。
  - `pipeline_schedule`/`num_micro_batches`/`gradient_accumulation`/
    `zero_stage` 等字段直接从 `TrainerConfig.parallelism` /
    `training.local_batch_size` 等已有配置读出，无需重新推断。

### 5.6 通信域（RankTable，`rank_table.py`）

`ParallelDims.build_mesh()` 之后，`parallel_dims._global_meshes`（`dataloading`/
`dense`/`sparse` 等复合 `DeviceMesh`）持有多维 rank 布局张量。`rank_table.py`
按**数据驱动**方式遍历（不是硬编码维度名列表——已用真实 `ParallelDims` +
fake PG 在沙盒环境实测验证，见下）：对每个复合 mesh 的
`mesh.mesh_dim_names`（如 `("pp", "dp_replicate", "efsdp", "ep")`），沿每个
命名轴固定其余轴、切片出该轴的全部通信组：

```python
# 实测（world_size=16, dp_shard=16, ep=8）：
# parallel_dims._global_meshes["sparse"].mesh_dim_names == ("pp","dp_replicate","efsdp","ep")
# parallel_dims._global_meshes["sparse"].mesh ==
#   tensor([[[[0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15]]]])
# 沿 "ep" 轴固定 (pp=0, dp_replicate=0, efsdp=0/1) 后切片，
# 精确得到两个 EP 组：[0..7] 和 [8..15]。
```

这比"从单个 rank 的 `get_optional_mesh(dim)` 反推所有其它 rank 的组"更可靠：
`get_optional_mesh("ep")` 只返回**当前 rank 自己所在的那一组**（实测确认
仅得到 `[0..7]`），必须用复合 mesh 的完整张量才能拿到全部分组。
注意实际轴名是 `"fsdp"`（`dp_shard × cp` 合并轴，见 `dense` mesh）而非
`"dp_shard"`——当验收配置 `context_parallel_degree=1` 时两者数值相同，
`rank_table.py` 直接使用 mesh 里出现的真实轴名，不额外杜撰
`"dp_shard"` 这个轴。torchtitan_npu 自身的 `_patch_for_parallel_dims_build_mesh`
（`torchtitan_npu/train.py`）会把 `"etp"` 加入 `sparse` mesh 的轴名列表，
数据驱动的遍历方式无需特殊处理就能自动纳入。

产出一个 `RankTable` 结构（JSON 可序列化）：

```json
{
  "world_size": 384,
  "dim_degrees": {"pp": 1, "dp_replicate": 1, "dp_shard": 2, "cp": 1,
                   "tp": 1, "ep": 192, "etp": 1},
  "rank_coordinates": {"0": {"dp_shard": 0, "ep": 0}, "1": {...}, ...},
  "process_groups": {
    "ep": [{"members": [0,1,...,191]}, {"members": [192,...,383]}],
    "dp_shard": [{"members": [0, 192]}, {"members": [1, 193]}, ...]
  }
}
```

这与真实 Ascend 集群的 `ranktable.json`（rank_id → server/device 物理映射）
概念对应，只是这里映射的是"逻辑并行坐标"而非"物理机架位置"——因为模拟器不
关心物理拓扑，只关心并行语义上的通信域划分，这与 L2 spec 的
`DataPass.src_device/dst_device` 字段（也是逻辑 rank 而非物理坐标）一致。
`RankTable` 本身作为 L2 `ScheduleGraph.annotations["rank_table"]` 挂载，同时
单独导出一份 `rank_table.json` 方便独立查看。

### 5.7 迭代语义（L3，`schedule_builder.py`）

`WorkloadGraph.workload_type="train"`；`data_inputs` 从
`HuggingFaceTextDataLoader.Config`（`dataset="c4_test"`）声明的
`local_batch_size`/`seq_len` 推出 `DataFlow`（`source="dataloader"`，
`tensor_shape=(local_batch_size, seq_len)`，`is_streaming=True`，
`interleave_strategy="synced"`）；`cross_iter_passes` 记录 optimizer 输出的
参数张量到下一轮 forward 输入参数的 `DataPass`（对应 spec 表格"跨迭代:
本轮 opt exit(param) → 下轮 fwd entry(param)"）。`num_iterations` 固定为 1
（只捕获一步），`warmup_iterations=0`。

### 5.8 NPU 算子成本模型（`cost/op_cost_model.py`）

可扩展注册表模式（`op_type -> handler`），覆盖：

- 通用：`matmul/addmm/bmm`（含 MoE 分组矩阵乘 `npu_gmm` 按每专家 token 数
  分别计入 FLOPs）、`layer_norm/rms_norm`、`gelu/silu/swiglu/softmax`。
- Attention 族：`sdpa/flash_attention_fwd`（含 `npu_sparse_flash_attention`/
  `npu_lightning_indexer`/`npu_sparse_attn_sharedkv`，按 `O(B·H·Sq·Sk·d)`
  或其稀疏变体的有效 token 数计算）。
- MoE 族：`npu_moe_token_permute/unpermute`（数据搬移量，非 FLOPs）、
  `npu_moe_re_routing`。
- 通信族：`allreduce/allgather/reduce_scatter/all_to_all`，
  `comm_bytes = numel * dtype_size`（乘 2 表示 allreduce 的
  reduce+broadcast 语义，与 spec 参考实现一致）。
- 未识别算子：返回全 0 估算并在 `annotations["cost_unknown"]=True`
  标记，**不抛异常**——保证新增/未覆盖算子不会打断整条捕获流水线，这是
  直接吸取姊妹项目 `simulator_defect_fix_plan.md` 里"MockCostModel 覆盖不足
  导致 FLOPs 评估为 0 且难以发现"的教训：我们让"未覆盖"显式可见
  （annotation + 汇总报告里单列"未识别算子 top list"），而不是静默归零。

### 5.9 可视化（`viz/`）

- `json_export.py`：导出完整四层 IR（`WorkloadGraph` 递归展开）为结构化
  JSON，供程序化消费/后续对接 cost/scheduler 等下游工具。
- `html_export.py`：自包含单文件 HTML（无外部 CDN 依赖，离线可看，参考
  姊妹项目 `trace.html` 的成熟模式，但内容按四层 IR 组织而非其自有 node
  model）：
  1. **L3 卡片**：workload 类型、迭代信息、dataloader 数据流。
  2. **L2 RankTable 网格 + 调度泳道**：384×维度坐标表（可按维度筛选/搜索
     rank），以及按 rank 分组的 StepInstance 时间线（forward/backward/
     optimizer 分段色块）+ DataPass 连线（跨 rank 通信）。
  3. **L1 Step 汇总**：forward/backward/optimizer 三张卡片（op 数、
     total_flops、peak_active_mem、comm_volume）。
  4. **L0 算子 DAG**：canvas 渲染的依赖图，支持按 `repeat_count`
     折叠/展开（默认折叠，点击展开某一层的完整子图），避免 61 层全展开
     导致浏览器卡死。
  - 同时导出 `compute_graph.dot`（Graphviz）与 `summary.txt`（纯文本
    摘要：op 计数、通信统计、FLOPs/内存/通信量汇总、"未识别算子"列表），
    与本仓 `docs/test_guides` 里已有的其他工具输出习惯保持一致。

## 6. 端到端数据流（一次 TrainStep 捕获）

```
1. entry.main() 解析 --module torchtitan_npu.simulator
                      --config deepseek_v4_pro_simulate_61_layers
2. config_registry 返回的 SimulationTrainer.Config：
   - model_spec = 原样复用 deepseek_v4 的 model_registry("v4_pro_debug_61_layers")
   - comm.mode 强制 "fake_backend"；debug.moe_force_load_balance 强制 True；
     compile.enable 强制 False
3. SimulationTrainer.__init__（继承 Trainer.__init__ 主流程）：
   - init_distributed() -> fake PG, world_size=384
   - meta_env: 模型在 torch.device("meta") 下构建，parallelize_fn/
     pipelining_fn 原样执行（FSDP2/TP/PP wrapper 都认为自己在正常初始化）
   - 跳过 to_empty()/init_weights()，模型保持在 meta
   - optimizer/lr_scheduler 正常构建（AdamW/swap_optimizer 等，均不依赖真实数值）
4. rank_table.py：build_mesh() 完成后，展开 384-rank 的通信域坐标
5. SimulationTrainer.train()（覆盖，只跑 1 步）：
   - 进入 fake_collectives 拦截上下文 + dispatch_capture(TorchDispatchMode)
   - 进入 step_boundary 边界钩子
   - 调用与真实训练相同的 forward_backward_step() + optimizer.step()
     （one micro-batch，数据来自 dataloader 或 synthetic token loader）
   - 所有 aten/npu 算子被 L0 捕获；所有集合通信被 fake_collectives 记录
6. 捕获退出：
   - step_boundary 产出的 op 缓冲区 -> L1 StepGraph（forward/backward/optimizer）
   - comm_events + rank_table -> L2 ScheduleGraph（StepInstance × 384 + DataPass）
   - dataloader 配置 + L2 -> L3 WorkloadGraph
7. viz/ 导出 json + html + dot + summary.txt 到 simulation.output_dir
```

## 7. 集成方式（用户怎么用）

复用现有 `scripts/run_train.sh` 约定（不新增脚本）：

```bash
NGPU=384 LOCAL_RANK=0 python3 -m torchtitan_npu.entry \
    --module torchtitan_npu.simulator \
    --config deepseek_v4_pro_simulate_61_layers \
    --comm.mode=fake_backend --training.steps=1
```

或直接：

```bash
MODULE=torchtitan_npu.simulator \
CONFIG=deepseek_v4_pro_simulate_61_layers \
COMM_MODE=fake_backend \
./scripts/run_train.sh
```

`deepseek_v4_pro_simulate_61_layers()`（新增于
`torchtitan_npu/simulator/config_registry.py`）内部直接调用并复制
`deepseek_v4_pro_debug_61_layers_4k_384die()` 的 `model_spec`，只替换/强制
`comm`/`debug`/`compile`/新增 `simulation` 子配置字段，其余训练/并行度配置
原样透传，保证"验收用例来自本仓真实配置"。

## 8. 测试与验证策略

分层验证，尽量让大部分测试不依赖真实 `torch_npu`：

1. **IR 单元测试**（`tests/unit_tests/simulator/ir/`，纯 Python，无 torch
   依赖）：`OpNode`/`StepGraph` DAG 校验、`ScheduleGraph`
   模板一致性、`WorkloadGraph` 构造。
2. **捕获机制单元测试**（`tests/unit_tests/simulator/capture/`，只需要
   `torch`，不需要 `torch_npu`）：用普通 `nn.Linear`/`nn.LayerNorm` 在
   `meta` device 上验证 `TorchDispatchMode` 捕获、`repeat_count` 去重、
   step 边界识别、`fake_collectives` 对 `dist.all_reduce`/`all_gather`
   的 shape 正确性（用小 world_size，比如 8，配合 `dist.init_process_group
   ("fake", ...)`，这一步本沙盒环境即可跑，已在小规模 spike 中验证可行）。
3. **NPU 算子集成测试**（`tests/smoke_tests/simulator/`，需要真实
   `torch_npu`，用 `npu_available`-同款的 `importorskip`/skip 装饰器跳过）：
   在带 CANN 的容器（见 §2，`titan-npu-sim-validate` 或团队正式 CI 镜像）内，
   跑一个 2 层/4 专家的极小 DeepSeek-V4 配置，world_size=8，验证 L0-L3
   全部产出且 `debug_force_load_balance` 路由正确、EP all-to-all
   转换出的 `DataPass` 数量/字节数与理论值吻合。
4. **验收测试**：同样在带 CANN 容器内，跑
   `deepseek_v4_pro_simulate_61_layers`（对应 61 层/384 专家/384 die），
   验证：
   - 捕获无异常退出；
   - `RankTable.world_size == 384`，各维度 degree 与配置
     （`expert_parallel_degree=192` 等）一致；
   - L0 图节点数在去重后为可控数量级（不因 61 层线性膨胀到不可视化的规模）；
   - `summary.txt`/`trace.html`/`simulation_result.json` 均正确生成；
   - 通信统计中 EP all-to-all 与 FSDP all-gather/reduce-scatter 均有非零
     `comm_bytes`。

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 部分未采样到的 NPU 自定义算子（本设计只验证了 4 个）在 meta 下可能报错 | Cost model/capture 层遇到未知失败时记录到 `annotations["capture_error"]`，跳过该子图节点而非整体崩溃（可配置严格模式用于调试）；实现阶段逐层跑通 61 层配置时会自然暴露，逐一补齐 |
| activation_checkpoint(mode="full") 下反传会重放 forward 算子 | 按"当前 step 阶段=backward 时产生的 op 归入 backward StepGraph"处理，允许 backward 图中出现"表面上是 forward 算子"的重算节点，符合真实语义 |
| 验证用的 `torch_npu==2.10.0` 与仓库锁定的 `2.12.0rc1` 不完全一致 | 设计与实现不依赖具体 patch 版本行为；正式验收前在团队 CI/CANN 环境用锁定版本复验一次 |
| 384 个 `StepInstance` 若逐一深拷贝 OpNode 会内存爆炸 | `StepInstance` 只存 `step_ref`（模板引用）+ 坐标信息，不复制 OpNode；只有 rank 0 的模板做完整 L0/L1 展开 |
| 集合通信 Python 层拦截可能遗漏某个内部调用路径（如 DTensor 内部直接调用 ProcessGroup 方法而不经过 public API） | 实现阶段用一次真实的 FSDP2+TP+EP 组合配置跑一遍，检查是否有未被拦截、直接触达 c10d 算子的调用；如发现遗漏，按需增加对应 Python 层拦截点（保持"Python 层拦截"这一统一模式，不引入 ATen 级 Meta 核 hack，因为 §2 验证 #4 已经证明后者容易撞到内部 Work 对象转换等更深的坑） |
| `pipeline_parallel_degree > 1` 时单一模板不足以代表所有 stage | 验收目标 `pp=1`，不受影响；设计已在 §5.5 明确"模板粒度=每 pipeline stage 一份"，多 stage 场景列为后续扩展（需要以不同 pp 坐标多次调用 `pipelining_fn` 切分逻辑分别捕获），不在本次实现范围内 |
| `set_determinism()` 在 `world_size>1` 且未设 seed 时对 meta 张量调用 `.to("cpu").item()` 崩溃（已用 pinned 源码复现） | `SimulationTrainer` 强制 `config.debug.seed` 为固定整数，规避该分支 |
| `Trainer.train_step()` 的 `dist_sum`（token 计数）与日志代码对 meta 张量调用 `.item()` 崩溃（已用 pinned 源码确认调用点） | `SimulationTrainer` 不调用 `train_step()`，直接调用 `forward_backward_step()` 并手动提供 `global_valid_tokens`（纯 Python float，由配置静态算出），跳过 `clip_grad_norm_`/损失日志 |

## 10. 与用户需求的对应关系

| 用户需求 | 设计对应 |
|---------|---------|
| 侧载包形式 | `torchtitan_npu/simulator/` 纯新增子包，零修改现有文件 |
| 不需要真实硬件/真实分配内存 | meta device 全程 + fake ProcessGroup + 集合通信拦截 |
| 捕获所有卡的完整计算图 | 单进程 fake world_size=384 + RankTable 展开 + StepInstance × 384（模板化，非重复展开） |
| 通信域/通信量正确 | RankTable（§5.6）+ DataPass 从真实拦截到的集合通信调用（含 group/shape）转换而来，非凭空构造 |
| MoE 强制负载均衡 | 复用并强制 `debug.moe_force_load_balance`，SimulationTrainer 兜底强制打开 |
| 四层 IR 可观测 + 可视化 | `ir/` 四层 dataclass（按 spec 原创）+ `viz/` HTML/JSON/DOT 导出 |
| 验收：61 层 DSV4-pro 跑通 | `deepseek_v4_pro_simulate_61_layers` 配置直接复用验收配置的 model_spec/并行度 |
