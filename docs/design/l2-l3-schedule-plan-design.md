# L2/L3 调度 IR 重构设计：从扁平 trace 到结构化 SchedulePlan

> 分支：feat/pipeline-multi-graph
> 日期：2026-07-14
> 涉及文件：`ir/schedule_graph.py`、`ir/workload_graph.py`、`capture/schedule_builder.py`、`capture/comm_events.py`、`trainer.py`、viz 导出

## 1. 背景

L1/L0 已能按 `(pp_stage, comp_type)` 完整还原每种计算图（见 `pipeline-multi-graph-capture-design.md`）。但 L2/L3 还不能清晰表达"整个负载如何调度这些 L1 图"。用户已能通过 `pp_schedule.pipeline_order` 看到 F/W/I/B 在各 stage×microbatch 上的分布，但看不到：

- **(G1) F/W/I/B 在各 stage 之间传递的数据**：哪个 stage 的 forward 出口激活发给哪个 stage 的 forward 入口、什么 shape/dtype/bytes；
- **(G2) FSDP unshard/reshard 的位置**：相对 F/W/I/B 它们出现在哪里、产出/消费什么张量；
- **(G3) 优化器步进的位置**：相对 backward 它在哪里、消费什么梯度、产出什么参数。

并且训练/推理等不同场景下 L1 调度的结构性差异（迭代相位、跨迭代数据流、optimizer/lr cadence）在 L3 里没有一等公民表达。

## 2. 现状与根因

### 2.1 torch 侧已有一等调度数据（用户尚未利用）

调研 `torch.distributed.pipelining.schedules`：

| 属性 | 内容 | 是否含通信/FSDP |
|------|------|----------------|
| `pp_schedule.pipeline_order` | `dict[rank, list[_Action]]`，**仅计算动作**（F/B/I/W/OVERLAP_F_B） | 否（用户当前看的就是这个） |
| `pp_schedule.pipeline_order_with_comms` | `dict[rank, list[_Action]]`，**lowered 完整计划** | 是 |

`_Action = (stage_index, computation_type, microbatch_index, sub_actions)`，`_ComputationType` 取值：

```
F(FORWARD)  B(FULL_BACKWARD=I+W)  I(BACKWARD_INPUT)  W(BACKWARD_WEIGHT)
UNSHARD     RESHARD               REDUCE_GRAD
SEND_F/RECV_F   SEND_B/RECV_B
```

lowering 流水线（`_prepare_schedule_with_comms`）：
1. `_add_unshard_reshard`：在计算动作前后按 prefetch 策略插入 `UNSHARD`(产出 full param)/`RESHARD`(释放)；
2. `_add_reduce_grad`：在某 stage 的最后一个 backward 后插入 `REDUCE_GRAD`（FSDP post_backward + DP 梯度归约）；
3. `_add_send_recv`：在跨 rank 的 F/B 前后插入 `SEND_F`/`RECV_F`/`SEND_B`/`RECV_B`，并保证不死锁的顺序。

P2P 传递的**数据**可从 stage 的 `act_send_info[idx]`（forward 第 idx 输出 → 目标 stage 列表）/`grad_send_info`（input grad → 目标 stage）+ `get_fwd_send_ops`/`get_bwd_send_ops` 得到（`isend` 的 tensor shape 即被传张量）。这些都在捕获时被 `comm_events.py` 记成 `CommEvent.tensor_shape/dtype/volume_bytes + p2p_peer_rank/p2p_mb_idx/p2p_stage`，但**没有被结构化成"哪个实例的出口 → 哪个实例的入口"的依赖边**。

**optimizer/lr_scheduler 不在 `pipeline_order_with_comms` 里**：它们在 `run_simulation_step` 中 `pp_schedule.step()` 之后单独调用（`trainer.py:177-178`）。所以 G3 的答案是：当前计划根本不含优化器——需要捕获侧把它作为计划尾部的独立动作补进来。

### 2.2 当前 L2/L3 IR 的结构缺陷

`ir/schedule_graph.py` / `ir/workload_graph.py` 现状：

| 字段 | 问题 |
|------|------|
| `ScheduleGraph.instances: list[StepInstance]` | 扁平列表、每模板仅 mb=0 一个实例；**不表达顺序、不表达 per-mb 重复**，看不出调度 |
| `ScheduleGraph.execution_timeline: list[TimelineEntry]` | 捕获**trace**（按 seq_idx 排序的事件流），混了 compute/comm/scheduling；不是"计划"；MB 1+ 只剩 comm+timeline 片段 |
| `ScheduleGraph.data_passes: list[DataPass]` | 从 comm 事件来，`src_instance`/`dst_instance` 用 `"rank{stage}"` 字符串，**不带 mb_idx**，连不到具体 (stage, mb, comp_type) 实例 |
| 无 "ScheduleAction" 概念 | UNSHARD/RESHARD/SEND/RECV/REDUCE_GRAD/OPTIMIZER 的**位置**埋在扁平 timeline 里，没有结构化的一等对象 |
| 无 "plan vs trace" 区分 | torch 已有的 `pipeline_order_with_comms`（计划）被完全忽略；只有事后 trace |
| `IterationSpec` | 仅 `schedule + microbatch_count + time_est`；不表达相位（warmup/steady/cooldown）、cadence、多场景（train/inference/eval） |
| `cross_iter_passes` | 字段在但为空；参数/优化器状态的跨迭代数据流未建模 |

根因：L2 把"计划"和"trace"混为一谈，且没有把"动作（action）+ 动作间数据依赖"作为核心抽象。

## 3. 设计

### 3.0 核心抽象：ScheduleAction + DataSlot + 依赖边

把 L2 从"扁平 trace"重构为"**有序动作计划 + 数据流图**"：

- **ScheduleAction** = 调度的最小单位（一次 forward_one_chunk / 一次 unshard / 一次 p2p send / 一次 optimizer step …）。每个 action 有 `action_type`、所属 `(rank, stage, mb_idx)`、`seq_idx`（执行序）、`consumes`/`produces`（DataSlot 引用）。
- **DataSlot** = 在动作间流动的一个张量（激活/输入梯度/全参/分片参/归约梯度/优化器状态…），带 shape/dtype/bytes，且记录 `producer_action_id` 与 `consumer_action_ids`。
- **依赖边**隐含在 DataSlot 的 producer/consumer 里：`A.produces ∋ slot` 且 `B.consumes ∋ slot` ⇒ A→B 数据依赖。可据此做拓扑排序、关键路径、气泡分析。

这样 G1/G2/G3 直接落位：

| 用户诉求 | 落到哪个结构 |
|---------|-------------|
| G1 F/W/I/B 跨 stage 传递的数据 | `SEND_F`/`RECV_F` action 的 DataSlot（激活）；`SEND_B`/`RECV_B` action 的 DataSlot（输入梯度）；producer=前 stage 的 F/B 出口、consumer=后 stage 的 F/B 入口 |
| G2 FSDP unshard/reshard 位置 | `UNSHARD`/`RESHARD` action（计划里一等对象，位置由 `_add_unshard_reshard` 决定）；UNSHARD 产出 `param_full` slot → 下一个 F/B 入口消费；backward 出口 → RESHARD 消费 |
| G3 优化器步进位置 | `OPTIMIZER` action（计划尾部，所有 REDUCE_GRAD 之后）；消费 `grad_reduced` slots、产出更新后的 `param_shard` slots（下一迭代 UNSHARD 消费） |

### 3.1 L2 数据结构

```python
@dataclass
class DataSlot:
    slot_id: str                      # "slot_123"
    kind: str                         # activation | grad_input | param_shard | param_full |
                                      # grad_reduced | optimizer_state | kv_cache | dataloader_input
    shape: tuple[int | str, ...]
    dtype: str
    volume_bytes: int
    producer_action_id: str          # 产出此 slot 的 action（""=外部，如 dataloader/上一迭代）
    consumer_action_ids: list[str]
    src_stage: int = -1               # P2P/集合通信源 stage
    dst_stage: int = -1
    mb_idx: int = -1
    comm_primitive: str = ""         # p2p_send | allgather | reduce_scatter | allreduce | "" (本地)
    src_exit_op: int = 0             # L1 内具体出口算子 op_id（0=未知）
    dst_entry_op: int = 0            # L1 内具体入口算子 op_id

@dataclass
class ScheduleAction:
    action_id: str                   # "r0_a12"
    rank: int
    stage: int                       # PP stage（-1 非 PP）
    mb_idx: int                      # microbatch（-1 非 mb 级，如 OPTIMIZER）
    action_type: str                 # COMPUTE | UNSHARD | RESHARD | SEND_F | RECV_F
                                      # | SEND_B | RECV_B | REDUCE_GRAD | OPTIMIZER
                                      # | LR_SCHEDULER | BARRIER | LOSS
    comp_type: str = ""              # COMPUTE 时: F/B/I/W/F_RECOMPUTE
    template_ref: str = ""           # COMPUTE 时: L1 模板 id（s{stage}_{comp_type}）
    seq_idx: int = 0                 # 执行序（捕获给出）
    consumes: list[str] = ...        # DataSlot id 列表
    produces: list[str] = ...        # DataSlot id 列表
    duration_est: float = 0.0
    annotations: dict = ...          # fsdp_state / comm_group_ranks / raw 等

@dataclass
class SchedulePlan:                  # L2 主体（取代 ScheduleGraph 的主体职责）
    plan_id: str
    workload_type: str               # train | inference | eval
    step_templates: dict[str, StepGraph]   # L1 模板（COMPUTE action 引用）
    actions: list[ScheduleAction]     # 有序（按 seq_idx，或按 plan 顺序）
    action_map: dict[str, ScheduleAction]
    data_slots: dict[str, DataSlot]
    # 并行维度（从 RankTable 来，供展开/可视化）
    pp_degree: int = 1
    tp_degree: int = 1
    dp_degree: int = 1
    num_micro_batches: int = 1
    pipeline_schedule: str = "none"
    gradient_accumulation: int = 1
    annotations: dict = ...          # rank_table 等
```

**向后兼容视图**：旧的 `ScheduleGraph.instances/data_passes/execution_timeline` 改为 `SchedulePlan` 的**派生视图**（property/方法），不丢老消费者：
- `instances` ← `actions` 中 `action_type=="COMPUTE"` 的 action（每条→一个 StepInstance，带 mb_idx）；
- `data_passes` ← `data_slots` 中 `comm_primitive != ""` 的 slot（每条→一个 DataPass，src/dst 用 producer/consumer action 的 rank+stage+mb）；
- `execution_timeline` ← `actions` 按 `seq_idx` 排序映射成 `TimelineEntry`。

### 3.2 L3 数据结构

```python
@dataclass
class IterationPhase:
    name: str                        # warmup | steady | cooldown | single
    mb_range: tuple[int, int]        # 该相位覆盖的 microbatch 区间
    action_id_range: tuple[str, str] # 计划中该相位对应的 action 区间（起止 action_id）

@dataclass
class IterationSpec:
    schedule_plan: SchedulePlan
    microbatch_count: int
    phases: list[IterationPhase]      # 1F1B: [warmup, steady, cooldown]；GPipe: [single]
    iteration_time_est: float = 0.0
    # cadence：相对"一次迭代"的步进频率（grad accumulation 等）
    optimizer_cadence: int = 1        # 每 N 次迭代执行一次 optimizer（>1=梯度累积）
    lr_scheduler_cadence: int = 1
    grad_accumulation: int = 1

@dataclass
class WorkloadGraph:                  # L3
    workload_id: str
    workload_type: str                # train | inference | eval | rag
    step_templates: dict[str, StepGraph]   # L1 模板（跨迭代共享）
    iteration: IterationSpec          # 主迭代（train）
    eval_iteration: IterationSpec | None = None   # 评估迭代（可选，eval 场景）
    num_iterations: int
    warmup_iterations: int = 0
    data_inputs: list[DataFlow]       # dataloader → 首 stage F 入口（首迭）
    data_outputs: list[DataFlow]      # 末 stage B 出口 / loss
    cross_iter_passes: list[DataPass] # 跨迭代：OPTIMIZER 产出的 param_shard → 下一迭代 UNSHARD；
                                      # 推理场景：KV-cache → 下一迭代 attention 入口
    total_runtime_est: float = 0.0
```

### 3.3 三场景在结构上的体现

| 场景 | actions 出现的 type | phases | cross_iter_passes |
|------|---------------------|--------|-------------------|
| **train** | F, (B 或 I,W), UNSHARD, RESHARD, SEND/RECV, REDUCE_GRAD, OPTIMIZER, LR_SCHEDULER | warmup→steady→cooldown | param_shard(OPTIMIZER 出)→下一 iter UNSHARD 入 |
| **eval** | F, LOSS（无 B/I/W/OPTIMIZER） | single | 无 |
| **inference (autoregressive)** | F（prefill 多 mb + decode 逐 token） | prefill→decode | kv_cache(本 iter F 出)→下一 iter F 入；dataloader_input 每 iter 推进 |

推理场景的 KV-cache 跨迭代数据流正是 `cross_iter_passes` 的用武之地——当前 L3 该字段为空，正是结构缺失点。

## 4. 数据来源：plan × capture 融合

`SchedulePlan` 的填充来自**两个互补源**的融合：

### 4.1 plan 源（结构）：`pp_schedule.pipeline_order_with_comms`

对 `_PipelineScheduleRuntime` 子类（ZB/DualPipe/Interleaved/LoopedBFS）调度，`pipeline_order_with_comms[rank]: list[_Action]` **就是**计划骨架。逐 `_Action` 映射成 `ScheduleAction`：

| `_Action.computation_type` | → `ScheduleAction.action_type` | 额外 |
|----------------------------|-------------------------------|------|
| FORWARD | COMPUTE, comp_type=F | template_ref=`s{stage}_F` |
| FULL_BACKWARD | COMPUTE, comp_type=B | template_ref=`s{stage}_B` |
| BACKWARD_INPUT | COMPUTE, comp_type=I | template_ref=`s{stage}_I` |
| BACKWARD_WEIGHT | COMPUTE, comp_type=W | template_ref=`s{stage}_W` |
| UNSHARD | UNSHARD | 产出 `param_full` slot |
| RESHARD | RESHARD | 消费 backward 出口 → 产出 `param_shard` |
| SEND_F / RECV_F | SEND_F / RECV_F | 产出/消费 `activation` slot |
| SEND_B / RECV_B | SEND_B / RECV_B | 产出/消费 `grad_input` slot |
| REDUCE_GRAD | REDUCE_GRAD | 消费各 mb 累积 grad → 产出 `grad_reduced` |

计划的**顺序**即 `pipeline_order_with_comms[rank]` 的列表序（已是死锁安全的拓扑序）。

### 4.2 capture 源（enrichment）：trace + comm 事件

plan 只给"要跑什么、什么序"，**不给**实际 seq_idx、L1 模板里具体 op_id 的连接、张量 shape。这些来自捕获：

- **seq_idx**：每个 COMPUTE action 的 seq_idx ← `record_timeline_event` 记录的 `forward_one_chunk`/`backward_one_chunk`/`backward_weight_one_chunk`（已有 comp_type/stage/mb）；
- **DataSlot shape/dtype/bytes**：← `CommEvent.tensor_shape/dtype/volume_bytes`（P2P/集合通信捕获已有）；
- **src_exit_op / dst_entry_op**：← `dispatch_capture._pending_comm_links` 的 producer/consumer 反查（已实现，需在 plan 侧把 CommEvent 的 op_id 与对应 COMPUTE action 的 template 节点对齐）；
- **template_ref 校验**：plan 里 `(stage, comp_type)` ↔ capture 里 `s{stage}_{comp_type}` 模板，一致即把 captured OpNode 集挂到该 action。

### 4.3 plan 之外的 action（optimizer/lr_scheduler/loss）

这些不在 `pipeline_order_with_comms`：在 `run_simulation_step` 的 `optimizer_step()`/`lr_scheduler_step()` 处补 `OPTIMIZER`/`LR_SCHEDULER` action（`trainer.py` 已 `boundary.mark("optimizer")` + `capture._capture_l0=True`，可直接从该区间捕获的 `comp_type=OPTIMIZER` 节点构造 action，seq_idx 接在计划尾部）。`LOSS` action 对应末 stage 的 loss 计算（F 内或独立）。

### 4.4 单 stage 调度（1F1B/GPipe/`PipelineScheduleSingle`）的退化

`PipelineScheduleSingle` 不走 `_perform_action`、无 `pipeline_order_with_comms`。对此**从捕获的 timeline 合成 plan**：timeline 已含每个 `forward_one_chunk`/`backward_one_chunk`（带 stage/mb/comp_type），按 seq_idx 排序即得 COMPUTE actions；UNSHARD/RESHARD 从 `comm_events` 里 `comm_layer=="L2"` 的 FSDP allgather/reduce_scatter 事件插入；SEND/RECV 从 P2P 事件插入。即"trace 即 plan"的退化形态——结构一致性靠同一套 `ScheduleAction` 表达。

## 5. 三处缺口的端到端落位（示意）

以 PP=2、ZBV、2 microbatch 为例，rank 0 的计划片段：

```
seq  action          stage mb  consumes              produces
1    UNSHARD          0   -                         param_full(S0)
2    COMPUTE F        0   0   param_full(S0)         activation(S0→S1,mb0)
3    SEND_F           0   0   activation(mb0)        ← DataSlot: shape=[B,S,H], dtype=bf16, bytes=…
4    UNSHARD          1   -                         param_full(S1)
5    RECV_F           1   0                          activation(mb0)   # 消费 SEND_F 产出
6    COMPUTE F        1   0   param_full(S1)+activation(mb0)  activation(S1→loss,mb0) | loss
7    COMPUTE I        1   0   …                      grad_input(S1→S0,mb0)
8    SEND_B           1   0   grad_input(mb0)        ← DataSlot: shape=[B,S,H], bytes=…
9    COMPUTE W        1   0   …                      grad(S1)
10   RECV_B           0   0                          grad_input(mb0)
11   COMPUTE I        0   0   …                      grad_input(S0→-,mb0)  # 首 stage 不发
12   COMPUTE W        0   0   …                      grad(S0)
…
17   REDUCE_GRAD      1   -   grad(S1,mb0..1)        grad_reduced(S1)
18   REDUCE_GRAD      0   -   grad(S0,mb0..1)        grad_reduced(S0)
19   RESHARD          0   -   param_full(S0)         param_shard(S0)
20   RESHARD          1   -   param_full(S1)         param_shard(S1)
21   OPTIMIZER        -   -   grad_reduced(S0,1)+param_shard(S0,1)  param_shard'(S0,1)  # 跨迭代
22   LR_SCHEDULER     -   -   …                      …
```

- **G1**：第 3 行 `SEND_F` 的 DataSlot = S0 forward 出口激活，shape/bytes 在 slot 上；producer=action#2 的 template exit_op、consumer=action#5（RECV_F→S1 F 入口）。`SEND_B` 同理连 S1 的 I 出口 → S0 的 B 入口。
- **G2**：第 1/4 行 `UNSHARD` 产出 `param_full`，被紧随的 F（#2/#6）`consumes`；第 19/20 行 `RESHARD` `consumes` backward 产出的 param_full、`produces param_shard`。位置（F 前/REDUCE_GRAD 后）由 `_add_unshard_reshard` 决定，一等可见。
- **G3**：第 21 行 `OPTIMIZER` 在所有 `REDUCE_GRAD`（#17/#18）之后，`consumes grad_reduced + param_shard`、`produces param_shard'`，后者进 `cross_iter_passes` 连到下一迭代的 UNSHARD。

## 6. 迁移与向后兼容

1. **新增** `ir/schedule_plan.py`（`DataSlot`/`ScheduleAction`/`SchedulePlan`），`ir/workload_graph.py` 扩 `IterationPhase`/`IterationSpec`/多场景。
2. **`schedule_builder.py`** 增加 `build_schedule_plan(...)`：优先读 `pp_schedule.pipeline_order_with_comms`（runtime 调度），退化则从 timeline+comm_events 合成；再与 capture 融合填 seq_idx/DataSlot/producer-consumer。
3. **`ScheduleGraph` 保留**为 `SchedulePlan` 的兼容视图（property 派生 instances/data_passes/execution_timeline），老 viz/CSV 不破。
4. **viz** 新增 `schedule_plan.csv`（一行一 action：seq, action_type, stage, mb, comp_type, template_ref, consumes→slot_ids, produces→slot_ids, slot shape/bytes, producer_action, consumer_actions），与现有 `l1_schedule.csv`（per-stage 时间线）互补：前者是结构化计划+数据流，后者是 per-stage 顺序。

## 7. 验证方案

1. **结构校验**：PP=2 ZBV 配置跑一次，断言 `SchedulePlan.actions` 的 action_type 序列与 `pipeline_order_with_comms[0]` 逐 `_Action` 一致；每个 `SEND_F` 的 DataSlot producer action = 同 mb 的前 stage F、consumer = 后 stage RECV_F→F。
2. **G1 验证**：`schedule_plan.csv` 里每对 SEND_F/RECV_F 的 slot shape/bytes 相同且与 `CommEvent.tensor_shape` 一致；producer 的 template exit_op 与 consumer 的 template entry_op 在各自 L1 图里存在。
3. **G2 验证**：每个 `UNSHARD` action 后续首个 `COMPUTE` action 的 `consumes` 含该 UNSHARD 产出的 `param_full` slot；每个 `RESHARD` 的 `consumes` 来自前序 backward。
4. **G3 验证**：`OPTIMIZER` action `seq_idx` > 所有 `REDUCE_GRAD`；其 `consumes` 覆盖所有 stage 的 `grad_reduced`；`cross_iter_passes` 含其产出的 `param_shard'`。
5. **场景校验**：构造 inference-only（forward-only，无 backward/optimizer）配置，断言 actions 无 B/I/W/REDUCE_GRAD/OPTIMIZER，`cross_iter_passes` 含 kv_cache（如启用）。

## 8. 与既有 IR 的边界

- **L0 OpNode**：不变（单算子）。
- **L1 StepGraph**：不变（`(stage, comp_type)` 模板图，`ScheduleAction.template_ref` 引用它）。
- **L2**：主体从"扁平 trace"升级为"SchedulePlan（有序 action + DataSlot 依赖图）"；`ScheduleGraph` 退为兼容视图。
- **L3**：`WorkloadGraph` 增迭代相位/cadence/多场景/cross-iter 数据流，覆盖 train/eval/inference。

本设计与 `pipeline-multi-graph-capture-design.md`（L0/L1）正交：L1 模板已就绪，L2 只需把模板按 `pipeline_order_with_comms` 实例化成有序 action 流、并用捕获的 comm/producer 数据补全 DataSlot 依赖。

## 9. DualPipeV 兼容性验证与方案完善

用 `--parallelism.pipeline_parallel_schedule DualPipeV`（PP=2、4 virtual stages、4 microbatch）实跑验证（容器 `titan-npu-sim-e2e`）。结论：**L0/L1 捕获完全兼容**（正确产出 `s0..3_{F,B,I,W}` 模板，I/W 拆分到位，V 型 stage 映射 rank0={0,3}/rank1={1,2} 各自捕获）；但**原 L2/L3 方案暴露三处不足**，需完善。

### 9.1 实测证据（`pipeline_order_with_comms`）

rank 0 计划前 25 个 action（`scripts/inspect_dualpipe_plan.py` 输出）：

```
0UNSHARD, 3UNSHARD, 0F0, 0SEND_F0, 0F1, 0SEND_F1, 3RECV_F0, 0F2, 0SEND_F2,
3F0, 3RECV_F1, 3I0, 3SEND_B0, 3W0, 3F1, 3RECV_F2, 0RECV_B0,
(0F3;3B1)OVERLAP_F_B, 0SEND_F3, 3SEND_B1, (3F2;0B0)OVERLAP_F_B, 3RECV_F3, ...
```

rank 0 action 计数：`UNSHARD×2, F×5, SEND_F×4, RECV_F×4, I×3, SEND_B×4, W×3, RECV_B×4, OVERLAP_F_B×3, B×2, REDUCE_GRAD×2, RESHARD×2`。

### 9.2 不足一：`OVERLAP_F_B` 复合动作

`(0F3;3B1)OVERLAP_F_B` 表示 stage0 的 F(mb3) 与 stage3 的 B(mb1) **重叠执行**。`_Action` 有 `sub_actions: tuple[_Action,...]`，`_step_microbatches` 对 `OVERLAP_F_B` 做 `for sub_a in action.sub_actions: _perform_action(sub_a)`——即**计划层是复合、执行/捕获层展开为两个独立 chunk**（各自一条 timeline event + 一个 (stage,comp_type) 模板）。

**原方案 `ScheduleAction` 只有单 `comp_type`/`template_ref`，无法表达此复合关系。** 完善：

```python
@dataclass
class ScheduleAction:
    ...
    action_type: str                 # 增 "OVERLAP_F_B"
    comp_type: str = ""
    template_ref: str = ""
    # OVERLAP_F_B 时非空：承载 F 与 B/I/W 两个子动作；子动作各自有 template_ref
    # 与 seq_idx（捕获侧已展开），父动作只保留"重叠"语义（同 time_step）。
    sub_actions: list["ScheduleAction"] | None = None
```

构建规则：plan 里遇到 `_Action(computation_type=OVERLAP_F_B, sub_actions=[F_a, B_a])` → 生成一个 `action_type=OVERLAP_F_B` 的父 `ScheduleAction`，其 `sub_actions=[F_sub, B_sub]`；F_sub/B_sub 的 `template_ref`/`seq_idx`/DataSlot 由 capture enrich（与普通 F/B 一致）。依赖边连在子动作上（F_sub 产 activation、B_sub 消费 grad），父动作仅作分组。

### 9.3 不足二：V 型同 rank 相邻 stage 的本地传递（无通信事件）

V 型映射 `generate_rank_to_stage_mapping(N, 2N, style="v")`：rank0={0,3}、rank1={1,2}。`_perform_action` 的 `[Note: V-schedule special case]`：当 `is_next_stage_on_this_rank`（如 rank1 上 stage1→stage2）时用 `set_local_fwd_input`/`set_local_bwd_input` 直接在进程内递交张量，**不发 `dist.isend`/`irecv`**——所以 `pipeline_order_with_comms` **根本不含** stage1→stage2 的 `SEND_F`/`RECV_F`（实测 rank1 有 `2SEND_F0` 但无 `1SEND_F0`），`comm_events` 也无对应记录。

**原方案"DataSlot 只来自 SEND_F/RECV_F action + CommEvent"会漏掉这部分跨 stage 数据流。** 完善：

```python
@dataclass
class DataSlot:
    ...
    comm_primitive: str = ""   # p2p_send | allgather | reduce_scatter | allreduce | ""
    is_local_transfer: bool = False   # V 型同 rank 相邻 stage：进程内递交，无通信
```

构建规则：plan 装载后，对每个 `COMPUTE F` action（stage S，mb M），若 `stage_index_to_group_rank[S+1] == stage_index_to_group_rank[S]`（S+1 同 rank），则**合成**一条本地 `activation` DataSlot：producer=本 F action（exit_op）、consumer=stage S+1 的下一个 F action（entry_op）、`is_local_transfer=True`、`comm_primitive=""`、shape 取 stage S 的 F 模板 exit 节点 shape（或 `act_send_info`）。backward 同理（stage S 的 I/B 出口 → stage S-1 的 B 入口，若同 rank）。`_resolve_peer_global_rank`/`stage_index_to_group_rank` 提供同 rank 判定（plan 里已有）。

这样 G1 的"跨 stage 数据"在 V 型下完整：跨 rank 走 P2P DataSlot（有 CommEvent），同 rank 走 local DataSlot（合成），两者结构一致、仅 `is_local_transfer`/`comm_primitive` 不同。

### 9.4 不足三：rank↔stage 非线性映射

V 型下 `stage_index_to_group_rank` 非线性（rank0 持 stage0 与 stage3）。原方案 `ScheduleAction` 同时有 `rank`/`stage` 字段已足够，但**P2P peer 不再是 `stage±1` 的 rank**——需经 `stage_index_to_group_rank[stage±1]` 解析。`DataSlot.src_stage`/`dst_stage` 用 **stage index**（与拓扑一致），实际 rank 由 plan 的 `stage_index_to_group_rank` 派生挂到 `annotations`。`src_exit_op`/`dst_entry_op` 仍是 L1 模板内 op_id，不受映射影响。

### 9.5 附带发现（非方案缺陷，记录）

- **首 stage 的 I 退化**：stage0（首 stage）的 `backward_one_chunk(full_backward=False)` 因无前序 stage 可发 input grad，`backward_maybe_with_nosync("input")` 跳过，实测 `s0_I` 仅 4 个 profiler 节点。方案对此自然处理（I 模板节点少即合法）。
- **W 模板规模大**：`s3_W=36041` 节点（末 stage loss+全参 W），`s0_W=3395`。`stage_backward_weight` 对全参跑 `autograd.grad` 产出大图——属捕获事实，方案按 `(stage, W)` 分模板已正确隔离，无需特殊处理。
- **`_comp_type_to_function_map` 路径**：`_step_microbatches` 对 torch.compile 路径走 function map，eager 走 `_perform_action`。simulator 强制 `compile.enable=False`，故 OVERLAP_F_B/SEND/RECV 都经 `_perform_action`，与方案假设一致。

### 9.6 更新后的 L2 构建流程（含 DualPipeV）

`build_schedule_plan(plan, capture, comm_events, rank_table, schedule_obj)`：

1. **读 plan**：`schedule_obj.pipeline_order_with_comms[rank]` 逐 `_Action` → `ScheduleAction`；`OVERLAP_F_B` → 父 action + `sub_actions`；`UNSHARD/RESHARD/SEND/RECV/REDUCE_GRAD` 直接映射。
2. **合成本地 DataSlot**：用 `schedule_obj.stage_index_to_group_rank` 检测同 rank 相邻 stage，为 F/I/B 的本地递交补 `is_local_transfer=True` 的 DataSlot（无 CommEvent）。
3. **enrich 自 capture**：每个 COMPUTE action 的 `seq_idx` ← timeline event；`template_ref` ← `s{stage}_{comp_type}`；P2P/集合通信 DataSlot 的 shape/dtype/bytes ← CommEvent；`src_exit_op`/`dst_entry_op` ← `_pending_comm_links`。
4. **补 plan 外 action**：`OPTIMIZER`/`LR_SCHEDULER` 接在计划尾部（来自 `boundary.mark("optimizer")` 区间的 `comp_type=OPTIMIZER` 节点）。
5. **派生兼容视图**：`ScheduleGraph.instances/data_passes/execution_timeline` 从 `SchedulePlan` 派生。

### 9.7 DualPipeV 验证方案补充

在 §7 基础上增：

6. **OVERLAP_F_B 校验**：DualPipeV 跑一次，断言 plan 中每个 `OVERLAP_F_B` 父 action 的 `sub_actions` 恰有 2 个（一个 F、一个 B/I/W），且子 action 的 `template_ref` 与 capture 的 `s{stage}_{comp_type}` 一致、`seq_idx` 与 timeline event 一致。
7. **本地传递校验**：rank1 的 stage1→stage2 数据流断言存在 `is_local_transfer=True` 的 `activation`/`grad_input` DataSlot，其 producer/consumer 均在 rank1、`comm_primitive==""`；跨 rank 的 stage0→1 / stage2→3 DataSlot `is_local_transfer=False` 且 `comm_primitive=="p2p_send"`。
8. **V 映射校验**：`ScheduleAction.rank` 与 `stage_index_to_group_rank[action.stage]` 一致；rank0 的 actions 仅含 stage∈{0,3}，rank1 仅 stage∈{1,2}。

## 10. UNSHARD/RESHARD ↔ L0 算子链接（落地修复 + 回放入口）

实测发现 UNSHARD/RESHARD plan action 与其 L0 comm 算子原本断开（`template_ref=""`、DataSlot 空壳），根因有三，均已修复：

1. **`_comm_layer` 全局赋值 bug**：`_patch_comm_layer_context` 里嵌套的 `_patched_unshard`/`_patched_reshard`/`_patched_allgather_seq`/`_patched_step_microbatches` 写 `_comm_layer = "L2"` 时缺 `global _comm_layer` 声明（外层 `global` 不传导到嵌套作用域），实际是局部赋值——模块全局从未被设成 "L2"，FSDP allgather 被错记成 `comm_layer="L1"`。修复：每个嵌套函数加 `global _comm_layer`。
2. **UNSHARD/RESHARD 期间 stage 陈旧**：非计算 action（UNSHARD/RESHARD/SEND/RECV/REDUCE_GRAD）不经 `forward/backward_one_chunk`，故 `_pp_context["stage"]` 停在上一个 chunk 的值（或首 chunk 前的 -1），其 L0 comm 算子落进 `s-1_F` 杂物桶。修复：patch `schedules._get_profiler_function_name`（`_step_microbatches` 对每个 action 调 `record_function(_get_profiler_function_name(action))`，含非计算 action）在入口 stamp `action.stage_index` + 按 action 类型设 `comp_type`（UNSHARD/RESHARD/REDUCE_GRAD→各自名；SEND_F/RECV_F→"F"；SEND_B/RECV_B→"B"），使这些算子落进 `s{stage}_UNSHARD`/`s{stage}_RESHARD` 等专用模板。计算 action 的 comp_type 仍由 chunk patch 在 body 内覆盖。单 stage 调度（1F1B）不走此函数，其 UNSHARD 在 `forward_one_chunk` 内、stage 已对，不受影响。
3. **元数据推理阶段的伪 CommEvent**：DYNAMIC 模式 `_prepare_forward_infra`/`_compute_outputs` 跑的 unshard 会经 `_record_comm_with_l0` 记 CommEvent，但 L0 算子被 `_in_metadata_inference` 跳过，导致 CommEvent 的 `op_id` 指向上一条已记的 aten 算子（陈旧）。修复：`build_schedule_plan` 收集 allgather/reduce_scatter 时过滤掉 `op_id` 解析不到 `comm.*` 算子的 CommEvent。

`ScheduleAction` 增 `comm_op_id: int`（该 action 的 L0 comm 算子 id）与 `is_noop: bool`（FSDP mesh=1 等无实际 collective 时为 True）。

### 仿真回放入口

给定一个 UNSHARD/RESHARD（及 SEND/RECV）action，取其 L0 算子详细信息：

```python
plan = workload_graph.schedule_plan
op = plan.find_op_node(action.comm_op_id)      # 跨所有 L1 step_templates 查 OpNode
sg = plan.find_template_for_op(action.comm_op_id)  # 它所在模板
# op.annotations["raw_op_type"] == "comm.allgather" / "comm.reduce_scatter" / "comm.p2p_send" ...
# op.outputs[0].shape, op.comm_bytes, op.flops, op.annotations["module_path"] ...
```

- `action.comm_op_id == 0` 且 `is_noop == True`：FSDP 无实际 collective（mesh=1），无算子可回放。
- `action.comm_op_id != 0`：`find_op_node` 返回 `comm.allgather`/`comm.reduce_scatter` OpNode，含 shape/comm_bytes/flops。DataSlot（`action.produces`/`consumes` 引用）另带 `shape`/`dtype`/`volume_bytes`/`src_exit_op`/`dst_entry_op`，可串到 producer/consumer 的 L1 模板边界算子。

实测（容器内）：
- DualPipeV（dp_shard=1，FSDP mesh=1）：UNSHARD/RESHARD 全 `is_noop=True`（诚实，无回放数据）；FSDP 算子已从 `s-1_F` 迁到 `s{stage}_UNSHARD`/`RESHARD`/`REDUCE_GRAD` 专用模板，`s-1_F` 缩到 2 节点。
- 1F1B PP4（dp_shard=4，真实 FSDP）：每个 UNSHARD action 链到独立 `comm.allgather` OpNode，如 `UNSHARD s0 seq=27 comm_op_id=34 -> OpNode#34 shape=[926679040] comm_bytes=115834880`，回放信息齐全。
