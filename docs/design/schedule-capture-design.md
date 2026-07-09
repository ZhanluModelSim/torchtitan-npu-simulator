# L2 ScheduleGraph 捕获方案设计

> 状态：设计中
> 作者：Copilot
> 日期：2026-07-09
> 修订：2026-07-09 — 取消 Rank 0 合并、增加 microbatch 分层捕获策略

## 1. 问题陈述

### 1.1 现状

Simulator 的 L0（OpNode）和 L1（StepGraph）层忠实捕获了每个算子的 dispatch 事件，包括 op_type、shape、seq_idx、phase 等。但 L2（ScheduleGraph）的 `build_schedule_graph` 存在以下**违反"不猜测、忠实 capture"原则**的问题：

| 问题 | 代码位置 | 性质 |
|------|----------|------|
| StepInstance 按 RankTable 编码生成 | 第 32-46 行 | ❌ 推断生成，非捕获 |
| P2P DataPass 的 src_rank 通过遍历 RankTable 查找 | 第 69-75 行 | ❌ 推断 |
| 集合通信 DataPass 做 all-to-all 展开 | 第 115-127 行 | ❌ 拓扑展开，非逐条捕获 |
| 非 P2P op 的 stage/mb 通过"最近 P2P anchor"推断 | 第 200-212 行 | ❌ 邻近性推断 |
| execution_timeline 的 rank 硬编码为 0 | 第 157 行 | ❌ 简化假设 |
| multi_proc 模式下 Rank 0 合并所有 stage IR | trainer.py | ❌ 不必要，各 stage 本身就是不同 rank |

### 1.2 目标

重新设计 L2 捕获方案，使 ScheduleGraph 的**所有字段**都来自捕获数据，而非推断或编码生成。核心原则：

- **PP 调度**：通过 multi_proc 方式忠实捕获（每个 PP stage 一个进程，真实执行 1F1B 调度）
- **CP/TP/EP/FSDP 通信**：通过 meta device + FakeProcessGroup 忠实捕获（通信拦截器记录每次调用）
- **进程数 = PP degree**：`torchrun --nproc_per_node=PP`，最小化资源开销
- **各 stage 独立输出**：不合并到 Rank 0，每个 stage 的 IR 直接以其 rank ID 标识
- **microbatch 分层捕获**：首个 microbatch 捕获完整 L0/L1，后续 microbatch 只捕获 L2 调度时序

## 2. 捕获架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  torchrun --nproc_per_node=PP                               │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐       ┌──────────────┐ │
│  │  PP Stage 0  │  │  PP Stage 1  │  ...  │  PP Stage N  │ │
│  │  (gloo rank) │  │  (gloo rank) │       │  (gloo rank) │ │
│  │  rank=0      │  │  rank=1      │       │  rank=N      │ │
│  │              │  │              │       │              │ │
│  │  Meta Device │  │  Meta Device │       │  Meta Device │ │
│  │  Fake PG     │  │  Fake PG     │       │  Fake PG     │ │
│  │  (CP/TP/EP/  │  │  (CP/TP/EP/  │       │  (CP/TP/EP/  │ │
│  │   FSDP 子组) │  │   FSDP 子组) │       │   FSDP 子组) │ │
│  │              │  │              │       │              │ │
│  │  Capture:    │  │  Capture:    │       │  Capture:    │ │
│  │  MB0: L0+L1+ │  │  MB0: L0+L1+ │       │  MB0: L0+L1+ │ │
│  │      L2+L3   │  │      L2+L3   │       │      L2+L3   │ │
│  │  MB1+: L2    │  │  MB1+: L2    │       │  MB1+: L2    │ │
│  │      only    │  │      only    │       │      only    │ │
│  │              │  │              │       │              │ │
│  │  Output:     │  │  Output:     │       │  Output:     │ │
│  │  rank_0/     │  │  rank_1/     │       │  rank_N/     │ │
│  └──────────────┘  └──────────────┘       └──────────────┘ │
│                                                             │
│  各 stage 独立输出，不合并到 Rank 0                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 各维度的捕获方式

| 维度 | 捕获方式 | 进程数 | PG 类型 | 通信行为来源 |
|------|----------|--------|---------|-------------|
| PP | 真实多进程执行 | PP degree | gloo（元数据交换） | `forward_one_chunk`/`backward_one_chunk` 真实调用 → `_pp_context` 捕获 |
| PP P2P | 拦截器记录 | — | gloo PG | `dist.isend`/`irecv` 拦截器记录 CommEvent |
| CP | Fake PG 模拟 | — | FakeProcessGroup | `_WindowExchange` + `_allgather_seq` 真实调用 → 拦截器记录 |
| TP | Fake PG 模拟 | — | FakeProcessGroup | `RowwiseParallel`/`ColwiseParallel` 的 all_reduce → 拦截器记录 |
| EP | Fake PG 模拟 | — | FakeProcessGroup | MoE dispatch 的 all_to_all → 拦截器记录 |
| FSDP | Fake PG 模拟 | — | FakeProcessGroup | `fully_shard` 的 all_gather/reduce_scatter → 拦截器记录 |

**关键点**：CP/TP/EP/FSDP 的通信行为是**真实触发的**——模型代码在 meta device 上执行时，这些通信函数被真实调用（只是数据是 meta tensor，通信被 no-op），拦截器记录每次调用的完整上下文。

## 3. Microbatch 分层捕获策略

### 3.1 问题分析

一个 train step 内可能有多个 microbatch：

- **PP 模式**：`PipelineScheduleSingle._step_microbatches()` 内部循环 `n_microbatches` 次，每次调用 `forward_one_chunk(mb_idx)` 和 `backward_one_chunk(mb_idx)`。1F1B 调度下，不同 microbatch 的 forward/backward 交错执行。
- **非 PP 模式**：`train_step()` 外部循环 `gradient_accumulation_steps` 次，每次调用 `forward_backward_step()`。

关键观察：**同一个 train step 内，不同 microbatch 的 L0/L1 计算图是相同的**——相同的模型权重、相同的 op 序列、相同的 shape（meta device 下 shape 不依赖输入数值）。差异仅在于：
1. **L2 调度时序**：哪个 microbatch 在何时被 forward/backward，P2P 通信的顺序
2. **L3 通信事件**：每次 microbatch 触发的通信调用（虽然类型和 shape 相同，但调用次数和顺序可能不同）

### 3.2 分层捕获设计

| Microbatch | L0 (OpNode) | L1 (StepGraph) | L2 (ScheduleGraph) | L3 (CommEvent) |
|------------|-------------|----------------|---------------------|-----------------|
| MB 0 | ✅ 完整捕获 | ✅ 从 L0 构建 | ✅ 完整捕获 timeline | ✅ 完整捕获 |
| MB 1+ | ❌ 跳过 | ❌ 跳过 | ✅ 只捕获 timeline + comm | ✅ 完整捕获 |

**首个 microbatch（MB 0）**：
- `OpDispatchCapture` 正常工作，记录每个 op 的 dispatch 事件
- `build_nodes()` / `build_step_graphs()` 正常构建 L0/L1
- L0/L1 作为 **template**（模板），后续 microbatch 复用

**后续 microbatch（MB 1+）**：
- `OpDispatchCapture` 进入 **pass-through 模式**：不记录 L0 op，但 `_pp_context` 仍然更新
- 只记录 L2 timeline 事件：每次 `forward_one_chunk`/`backward_one_chunk` 的调用（stage, mb_idx, phase, seq_idx）
- 只记录 L3 CommEvent：每次通信调用的拦截器记录（comm_primitive, group, shape, bytes, p2p_context）

### 3.3 实现机制

#### 3.3.1 OpDispatchCapture 增加 pass-through 模式

```python
class OpDispatchCapture(TorchDispatchMode):
    def __init__(self, ..., phase_provider=None):
        ...
        self._capture_l0 = True  # 首个 microbatch 默认捕获 L0

    def _record_event(self, raw_op_type, flat_inputs, flat_outputs, module_path):
        if not self._capture_l0:
            return  # pass-through: 不记录 L0 op
        # ... 现有的 L0 捕获逻辑 ...
```

#### 3.3.2 按 (phase, mb_idx) 控制捕获

1F1B 调度中，MB 0 的 forward 完成后，MB 1 的 forward 开始，但 MB 0 的 backward 稍后才执行。因此不能简单地在 MB 0 forward 后关闭 L0 捕获——MB 0 的 backward 也需要完整捕获。

`_capture_l0` 由 patched `forward_one_chunk`/`backward_one_chunk` 按 `(phase, mb_idx)` 控制：

```python
def _should_capture_l0(phase: str, mb_idx: int) -> bool:
    """只有 MB 0 的 forward 和 backward 完整捕获 L0。"""
    return mb_idx == 0

def _patched_fwd_one_chunk(self, mb_idx, *args, **kwargs):
    _pp_context["mb_idx"] = mb_idx
    _pp_context["phase"] = "forward"
    _pp_context["stage"] = self.stage_index
    cap = get_active_capture()
    if cap is not None:
        cap._capture_l0 = _should_capture_l0("forward", mb_idx)
    result = _original_fwd_one_chunk(self, mb_idx, *args, **kwargs)
    # 记录 L2 timeline 事件（无论是否捕获 L0）
    _record_timeline_event("forward_one_chunk", mb_idx, self.stage_index)
    return result

def _patched_bwd_one_chunk(self, mb_idx, *args, **kwargs):
    _pp_context["mb_idx"] = mb_idx
    _pp_context["phase"] = "backward"
    _pp_context["stage"] = self.stage_index
    cap = get_active_capture()
    if cap is not None:
        cap._capture_l0 = _should_capture_l0("backward", mb_idx)
    result = _original_bwd_one_chunk(self, mb_idx, *args, **kwargs)
    _record_timeline_event("backward_one_chunk", mb_idx, self.stage_index)
    return result
```

**效果**：
- MB 0 forward: `_capture_l0=True` → 完整捕获 L0
- MB 0 backward: `_capture_l0=True` → 完整捕获 L0
- MB 1+ forward: `_capture_l0=False` → pass-through
- MB 1+ backward: `_capture_l0=False` → pass-through

**L1 StepGraph 分桶**：MB 0 的 forward ops 进入 `forward` StepGraph，MB 0 的 backward ops 进入 `backward` StepGraph。MB 1+ 的 ops 不被捕获，不进入任何 StepGraph。

#### 3.3.3 非 PP 模式

非 PP 模式下，`train_step` 循环调用 `forward_backward_step`。Simulator 的 `run_simulation_step` 只调用一次 `forward_backward_step`（因为 `training.steps=1` 且 gradient accumulation 在 PP 模式下由 schedule 内部处理）。

如果需要支持非 PP 的多 microbatch 捕获，需要修改 `run_simulation_step` 支持循环调用，并在第 2 次及之后的调用前设置 `_capture_l0 = False`。

> **当前限制**：Simulator 目前 `steps=1` 且非 PP 模式下实际只有一个 microbatch。多 microbatch 捕获主要针对 PP 模式。

#### 3.3.4 L2 Timeline 的 microbatch 记录

无论是否捕获 L0，L2 timeline 都需要记录每个 microbatch 的调度事件。这通过 `_record_timeline_event` 实现：

```python
def _record_timeline_event(action: str, mb_idx: int, stage: int):
    """记录 L2 调度级 timeline 事件（不依赖 L0 捕获）。"""
    from torchtitan_npu.simulator.capture.comm_events import get_active_recorder
    recorder = get_active_recorder()
    if recorder is not None:
        recorder.record_timeline_event(
            seq_idx=next(_seq_counter),
            action=action,
            pp_stage=stage,
            pp_mb_idx=mb_idx,
            phase=_pp_context.get("phase", ""),
        )
```

### 3.4 L2 Timeline 结构

```python
@dataclass
class TimelineEntry:
    seq_idx: int           # 全局执行序号（跨 microbatch 递增）
    op_id: int             # L0 OpNode ID（MB 0 的 op 有 ID，MB 1+ 为 -1）
    rank: int              # 当前进程的 gloo rank
    pipeline_stage: int    # PP stage（from _pp_context, captured）
    micro_batch_idx: int   # microbatch 序号（from _pp_context, captured）
    phase: str             # "forward" / "backward" / "optimizer"
    comm_type: str = ""    # 通信类型（from CommEvent）
    comm_peer_rank: int = -1
    action: str = ""       # "compute" / "forward_one_chunk" / "backward_one_chunk" / "comm"
```

**MB 0 的 timeline**：每个 L0 op 都有一个 timeline entry（`op_id` 有值，`action="compute"`），加上通信 op 的 entry。

**MB 1+ 的 timeline**：只有调度级 entry（`op_id=-1`，`action="forward_one_chunk"`/`"backward_one_chunk"`），加上通信 op 的 entry。不展开到 L0 op 级别。

### 3.5 效率分析

| 配置 | 当前（全量捕获） | 新方案（MB0 全量 + MB1+ L2 only） |
|------|-----------------|-----------------------------------|
| PP=4, 4 microbatch, 16层 | 4 × 26K ops = 104K OpNode | 26K OpNode + 3×12 timeline entries |
| PP=16, 8 microbatch, 61层 | 16 × 70K ops = 1.12M OpNode | 70K OpNode + 7×16 timeline entries |
| 内存（PP=16） | ~8GB | ~500MB |

**L0/L1 捕获开销从 O(PP × num_mb × ops_per_mb) 降为 O(ops_per_mb)**，与 microbatch 数量无关。

## 4. 各 Stage 独立输出（不合并）

### 4.1 设计

multi_proc 模式下，每个 PP stage 进程独立捕获自己的 IR，**直接以 rank ID 输出**，不合并到 Rank 0：

```
simulator_output/<config_name>/
├── rank_0/                    # PP Stage 0 的输出
│   ├── step_templates.json    # L0/L1 模板（MB 0 捕获）
│   ├── execution_timeline.csv # L2 timeline（所有 microbatch）
│   ├── comm_events.json       # L3 通信事件（所有 microbatch）
│   ├── kernel_summary.csv     # L0 算子汇总
│   └── trace.html             # 可视化
├── rank_1/                    # PP Stage 1 的输出
│   └── ...
├── rank_2/
│   └── ...
└── rank_3/
    └── ...
```

### 4.2 理由

1. **不同 PP stage 本身就是不同 rank**：PP Stage 0 对应 rank 0（该 stage 的第一个 rank），PP Stage 1 对应 rank 1，以此类推。各 stage 的计算图、调度时序、通信模式都不同，合并反而丢失信息。

2. **避免 Rank 0 瓶颈**：合并需要所有进程 barrier 等待，Rank 0 串行处理所有 stage 的 IR。独立输出消除了这个瓶颈。

3. **简化实现**：不需要 `_merge_per_rank_ir()`，每个进程直接写自己的输出目录。

### 4.3 跨 stage 视图

虽然各 stage 独立输出，但用户可能需要查看跨 stage 的调度关系（如 PP P2P 通信的 send/recv 配对）。通过以下方式提供：

- **各 stage 的 `execution_timeline.csv`** 包含 P2P 通信的 `comm_peer_rank` 字段，用户可以自行关联 send 和 recv。
- **可视化工具**（trace.html）可以加载多个 stage 的 timeline，按 seq_idx 对齐展示。

## 5. 数据模型变更

### 5.1 L0 OpNode：增加 PP context 字段

```python
@dataclass
class _RawEvent:
    # ... 现有字段 ...
    phase: str = "forward"
    seq_idx: int = 0
    # 新增：PP context（每次 _record_event 时从 _pp_context 读取）
    pp_stage: int = -1      # PP stage index (-1 if not PP)
    pp_mb_idx: int = -1     # microbatch index (-1 if not PP)
```

**捕获时机**：在 `_record_event()` 中，与 `phase` 同时读取 `_pp_context`。

### 5.2 L2 TimelineEntry

```python
@dataclass
class TimelineEntry:
    seq_idx: int           # from capture (global, cross-microbatch)
    op_id: int             # from OpNode.op_id (MB 0 only, -1 for MB 1+)
    rank: int              # from gloo rank (captured)
    pipeline_stage: int    # from OpNode.pp_stage or _pp_context (captured)
    micro_batch_idx: int   # from OpNode.pp_mb_idx or _pp_context (captured)
    phase: str             # from OpNode.phase (captured)
    comm_type: str = ""    # from CommEvent (captured)
    comm_peer_rank: int = -1  # from CommEvent (captured)
    action: str = ""       # "compute" / "forward_one_chunk" / "backward_one_chunk" / "comm"
```

### 5.3 L2 DataPass：从 CommEvent 直接映射

**P2P 通信**（PP）：
```python
DataPass(
    src_instance=f"rank{event.p2p_stage}",        # captured
    dst_instance=f"rank{event.p2p_peer_rank}",    # captured
    slots=[TensorSlot(shape=event.tensor_shape, volume_bytes=event.volume_bytes)],
    comm_primitive=f"p2p_{event.p2p_direction}",   # captured
)
```

**集合通信**（CP/TP/EP/FSDP）：
```python
DataPass(
    src_instance=f"rank{rank}",
    dst_instance=f"group:{event.comm_dim}",
    slots=[TensorSlot(shape=event.tensor_shape, volume_bytes=event.volume_bytes)],
    comm_primitive=event.comm_primitive,            # captured
    comm_group_ranks=event.comm_ranks,              # captured
)
```

**关键变更**：不再对 group 内所有 rank 对生成 all-to-all DataPass。每次集合通信调用生成**一个** DataPass，`comm_group_ranks` 字段记录参与方。

### 5.4 DataPass 新增字段

```python
@dataclass
class DataPass:
    # ... 现有字段 ...
    comm_group_ranks: list[list[int]] = field(default_factory=list)  # 新增
```

### 5.5 CommEvent 增加 seq_idx

当前 CommEvent 没有 `seq_idx` 字段，无法在 timeline 中排序。需要增加：

```python
@dataclass
class CommEvent:
    # ... 现有字段 ...
    seq_idx: int = 0  # 新增：全局执行序号（与 L0 op 共享同一个 _seq_counter）
```

## 6. 捕获流程

### 6.1 multi_proc 模式（PP>1）

每个 PP stage 进程独立执行：

```
1. patch_device_type_to_meta() → meta device + gloo PG + FakePG 子组
2. init_distributed() → 返回 full_ws, ParallelDims 验证通过
3. build_mesh() → world_mesh 用 FakePG, PP 子组用 gloo
4. pipeline_module_split() → 本进程只构建本 stage 的模型分片
5. forward_backward_step() → pp_schedule.step() 真实执行 1F1B
   │
   ├── MB 0 forward_one_chunk(0):
   │   ├── _pp_context 更新: stage=N, mb=0, phase=forward
   │   ├── _capture_l0 = True → OpDispatchCapture 完整捕获 L0 ops
   │   ├── CP/TP/EP/FSDP 通信 → 拦截器记录 CommEvent
   │   └── PP P2P (isend/irecv) → 拦截器记录 p2p_stage/mb/direction
   │
   ├── MB 1 forward_one_chunk(1):
   │   ├── _pp_context 更新: stage=N, mb=1, phase=forward
   │   ├── _capture_l0 = False → pass-through（不记录 L0 ops）
   │   ├── L2 timeline 记录: action="forward_one_chunk", mb=1
   │   ├── CP/TP/EP/FSDP 通信 → 拦截器记录 CommEvent
   │   └── PP P2P → 拦截器记录 CommEvent
   │
   ├── MB 0 backward_one_chunk(0):
   │   ├── _pp_context 更新: stage=N, mb=0, phase=backward
   │   ├── _capture_l0 = True → 完整捕获 L0 ops
   │   └── 通信 → 拦截器记录
   │
   ├── MB 1+ backward_one_chunk(i):
   │   ├── _capture_l0 = False → pass-through
   │   └── L2 timeline + CommEvent 记录
   │
6. optimizer_step() → FSDP all_gather/reduce_scatter → 拦截器记录
7. build_nodes() → 本 stage 的 OpNode（仅 MB 0 的 ops）
8. build_step_graphs() → 按 phase 分桶（仅 MB 0）
9. build_schedule_graph() → 本 stage 的 ScheduleGraph
   ├── step_templates: MB 0 的 L0/L1 模板
   ├── execution_timeline: 所有 microbatch 的调度时序
   ├── StepInstance: rank=gloo_rank, stage=gloo_rank
   └── DataPass: 从 CommEvent 直接映射
10. _export(rank) → 写入 rank_N/ 目录（不合并）
```

### 6.2 fake_backend 模式（PP=1）

```
1. patch_device_type_to_meta() → meta device + FakePG
2. forward_backward_step() → 模型在 meta device 上执行
   ├── MB 0: OpDispatchCapture 完整捕获 L0
   ├── MB 1+ (如有): pass-through，只捕获 L2 timeline + CommEvent
   └── 通信 → 拦截器记录
3. optimizer_step() → 同上
4. build_nodes() / build_step_graphs() / build_schedule_graph()
5. _export(rank=0) → 写入 rank_0/ 目录
```

## 7. build_schedule_graph 重构

### 7.1 新签名

```python
def build_schedule_graph(
    *,
    step_templates: dict[str, StepGraph],
    rank_table: RankTable,
    comm_events: list[CommEvent],
    timeline_events: list[TimelineEvent],  # 新增：从 _pp_context 捕获的调度事件
    pipeline_schedule: str = "none",
    num_micro_batches: int = 1,
    gradient_accumulation: int = 1,
    rank: int = 0,           # 当前进程的 gloo rank
) -> ScheduleGraph:
```

### 7.2 新实现逻辑

```python
# 1. StepInstance: 从捕获数据生成
instances = []
for template_id, template in step_templates.items():
    instances.append(StepInstance(
        instance_id=f"rank{rank}_{template_id}",
        step_ref=template_id,
        step_type=template.step_type,
        micro_batch_idx=0,
        pipeline_stage=rank,
        device_ids=[rank],
        dp_group=0,
    ))

# 2. DataPass: 从 CommEvent 直接映射（不做 all-to-all 展开）
data_passes = []
for event in comm_events:
    if event.comm_primitive in ("p2p_send", "p2p_recv"):
        if event.comm_primitive != "p2p_send":
            continue
        data_passes.append(DataPass(
            src_instance=f"rank{event.p2p_stage}",
            dst_instance=f"rank{event.p2p_peer_rank}",
            slots=[TensorSlot(...)],
            comm_primitive=f"p2p_{event.p2p_direction}",
        ))
    else:
        data_passes.append(DataPass(
            src_instance=f"rank{rank}",
            dst_instance=f"group:{event.comm_dim}",
            slots=[TensorSlot(...)],
            comm_primitive=event.comm_primitive,
            comm_group_ranks=event.comm_ranks,
        ))

# 3. execution_timeline: 合并 L0 级、调度级和通信级 timeline
execution_timeline = []

# 3a. MB 0 的 L0 ops → timeline entries（op_id 有值）
for template_id, template in step_templates.items():
    for op_id, node in template.nodes.items():
        ann = node.annotations
        execution_timeline.append(TimelineEntry(
            seq_idx=node.seq_idx,
            op_id=op_id,
            rank=rank,
            pipeline_stage=ann.get("pp_stage", -1),
            micro_batch_idx=ann.get("pp_mb_idx", -1),
            phase=ann.get("phase", template.step_type),
            action="compute",
        ))

# 3b. MB 1+ 的调度事件 → timeline entries（op_id=-1）
for ev in timeline_events:
    execution_timeline.append(TimelineEntry(
        seq_idx=ev.seq_idx,
        op_id=-1,
        rank=rank,
        pipeline_stage=ev.pipeline_stage,
        micro_batch_idx=ev.micro_batch_idx,
        phase=ev.phase,
        action=ev.action,
    ))

# 3c. 通信事件 → timeline entries
for event in comm_events:
    execution_timeline.append(TimelineEntry(
        seq_idx=event.seq_idx,
        op_id=event.op_id if event.op_id > 0 else -1,
        rank=rank,
        pipeline_stage=event.p2p_stage if event.p2p_stage >= 0 else rank,
        micro_batch_idx=event.p2p_mb_idx if event.p2p_mb_idx >= 0 else -1,
        phase=...,
        comm_type=event.p2p_direction or event.comm_primitive,
        comm_peer_rank=event.p2p_peer_rank,
        action="comm",
    ))

# 4. 排序（按 seq_idx）
execution_timeline.sort(key=lambda e: e.seq_idx)
```

### 7.3 移除的代码

- ❌ StepInstance 的 RankTable 遍历
- ❌ P2P DataPass 的 src_rank 查找
- ❌ 集合通信的 all-to-all 展开
- ❌ P2P anchor 推断逻辑
- ❌ `rank=0` 硬编码
- ❌ `_merge_per_rank_ir()` 合并逻辑

## 8. 输出文件结构

### 8.1 目录结构

```
simulator_output/<config_name>/
├── rank_0/                        # PP Stage 0 (gloo rank 0)
│   ├── step_templates.json        # L0/L1 模板（MB 0 only）
│   ├── execution_timeline.csv     # L2 timeline（所有 microbatch）
│   ├── comm_events.json           # L3 通信事件（所有 microbatch）
│   ├── data_passes.csv            # L2 DataPass
│   ├── kernel_summary.csv         # L0 算子汇总
│   ├── step_forward_l0_ops.csv    # L1 forward 的 L0 ops
│   ├── step_backward_l0_ops.csv   # L1 backward 的 L0 ops
│   ├── step_optimizer_l0_ops.csv  # L1 optimizer 的 L0 ops
│   └── trace.html                 # 可视化
├── rank_1/
│   └── ...
├── rank_2/
│   └── ...
└── rank_3/
    └── ...
```

### 8.2 execution_timeline.csv 格式

| 列 | 说明 | 来源 |
|----|------|------|
| seq_idx | 全局执行序号 | captured |
| rank | 进程 rank | captured (gloo rank) |
| pipeline_stage | PP stage | captured (_pp_context) |
| micro_batch_idx | microbatch 序号 | captured (_pp_context) |
| phase | forward/backward/optimizer | captured |
| action | compute/forward_one_chunk/backward_one_chunk/comm | captured |
| op_id | L0 OpNode ID（MB 0 有值，MB 1+ 为 -1） | captured |
| comm_type | 通信类型 | captured (CommEvent) |
| comm_peer_rank | P2P peer rank | captured (CommEvent) |

## 9. 实施计划

| 步骤 | 文件 | 改动 | 依赖 |
|------|------|------|------|
| 1 | `dispatch_capture.py` | `_RawEvent` 增加 `pp_stage`/`pp_mb_idx`；`_record_event` 读取 `_pp_context`；增加 `_capture_l0` pass-through 模式 | — |
| 2 | `op_node.py` | `OpNode` 增加 `pp_stage`/`pp_mb_idx` annotations | 步骤 1 |
| 3 | `comm_events.py` | `CommEvent` 增加 `seq_idx`；`_record_comm_with_l0` 记录 seq_idx；增加 `record_timeline_event` API | — |
| 4 | `meta_env.py` | `_patched_fwd_one_chunk`/`_patched_bwd_one_chunk` 设置 `_capture_l0`；记录 timeline event | 步骤 1, 3 |
| 5 | `schedule_graph.py` | `DataPass` 增加 `comm_group_ranks`；`TimelineEntry` 增加 `action`/`rank` 字段 | — |
| 6 | `schedule_builder.py` | 重构 `build_schedule_graph`：从 OpNode + CommEvent + timeline_events 直接映射，移除推断逻辑 | 步骤 1-5 |
| 7 | `trainer.py` | 移除 `_merge_per_rank_ir`；`_export_per_rank` 改为 `_export(rank)` 写入 `rank_N/` 目录；`run_simulation_step` 传入 `rank` 和 `timeline_events` | 步骤 6 |
| 8 | 测试 | 运行 PP=4+CP=4 验证：各 stage 独立输出、MB 0 有 L0、MB 1+ 只有 timeline、所有字段来自捕获 | 步骤 1-7 |

## 10. 验证标准

1. **各 stage 独立输出**：`rank_0/`、`rank_1/`、... 目录各自独立，无 Rank 0 合并步骤。

2. **MB 0 完整捕获**：`step_forward_l0_ops.csv` 和 `step_backward_l0_ops.csv` 包含完整的 L0 op 序列。

3. **MB 1+ 跳过 L0**：`execution_timeline.csv` 中 MB 1+ 的 entry 的 `op_id=-1`，`action="forward_one_chunk"`/`"backward_one_chunk"`。

4. **PP 调度时序**：timeline 中不同 microbatch 的 forward/backward 交错顺序反映 1F1B 调度（如 stage 0 的 MB 0 forward → MB 1 forward → MB 0 backward → MB 1 backward）。

5. **CP 通信**：`comm_events.json` 中 CP 的 allgather 和 P2P 数量与 microbatch 数量成正比。

6. **FSDP 通信**：`comm_events.json` 中 allgather/reduce_scatter 数量与 FSDP2 的 unshard/reshard 调用次数一致。

7. **无推断代码**：`build_schedule_graph` 中不存在遍历 RankTable 生成 instance、查找 src_rank、all-to-all 展开、P2P anchor 推断等逻辑。

8. **效率**：PP=4, 4 microbatch, 16层配置下，L0 OpNode 数量与单 microbatch 一致（不随 microbatch 数量增长）。
