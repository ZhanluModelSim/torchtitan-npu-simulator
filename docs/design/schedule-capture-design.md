# L0-L3 IR 层级归属设计

> 历史说明：本文保留早期 L0/L1/L2 通信分层讨论，不再定义当前 PP L2
> 组装方案。当前的 rank 身份、语义 action、FSDP residency、跨 stage
> 依赖和 `SchedulePlan` 主数据流以
> [PP L2 Capture Architecture](./pp-l2-capture-architecture.md) 为准。
> 特别地，RESHARD 是本地 full-parameter residency 释放，不等同于梯度
> reduce-scatter。

> 状态：设计中
> 作者：Copilot
> 日期：2026-07-09
> 修订：2026-07-09 v2 — 基于调用路径上下文区分通信层级，不依赖名称匹配

## 1. 问题陈述

当前实现中 L1 StepGraph 和 L2 ScheduleGraph 的通信层级归属有误：

| 通信类型 | 当前归属 | 正确归属 | 原因 |
|----------|----------|----------|------|
| CP P2P (`_WindowExchange`) | L1 + L2 混杂 | **L1** | attention 计算的一部分 |
| CP allgather (`_allgather_seq`) | L1 + L2 混杂 | **L1** | attention 计算的一部分 |
| PP P2P (`forward_send/recv`) | L1（错误） | **L2** | stage 间调度依赖 |
| FSDP allgather (unshard) | L1 + L2 混杂 | **L2** | 参数管理，StepGraph 间依赖 |
| FSDP reduce_scatter (reshard) | L1 + L2 混杂 | **L2** | 参数管理，StepGraph 间依赖 |
| FSDP allreduce | L2 | **L2** | 正确 |

## 2. 层级定义

### L0 OpNode

纯计算算子（aten/npu/triton）。**不含任何通信算子**。

### L1 StepGraph

一个 microbatch 的一个 phase（forward/backward/optimizer）的计算 DAG。

包含：
- 所有 L0 计算算子
- **CP 通信**：`_WindowExchange` 的 P2P、`_allgather_seq` 的 allgather——这些在 `CompressorAttentionCP` 的 forward/backward 函数内部调用，是 attention 计算的一部分
- Activation checkpointing 的 recomputation 算子

不包含：PP P2P、FSDP 通信。

### L2 ScheduleGraph

L1 StepGraph 之间的调度依赖关系。

包含：
- StepInstance（每个 microbatch × phase 的 L1 实例）
- **PP P2P DataPass**：stage 间 activation/gradient 传递
- **FSDP DataPass**：unshard/reshard/allreduce
- execution_timeline：microbatch 调度时序 + PP/FSDP 通信时序

不包含：L0 计算算子、CP 通信。

## 3. 基于调用路径的层级判定

### 3.1 核心思路

不通过通信算子名称（如 `cp_` 前缀）判定层级，而是通过**捕获时的调用路径上下文**判定：通信发生在模型计算内部（L1）还是框架调度层（L2）。

### 3.2 调用路径分析

每种通信类型的调用栈：

| 通信类型 | 调用入口 | 调用栈特征 |
|----------|----------|-----------|
| CP P2P | `CompressorAttentionCP._pre_hook` | model.forward → attention → _pre_hook → _WindowExchange → c10d.isend |
| CP allgather | `CompressorAttentionCP._post_hook` | model.forward → attention → _post_hook → _allgather_seq → funcol.all_gather |
| PP P2P | `PipelineSchedule._step_microbatches` | pp_schedule.step → _step_microbatches → _batch_p2p → dist.isend |
| FSDP unshard | `FSDPParamGroup.pre_forward` | model.forward → module → pre_forward → unshard → all_gather |
| FSDP reshard | `FSDPParamGroup.post_backward` | autograd.backward → post_backward → reshard → reduce_scatter |
| FSDP allreduce | 梯度同步 | optimizer → all_reduce |

### 3.3 方案：`_comm_layer` 上下文变量

在 `meta_env.py` 中维护一个全局 `_comm_layer` 变量，由**调用入口的 patch** 设置：

```python
# meta_env.py
_comm_layer: str = ""  # "", "L1" (model compute), "L2" (framework scheduling)
```

**设置时机**——patch 各调用入口：

| 调用入口 | 设置值 | 实现方式 |
|----------|--------|----------|
| `_WindowExchange.forward/backward` | `"L1"` | 已有 patch（`_meta_safe_forward`），在记录 CP P2P 前设置 |
| `_allgather_seq` | `"L1"` | 新增 patch，在函数入口设置 |
| `FSDPParamGroup.unshard` | `"L2"` | 新增 patch，在方法入口设置 |
| `FSDPParamGroup.reshard` | `"L2"` | 新增 patch，在方法入口设置 |
| `FSDPParamGroup.pre_forward` | `"L2"` | 新增 patch（覆盖 unshard 调用） |
| `FSDPParamGroup.post_backward` | `"L2"` | 新增 patch（覆盖 reshard 调用） |
| `PipelineSchedule._step_microbatches` | `"L2"` | 新增 patch，在方法入口设置 |
| 其他（无 patch 覆盖的路径） | `""` | 默认，由 `_record_comm` 回退判定 |

**读取时机**——在 `comm_events.py` 的 `_record_comm` 中读取：

```python
def _record_comm(recorder, comm_primitive, group, tensor, ...):
    from torchtitan_npu.simulator.meta_env import _comm_layer
    event = recorder.record(...)
    event.comm_layer = _comm_layer  # "L1" or "L2" or ""
    ...
```

### 3.4 回退判定

当 `_comm_layer` 为空时（未被任何 patch 设置），使用调用栈特征回退判定：

```python
def _infer_comm_layer() -> str:
    """Fallback: infer layer from call stack when _comm_layer is not set."""
    import inspect
    for frame_info in inspect.stack():
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            continue
        name = module.__name__ or ""
        if "compressor_attention_cp" in name:
            return "L1"
        if "fsdp" in name or "pipelining" in name:
            return "L2"
    return "L2"  # default to L2 for unknown comm
```

> 回退判定只在 patch 未覆盖的路径上触发，正常情况下 `_comm_layer` 已被设置。

### 3.5 为什么不只用 ModulePathTracker

`ModulePathTracker` 通过 forward hooks 追踪模块调用栈，可以区分"是否在模型 forward 内部"。但：
- FSDP 的 `pre_forward` hook 也在模型 forward 内部执行（作为 forward pre_hook），`ModulePathTracker` 会显示模块路径
- PP P2P 在 `PipelineSchedule._step_microbatches` 中调用，此时不在任何模块 forward 内部，`ModulePathTracker` 栈为空

因此 `ModulePathTracker` 能区分 PP P2P（栈空）vs CP/FSDP（栈非空），但**无法区分 CP vs FSDP**（两者都在模块 forward 内部）。`_comm_layer` 上下文变量通过精确 patch 调用入口解决了这个问题。

## 4. 实现变更

### 4.1 CommEvent 增加 comm_layer 字段

```python
@dataclass
class CommEvent:
    # ... 现有字段 ...
    comm_layer: str = ""  # "L1" (model compute) or "L2" (framework scheduling)
```

### 4.2 meta_env.py 增加 _comm_layer 和 patch

```python
_comm_layer: str = ""

# 在 _meta_safe_forward (已有) 中设置:
def _meta_safe_forward(ctx, tensor, window, group):
    global _comm_layer
    _comm_layer = "L1"  # CP P2P is part of attention compute
    ...

# 新增 patch: _allgather_seq
def _patch_allgather_seq_for_comm_layer():
    from torchtitan_npu.distributed.context_parallel.compressor_attention_cp import _allgather_seq
    orig = _allgather_seq
    def _patched(tensor, mesh, seq_dim=1):
        global _comm_layer
        _comm_layer = "L1"
        return orig(tensor, mesh, seq_dim)
    # patch in module

# 新增 patch: FSDPParamGroup.unshard/reshard
def _patch_fsdp_for_comm_layer():
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    orig_unshard = FSDPParamGroup.unshard
    orig_reshard = FSDPParamGroup.reshard
    def _patched_unshard(self, async_op=False):
        global _comm_layer
        _comm_layer = "L2"
        return orig_unshard(self, async_op)
    def _patched_reshard(self):
        global _comm_layer
        _comm_layer = "L2"
        return orig_reshard(self)
    FSDPParamGroup.unshard = _patched_unshard
    FSDPParamGroup.reshard = _patched_reshard

# 新增 patch: PipelineSchedule._step_microbatches
def _patch_pipeline_schedule_for_comm_layer():
    from torch.distributed.pipelining.schedules import _PipelineSchedule
    orig = _PipelineSchedule._step_microbatches
    def _patched(self, *args, **kwargs):
        global _comm_layer
        _comm_layer = "L2"
        return orig(self, *args, **kwargs)
    _PipelineSchedule._step_microbatches = _patched
```

### 4.3 comm_events.py: _record_comm 记录 comm_layer

```python
def _record_comm(recorder, comm_primitive, group, tensor, ...):
    from torchtitan_npu.simulator.meta_env import _comm_layer
    event = recorder.record(...)
    event.comm_layer = _comm_layer
    ...
```

### 4.4 schedule_builder.py: 按 comm_layer 分流

```python
# L1 DataPasses (CP comm, comm_layer="L1")
l1_data_passes = [e for e in comm_events if e.comm_layer == "L1"]

# L2 DataPasses (PP/FSDP comm, comm_layer="L2")
l2_data_passes = [e for e in comm_events if e.comm_layer == "L2"]

# L2 execution_timeline: only L2 events
l2_timeline = [e for e in all_events if e.comm_layer != "L1" and not e.is_l0_compute]
```

### 4.5 step_boundary.py: L1 StepGraph 包含 CP 通信

`build_step_graphs` 将 `comm_layer="L1"` 的 CommEvent 关联到对应 phase 的 StepGraph，作为内部 DataPass。

## 5. 实施计划

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `meta_env.py` | 增加 `_comm_layer` 全局变量；patch `_allgather_seq`、`FSDPParamGroup.unshard/reshard`、`PipelineSchedule._step_microbatches` 设置 `_comm_layer` |
| 2 | `comm_events.py` | `CommEvent` 增加 `comm_layer` 字段；`_record_comm` 读取并记录 `_comm_layer` |
| 3 | `schedule_builder.py` | 按 `comm_layer` 分流：L1 通信归 StepGraph，L2 通信归 ScheduleGraph |
| 4 | `step_boundary.py` | `build_step_graphs` 关联 L1 通信到 StepGraph |
| 5 | `schedule_graph.py` | StepGraph 增加 `internal_data_passes` 字段 |
| 6 | 测试 | 验证 L1 只含 CP 通信、L2 只含 PP/FSDP 通信 |

## 6. L2 DataPass 与 L1 StepGraph 的连接关系

### 6.1 问题

当前 DataPass 的 `src_exit_op` 和 `dst_entry_op` 都指向**通信算子自身的 op_id**（`src_exit_op == dst_entry_op`），没有建立与 L1 StepGraph 入口/出口算子的连接。回放时无法从 L2 顶层沿着 DataPass 链找到 L1 的具体入口/出口算子。

### 6.2 需要的连接关系

每条 L2 DataPass 应连接两个 L1 StepGraph 实例（或一个 L1 实例与一个通信端点）：

```
L1 StepGraph (src)  ──src_exit_op──▶  L2 DataPass  ──dst_entry_op──▶  L1 StepGraph (dst)
   (出口算子)                              (通信)                          (入口算子)
```

| L2 通信类型 | src_exit_op 指向 | dst_entry_op 指向 | 语义 |
|------------|-----------------|------------------|------|
| FSDP unshard (allgather) | 通信算子自身 op_id | L1 forward 的**入口算子**（第一个消费 unshard 参数的算子） | unshard 后参数进入 forward 计算 |
| FSDP reshard (reduce_scatter) | L1 backward 的**出口算子**（梯度产出算子） | 通信算子自身 op_id | backward 产出的梯度被 reshard |
| FSDP allreduce | L1 optimizer 的**出口算子** | 通信算子自身 op_id | optimizer 产出的梯度被同步 |
| PP forward_send | L1 forward 的**出口算子**（activation 产出算子） | 通信算子自身 op_id | forward 产出的 activation 被 send |
| PP forward_recv | 通信算子自身 op_id | L1 forward 的**入口算子**（第一个消费 recv activation 的算子） | recv 的 activation 进入 forward 计算 |
| PP backward_send | L1 backward 的**出口算子**（gradient 产出算子） | 通信算子自身 op_id | backward 产出的 gradient 被 send |
| PP backward_recv | 通信算子自身 op_id | L1 backward 的**入口算子**（第一个消费 recv gradient 的算子） | recv 的 gradient 进入 backward 计算 |

### 6.3 连接关系的方向性

```
                    ┌─────────────────────────────────────────┐
                    │           L2 ScheduleGraph              │
                    │                                         │
  FSDP unshard ──▶  │  dst_entry_op → forward StepGraph entry  │
                    │                      │                  │
                    │                 (L1 compute)             │
                    │                      │                  │
  PP forward_send ◀│  src_exit_op ← forward StepGraph exit     │
                    │                      │                  │
                    │                 (PP P2P)                │
                    │                      │                  │
  PP forward_recv ─▶│  dst_entry_op → next stage forward entry│
                    │                      │                  │
                    │                 (L1 compute)             │
                    │                      │                  │
  FSDP reshard ◀│  src_exit_op ← backward StepGraph exit      │
                    │                                         │
                    └─────────────────────────────────────────┘
```

### 6.4 实现方案

#### 6.4.1 捕获时记录连接点

在 `_record_comm` 中，除了记录通信事件本身，还需要记录它与 L1 StepGraph 的连接关系。关键信息是：**通信发生时，当前 L1 计算图的入口/出口算子是什么**。

**方案 A：基于 producer/consumer 关系（推荐）**

`OpDispatchCapture` 已经维护了 `_producer` 字典（`id(tensor) → op_id`），记录每个张量由哪个算子产出。当通信算子被调用时：

- **unshard（allgather）**：通信产出的张量被后续计算算子消费。`dst_entry_op` = 第一个消费通信产出张量的算子 op_id（通过 `_producer` 反查通信算子的消费者）。
- **reshard（reduce_scatter）**：通信消费的张量由前序计算算子产出。`src_exit_op` = 产出通信输入张量的算子 op_id（通过 `_producer` 查找）。
- **PP send**：通信消费的张量由 forward 计算产出。`src_exit_op` = 产出 send 输入张量的算子 op_id。
- **PP recv**：通信产出的张量被后续计算消费。`dst_entry_op` = 第一个消费 recv 产出张量的算子 op_id。

```python
def _record_comm(recorder, comm_primitive, group, tensor, output_tensor=None):
    ...
    event = recorder.record(...)
    
    # Determine connection to L1 StepGraph
    capture = get_active_capture()
    if capture is not None:
        if output_tensor is not None:
            # Comm produces a tensor that will be consumed by L1 compute
            # dst_entry_op will be set later when the consumer is captured
            # For now, register a pending link
            capture._pending_comm_links[id(output_tensor)] = event
        else:
            # Comm consumes a tensor produced by L1 compute
            # src_exit_op = the producer of the input tensor
            producer_op = capture._producer.get(id(tensor))
            if producer_op is not None:
                event.src_exit_op = producer_op
    ...
```

**延迟解析 dst_entry_op**：通信产出的张量可能稍后才被计算算子消费。在 `_record_event` 中，当捕获一个新算子时，检查其输入张量是否有 pending comm link：

```python
def _record_event(self, raw_op_type, flat_inputs, ...):
    ...
    # Check if any input tensor was produced by a comm op (pending link)
    for t in flat_inputs:
        if id(t) in self._pending_comm_links:
            event = self._pending_comm_links.pop(id(t))
            event.dst_entry_op = op_id  # this op is the first consumer
    ...
```

#### 6.4.2 StepInstance 标识

DataPass 的 `src_instance` / `dst_instance` 需要标识具体的 L1 StepGraph 实例（哪个 rank、哪个 microbatch、哪个 phase）：

```python
# 当前: src_instance = "rank1" (只有 rank)
# 改为: src_instance = "rank1_mb0_forward" (rank + microbatch + phase)
```

这需要从 `_pp_context` 获取 `mb_idx` 和 `phase`，在 `_record_comm` 时记录到 CommEvent。

#### 6.4.3 DataPass 结构更新

```python
@dataclass
class DataPass:
    src_instance: str          # "rank1_mb0_forward" (rank + mb + phase)
    dst_instance: str          # "rank2_mb0_forward" or "group:fsdp"
    slots: list[TensorSlot]
    src_device: int | None = None
    dst_device: int | None = None
    requires_communication: bool = False
    comm_primitive: str = ""
    comm_group_ranks: list[list[int]] = field(default_factory=list)
    # 连接关系:
    # src_exit_op: src_instance 的出口算子 op_id (产出被通信的数据)
    # dst_entry_op: dst_instance 的入口算子 op_id (消费通信的数据)
    # 如果指向通信算子自身，表示该端是通信端点（非 L1 计算端）
```

TensorSlot 的 `src_exit_op` / `dst_entry_op` 语义：

| 字段 | 值 | 含义 |
|------|-----|------|
| `src_exit_op` | L1 计算算子 op_id | 数据由该 L1 算子产出，然后进入通信 |
| `src_exit_op` | 通信算子 op_id | 数据由通信产出（如 recv/unshard），src 端是通信端点 |
| `dst_entry_op` | L1 计算算子 op_id | 数据被该 L1 算子消费，通信后进入计算 |
| `dst_entry_op` | 通信算子 op_id | 数据被通信消费（如 send/reshard），dst 端是通信端点 |

### 6.5 回放路径示例

以 PP=4, MB=0 的 forward 为例，回放路径：

```
1. L2: PP forward_recv (stage 1, MB 0)
   └─ dst_entry_op → L1 forward StepGraph (stage 1, MB 0) 的入口算子

2. L1: forward StepGraph (stage 1, MB 0) 内部按拓扑序回放
   └─ exit_nodes → 出口算子

3. L2: PP forward_send (stage 1, MB 0)
   └─ src_exit_op → L1 forward StepGraph (stage 1, MB 0) 的出口算子
   └─ dst_entry_op → 通信算子自身（send 端点）

4. L2: PP forward_recv (stage 2, MB 0)
   └─ dst_entry_op → L1 forward StepGraph (stage 2, MB 0) 的入口算子

5. L1: forward StepGraph (stage 2, MB 0) 内部回放
   ...
```

FSDP 的回放路径：

```
1. L2: FSDP unshard (allgather, rank 1, MB 0, forward)
   └─ dst_entry_op → L1 forward StepGraph (rank 1, MB 0) 的入口算子

2. L1: forward StepGraph 内部回放
   └─ exit_nodes → 出口算子

3. L2: FSDP reshard (reduce_scatter, rank 1, MB 0, backward)
   └─ src_exit_op → L1 backward StepGraph (rank 1, MB 0) 的出口算子
```

### 6.6 实施补充

在原实施计划基础上增加：

| 步骤 | 文件 | 改动 |
|------|------|------|
| 7 | `dispatch_capture.py` | 增加 `_pending_comm_links` 字典；`_record_event` 中检查输入张量是否有 pending comm link，设置 `dst_entry_op` |
| 8 | `comm_events.py` | `_record_comm` 中通过 `_producer` 查找 `src_exit_op`；注册 pending link for `dst_entry_op` |
| 9 | `schedule_builder.py` | DataPass 的 `src_instance`/`dst_instance` 改为 `rank_mb_phase` 格式 |
| 10 | 测试 | 验证 DataPass 的 `src_exit_op`/`dst_entry_op` 指向 L1 计算算子（非通信算子自身） |
