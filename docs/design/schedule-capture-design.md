# L0-L3 IR 层级归属设计

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
