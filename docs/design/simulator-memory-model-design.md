# Simulator Memory Model 改造设计

> 状态：设计建议
> 日期：2026-07-13
> 目标：把当前 simulator 中不准确的 `peak_mem` 累加口径，改造成可解释、可回放、可供 DES 消费的显存事件模型。

## 1. 背景

当前 simulator 在 meta device 上捕获一个训练 step 的 L0-L3 IR：

- `OpDispatchCapture` 通过 `TorchDispatchMode` 捕获 aten/npu 算子，记录输入输出 shape/dtype、producer/consumer 边、phase、seq_idx。
- `CommEventRecorder` 拦截 FSDP/TP/PP/EP/CP 等通信，记录通信 primitive、tensor shape/dtype/bytes、comm_layer、src_exit_op、dst_entry_op。
- `StepGraph` 按 forward/backward/optimizer 分桶。
- `ScheduleGraph` 记录 L2 调度与 DataPass。

这些数据已经足够作为内存建模的基础，但当前 `peak_mem` 字段语义不对。

## 2. 当前问题

### 2.1 `peak_mem` 不是峰值显存

`OpCostModel` 当前在 matmul、attention、norm、elementwise、data_move 等 handler 中把 `outputs[0]` 的 bytes 写入 `CostEstimate.peak_mem`。后续 summary 又按 step 做：

```text
total_peak_mem_bytes = sum(node.peak_mem for node in step_graph.nodes.values())
```

这等价于“阶段内所有 op 输出大小求和”，不是某个时刻活跃 tensor bytes 的峰值。

直接后果：

- 会把生命周期互斥的输出全部累加，明显高估。
- 无法表达 forward activation 是否活到 backward。
- 无法表达 checkpoint recompute 后原 forward activation 被释放。
- 无法表达 FSDP full-param buffer 的 allgather/reshard 生命周期。
- 无法作为后续 DES 的显存曲线输入。

### 2.2 `param_mem` 当前不可靠

`TensorMeta.is_parameter` 默认是 `False`，dispatch 捕获时没有建立 `id(param) -> parameter name/shard info` 的映射。因此通过 op input 判断参数大小不可靠，FSDP 后尤其不可靠：compute op 看到的可能是 allgather 回来的 full-param buffer 或其 view，而不是原始 `nn.Parameter` 对象。

### 2.3 L0 折叠不适合直接算生命周期

当前 L0 有相邻重复算子的 `repeat_count` 折叠。折叠适合展示和摘要，不适合内存生命周期计算，因为它会丢失：

- 具体 producer/consumer 顺序；
- last consumer；
- checkpoint recompute 的实际出现位置；
- microbatch overlap 下的峰值位置。

内存模型需要基于未折叠 raw event stream，或先由 raw event stream 生成 memory events 后再聚合展示。

## 3. 目标与非目标

### 3.1 目标

第一版目标是产出合理、有据、可解释的 `active tensor bytes` 曲线：

```text
active_bytes =
  persistent local params
+ external inputs / labels
+ FSDP full-param buffers
+ non-alias op outputs live by use-def
+ comm buffers live by CommEvent
```

并导出：

- 每 rank 的 `persistent_param_bytes`。
- 每 tensor 的生命周期：alloc seq、free seq、bytes、kind、producer、consumers。
- 每 seq 的 memory event：alloc/free、bytes、active_bytes_after。
- 每 step/rank 的 `active_bytes_peak` 和 peak 发生位置。
- 未分类或高风险 tensor/op 的审查列表。

### 3.2 非目标

第一版不追求对齐真实设备 allocator 的 `reserved_bytes`：

- 不模拟 allocator cache。
- 不模拟 fragmentation。
- 不模拟 kernel workspace。
- 不模拟真实 NPU 内部临时 buffer。
- 不强行复刻 PyTorch/FSDP 内部所有释放细节。

报告与导出字段应使用 `active_bytes_peak`，不要继续称为真实 device peak。

## 4. 生命周期规则

生命周期不能边捕获边决定，必须在完整 step 捕获结束后离线计算。原因是 forward tensor 是否会被 backward 消耗，只有 backward 捕获完才知道。

### 4.1 基础规则

对每个 tensor record：

```text
alloc_seq = producer_seq
free_seq = max(consumer.seq_idx)

if no consumers and not escaped:
    free_seq = producer_seq
```

分类规则：

| kind | 生命周期 |
|---|---|
| `parameter_shard` | step_start -> step_end |
| `optimizer_state` | step_start -> step_end，可作为开关 |
| `external_input` / `label` | first use 前 alloc，last consumer 后 free |
| `op_output` | producer 后 alloc，last consumer 后 free |
| `dead_temp_output` | producer 后 alloc，producer 后 free |
| `alias` | 不分配 bytes，只延长 base buffer lifetime |
| `comm_buffer` | comm alloc -> comm consumer/wait/phase boundary |
| `fsdp_full_param` | allgather/unshard alloc -> reshard/free point |

### 4.2 Forward activation 与 backward

非 checkpoint 情况：

```text
forward producer -> backward consumer
free at last backward consumer
```

checkpoint 情况：

```text
original forward activation:
  only consumed inside forward
  free at last forward consumer

recomputed activation:
  produced in backward phase
  consumed by backward gradient op
  free at last backward consumer
```

这不需要特殊猜 checkpoint，只要 raw event stream 保留完整 forward+backward use-def 链即可自然表达。

### 4.3 Escape 标记

没有 consumer 的 tensor 不能一概 producer 后释放，需要标记逃逸对象：

- loss / model output；
- L2 DataPass slot；
- P2P send buffer；
- FSDP allgather output；
- optimizer state / grad shard；
- 其他框架持有的跨 phase tensor。

初版可以保守处理：无法分类的无 consumer tensor 导出到 `unclassified_dead_outputs.csv`，默认不进入核心 peak，或按配置决定 producer 后释放。

## 5. FSDP 建模

FSDP 是内存模型中影响最大的部分，不能只依赖普通 op input/output。

### 5.1 需要表达的对象

```text
persistent local shard param:
  step_start -> step_end

fsdp allgather output / full param buffer:
  allgather/unshard seq -> matching reshard/free seq

compute view of full param:
  alias of full param buffer
  no extra bytes

reduce_scatter input / full grad buffer:
  free at reduce_scatter seq or matching reshard point

reduce_scatter output / grad shard:
  reduce_scatter seq -> optimizer consumer or step_end
```

### 5.2 匹配信号

FSDP-aware planner 使用以下信号组合，而不是靠单一字符串猜测：

- `CommEvent.comm_layer == "L2"`；
- `CommEvent.comm_primitive in {"allgather", "reduce_scatter", "allreduce"}`；
- `comm_dim` / `comm_ranks` 对应 FSDP 或 DP shard 维度；
- `src_exit_op` / `dst_entry_op` 连接到 L1 compute；
- tensor shape/dtype 与参数 shard/full param shape 匹配；
- module_path 或 FSDP hook 上下文作为辅助信息。

### 5.3 释放点

FSDP full-param buffer 的释放点优先级：

1. 明确捕获到的 reshard/free hook 或 reduce_scatter event。
2. matching FSDP reduce_scatter。
3. last compute consumer。
4. phase boundary fallback。

这个释放点比单纯 last consumer 更接近 FSDP 语义，因为真实框架会在 hook 中管理 full-param buffer 的保留和释放。

## 6. Alias、Mutation 与 Unknown

### 6.1 Alias

以下 op 初版应视为 alias，不额外分配 bytes：

- `view`
- `reshape`
- `transpose`
- `permute`
- `slice`
- `select`
- `split`
- `narrow`
- `as_strided`

以下 op 通常是真分配：

- `cat`
- `clone`
- `contiguous`
- `empty_like`
- `zeros_like`
- 通信 output buffer
- fused/custom op 的真实 output

### 6.2 In-place / mutation

`add_`、`copy_`、optimizer update、foreach mutation、`out=` 参数可能复用已有 buffer。初版至少需要：

- 识别 raw op name 中的 trailing `_`。
- 识别 kwargs 中的 `out` tensor。
- 对 mutation op 不创建新 allocation，只记录 input buffer 被写。

### 6.3 Unknown op

Unknown op 不能默认 output 分配，也不能默认 alias。处理策略：

- 白名单逐步分类。
- 未分类 op 导出 `unclassified_memory_ops.csv`。
- 核心 peak 报告展示 unknown coverage。
- 必要时提供 conservative/aggressive 两种模式：
  - conservative：unknown output 不计入核心 peak，只报告风险；
  - aggressive：unknown output 按新分配计入，作为上界。

## 7. Microbatch 与 PP

当前 simulator 对首个 microbatch 捕完整 L0，后续 microbatch 多为 L2 timeline/pass-through。内存模型不能只用 MB0 的曲线，否则 PP/gradient accumulation 下峰值位置可能错。

合理做法：

1. 使用 MB0 raw event stream 生成 L1 memory template。
2. 按 L2 execution timeline 实例化 template：

```text
mb3/op42/out0
rank7/stage2/forward/mb3/...
```

3. 对每个实例生成独立 tensor id，但复用模板中的 shape/dtype/bytes/relative lifetime。
4. 对 PP P2P、FSDP DataPass 使用 captured CommEvent 和 ScheduleGraph 补齐跨 instance 生命周期。

这样不需要重新捕获所有 microbatch 的 L0，也不需要把展示图完全展开。

当前实现采用该方案，并限定在 `pp_degree > 1` 的内存路径：

- 从 raw memory events 中为每个 `(stage, comp_type)` 选择完整模板；
- 按 `SchedulePlan.actions` 展开任意调度，不根据 1F1B、GPipe 或 DualPipeV 名称分支；
- `OVERLAP_F_B` 在内存时间轴中按 sub-action 稳定展开，保留 action span 供 Perfetto 展示；
- 参数/buffer tensor ID 保持持久，microbatch 相关输入输出生成独立 ID；
- PP P2P 接收由下游 stage input 表达，避免再叠加一份通信 buffer；
- FSDP 显式驻留 marker 保持真实调用次数，不随计算模板复制；
- 非 PP 直接调用原有 `estimate_static_memory`，不经过模板重放。

`ScheduleAction` 同时保留两类位置：`schedule_order` 是 lowered pipeline plan 的逻辑
顺序，`seq_idx` 是捕获来源位置。compute 的 `seq_idx` 可能是 L0 op 位置，而通信 action
的 `seq_idx` 可能是 plan index，因此不能把二者作为同一时间轴排序。

## 8. 优先级

### P0：最小可信闭环

| 项 | 影响 | 复杂度 | 说明 |
|---|---:|---:|---|
| 移除或改名 `sum(node.peak_mem)` | 极大 | 低 | 先停止输出误导性 peak |
| 本地参数 shard bytes | 极大 | 低 | 从 `model_parts.named_parameters()` 去重统计 |
| raw tensor use-def | 极大 | 中 | 记录 tensor_id、producer、consumers、bytes |
| alias 白名单 | 极大 | 中 | 避免 view/reshape 重复分配 |
| FSDP allgather/full-param/reshard | 极大 | 中高 | 决定大模型峰值是否合理 |
| memory events 导出 | 极大 | 中 | DES 消费的直接输入 |

### P1：训练显存与调度精度

| 项 | 影响 | 复杂度 | 说明 |
|---|---:|---:|---|
| microbatch/PP template 实例化 | 大 | 中 | 已实现通用 action-plan 重放 |
| in-place/out mutation | 中到大 | 中 | 避免 optimizer/copy 重复算 |
| grad/optimizer state 开关 | 中到极大 | 中 | 是否做训练峰值取决于目标 |
| async comm wait/phase boundary | 中 | 中 | 初版可保守近似 |
| checkpoint trace 校验 | 大 | 低 | 主要验证 use-def 是否自然表达 |

### P2：真实设备贴近度

| 项 | 影响 | 复杂度 | 说明 |
|---|---:|---:|---|
| allocator reserved/cache | 中 | 高 | 第一版不做 |
| fragmentation | 中 | 高 | 第一版不做 |
| kernel workspace | 小到中 | 高 | 当前需求可忽略 |
| alignment/padding | 小到中 | 低 | 后续可加简单 align |

## 9. 技术可达性

该方案不需要黑魔法，也不需要复刻 NPU allocator。核心依赖都是当前 simulator 已经有的信号：

- `TorchDispatchMode` 已经能捕获 op 输入输出和 seq_idx。
- `_producer` 已经能建立 tensor producer 边。
- `CommEvent` 已经有 comm primitive、shape/dtype/bytes、comm_layer、src/dst op。
- `ScheduleGraph` 已经有 L2 timeline 和 DataPass。
- 模型参数可以从 `model_parts` 静态遍历。

主要新增的是数据结构和离线 planner，不是新的执行路径。

## 10. 模块设计

建议新增：

```text
torchtitan_npu/simulator/memory/
├── __init__.py
├── tensor_record.py        # TensorRecord / MemoryEvent dataclass
├── parameter_snapshot.py   # local parameter shard bytes
├── raw_event_recorder.py   # raw tensor use-def extraction helpers
├── alias_rules.py          # alias / allocation / mutation op classification
├── fsdp_rules.py           # FSDP-aware lifetime rules
├── planner.py              # build_memory_plan(workload_graph, capture, ...)
└── export.py               # CSV/JSON summary exports
```

需要改动：

| 文件 | 改动 |
|---|---|
| `capture/dispatch_capture.py` | 保留未折叠 raw event stream；记录 tensor_id、producer、consumer、alias/mutation 信息 |
| `capture/tensor_utils.py` | 增加 tensor bytes helper 或直接复用现有 `tensor_volume_bytes` |
| `trainer.py` | 在 `build_nodes()` / `build_step_graphs()` 后调用 memory planner |
| `ir/tensor_meta.py` | 可选增加 `tensor_id`、`num_bytes`、`alias_of`；也可以先放在独立 memory record 中，避免扰动 IR |
| `ir/step_graph.py` | 填充 `peak_active_mem`、`param_mem`、`tensor_lifetimes` |
| `viz/text_summary.py` | 输出 `active_bytes_peak`，移除或改名 `total_peak_mem_bytes` |
| `viz/json_export.py` | 导出 memory plan |
| `viz/csv_export.py` | 导出 memory events / tensor lifetimes |
| `viz/html_export.py` | 可选展示 per-rank memory curve |
| `docs/user-guides/simulator.md` | 更新 `peak_mem` 字段说明 |

## 11. 工作量预估

按一个熟悉当前 simulator 的工程师估算：

| 阶段 | 内容 | 预估 |
|---|---|---:|
| 1 | 止血：改 summary 字段语义，新增文档说明 | 0.5-1 天 |
| 2 | 参数 shard bytes 统计与导出 | 1 天 |
| 3 | raw tensor use-def recorder 与基础 lifetime planner | 2-3 天 |
| 4 | alias/mutation 基础规则 | 1-2 天 |
| 5 | FSDP-aware allgather/reshard 生命周期 | 2-4 天 |
| 6 | memory events CSV/JSON + summary peak | 1-2 天 |
| 7 | microbatch/PP template 实例化 | 2-4 天 |
| 8 | 测试与 trace 校验 | 2-3 天 |
| 9 | HTML 曲线与用户文档 | 1-2 天 |

最小可信闭环：约 7-12 人日。

带 PP/microbatch 实例化和较完整导出：约 12-18 人日。

不建议第一阶段投入 allocator/cache/fragmentation/workspace，这些会显著增加复杂度，但对“合理有据”的 active tensor bytes 曲线不是必要条件。

## 12. 验证计划

### 12.1 单元测试

- 简单链：`a -> b -> c`，验证 b 在 c 消费后释放。
- dead output：无 consumer 的临时 output 在 producer 后释放。
- cross phase：forward output 被 backward op 消费，释放点在 backward。
- checkpoint-like：forward output 只在 forward 内消费，backward 重算生成新 tensor。
- alias：view/reshape 不新增 bytes，base lifetime 延长。
- mutation：in-place op 不新增 bytes。

### 12.2 FSDP 场景测试

- allgather output 被 compute 消费，形成 `fsdp_full_param`。
- full-param view 进入 matmul，不重复分配。
- reduce_scatter 后 full buffer 释放。
- local shard param step 常驻。
- 无法匹配的 FSDP event 进入 unclassified 报告。

### 12.3 集成测试

- `deepseek_v4_pro_simulate_16_layers`：验证基础 active peak 非零且 summary 不再出现 `sum(node.peak_mem)`。
- CP/EP 配置：验证 L1 comm buffer 进入 memory events。
- PP/FSDP 配置：验证 memory curve 随 microbatch timeline 实例化。

## 13. 输出示例

建议新增文件：

```text
simulator_output/<config>/
└── memory/
    ├── memory_summary.json
    ├── memory_events.csv
    ├── memory_timeline.csv
    ├── tensor_lifetimes.csv
    ├── memory_actions.csv
    ├── memory_trace.json
    └── unclassified_memory_ops.csv
```

`memory_events/rank_0.csv`：

```text
seq_idx,phase,op_id,event,tensor_id,kind,bytes,active_bytes_after,reason
0,init,-1,alloc,param:tok_embeddings.weight,parameter_shard,1048576,1048576,persistent_param
12,forward,45,alloc,t:45:out0,op_output,262144,1310720,producer
18,forward,52,free,t:45:out0,op_output,262144,1048576,last_consumer
30,forward,80,alloc,fsdp:80:full_param,fsdp_full_param,16777216,17825792,allgather
45,forward,97,free,fsdp:80:full_param,fsdp_full_param,16777216,1048576,reshard
```

`memory_summary.json`：

```json
{
  "metric": "active_tensor_bytes",
  "rank": 0,
  "persistent_param_bytes": 123456789,
  "active_bytes_peak": 234567890,
  "peak_seq_idx": 1234,
  "included": [
    "local parameter shards",
    "external inputs and labels",
    "op outputs by use-def",
    "comm buffers",
    "FSDP full parameter buffers"
  ],
  "excluded": [
    "allocator reserved/cache",
    "fragmentation",
    "kernel workspace",
    "device internal temporary buffers"
  ]
}
```

## 14. 结论

当前 `peak_mem` 的核心问题不是公式不够精细，而是计算位置错了：显存峰值不是单 op cost model 能给出的字段，而是需要基于执行序、tensor use-def、通信事件和 FSDP 语义离线扫描得到的 active set。

推荐路线：

1. 立即停止把 `sum(node.peak_mem)` 展示为 peak。
2. 保留现有折叠图用于展示，但新增未折叠 raw event stream 给 memory planner。
3. 第一版只承诺 `active_tensor_bytes`，不承诺真实 allocator peak。
4. P0 优先覆盖参数 shard、use-def、alias、FSDP allgather/reshard。
5. 后续再补 microbatch/PP 实例化、optimizer state、async comm 和 HTML 曲线。

这样可以在工程复杂度可控的前提下，得到一个合理、有据、能被 DES 消费的显存变化模型。

## 15. P0 开发计划

第一版只实现静态 active tensor bytes，不做 allocator/reserved/workspace。它必须能在单进程 meta simulator 和 multi-proc meta 每 rank 独立导出下稳定运行。

### 15.1 核心数据结构

```text
TensorRef:
  tensor_id: int
  shape: tuple[int, ...]
  dtype: str
  device: str
  num_bytes: int

RawMemoryEvent:
  event_id: int
  op_id: int
  seq_idx: int
  raw_op_type: str
  op_type: str
  phase: str
  module_path: str
  inputs: list[TensorRef]
  outputs: list[TensorRef]

TensorLifetime:
  tensor_id: str
  kind: parameter_shard | external_input | activation | temporary | dead_temp_output | alias | comm_buffer | fsdp_full_param
  num_bytes: int
  birth_seq: int
  death_seq: int
  producer_op: int
  consumer_ops: list[int]
  alias_of: str
  reason: str

MemoryTimelineEvent:
  seq_idx: int
  action: alloc | free
  tensor_id: str
  kind: str
  num_bytes: int
  active_bytes_after: int
  reason: str

MemoryPlan:
  persistent_param_bytes: int
  peak_active_bytes: int
  peak_seq_idx: int
  tensor_lifetimes: list[TensorLifetime]
  timeline_events: list[MemoryTimelineEvent]
  unclassified_ops: list[dict]
```

### 15.2 P0 算法伪代码

```text
raw_events = capture.memory_events()
params = snapshot_model_parameters(model_parts)
comm_by_op = {comm.op_id: comm for comm in comm_events}

for param in params:
  add persistent lifetime(step_start, step_end)

for event in raw_events ordered by seq_idx:
  for input in event.inputs:
    lifetime = find_or_create_external_input(input)
    lifetime.last_consumer = event.seq_idx
    lifetime.consumer_ops.add(event.op_id)

  for output in event.outputs:
    if output is known parameter or in-place mutation target:
      continue
    if alias_rules.is_alias(event):
      add zero-byte alias lifetime(alias_of=first_input)
      continue
    kind = classify_output(event, comm_by_op)
    add allocated lifetime(birth=event.seq_idx, kind=kind)

for lifetime in allocated_lifetimes:
  if no consumers:
    lifetime.death_seq = lifetime.birth_seq
    lifetime.kind = dead_temp_output unless comm/FSDP rule extends it
  else:
    lifetime.death_seq = max(consumer.seq_idx)
    if produced in forward and consumed in backward:
      lifetime.kind = activation
    elif kind was op_output:
      lifetime.kind = temporary

timeline = sweep alloc/free events ordered by seq_idx
peak = max(active_bytes_after)
```

### 15.3 第一版明确取舍

- 使用未折叠 raw event stream 算生命周期，保留现有折叠 L0 图用于展示。
- alias 只做白名单，不尝试从 storage 推断。
- mutation 只做 raw op 名称和输入输出 id 的保守识别。
- FSDP 先基于 `CommEvent` 分类 allgather/reduce_scatter buffer，并以 last consumer 释放；reshard hook 显式释放留到 P1。
- PP/microbatch 按 L2 `SchedulePlan` 实例化 MB1+ 的 memory events；L0/L1 展示图仍保持折叠。
- 输出名使用 `active_bytes_peak`，旧 `sum(node.peak_mem)` 在 summary 中改名为 `total_op_output_bytes_estimate`。

### 15.4 验收

- 单元测试覆盖简单链、dead output、cross phase activation、checkpoint-like 重算、alias、comm/FSDP 分类。
- smoke 脚本输出人可读日志：参数常驻、峰值、峰值位置、top lifetimes、导出路径。
- multi-proc meta 下每 rank 单独导出 `memory/`，PP 额外包含 `memory_actions.csv` 和 action span trace。
