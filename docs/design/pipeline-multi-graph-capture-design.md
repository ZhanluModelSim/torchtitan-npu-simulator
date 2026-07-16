# 复杂 Pipeline 策略下完整计算图捕获方案设计

> 分支：feat/npu-simulator
> 日期：2026-07-14
> 涉及文件：`meta_env.py`、`capture/dispatch_capture.py`、`capture/step_boundary.py`、`capture/schedule_builder.py`、`capture/comm_events.py`、`ir/step_graph.py`、`ir/schedule_graph.py`、`trainer.py`

## 1. 问题

用户反馈：`meta_env.py` 与 `dispatch_capture.py` 对复杂 Pipeline 策略的捕获不完整，具体表现为：

1. **只捕获 microbatch=0 的 L0 计算图**：MB 1+ 被整体跳过 L0 捕获。
2. **只按 forward/backward/optimizer 三分类**：无法区分复杂策略下 backward 的不同形态。
3. **不同 microbatch 间的 FSDP reshard 差异丢失**：MB 0 的图不能代表后续 MB 的图。

针对"复杂 Pipeline 策略"，用户特别指出两点结构性差异当前无法还原：
- backward 有 **I（input grad）与 W（weight grad）的区别**；
- 不同 microbatch 之间可能存在 **是否 fsdp reshard 的差异**。

## 2. 根因分析

### 2.1 复杂策略的真实计算流：`_ComputationType` 枚举

`torch.distributed.pipelining.schedules._ComputationType` 直接对应复杂策略（`ScheduleZBVZeroBubble` / `ScheduleDualPipeV` / `ScheduleInterleavedZeroBubble` / `ScheduleLoopedBFS`，均继承 `_PipelineScheduleRuntime`）的动作类型：

```python
class _ComputationType(str, Enum):
    FORWARD          = "F"   # forward_one_chunk
    FULL_BACKWARD    = "B"   # backward_one_chunk(full_backward=True)  → I+W 一次 autograd.backward
    BACKWARD_INPUT   = "I"   # backward_one_chunk(full_backward=False) → 仅 stage_backward_input（torch.autograd.grad 保留图）
    BACKWARD_WEIGHT  = "W"   # backward_weight_one_chunk              → stage_backward_weight（第二次 autograd.grad，仅权重）
    UNSHARD          = "UNSHARD"   # FSDPParamGroup.unshard (all_gather)
    RESHARD           = "RESHARD"    # FSDPParamGroup.reshard (reduce_scatter)
    SEND_F / RECV_F / SEND_B / RECV_B / REDUCE_GRAD ...
```

**关键发现**：用户说的"I 与 W 的区别"恰好就是 `BACKWARD_INPUT="I"` 与 `BACKWARD_WEIGHT="W"`。zero-bubble 类策略把一次 backward 拆成两次独立的 autograd 传递（`stage_backward_input` 用 `torch.autograd.grad` 只算输入梯度并保留图；`stage_backward_weight` 用第二次 `torch.autograd.grad` 只算权重梯度），产生**两个互相独立、拓扑不同的 L0 计算图**。`FULL_BACKWARD="B"` 则是 1F1B/GPipe 的单次 `torch.autograd.backward`（I+W 混在一张图里）。

`_PipelineScheduleRuntime._perform_action` 按 `pipeline_order_with_comms` 顺序对**每个 stage、每个 microbatch** 依次执行 F / I / W / B / UNSHARD / RESHARD / SEND / RECV，因此一个训练 step 内实际存在 `num_stages × {F, B or (I,W)} × num_microbatches` 个独立计算块，且 I/W/UNSHARD/RESHARD 在不同 MB 间的交错顺序由策略决定。

### 2.2 当前捕获的三处 gating/分桶缺陷

**(a) MB 1+ 的 L0 捕获被整体关闭** — `meta_env.py:1174-1178` / `1196-1200`：

```python
def _patched_fwd_one_chunk(self, mb_idx, *args, **kwargs):
    ...
    cap._capture_l0 = (mb_idx == 0)   # ← 只有 MB 0 进 L0 捕获
```

`dispatch_capture.py:148-149` 据此直接 `return`，MB 1+ 的全部算子（含零气泡策略下可能仅在后续 MB 才出现的某种 I/W 形态）都被丢弃。而 `comm_events` 对所有 MB 始终记录，导致 L2 timeline 与 L0 模板对不上。

**(b) phase 仅三值，backward 形态被合并** — `step_boundary.py:21` `_PHASES = ("forward","backward","optimizer")`，`build_step_graphs`（`step_boundary.py:88-101`）只按 `annotations["phase"]` 分桶。`FULL_BACKWARD`/`BACKWARD_INPUT`/`BACKWARD_WEIGHT` 三种拓扑不同的图全部塞进同一个 `"backward"` StepGraph，I/W 的边界彻底丢失。

**(c) 不同 stage 的同 phase 图被合并** — fake-PG 单进程下 `_patch_get_stage_indices_for_fake_pg`（`meta_env.py:1218`）把所有 PP stage 建在同一个进程里顺序执行，所有 stage 的 forward/backward 算子都流进 `OpDispatchCapture._events` 这一个扁平 list，再被 (b) 按 phase 合并 → stage0 与 stageN 的不同模型块图被混进同一张图。

**(d) FSDP reshard 状态不在分类键里** — `UNSHARD`/`RESHARD` 当前只被 `comm_events` 作为 L2 collective 记录（`_patch_comm_layer_context` 给 unshard/reshard 打 "L2" 标），但**不进入 L0 图的分类键**，所以"本次 forward 前是否刚 unshard、本次 backward 后是否 reshard"这种跨 MB 的状态差异不会生成不同的 L0 模板，也不会在 L1 层体现。

### 2.3 当前的"效率机制"与完整性的冲突

`dispatch_capture.py:182-184` 用"连续同 signature → `repeat_count++`"折叠重复算子，这是面向"同一张图内相邻重复层"的，不是面向"跨 MB 去重"。把 MB 1+ 全关掉（用牺牲完整性换效率）是过激的取舍：复杂策略下不同 MB 的图**大多结构相同**，只有少量形态（首/末 MB 边界、FSDP reshard 切换点、recompute）不同。正确做法是"按形态去重、每形态只捕获一次"，而非"只捕 MB 0"。

## 3. 方案设计

### 3.0 设计目标与约束

- **完整性**：训练 step 内出现的每一种 `(stage, 计算形态, FSDP状态, 边界)` 组合的计算图都被捕获至少一次，且每个 microbatch 的每个计算块都能在 L1/L2 被映射到正确的模板实例。
- **效率**：捕获开销 ∝ **不同形态数 × 单块算子数**，而非 `num_microbatches × 单块算子数`。复杂策略下不同形态数 ≪ `num_microbatches × num_stages`，因此 MB 1+ 的绝大多数计算块走 pass-through。
- **不破坏现有 L1/L2/L3 IR 形态**：`StepGraph`/`StepInstance`/`DataPass`/`TimelineEntry` 结构保留，仅扩充分类维度。

### 3.1 核心思路：用 `_ComputationType` 替换扁平 phase，按"每形态首现捕获"去重

把"图分类键"从 `phase ∈ {forward, backward, optimizer}` 升级为多维度 **GraphClassKey**：

```
GraphClassKey = (pp_stage, comp_type, fsdp_state, mb_boundary)
```

- `pp_stage`：该计算块所属 PP stage（解决 2.2(c)）。
- `comp_type` ∈ {`F`, `B`, `I`, `W`, `F_RECOMPUTE`, `OPTIMIZER`}（解决 2.2(b) + 用户的 I/W 诉求）。
  - `F_RECOMPUTE`：activation checkpointing 在 backward 阶段重算的 forward（autograd 上下文不同，是独立图）。
  - `OPTIMIZER`：沿用现有 `boundary.mark("optimizer")`。
- `fsdp_state` ∈ {`SHARDED`, `UNSHARDED`, `NA`}：本 stage 当前参数分片状态（解决 2.2(d) + 用户的 reshard 诉求）。**注意：这只影响 L0 模板是否含 unshard/reshard 集体通信算子，以及 L2 DataPass；同一 comp_type 在两种 fsdp_state 下的"纯计算算子"通常相同**，见 3.5 的取舍。
- `mb_boundary` ∈ {`FIRST`, `STEADY`, `LAST`}：仅对**同 comp_type 但首/末 MB 边界行为不同**的场景产生新模板（如 FSDP2 首次 forward 的 `_lazy_init`、末次 backward 的 `REDUCE_GRAD`）。

**捕获 gating 规则**（替换 `cap._capture_l0 = (mb_idx == 0)`）：

```python
key = (stage, comp_type, fsdp_state, mb_boundary)
should_capture = key not in self._captured_classes
```

- 首次出现 → 全量 L0 捕获，作为该 key 的模板，记入 `_captured_classes`。
- 再次出现 → pass-through（仅记 timeline + comm，不记 L0 算子），并把该次记为对应模板的一个 instance。

这样**每种形态恰好捕获一次**，既完整又高效。

### 3.2 I/W 区分的落地：`comp_type` 直接来自 schedule 动作

**复杂策略（`_PipelineScheduleRuntime`）**：patch `_PipelineScheduleRuntime._perform_action`，在分发前读 `action.computation_type` 与 `action.stage_index`，写入扩展后的 `_pp_context`：

```python
_pp_context = {
    "stage": 0, "mb_idx": 0,
    "comp_type": "F",        # 新增：F / B / I / W
    "fsdp_state": "UNSHARDED",  # 新增
    "chunk_id": 0,           # 新增：用于 chunk 边界
}
```

`_perform_action` 对 `UNSHARD`/`RESHARD` 不开 L0 捕获（它们是 L2 通信，由 `comm_events` 记录），但**更新对应 stage 的 `fsdp_state`**（见 3.5 状态机），使得紧随其后的 `FORWARD`/`BACKWARD*` 拿到正确的 fsdp_state 参与分类键。

**简单策略（`PipelineScheduleSingle`：1F1B / GPipe）**：不走 `_perform_action`，仍 hook `forward_one_chunk`/`backward_one_chunk`。`backward_one_chunk` 带 `full_backward: bool` 参数：
- `full_backward=True` → `comp_type = "B"`
- `full_backward=False` → `comp_type = "I"`（随后 `backward_weight_one_chunk` 被调用时设 `"W"`）

再 hook `backward_weight_one_chunk` 把 `comp_type` 切到 `"W"`。这覆盖所有策略类型，单一 patch 点集合。

> 说明：对 `FULL_BACKWARD="B"` 这种 I+W 混在一张图的情况，本设计在**模板/StepGraph 粒度**已用 `comp_type` 标注该图为"B（I+W 合一）"；若还需对 B 图内部**逐算子**标注 I 路径 vs W 路径，需要 autograd 节点级追溯（见 3.6 可选扩展）。零气泡策略下真正需要区分 I/W 的场景，schedule 本身已经把 I/W 拆成两次独立 autograd 传递，模板级 `comp_type=I` 与 `comp_type=W` 已天然分开，无需逐算子标注。

### 3.3 L0 捕获侧改动（`dispatch_capture.py`）

1. **`_RawEvent` 增字段**：`comp_type: str`、`fsdp_state: str`、`mb_boundary: str`（原 `phase` 保留向后兼容，由 `comp_type` 派生：`I/W/B → "backward"`）。
2. **chunk 级捕获 scope**：新增 `_chunk_events: list[_RawEvent]` 与 `_chunk_classes_seen: set[tuple]`。当前一个 `_events` 扁平 list 改为"按 chunk 收集 → chunk 结束时按 class_key 决定提交/丢弃"：
   - `_begin_chunk(key)`：若 key 未见过 → `_capture_mode="capture"`；否则 `"passthrough"`。
   - `_record_event`：`passthrough` 模式下直接 return（同现状的 `_capture_l0=False`）；`capture` 模式写入 `_chunk_events`，**chunk 内仍用现有连续 signature 折叠 repeat_count**（同一 chunk 内的重复层仍折叠，保效率）。
   - `_end_chunk(key)`：`capture` 模式 → 把 `_chunk_events` 提交为一个新 `StepGraph` 模板（`step_type = comp_type`），清空；`passthrough` 模式 → 丢弃 `_chunk_events`，给对应模板 `instance_count += 1`。
3. **op 的 stage/mb 归因**：仍从 `_pp_context` 读 `stage`/`mb_idx` 写入 `_RawEvent.pp_stage`/`pp_mb_idx`（已有），保证 L2 timeline 全量。

`build_nodes` 改为返回 `dict[str, StepGraph]`（key = class_key 的字符串化，如 `stage0_F_UNSHARDED_FIRST`），而非按 phase 的三桶。

### 3.4 FSDP reshard 状态机（`meta_env.py` + `comm_events.py`）

维护 `_fsdp_state: dict[int, str]`（stage → `SHARDED`/`UNSHARDED`）：

- `_perform_action(UNSHARD, stage)` 或 `FSDPParamGroup.unshard` patch → `_fsdp_state[stage] = "UNSHARDED"`，并保证随后该 stage 的 `FORWARD`/`BACKWARD*` 读到 `UNSHARDED`。
- `_perform_action(RESHARD, stage)` 或 `FSDPParamGroup.reshard` patch → `_fsdp_state[stage] = "SHARDED"`。
- 初始 `SHARDED`（FSDP2 默认参数分片）。无 FSDP 时 `NA`。

**取舍**：把 `fsdp_state` 放进 class_key 会在"参数 reshard 了但纯计算算子不变"时生成两个 L0 模板。这通常是期望行为——因为 unshard/reshard 集体通信算子会进 L0 图（`comm.*` 节点），导致两模板拓扑确实不同（一个带 allgather、一个带 reduce_scatter）。若发现纯计算部分完全相同、只想在 L2 DataPass 层区分，可把 `fsdp_state` 从 L0 class_key 移出、仅用于 L2 `StepInstance` 标注（见 3.7 兜底）。

### 3.5 L1/L2 装配改动

**`step_boundary.py`**：`_PHASES` 扩为动态集合；`build_step_graphs` 改为按 `class_key`（或等价的 `(stage, comp_type)`）分桶，而非 `phase`。`step_type` 用 `comp_type`（`F`/`B`/`I`/`W`/`F_RECOMPUTE`/`OPTIMIZER`）+ stage。

**`schedule_builder.py`**：
- `StepInstance` 改为 per `(microbatch, stage, comp_type)`：`instance_id = f"rank{rank}_s{stage}_mb{mb}_{comp_type}"`，`micro_batch_idx=mb`，`pipeline_stage=stage`，新增 `comp_type`/`fsdp_state` 字段。
- 全量 `execution_timeline`：对**每个**计算块（含 MB 1+）都产出一条 `TimelineEntry`（seq 来自捕获、mb_idx/stage/comp_type 来自 `_pp_context`/timeline_events），不再只对 MB 0。
- `DataPass`：P2P/FSDP 通信按 `(src_stage, src_mb, dst_stage, dst_mb)` 连接正确的 StepInstance，而非笼统 `rank{stage}`。`src_exit_op`/`dst_entry_op` 需要在"该 MB 的 chunk 内"反查 producer——见 3.7 的 chunk 局部 producer 表。

**`ir/schedule_graph.py`**：`StepInstance` 增 `comp_type`/`fsdp_state`；`TimelineEntry` 增 `comp_type`；`StepGraph.step_type` 语义改为 comp_type。

### 3.6 可选扩展：B 图内部逐算子 I/W 归因

若需要对 `FULL_BACKWARD` 模板内部逐算子标注 I 路径/W 路径：在 chunk 捕获期间记录每个 leaf 的 `id()`——参数 leaf → W 汇聚点，输入 leaf → I 汇聚点。backward 算子产出张量若链路可达参数 leaf 记 `"W"`、可达输入 leaf 记 `"I"`、两者皆可达记 `"I+W"`。实现上需要在 chunk 结束时跑一次轻量反向 BFS（沿 `_producer` 图从两类 leaf 出发打标），开销 O(单块算子数)。本设计将其列为可选，因为零气泡场景下 I/W 已在模板级天然分离。

### 3.7 chunk 局部 producer 表与跨 chunk 依赖

当前 `_producer: dict[id(tensor), op_id]` 是 step 级全局表。改为 **per-chunk 局部表** `_chunk_producer`，chunk 结束随模板提交。跨 chunk 的依赖（如 backward 引用 forward 保存的激活、I 块引用 W 块的中间节点）通过**外部 predecessor**表达（`step_graph.py:_compute_entry_exit` 已支持"前驱不在本图内即视为外部依赖"，见 `step_graph.py:17-28` 注释），无需跨 chunk 维护全局 id 映射。`src_exit_op`/`dst_entry_op` 在 chunk 内反查 `comm.*` 算子的 producer/consumer 时也用局部表，保证 DataPass 连到正确 chunk 的边界算子。

## 4. 改动清单（按文件）

| 文件 | 改动 |
|------|------|
| `meta_env.py` | (1) `_pp_context` 增 `comp_type`/`fsdp_state`/`chunk_id`；(2) 新增 patch `_perform_action`（runtime 策略）写 comp_type + 驱动 fsdp 状态机；(3) `_patch_pipeline_stage_for_pp_context` 的 `forward_one_chunk`/`backward_one_chunk` 改用 class_key gating 替换 `mb_idx==0`，并读 `full_backward` 区分 B/I；(4) 新增 patch `backward_weight_one_chunk` 设 `comp_type="W"`；(5) `_patch_comm_layer_context` 的 unshard/reshard patch 同步更新 `_fsdp_state`；(6) `_patch_get_stage_indices_for_fake_pg` 已建全部 stage，无需改。 |
| `capture/dispatch_capture.py` | (1) `_RawEvent` 增 `comp_type`/`fsdp_state`/`mb_boundary`；(2) 把扁平 `_events` + `_capture_l0` bool 改为 chunk scope（`_begin_chunk`/`_end_chunk` + `_chunk_classes_seen`），首现捕获、重复 pass-through；(3) `build_nodes` → 返回 `dict[class_key_str, StepGraph]`。 |
| `capture/step_boundary.py` | `_PHASES` 动态化；`build_step_graphs` 按 `(stage, comp_type)` 分桶。 |
| `capture/comm_events.py` | CommEvent 增 `comp_type`/`fsdp_state`（从 `_pp_context` 读），便于 L2 按块连接 DataPass。 |
| `capture/schedule_builder.py` | StepInstance per `(mb, stage, comp_type)`；全量 timeline；DataPass 连到具体 StepInstance（带 mb_idx）。 |
| `ir/schedule_graph.py` / `ir/step_graph.py` | `StepInstance`/`TimelineEntry` 增 `comp_type`/`fsdp_state`；`step_type` 语义改 comp_type。 |
| `trainer.py` | `run_simulation_step` 透传 `comp_type` 维度；`num_micro_batches` 已有；导出（CSV/JSON/HTML）按 class_key 多模板输出。 |
| viz 导出 | per-stage×comp_type 的 L0 CSV 文件名（如 `stage0_I_l0_ops.csv`、`stage0_W_l0_ops.csv`）。 |

## 5. 效率分析

设 `S` = PP stage 数，`M` = microbatch 数，`C` = 单块平均算子数。

- **当前**：L0 捕获 ≈ `S × C`（仅 MB 0 全 stage），但丢失 MB 1+ 的形态差异；timeline 记 `S×M` 块但无对应 L0。
- **朴素全捕**：`S × M × C`，完整但 `M` 倍开销。
- **本方案**：`D × C`，其中 `D = 不同 class_key 数 ≤ S × {F, B or (I,W), F_RECOMPUTE?} × {SHARDED, UNSHARDED} × {FIRST, STEADY, LAST}`。实际 D ≪ S×M（典型 D ≈ S×2~4），与朴素全捕相比降到 `D/(S×M) ≈ 2/M ~ 4/M`。对 M=4 的 DSV4 约 1/4 ~ 1/2 开销，但换得全量形态。

L1/L2 装配开销仍 ∝ timeline 事件数（`S×M`），但这是轻量 dict 操作，远低于 L0 dispatch 拦截开销。

## 6. 验证方案

1. **单元**：构造 fake schedule（CSV 形式 `pipeline_parallel_schedule_csv`）含 `F`/`I`/`W`/`UNSHARD`/`RESHARD` 交错，断言每个 `(stage, comp_type)` 都产出独立 StepGraph 模板，且 `instance_count` 之和 = `S×M`。
2. **端到端（容器 `titan-npu-sim-e2e`）**：用 DSV4 `ZBVZeroBubble` 配置跑一次，检查输出：
   - `ir_export/` 下出现 `stage{s}_{F|I|W|B}_l0_ops.csv`，而非单一 `backward`；
   - `l1_schedule.csv` 每个 MB 的每个计算块都有 comp_type 标注，timeline 行数 ≈ `S×M`；
   - I 模板的 exit 节点 = P2P send 的源（input grad），W 模板不含 P2P send、其 exit 指向 optimizer 入口。
3. **回归**：`GPipe`/`1F1B`（仅 F/B/O）下输出退化为 3 类 × S 模板，与现状等价（仅多了 per-stage 分桶），不破坏既有对照基线。
