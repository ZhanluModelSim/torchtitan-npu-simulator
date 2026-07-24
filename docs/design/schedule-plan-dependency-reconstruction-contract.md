# SchedulePlan 下游依赖重建契约

> 目标读者：拿到 simulator L2 输出、负责组装全局训练 step 或实现 DES 的开发者和 agent。
>
> 本文是 `capture_schema_version = 2` 的下游消费契约。它只说明输入中已经有什么、
> 如何建立依赖和通信事件，以及哪些情况必须拒绝；不讨论这些数据如何捕获。

## 1. 重建目标

下游需要把每个 capture process 导出的 rank-local `SchedulePlan` 组装成一个可推进的全局
训练 step。最终结果必须同时表达：

1. 每个 rank 上 action 的发布顺序；
2. action 之间由 tensor 或 readiness token 形成的本地依赖；
3. PP SEND/RECV 之间的跨 rank rendezvous；
4. collective 的参与组和共享完成事件；
5. FSDP full parameter 的驻留和释放；
6. compute action 对 L1 `StepGraph` 的实例化；
7. 无法推进时可定位到具体 action、slot 或通信的死锁信息。

`SchedulePlan` 已经给出依赖事实。下游的职责是实例化和连接这些事实，而不是根据
`1F1B`、`GPipe` 或某个 TorchTitan schedule 的公式重新猜一份计划。

## 2. 权威输入和版本检查

每个 capture process 提供一份：

```text
SchedulePlan
  actions[]                 rank-local action stream
  data_slots{}              rank-local producer/consumer dependencies
  step_templates{}          L1 compute/communication templates
  annotations.rank_table    complete logical mesh description
```

开始组装前必须检查：

```text
plan.annotations["capture_schema_version"] == 2
action.annotations["capture_schema_version"] == 2
plan.annotations["capture_process_rank"] is present
```

不支持的版本必须显式失败，不能退化为按 CSV 行号或 `seq_idx` 猜依赖。

进程内集成应直接消费 `SchedulePlan` 对象。`schedule_plan.csv` 是可读导出，只适合调试；
如果必须解析 CSV，列表、布尔值和 annotations 必须使用结构化解析，不能按显示字符串做模糊匹配。

## 3. 四种关系必须分开

### 3.1 发布顺序

同一 plan 的顶层 action 按以下字段排序：

```text
schedule_order
```

schema v2 中，顶层 `schedule_order` 必须非负且在 plan 内唯一。它是 rank-local 语义发布顺序；
`seq_idx` 只用于定位 L0 捕获位置和诊断，不参与依赖或顺序重建。禁止：

- 跨 rank 比较 `schedule_order`；
- 使用 `seq_idx` 构造全局时间；
- 因为两个 action 相邻就自动添加数据依赖。

发布顺序控制 rank 的 issue cursor。它不是 DAG 边。

### 3.2 本地数据依赖

唯一权威来源是 `DataSlot`：

```text
producer_action_id -> slot_id -> consumer_action_ids[]
```

一个 consumer 只有在其全部 `consumes` slot ready 后才满足本地输入条件。slot 的 `kind` 用于
解释数据和内存，不用于重新推断 producer。

### 3.3 跨 rank P2P

PP SEND/RECV 通过以下字段配对：

```text
action.comm.transfer_id
```

SEND 和 RECV 之间没有普通 `DataSlot` 边。RECV 可以先发布并等待远端 SEND。禁止使用
`action_id`、`slot_id`、`seq_idx`、`comm_op_id` 或进程到达顺序替代 `transfer_id`。

### 3.4 资源约束

compute stream、communication stream、link、collective group 等资源约束由 DES 建模。资源边
不是 `SchedulePlan` 的数据边，不应写回 `DataSlot` DAG。

## 4. ID 和命名空间

### 4.1 原始代表 rank 模式

若 DES 只仿真每个 PP stage 的一个代表 rank，原始 ID 可以直接使用：

```text
action_key   = (capture_process_rank, action_id)
slot_key     = (capture_process_rank, slot_id)
transfer_key = transfer_id
```

即使 `action_id`/`slot_id` 当前带有 rank 前缀，也建议消费端保留 tuple 命名空间，不依赖字符串
格式。

### 4.2 完整 logical world 模式

真实 capture process 数通常等于 PP degree，但 `rank_table.world_size` 可能更大。此时每个
capture process 代表完整 logical mesh 中一个 PP stage 的模板：

```text
capture_process_rank: 真实 Gloo 抓取进程，也是该 plan 的 PP mesh 坐标
logical_global_rank:   完整 mesh 中的目标逻辑设备，由下游展开产生
stage:                 模型虚拟 pipeline stage，V schedule 中不一定等于 PP mesh 坐标
```

展开完整 world 时：

1. 在 `rank_table.rank_coordinates` 中选择 `pp == capture_process_rank` 的 logical rank；
2. 为每个目标 logical rank 克隆 action、slot 和 template instance；
3. 将 action/slot ID 重写到 logical-rank 命名空间；
4. 同步重写 `consumes`、`produces`、producer 和 consumer 引用；
5. collective 选择包含该 logical rank 的 `comm_group_ranks`；
6. PP 只连接非 PP 坐标相同的相邻虚拟 stage。

不能按 `action.stage` 选择 logical rank。一份 V schedule plan 可能同时包含 stage 0 和 stage 3，
但它们都属于同一个 capture process 和 PP mesh 坐标。

直接读取 Python `RankTable` 时，`rank_coordinates` 的 key 是整数；从 annotation/JSON 读取时 key
可能是数字字符串。消费端应在入口规范化一次，不要在匹配过程中同时维护两种 key。

克隆后的 P2P 键必须增加 lane 身份：

```text
logical_transfer_key = (
    base_transfer_id,
    coordinates_except_pp,
)
```

也可以使用 `(base_transfer_id, src_logical_rank, dst_logical_rank)`。同一次 DES 不得混用
代表 rank 模式和完整 world 模式。

## 5. 第一遍：建立本地 action/slot 图

对每份 plan 独立执行。

### 5.1 展开 action 索引

建立：

```text
action_map[action_key] -> ScheduleAction
issue_queue[rank]       -> top-level actions sorted by schedule_order
```

`OVERLAP_F_B` 是一个顶层调度单元，`sub_actions` 是其内部 F/B compute。slot 可能连接到
sub-action，因此 `action_map` 必须递归包含 sub-action。

消费端有两种合法实现：

- 将 parent 作为并发容器，分别执行 sub-action，所有 sub-action 完成后 parent 完成；
- 将 parent 作为一个复合事件，但其 readiness 必须是全部 sub-action 输入的并集，完成后必须
  发布全部 sub-action 输出。

不能忽略 sub-action 上的 `consumes`/`produces`。

### 5.2 验证并连接 DataSlot

对每个 `DataSlot`：

```text
if external:
    require producer_action_id == ""
    initial_ready.add(slot_key)
else if consumer_action_ids is not empty:
    require producer_action_id exists

if producer_action_id:
    require producer exists
    require slot_id in producer.produces

for consumer_id in consumer_action_ids:
    require consumer exists
    require slot_id in consumer.consumes
    add edge producer -> consumer when producer exists
```

同时反向检查每个 action 的 `consumes` 和 `produces` 都能在 `data_slots` 中找到。

以下情况必须报错：

- 非 external 的 consumed slot 没有 producer；
- producer/consumer action 不存在；
- action 与 slot 的双向引用不一致；
- 一个 slot 被两个 action 声称为 producer；
- `is_noop=True` 的 action 带有 blocking slot。

不要通过“producer 不存在则标记 ready”来兜底。

### 5.3 Slot 类型只决定语义

常见类型：

| kind | 语义 |
|---|---|
| `dataloader_input` | 首 stage 的外部输入 |
| `loss_grad` | 真正末 stage 的外部 loss 梯度 |
| `activation` | forward stage 间激活，可能是 P2P 或同 rank local transfer |
| `grad_input` | backward stage 间输入梯度 |
| `forward_state` | 同一 `(stage, mb)` 的 F 到 B/I/W readiness |
| `param_full` | UNSHARD 后的完整参数 |
| `grad_local` | 本地 backward 产生的梯度 readiness |
| `grad_reduced` | gradient collective 完成后的 readiness |
| `control` | 零字节 action-completion token |

`is_local_transfer=True` 表示相邻虚拟 stage 位于同一 rank，没有网络事件；它仍是一条普通本地
DataSlot 依赖。

典型连接如下。这些关系应已经体现在 slot 中，只用于下游验收，不能在缺失时自行补边：

```text
F(src, mb)       -> activation -> SEND_F
RECV_F           -> activation -> F(dst, mb)
I/B(src, mb)     -> grad_input -> SEND_B
RECV_B           -> grad_input -> I/B(dst, mb)
F(stage, mb)     -> forward_state -> B/I/W(stage, mb)
F(stage, mb)     -> local activation -> F(next virtual stage, mb)
I/B(stage, mb)   -> local grad_input -> I/B(previous virtual stage, mb)
B/W              -> grad_local -> REDUCE_GRAD -> grad_reduced -> OPTIMIZER
```

## 6. 第二遍：建立跨 rank PP rendezvous

收集所有计划中的：

```text
SEND_F, RECV_F, SEND_B, RECV_B
```

按 `transfer_key` 建立：

```text
transfers[key].send
transfers[key].recv
```

启动 DES 前验证：

```text
exactly one SEND endpoint
exactly one RECV endpoint
same src_stage/dst_stage
same mb_idx
same volume_bytes, shape and dtype when both sides provide them
SEND.comm.role == "send"
RECV.comm.role == "recv"
```

一次 transfer 只创建一个共享网络事件：

```text
SEND local input ready
        \
         +-- both endpoints posted --> one network event --> SEND done
RECV posted
                                                \-------> RECV done
                                                          publish recv slots
```

注意：

- SEND 必须先等本地 activation/grad slot ready；
- RECV 到达 issue cursor 后即可 posted，不等待远端 slot；
- 两端都 posted 后才可启动网络传输；
- 网络传输只计算一次 cost；
- SEND 与 RECV 各自设备上的 buffer residency 可以分别计入内存。

`transfer_id` 当前语义格式为：

```text
pp:{forward|backward}:s{src}->s{dst}:mb{microbatch}:t{tensor_ordinal}
```

下游应把它当作 opaque stable ID，不能依赖字符串解析实现正确性。

## 7. 第三遍：建立 collective 和 FSDP 状态

### 7.1 Collective

`UNSHARD` 和 `REDUCE_GRAD` 的 `comm.role == "collective"`。完整 logical world 模式下，为同一
collective 的所有 endpoint 创建一个共享事件。至少按以下信息区分实例：

```text
primitive
normalized comm_group_ranks
logical lane / group
stage
schedule action instance
fsdp_transition_id or fsdp_group_id when present
```

不能只按 `primitive + group` 合并，因为同一 group 在一个 step 内会多次执行 collective。
也不能用 `comm_op_id` 作为实例 ID；折叠的 L0 模板可被多个 microbatch/action 复用。

若当前输入只包含一个代表 endpoint，DES 可以使用 group size 估算该 action 的通信 cost，但必须
明确处于代表 rank 模式，不能等待并不存在的 logical-rank endpoint。

`action.comm` 是通信量、peer 和 group 的直接描述。若 cost model 还需要 L0 节点，可调用：

```python
op = plan.find_op_node(action.comm_op_id)
```

`comm_op_id == 0` 表示没有捕获到实际 L0 通信，合法情况必须同时具有明确的 no-op 语义。

### 7.2 FSDP residency

FSDP 状态身份优先使用：

```text
fsdp_transition_id
fsdp_group_id
stage
logical_rank
```

状态转换：

```text
UNSHARD collective done
    -> param_full slot ready
    -> full parameter residency opens

owning compute done
    -> control slot ready

RESHARD done
    -> full parameter residency closes
```

`RESHARD` 是本地释放，不是 reduce-scatter。梯度 reduce-scatter/all-reduce 由
`REDUCE_GRAD` 表达。

`is_noop=True` 表示该 schedule intent 没有实际通信或驻留变化：

- 不创建 collective event；
- 不等待 DataSlot；
- 到达 issue cursor 后立即完成；
- 不得借用相邻 action 的通信补齐。

若一个非 no-op UNSHARD 缺少 collective 描述，或 RESHARD 没有可关闭的 active residency，
必须报错。

## 8. Compute、I/W、重计算和虚拟 stage

### 8.1 Compute 实例

每个 `COMPUTE` action 通过：

```text
plan.step_templates[action.template_ref]
```

实例化 L1 `StepGraph`。同一模板可被多个 microbatch 复用，但以下状态必须独立：

```text
template_instance_id
L0 op state
temporary tensor instance
start/end time
memory lifetime
```

`template_ref` 相同不代表 action 或 tensor 相同。

### 8.2 不根据 comp_type 猜边

`F/B/I/W/F_RECOMPUTE` 只用于选择模板和解释阶段。真正依赖仍来自 DataSlot。因此：

- split backward 的 `I`、`W` 按各自 slot 执行；
- `forward_state` 可以连接到 B、I 或 W；
- recompute action 是否消费/产生 activation 以它的 slot 为准；
- optimizer 等待哪些 backward/reduce action，以 `consumes` 为准。

禁止硬编码 `F -> B` 后忽略 I/W，也禁止把 schedule 中相邻的 F_RECOMPUTE 和 B 自动合并。

### 8.3 虚拟 stage

一个 rank 可以拥有多个不相邻的 stage，例如 ZeroBubble V schedule。必须始终使用
`action.stage`，不能使用 `action.rank` 推导 stage。

同 rank 相邻 stage 的 forward/backward 传递由 `is_local_transfer=True` 的 DataSlot 表达，
不创建 SEND/RECV；跨 rank stage 才使用 P2P rendezvous。真正的最后 stage 是计划中 stage
拓扑的最大 stage，不一定是 `capture_process_rank == pp_degree - 1`。

### 8.4 其他 action

`OPTIMIZER`、`LOSS`、`LR_SCHEDULER` 等本地 action 同样由 `consumes/produces` 决定 readiness。
有 `template_ref` 时实例化对应模板；没有模板时由已注册的 action executor 处理。遇到未知
`action_type` 必须要求消费端插件显式注册，不能默认按零时长 action 完成。

## 9. DES 最小状态机

推荐 action 状态：

```text
NOT_ISSUED -> POSTED -> RUNNING | WAITING_COMM -> DONE
```

全局状态至少包括：

```text
issue_cursor[rank]
action_state[action_key]
ready_slots
posted_transfers
collective_instances
event_queue
resource_state
active_fsdp_residency
remaining_slot_consumers
```

最小推进算法：

```text
validate every plan
build local slot graph
build P2P matcher
build collective instances
mark external slots ready

while unfinished actions exist:
    progressed = false

    for each rank:
        action = issue_queue[rank][issue_cursor[rank]]

        if action is RECV:
            post recv endpoint
            advance issue cursor
            progressed = true

        else if all action.consumes are ready:
            if action.is_noop:
                complete action immediately
            else if action is SEND:
                post send endpoint
                advance issue cursor
            else:
                schedule local, compute, or collective event
            progressed = true

    for each P2P transfer whose SEND and RECV are posted:
        schedule exactly one network event

    for each collective whose required endpoints are posted:
        schedule exactly one collective event

    if event_queue is not empty:
        pop earliest event
        complete action or shared communication endpoints
        publish every produced slot
        update FSDP residency and memory
        advance cursor for synchronous actions
        progressed = true

    if not progressed:
        report structural/resource deadlock and fail
```

第一版只有 P2P endpoint 明确允许 posted 后推进 cursor。普通 compute、RESHARD、optimizer 和
没有显式 launch/wait 拆分的 collective，应在 DONE 后推进 cursor，避免凭空制造异步重叠。

## 10. 内存生命周期

依赖图也提供静态 tensor 生命周期：

1. persistent parameter 是 action 外的 step 初始基线；
2. 普通 slot 在 producer 完成时分配；
3. external slot 在 step 初始化时存在；
4. slot 在最后一个 consumer **完成**后释放，不是在 consumer 发布或开始时释放；
5. `param_full` 在 UNSHARD 完成后增加，在匹配 RESHARD 完成后释放；
6. `control`、`forward_state` 等零字节 slot 只影响 readiness；
7. SEND buffer 和 RECV buffer 属于不同 logical rank，不能合并；
8. 每个 microbatch/template instance 的临时 tensor 必须独立。

若一个 slot 没有 consumer，下游可以在 producer 完成后立即释放，但应先确认它不是：

- external output；
- persistent/cross-step state；
- FSDP residency；
- 由专门 memory annotation 管理的状态。

第一版只计算静态基线、L1 输入输出、DataSlot 和 FSDP residency，不要求模拟 allocator
fragmentation 或未捕获的算子中间激活。

## 11. 必须执行的校验

### 11.1 单 plan

- action ID 唯一，包括 sub-action；
- 顶层 `schedule_order` 非负且唯一；
- action/slot 双向引用一致；
- 每个非 external consumed slot 有 producer；
- no-op action 没有 blocking slot；
- real UNSHARD 有 all-gather；
- RESHARD 不携带网络 collective；
- real REDUCE_GRAD 有 collective；
- 本地 DataSlot 图无环。

### 11.2 全局

- 每个 required P2P key 恰好一个 SEND 和一个 RECV；
- P2P 两端拓扑、microbatch 和 tensor metadata 一致；
- logical-world 模式下每个 collective endpoint 数量等于 group size；
- 每个 FSDP residency 有且只有一次 open/close；
- 所有 external slot 只位于合法 step 边界；
- 所有 action 最终可达。

仓库中的零时长参考检查：

```python
validate_schedule_plan(plan, strict_1f1b=True)
validate_1f1b_transfer_pairs(all_rank_plans)
replay_1f1b_readiness(all_rank_plans)
```

这些函数验证结构可推进，不替代带资源和 cost 的正式 DES。

## 12. 禁止的兜底

遇到缺失数据时，不允许：

- 用 `seq_idx` 代替缺失的 `schedule_order` 或跨 rank identity；
- 把 producer 缺失的内部 slot 自动标记 ready；
- 把未配对 RECV 自动完成；
- 为找不到通信的 UNSHARD 借用“最近的”all-gather；
- 把缺失 collective 的 action 自动改为 no-op；
- 默认 `capture_process_rank == logical_global_rank == stage`；
- 因为 action 相邻就补一条控制边；
- 在 event queue 为空时把剩余 action 批量完成。

以上情况都应 fail fast，并打印输入身份和缺失关系。

## 13. 死锁报告最低要求

无法推进时至少输出：

```text
schema version and plan ids
each rank's issue cursor and next action
missing slot ids for each blocked action
producer and state of each missing slot
posted but unmatched transfer keys and endpoint roles
waiting collective key and missing logical ranks
active FSDP residency and expected RESHARD
running events and occupied resources
event queue snapshot
```

这样可以区分：

- L2 输入缺边；
- 跨 rank plan 缺失；
- logical-rank 展开错误；
- 通信 matcher 错误；
- 资源模型造成的真实死锁。

## 14. 完成判据

一次 step 只有在以下条件全部满足时完成：

```text
all issue cursors reach queue end
all non-noop actions are DONE
all required P2P transfers are paired and DONE
all required collectives are DONE
all internal consumed slots were produced
all nonpersistent FSDP residency is closed
optimizer/LR scheduler terminal actions are DONE
event queue and waiting communication sets are empty
```

建议输出以下摘要用于验收：

```text
actions by logical rank / stage / action_type
slots by kind and total bytes
P2P paired/unmatched count
collective count by primitive/group
FSDP residency intervals
step completion time
per-rank peak memory and peak action
```

只按 `schedule_order` 逐行播放、但没有 DataSlot readiness、P2P rendezvous、collective 和 FSDP
状态机的实现，不算完整依赖重建。
