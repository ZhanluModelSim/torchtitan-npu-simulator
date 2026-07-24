# 1F1B 完整依赖组装与 DES 回放实现指南

> 目标读者：负责继续实现 simulator L2/L3 与 DES 的 AI 或开发者。
>
> 本文只聚焦第一个可验收版本：`PipelineScheduleSingle` 的 1F1B，优先支持
> `PP > 1`、每个进程一个 PP stage、`B` 为完整 backward。不要在这一版同时扩展
> DualPipe、V schedule、I/W split、跨迭代流水或复杂 overlap。

如果下层已经生成 `SchedulePlan`，上层不需要理解本文的捕获和组装过程。请直接阅读
[`schedule-plan-dependency-reconstruction-contract.md`](./schedule-plan-dependency-reconstruction-contract.md)，
按现有 action、slot 和 communication descriptor 重建依赖。DES 状态机和内存消费的扩展说明见
[`schedule-plan-des-consumer-guide.md`](./schedule-plan-des-consumer-guide.md)。

开始实现前先阅读：

- `torchtitan_npu/simulator/capture/schedule_builder.py`
- `torchtitan_npu/simulator/capture/comm_events.py`
- `torchtitan_npu/simulator/capture/fsdp_residency.py`
- `torchtitan_npu/simulator/meta_env.py` 中的 PP context patches
- `torchtitan_npu/simulator/ir/schedule_plan.py`
- `torchtitan_npu/simulator/memory/schedule_replay.py`

## 1. 最终目标

从一次 meta capture 产生的信息中，构造一份可以独立驱动 DES 的 1F1B 计划。DES 必须能够：

1. 按每个 rank 的真实 1F1B 顺序发布 F、B、P2P、FSDP 和 optimizer action。
2. 根据真实数据依赖阻塞或激活 action，不产生无 producer 的内部依赖。
3. 正确配对跨 rank 的 SEND/RECV，允许 RECV 先发布并等待 SEND。
4. 正确区分 FSDP all-gather、full-parameter 释放和 gradient reduce-scatter。
5. 展开 `template_ref` 指向的 L1 StepGraph，计算每个 compute action 的耗时和静态输入输出内存。
6. 在事件队列为空但 step 未结束时，输出明确的死锁原因，而不是静默停止。

第一个版本的正确性标准不是与真实执行逐纳秒对齐，而是：action 顺序合理、依赖完备、
通信可推进、内存状态转换有来源，并且 PP/FSDP 规模变化符合常理。

## 2. 当前实现中必须先修正的问题

当前 `build_schedule_plan()` 对 `PipelineScheduleSingle` 使用
`timeline_events + comm_events` 合成 action。这个分支只能作为数据来源验证，不能直接作为最终实现。

### 2.1 reduce-scatter 被错误映射成 RESHARD

当前 fallback 把：

```text
allgather      -> UNSHARD
reduce_scatter -> RESHARD
```

第二条是错误的。PyTorch pipeline 语义是：

```text
UNSHARD     = all-gather full parameter
RESHARD     = 本地释放 full parameter，不发生 collective
REDUCE_GRAD = gradient reduce-scatter 或 all-reduce
```

因此不得使用“有没有 reduce-scatter CommEvent”判断 RESHARD 是否存在。

### 2.2 compute timeline 目前只有结束点

`forward_one_chunk`/`backward_one_chunk` 当前在函数返回后记录 timeline event。
FSDP unshard、reshard 和部分通信可能发生在 chunk 内部，仅按所有事件的 `seq_idx` 排序，
可能得到错误的 action 包含关系。

1F1B assembler 至少需要每个 compute instance 的：

```text
instance_id, rank, stage, mb_idx, comp_type, start_seq, end_seq, template_ref
```

建议保留当前结束事件以兼容导出，同时补充 begin/end 或一次带 start/end 的结构化记录。

### 2.3 P2P 缺少跨 rank 稳定标识

不同进程本地生成的 `slot_N` 不能用于跨 rank 配对。必须为同一次逻辑传输生成确定性的：

```text
transfer_id = (iteration, direction, src_stage, dst_stage, mb_idx, tensor_ordinal)
```

第一版每个 stage 边界、每个 microbatch、每个方向只有一个主 tensor 时，
`tensor_ordinal=0`。不要使用本地 action id、op id 或捕获顺序作为跨 rank 主键。

### 2.4 FSDP residency 缺少 action 归属

`FSDPResidencyEvent` 已记录 `group_id/action/seq_idx/num_bytes`，但完整组装还需要：

```text
rank, stage, mb_idx, comp_type, parent_compute_instance_id
```

这些字段应在 hook 触发时从 `_pp_context` 固化，不能在构图阶段读取已经变化的全局 context。

### 2.5 一个 UNSHARD 可能包含多个 FSDP group

stage 内可能有多个嵌套 FSDP group。一个 stage 只保存一个 `CommDetail` 会丢通信和内存信息。
第一版应选择以下一种显式表达，推荐前者：

1. 每个 `group_id` 建一个 UNSHARD/RESHARD action。
2. 一个聚合 action 持有 `comm_details[]` 和 `residency_transitions[]`。

不要只保留第一条 all-gather，其余静默丢弃。

### 2.6 capture process rank 不一定是逻辑 global rank

`multi_proc_meta` 可能只启动 PP 数量的真实 Gloo 进程，同时模拟更大的逻辑 world。
不得默认 `os process rank == simulated global rank`。结构化记录至少要区分：

```text
capture_process_rank
pp_stage
logical_rank or logical_rank_coordinates
```

第一版 P2P 配对以 `src_stage/dst_stage/mb/direction` 为稳定拓扑主键，再通过 RankTable
解析或展开逻辑 rank。不要使用 Gloo rank 推导 DP/TP/FSDP group。

## 3. 模块边界

不要继续在 `build_schedule_plan()` 中线性增加 1F1B 特判。使用 assembler 插件边界：

```text
SchedulePlanAssembler
  - RuntimePlanAssembler
      输入 pipeline_order_with_comms
      服务 DualPipe/ZBV 等 lowered schedule

  - SingleStageTraceAssembler
      输入 compute spans、P2P、FSDP residency、collectives
      服务 PipelineScheduleSingle: 1F1B/GPipe
```

公共层只负责：

- action/slot 注册与唯一 ID；
- producer/consumer 双向索引；
- plan invariant 校验；
- export；
- L1 template 查询。

1F1B assembler 不应修改 runtime-plan assembler 已验证的行为。`PP == 1` 继续走非 PP 路径，
除非有单独测试证明切换不会改变原有内存估计。

## 4. DES 所需的权威对象

DES 不能只读取 DataSlot，也不能只读取 action 列表。完整输入由以下部分组成。

### 4.1 ScheduleAction

Action 是可发布、可执行、可完成的主体：

```text
action_id
rank, stage, mb_idx
action_type: COMPUTE | SEND_F | RECV_F | SEND_B | RECV_B |
             UNSHARD | RESHARD | REDUCE_GRAD | OPTIMIZER
comp_type: F | B
issue_order
template_ref
consumes[]
produces[]
is_noop
```

`issue_order` 表示本 rank 的程序发布顺序，不等价于所有相邻 action 都存在数据依赖。

### 4.2 DataSlot

DataSlot 表示 action 完成后可用的数据或 readiness token：

```text
slot_id
kind
producer_action_id
consumer_action_ids[]
shape, dtype, volume_bytes
src_stage, dst_stage, mb_idx
lifetime/release_policy
```

第一版至少支持：

```text
dataloader_input
activation_local
activation_recv
grad_local
grad_recv
forward_state
param_full
control
grad_reduced
```

外部输入允许没有内部 producer，但必须显式标记 `external=True`。其他 consumed slot 必须有且只有一个 producer。

### 4.3 Communication descriptor

SEND/RECV 通过 `transfer_id` 配对，通信描述至少包含：

```text
transfer_id
role: send | recv
primitive
src_rank, dst_rank
src_stage, dst_stage
mb_idx
shape, dtype, volume_bytes
group/peer
comm_op_id
```

同一次传输只能计一次网络代价。可以让通信 matcher 创建一个共享 transfer event，
SEND/RECV 都等待该 event 完成；不能把两边各算一次完整通信。

### 4.4 L1 template

每个 `COMPUTE(stage, mb, F/B)` 必须解析到：

```text
s{stage}_F
s{stage}_B
```

DES 展开模板中的 L0 op 和内部 tensor 依赖。不同 microbatch 可以复用同一模板，
但每次实例化必须有独立 action instance、时间状态和非持久 tensor 标识。

## 5. 1F1B action skeleton 的构建

不要根据 1F1B 公式重新猜测本地顺序。捕获到的 compute span 和 P2P issue 顺序是权威来源。

### 5.1 创建 compute actions

对当前 rank 的每个 compute span：

```text
F(stage, mb) -> COMPUTE comp_type=F template=s{stage}_F
B(stage, mb) -> COMPUTE comp_type=B template=s{stage}_B
```

按 `start_seq` 确定 `issue_order`。必须验证每个本地 microbatch 恰好有一个 F 和一个 B。

### 5.2 创建 P2P actions

按 CommEvent 的 `p2p_direction` 创建：

```text
forward_send  -> SEND_F
forward_recv  -> RECV_F
backward_send -> SEND_B
backward_recv -> RECV_B
```

只接受以下四个精确值：`forward_send`、`forward_recv`、`backward_send`、
`backward_recv`。不能用简单的 `"send" in direction`，否则
`cp_forward_send/cp_backward_send` 会被误当成 PP action。

当前 PP P2P capture 不保证 `comm_layer == "L2"`，因此第一版不能仅靠该字段过滤。
应联合校验精确 direction、P2P primitive、stage、mb 和 peer；peer 的逻辑 rank 由 RankTable 解析。

各 rank 的理论边界如下，用于校验而不是生成顺序：

```text
非首 stage: RECV_F(m) -> F(m)
非末 stage: F(m) -> SEND_F(m)
非末 stage: RECV_B(m) -> B(m)
非首 stage: B(m) -> SEND_B(m)
```

### 5.3 创建 F 与 B 的同 microbatch 依赖

每个 `B(stage, mb)` 必须依赖同 stage 的 `F(stage, mb)` 已完成。建立零字节
`forward_state` 或 `control` slot：

```text
F(stage, mb) -> B(stage, mb)
```

这条依赖不能因为开启 full recompute 而删除。重计算改变 activation 的内存生命周期，
不改变 B 必须发生在原始 F 之后这一控制事实。

### 5.4 不要把整个 issue_order 变成依赖链

本 rank 的 issue cursor 保证 action 按程序顺序发布。只有真实的数据或完成条件才创建 slot。
否则会把异步通信和可重叠计算强制串行化。

## 6. P2P 依赖的正确表达

P2P 必须区分“action 可以发布”和“数据传输已经完成”。推荐的数据链为：

```text
发送端：
F(src,m) --activation_local--> SEND_F(transfer_id)
B(src,m) --grad_local-------> SEND_B(transfer_id)

接收端：
RECV_F(transfer_id) --activation_recv--> F(dst,m)
RECV_B(transfer_id) --grad_recv-------> B(dst,m)
```

SEND 与 RECV 的远端关系由 `transfer_id` 和 communication matcher 表达，不创建普通的
`SEND -> RECV` 开始依赖。

原因：真实系统允许 RECV 先发布并等待。如果 RECV 必须先消费 SEND 产出的普通 slot 才能发布，
某些 rank 顺序会在 rendezvous 之前互相等待，产生人为死锁。

DES 的通信推进逻辑：

```text
post SEND when local input slot is ready
post RECV when rank issue cursor reaches it
when SEND and RECV with same transfer_id are both posted:
    schedule one network transfer event
when transfer completes:
    mark SEND done
    mark RECV done
    publish RECV output slot
```

如果后端希望使用 eager-send 模型，也必须保留相同的 `transfer_id`，并保证 RECV 可以先挂起。

## 7. FSDP 状态和依赖

FSDP 以 residency interval 为权威，不要从算子名字猜测释放时间。

### 7.1 UNSHARD

对每个真实 `alloc` interval：

1. 找到同 group 的 all-gather CommEvent。
2. 创建 UNSHARD action，绑定 all-gather 的 `comm_op_id/shape/bytes/group`。
3. UNSHARD 完成后产生 `param_full(group_id)`。
4. interval 内使用该参数的 F/B action 消费 `param_full`。

捕获会折叠 MB1+ 的 L0 图，但 residency 仍逐 microbatch 保留。因此应按
`(stage, group_id, comp_type)` 复用首个已捕获 microbatch 的 all-gather L0 模板，
同时为每个 microbatch 保留独立 UNSHARD/RESHARD action。元数据推理阶段产生、且没有合法
`stage/mb/parent_compute_instance_id` 的 residency 不属于训练 schedule，应在 assembler 入口过滤；
若 stage/mb 合法但 parent compute 不存在，则必须报错。

若 stage 使用 FSDP 且 shard degree 大于 1，却找不到 all-gather，立即构图失败。
mesh size 为 1 时可以保留 `is_noop=True` action，但不得创建阻塞 slot。

### 7.2 RESHARD

对每个真实 `free` interval：

1. 创建 RESHARD action，不绑定 reduce-scatter。
2. 找到 interval 内最后一个使用该 full parameter 的 compute action。
3. 建立零字节 control slot：`last_compute -> RESHARD`。
4. RESHARD 完成时结束 `param_full` 的 residency，释放对应字节。

缺少配对 alloc、最后 consumer 或 group_id 时直接失败。不要创建 producer 为空的 slot。

### 7.3 REDUCE_GRAD

FSDP reduce-scatter 或 DDP all-reduce 应创建 REDUCE_GRAD：

```text
last relevant B -> REDUCE_GRAD -> grad_reduced -> OPTIMIZER
```

如果梯度通信在 B 内异步发起，使用捕获到的 parent compute 和 seq 确定发布位置；
依赖仍必须保证对应梯度已经产生。

## 8. optimizer 与 step 收尾

为捕获到的 optimizer template 创建一个 OPTIMIZER action：

- `template_ref` 指向 optimizer L1 graph；
- 消费本 rank 所有 `grad_reduced`；
- 没有 DP gradient collective 时，消费最后 B 产生的本地 grad readiness；
- 只能在所有本地 microbatch 的 backward 完成后执行。

第一版不要求完整表达下一 iteration 的 optimizer state，但必须保证 step 在 optimizer 完成后结束，
并且没有仍处于 active 状态的 full parameter residency 或未完成 P2P transfer。

## 9. Plan 校验器是必需功能

构建完成后、导出前运行统一 validator。至少检查：

### 9.1 引用完整性

- action id、slot id 全局唯一；
- 每个 `consumes/produces` 引用的 slot 存在；
- action 与 slot 两侧索引互相一致；
- 非 external consumed slot 恰好一个 producer；
- compute action 的 template 存在；
- 非 no-op 通信 action 的通信描述完整。

### 9.2 1F1B 结构

- 每个本地 `(stage, mb)` 恰好一个 F 和一个 B；
- `F(stage,mb)` 在 `B(stage,mb)` 之前；
- 首 stage 没有 RECV_F，末 stage 没有 SEND_F；
- 末 stage 没有 RECV_B，首 stage 没有 SEND_B；
- 每个 SEND/RECV 均能通过 transfer_id 在全 rank plan 中唯一配对。

### 9.3 FSDP

- real UNSHARD 有 all-gather；
- MB1+ 可复用同 group/comp_type 的 MB0 all-gather 模板，但 residency action 不折叠；
- real RESHARD 没有 collective，且有最后 compute producer；
- reduce-scatter 只归属 REDUCE_GRAD；
- no-op action 没有阻塞 slot；
- 每个 residency alloc 有且只有一个匹配 free，允许显式标记跨迭代常驻的例外。

### 9.4 图性质

- 普通数据和 control 边无环；
- P2P channel 单独由 communication matcher 校验，不参与 SEND->RECV 的开始依赖拓扑；
- 所有非 external action 从初始 ready 集合最终可达；
- 不允许通过“把未解析依赖当 ready”绕过错误。

## 10. DES 最小状态机

每个 action 至少具有：

```text
NOT_ISSUED -> POSTED/READY -> RUNNING/WAITING_COMM -> DONE
```

推进规则：

```text
for each rank:
    inspect next action by issue_order
    if action is RECV:
        post it even when remote SEND is absent
    elif all local consumed slots are ready:
        post it

for each posted action:
    noop:
        complete immediately
    compute:
        start when compute resource is free
    SEND/RECV:
        register with communication matcher
    UNSHARD/REDUCE_GRAD:
        start when communication resource and inputs are ready
    RESHARD:
        complete after its control dependency, then release param_full

on action completion:
    publish output slots
    release input slots whose last consumer completed
    advance dependent actions
```

注意：rank issue cursor 是否在 action POSTED 后前进，必须与捕获的 API 语义一致。
第一版可以采用保守策略：普通 compute 等待 DONE；异步 SEND/RECV 在成功 POSTED 后允许 cursor 前进。

### 死锁诊断

当 event queue 为空但仍有未完成 action 时，输出：

```text
rank cursor and next action
unresolved local slot ids and expected producers
posted but unmatched transfer_ids
active FSDP group_ids
resource owners
```

随后抛出异常。禁止把这些 action 自动标成完成。

## 11. 实现顺序

### P0.1 捕获契约

1. compute timeline 增加稳定 `instance_id/start_seq/end_seq`。
2. FSDP residency 增加 stage/mb/comp_type/parent instance。
3. P2P 生成确定性 transfer_id。
4. 保留旧字段和导出，避免破坏现有 runtime schedule。

### P0.2 独立 assembler

1. 增加 `SingleStageTraceAssembler`。
2. 先只生成 compute + P2P + F->B control。
3. 通过 PP=2、无 FSDP 的 validator 和 DES。
4. 再接入 FSDP residency、UNSHARD/RESHARD/REDUCE_GRAD。
5. 最后接 optimizer 和内存释放。

### P0.3 DES

1. 实现 rank issue cursor。
2. 实现 local slot readiness。
3. 实现基于 transfer_id 的 SEND/RECV matcher。
4. 实现 compute/comm 资源与 cost model duration。
5. 实现 no-op、RESHARD release 和死锁诊断。

不要在第一版实现：任意 overlap group、I/W split、V schedule 本地跨 stage、跨 iteration pipeline、
精确 allocator fragmentation。接口应允许后续扩展，但当前验收不依赖这些功能。

## 12. 验收矩阵

### Case A：PP=2，M=4，DP shard=1

每个 stage 必须有 `F x4 + B x4`。

stage 0：

```text
SEND_F x4
RECV_B x4
RECV_F x0
SEND_B x0
```

stage 1：

```text
RECV_F x4
SEND_B x4
SEND_F x0
RECV_B x0
```

全局有 4 个 forward transfer_id 和 4 个 backward transfer_id，全部唯一配对。
DES 能完成整个 step，无 dangling slot、unmatched transfer 或 active residency。

### Case B：PP=2，M=4，DP shard=2

在 Case A 基础上验证：

- all-gather 只属于 UNSHARD；
- reduce-scatter 只属于 REDUCE_GRAD；
- RESHARD 没有 comm_op_id，消费最后 compute 的 control slot；
- `param_full` bytes 在 alloc 后增加、free 后减少；
- persistent parameter bytes 约为 shard=1 的一半，full-param 峰值只在 residency interval 内出现。

### Case C：microbatch 对照

比较 M=2 与 M=4：

- action 和 P2P 数量按 microbatch 线性增长；
- persistent parameter bytes 不随 M 变化；
- 非重计算 activation residency 可以增长；
- full recompute 下 activation 峰值不应出现无界增长。

### Case D：非 PP 回归

`PP=1` 必须继续使用原路径，输出的 persistent params、peak memory 和 action 计数不能因
1F1B assembler 引入明显变化。

## 13. 交付要求

完成时应同时提交：

1. assembler 与 capture schema 改动；
2. plan validator；
3. DES SEND/RECV matcher 与死锁诊断；
4. 单元测试；
5. PP=2 的 shard=1/shard=2 可读日志；
6. 每个 rank 的 schedule plan，以及聚合后的 transfer 配对摘要；
7. memory timeline，至少展示 F/B 阶段、P2P、param_full alloc/free 和峰值。

最终日志至少包含：

```text
actions by type/rank/stage
slots by kind
paired/unmatched transfers
FSDP residency intervals
ready/running/waiting action counts
step completion time
peak memory and peak action
```

只有“所有 action 完成、所有 required transfer 配对、所有内部 consumed slot 可追溯、
所有非跨迭代 residency 已关闭”同时成立，才能认为一次 1F1B 仿真完成。
