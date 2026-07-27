# DualPipeV SchedulePlan 下游接入指南

> 目标读者：消费 simulator L2 `SchedulePlan`、负责训练 step 组装和 DES 的开发者或 agent。
>
> 本文只描述 DualPipeV 相比普通 1F1B/GPipe 新增的消费规则。通用依赖、P2P、collective
> 和 logical world 展开规则以
> [`schedule-plan-dependency-reconstruction-contract.md`](./schedule-plan-dependency-reconstruction-contract.md)
> 为准。

## 1. 接入边界

DualPipeV 不要求下游根据 schedule 名称重新生成流水计划。捕获端已经输出：

- 每个物理 PP 进程的顶层 action 发布顺序；
- 同一物理 rank 上多个虚拟 stage 的 compute action；
- F/B overlap parent 及其两个 compute sub-action；
- stage 间本地 DataSlot 和跨 rank P2P `transfer_id`；
- 每个 compute sub-action 对应的 L1 `StepGraph`；
- FSDP UNSHARD/RESHARD/REDUCE_GRAD action 和依赖。

下游仍只需要实现“按顺序发布、按依赖就绪、按资源执行”。`pipeline_schedule` 仅用于日志和
验收，不能成为硬编码 DualPipeV 时序的开关。

如果从 CSV 调试导出读取，`schedule_plan.csv` 包含两个连续 section：

```text
action header + action rows
slot_id,kind,... header + DataSlot rows
```

遇到 `slot_id,kind,...` 后必须切换到 DataSlot schema，不能继续按 action header 解析。生产接入
仍应直接消费进程内 `SchedulePlan` 对象。

## 2. 虚拟 stage 与物理 rank

PP degree 为 `p` 时，DualPipeV 通常有 `2p` 个虚拟 stage。一份 rank-local plan 可能拥有两个
不相邻的 stage。例如 PP=2：

| capture process | 虚拟 stage |
|---|---|
| rank 0 | 0、3 |
| rank 1 | 1、2 |

因此：

1. action 的归属 rank 使用 `plan.annotations["capture_process_rank"]`；
2. `action.stage` 只表示模型虚拟 stage，不能用来推导物理 rank；
3. logical world clone 仍按 PP mesh 坐标选择 capture plan；
4. 同 rank 相邻虚拟 stage 之间使用 `DataSlot.is_local_transfer=True`，不创建网络事件；
5. 跨 rank stage 之间按 `comm.transfer_id` 配对 SEND/RECV。

## 3. Overlap action

`OVERLAP_F_B` 是顶层发布单元，`sub_actions` 是实际 F/B compute：

```text
OVERLAP_F_B parent
  sub_actions[0] -> COMPUTE(F or B)
  sub_actions[1] -> COMPUTE(B or F)
```

消费端必须：

1. 仅将 parent 放入 rank issue queue；
2. 将 parent 和所有 sub-action 都加入 `action_id` 索引；
3. 在 sub-action 上解析 `consumes`、`produces` 和 `template_ref`；
4. 每个 sub-action 独立判断输入 readiness；
5. 所有 sub-action 完成后，parent 才完成并推进顶层 issue cursor；
6. 禁止再把 sub-action 当作顶层 action 发布，否则会重复执行。

DataSlot 可以直接引用 sub-action ID。不能把依赖提升到 parent 后丢弃原始 child 引用。

## 4. 后端暂不支持双图并发时

允许先将 overlap 串行执行，但应把 parent 实现为“单资源复合组”，而不是合并成一个不可见的
compute action：

```text
on parent issued:
    pending = sub_actions in captured order
    while pending:
        child = first ready child
        if no child is ready:
            wait for dependency or communication completion
        execute child exclusively
        publish this child's output slots immediately
        remove child from pending
    complete parent
```

关键约束：

- 不要等待两个 child 的输入全部就绪后才启动 parent；第一个 child 的输出可能推动远端事件，
  进而使第二个 child 就绪，预先等待并集可能制造死锁。
- child 串行时仍分别实例化各自的 `StepGraph`，不能只保留其中一张图。
- child 完成后立即发布其输出，不要延迟到 parent 完成。
- 多个 child 同时 ready 时使用 `sub_actions` 记录顺序，保证确定性。
- 串行模式只改变资源重叠，不改变 DataSlot、P2P 和 collective 依赖。

这种模式可以正确完成 step，但时间与峰值内存是保守近似：没有计算重叠，两个 child 的临时
tensor 也不会同时驻留。输出中应标记 `dualpipe_overlap_mode=serialized`，避免将结果误认为真实
DualPipeV 并发性能。

## 5. P2P 与本地 V 转移

对每个跨 rank transfer：

```text
key = comm.transfer_id
require exactly one SEND
require exactly one RECV
require same src_stage, dst_stage, mb_idx and volume_bytes
```

同 rank 的 V 形 stage 转移没有 SEND/RECV：

```text
producer compute
  -> DataSlot(is_local_transfer=True)
  -> consumer compute
```

不要因为看不到通信 action 就补一个虚假 P2P，也不要把 local transfer 当 external input。

## 6. FSDP 与内存

DualPipeV 与 1F1B 使用相同的通信归属契约：

```text
stage 内触发的 all-gather/reduce-scatter -> COMPUTE 的 L1 StepGraph
stage 外显式 prefetch -> UNSHARD -> COMPUTE -> RESHARD
stage 外真实梯度归约 -> BACKWARD -> REDUCE_GRAD -> OPTIMIZER
```

下游必须完整回放 `action.template_ref` 指向的图，不能再根据 FSDP 配置向 L2
补通信。只有 `communication_owner=L2_PREFETCH/L2_STANDALONE` 的 action
需要由 L2 单独调度。详细规则见
[`communication-ownership-contract.md`](./communication-ownership-contract.md)。

同一个 virtual stage 的相邻 F/B/I/W action 之间可能发生 FSDP 跨 action
prefetch。例如 B 或 W 尾部发起 all-gather，为稍后的 F 准备参数。该通信已经归入实际发起
它的 L1 模板，并标记为 `ownership_placement=cross_action_prefetch`；目标 F 模板不会重复包含
它。下游仍按原有规则执行，不需要新增边：

1. 完整执行 launch action 引用的 L1 图，包括该 all-gather；
2. all-gather 作为 L1 图出口时，launch action 要等它完成；
3. 继续遵守 rank-local 顶层 `schedule_order`；
4. 不要因为目标 F 中没有相同 all-gather 而在 L2 补通信。

层内 prefetch 则直接表现为 L1 DAG 中两个可并行分支：目标层 all-gather 和来源层 compute
共享前置 readiness，目标层 compute 等待 all-gather。即使后端暂时把 DualPipeV 的双图
overlap 串行化，也必须保留单张 L1 图内部的通信/计算并行关系。

一个物理 rank 拥有多个虚拟 stage 时，参数基线和 FSDP residency 都按物理 rank 汇总。小模型中
embedding/output 可能远大于 transformer layer，因此 V 形切分可能产生明显 rank 间显存不均；
不能据此把较小 rank 的参数补齐到平均值。

串行 overlap 的内存回放应遵循实际采用的 child 执行顺序。它能保持生命周期和量级正确，但不
表达两个 compute graph 真正并发时的瞬时峰值。

## 7. 接入验收

组装前至少检查：

1. 每个顶层 `schedule_order` 唯一且非负；
2. parent 不在任何其他 parent 下，sub-action 不进入顶层 issue queue；
3. 每个 overlap parent 恰有两个有效 compute child；
4. child 的 `template_ref`、consumes 和 produces 全部可解析；
5. 所有非 external consumed slot 都有有效 producer；
6. 所有跨 rank `transfer_id` 恰有一对 SEND/RECV；
7. local transfer 只有 DataSlot，没有 P2P endpoint；
8. 每个 rank 最终完成全部 parent、child、通信和 FSDP action；
9. 无法推进时打印 pending parent、child readiness、缺失 slot 和未配对 transfer。
10. L1 中不存在 `sim.fsdp_*`，且 `cross_action_prefetch` 未在目标 action 重复。

PP=2 的最小验收应覆盖：

- rank0 拥有 stage 0 和 3，rank1 拥有 stage 1 和 2；
- 至少一个 overlap parent 在串行模式下能完成两个 child；
- 前向 activation 和反向 gradient 同时覆盖跨 rank与同 rank local transfer；
- 开启 DP shard 后，外部 prefetch 的 UNSHARD/RESHARD 成对；stage 内 FSDP
  通信位于 L1，且两者都不改变 PP transfer 配对。
