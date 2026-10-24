# WorkloadGraph 的流控制依赖

当前捕获将 Meta 设备上的原生 `wait_event`、`wait_stream` 转换为零耗时的 `stream_wait` 控制节点，并用前驱边保留同一流上的设备工作顺序。该机制补齐 FSDP 中独立于权重／梯度 tensor 数据流的等待，不需要新增配置。

## 上游调度约定

遍历全部 `workload.step_templates`，建立跨模板的全局节点索引。调度任一节点前，必须等待其全部 `predecessors` 完成；通信节点结束表示通信完成，不是 Python launch 返回。

`stream_wait` 使用现有 `OpNode`，合同如下：

| 字段 | 含义 |
| --- | --- |
| `op_type` | `stream_wait` |
| `annotations.control_node` | `true`，控制节点 |
| `annotations.zero_duration` | `true`，执行时长为零；不查询硬件算子性能表 |
| `annotations.stream_id` | 本次捕获内的目标流 ID，不是硬件 stream handle |
| `attrs.kind` | `wait_event` 或 `wait_stream` |
| `attrs.source_stream_id` | 事件记录／等待源流的捕获内 ID；捕获前的事件可为空 |
| `predecessors` | 目标流先前工作与记录时源流工作；即使零耗时也必须全部完成 |
| `inputs`／`outputs` | 均为空；不生成占位 tensor |
| FLOPs／显存／通信字节 | 均为零；不新增内存生命周期事件 |

例如 `RS → AR → stream_wait → MM`，将 AR 延后会延后 MM。未等待此事件的另一条流没有这条控制约束。已有 tensor producer、alias 和 storage mutation 数据边继续保留，不能用流边替换。

`annotations.stream_predecessors` 和 `event_predecessors` 记录原始 `op_id`，供审计使用；调度使用正式的 `predecessors`。当前组装不克隆节点或重新编号，不再需要 `source_op_id` 映射。

## 捕获语义

- 每条流保留已提交设备工作的执行前沿；后续工作连接该流前沿，而不是把所有流串行化。
- `Event.record` 保存当时源流的前沿。之后源流再提交的工作不自动加入该事件；同一事件重新记录也不改变已经生成的等待节点。
- `wait_stream` 同样只等待调用时已经提交的源流工作。
- `Event.wait` 转入相应目标流的 `wait_event`。捕获前或上一轮捕获的事件不引用旧图节点；事件只弱引用捕获对象。
- 只改元数据的 view／alias 节点保留数据边，但不推进设备流前沿，避免将异步预取 buffer 的视图变成意外屏障。
- `Tensor.record_stream` 表示 allocator 生命周期，不作为执行等待。

本轮覆盖 FSDP 的异步流等待。`query`、`elapsed_time` 和 host `synchronize` 仍是 Meta 协议占位行为，不表达实际设备进度；不能据此推广为任意基于轮询或 host 同步的程序均已支持。真实 NPU/HCCL 时序与性能未验证，PP 训练入口继续拒绝。

## 图折叠与文件输出

当前已移除 metadata-view 折叠、FSDP 通信模板变体、预取依赖推断、节点恢复补偿及旧 PP 通信 fragment 图改写。WorkloadGraph 直接保留原始 view／alias、通信和控制节点；文件输出仍只有 mem。清理范围与验证见 [精简记录](simulator-cleanup.md)。

零耗时不意味着可以忽略节点依赖。上游不得直接删除控制节点或其前驱边，否则会重新出现归约未完成而后续计算提前执行的问题。

## 验证

独立测试见 [stream 单测](../tests/capture/test_stream_dependencies.py)：通信延后传播、未等待分支重叠、事件快照／重新记录、同流顺序、跨捕获隔离，以及元数据视图不会引入屏障。

原生 HSDP 最终图测试见 [HSDP 等待回归](../tests/test_fsdp_stream_capture.py)：观察实际 `post_backward` 的 `_post_reduce_event` 等待，检查控制边经过模板组装仍保留，并验证 AR 延后会延后同一目标流后续的 MM。测试使用 Meta 和 fake process group，不是 NPU ST。

本轮四配置与流探针的执行结果补充到 [依赖审计](fsdp-dependency-audit.md)。
