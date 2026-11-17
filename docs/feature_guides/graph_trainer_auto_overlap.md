# GraphTrainer NPU Auto Overlap

NPU Auto Overlap 用于 TorchTitan GraphTrainer 的双 chunk MoE 通信掩盖自动调度。它在 NPU 上测量计算和通信耗时，再以依赖安全、跨 rank 一致为约束重排图，使双 chunk 的计算和通信互相掩盖。

该特性复用原生 EP overlap 的切图、process-group 隔离、结构校验和符号 shape 处理，将 `ep_overlap_schedule_pass` 的固定调度替换为基于实测 cost 的自动调度。使用时需要：

1. 设置 `TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER=1`；
2. 设置 `TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER=0`；
3. 开启 EP overlap，并选择对应的 Auto Overlap pipeline。

完整配置参考 [`deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer.sh`](../../examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer.sh)：该脚本同时设置 EP overlap、两个 Inductor benchmark 环境变量和 NPU auto overlap pipeline。

## 1. 整体流程

### 1.1 与原生 EP overlap 的关系

原生 EP overlap 将目标 MoE 区域切成两个 chunk，并以固定规则重排节点。NPU Auto Overlap 保留切图结果，等动态 shape 具体化后再根据 NPU 实测 cost 调度：

```text
原生 EP overlap
  chunk metadata -> EP chunk -> 固定调度 -> shape 具体化 -> 后续编译

NPU Auto Overlap
  chunk metadata -> EP chunk -> EP 结构校验 -> shape 具体化 -> 自动调度 -> 后续编译
```

启用时，pipeline 组合逻辑执行以下操作：

1. 从完整的原生 GraphTrainer pipeline 开始。
2. 将原生 EP 调度 pass 替换为结构校验的 `ep_overlap_validate_pass`，不应用原生固定调度。
3. 在 EP chunk shape 具体化之后插入 NPU Auto Overlap pass。
4. 保持其他 pass 的内容和相对顺序不变。

如果必要的 EP pass 缺失、目标 pass 重复，或者当前正在加载预编译产物，则不改写 pipeline。

### 1.2 pass 流程

当前流程包含 1 次单算子建模和 2 轮整图校准：

```text
单计算/通信算子 benchmark
  -> 第 1 轮调度
  -> 整图 benchmark
  -> 第 2 轮调度
  -> 整图 benchmark
  -> 第 3 轮调度
```

第一次调度在原始 FX 图的 MoE 区域内逐算子 benchmark，获取初始 cost model 并执行自动调度。之后，每轮都执行上一轮排好的整图，通过 profiling 采集训练场景下通算并发时的算子耗时，校准 cost model，再次执行自动调度。

跨轮次保留三类信息：

- 上一轮生成的图顺序，作为下一轮的输入；
- 原始图节点的 canonical 顺序，作为各轮确定性排序和通信算子编号的稳定基准；
- 最近一次可用的节点 cost，用于整图测量缺失时回退。

整图校准需要当前 traced graph、模型、batch 和训练上下文。runtime context 不可用时，只执行首次单算子 benchmark 和调度。

## 2. 单算子 Benchmark

### 2.1 MoE 动态 shape 处理

原生 runtime estimator 可以从 FakeTensor metadata 为普通静态 shape 算子构造输入，但 MoE 的 routed-token 维度由当步路由结果决定，常以动态 shape 出现在 FX 图中。直接沿用原生物化逻辑通常有两种结果：

- 动态维度无法物化，算子不能 benchmark；
- 使用统一的符号 fallback 值，会使 dispatch、combine、GMM 和 permutation 的 workload 彼此不匹配。

Auto Overlap 采用负载均衡假设：保留可确定的总 token 数，并将 routed tokens 尽量均匀分配给 experts 和 EP ranks，使同一 chunk 内的通信和计算看到一致、合法的 workload。

### 2.2 计算算子

普通计算节点从 FakeTensor metadata 恢复 shape、stride、dtype、device 和非 tensor 参数，并生成相同布局的随机 tensor。只有纯函数、输入 metadata 完整且 tensor 值不决定算子合法性的节点才使用通用物化；包含 index、mask、offset 或 split 信息的整数/布尔 tensor 不会被任意随机化。

MoE 中两类动态算子使用专门输入构造：

| 算子 | 合成 workload |
| --- | --- |
| Grouped MM | 从同 chunk 的 dispatch 推导 routed-token 总量，均匀生成每个 expert 的 token count 和累计 offset；覆盖 forward、dgrad 和 wgrad 形态 |
| Permute / unpermute | 根据 routed-token 总量、top-k 和 expert 数构造均匀 routing；unpermute 使用匹配 NPU permute 返回语义的 inverse indices |

只改变 shape、stride、view 或 alias 等 tensor metadata 的节点按零 cost 处理。无法安全物化或 benchmark 失败的计算节点也回退为零 cost，并打印告警；整图校准有机会在后续轮次补上真实测量。

计算算子通过 NPU profiler benchmark backend 多次执行并得到稳定耗时。cache key 包含算子、tensor shape/stride/dtype/device、非 tensor 参数以及动态 workload 假设，因此相同 workload 可以复用结果，不同布局或路由规模不会错误共享 cost。

### 2.3 通信算子

为每个通信 launch 找到对应 wait，并以 launch 到完成的耗时作为通信 cost。单算子通信 benchmark 优先使用 CANN profiler；当前 CANN 或 `torch_npu` 无法导出、关联 HCOM 活动时，回退到 NPU event 计时。

A2A 的输入在通信组内共同构造：

1. 每个 rank 提供本地 routed-token 总量。
2. 通信组内交换这些总量，并将每行尽量均匀分给目标 ranks，形成兼容的 split matrix。
3. dispatch 使用该矩阵的本地发送行和接收列。
4. combine 反转与配对 dispatch 的输入、输出 splits，使其 workload 保持一致。

同一 chunk 的 dispatch token 总量也用于 GMM 和 permutation，避免各算子独立固化动态 shape 后得到互相矛盾的 workload。

CANN 路径会先按 workload 去重和查询缓存，再逐个测量缺失的通信签名。每次样本取通信组内最慢 rank，最后对多次样本取中位数，从而表示一次通信的全局完成时间。

### 2.4 单算子 cost 的跨 rank 对齐

首轮 benchmark 基于尚未调度的 MoE 区域，各 rank 具有相同的可 benchmark 节点数量和 canonical 顺序。因此计算、通信 cost 可以在每个 region 内按位置交换，并对各 rank 的值取中位数。

## 3. 整图 Benchmark

### 3.1 为什么需要整图校准

单算子 benchmark 提供稳定、可启动调度的初始 cost，但它不能反映训练中算子的上下文和并发环境。

整图 benchmark 直接执行上一轮排好的 GraphTrainer 图，因而能够观察：

- 计算、HCCL 通信和 copy 并发时的实际设备耗时；
- 当前节点顺序、stream dependency 和资源竞争带来的影响；
- 前一轮 schedule 改变后，各节点 cost 随执行上下文发生的变化。

这些 cost 不是单算子的硬件延迟，而是当前训练 schedule 中的设备侧测量，更适合用来校准下一轮调度。

### 3.2 校准执行

每轮校准复用模型和首个 batch，不额外消耗 dataloader 数据。执行前复制输入 tensor，执行后恢复模型 buffer 和 CPU/NPU RNG 状态；不修改 parameter 或 gradient。

在整图 profiling 采集前，各卡会对进行同步并 warmup ，之后再执行被测整图。

### 3.3 CANN cost 获取

整图 profiling 不向 FX 图插入逐节点 marker，而是：

1. 按 FX 执行顺序匹配 CANN 中已有的 CPU dispatcher scope；
2. 沿 `torch_to_npu` flow 找到该 scope 下发的设备任务；
3. 计算节点保留 kernel 和 copy，过滤 host enqueue/dequeue 及设备控制事件；
4. 通信节点使用对应 HCOM 活动的完成耗时。

### 3.4 cost 跨 rank 对齐

非均匀切分（例如切分 DeepSeekV4 的 MHC）可能使各 rank 的 FX 图存在差异，无法采用按节点位置顺序对齐。为此，每个本地节点使用不依赖本地编号的语义 key，信息包含：

- 计算节点使用所在 MoE region、前向/反向、chunk、module scope、算子类型、输入/输出 tensor metadata，以及相同签名的出现次序；
- 通信节点额外使用逻辑通信组序号、通信类型和 count/dispatch/combine 角色。

各 rank 交换“语义 key -> cost”映射。只有所有 ranks 都存在的 key 才采用本轮整图 cost，并取各 rank 的最小值；仅部分 ranks 测得的节点统一回退到上一轮，避免下一轮调度因 cost 来源不同而分叉。通信执行前还会校验每个逻辑通信组中的通信类型和数量，防止不一致的通信图进入校准执行。

## 4. 自动调度

### 4.1 调度范围与 cost model

调度仅针对双 chunk MoE 区域。调度范围包括两个 chunk body，以及同一 MoE 内会影响 overlap 的共享计算，例如未归入任一 chunk 的权重转换。

当前调度模型基于以下假设：

- 计算节点在一条有序计算时间线上执行；
- MoE 区域内的 EP 通信共享一条有序通信资源；
- 通信下发后，后续计算消耗其尚未覆盖的时间；
- 执行 wait 时仍未覆盖的通信尾部计为暴露时间。

当前跨 rank cost 对齐和调度投票使用 WORLD，因此参与调度的图通信组必须覆盖全部 rank，多个 EP subgroup 或 PP stage 暂不支持。

模型估算的隐藏时间和暴露时间用于比较候选顺序，不等同于整图实测耗时。copy、AI Core 和 HCCL 之间更细粒度的资源竞争由整图校准间接反馈。

### 4.2 依赖与安全约束

调度前会完成以下处理：

1. 收集调度区域内的通信 launch，并关联所有对应 wait。
2. 从完整 FX 图向调度区域投影依赖，保留“离开调度区域后又重新进入”的间接路径，避免局部合法顺序在全图形成环。
3. 补充动态 split-list host 协议无法由 tensor edge 完整表达的依赖。
4. 基于这些依赖维护可执行节点集合，并逐步生成新的 region 顺序。
5. 将多个调度区域的顺序约束合入全图；发生冲突时保留最大无环子集。
6. 完成稳定拓扑排序后，再校验通信顺序及其跨 rank 一致性。

### 4.3 Token-count D2H 优化

原始两个 chunk 各自包含一次异步 D2H 和一次同步 D2H：

```text
chunk 0: count0 -> D2H0 async -> D2H1 sync -> dispatch0
chunk 1: count1 -> D2H2 async -> D2H3 sync -> dispatch1
```

Auto Overlap 将四次 copy 按稳定顺序下发，只保留最后一次 host 同步：

```text
count0 -> count1 -> D2H0/D2H1/D2H2 async -> D2H3 sync -> dispatch0/1
```

为保证正确性，读取 host 数据的消费者都依赖最终同步点，而两个 dispatch 只依赖各自的 split-list 物化路径。token-count 通信不会和相邻通信交换；其 wait 进入数据通信关键路径后也会被优先完成。

### 4.4 节点选择优先级

主循环优先保持通信连续下发，同时用不会延迟下一次下发的计算填充空闲窗口。

准备下一次通信时，调度器会比较原始顺序中相邻的两个候选通信。只有在两者属于同一通信组、彼此无依赖、准备路径安全且都不是 token-count 通信时，才允许交换。

候选顺序的 score 考虑：

- 下发前的准备工作是否造成通信资源空闲；
- 第一段通信还有多少尾部无法被准备工作和可执行计算覆盖；
- 第一段通信完成后会解锁多少后续计算；
- 避免重复计算两个候选共同的准备工作。

只有所有 ranks 都认为交换合法、且跨 rank 平均收益为正，才接受交换，保证通信顺序相同。

确定好下个通信后，按如下优先级下发算子：

| 优先级 | 场景 | 处理 |
| ---: | --- | --- |
| 1 | 下一个通信已可执行 | 立即下发 |
| 2 | 通信正在执行，下一次通信下发尚未进入准备阶段 | 在预留准备预算后，用能够放入剩余窗口的独立计算填充 |
| 3 | 下一次下发仍有可执行的非 wait 前置节点 | 按稳定顺序推进关键准备路径 |
| 4 | token-count wait 阻塞数据通信准备 | 尽快完成 host 协议，不把它当作普通长通信窗口 |
| 5 | 已无待下发通信，但仍有通信在执行 | 执行较大的就绪计算，覆盖剩余通信时间 |
| 6 | 下一通个信依赖尚未完成的通信结果 | 在执行可能阻塞的 wait 前，先用就绪计算掩盖通信尾部 |
| 7 | 没有需要优先处理的通信 | 按稳定顺序执行节点，继续推进图 |
| 8 | 只剩 wait | 完成最早的 wait |
| 9 | 防御性兜底 | 选择稳定顺序最早的就绪节点 |

当调度器已经选择某个通信后，会持续推进它的输入准备直到下发，避免准备过程中反复改选目标。

### 4.5 跨 Rank 一致性

调度器通过以下机制避免 ranks 之间的通信顺序分叉：

- 调度前比较各 rank 的通信清单，包括通信组内序号、算子类型与角色，以及所属 MoE；
- 通信交换由所有 ranks 共同决策；
- 多个节点在依赖和调度优先级上等价时，按首次调度前的 canonical 顺序确定先后；
- 调度后按通信组校验最终 launch-order signature。

非均匀切分可以使计算节点和 wait 的具体位置不同，但同一通信组内的通信 launch 顺序必须一致。

## 5. 调试与实现索引

设置 `--compile.debug-graph-passes` 后，除原生 graph pass 调试信息外，auto overlap 还会：

- 保留内部 CANN profiling 目录，位于 `$TMPDIR` 或系统临时目录；
- 保存自动调度前后的可读 FX 图，位于 `./fx_graphs/` 目录；
- 日志打印单算子本地/对齐 cost、整图应用 cost 和逐步调度决策。

相关代码：

| 文件 | 作用 |
| --- | --- |
| `auto_overlap.py` | pipeline 注册、调度与整图校准流程编排 |
| `npu_moe_auto_scheduler.py` | 调度区域收集、节点重排和安全校验 |
| `compute_benchmark.py` | 计算节点输入物化与单算子 benchmark |
| `collective_benchmark.py` | 通信输入物化、CANN benchmark 和 event fallback |
| `whole_graph_benchmark.py` | 整图校准执行、CANN cost 获取和对齐 |
| `utils.py` | 设置 profiling 工作目录、FX 图保存和通用方法 |

## 相关文档

- 原生 EP overlap 与 NPU 适配：[`graph_trainer_ep_overlap.md`](graph_trainer_ep_overlap.md)
- DeepSeek-V4 GraphTrainer 编译路径：[`deepseek_v4_graph_trainer.md`](deepseek_v4_graph_trainer.md)
