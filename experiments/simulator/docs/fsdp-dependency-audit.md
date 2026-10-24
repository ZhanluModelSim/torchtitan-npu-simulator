# FSDP 依赖审计与流控制修复

流控制缺口已在当前实现补齐：原生 `wait_event`／`wait_stream` 生成零耗时控制节点，同流设备工作保留顺序。上游必须执行完整前驱依赖，接口见 [流控制依赖](stream-control-dependencies.md)。下文保留修复前审计与反例，节点 ID 仅适用于各自捕获。

后续已删除 FSDP 模板折叠、推断补边和恢复补偿，并再次复验同样的四配置，见 [精简记录](simulator-cleanup.md)。下方节点 ID 与数字保留对应历史快照，不跨捕获复用。

## 修复与复验

本轮只在 sidecar 的 Meta stream/event 协议与 dispatch 捕获层增加控制语义；模型、通信 API、梯度和 tensor 身份保持原路径。没有恢复图折叠或额外图导出。

| 配置 | 修复后图节点 | 零耗时控制节点 | AG 权重读取检查 | AG 延后检查区间 |
| --- | ---: | ---: | ---: | ---: |
| FSDP | 7,311 | 102 | 263 | 17 |
| HSDP | 7,333 | 114 | 263 | 17 |
| 组合并行 | 9,506 | 173 | 263 | 27 |
| HSDP + SAC + MXFP8 | 11,293 | 114 | 297 | 17 |

四配置沿用下方表格的 world／rank、MTP 和序列长度。1,086 次参数读取的 AG 祖先关系、78 个区间的 AG 延后传播、35 对 RS→AR，以及归约→clip／每个 optimizer 的依赖全部通过。

保留原生行为的 HSDP 探针观察到 110 次具有源／目标设备工作的等待，全部在最终 DAG 可达；2,961 对同流相邻设备工作也全部可达。这里排除了 host metadata view，口径不同于修复前「全部 dispatch」的探针，不能把计数差异解释为丢失等待。

其中 11 条 post-backward 等待观测验证了 AR 延后向后续 MM 传播。原反例对应的新捕获为 AR `2746` → 后续 MM `2794`，将 AR 时长增加 1,000,000 个抽象单位后，MM 最早启动从 1,008 变成 1,001,008。抽象单位不代表硬件耗时。独立流不应被阻塞的反例由单元测试另外验证。

四组修复前后原始 tensor 事件的算子、阶段、输入输出 shape／dtype／storage bytes 逐项一致，通信顺序与 payload 一致，梯度参数数量及捕获内存峰值一致。控制节点没有 tensor 操作数，也不生成内存生命周期事件。

验证：完整回归 460 passed；随后新增的原生 HSDP 最终图回归 1 passed；容器内 Pyrefly 0 errors。变更文件的格式、Ruff、拼写及架构检查通过。宿主 pre-commit 的 Pyrefly 因 Python 3.9 环境缺失可执行文件失败，不能称为完整 Lint 通过；OAT 因无暂存文件跳过实际扫描。静态增量审查未发现必修问题。

本轮证据位于 `torchtitan-migration-audit-20261020/stream-control-20261023/`，包含四配置 JSON、`analysis.json`、`stream-analysis.json`、`comparison.json` 与测试／检查日志。host synchronize、轮询式进度和真实 NPU/HCCL 时序不在本轮验证范围；PP 入口继续拒绝。

## 修复前审计范围与方法

修复前检查未提交实现，使用既有固定 TorchTitan v0.3.0、NPU master 快照与 torch 2.14.0+cpu 环境。该轮没有修改运行源码；执行了四次真实 DSV4 debugmodel Meta F/B/O，以及一次只增加观测的 HSDP stream 探针。

| 配置 | world／rank | DP replicate／shard | TP／CP／EP | AC／计算 | AG 节点 | 已完成 unshard 区间 | 参数 storage 读取检查 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| FSDP | 2／0 | 1／2 | 1／1／1 | none／BF16 | 18 | 18 | 263 |
| HSDP | 4／0 | 2／2 | 1／1／1 | none／BF16 | 18 | 18 | 263 |
| 组合并行 | 16／15 | 2／2 | 2／2／2 | none／BF16 | 30 | 28 | 263 |
| HSDP + SAC + MXFP8 | 4／0 | 2／2 | 1／1／1 | selective／MXFP8 | 18 | 18 | 297 |

全部 MTP=1；组合并行使用 S=512，其余 S=128。最初组合并行 S=128 被原生 metadata 以压缩块长度不足拒绝，改为既有已验证的 S=512 后完成捕获。没有将这次配置拒绝当作依赖失败。

按 FSDP residency allocation 的实际 storage_key/generation、有效区间定位参数读取，不按 shape 猜测权重。随后在全部模板的全局 DAG 中检查祖先关系。组合配置包含两个没有进入已完成 unshard 区间的实际预取 launch，保留其通信节点，但不伪造权重消费者。

另外，将每个有使用者的 AG 节点时长增加 1,000,000 个抽象调度单位，重新求 DAG 最早启动时间。这只验证依赖传播，不代表硬件时延。

## 通过的部分

### AG 完成约束权重计算

共检查 1,086 次读取参数 storage 的算子，未发现对应 AG 不是祖先的情况。78 个有参数消费者的 unshard 区间逐个延后 AG 后，该区间全部被检查的使用者都会延后。

HSDP 捕获中的具体链路：

```text
104 c10d._allgather_base_
  → 130 aten.view
  → 200 fsdp.split_with_sizes_copy
  → 233 aten.mm（layers.0.attention.wq_a）
```

`233.inputs[1]` 的实际 storage_key 与 `200` 的写出 storage 匹配，激活输入 `233.inputs[0]` 不匹配。因此这条边确实覆盖权重输入，并非仅从激活路径偶然连通。

源码依据：[输入生产者与 storage mutation frontier](../torchtitan_simulator/capture/dispatch_capture.py)。AG output 是通信 buffer，经过原生 copy-out／view／cast 才成为权重使用者的输入；不要求 AG output 与最终权重是同一个 tensor 对象。

结论以消费者遵守完整、跨模板 predecessors，且把 AG 节点结束视为通信完成为前提。仅按节点列表或 Python launch 返回顺序执行，不属于本轮验证的 DAG 消费方式。

### RS、AR 与最终梯度消费

HSDP 相关三组共检查 35 对同参数组的 RS→AR，全部可达。四组全部归约节点都可达 clip，并且是每个实际 fused AdamW 节点的祖先，未发现 optimizer 绕过归约。

源码依据：[归约的实际梯度依赖](../torchtitan_simulator/capture/native_transport.py)、[FSDP post-backward 观测](../torchtitan_simulator/capture/fsdp_residency.py)。

这不意味着每个 RS／AR 都应阻塞下一层反向。当前层 dX 供下一层反向，当前层 dW 供 RS／AR；二者可以有独立分支并重叠。应保留实际数据依赖及原生明确执行的等待，而不是统一添加「归约完成后才能算下一层」的屏障。

## 修复前的 S1：stream/event 控制依赖缺失

[MetaDeviceModule.Stream.wait_event / wait_stream](../torchtitan_simulator/compat/device.py) 在修复前直接返回，Event.record 不保存流前沿；捕获侧主要保留 tensor 生产者、alias 和 storage mutation 数据边，没有完整记录流序号与事件依赖。

HSDP stream 探针保留原行为，只观察原生 record/wait 与每个 dispatch 所在的 Meta stream，再与最终 DAG 比较。112 次可观察等待中有 74 次未形成相应可达路径；其中包含 allocation／profiler 节点，不能将 74 当作独立的实际错误数量。继续检查同一目标流上的后续实际计算，得到 12 条等待观测缺失的例子。

一个明确反例来自当前安装 PyTorch 的 `_fsdp_param_group.py:776`：partial-group backward 在返回 autograd 前，对 `_post_reduce_event` 执行 `current_stream().wait_event(...)`。这次原生等待真实发生，但图中没有以下路径：

```text
2699 RS → 2700 AR → 2701 post-reduce view
                         ↓ 原生 wait_event
2743 aten.mm（lm_head，后续 chunk 的计算）
```

对当前 DAG 延后 `2699` 或 `2700` 1,000,000 个抽象单位，`2743` 的最早启动时间仍为 269，与未延后时相同。因此这是能使调度提前的反例，不只是「看不到 wait 节点」。

其他观测还包括 RS buffer throttle 的等待，以及同一 AG stream 上的连续 AG 在 DAG 中没有顺序路径。数据依赖能够覆盖其中一部分，但不能替代全部流控制语义。若上游另有资源队列，可能补偿某些同队列顺序；当前图本身未输出足以完整复现这些 stream/event 约束的信息，也未验证上游能补偿。

### 修复方向

在 Meta stream/event 层观察原生行为，保留每条流的执行前沿、event record 时的前沿，以及 wait 对后续该流的约束；通过明确控制边或可被消费者正确执行的流调度信息进入图。必须保持原生允许的重叠。

不要给所有 RS／AR 与下一层计算统一加边；不要把仅用于 storage 生命周期的 `Tensor.record_stream` 当成执行同步；也不要使用全局 barrier 代替实际 event wait。

验证至少增加：本反例延后 AR 后目标 MM 必须延后；没有原生 wait 的独立反向分支仍可重叠；AG 权重链路、RS→AR、归约→clip／optimizer 不退化。

## 审查与证据

设计判断：dispatch 捕获、storage 身份、FSDP 观测、L0–L3 组装的分层合理；当前缺失的是独立于 tensor 数据流的控制语义，不能在末端按层号猜测补边。sidecar 路径未匹配 `torchtitan_npu/` 专项规则。CodeCheck 不适用；本轮没有运行源码修改，也没有重新执行历史 455 项回归。

本轮执行：四配置捕获、参数读取／AG 延后检查、RS→AR 与梯度消费可达性、stream/event 观察与延后反例。结论只覆盖这些 Meta 路径，不证明 NPU/HCCL 真实时序。

完整节点、TensorRef、探针与机器结果位于本机 `torchtitan-migration-audit-20261020/fsdp-dependency-audit-20261023/`：`capture.py`、`analyze.py`、`stream_probe.py`、`analyze_streams.py`、`analysis.json`、`stream-analysis.json` 及各捕获日志。节点 ID 仅标识本轮对应捕获，不能跨运行使用。
