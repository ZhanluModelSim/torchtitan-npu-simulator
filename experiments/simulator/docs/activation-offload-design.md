# 激活 offload 执行图设计与真实捕获核对

本文记录 2026-11-02 的真实 Meta 捕获核对与激活 offload 实现。训练 simulator 在通信视图上插入激活传输与等待依赖，并修正静态显存；平台提供传输耗时、资源与 DES。沿用 `memory_offload_ac_saved_tensors=True` 启用。优化器卸载不在范围内；本仓不执行真实设备 copy。

## 实现前的真实捕获证据

训练基线 `3189b01e2553df642003e87ecce9c4e09ef94aa8`，平台基线 `e33f84f08e518192dea9fad99d2788e9094176a5`。在 `ttsim` 容器使用工作区 editable 安装、PyTorch `2.14.0+cpu` 运行，实际 simulator 导入来自本 checkout。以下为 Meta 捕获及公共图适配验证，不是 NPU 实测，也未验证传输时间精度。

共同参数：world size 4、DP replicate 2、DP shard 2、EP 2、local batch 1、global batch 8、seq_len 128，启用 `memory_offload_ac_saved_tensors`。DeepSeek 使用 `deepseek_v4_smoketest`；Kimi 使用 `kimi_k3_smoketest`。所有案例均完成捕获及平台适配。

| 案例 | 公共图节点数 | stream_wait 数 | 逐层 payload 合计（bytes） |
| --- | ---: | ---: | ---: |
| DeepSeek-V4 selective AC | 5164 | 117 | 5,301,376 |
| DeepSeek-V4 full AC | 5162 | 117 | 1,089,536 |
| DeepSeek-V4 无 AC | 3492 | 117 | 63,053,732 |
| DeepSeek-V4 selective AC、MTP 1 | 6276 | 152 | 20,702,600 |
| Kimi selective AC | 4395 | 106 | 4,066,496 |

完整本地证据位于工作区 `runs/activation-offload-audit-20261102/`：`audit.py`、各案例 `summary.json`、`nodes.json`、`offload-records.json`、`action-spans.json` 及运行日志。这些文件为未入库的实验产物；本表和下述结论保留审阅所需摘要。未覆盖量化、PP、多 MTP 或关闭 FSDP 的组合。

### 层、head 与 MTP

以下为捕获观察到的执行块顺序，不代表已经验证了新增传输后的 DES 调度。

- 普通 DeepSeek 的 F 顺序是 decoder layers → hc_head → norm → 分块 lm_head/loss。8 次 lm_head 前向和对应反向交错发生在 **F 图内**。B 图从 norm/hc_head 反向进入最后 decoder 层；有 AC 时先重计算，无 AC 时直接反向。
- Kimi 的尾部包含 output_attn_res 和 norm，同样存在 F 图内的分块 lm_head 反向。因此不能仅按名字 `lm_head` 寻找 B 图预取窗口。
- MTP 1 的 F 顺序为主 decoder → hc_head/norm → MTP → 分块 head/loss；B 先执行 MTP 反向，再执行主模型 norm/hc_head 和 decoder 反向。不能按报表的层名排序将 MTP 接成普通 decoder 的最后一层。
- 恢复数据必须在首次消费前完成；有 AC 时首次消费可能是 **recompute**，不能只连接普通 backward。

### 原生 wait 与原始内存记录

`compat/device.py` 的 `wait_event` / `wait_stream` 经 `capture/dispatch_capture.py::record_stream_wait` 生成 `simulator.stream_wait`。它们等待源流的已记录前沿，不是按层生成的屏障。本轮所有这类节点均无 module path。FSDP 通信投影明确保留这些节点。

可以复用控制节点和流前沿依赖机制；不能直接给现有 FSDP wait 增加传输前驱，否则可能扩大等待范围、阻塞无关预取。新增等待应记录建模来源，使用独立传输流，只阻塞实际消费者或缓冲复用边界。

`memory/records.py::_activation_prefetch_records()` 只有候选记录，不代表 H2D 执行事件。本轮 timeline 只有 alloc/free，action spans 为空。`memory/activation_offload.py` 将候选 lifetime 的 `modeled_num_bytes` 设为零，因此当前并未保留一层设备缓冲。head/norm 及 F 内即时消费的数据保持驻留，依据该模块及 `tests/memory/test_interleaved_loss_offload.py`；本轮导出的候选记录未独立测量 head 驻留字节。

逐层 payload 按层内 storage 去重，但层间可能重复：DeepSeek 的两块 1024 B 输入被四层共享；Kimi 有一块 1024 B 被四层共享、一块 65536 B 被三层共享。payload 的归属层也可能不同于生产层，例如 checkpoint 输入由前层产生。逐层预算不是全局唯一字节数，也不足以单独决定释放时机。

## 实现组织与交付契约

`capture/activation_transfers.py` 在捕获、内存分析和 FSDP 通信投影之后构造执行视图，不修改 `captured_workload`。内部计划由现有保存候选、storage 身份和真实 producer 生成。`memory/activation_transfer_budget.py` 消费同一计划，构造静态缓冲区预算。

- 按 `comp_type=F/B` 划分阶段，不能将 F 内 `phase=backward` 的分块 head 当成 B 预取窗口。根据已观察到的模块作用域形成计算块；当前要求同一层在各阶段只有一个连续块，有重入时明确拒绝。重计算与该层正常反向属于同一消费块。
- 层边界取完整 DAG 可达关系上的入口／出口前沿：中间隔有通信、wait 或隐藏 FSDP 节点时，仍保持原始因果关系。不能把捕获序号直接转换成全串行依赖。
- D2H 同时等待层完成及实际数据 producer。正常情况下覆盖下一块，完成等待连接到再下一块的入口；最后的传输由 F 阶段完成屏障收尾。
- H2D 在前一个 B 计算块入口条件就绪时启动，在当前层的第一次消费（包括 recompute）前等待。B 内没有前一个计算块时，H2D 从 B 入口启动，不能假定免费完成；无 FQN/FSDP 前奏不推迟其静态缓冲驻留起点。
- 拷贝使用独立逻辑流、同向顺序依赖；等待是新建零耗时 `simulator.stream_wait`，不改原生 FSDP wait 的含义。跨 F/B 不延续异步传输，保持平台现有阶段屏障及 GA 组装。
- 同一 storage 仍被后续前向读取时继续驻留。dispatch 从真实算子 schema 记录 `mutated_storage_keys`，包括 `out=` 写入；若已复制数据之后又发生前向写入，当前明确拒绝，不能使用过期主机副本。

传输节点不伪造输入 dtype 或输出张量，平台消费以下窄属性：

```python
annotations["raw_op_type"] = "simulator.activation_h2d"  # 或 simulator.activation_d2h
attrs["activation_transfer"] = {"direction": "h2d", "bytes": 65536, "layer": "layers.3"}
annotations["modeling_source"] = "activation_offload_neighbor_v1"
```

新增 ID 从完整原图最大 ID 之后分配，原 ID 不变，不加入 `native_dispatch_op_ids`。新流为现有最大 stream ID 之后的独立整数。`activation_transfer_plan` 注记记录新增节点、层列表和阶段策略。模型生成节点与原始捕获事实分别验证。

### 晚产生的 payload

真实 MTP 存在归属该层、但在层结束后的 head 中才产生的保存值。整层 D2H 必须等待这些 producer；此时采用 `late_payload_phase_drain` 保守策略：取消该层原邻层完成等待，由 F 末屏障等待整层复制完成，候选 storage 保留到 F 末。此近似可能损失本可利用的掩盖窗口，也可能超过两层同时驻留；未按各 storage 的实际 ready 时刻拆批。

`activation_transfer_plan.phase_drain_layers` 与 `MemoryPlan.notes` 保存此说明，标准 logger 发出列明层名的 WARNING，平台可收集到 `task.log`。普通路径 INFO 记录传输节点数、单向总字节和阶段屏障策略。缺失非初始 producer、重复层调用及复制后写入均明确失败。

## 静态显存与平台统计

原始 `raw_events` 和 `tensor_lifetimes` 不变；模型新增的驻留区间单独保存在 `activation_buffer_lifetimes`。原始与新增区间共同组成修正后的 timeline 和预算，内存明细 CSV 以 `activation_transfer_buffer` 区分新增区间。

前向从真实 allocation 出生至下一计算窗口结束保留候选；晚产生 payload 保留至 F 末。反向从预取窗口开始至消费结束保留，跨层共享 storage 合并区间，仍有实际消费者时延长驻留。零字节 alias 不能覆盖真实 allocation 的生命周期。head 自身的原驻留不重复补入。重新求 F/B 峰值，O 保持原口径，总峰值取各阶段最大值；这是静态策略预算，不是 DES 时间驱动的动态显存回放。

平台将传输归为 h2d/d2h，通过传输字节和硬件能力计算耗时，DES 统一处理并发、等待和掩盖。传输参与算子统计和时间线，不再额外累加同一传输的后处理拖尾。激活传输字节不加入 MBU 分子；MFU/HFU/MBU 使用更新后的端到端分母。训练仓不反向依赖平台或其硬件配置。

## 验证入口与边界

- `tests/capture/test_activation_transfers.py`：独立手算完整路径前沿、正常邻层等待、重计算首用、B 入口加载、前奏显存、共享 storage 去重、晚生 alias、晚 producer F 末收尾及复制后写入。
- `tests/capture/test_dispatch_capture.py::test_capture_records_out_overload_storage_writes`：真实 Meta dispatch 的 `aten.add.out` 写入事实进入内存事件。
- `tests/test_memory_delivery.py::test_activation_transfer_mtp_capture_preserves_raw_facts_and_restores_at_b_entry`：真实 MTP CLI 捕获、阶段归属、原始审计保留及 mem 导出。
- 现有 `tests/memory/test_interleaved_loss_offload.py`：DeepSeek/Kimi 的 none/full/selective AC 原生 head/loss 驻留；跨仓系统验证另外检查 DES、导出、MBU 分子和快照回放。

本轮独立审阅推动修正了完整 DAG 前沿、F 内反向阶段归属、schema 写入事实、晚 producer 与晚 alias 的区别，以及 B 入口前奏的驻留起点。Meta 捕获和 synthetic 计算耗时仅验证结构与流水线，不证明 NPU 数值或真实传输性能。优化器卸载、跨阶段异步执行、按 storage ready 拆批和动态显存回放仍未实现。
