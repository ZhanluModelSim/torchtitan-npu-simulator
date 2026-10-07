# Simulator 输出与内存口径

> 现状核对：2026-10-07，`feat/npu-simulator`，实现基线 `ecde2f0`。依据源码与 Git 历史静态核对，未执行测试或仿真；后续行为以当前源码为准。

返回[使用入口](../simulator.md)。格式开关见[配置参考](configuration.md)，调度依赖见[通信与 schema v2 契约](../../design/schedule-plan-dependency-reconstruction-contract.md)。

## 格式与目录

`simulation.output_formats=[]` 是当前默认值。内存跟踪默认开启，所以即使不指定输出格式，也会写出 `memory/memory_summary.json`。`mem` 控制详细内存产物；关闭内存跟踪后，单独选择 `mem` 不会建立内存计划。

| 文件或目录 | 启用格式 | 内容 |
|---|---|---|
| `summary.txt` | `text` | 算子、执行类型、通信与逻辑内存摘要 |
| `trace.html` | `html` | 层级图和调度展示 |
| `compute_graph.dot` | `dot` | 算子依赖图 |
| `simulation_result.json` | `json` | WorkloadGraph、SchedulePlan 和模板等结构化内容 |
| `kernel_summary/rank_N.csv` | `csv` | 各逻辑 rank 的 L0 算子展开 |
| `ir_export/rank_schedule.csv` | `csv` | L3/兼容调度导出 |
| `ir_export/schedule_plan.csv` | `csv` | L2 action 与 DataSlot，包含两个 section |
| `ir_export/l1_schedule/` | `csv` | 各 stage 的 L1 模板实例调度 |
| `ir_export/step_*_l0_ops.csv` | `csv` | L1 模板的 L0 算子；多个同类模板时文件名含模板标识 |
| `memory/memory_summary.json` | 不依赖格式 | 内存跟踪开启且完成建模时生成 |
| `memory/` 详细 trace/CSV | `mem` | 生命周期、保存激活、回放 action 和诊断记录 |

PP=1 直接写入 `simulation.output_dir`；PP>1 各 capture process 写入 `rank_N/`，这里的 N 是真实 PP 进程编号，不能与 `kernel_summary/rank_N.csv` 的逻辑 rank 混淆。一个进程可拥有多个虚拟 stage，不能把目录固定解释为某一个 stage。输出不自动合并为全局 graph。

完整导出示例：

```bash
--simulation.output-formats text html dot json csv mem
```

大 world 下可按需省略 CSV/JSON，或用 `--simulation.csv-max-ranks 4` 限制部分 CSV 的逻辑 rank 展开；该选项不缩小 world、mesh 或捕获范围，也不限制所有输出文件。

## L2 与跨进程消费

生产消费优先读取 SchedulePlan 对象。CSV 用于诊断，不能把第二个 DataSlot section 当作 action 行继续解析。DualPipeV overlap child 的 parent ID、template_ref 和 slot 必须保留。

- `schedule_order` 是 rank-local 发布顺序；`seq_idx` 是来源/诊断位置，不能当全局时间。
- PP SEND/RECV 按 `transfer_id` 配对，再核对逻辑 rank、stage 和 tensor 信息；只用 `comm_peer_rank` 不足以区分多条传输。
- 计算块内通信已经在 L1，不能再从兼容 DataPass 补一份费用。
- PP replay 按 action 实例化各 microbatch，展示模板仍按首次出现折叠。

Memory Perfetto/Chrome trace 横轴由事件序号构造，不是设备运行时延。launcher 的完成记录检查 worker 状态和导出完整性，不等于模型数值或硬件性能验收。

## 内存建模口径

核心 use-def 扫描使用未折叠 raw event stream，而不是把 L0 `peak_mem` 相加。每个 tensor 记录 producer、consumer、last use、alias/mutation、logical bytes、modeled resident bytes 和分类。

1. 从真实 model_parts 获取本地参数 shard 并去重。
2. 保留参数 materialization alias 与 FSDP full-parameter/staging 的区别。
3. FSDP plugin 根据 wait/reshard 状态事件建驻留，不用 schedule intent 冒充驻留开始/释放。
4. Checkpoint plugin 区分边界保存、内部临时值、selective 保存且在 recompute 中复用的值。
5. non-PP 下结合 autograd `saved_tensors_hooks` pack/unpack 识别实际保留值，排除参数、输入、通信和 FSDP buffer，再按 storage/alias 去重。
6. 缺失 meta parameter gradient 由专门 plugin 建模，避免少计训练梯度。
7. PP 下按 SchedulePlan action 实例化模板，区分 microbatch tensor 身份、持久参数和 residency。
8. 导出整体 active peak、pre-optimizer model peak、forward/backward/optimizer 分峰及 checkpoint/activation prefetch 逻辑量。

**精度边界：** PP replay 当前明确不重放精确 autograd save slots，`autograd_saved_tensor_events=None`，避免把 MB0 save ID 错用到其他 MB；该路径采用 use-def/checkpoint 推断。none 下的 save slots/unique storages 数量和卸载 inventory 累计字节都不能直接当作时间轴峰值。

卸载仅是驻留假设：当前不是实际 host/device offload executor，也没有据此测量回捞时延。FP8 AG 当前是 transport/staging precision model，不是完整的 quantize/通信/dequantize 性能实现；EP 则显式建模 payload+scale 与依赖。

证据：[estimator](../../../torchtitan_npu/simulator/memory/estimator.py)、[PP replay 边界](../../../torchtitan_npu/simulator/memory/schedule_replay.py)、[memory plugin 接口](../../../torchtitan_npu/simulator/memory/plugins.py)。

## 内存明细文件

`memory/` 中的主要文件：

| 文件 | 内容 |
|------|------|
| `memory_summary.json` | 默认导出；参数常驻、前向/反向/optimizer 峰值及 checkpoint 聚合 |
| `memory_events.csv` | `mem`；未折叠的算子输入输出事件，含 PP stage/microbatch/comp_type |
| `memory_timeline.csv` | `mem`；alloc/free 后的 active tensor bytes 曲线 |
| `tensor_lifetimes.csv` | `mem`；每个 tensor 的 birth、last consumer、death、逻辑大小、建模驻留大小和分类 |
| `checkpoint_tensors.csv` | `mem`；AC wrapper 边界输入/输出及 selective 内部保存 tensor 元数据；存在 AC 边界时生成 |
| `activation_offload_tensors.csv` | `mem`；不属于 AC wrapper 记录、但被统一激活卸载策略覆盖的 tensor 及逐层归属 |
| `autograd_saved_tensors.csv` | `mem`；存在 save/unpack 记录时生成，供核对 autograd 保留值 |
| `memory_actions.csv` | `mem`；PP 调度 action 与展开后内存事件区间的映射；非 PP 不生成 |
| `memory_trace.json` | `mem`；可由 Chrome Trace 或 Perfetto 打开的阶段趋势图 |

别名去重按 ATen view 算子的精确名称识别，不以 `permute`、`slice` 等子串判断。
MoE token permutation 的路由输出、排序索引以及 `slice_backward` 等分配型算子
分别计入独立 tensor；真正的 view/transpose/split 则保留与原 tensor 的别名关系。
AC `none` 下可用 `autograd_saved_tensors.csv` 核对实际 save/unpack，再与
`activation_offload_tensors.csv` 对照；`requires_grad` 本身不代表需要保存或卸载。

## 保存激活与回捞字段

示例数值仅说明字段结构，不代表当前模型的测量结果。逻辑回捞量不直接给出有效带宽或传输耗时。

selective checkpoint 保存的内部 tensor 在 `tensor_lifetimes.csv` 中标记为
`checkpoint_saved_for_recompute`。summary 中的
`checkpoint_recompute_saved_{tensor_count,logical_bytes,modeled_bytes}`
分别给出数量、逻辑大小和建模驻留大小；全量明细中对应的
`checkpoint_tensors.csv` 记录使用 `role=recompute_saved`。

`memory_summary.json` 的 `checkpoint_saved_activations` 按稳定的
`checkpoint_id` marker 聚合保存激活。例如：

```json
{
  "checkpoint_saved_activations": {
    "part0:layers.0": {
      "marker_kind": "module",
      "logical_bytes_per_instance": 262144,
      "modeled_bytes_per_instance": 0,
      "instance_count": 2,
      "logical_bytes_total": 524288,
      "modeled_bytes_total": 0,
      "pp_stages": [0],
      "microbatches": [0, 1],
      "size_variants": [
        {
          "logical_bytes": 262144,
          "modeled_bytes": 0,
          "tensor_count": 1,
          "instance_count": 2
        }
      ]
    }
  }
}
```

`logical_bytes_per_instance` 可直接作为后续一次恢复/load 的大小。若同一个 marker
存在动态 shape，该字段为 `null`，具体大小和出现次数记录在 `size_variants`。
marker 当前使用 AC wrapper 模块路径且 `marker_kind="module"`；该结构不依赖“层”的
概念，后续 selective AC 可以使用 op 级 marker 和 `marker_kind="op"` 并复用相同格式。

`checkpoint_prefetch` 保留 AC 专用的边界输入与 selective 保存值统计。汇总三种
AC 模式下的统一激活逻辑回捞量时，应使用 `activation_prefetch`：

```json
{
  "activation_offload_tensor_count": 567,
  "activation_offload_logical_bytes": 32992845480,
  "activation_offload_modeled_bytes": 0,
  "activation_prefetch_logical_bytes": 32992845480,
  "activation_prefetch": {
    "part0:layers.0": {
      "marker_kind": "layer",
      "logical_bytes_per_instance": 1771982464,
      "modeled_bytes_per_instance": 0,
      "tensor_count_per_instance": 29,
      "instance_count": 1,
      "logical_bytes_total": 1771982464,
      "modeled_bytes_total": 0,
      "pp_stages": [0],
      "microbatches": [0],
      "size_variants": [
        {
          "logical_bytes": 1771982464,
          "modeled_bytes": 0,
          "tensor_count": 29,
          "instance_count": 1
        }
      ]
    }
  }
}
```

none 模式按正常 backward consumer 的 `layers.N` 归档保存激活；full/selective
复用 checkpoint marker，并合并该层其他保存激活。没有唯一层归属的 loss、logits
等记录在 `part0:<unattributed>`，其总量同时写入
`activation_prefetch_unattributed_{tensor_count,logical_bytes}`。

AC 专用的 `checkpoint_prefetch` 结构如下：

```json
{
  "checkpoint_prefetch": {
    "part0:layers.0": {
      "marker_kind": "module",
      "boundary_logical_bytes_per_instance": 262144,
      "recompute_saved_logical_bytes_per_instance": 1048576,
      "prefetch_logical_bytes_per_instance": 1310720,
      "boundary_modeled_bytes_per_instance": 0,
      "recompute_saved_modeled_bytes_per_instance": 0,
      "modeled_bytes_per_instance": 0,
      "boundary_tensor_count_per_instance": 1,
      "recompute_saved_tensor_count_per_instance": 3,
      "tensor_count_per_instance": 4,
      "instance_count": 2,
      "prefetch_logical_bytes_total": 2621440,
      "modeled_bytes_total": 0,
      "pp_stages": [0],
      "microbatches": [0, 1],
      "size_variants": [
        {
          "boundary_logical_bytes": 262144,
          "boundary_modeled_bytes": 0,
          "boundary_tensor_count": 1,
          "recompute_saved_logical_bytes": 1048576,
          "recompute_saved_modeled_bytes": 0,
          "recompute_saved_tensor_count": 3,
          "prefetch_logical_bytes": 1310720,
          "modeled_bytes": 0,
          "tensor_count": 4,
          "instance_count": 2
        }
      ]
    }
  }
}
```

`prefetch_logical_bytes_per_instance` 是该实例回捞大小的逻辑估算：
wrapper 边界保存输入与 selective AC 内部保存结果之和。full AC 通常只有前者；
selective AC 两部分都可能存在。`prefetch_logical_bytes_total` 是该 marker 在整个
训练 step 中所有实例的逻辑回捞总量；当前未建模实际传输、复用、并发和时延。开启保存激活 offload 建模后，logical bytes 保持不变，
而 modeled bytes 为 0，表示这些 tensor 不计入设备驻留。

同一 marker 存在动态 shape 时，所有 `*_per_instance` 字段为 `null`，应从
`size_variants` 读取各尺寸及出现次数。若 selective 保存值无法可靠匹配 checkpoint
边界，`checkpoint_prefetch_unattributed_{tensor_count,logical_bytes}` 会非零；
此时分层结果不完整，不应直接作为总带宽结论。

## Kernel CSV 字段

| 列 | 说明 |
|----|------|
| `rank` | 逻辑 rank 编号（0 ~ world_size-1） |
| `step_type` | 步骤类型：`forward` / `backward` / `optimizer` |
| `step_id` | 步骤模板 ID |
| `topo_order` | 在该步骤模板内的拓扑序（从 0 开始，Kahn 算法） |
| `op_id` | 算子唯一 ID |
| `op_type` | 规范化算子类型（如 `matmul`、`rms_norm`），未映射的显示原始算子名 |
| `raw_op_type` | 原始 dispatcher 算子名（如 `aten.addmm.default`、`npu.npu_rms_norm.default`） |
| `inputs_shape` / `outputs_shape` | 输入/输出张量 shape，格式 `[d0,d1];[d0,d1]` |
| `inputs_dtype` / `outputs_dtype` | 输入/输出 dtype |
| `flops` / `peak_mem` / `param_mem` / `comm_bytes` | 单算子辅助成本；`peak_mem` 不能累加为训练峰值 |
| `repeat_count` | 去重折叠的重复次数 |
| `module_path` | 算子所属模块路径（如 `layers.2._checkpoint_wrapped_module.moe`） |
| `phase` | 捕获阶段：`forward` / `backward` / `optimizer` |
| `execution_kind` | 实际执行类型：`original_forward` / `recompute` / `backward` / `optimizer` |
| `is_recompute` | 当前算子是否由 activation checkpoint 在 backward 中实际重放 |
| `group_name` | 解析后的通信维度名（如 `tp`、`ep`、`fsdp`）；无法唯一解析时回退为框架原始组 ID |
| `raw_group_name` | 框架生成的原始 ProcessGroup 名称（如 `3713`） |
| `comm_dim` | `group_name` 的兼容别名 |
| `comm_ranks` | 通信域包含的 Rank 列表（仅通信算子有值，如 `0,1,2,3,...,15` 表示这 16 个 rank 属于同一通信组） |

未映射算子保留 raw 名字；`cost_unknown` 表示成本尚未覆盖，不能按零工作量消费。recompute 属于 backward phase，但有独立 `execution_kind`；缓存命中且未重放的算子不应算作 recompute。通信维度无法唯一解析时保留原组名，不能猜测。
