# 对接 GE Graph-Mode 算子融合设计

> 状态：已实现（P2 消费端），P1 捕获端待 NPU 机器补全
> 分支：`feat/autofusion`（基于 `feat/npu-simulator` `c67bc9d`）
> 日期：2026-07-16

## 1. 问题陈述

模拟器在 meta 设备 `__torch_dispatch__` 下捕获 L0 算子，得到的是**过度分解**的图——真实 NPU 的图引擎（GE/CANN）会融合掉的 aten 原语占绝大多数节点：

| step | L0 nodes | `unknown`(aten 原语) | 占比 |
|------|----------|----------------------|------|
| s0_F | 8487 | 6338 | 75% |
| s0_B | 18396 | 14965 | 81% |
| s0_OPTIMIZER | 978 | 975 | 99.7% |

`unknown` 的实际 raw_op 是 `aten.mul/add/unsqueeze/empty/slice/clone/_to_copy/detach/sum` 等 elementwise/shape 原语，GE 会把它们折进相邻 matmul/attention/norm kernel 或合成 elementwise fusion kernel。当前实现两个后果：

1. **`StepGraph.fused_regions` 字段已存在但恒为空**（`step_graph.py:70`，设计时预留，从未实现）。
2. **`OpCostModel` 对所有 `unknown` op 返回 `unknown_cost()`（flops/mem 全 0）**——既不计 flops 也不计内存，无法反映融合消除的中间张量与 launch 开销。

用户诉求：直接对接 torch_npu 的 graph-mode GE fusion，而非自行编写融合规则。

## 2. 核心约束（已验证）

GE 的 op 融合发生在 `aclgrph` 编译管线内，其入口 `ge.offline_compile.build_initialize` 需要 NPU runtime/驱动：

| 验证项 | 结果 |
|--------|------|
| `build_initialize()` | `RuntimeError: Failed to initialize aclgrph build`（本容器 `npu-smi` 不存在，无 NPU） |
| `ge.passes` 模块 | 空目录——融合 pass 未作为独立 graph-pass 暴露，锁在 aclgrph build 内部 |
| 模拟器 meta 捕获与 GE 编译 | 本质分离：GE 需要真实 device tensor，meta 设备不行 |

**结论**："在模拟器里实时跑 GE fusion"不可行。模拟器的 `__torch_dispatch__` 捕获与 GE 图编译是两条独立路径，无法在 meta 设备上合一。

## 3. 可行能力（已验证，无需 NPU）

GE 图的 **build / serialize / load** 全程不需要 NPU runtime：

```
ge.es.GraphBuilder("g") → Graph(6 nodes) → save_to_air(3972B) → load_from_air → 6 nodes roundtrip OK
```

- `ge.graph.Graph`：`get_all_nodes` / `add_data_edge` / `save_to_air` / `load_from_air` / `dump_to_file`
- `ge.graph.Node`：`name` / `type` / `get_in_data_nodes_and_port_indexes(port)` / `get_out_data_nodes_and_port_indexes(port)` / `get_inputs_size` / `get_outputs_size`
- `ge.es.nn`：真实融合算子类型——`AddRmsNorm`（residual+RMSNorm）、`FusedMatMul`（matmul+bias/act）、`AdamApplyOneWithDecay`、`MatMulV3`、`GemmV2`；`ge.es.math`：`Mul/Add/Sub` 等 elementwise

**结论**：可走"离线 profile + 模拟器消费"路线——在真实 NPU 上捕获 GE 编译后的融合图作为 profile，模拟器离线载入并映射到捕获的 L0 算子上。

## 4. 总体架构：离线捕获（P1）+ 模拟器消费（P2）

```
┌──────────────── 真实 NPU 机器（P1）─────────────────┐
│  torch.npu.set_compile_mode(jit_compile=False)     │   GE graph mode
│  → torch_npu 将 aten 下沉到 GE 图                    │
│  → GE 编译跑融合 pass（aclgrph build）              │
│  → build_model + save_to_air 产出 .air（融合后拓扑）│
│  + GE debug dump 产出 original→fused 算子映射        │
│  → 写 JSON profile（可移植契约）                     │
└────────────────────────┬───────────────────────────┘
                         │  .air + .json
   ┌─────────────────────▼─────────────────────┐
   │  本容器（P2，无 NPU）ir/ge_fusion.py       │
   │  load_fusion_profile_json / _air           │
   │  → apply_ge_fusion_profile(StepGraph)      │
   │  → 填充 StepGraph.fused_regions            │
   │  → fusion_summary（压缩比 / 消除中间张量） │
   └───────────────────────────────────────────┘
```

两条 profile 来源：

| 来源 | 载入函数 | 内容 | 是否含 original→fused 映射 |
|------|----------|------|---------------------------|
| **JSON profile** | `load_fusion_profile_json` | 可移植契约：每融合节点带 `fused_op_type` + `original_op_seq_idxs` | **是**（P1 显式产出） |
| **`.air` 图** | `load_fusion_profile_air` | GE 原生序列化图，`extract_graph_topology` 恢复融合节点类型 + 数据边 | 否（仅拓扑，用于融合算子 cost 归类） |

## 5. 模块设计：`torchtitan_npu/simulator/ir/ge_fusion.py`

### 5.1 数据结构

```python
@dataclass
class FusedNode:            # GE 编译图中的一个融合 kernel
    node_id: int
    fused_op_type: str                # 如 "AddRmsNorm"/"FusedMatMul"/"Mul"
    original_op_seq_idxs: list[int]  # 融并进来的捕获 L0 OpNode.seq_idx（.air 拓扑为空）
    input_fused_ids: list[int]       # 生产者 FusedNode.node_id（按端口序）
    output_bytes: int = 0

@dataclass
class GEFusionProfile:      # 载入的融合 profile
    graph_name: str
    fused_nodes: list[FusedNode]
    seq_to_fused: dict[int, int]  # seq_idx→fused 查找表（__post_init__ 建）

@dataclass
class FusedRegion:          # 写入 StepGraph.fused_regions
    region_id: int
    fused_op_type: str
    op_ids: list[int]                 # 该 region 内的 L0 OpNode.op_id
    eliminated_intermediates_bytes: int = 0  # 融合消除的中间张量字节
    is_unfused: bool = False         # 单算子 region（GE 原样保留）
```

### 5.2 核心 API

| 函数 | 职责 |
|------|------|
| `load_fusion_profile_json(path)` | 载入 P1 产出的可移植 JSON profile |
| `extract_graph_topology(graph)` | 遍历 `ge.graph.Graph`，跳过 Data/Const/NetOutput 叶子，按端口查 `get_in_data_nodes_and_port_indexes` 恢复生产者边，best-effort 读 `get_output_attr` 算 output_bytes |
| `load_fusion_profile_air(path)` | 懒导入 `ge`，`load_from_air` + `extract_graph_topology`，返回拓扑 profile（`ge` 未安装时模块仍可导入） |
| `apply_ge_fusion_profile(step_graph, profile)` | 把 profile 映射到捕获 StepGraph：按 `seq_idx→op_id` 聚合每个 FusedNode 的 op_ids 成 FusedRegion；region 内部边（生产者与消费者同 region）的 `peak_mem` 之和 = `eliminated_intermediates_bytes`；未归属 op 成单算子 unfused region；赋值 `step_graph.fused_regions` |
| `fusion_summary(step_graph)` | 融合前后统计：l0_nodes / fused_regions / fused_multi_op_regions / unfused_singletons / compression_ratio / eliminated_intermediates_bytes / fused_op_types |

### 5.3 JSON profile 契约（P1 产出、P2 消费）

```json
{
  "graph_name": "deepseek_v4_pro_simulate_16_layers",
  "fused_nodes": [
    {"node_id": 0, "fused_op_type": "AddRmsNorm",
     "original_op_seq_idxs": [12, 13, 14],
     "input_fused_ids": [..], "output_bytes": 262144},
    {"node_id": 1, "fused_op_type": "FusedMatMul",
     "original_op_seq_idxs": [15], "input_fused_ids": [0], "output_bytes": 524288}
  ]
}
```

`original_op_seq_idxs` 指向捕获的 L0 `OpNode.seq_idx`——GE 编译前的预编译图节点与 aten op 一一对应，P1 据此产出映射。

## 6. 映射算法（`apply_ge_fusion_profile`）

1. 建 `seq_to_opid = {OpNode.seq_idx: OpNode.op_id}`。
2. 对每个 `FusedNode`，把 `original_op_seq_idxs` 解析成 op_ids，组装 `FusedRegion`（单 op → `is_unfused=True`）。
3. 算 `eliminated_intermediates_bytes`：对每条 region 内部边（生产者与消费者同 region_id），累加生产者 `peak_mem`——这些中间张量被融合消除、不再物化。
4. 未被任何 FusedNode 归属的 op → 单算子 unfused region（GE 原样保留）。
5. 赋值 `step_graph.fused_regions`。

**不变量**：每个 L0 OpNode 恰好归属一个 region。

## 7. 验证结果

### 7.1 `.air` loader（真实 GE 图）

构造含真实融合算子的样例图（`scripts/build_sample_ge_air.py`，用 `ge.es.nn.MatMulV3`/`AddRmsNorm`/`FusedMatMul`），`save_to_air` → `load_fusion_profile_air`：

```
fused_nodes: 3
  #5 type=MatMulV3  in=[0, 1]   # Data + Const
  #6 type=AddRmsNorm in=[0, 5, 2]  # residual + matmul + gamma
  #7 type=FusedMatMul in=[6, 3, 4]
```

`.air` roundtrip OK，节点类型 + 生产者边正确恢复。样例资产 `tests/assets/ge_fusion_sample.air`。

### 7.2 真实 16-layer StepGraph（合成 profile 占位）

`scripts/validate_ge_fusion.py` 跑 16-layer 捕获，对每个 StepGraph 用 P1 占位合成器（按 anchor/fusible 分类分组连续 aten 原语链）产出 profile，再 `apply_ge_fusion_profile`：

| step | L0 nodes | → regions | fused_multi | unfused | 压缩比 | 消除中间张量 |
|------|----------|-----------|-------------|---------|--------|--------------|
| s0_F | 8487 | 1076 | 496 | 580 | **7.9x** | 2.57 GB |
| s0_B | 18396 | 1884 | 881 | 1003 | **9.8x** | 5.13 GB |
| s0_OPTIMIZER | 978 | 5 | 3 | 2 | **195.6x** | 0 |

- optimizer 978→5（195x）印证 foreach-adamw 分解（`aten.zeros/sub/sign/mul/add_`）被融合成数个 kernel。
- fwd+bwd 共消除 ~7.7 GB 中间张量。

> **说明**：7.2 用的是 P1 占位合成器（按算子类型启发式分组），非真实 GE 融合。它仅用于验证 P2 消费端管线，量化压缩潜力。真实 profile 须由 P1 在 NPU 机器上产出。

## 8. P1 离线捕获端（脚手架，`scripts/ge_fusion_capture.py`）

真实 NPU 机器上运行：

```
NGPU=<n> python3 scripts/ge_fusion_capture.py \
  --config deepseek_v4_pro_simulate_16_layers \
  --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer \
  --output_dir ./ge_profiles
```

流程：
1. `torch.npu.set_compile_mode(jit_compile=False)` → op 经 GE。
2. torch_npu 将 aten 下沉到 GE 图，`ge.offline_compile.build_model` 编译跑融合 pass。
3. `save_to_air` 导出 `.air`（融合后拓扑）。
4. GE debug dump 配置记录 per-op 融合归因 → 写 JSON profile（`original_op_seq_idxs`）。

> 状态：脚手架。`.air` 拓扑捕获已实现（镜像 `build_sample_ge_air.py`），`original→fused` 映射需 GE debug dump 配置（机器/GE 版本相关），标 TODO 待 NPU 机器补全。

## 9. 相关文件

| 文件 | 职责 |
|------|------|
| `torchtitan_npu/simulator/ir/ge_fusion.py` | P2 消费端：dataclass + JSON/.air loader + apply + summary |
| `tests/assets/ge_fusion_sample.air` | 含真实融合算子的样例 GE 图（loader 验证用） |
| `scripts/build_sample_ge_air.py` | 用 `ge.es.nn` 构造样例 `.air` |
| `scripts/validate_ge_fusion.py` | P2 端到端验证（16-layer 捕获 + 占位 profile） |
| `scripts/ge_fusion_capture.py` | P1 离线捕获脚手架（待 NPU 机器补全） |
| `scripts/op_inspect.py` | 捕获 L0 op_type 分布诊断 |
| `scripts/ge_fusion_probe.py` | GE API 能力探测 |

## 10. 后续工作

1. **真实 NPU 跑 P1**：补全 JSON profile 的 `original_op_seq_idxs` 映射，替换占位合成 profile。
2. **接入 cost 管线**：扩展 `OpCostModel`，融合算子类型（`AddRmsNorm`/`FusedMatMul` 等）走专用成本估算；`StepGraph.total_flops`/`peak_active_mem` 从 region 聚合（扣除 `eliminated_intermediates_bytes`）而非逐 node 求和。
3. **CSV 导出**：`StepGraph.export_l0_csv` 增 `region_id`/`fused_op_type` 列。
4. **runtime 调度联动**：`fused_regions` 反映到 L2 SchedulePlan 的 action 粒度（融合后的 kernel 作为单 action 建模）。
