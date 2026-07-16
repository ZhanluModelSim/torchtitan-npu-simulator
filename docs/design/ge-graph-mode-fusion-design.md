# 纯无 NPU 环境达成 GE Fusion 效果设计

> 状态：已实现（host-only 融合 pass）
> 分支：`feat/autofusion`（基于 `feat/npu-simulator` `c67bc9d`）
> 日期：2026-07-16

## 1. 目标

在**纯无 NPU 环境**（本容器无 `npu-smi`、无 NPU 硬件/驱动）下，达成 torch_npu graph-mode GE fusion 的**效果**——把捕获的 over-decomposed L0 图压缩为融合后的 kernel 区域，消除中间张量，并用 GE 真实融合算子类型标注。**不依赖任何离线 NPU 捕获**。

## 2. 问题陈述

模拟器在 meta 设备 `__torch_dispatch__` 下捕获 L0 算子，得到过度分解的图——GE 会融合掉的 aten 原语占绝大多数节点：

| step | L0 nodes | `unknown`(aten 原语) | 占比 |
|------|----------|----------------------|------|
| s0_F | 8487 | 6338 | 75% |
| s0_B | 18396 | 14965 | 81% |
| s0_OPTIMIZER | 978 | 975 | 99.7% |

`unknown` 的实际 raw_op 是 `aten.mul/add/unsqueeze/empty/slice/clone/_to_copy/detach/sum` 等 elementwise/shape 原语。后果：`StepGraph.fused_regions` 字段恒空（`step_graph.py:70`，预留未实现），`OpCostModel` 对所有 `unknown` op 返回零成本。

## 3. 诚实探索：能否在无 NPU 下跑真实 GE fusion？

GE 的 op 融合发生在 `aclgrph` 编译管线内。我们逐层深挖了 host-only 可能性：

### 3.1 build_initialize 带 soc_version —— ✅ 部分可行

| 调用 | 结果 |
|------|------|
| `build_initialize()`（默认） | `RuntimeError: Failed to initialize aclgrph build`（需 NPU runtime） |
| `build_initialize({"ge.socVersion":"Ascend910B"})` | **成功**（不抛异常）——指定 SOC 版本后 runtime 可 host 初始化 |

### 3.2 DUMP_GE_GRAPH 各阶段 dump —— ✅ 可行

设置 `DUMP_GE_GRAPH=1` + `DUMP_GRAPH_PATH=...`，GE 在编译的**每个阶段**导出可解析的 proto-text 图：

```
ge_proto_00000000_graph_0_PreRunBegin.txt
ge_proto_00000006_graph_0_RunCustomPassAfterBuiltinFusionPass.txt
ge_proto_00000007_graph_0_PreRunAfterOptimizeOriginalGraph.txt
ge_proto_00000008_graph_0_PreRunAfterOptimizeAfterStage1.txt
...
ge_proto_00000011_graph_0_PreRunAfterOptimizeGraphBeforeBuild.txt
```

这是华为内部检视 GE 融合的标准机制，全程无需 NPU。

### 3.3 build_model 编译真实算子 —— ❌ 失败

| 图 | build_model | 原因 |
|------|-------------|------|
| 纯 elementwise 链（Mul→Add→Mul） | 成功 | elementwise 算子可 host 编译 |
| 含 LayerNorm/MatMul 的图 | **失败** | `BuildModel: Build ir model Init failed`——device-kernel codegen 需要硬件 |

### 3.4 融合 pass 在 host-built 图上是否触发 —— ❌ 负结论

构造**真实融合模式**（residual Add（两个 tensor 输入）+ LayerNorm，GE 应融合为 `AddLayerNorm`），跑 `DUMP_GE_GRAPH`，检查所有 10 个 stage dump：

```
所有阶段: ['Add', 'LayerNorm', 'NPU']   ← 全程未融合
```

Add 与 LayerNorm 在**每个阶段都保持分离**，未出现 `AddLayerNorm`。即 GE 的 BuiltinFusionPass / OptimizeOriginalGraph / Stage1 在 `ge.es` 构造的图上**不触发融合**——融合规则注册在 op-impl（OPP）层，bare `ge.es` 构造不携带该元数据。

### 3.5 结论

| 能力 | host-only |
|------|-----------|
| GE runtime 初始化（soc_version） | ✅ |
| 编译阶段图 dump | ✅ |
| 真实算子完整编译（codegen） | ❌（需硬件） |
| 融合 pass 在 host-built 图上触发 | ❌（需 op-impl 融合元数据） |

**真实 GE fusion 无法在纯无 NPU 环境产出可用结果**。模拟器的 meta 捕获与 GE 图编译本质分离。

## 4. 方案：纯 host-only GE 目录落地的融合 pass

既然不能跑真实 GE，就**复现 GE 的融合行为**——在捕获的 L0 DAG 上做图改写，目标是 GE 真实算子目录（`ge.es.nn` catalog）里的融合算子类型。100% host-only，零 NPU 依赖。

### 4.1 GE 融合算子目录（`ge.es.nn`，已验证安装）

| 融合目标（GE 真实类型） | 对应的融合模式 |
|------------------------|----------------|
| `AddRmsNorm` | residual Add + RMSNorm |
| `AddLayerNorm` | residual Add + LayerNorm |
| `FusedMatMul` | MatMul/BMM + trailing bias-Add |
| `AdamApplyOneWithDecay` | optimizer foreach-adamw 子算子块 |
| `ElementwiseFusion` | 连续单消费者 elementwise/shape 原语链 |

### 4.2 pass 算法（`build_ge_fusion_profile`）

按 `seq_idx` 序遍历捕获 StepGraph 的 L0 节点，模式匹配建区：

1. **残差 + norm**：遇到 `rms_norm`/`layer_norm`，回溯前驱找单消费者、raw 含 `add` 的 elementwise 前驱 → 合并为 `AddRmsNorm`/`AddLayerNorm`。
2. **matmul + bias**：遇到 `matmul`/`bmm`，前向找 raw 含 `add` 的单 bias 后继 → 合并为 `FusedMatMul`。
3. **elementwise 链**：连续单消费者 elementwise/shape 原语 → 累积成 `ElementwiseFusion` 区，遇锚点时 flush。
4. **foreach-adamw**：optimizer step 中连续的 `zeros/zero_/sub/sign/mul/add_/mean/select` 等 → 单个 `AdamApplyOneWithDecay` 区。
5. **锚点**（attention/rope/softmax/comm/adamw_step）→ 单算子区（GE 原样保留）。

产出 `GEFusionProfile`（含 `original_op_seq_idxs` 映射），交 `apply_ge_fusion_profile` 填 `StepGraph.fused_regions`。

> 注：`AddRmsNorm`/`AddLayerNorm` 在当前捕获图上不触发——因为 converter 级融合（`npu_rms_norm`）已在捕获时把 residual+norm 合成单算子，L0 图里没有可再融合的 Add+RmsNorm 分解。这印证了 converter 与 GE 的分层：converter 级融合已吸收 residual+norm，本 pass 接力吸收 GE 级 elementwise/matmul+bias/foreach 融合。

## 5. 模块设计：`torchtitan_npu/simulator/ir/ge_fusion.py`

### 5.1 数据结构

```python
@dataclass class FusedNode:        # 一个融合 kernel
    node_id: int; fused_op_type: str        # GE 真实类型
    original_op_seq_idxs: list[int]         # 融并的 L0 OpNode.seq_idx
    input_fused_ids: list[int]; output_bytes: int = 0

@dataclass class GEFusionProfile:  # 载入/产出的融合 profile
    graph_name: str; fused_nodes: list[FusedNode]
    seq_to_fused: dict[int,int]             # seq_idx→fused 查找表

@dataclass class FusedRegion:      # 写入 StepGraph.fused_regions
    region_id: int; fused_op_type: str; op_ids: list[int]
    eliminated_intermediates_bytes: int = 0; is_unfused: bool = False
```

### 5.2 API

| 函数 | 职责 |
|------|------|
| `build_ge_fusion_profile(step_graph)` | **主路径**：host-only 图改写，按 GE 融合规则产出 profile，目标类型取自 `ge.es.nn` 目录 |
| `apply_ge_fusion_profile(step_graph, profile)` | 把 profile 映射到 StepGraph，填 `fused_regions`，算 region 内部边消除的中间张量字节 |
| `fusion_summary(step_graph)` | 融合前后统计：l0_nodes / regions / compression_ratio / 消除中间张量 / 融合算子类型集 |
| `load_fusion_profile_json(path)` | （可选）载入外部 JSON profile 做交叉校验 |
| `load_fusion_profile_air(path)` | （可选）载入 GE 原生 `.air` 图，`extract_graph_topology` 恢复融合节点拓扑 |

## 6. 映射算法（`apply_ge_fusion_profile`）

1. 建 `seq_to_opid`。
2. 每个 `FusedNode.original_op_seq_idxs` → op_ids 组装 `FusedRegion`（单 op → `is_unfused=True`）。
3. `eliminated_intermediates_bytes`：对每条 region 内部边（生产者与消费者同 region_id），累加生产者 `peak_mem`。
4. 未归属 op → 单算子 unfused region。
5. 赋值 `step_graph.fused_regions`。

**不变量**：每个 L0 OpNode 恰好归属一个 region。

## 7. 验证结果（16-layer 真实捕获，纯 host pass）

`scripts/validate_ge_fusion.py`：16 rank 捕获 → `build_ge_fusion_profile` → `apply_ge_fusion_profile` → `fusion_summary`：

| step | L0 nodes | → regions | 压缩比 | 消除中间张量 | 命中 GE 融合类型 |
|------|----------|-----------|--------|--------------|------------------|
| s0_F | 8487 | 1076 | **7.9x** | 3.57 GB | ElementwiseFusion:496, FusedMatMul:17 |
| s0_B | 18396 | 1884 | **9.8x** | 7.13 GB | ElementwiseFusion:881, FusedMatMul:34 |
| s0_OPTIMIZER | 978 | 5 | **195.6x** | 0 | AdamApplyOneWithDecay:1, ElementwiseFusion:2 |

- `FusedMatMul` 命中 17（fwd）/34（bwd）——对应 17 个 matmul/bmm 算子及其 bias-add 融并。
- `AdamApplyOneWithDecay` 命中 1——optimizer 的 foreach-adamw 分解（978 个 `aten.zeros/sub/sign/mul/add_`）融成单个 kernel，这是 195x 压缩的主因。
- fwd+bwd 共消除 ~10.7 GB 中间张量。
- `.air` loader 在含真实融合算子（MatMulV3/AddRmsNorm/FusedMatMul）的样例图上 roundtrip 验证通过。

## 8. 相关文件

| 文件 | 职责 |
|------|------|
| `torchtitan_npu/simulator/ir/ge_fusion.py` | host-only 融合 pass + apply + summary + 可选 json/air loader |
| `tests/assets/ge_fusion_sample.air` | 含真实融合算子的样例 GE 图（loader 验证用） |
| `scripts/build_sample_ge_air.py` | 用 `ge.es.nn` 构造样例 `.air` |
| `scripts/validate_ge_fusion.py` | host-only pass 端到端验证（16-layer） |
| `scripts/ge_host_fusion_proof.py` | 探索负结论的复现脚本（DUMP_GE_GRAPH 各阶段，证明 host-built 图不触发融合） |
| `scripts/ge_fusion_probe.py` | GE API 能力探测 |
| `scripts/op_inspect.py` | 捕获 L0 op_type 分布诊断 |

## 9. 后续工作

1. **接入 cost 管线**：扩展 `OpCostModel`，融合算子类型（`FusedMatMul`/`AddRmsNorm` 等）走专用成本估算；`StepGraph.total_flops`/`peak_active_mem` 从 region 聚合（扣 `eliminated_intermediates_bytes`）而非逐 node 求和。
2. **CSV 导出**：`StepGraph.export_l0_csv` 增 `region_id`/`fused_op_type` 列。
3. **L2 联动**：`fused_regions` 反映到 L2 SchedulePlan 的 action 粒度（融合后 kernel 作单 action 建模）。
4. **融合规则增强**：补充更多 GE 目录模式（如 `npu_gelu_mul`、MoE dispatch 融合），与 `ge.es.nn` 完整目录对齐。
