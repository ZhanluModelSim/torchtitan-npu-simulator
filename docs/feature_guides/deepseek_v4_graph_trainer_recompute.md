# DeepSeek-V4 GraphTrainer 激活重计算

本文说明 `torchtitan-npu` 中 DeepSeek-V4 GraphTrainer 的激活重计算设计，包括默认 `full` 策略、可选 `dsv4-mhc` / `dsv4-mhc-moe-save` 策略、使用方式和已知限制。


## 1. 特性概述

DeepSeek-V4 同时包含 attention、压缩 attention、MoE、SimpleFSDP 通信和 Ascend 融合算子。若前向激活全部保存，显存峰值较高；若只在 module 边界统一 checkpoint，又会重算昂贵的 attention/MoE 子图。GraphTrainer 先将前向、loss 和 `torch.autograd.grad` 反向捕获为联合 FX 图，再按节点决定保存或重算，从而用计算换取显存。

当前重计算只作用于联合前向/反向图，`optimizer.step()`、梯度裁剪和 scheduler 仍在图外执行。

核心原则：

- 在 joint FX graph 上做节点级决策，而不是重新包 eager checkpoint hook；
- 随机数状态、反向必需值和 FSDP 强制节点优先保存；
- 可重算子图保持函数式、无隐式状态；
- 通用策略由上游提供，NPU 仓只扩展 DSV4 所需规则。

## 2. 配置接口

### 2.1 默认配置

DSV4 GraphTrainer 工厂默认配置如下：

```text
enable = true
mode = "aot_fx_trace"
memory_policy = "full"
pass_pipeline = "mutation-functionalization"
disable_passes = ["cudagraph_pass"]
```

配置位置：

```text
torchtitan_npu/models/deepseek_v4/config_registry.py
```

五个 GraphTrainer recipe 均使用 `num_mtp_layers=0` 和 `memory_policy="full"`。

### 2.2 切换 DSV4 专用策略

```bash
--compile.memory-policy dsv4-mhc
```

`dsv4-mhc` 不是默认策略。建议先用 `full` 完成数值和编译验证，再在相同模型、batch、并行度和 step 数下对比显存与吞吐。

如需额外保存 MoE expert 激活、减少反向 W13 或 W1/W3 GEMM replay，可使用：

```bash
--compile.memory-policy dsv4-mhc-moe-save
```

### 2.3 配置类型要求

NPU ConfigManager 会跳过 `GraphTrainer.Config` 的普通包装，相关代码位于：

```text
torchtitan_npu/config/manager.py
```

如果误转成普通 `TrainerConfig`，训练会退回 `TrainerEx`，GraphTrainer 的联合图和重计算策略不会生效。

## 3. 执行流程

```mermaid
flowchart LR
    A["DSV4 recipe"] --> B["TP / EP"] --> C["SimpleFSDP"] --> D["首个 batch make_fx"]
    D --> E["joint forward/backward FX graph"] --> F["FQN/FSDP annotation"]
    F --> G["memory policy"] --> H["save/recompute partition"]
    H --> I["AOTAutograd + Inductor"] --> J["后续 batch replay"]
    J --> K["图外 optimizer.step"]
```

首个 batch 完成 trace、重计算分区和编译后，GraphTrainer 会缓存 TracedResult。后续 batch 复用已捕获的图和重计算策略，只更新实时输入、参数和 metadata。
后续 batch 必须满足首次捕获时的输入契约：
- pytree/input 结构保持不变；
- dtype、device 和未标记的静态 shape 保持不变；
- ratio 集合、metadata 字段结构和模块结构保持不变；
- 只有预先标记为 dynamic 且满足约束的维度可以变化。
当前 replay 路径不会因为输入契约变化而自动重新 trace。契约变化会导致 guard/replay 失效，必须显式重新捕获和编译；

## 4. 重计算实现

### 4.1 上游 `full` 策略

上游 full policy 默认使用一层一保存输出的策略，其余节点默认使用重计算方式。

代码位置：

```text
torchtitan/experiments/graph_trainer/memory_policy.py
```


### 4.2 NPU 策略框架

```text
torchtitan_npu/patches/torchtitan/experiments/graph_trainer/memory_policy.py
```

NPU 扩展使用 `NodePolicyKey` 按 FX target、模块 FQN 和 occurrence 定位节点，并提供 layer boundary、`MUST_SAVE`、反向节点跳过、`lm_head/loss` 跳过和 SymInt 保存等规则。

### 4.3 `dsv4-mhc` 策略

```text
torchtitan_npu/models/deepseek_v4/memory_policy.py
```

该策略以 `full` 为基线，当前主要做三件事：

1. 保存 `layers.*.moe` 中第一次出现的 `aten.add.Tensor`，保留已验证的 MoE 输出汇合边界；
2. 保留 `reshard_after_forward=False` 时的 FSDP 强制节点；
3. `save_input_every_n_layers=1`，每层建立跨层保存边界。

当前不再保存 `layers.*.attention.wo_b` 或 mHC post 输入，attention 到 mHC post 保持连续重算；router 的 gate、归一化、top-k 和 dispatch metadata 也保持为一个完整重算区域。

```mermaid
flowchart TD
    A["FX node"] --> B{"FSDP 强制保存?"}
    B -- "是" --> S["MUST_SAVE"]
    B -- "否" --> C{"命中 MoE 汇合?"}
    C -- "是" --> S
    C -- "否" --> D{"输出跨越层边界?"}
    D -- "是" --> S
    D -- "否" --> F["沿用 full 重算决策"]
```

规则依赖 FQN、算子 target 和 occurrence，模型重构后可能静默失配。



### 4.4 `dsv4-mhc-moe-save` 策略

该策略继承 `dsv4-mhc`，并额外保存 shared/routed experts 的激活：

- 开启 `swiglu_group` 时，保存 SwiGLU 实际接收的唯一 `2F` 输入，可兼容量化/反量化中间链；
- 未开启 `swiglu_group` 时，分别保存 W1、W3 投影输出；
- 不为 router 增加局部保存点，避免 expert 激活与 routing metadata 来自不同 replay 区域。

保存点在 remat 和 EP chunk pass 之前识别。该策略以更多激活显存换取更少的 expert GEMM replay。

### 4.5 前向变异算子的重算保真

原生 SAR 按数据流回放重算节点，无数据输出的前向变异算子（如 partial-RoPE 对
clone 的原地旋转）不会进入回放链，反向会读到未旋转的值（上游 issue
[#4688](https://github.com/pytorch/torchtitan/issues/4688)）。DSV4 GraphTrainer 默认
启用上游 PR [#4708](https://github.com/pytorch/torchtitan/pull/4708) 的
`functionalize_recompute_mutations_pass`（见 2.1）：在 memory-policy 打标之后、
CPU offload 与 SAR 之前把变异写入显式化为数据流，原生 SAR 随之自然重算；对无可
重算变异的图是 no-op。已验证 A3（CANN w0902，backward 走临时 shim）2p/8p 精度
对齐；A5 原生反向未验证。固定 TorchTitan 依赖自带等价 pass 后，删除本 backport
与 `pass_pipeline` 设置即可。

## 5. 支持边界与风险
- `dsv4-mhc` 依赖 FQN、FX target 和 occurrence，模型或编译器升级后必须检查规则命中数。
- `dsv4-mhc-moe-save` 的融合路径要求 `swiglu_group` 有唯一的 `2F` 输入；未融合 fallback 依赖 W1/W3 FQN 或受支持的 GEMM target。
- `dsv4-mhc-moe-save` 会提高激活显存，应同时比较峰值显存、吞吐和数值。
- 变异算子重算保真依赖上游 #4708 的 functionalization 语义与原生 SAR 的交互；升级
  PyTorch 或 TorchTitan 后需重跑 `test_rope_recompute_integration.py` 的正/负对照。
- 重计算链路依赖 FX tracer、AOTAutograd、Dynamo dynamic annotation 和 Inductor 私有 API，升级 PyTorch、torch-npu 或 TorchTitan 后需重新回归。

## 6. 关键文件

| 功能 | 文件 |
| --- | --- |
| DSV4 GraphTrainer 配置 | `torchtitan_npu/models/deepseek_v4/config_registry.py` |
| 通用 full policy | `torchtitan/experiments/graph_trainer/memory_policy.py` |
| NPU policy 框架 | `torchtitan_npu/patches/torchtitan/experiments/graph_trainer/memory_policy.py` |
| `dsv4-mhc` / `dsv4-mhc-moe-save` 策略 | `torchtitan_npu/models/deepseek_v4/memory_policy.py` |
| FX tracer/replay | `torchtitan/experiments/graph_trainer/make_fx_tracer.py`、`torchtitan/experiments/graph_trainer/trainer.py` |
| SimpleFSDP | `torchtitan/experiments/graph_trainer/simple_fsdp.py` |
| 函数式 compressor | `torchtitan_npu/models/deepseek_v4/compressor.py` |
| 变异重算保真（functionalization backport） | `torchtitan_npu/patches/torchtitan/experiments/graph_trainer/functionalize_recompute_mutations.py` |
| MoE 保存策略单测 | `tests/unit_tests/models/deepseek_v4/memory_policy/test_swiglu_memory_policy.py` |

## 7. 结论

DeepSeek-V4 GraphTrainer 重计算是在联合 FX 图上进行节点级 save/recompute 决策，并与 SimpleFSDP 参数通信协同。`full` 是显存优先的通用基线，`dsv4-mhc` 保存 MoE 汇合，`dsv4-mhc-moe-save` 再保存 expert W13 或 W1/W3 激活以减少 GEMM replay。实际启用前应同时验证数值、峰值显存、重编译次数和稳态吞吐；不能只以训练成功启动作为结论。
