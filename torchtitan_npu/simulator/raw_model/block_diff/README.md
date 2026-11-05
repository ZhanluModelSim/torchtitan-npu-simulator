# Block diffusion：训练与推理负载参考

这个目录提供独立的 PyTorch 模型，供数值核对、meta 形状捕获和 NPU 成本模型
适配使用。核心只依赖 PyTorch 和标准库，不需要导入 `torchtitan_npu` 来运行。
它不是已完成原生模拟器接入的 `ModelSpec`；现有
[`models/block_diffusion`](../../../models/block_diffusion) 训练计算代理保持独立。

## 文件与接口

| 文件 | 职责 |
| --- | --- |
| `model.py` | 共享 dense/MoE 网络，显式训练和推理入口 |
| `workload.py` | 推理阶段、TP/EP 分片、通信与采样的语义描述 |
| `training_workload.py` | 实际 CPU/meta forward、backward 和 AdamW 算子捕获 |
| `legacy_state_dict.py` | 原始 stacked-expert 权重格式的显式双向转换 |
| `example.py` / `export_example.py` | 推理回放与语义 JSON 导出 |
| `tests/` | 独立数值参考、梯度、状态、分片和转换验收 |

从本目录导入时使用：

```python
from model import BlockDiffusionModel, ModelConfig, TrainingSpec
```

从仓库根目录运行下面的脚本不会触发 NPU 插件初始化：

```bash
python torchtitan_npu/simulator/raw_model/block_diff/example.py --device cpu --moe
python torchtitan_npu/simulator/raw_model/block_diff/example.py --device meta --moe
python torchtitan_npu/simulator/raw_model/block_diff/export_example.py --tp 2 --ep 4 --output /tmp/block-diff-inference.json
python torchtitan_npu/simulator/raw_model/block_diff/training_workload.py --device meta --optimizer adamw --output /tmp/block-diff-training.json
```

## 推理

`model(input_ids, cache=..., phase=..., attention_policy=..., logits_positions=...)`
执行一次前向，返回 `PhaseOutput(logits, cache)`。

- `prefill` 和 `commit` 返回追加后的逐层 KV；输入 cache 不被修改。
- `denoise` 读取前缀，重算当前块，不提交 KV。
- `attention_policy` 明确选择前缀可见、块内 causal 或 bidirectional。
- `logits_positions=None` 计算全部位置，空 tuple 跳过输出头；tuple 选择所有
  请求共享的位置，输入设备上的 int64 `[B,H]` tensor 选择各请求自己的位置。
  位置选择发生在 final norm 和输出头之前。
- 步数、token 更新、停止和 commit 策略属于调用方，不再默认所有算法都为
  `D+1` 次前向。模型没有内置置信度采样循环。

CPU MoE 可以使用真实 top-k 路由。`RoutingPlan` 提供可复现的逐 token 专家
选择，用于 meta 捕获或特定分布回放；专家按 token 分组计算，不展开逐 pair
权重。可选 shared expert、权重共享与 KV 浮点存储精度均实际生效。

推理语义适配层的 `build_schedule` 支持多个块、每轮接受数、零接受轮次和
末块无后续消费者时省略 commit。未确定位置由 bool 状态动态压缩成每条请求
自己的索引映射，不把前 U 个位置伪装成真实未确定位置。一个批次的形状与
接受计数相同，异步请求调度不在本目录中实现。

`ParallelSpec` 只支持明确的 batch-sharded EP × TP：请求按完整 batch 行分给
EP；attention heads、词表和所有 FFN intermediate 沿 TP 切分；专家沿 EP
分配。dispatch/combine、TP 归约、词表采样归约及路由元数据分别记录。允许
没有本地请求但仍接收专家任务的 rank。它不是原生 TorchTitan 并行支持矩阵；
独立 ETP、CP、PP、FSDP 和 KV-head 复制未实现，会明确拒绝不支持的布局。

## 训练

`model.forward_training(input_ids, labels, spec=TrainingSpec(...), routing_plan=...)`
执行可反向传播的完整窗口前向，不复用或 detach 推理 KV。`forward()` 本身仍是
无梯度推理入口；仅调用 `model.train()` 不会切换到训练接口。

```python
import torch
from model import BlockDiffusionModel, ModelConfig, TrainingSpec

model = BlockDiffusionModel(ModelConfig())
tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
labels = torch.full_like(tokens, -100)
labels[:, -2:] = tokens[:, -2:]
corrupted = tokens.clone()
corrupted[:, -2:] = 0
output = model.forward_training(corrupted, labels, spec=TrainingSpec(block_size=4))
output.loss.backward()
```

`TrainingSpec` 提供两种精确 mask，要求序列长度为正的 block-size 整数倍：

- `block_causal`：可以看到所有先前块与当前整个块。
- `prefix_causal_canvas_bidirectional`：前缀内部逐 token causal，最后一个
  canvas 对前缀和自身双向可见。

标签与输入位置相同，不做 AR 的 next-token shift；由调用方将未监督位置设为
`ignore_index`。loss 是 FP32 cross-entropy 求和除以有效标签数；没有有效标签时
返回可微分的零值。`TrainingOutput` 同时保留 `loss_sum` 和 `valid_tokens`，
跨 rank 的全局归一化由训练驱动负责。训练输出头计算整个窗口，不隐式减少
ignored positions 的投影工作。

`capture_training_step` 通过实际执行记录 forward、backward 和可选 AdamW，
不会用“全部参数 FLOPs × 固定倍数”代替 MoE 的真实反向图。合成输入的 mask
位置和路由是明确的固定回放；这不代表训练数据分布或模型收敛。AdamW 为实际
执行的非 fused、非 foreach 单次更新，其超参数写入 JSON。零目标批次是否跳过
optimizer 由训练驱动决定；选择 AdamW 时捕获器仍会执行该更新。

捕获器显式使用 PyTorch SDPA math backend，包含它实际产生的 GQA 展开等操作。
这些是参考分解，不等同于融合 NPU attention 的流量。输出包括张量身份、存储
别名、stride、原地写入、已知矩阵 FLOPs、实际梯度与优化器状态容量。
未建模算子的成本为 `null`，不是零。为保持身份稳定，捕获期间保留张量引用，
因此不能从该执行的张量寿命推导正常运行的峰值显存。

## 原始接口与 checkpoint 迁移

本次更新将原始的 `BlockDiffusionArgs` / `RuntimeConfig` / `BlockDiffusionLM`
及隐式 Zhanlu 注册改为显式 `ModelConfig` / `TrainingSpec` / `BlockDiffusionModel`。
旧 Python 调用方需要迁移；本文件不会再自动注册 `MindIBlockDiffusion`。
下游注册、kernel 转换及硬件参数应在各自后端适配层完成。

state-dict key 也已更新。不要直接把旧 checkpoint 当成新格式加载：

```python
from legacy_state_dict import from_legacy_state_dict, to_legacy_state_dict

reference_state = from_legacy_state_dict(old_raw_state)
model.load_state_dict(reference_state, strict=True)
native_adapter_input = to_legacy_state_dict(model.state_dict())
```

`native_adapter_input` 保持现有 `BlockDiffusionStateDictAdapter.from_hf()` 使用的
`backbone.*`、stacked `w13/w2` 格式。原生训练模型和 adapter 的实现未变。
转换保留张量值与 dtype，拒绝未知字段、残缺专家或无法表达的多个 shared
experts；它不承诺任意公开 HF checkpoint、旧优化器状态或 DCP 元数据的兼容。

## 验收与适用范围

安装 PyTorch 和 pytest 后，从仓库根目录运行独立测试：

```bash
python -m pytest -q --import-mode=prepend \
  --confcutdir=torchtitan_npu/simulator/raw_model/block_diff/tests \
  torchtitan_npu/simulator/raw_model/block_diff/tests
```

这只隔离 raw 模型测试，不加载根测试配置中的 NPU 插件；它不是原生训练
集成测试的替代品。执行证据见
[`docs/test_guides/block_diffusion_raw_acceptance.md`](../../../../docs/test_guides/block_diffusion_raw_acceptance.md)。

容量、逻辑 KV 读写、collective payload、实际 HBM 与链路流量分别计量。推理
JSON 是有序语义事件；训练 JSON 是实际参考执行轨迹。两者都还需要目标后端
算子映射和调度才能得到 NPU 时延。原生训练路径中的 `attention_compute_alpha`
计算代理未移植到本模型，也不能用来缩放推理 KV 容量。
