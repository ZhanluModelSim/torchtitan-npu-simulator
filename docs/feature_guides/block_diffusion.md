# Block Diffusion 契约、实现与工具

Block Diffusion 的规范由
[`raw_model/block_diff/README.md`](../../torchtitan_npu/simulator/raw_model/block_diff/README.md)
和同目录的 [`model.py`](../../torchtitan_npu/simulator/raw_model/block_diff/model.py)
定义，当前 `MODEL_CONTRACT_VERSION=1`。raw 目录只保留这两个文件。

`models/block_diffusion` 是独立维护的 TorchTitan 实现，必须人工遵循 raw 契约；
运行时不得导入 raw 或工具 loader。工具与测试可以加载规范实现，用于回放、
算子捕获和比较，不构成 native 的继承层或运行时适配层。

历史 native 接入基线是 `feat/npu-simulator` 的
`3eeed771d70b733f4f7eadd60ccd38bb77ac3aea`；本次参考重构从
`feat/npu-simulator-local` 的 `4f200e8abd318a0a813061d9f3946e85dcbc2e03` 开始。
原始 raw 文件的固定来源见规范 README。这些来源不构成公开 checkpoint 兼容声明。

## 目录职责

| 位置 | 职责 |
| --- | --- |
| `torchtitan_npu/simulator/raw_model/block_diff/` | 规范 README 和独立参考 model.py |
| `torchtitan_npu/models/block_diffusion/` | 人工维护的 native 模型、配置、并行计划及权重适配 |
| `scripts/block_diffusion/raw_reference.py` | 不触发 NPU 插件初始化的独立参考 loader |
| `scripts/block_diffusion/workload.py` | 推理调度、分片、通信和采样的语义事件 |
| `scripts/block_diffusion/training_workload.py` | 实际 CPU/meta 前向、反向和 AdamW 算子捕获 |
| `scripts/block_diffusion/legacy_state_dict.py` | 旧 stacked-expert 权重格式的显式双向转换 |
| `tests/unit_tests/models/block_diffusion_reference/` | 数值、状态、工作量及 raw/native 对照验收 |

原始规模是 97 层、hidden 8192、64/8 query/KV heads、head dim 128、词表
262144、1024 routed experts、top-8、routed intermediate 4096、一个宽度
14336 的 shared expert。它是一个历史配置，不是范式的固定结构。规范支持
不同规模、dense、可选 shared experts 和 tied embeddings；native 能覆盖哪些
组合必须由配置校验和比较测试明确说明。

## 训练与推理的使用范围

训练使用完整窗口的精确 `block_causal` 或
`prefix_causal_canvas_bidirectional` mask、同位置 masked-token labels 和
有效标签数归一化。推理通过 prefill／denoise／commit、逐层 KV 和显式输出位置
表达工作量。精确条件、形状和精度规则以 raw README 为准。

Native 现在显式区分 `attention_mode="exact"` 与 `"compute_proxy"`。
Exact 按 raw v1 的两种 mask 分段计算，使用无缩放 split-half RoPE、FP32 router
和有效标签归一化；alpha 必须为 1。原来的 flavors 保留 compute-proxy 模式，
其全长 causal 前向和缩放 ledger 不能当成 raw v1 的精确数值结果。

Exact 的 `block_causal` 为每块的 query 调用一次对完整已知前缀及当前块的
双向 attention；`prefix_causal_canvas_bidirectional` 则是 causal prefix 加
双向 canvas 两段。CPU attention、NPU converter 和 simulator shim 共享此
native 分段定义。精确 ledger 保留各段真实 Q/KV 长度，GQA 反向显式记录
query-head contributions 到 KV heads 的归约，布局转换与 matmul 分开。

| 路径或能力 | 验收范围 |
| --- | --- |
| 独立 raw CPU/meta 训练和推理 | 已有数值、梯度、状态与工作量测试；见参考验收记录 |
| native exact 训练 logits／loss／全部参数梯度 | 已通过真实 TorchTitan 单 rank FP32 对照，覆盖 dense、一个 shared expert 的 MoE 与两种策略 |
| native converter 保持两种精确 mask | CPU 硬件入口仿真及 native meta ledger 检查通过；未执行真实 NPU |
| native 推理 KV／多轮去噪 | 尚未适配；raw 推理和独立工具的通过不代表 native 推理已实现 |
| native exact + CP | 明确拒绝，尚未实现全局位置与分片所有权语义 |
| TP／EP／ETP／CP／PP／FSDP 等 native 组合 | 各组合单独验证，不从参考单 rank 或代码注册推导通过 |
| 原生 NPU 数值训练、性能和收敛 | 不由本页声明；需对应设备和训练证据 |
| 独立 JSON 到 native simulator IR | 尚需后端映射和调度集成 |
| 公开 Hugging Face checkpoint 数值兼容 | 未声明；历史来源未给出权威 checkpoint 契约 |

## Native 配置与计量

保留 `dense_debug`、`debug`、`reduced`、`full` 的 compute-proxy 行为；使用
`dense_debug_exact`、`debug_exact`、`reduced_exact`、`full_exact` 选择 exact。
新增的训练配置是 `block_diffusion_dense_exact_smoketest` 和
`block_diffusion_exact_smoketest`。Exact dataloader 使用窗口内 `0..S-1`
位置，旧 proxy 的 document-position 规则保持不变。

手工调用 `make_block_diffusion_config` 时指定 `attention_mode="exact"`、
`attention_compute_alpha=1.0` 及所需 `attention_policy`。不得用顶层开关把
不一致的复杂 RoPE 配置静默变成 exact。Exact 禁止 raw v1 未定义的额外专家
bias、路由重缩放、分组选择和辅助负载均衡策略；不支持项会报错。

框架 `get_nparams_and_flops` 返回的第二项已修正为 **每 token** 指标，使用
活跃专家口径，并保留 tied output head 的计算。该指标沿用 TorchTitan 的
dense-attention MFU 估算约定，不是 exact 分段 ledger，也不用于估计峰值
显存或 NPU 时延。

通用 MoE dispatch 的 gate helper 按 gate 权重 dtype 计算，支持 BF16 hidden
与 FP32 router；CPU 外层 autocast 不再降低明确要求的 gate 精度。NPU 融合
attention 的 noncausal 段使用完整可见范围，参数契约依据
[TorchNPU 官方 API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md)。

## 独立工具

从仓库根目录运行，不导入 `torchtitan_npu` 包，不需要原生 NPU 依赖：

```bash
python scripts/block_diffusion/example.py --device cpu --moe
python scripts/block_diffusion/example.py --device meta --moe
python scripts/block_diffusion/export_example.py --tp 2 --ep 4 --output /tmp/block-diff-inference.json
python scripts/block_diffusion/training_workload.py --device meta --optimizer adamw --output /tmp/block-diff-training.json
```

脚本和独立测试通过 `raw_reference` 加载真正的规范文件；同一 checkout 使用
同一个 module cache，避免 dataclass 身份不一致。工具目录加入 `PYTHONPATH`
后也可在 Python 中使用：

```python
from raw_reference import BlockDiffusionModel, ModelConfig, TrainingSpec
from training_workload import capture_training_step
```

`workload.build_schedule` 支持多块、每轮接受数、零接受轮次和末块无后续消费者
时省略 commit。未确定位置由 bool mask 动态压缩成逐请求位置映射。当前一个
batch 的块长与每轮接受数一致；异步 serving 调度未实现。

推理 `ParallelSpec` 的单一布局是 batch-sharded EP × TP：EP 按完整 batch 行
分配请求；attention heads、词表及 FFN intermediate 沿 TP 切分，专家沿 EP
分配。dispatch／combine、TP 归约、词表采样归约和路由元数据分别记录。
空请求 rank 仍可接收专家任务。这不是 native 支持矩阵；独立 ETP、CP、PP、
FSDP 和 KV-head 复制不在该语义工具中实现。

训练捕获实际执行前向、反向和可选 AdamW，不使用固定反向倍数。捕获器显式
选择 PyTorch SDPA math backend，记录其真实参考分解；这不等同于融合 NPU
attention。已知 mm／bmm／addmm FLOPs、实际梯度和 lazy optimizer state 分别
记录，未知算子成本为 null。捕获期间为稳定身份保留 tensor 引用，因此不能
从其寿命推导峰值显存。合成 corruption／路由不代表真实数据分布或收敛。

## 接口及权重迁移

原始 `BlockDiffusionArgs`／`RuntimeConfig`／`BlockDiffusionLM` 和隐式 Zhanlu
注册已替换为规范中的显式接口。旧 Python 调用方需要迁移，不能依赖自动注册。
旧工具路径也已迁移到 `scripts/block_diffusion`，raw 下不提供兼容转发脚本。

旧 stacked `w13/w2` 与新参考 state-dict 的转换必须显式执行：

```python
from legacy_state_dict import from_legacy_state_dict, to_legacy_state_dict

reference_state = from_legacy_state_dict(old_raw_state)
model.load_state_dict(reference_state, strict=True)
legacy_state = to_legacy_state_dict(model.state_dict())
```

转换保留值与 dtype，拒绝未知字段、残缺专家，以及旧格式无法表达的多个
shared experts。它不是任意 HF checkpoint、优化器状态或 DCP 元数据的迁移器。
是否可进一步加载到 native，必须经过对应 state-dict adapter 和严格比较验证。

## 人工维护流程

本范式在 `feat/npu-simulator-local` 维护；从 fork 的同名分支提交上游 PR，
不另外创建 `fix/block-diff-*` 开发分支。

1. 在 raw README 与 model.py 中明确拟修改的结构、mask、状态或精度语义。
2. 语义变化递增 `MODEL_CONTRACT_VERSION`，先验证规范参考本身。
3. 人工更新 native 模型、相关配置、state-dict adapter 与 converter；不得通过
   导入 raw 或复用工具 loader 来规避独立实现。
4. 同配置、同权重比较 logits、loss、参数梯度和适用的 KV 状态，再验证每个
   声明支持的并行／后端组合。记录明确的不支持项，禁止静默降级为计算代理。
5. 更新支持矩阵和执行证据；没有执行的检查保留为待验证。

独立测试命令：

```bash
python -B -m pytest -q --import-mode=prepend \
  --confcutdir=tests/unit_tests/models/block_diffusion_reference \
  tests/unit_tests/models/block_diffusion_reference
```

参考证据见 [raw 验收记录](../test_guides/block_diffusion_raw_acceptance.md)和
[native 对齐验收](../test_guides/block_diffusion_alignment.md)。
[旧 native 接入记录](../test_guides/block_diffusion_acceptance.md) 保留为历史，
不替代契约版本 1 的对照验收。新模型接入与最终判定仍遵循
[开发流程](new_model_onboarding.md)及[验收规范](../test_guides/model_acceptance.md)。
