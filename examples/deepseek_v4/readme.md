# DeepSeek-V4 training examples

本目录提供选择完整 Python recipe 的启动入口。模型规模、序列长度、并行策略和量化方式由 `torchtitan_npu_recipes.examples.deepseek_v4` 中的配置函数定义，自定义组合使用 `MODULE` / `CONFIG` 选择，见[Python 配置指南](../../docs/developer_guides/python_recipes.md)。

示例默认使用非 golden 的 NPU 融合算子组合，参考组合需显式设置 `USE_GOLDEN=1`。A3 使用 CANN 1104 和 torch-npu 1105.3，下载与安装见[安装指南](../../docs/user-guides/installation.md)。

## 入口关系

Flash 多机训练以 `deepseek_v4_flash_cpt_4k_a3.sh` 为公共基线：

```text
deepseek_v4_flash_cpt_4k_a3.sh
└── deepseek_v4_flash_cpt_4k_a5.sh
    └── deepseek_v4_flash_cpt_1024k_a5.sh
```

- `deepseek_v4_flash_cpt_4k_a3.sh`：公共训练参数、override、checkpoint、optimizer 等基线配置，默认选择 `deepseek_v4_flash_cpt` recipe，显式启用 model 编译和 checkpoint，backend 为 `inductor`。
- `deepseek_v4_flash_cpt_4k_a5.sh`：增加 A5 运行环境和默认 block-FP8 量化，其他配置复用 A3 基线。
- `deepseek_v4_flash_cpt_1024k_a5.sh`：继续复用 A5 4K 入口，只通过 CLI 覆盖 1M 所需的 CP、DP、序列长度和 global batch size，量化默认随 A5 4K 入口继承。
- 用户普通 CLI 参数位于默认选项之后，因此可以覆盖默认值（A5 的 activation-checkpoint 子命令仍在末尾），包括使用 `--extension.quantization.no-enable-quantized-training` 切换到 BF16。

## 编译配置

编译和 checkpoint 的启用状态在 Python recipe 中定义。A3 8P Flash CPT debug
默认选择 `deepseek_v4_flash_8p_cpt_4k_a3_eager`，使用 `compile=None` 关闭模型编译。
其他 Flash 配置函数保留各自的编译配置；配置存在不表示当前 NPU trainer 支持该编译路径。
更改编译 backend、启用或关闭组件时直接修改 recipe，不使用通用字段 CLI。

例如将以下函数保存为仓库根目录的 `my_recipes.py`，再通过 `MODULE=my_recipes CONFIG=my_run` 选择：

```python
from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_flash_cpt

def my_run():
    config = deepseek_v4_flash_cpt()
    config.compile = None
    config.checkpointer = None
    return config
```

自定义 recipe 关闭 checkpoint 时，应选用未附加 checkpoint 字段的 debug launcher，
或直接使用 `scripts/run_train.sh` 并在 recipe 中配置训练参数。Pro、CP 等其他入口需要
编译时，也在对应 recipe 中显式设置 `CompileConfig`，再通过 `MODULE` / `CONFIG` 选择。

## 常用环境变量

运行前至少根据环境设置以下路径和节点信息：

```sh
export NODE_IPS="192.168.1.10,192.168.1.11,..."
export HF_ASSETS_PATH=/path/to/DeepSeekV4_tokenizer
export CKPT_SAVE_LOAD_PATH=/path/to/save_or_resume_ckpt
export CKPT_INIT_LOAD_PATH=/path/to/init_hf_ckpt
```

A5 默认每节点使用 8 张 NPU，A3 默认每节点使用 16 张 NPU；需要改变时设置 `NGPU`。
训练参数在基础脚本中直接设置；派生入口通过 CLI 覆盖不同的默认值。命令末尾的
`--training.max-context-length`、`--training.num-tokens-per-microbatch-per-dp-rank` 和
`--training.num-tokens-per-train-step` 可覆盖相应参数。单独修改上下文长度不会重算 microbatch，
单独修改 microbatch 也不会重算每步预算；需要调整时应同时指定相关参数。
每步预算须能被「每 rank microbatch token 数 × 有效 DP 度 × PP microbatch 数」整除。
例如 8 卡 Flash debug 的 A3 默认梯度累积 8 次，A5 默认梯度累积 4 次。

## 裁剪 Debug 模型

Debug/performance 验证优先使用等价裁剪模型，避免为了快速验证直接启动完整规模模型。这里统一放置不同规模的裁剪 Debug 入口，卡数不是这一节的分类依据。

### 1P mini 模型

固定 TorchTitan 基线使用展平的 token 布局：输入与标签为 `[T]`，隐藏状态为
`[T, D]`。此入口默认每步处理 2048 个 token，最大上下文长度为 2048。
数据集及源文件由 `MODULE` / `CONFIG` 选择的 recipe 配置，入口不再读取
`DATASET` / `DATASET_PATH`。自定义数据源通过独立 recipe 指定。

`USE_GOLDEN=1` 选择模型默认的参考 attention、实数缓存 RoPE 和 AdamW。
在当前 Ascend A3 环境中，此组合已完成单卡、确定性 eager 前向、反向及
optimizer step；参考 indexer 的离散选择不提供 indexer 训练目标。
默认融合组合使用 Muon。融合算子组合另以 AdamW 完成两步功能验证；当前
CANN 的 `SparseFlashMlaGrad` 拒绝确定性模式，因此该功能验证未启用
`--debug.deterministic`，不能作为数值对齐证据。Muon、多卡及编译仍需独立验证。

```sh
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh

USE_GOLDEN=1 HF_ASSETS_PATH="$PWD/tests/assets/deepseek_v3" \
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh \
  --training.steps 2 --debug.seed 42 --debug.deterministic
```

token 预算使用 `--training.num-tokens-per-microbatch-per-dp-rank` 和
`--training.num-tokens-per-train-step`；两者均统计 token slot。模型上下文
在 registry 构造时选定，recipe 同步设置 `training.max_context_length`。
单独覆盖训练上下文不会重新配置 RoPE；修改上下文长度需使用匹配的 recipe。
其他入口的迁移与验证状态独立于本单卡 mini 记录。

### 8P Flash 16-expert Debug

默认使用 43 层、16 个专家、4K 上下文、Muon 优化器和非 golden 融合算子，以 eager 模式运行。
训练预算与 `upstream/dev_0.4.0` 的 A3 8P 脚本一致：每卡每个 microbatch 4096 个 token，
每步 262144 个 token，共 100 步，并启用全量激活重计算。
`USE_GOLDEN` 只切换算子组合，不改变优化器或文档打包上限。
参考组合不提供 indexer 训练梯度，使用 Muon 会报 `missing gradients`；需要该组合时，
应在自定义配置中显式选择 AdamW，不能作为本用例的 Muon 训练对照。

```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh
```

在 A3、CANN 1104 与 torch-npu 1105.3 上，修复前的完整 4K 配置已完成 50 步训练，
峰值显存约 `40.19 GiB`，未出现 OOM。与 `upstream/dev_0.4.0` 固定版本
`775fdcd5036de8c3311fb2eeaa45e1258f387227` 的 eager 运行比较时，使用同一份
FP32 初始权重、相同输入、seed 42、严格确定性检查，并关闭 HF32。
10 步比较中，loss 和 grad_norm 的最大相对误差分别为 `0.117694%` 和 `0.239883%`，
两项均满足仓库的数值近似阈值，比特一致未通过。进一步的完整 50 步比较中，
两项的最大相对误差分别为 `0.575068%` 和 `4.943348%`，平均绝对误差分别为
`0.02604002` 和 `0.12830677`，均未通过数值近似阈值。8 个 rank 每侧各消费
400 个 microbatch，输入全部一致；grad_norm、loss 分别从第 17、36 步开始逐步超限。
首个前向差异已定位到 Q head 归一化：旧版使用融合 RMSNorm，当前上游使用
BF16 算式，中间舍入不同。这些 50 步数据属于修复前版本，不能作为修复后的精度结论。

当前已恢复可配置的 Q head 归一化节点，由非 golden 组合启用融合 RMSNorm；
同时修正 Muon 的两处迁移遗漏：`wo_a` 按每组 `o_lora_rank` 分块，
`compressor_128` 的投影与 APE 参数使用与旧版相同的 Muon 配置。
修复后以相同的完整配置比较第 1 步，loss 均为 `12.281325340270996`，
grad_norm 均为 `9.061148643493652`；8 个 rank 各 1199 个优化器参数的梯度
与更新后值逐字节一致。参数比较仅统一新旧名称，并移除 attention sink 新增的尾部单维。
捕获的第 0 层 attention 输出梯度仍有正负零的位表示差异，数值差为 0；
因此该结论限定为首步训练指标、参数梯度和参数更新，不表示所有中间张量的原始位模式相同。
修复后的 10/50 步尚未验证，不能由首步结果推断多步精度或完整收敛。
此前关闭激活重计算的 4K 配置曾出现 OOM，该记录不适用于当前全量重计算配置。
训练步数、上下文长度和 token 预算统一在自定义 Python 配置中设置，
本入口不再读取 `STEPS`、`MAX_CONTEXT_LENGTH`、`TOKENS_PER_MICROBATCH`、
`TOKENS_PER_STEP` 或 `MAX_NUM_DOCUMENTS`。

运行前按安装指南设置 `ASCEND_SET_ENV_PATH` 指向 1104 CANN。需要 profiling 时在自定义 Python 配置函数中设置 `config.profiler.enable_profiling=True`。

### 32P Pro 32-expert Debug

Pro 32-expert 是独立的裁剪模型 debug/performance 入口，不与 8P Flash launcher 强制复用：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --training.steps 20
```

128K 场景不新增 launcher，只通过 CLI 改变并行和训练配置：

```sh
CP=32 NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --parallelism.data-parallel-shard-degree 1 \
  --parallelism.data-parallel-replicate-degree 1 \
  --training.max-context-length 131072 \
  --training.num-tokens-per-microbatch-per-dp-rank 131072 \
  --training.num-tokens-per-train-step 1048576 \
  --training.steps 20
```

A5 脚本默认启用 block-FP8；如需 BF16，请在命令末尾追加后文的禁用选项。

### 导出和加载 Debug Hugging Face checkpoint

debug 默认关闭 checkpoint。先在 `my_recipes.py` 中定义需要 checkpoint 的配置：

```python
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_flash_43layers_16experts_compiled

def debug_checkpoint():
    config = deepseek_v4_flash_43layers_16experts_compiled()
    config.checkpointer = CheckpointManager.Config(load_only=True)
    return config
```

导出 checkpoint：

```sh
MODULE=my_recipes CONFIG=debug_checkpoint \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --checkpointer.folder ./export_ckpt \
  --checkpointer.no-load-only \
  --checkpointer.last-save-in-hf \
  --training.steps 1 \
  --debug.seed 42 \
  --debug.deterministic
```

从导出的 HF checkpoint 冷启动时，使用新的或空的 `checkpointer.folder`，避免已有 step checkpoint 优先触发 resume：

```sh
MODULE=my_recipes CONFIG=debug_checkpoint \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --checkpointer.folder ./load_ckpt \
  --checkpointer.initial-load-path ./export_ckpt/checkpoint/step-1 \
  --checkpointer.initial-load-in-hf \
  --training.steps 1 \
  --debug.seed 42 \
  --debug.deterministic
```

## 多机完整模型训练

### A3 4K

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh \
  --training.steps 500
```

### A5 4K

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
  --training.steps 20
```

### A5 64K

64K 不需要新增 launcher，直接在 A5 4K 入口上覆盖训练和并行参数：

```sh
CP=8 NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
  --parallelism.data-parallel-shard-degree 16 \
  --parallelism.data-parallel-replicate-degree 1 \
  --training.max-context-length 65536 \
  --training.num-tokens-per-microbatch-per-dp-rank 131072 \
  --training.num-tokens-per-train-step 8388608 \
  --training.steps 20
```

### A5 1M

1M wrapper 只提供该场景的默认 CLI 覆盖；仍可在命令末尾继续覆盖：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_1024k_a5.sh \
  --training.steps 20
```

等价的主要差异参数为：

```sh
CP=128 \
--parallelism.data-parallel-shard-degree 1 \
--parallelism.data-parallel-replicate-degree 1 \
--training.max-context-length 1048576 \
--training.num-tokens-per-microbatch-per-dp-rank 1048576 \
--training.num-tokens-per-train-step 8388608
```

## A5 量化与 BF16

所有 A5 脚本（`*_a5.sh`）默认启用 block-FP8 量化训练，等价于自动追加：

```sh
--extension.quantization.enable-quantized-training \
--extension.quantization.recipe all_block_fp8
```

例如 A5 64K、1M 和 Pro 32P 命令均默认使用该量化配置。其中 Flash CPT 的 A5 入口（多机 4K 及其派生 64K/1M、8P debug）在此基础上还默认启用 FSDP 权重预量化（`--extension.quantization.enable-fsdp-prequantize`）。多机 4K 入口默认 DP_SHARD=128，64K/1M 通过 CP/DP 覆盖得到的有效 FSDP 切分度同为 128；该切分度下均分切分的投影（wq_a、wkv、wo_b、shared experts）无法满足预量化的 64 对齐约束、启动即报错，因此 4K 入口将 `fsdp-prequantize-fqns` 收窄为任何切分度下都保持对齐的投影（wq_b、wo_a、indexer.wq_b、routed experts）。8P debug 入口在 DP_SHARD=8 下使用 recipe 默认白名单。对齐约束及“结构性 no-op 回退 / 对齐失败报错”的区别详见 `docs/feature_guides/low_precision_training.md`。

需要保留 BF16 训练方案时，在命令末尾追加以下选项，覆盖脚本默认值：

```sh
--extension.quantization.no-enable-quantized-training
```

A3 脚本不默认启用量化；如需 A3 量化训练，请显式传入上面的 enable 和 recipe 参数。

## Profiling

需要采集 profile 时显式开启：

```sh
--profiler.enable-profiling
```

公共 A3 基线提供如下 schedule，A5 Flash 直接继承，用户仍可通过 CLI 覆盖：

```sh
--profiler.profile-freq 1 \
--profiler.profiler-warmup 0 \
--profiler.profiler-active 1 \
--profiler.profiler-repeat 1 \
--profiler.profiler-skip-first 4
```

如需不依赖 recipe 固定单次采样窗口，请在命令末尾完整追加：

```sh
--profiler.enable-profiling \
--profiler.profile-freq 10 \
--profiler.profiler-warmup 3 \
--profiler.profiler-active 1 \
--profiler.profiler-repeat 1 \
--profiler.profiler-skip-first 0
```

## Checkpoint 说明

Flash 多机 CPT 基线默认启用 `--checkpointer.load-only`，即只负责加载，不会保存新的训练 checkpoint。需要保存时必须显式覆盖：

```sh
--checkpointer.no-load-only
```

当 `checkpointer.folder` 已存在有效的 `step-*` checkpoint 时，TorchTitan 会优先 resume，该情况下 `checkpointer.initial-load-path` 不作为冷启动来源；要从 `initial-load-path` 冷启动，请使用新的或空的 `checkpointer.folder`。

## LoRA 微调

完整 Flash 模型的 BF16 LoRA 训练复用 A3 4K 公共入口，节点、数据和基座路径使用上面的环境变量：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_lora_4k_a3.sh \
  --optim.optimizer.name AdamW \
  --training.steps 500 \
  --checkpointer.interval 100
```

该入口使用 `deepseek_v4_flash_lora_compiled` recipe（dense/expert rank 16、alpha 32），并开启 checkpoint 保存。优化器沿用 CPT 默认值，传入 `--optim.optimizer.name` 可覆盖。周期 checkpoint 用于续跑，最后一步导出 PEFT adapter。模型规模、并行配置和其余默认参数继承 A3 公共入口，命令末尾的 CLI 参数仍可覆盖默认值。A5 使用 `CONFIG=deepseek_v4_flash_lora_compiled bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh --checkpointer.no-load-only --checkpointer.save-training-state`，继承 A5 CPT 的 block-FP8 默认配置。

配置、部分参数训练、native checkpoint 恢复、PEFT 导出和离线合并见 [LoRA 功能指南](../../docs/feature_guides/deepseek_v4_lora.md)。
