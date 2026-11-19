# DeepSeek-V4 training examples

本目录提供选择完整 Python recipe 的启动入口。模型规模、序列长度、并行策略和量化方式由 `torchtitan_npu_recipes.examples.deepseek_v4` 中的配置函数定义，自定义组合使用 `MODULE` / `CONFIG` 选择，见[Python 配置指南](../../docs/developer_guides/python_recipes.md)。

示例默认使用非 golden 的 NPU 融合算子组合，参考组合需显式设置 `USE_GOLDEN=1`。A3 使用 CANN 1104 和 torch-npu 1105.3，下载与安装见[安装指南](../../docs/user-guides/installation.md)。

## 入口关系

每个 launcher 选择 `torchtitan_npu_recipes.examples.deepseek_v4` 中的完整配置函数，
不再通过 shell 继承和字段 CLI 拼接训练配置：

| launcher | 配置函数 |
| --- | --- |
| `deepseek_v4_flash_cpt_4k_a3.sh` | `deepseek_v4_flash_cpt_4k_a3` |
| `deepseek_v4_flash_cpt_4k_a5.sh` | `deepseek_v4_flash_cpt_4k_a5` |
| `deepseek_v4_flash_cpt_1024k_a5.sh` | `deepseek_v4_flash_cpt_1024k_a5` |

修改训练参数时，用 `MODULE` / `CONFIG` 选择自定义 Python 工厂。CLI 仅保留
`--print-config`、`--output-dir`、`--resume-step`、`--comm-backend` 和 `--override` 等启动选项。

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

自定义 recipe 通过 `config.checkpointer = None` 关闭 checkpoint，
再由对应 launcher 或 `scripts/run_train.sh` 选择该工厂。Pro、CP 等其他入口需要
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
训练参数由 recipe 设置。构造基础配置时传入 `seq_len` 和 `parallelism`，
并设置 `config.training.num_tokens_per_microbatch_per_dp_rank` 与
`config.training.num_tokens_per_train_step`。单独修改训练字段不会重建模型的 RoPE 或分片声明，
也不会自动重算 token 预算，修改时需保持这些设置一致。
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
`config.debug.deterministic`，不能作为数值对齐证据。该记录仅覆盖所述单卡配置。

```sh
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh

USE_GOLDEN=1 HF_ASSETS_PATH="$PWD/tests/assets/deepseek_v3" \
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh --print-config
```

需要复现两步确定性配置时，在 `my_recipes.py` 中定义：

```python
from torchtitan_npu_recipes.examples.deepseek_v4 import deepseek_v4_mini_1p_cpt_2k_a3


def mini_two_steps():
    config = deepseek_v4_mini_1p_cpt_2k_a3()
    config.training.steps = 2
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
```

```bash
USE_GOLDEN=1 HF_ASSETS_PATH="$PWD/tests/assets/deepseek_v3" \
MODULE=my_recipes CONFIG=mini_two_steps \
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh
```

token 预算使用 `config.training.num_tokens_per_microbatch_per_dp_rank` 和
`config.training.num_tokens_per_train_step`；两者均统计 token slot。模型上下文
在 registry 构造时选定，recipe 同步设置 `training.max_context_length`。
单独覆盖训练上下文不会重新配置 RoPE；修改上下文长度需使用匹配的 recipe。
其他入口的迁移与验证状态独立于本单卡 mini 记录。

### 8P Flash 16-expert Debug

默认使用 43 层、16 个专家、4K 上下文、Muon 优化器和非 golden 融合算子，以 eager 模式运行。
训练预算与 `upstream/dev_0.4.0` 的 A3 8P 脚本一致：每卡每个 microbatch 4096 个 token，
每步 262144 个 token，共 100 步，并启用全量激活重计算。
`USE_GOLDEN` 只切换算子组合，不改变优化器或文档打包上限。
参考 Muon 组合通过现有 `SparseIndexerLoss` 提供 indexer 梯度，系数为 `1e-3`，
无需冻结 indexer 或改用 AdamW。配置与融合路径的区别见 [Muon 指南](../../docs/feature_guides/muon_optimizer.md)。

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

Pro 32-expert 是独立的裁剪模型入口：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh
```

训练步数在对应的自定义 recipe 中设置 `config.training.steps = 20`。
128K 场景需在模型构造前指定上下文长度和并行配置，并同步 token 预算，
通过 `MODULE` / `CONFIG` 选择工厂，不再在命令末尾追加字段参数。
构造顺序见 [Python 配置指南](../../docs/developer_guides/python_recipes.md)。
A5 脚本默认启用 block-FP8，BF16 切换方式见下文。

### 导出和加载 Debug Hugging Face checkpoint

debug 默认关闭 checkpoint。以下工厂保存到仓库根目录的 `my_recipes.py`，
复用 8P eager 示例并显式配置 checkpointer：

```python
from pathlib import Path

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan_npu_recipes.examples.deepseek_v4 import deepseek_v4_flash_8p_cpt_4k_a3_eager


def export_debug_checkpoint():
    config = deepseek_v4_flash_8p_cpt_4k_a3_eager()
    config.training.steps = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    config.checkpointer = CheckpointManager.Config(
        load_only=False, last_save_model_only=True, last_save_in_hf=True
    )
    return config


def load_debug_checkpoint():
    config = export_debug_checkpoint()
    config.checkpointer.load_only = True
    config.checkpointer.initial_load_path = str(Path("export_ckpt/checkpoint/step-1").resolve())
    config.checkpointer.initial_load_in_hf = True
    return config
```

使用参考算子组合导出与加载：

```sh
USE_GOLDEN=1 MODULE=my_recipes CONFIG=export_debug_checkpoint \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh --output-dir ./export_ckpt

USE_GOLDEN=1 MODULE=my_recipes CONFIG=load_debug_checkpoint \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh --output-dir ./load_ckpt
```

两侧模型配置需一致。从 HF 权重冷启动时，使用新的或空的输出目录，
避免已有 step checkpoint 优先触发 resume。HF 权重加载不恢复优化器和训练步数。

## 多机完整模型训练

### A3 4K

```sh
NODE_IPS="${NODE_IPS}" bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh
```

### A5 4K

```sh
NODE_IPS="${NODE_IPS}" bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh
```

### A5 64K

64K 使用自定义 Python recipe，在构造模型前指定 `seq_len=65536`、CP 和 DP，
并设置相应 microbatch 和每步 token 预算。通过 `MODULE` / `CONFIG` 选择该工厂，
不能对已构造的 4K 模型仅修改训练上下文。

### A5 1M

```sh
NODE_IPS="${NODE_IPS}" bash examples/deepseek_v4/deepseek_v4_flash_cpt_1024k_a5.sh
```

1M 工厂声明 CP=128、DP shard=1、上下文和 microbatch token 数均为 1048576、
每步 token 数为 8388608，并保留模型的 CP 迁移检查。
其他步数和训练预算统一在自定义 recipe 中设置。

## A5 量化与 BF16

A5 Flash 和 Pro 示例在 Python recipe 中启用 block-FP8，HiF8 与 QAT 专用入口
使用各自配置。普通 block-FP8 设置为：

```python
config.extension.quantization.enable_quantized_training = True
config.extension.quantization.recipe = "all_block_fp8"
```

Flash A5 工厂还启用 FSDP 权重预量化。完整 Flash 工厂使用显式的
`fsdp_prequantize_fqns` 白名单，以满足参数分片的对齐约束，8P debug 使用默认白名单。
对齐检查见 [低精度训练](../../docs/feature_guides/low_precision_training.md)。

需要 BF16 时，在所选工厂返回前设置：

```python
config.extension.quantization.enable_quantized_training = False
```

A3 示例默认不启用量化。自定义量化配置也直接在 recipe 中设置。

## Profiling

在自定义 recipe 中启用 profiling，并设置采样窗口：

```python
config.profiler.enable_profiling = True
config.profiler.extension.profiler_start = 6
config.profiler.extension.profiler_end = 7
config.profiler.extension.profile_ranks = [0]
```

上述设置沿用示例工厂的 NPU profiler 扩展，训练步数需覆盖所选窗口。
使用 `MODULE` / `CONFIG` 选择该配置。

## Checkpoint 说明

Flash 多机 CPT 基线的 `config.checkpointer.load_only=True`，只加载，不保存新的 checkpoint。
需要保存时，在 recipe 中设置：

```python
config.checkpointer.load_only = False
config.checkpointer.interval = 100
```

若基础工厂的 `checkpointer=None`，先创建 `CheckpointManager.Config`。
当输出目录内存在有效的 step checkpoint 时，会优先 resume；从
`config.checkpointer.initial_load_path` 冷启动需使用新的或空的输出目录。
已有 checkpoint 配置可通过 `--resume-step` 选择恢复步数。

## LoRA 微调

完整 Flash 模型的 BF16 LoRA 训练复用 A3 4K 公共入口，节点、数据和基座路径使用上面的环境变量：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_lora_4k_a3.sh
```

该入口选择 `torchtitan_npu_recipes.examples.deepseek_v4.deepseek_v4_flash_lora_4k_a3`，基座为 compiled LoRA 模型（dense/expert rank 16、alpha 32），并开启 checkpoint 保存。优化器列表及超参数直接在 Python recipe 中配置；修改步数和保存间隔也应修改 recipe，通过 `MODULE` / `CONFIG` 选择自定义配置。周期 checkpoint 用于续跑，最后一步导出 PEFT adapter。A5 量化 LoRA 需在对应 A5 recipe 中配置 LoRA transform 和检查点管理器。

配置、部分参数训练、native checkpoint 恢复、PEFT 导出和离线合并见 [LoRA 功能指南](../../docs/feature_guides/deepseek_v4_lora.md)。
