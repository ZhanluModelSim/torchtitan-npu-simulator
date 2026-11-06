# 数值一致性检查（精度对比）

本目录提供三类精度对比：局部模块前反向、Transformers 推理基线对齐、两个训练脚本或 Git 版本的训练指标对比。
整网对比以 DeepSeek-V4（DSV4）为例；**只比较训练 `loss` 和 `grad_norm` 时，直接看「步骤四」即可，无需准备推理权重。**

指标定义与验收要求见[数值对齐开发指南](../../docs/developer_guides/numerical_alignment.md)，完整参数见各入口的 `--help`。

## 环境准备

以下命令均从仓库根目录执行。需先安装项目依赖，配置 CANN、`torch_npu`，并准备空闲的 Ascend NPU。脚本不会自动 source 环境：

```bash
source /path/to/cann/set_env.sh
```

推理基线固定使用 `transformers==5.13.0`（运行时会校验版本），建议在独立环境中安装：

```bash
python -m pip install -r scripts/numerical_alignment/requirements.txt
```

## 局部模块对比

以 RMSNorm 为例，比较 Torch 实现与 AscRMSNorm 融合实现的前向输出、输入梯度和权重梯度：

```bash
python -m scripts.numerical_alignment.compare_module_example
```

默认 shape `(2, 17, 4096)`、`bfloat16`、seed `1234`；`--dtype` 支持 `bfloat16` / `float16` / `float32`，`--device` 仅支持 NPU。
三项检查均须满足 `mean_ae <= 0.02`、`mean_re <= 0.02`、`cos >= 0.9999`（`max_re` 仅用于诊断）。全部通过返回 `0`，否则返回 `1`。

两侧任一张量含 NaN/Inf 时，工具打印各侧 NaN/Inf 计数与首个非有限下标，并把该项判为 `FAIL`（失败原因名 `non_finite`），不再抛出异常。

## 整网精度对比

推理基线对齐使用 DSV4 Flash 的前 1～4 层，保留前 4 个 routed experts、top-3，关闭 MTP。

### 步骤一：准备 HF 权重

需要覆盖所选层、embedding、最终 norm、head 和 mHC head 的官方命名 Safetensors。
默认四层可从 [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) 下载 `config.json`、`model.safetensors.index.json`、分片 1～5（embedding 与前四层）和分片 45（最终 norm、head、mHC head）。更换版本或层数时按官方索引重新确认分片。

```bash
export HF_QUANTIZED=/path/to/dsv4        # 量化分片及 config.json
export HF_BF16=/path/to/dsv4-bf16        # 反量化输出目录（须为空）
export TT_DCP=/path/to/dsv4-dcp-4L       # DCP 输出目录（须为空）
```

已有 BF16 HF 权重时，将 `HF_BF16` 指向该目录并跳过下一步的反量化。

### 步骤二：权重转换

```bash
# FP8 / FP4 → BF16（CPU 上逐分片反量化）
python scripts/checkpoint_conversion/dequant_save_hf_dsv4.py "$HF_QUANTIZED" "$HF_BF16"

# HF → DCP（四层、四专家、top-3）
python scripts/checkpoint_conversion/convert_dsv4_hf_to_dcp.py "$HF_BF16" "$TT_DCP" \
    --num-experts 4 --top-k 3
```

`--n-layers`、专家数和 top-k 必须与下一步的推理配置一致。输出目录须为空或不存在，并需足够的 CPU 内存和磁盘空间。格式细节见[权重转换说明](../checkpoint_conversion/README.md)。

### 步骤三：Transformers 和 TorchTitan 前向对比

在同一张 NPU 上依次加载两侧模型，用相同随机 token 比较最后一个 token 的 BF16 logits（默认 batch 1、seq 4096、seed 1234，无需 tokenizer）：

```bash
python -m scripts.numerical_alignment.deepseek_v4.compare_dsv4_with_transformers \
    --hf-checkpoint-path "$HF_BF16" \
    --tt-checkpoint-path "$TT_DCP"
```

HF 与 DCP 必须来自同一份权重。两种 attention 模式都要求 batch size 为 `1`；参考模式（`--attention golden`）还要求序列长度为 `128` 的倍数，不支持的配置会在加载权重前报错。
默认启用 Transformers 数值补丁，使两侧的计算顺序与 BF16 边界一致，因此对齐参考是**打补丁后**的 Transformers 实现。

常用选项：

- `--mode bitwise`：默认，要求 logits 逐元素完全相同。
- `--mode close`：要求 `mean_ae <= 0.02`、`mean_re <= 0.02`、`cos >= 0.9999`。
- `--no-apply-transformers-patch`：观察未修改的 Transformers 结果。
- `--attention test`：使用 AscendC override 组合，需对应 CANN 算子。
- `--trace-modules`：输出逐模块对比指标，仅用于定位，不参与判定。

精度通过返回 `0`，否则返回 `1`。

### 步骤四：训练 loss 和 grad_norm 对比

[`loss_compare.py`](loss_compare.py) 比较两个训练脚本或两个 Git 版本，从 TensorBoard 事件读取逐 step 的 `loss` 和 `grad_norm`。
默认比较 100 步、序列长度 4096、训练 seed 42、开启确定性算法，默认入口为 `examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh`。

省略 Git ref 时直接使用当前工作区（含未提交改动）；指定两个 ref 时，两侧在独立 worktree 中运行，ref 须已存在于本地且改动已提交。
两个 ref 须同时提供或同时省略，暂不支持单个 ref 与当前未提交改动的比较。

#### 比较两个 Git 版本

```bash
export HF_ASSETS_PATH=/path/to/tokenizer

# 比特一致对比：前后两个提交
python scripts/numerical_alignment/loss_compare.py HEAD~1 HEAD
```

#### 比较上游适配前后的版本

`93f956c` 依赖 TorchTitan `dc3985ad`，使用旧的 `checkpoint`、`optimizer` 配置；当前版本依赖 `ebdfc8c`，使用可选的 `checkpointer` 和嵌套 `optim`。通过 `--baseline-config-api legacy` 选择旧接口，test 侧默认使用 `native`。两侧均已使用 token 训练 API，无需设置 `--baseline-training-api legacy`。

准备两个分别固定到上述 commit 的 TorchTitan 源码目录，使用同一套 CANN、PyTorch、torch-npu 与输入数据，从仓库根目录执行：

```bash
export HF_ASSETS_PATH=/path/to/tokenizer
export OLD_TORCHTITAN=/path/to/torchtitan-dc3985ad
export NEW_TORCHTITAN=/path/to/torchtitan-ebdfc8c
export RUNNER=examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh

python scripts/numerical_alignment/loss_compare.py 93f956c HEAD \
    --runner "$RUNNER" --steps 100 --seq-len 4096 \
    --baseline-config-api legacy --test-config-api native \
    --baseline-upstream "$OLD_TORCHTITAN" --test-upstream "$NEW_TORCHTITAN"
```

DeepSeek-V4.1 使用 `examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh`。在 A5 上使用对应的 `_a5.sh` runner；两侧分别读取各自 commit 中的脚本。运行前按实际设备配置 `ASCEND_RT_VISIBLE_DEVICES` 和编译缓存路径。

脚本从旧侧生成一份 FP32、仅模型的 seed DCP，两侧从该文件加载权重，并使用各自的空 checkpoint 输出目录，避免恢复其他训练状态。默认清理 seed DCP 和临时 worktree，保留 `seed.log`、两侧训练日志、TensorBoard 事件及 `comparison.json`。native 侧使用本工具的训练入口设置可选 checkpointer，不能同时指定自定义 `TRAIN_FILE`。

短训练只能验证启动、前反向、更新和比较流程；精度验收仍需完整运行 100 步。指标不一致时返回 `1`，以报告中的逐项结果判定，不把「两侧训练完成」作为精度通过。

#### 数值近似对比

同一提交下比较 Golden 实现与融合算子，只需覆盖 test 的环境变量：

```bash
python scripts/numerical_alignment/loss_compare.py HEAD HEAD \
    --mode close --test-env USE_GOLDEN=0
```

若融合算子不支持确定性算法，可加 `--no-deterministic`。

#### 比较自备的 base / test 脚本

将模型配置、数据路径和 override 分别写入两个脚本，二者须有执行权限，并在训练命令末尾追加 `"$@"` 以接收比较脚本控制的参数：

```bash
python scripts/numerical_alignment/loss_compare.py \
    --baseline-runner examples/deepseek_v4/debug/base.sh \
    --test-runner examples/deepseek_v4/debug/test.sh
```

同样可在命令前加上两个 Git ref 来比较历史版本的代码；入口脚本始终从当前工作区读取，无需存在于被测提交中。
显式指定两侧入口时，相对入口路径会先基于当前仓库根目录解析为绝对路径，因此命令中仍可使用相对路径；脚本中的相对训练路径则以对应 worktree 为基准。

指定两个 Git ref 时，两侧分别从对应 worktree 根目录启动。入口脚本须沿用该工作目录，以相对路径启动训练，例如 `bash scripts/run_train.sh "$@"`。若基于 `dirname "$0"` 切回原工作区，或使用原工作区的绝对路径定位训练入口，两侧可能运行同一份源码，产生未实际比较版本差异的 `PASS`。
此组合会在运行前打印警告，列出入口、目标提交及预期训练目录；该提示不会验证 runner 内部实际选择的源码。

#### 比较不同训练参数或 runner

```bash
# 只改 test 的学习率
python scripts/numerical_alignment/loss_compare.py HEAD HEAD \
    --mode close --test-options=--optimizer.lr=2e-5
```

常用参数：

- `--steps` / `--seq-len`：统一控制两侧步数和序列长度。
- `--baseline-options` / `--test-options`：分别追加训练参数。
- `--baseline-env` / `--test-env`：分别注入 `KEY=VALUE[,...]` 环境变量，可重复指定。
- `--runner` 设置两侧默认入口。指定 Git ref 时，相对路径使用对应 ref 中的脚本；绝对路径让两侧共用同一份脚本。绝对路径入口同样须沿用对应 worktree 的工作目录，运行前会打印相同告警。
- `--baseline-runner` / `--test-runner` 分别覆盖单侧入口，相对路径始终以当前工作区根目录为基准（不受 ref 影响），绝对路径直接使用指定文件。
- `--ngpus` 统一覆盖卡数，`--baseline-ngpus` / `--test-ngpus` 分别覆盖且优先。省略卡数参数时沿用环境变量和各自 runner 的设置；入口脚本需支持 `NGPU="${NGPU:-8}"` 形式接收卡数。

比较并行策略时，需自行保证全局 batch、输入数据和训练配置一致。

**卡数默认行为变更**：旧版未传 `--ngpus` 时固定注入 `NGPU=1`；现在两次训练依次采用显式卡数参数、单侧环境参数、继承的 `NGPU`、runner 默认值。已有命令可能因此使用更多设备；需要保持旧版单卡行为时，显式添加 `--ngpus=1`，并确认 runner 接受该环境变量。

#### Seed checkpoint

默认使用 baseline 入口，以单进程 CPU 初始化完整权重并保存 seed checkpoint，两侧训练加载同一份权重。seed 阶段单独使用 `NGPU=1`，不改变两次训练本身的卡数。跨并行布局加载是否支持取决于模型的 checkpoint 实现；当前不支持恢复完整训练状态的续训比较。
若两侧无法共用 seed checkpoint，可用 `--no-seed-checkpoint` 改为固定种子初始化。

两侧默认使用 token 预算接口。迁移比较可分别选择序列长度参数：`--baseline-training-api legacy` 使用 `--training.seq-len`，`--test-training-api tokens` 使用 `--training.max-context-length`；长度仍由 `--seq-len` 统一指定。旧侧 local/global batch 与新侧 microbatch/train-step token 预算通过各自 options 显式传入，工具不推断数据并行度或梯度累积。`--baseline-upstream` / `--test-upstream` 可分别指定 TorchTitan 源码目录，并加入对应进程的 `PYTHONPATH`；解释器及依赖仍由各侧环境提供。

#### 输出与清理

默认在 `outputs/compare_results/` 下创建独立结果目录，可用 `--output-dir` 指定空目录或尚不存在的目录。
控制台实时打印 seed 创建和两次训练日志，结束时打印误差摘要及 `PASS` / `FAIL`，详细结果写入 `comparison.json`。两侧指标须完整覆盖 `1..steps`；通过返回 `0`，否则返回 `1`。

报告的 `baseline.npu_processes` / `test.npu_processes` 记录启动配置：`override` 为显式卡数覆盖值，`ngpu_env` 为传给 runner 的 `NGPU`（未设置时为 `null`），`source` 为 `override`、`environment` 或 `runner`。runner 可在内部覆盖环境变量，因此这些字段不代表实际运行卡数；`source=runner` 时由脚本自行决定。

退出时默认清理 seed 目录和临时 worktree，保留训练日志、TensorBoard 数据和 `comparison.json`；`--no-cleanup-after-test` 可保留全部产物（含 seed checkpoint）。
