# 数值一致性检查（精度对比）

本目录提供局部模块前反向对比、Transformers 推理基线对齐，以及两个 Git 版本的训练指标对比。
整网对比以 DeepSeek-V4（DSV4）为例；只比较训练 `loss` 和 `grad_norm` 时，可直接使用「步骤四」，无需准备推理权重。

`utils/` 维护模型无关的指标、checkpoint 加载和 trace hook。DSV4 的配置、权重适配、前向协议及模块映射集中在 `deepseek_v4/`；训练入口和 override 选择由 `examples/` 下的 runner 提供。

## 目录结构

```text
scripts/numerical_alignment/
├── README.md
├── compare_module_example.py          # RMSNorm 前向输出和梯度对比
├── loss_compare.py                    # 两个 Git 版本的 loss、grad_norm 对比
├── requirements.txt                   # 推理基线的额外依赖
├── configs/
│   └── dsv4_4l.json                    # DSV4 四层、四专家推理配置
├── deepseek_v4/
│   ├── compare_dsv4_with_transformers.py  # Transformers 与 TorchTitan 前向对比
│   ├── config.py                      # 模型裁层、专家缩减和 hash routing
│   ├── hf_model.py                    # HF 权重视图、模型加载及前向
│   ├── tt_model.py                    # TorchTitan 模型构造、DCP 加载及前向
│   ├── overrides.py                   # golden、test 两组 override
│   ├── trace.py                       # 两侧模块输出的统一映射
│   └── transformers_dsv4_patch.py      # Transformers 数值对齐补丁
└── utils/
    ├── checkpoint.py                  # DCP 权重过滤和加载
    ├── metrics.py                     # tensor 精度指标和 logits 阈值
    └── trace_hooks.py                 # hook 注册、记录和清理
```

## 环境准备

以下命令从仓库根目录执行。需先安装项目依赖，配置 CANN、`torch_npu`，并准备空闲的 Ascend NPU。
CANN 路径按实际安装位置替换；脚本不会自动 source 环境：

```bash
source /path/to/cann/set_env.sh
```

推理基线固定使用 `transformers==5.13.0`，运行时会校验版本。在独立的对齐环境中，安装项目依赖后再安装：

```bash
python -m pip install -r scripts/numerical_alignment/requirements.txt
```

## 局部模块对比

[`compare_module_example.py`](compare_module_example.py) 以 RMSNorm 为例，比较 Torch 实现与 AscRMSNorm 融合实现。
两侧共享相同输入、权重、dtype 和上游梯度，分别检查前向输出、输入梯度和权重梯度。

```bash
python -m scripts.numerical_alignment.compare_module_example
```

默认 shape 为 `(2, 17, 4096)`、dtype 为 `bfloat16`、eps 为 `1e-6`、seed 为 `1234`。`--dtype` 支持 `bfloat16`、`float16`、`float32`，`--device` 仅支持 NPU。

控制台按 `forward`、`input_grad`、`weight_grad` 打印 `mean_ae`、`max_re`、`mean_re`、`cos` 和 `PASS` / `FAIL`。
三项检查均须满足 `mean_ae <= 0.02`、`mean_re <= 0.02`、`cos >= 0.9999`；`max_re` 仅用于诊断。
全部通过返回 `0`，任一检查不通过返回 `1`。

## 整网精度对比

### 步骤一：准备 HF 权重

推理对齐使用 DeepSeek-V4 Flash 的前 1～4 层，保留前 4 个 routed experts、top-3，并关闭 MTP。需准备官方 HF 命名的 Safetensors 权重，覆盖所选层、embedding、最终 norm、head 和 mHC head 等全局参数。

默认四层使用官方 [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) 的 48 分片版本，下载 `config.json`、`model.safetensors.index.json`、`model-00001-of-00048.safetensors`～`model-00005-of-00048.safetensors` 和 `model-00045-of-00048.safetensors` 即可。第 1～5 号分片覆盖 embedding 和前四层，第 45 号分片包含最终 norm、head 和 mHC head。更换版本或层数时，按官方索引重新确认所需分片。

本例使用以下目录，按实际位置替换：

```bash
export HF_QUANTIZED=/path/to/dsv4
export HF_BF16=/path/to/dsv4-bf16
export TT_DCP=/path/to/dsv4-dcp-4L
```

`HF_QUANTIZED` 存放量化分片及 `config.json`；`HF_BF16` 和 `TT_DCP` 是后续转换的空输出目录。
已有 BF16 HF 权重时，将 `HF_BF16` 指向该目录，跳过下一步的反量化。

### 步骤二：权重转换

#### FP8 / FP4 → BF16

[`dequant_save_hf_dsv4.py`](../checkpoint_conversion/dequant_save_hf_dsv4.py) 在 CPU 上逐分片反量化。
FP8 E4M3FN 权重按 block scale 还原，int8 打包的 FP4 权重按分组 scale 还原；每个量化权重及其 `.scale` 须在同一分片中。
输出保留 HF 参数名，移除 scale tensor，并生成新的权重索引和去除量化配置的 `config.json`。

```bash
python scripts/checkpoint_conversion/dequant_save_hf_dsv4.py \
    "$HF_QUANTIZED" "$HF_BF16"
```

#### HF → DCP

TorchTitan 使用 DCP 格式。将同一份 BF16 HF 权重转换为四层、四专家、top-3 的 DCP：

```bash
python scripts/checkpoint_conversion/convert_dsv4_hf_to_dcp.py \
    "$HF_BF16" "$TT_DCP" \
    --num-experts 4 --top-k 3
```

转换使用当前 `DeepSeekV4StateDictAdapter` 和 `routed_experts.inner_experts` 权重格式。
`--n-layers` 必须与步骤三一致，专家数和 top-k 必须与推理配置一致。
两项转换的输出目录均须为空或不存在，并需准备足够的 CPU 内存和磁盘空间。详细格式见[权重转换说明](../checkpoint_conversion/README.md)。

### 步骤三：Transformers 和 TorchTitan 前向对比

[`deepseek_v4/compare_dsv4_with_transformers.py`](deepseek_v4/compare_dsv4_with_transformers.py) 在同一张 NPU 上依次加载两侧模型，
使用相同的随机 token，比较最后一个 token 的 BF16 logits。默认 batch size 为 `1`、sequence length 为 `4096`、输入 seed 为 `1234`，无需 tokenizer。

```bash
python -m scripts.numerical_alignment.deepseek_v4.compare_dsv4_with_transformers \
    --hf-checkpoint-path "$HF_BF16" \
    --tt-checkpoint-path "$TT_DCP"
```

脚本关闭 NPU matmul、conv 和 ACLNN 的 HF32，并启用确定性算法；确定性补丁自动设置 `CLOSE_MATMUL_K_SHIFT=1` 和 `HCCL_DETERMINISTIC=true`。
HF 与 DCP 必须来自同一份权重。Golden 模式仅支持 batch size 为 `1`、序列长度为 `128` 的倍数；不支持的配置会在加载权重前报错。

默认启用 Transformers 数值补丁，使 RoPE、RMSNorm、mHC、indexer、稀疏 attention 和 MoE 的计算顺序及 BF16 边界匹配 TorchTitan。因此，对齐结果的参考是打补丁后的 Transformers 实现。

常用选项：

- `--mode bitwise`：默认模式，要求 logits 逐元素完全相同。
- `--mode close`：要求 `mean_ae <= 0.02`、`mean_re <= 0.02`、`cos >= 0.9999`。
- `--no-apply-transformers-patch`：观察未经修改的 Transformers 模型结果。
- `--attention test`：使用 AscendC override 组合，需要对应 CANN 算子，结果按所选模式独立判定。
- `--trace-modules`：输出 embedding、各层 attention / MoE / mHC、最终 norm 和 logits 的对比指标；中间结果只用于定位，不参与最终判定。

控制台先输出 HF32 状态、Transformers 版本和两侧加载过程，再输出 logits shape、四项指标及 `PASS` / `FAIL`。`max_re` 仅用于诊断。精度通过返回 `0`，不通过返回 `1`。

### 步骤四：训练 loss 和 grad_norm 对比

[`loss_compare.py`](loss_compare.py) 在独立 Git worktree 中运行两个版本，默认入口为
`examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh`。
脚本从 TensorBoard 事件读取逐 step 的 `loss` 和 `grad_norm`，默认比较 100 步、序列长度 4096，开启确定性算法，固定训练 seed 为 42。

执行前需确保两个 Git ref 已存在于本地，相关修改已提交；未提交改动不会出现在 worktree 中。
准备 tokenizer、checkpoint 环境变量，并为以下示例统一选择 Golden 实现：

```bash
export HF_ASSETS_PATH=/path/to/tokenizer
export CKPT_SAVE_LOAD_PATH=checkpoint
export USE_GOLDEN=1
```

#### 比特一致对比

使用 Golden 实现比较前后两个提交：

```bash
python scripts/numerical_alignment/loss_compare.py HEAD~1 HEAD
```

#### 数值近似对比

同一提交下比较 Golden 和融合算子，只需覆盖 test 的环境变量：

```bash
python scripts/numerical_alignment/loss_compare.py HEAD HEAD \
    --mode close --test-env USE_GOLDEN=0
```

如有不支持确定性算法的融合算子，可开启 `--no-deterministic`。

#### 不同训练参数或 runner

使用 `--baseline-options` / `--test-options` 比较两组训练参数。例如，将 test 的学习率从默认 `1e-5` 改为 `2e-5`：

```bash
python scripts/numerical_alignment/loss_compare.py HEAD HEAD \
    --mode close --test-options=--optimizer.lr=2e-5
```

步数和序列长度统一通过 `--steps`、`--seq-len` 控制。

`--runner` 可指定其他入口。相对路径使用各自 Git ref 中的脚本，可比较同一路径下不同版本的 runner；绝对路径让两侧共用同一份脚本。
例如使用 DeepSeek-V4 Flash 八卡入口：

```bash
python scripts/numerical_alignment/loss_compare.py HEAD~1 HEAD \
    --runner examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
    --ngpus 8
```

两侧卡数不同时使用 `--baseline-ngpus` / `--test-ngpus`，并通过各自 options 设置对应并行拓扑。
比较并行策略时，还需保证全局 batch、输入数据和训练配置一致。

#### 输出与清理

默认在当前目录的 `outputs/compare_results/` 下创建独立结果目录；`--output-dir` 可指定空目录或尚不存在的目录。
控制台实时打印 seed 创建和两次训练日志，结束时打印指标误差摘要及 `PASS` / `FAIL`，详细结果保存为 `comparison.json`。
两侧指标须完整覆盖 `1..steps`；全部通过返回 `0`，比较不通过返回 `1`。

默认在退出时清理 seed 目录和临时 worktree，保留 baseline / test 训练日志、TensorBoard 数据及 `comparison.json`；
使用 `--no-cleanup-after-test` 保留全部产物，包括 seed checkpoint。训练失败或比较不通过时也按该开关清理。
若两侧不能共用 seed checkpoint，可用 `--no-seed-checkpoint` 改为固定种子初始化，但仍需确保两侧初始权重可比。

指标定义与验收要求见[数值对齐开发指南](../../docs/developer_guides/numerical_alignment.md)，完整参数见各入口的 `--help`。
