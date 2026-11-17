# DeepSeek-V4 training examples

本目录只保留少量稳定入口，模型规模、序列长度、并行策略和量化方式优先通过 CLI 覆盖，不为每一种组合新增独立脚本。

## 入口关系

Flash 多机训练以 `deepseek_v4_flash_cpt_4k_a3.sh` 为公共基线：

```text
deepseek_v4_flash_cpt_4k_a3.sh
└── deepseek_v4_flash_cpt_4k_a5.sh
    └── deepseek_v4_flash_cpt_1024k_a5.sh
```

- `deepseek_v4_flash_cpt_4k_a3.sh`：公共训练参数、override、checkpoint、optimizer 等基线配置，默认通过 CLI 启用 model 编译，backend 为 `inductor`。
- `deepseek_v4_flash_cpt_4k_a5.sh`：增加 A5 运行环境和默认 block-FP8 量化，其他配置复用 A3 基线。
- `deepseek_v4_flash_cpt_1024k_a5.sh`：继续复用 A5 4K 入口，只通过 CLI 覆盖 1M 所需的 CP、DP、序列长度和 global batch size，量化默认随 A5 4K 入口继承。
- 用户普通 CLI 参数位于默认选项之后，因此可以覆盖默认值（A5 的 activation-checkpoint 子命令仍在末尾），包括使用 `--extension.quantization.no-enable-quantized-training` 切换到 BF16。

## 编译配置

编译通过 CLI 配置。Flash A3/A5 默认使用 `inductor`，只编译 `model`。在示例命令末尾追加
`--compile.backend inductor` 可显式指定 backend，追加 `--compile.no-enable` 可关闭编译。
独立入口若未默认启用编译，需同时传入 `--compile.enable --compile.components model`。

## 常用环境变量

运行前至少根据环境设置以下路径和节点信息：

```sh
export NODE_IPS="192.168.1.10,192.168.1.11,..."
export HF_ASSETS_PATH=/path/to/DeepSeekV4_tokenizer
export CKPT_SAVE_LOAD_PATH=/path/to/save_or_resume_ckpt
export CKPT_INIT_LOAD_PATH=/path/to/init_hf_ckpt
```

A5 默认每节点使用 8 张 NPU，A3 默认每节点使用 16 张 NPU；需要改变时设置 `NGPU`。

## 裁剪 Debug 模型

Debug/performance 验证优先使用等价裁剪模型，避免为了快速验证直接启动完整规模模型。这里统一放置不同规模的裁剪 Debug 入口，卡数不是这一节的分类依据。

### 1P mini 模型

```sh
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh

USE_GOLDEN=1 \
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh
```

### 8P Flash 16-expert Debug

```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --training.steps 10
```

需要采集 profile 时显式追加：

```sh
--profiler.enable-profiling
```

### 32P Pro 32-expert Debug

Pro 32-expert 是独立的裁剪模型 debug/performance 入口，不与 8P Flash launcher 强制复用：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --compile.enable \
  --compile.components model \
  --compile.backend inductor \
  --debug.moe-force-load-balance \
  --training.steps 20
```

128K 场景不新增 launcher，只通过 CLI 改变并行和训练配置：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
  --compile.enable \
  --compile.components model \
  --compile.backend inductor \
  --parallelism.context-parallel-degree 32 \
  --parallelism.data-parallel-shard-degree 1 \
  --parallelism.data-parallel-replicate-degree 1 \
  --training.seq-len 131072 \
  --training.global-batch-size 8 \
  --debug.moe-force-load-balance \
  --training.steps 20
```

A5 脚本默认启用 block-FP8；如需 BF16，请在命令末尾追加后文的禁用选项。

### 导出和加载 Debug Hugging Face checkpoint

导出 checkpoint：

```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --checkpoint.folder ./export_ckpt \
  --checkpoint.enable \
  --checkpoint.no-load-only \
  --checkpoint.last-save-in-hf \
  --training.steps 1 \
  --debug.seed 42 \
  --debug.deterministic
```

从导出的 HF checkpoint 冷启动时，使用新的或空的 `checkpoint.folder`，避免已有 step checkpoint 优先触发 resume：

```sh
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --checkpoint.folder ./load_ckpt \
  --checkpoint.enable \
  --checkpoint.initial-load-path ./export_ckpt/checkpoint/step-1 \
  --checkpoint.initial-load-in-hf \
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
  --compile.enable \
  --compile.components model \
  --compile.backend inductor \
  --debug.moe-force-load-balance \
  --training.steps 20
```

### A5 64K

64K 不需要新增 launcher，直接在 A5 4K 入口上覆盖训练和并行参数：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
  --compile.enable \
  --compile.components model \
  --compile.backend inductor \
  --parallelism.context-parallel-degree 8 \
  --parallelism.data-parallel-shard-degree 16 \
  --parallelism.data-parallel-replicate-degree 1 \
  --training.seq-len 65536 \
  --training.global-batch-size 128 \
  --debug.moe-force-load-balance \
  --training.steps 20
```

### A5 1M

1M wrapper 只提供该场景的默认 CLI 覆盖；仍可在命令末尾继续覆盖：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_1024k_a5.sh \
  --compile.enable \
  --compile.components model \
  --compile.backend inductor \
  --debug.moe-force-load-balance \
  --training.steps 20
```

等价的主要差异参数为：

```sh
--parallelism.context-parallel-degree 128 \
--parallelism.data-parallel-shard-degree 1 \
--parallelism.data-parallel-replicate-degree 1 \
--training.seq-len 1048576 \
--training.global-batch-size 8
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

Flash 多机 CPT 基线默认启用 `--checkpoint.load-only`，即只负责加载，不会保存新的训练 checkpoint。需要保存时必须显式覆盖：

```sh
--checkpoint.no-load-only
```

当 `checkpoint.folder` 已存在有效的 `step-*` checkpoint 时，TorchTitan 会优先 resume，该情况下 `checkpoint.initial-load-path` 不作为冷启动来源；要从 `initial-load-path` 冷启动，请使用新的或空的 `checkpoint.folder`。

## LoRA 微调

完整 Flash 模型的 BF16 LoRA 训练复用 A3 4K 公共入口，节点、数据和基座路径使用上面的环境变量：

```sh
NODE_IPS="${NODE_IPS}" \
bash examples/deepseek_v4/deepseek_v4_flash_lora_4k_a3.sh \
  --optimizer.name AdamW \
  --training.steps 500 \
  --checkpoint.interval 100
```

该入口使用 `deepseek_v4_flash_lora` recipe（dense/expert rank 16、alpha 32），并开启 checkpoint 保存。优化器沿用 CPT 默认值，传入 `--optimizer.name` 可覆盖。周期 checkpoint 用于续跑，最后一步导出 PEFT adapter。模型规模、并行配置和其余默认参数继承 A3 公共入口，命令末尾的 CLI 参数仍可覆盖默认值。A5 使用 `CONFIG=deepseek_v4_flash_lora bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh --checkpoint.no-load-only --checkpoint.save-training-state`，继承 A5 CPT 的 block-FP8 默认配置。

配置、部分参数训练、native checkpoint 恢复、PEFT 导出和离线合并见 [LoRA 功能指南](../../docs/feature_guides/deepseek_v4_lora.md)。
