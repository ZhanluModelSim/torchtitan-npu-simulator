# DeepSeek V4 系列 SFT

## 1. 准备 DeepSeek 官方发布模型资产

沿用所在集群的 CPT 环境。从官方权重启动 SFT 时，需要准备**完整的 DeepSeek 官方发布模型资产**，包括模型配置、权重分片及索引、tokenizer 和 `encoding/` 下的官方编码文件。**不同发布日期的 `encoding_dsv4.py` 或 `encoding.py` 即使同名也不可混用。** 编码文件须与权重和 tokenizer 来自同一发布版本。配方的 `hf_assets_path` 指向包含 tokenizer 和 `encoding/` 的目录，初始权重由原 CPT 配方加载。

- **DeepSeek V4** 的 Flash、Pro、Flash-0731 和 Pro-0813 都在 `encoding/encoding_dsv4.py` 提供文本对话编码。各版本沿用同一文件名，具体编码规则以随权重发布的文件为准。[V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1/encoding) · [V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1/encoding) · [Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding) · [Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/tree/72e1d3230f6c080a530b0a1d46f8eeb4602340597/encoding)
- **DeepSeek V4.1** 的 Flash 和 Flash-1017 在 `encoding/encoding.py` 提供文本与图文对话编码，可同时返回 prompt 和媒体记录。[V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6/encoding) · [Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/08c8657f02291e11b6f6b63e0e16014f8e748518/encoding)

SFT 沿用 DeepSeek 官方示例的做法，直接调用对应版本 Python 编码器的 `encode_messages()`。官方编码器生成对话文本、特殊标记和 V4.1 的媒体信息，训练侧在编码结果上定位 assistant 片段并生成监督标签。[V4 官方推理示例](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/generate.py#L125) · [V4.1 官方编码示例](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/encoding/README.md)

从官方 FP8/FP4 权重开始时，先转换为 BF16 safetensors。将配方中 `checkpointer.initial_load_in_hf` 设为 `True`，`initial_load_path` 指向转换后的目录。V4.1 Flash 的 CPT 配方用 `HF_ASSETS_PATH` 指向 DeepSeek 官方发布模型资产目录，用 `CKPT_INIT_LOAD_PATH` 指定 BF16 权重。[V4 转换脚本](../../scripts/checkpoint_conversion/dequant_save_hf_dsv4.py) · [V4.1 转换脚本](https://gitcode.com/cann/cann-recipes-train/blob/78a47e5e3459601ee8e21fd64c1f8ac68e2bbd17/llm_pretrain/deepseek_v4_1/utils/convert_bf16.py)

## 2. 准备数据

每行写一条 JSONL 对话。两种模型都接受纯文本多轮 `messages`。V4.1 的图文混排使用 JSONL 保留消息中的文字和图片块，图片相对路径以数据文件所在目录为基准。下列 `...` 表示省略的样本。

```jsonl
{"messages":[{"role":"user","content":"Translate ‘晨雾’ into English."},{"role":"assistant","content":"Morning mist."},{"role":"user","content":"Use it in a sentence."},{"role":"assistant","content":"Morning mist covered the lake."}]}
...
{"messages":[{"role":"user","content":[{"type":"text","text":"Compare this view "},{"type":"image","path":"images/first.jpg"},{"type":"text","text":" with this one."},{"type":"image","path":"images/second.jpg"}]},{"role":"assistant","content":"The first view has clear skies, while the second is foggy."}]}
```

V4 使用第一种文本样式。工具对话沿用 DeepSeek 官方编码器的消息格式，函数定义放在 `tools`，assistant 的调用放在 `tool_calls`，工具返回以 `tool_call_id` 对应调用。

## 3. 复用 CPT 配方启动

**SFT 复用现有 CPT 配方和启动脚本。** `MODULE` 和 `CONFIG` 选择已能运行的 CPT 配方，沿用其模型、权重加载、优化器和并行配置。设置 `DATASET_PATH` 并追加相应的 SFT `--override`，即可切换 dataloader。

按原 CPT 命令填写两个 Bash 数组，没有对应设置时使用空数组 `()`。

- `ENV_ARGS` 放原命令中的环境变量赋值，例如 `NGPU=8`、`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。通过 `env "${ENV_ARGS[@]}"` 在启动脚本前设置。
- `CPT_ARGS` 放原命令中写在 `bash scripts/run_train.sh` 后面的参数，例如 `--output-dir /path/to/sft-run`。通过 `"${CPT_ARGS[@]}"` 按原顺序传入，然后追加 SFT override。

下面以仓库现有的 V4 eager 配方和 V4.1 图文调试配方为例，说明如何复用 CPT 配方启动 SFT。将官方 tokenizer 和编码文件放在配方的 `hf_assets_path` 下，V4 示例使用 `/path/to/DeepSeekV4_tokenizer`，V4.1 示例使用 `tests/assets/tokenizer`。

```bash
ENV_ARGS=("NGPU=${NGPU:-}")
CPT_ARGS=(--output-dir /path/to/run)

# V4
env "${ENV_ARGS[@]}" \
  MODULE=torchtitan_npu_recipes.tests.models.deepseek_v4 \
  CONFIG=deepseek_v4_flash_43layers_16experts_eager_bf16 \
  DATASET_PATH=tests/assets/deepseek_v4/alpaca_samples.jsonl bash scripts/run_train.sh \
  "${CPT_ARGS[@]}" \
  --override torchtitan_npu.override.deepseek_v4.chat_dataloader.sft

# V4.1
env "${ENV_ARGS[@]}" \
  MODULE=torchtitan_npu_recipes.models.deepseek_v4_1 \
  CONFIG=deepseek_v4_1_debugmodel_multimodal DATASET=cc12m-test \
  DATASET_PATH=/path/to/docci_sft.jsonl bash scripts/run_train.sh \
  "${CPT_ARGS[@]}" \
  --override torchtitan_npu.override.deepseek_v4_1.vision_language_dataloader.sft
```

训练步数和权重设置由 Python 配方提供。

V4.1 的每个微批次包含一个对话上下文，`num_tokens_per_microbatch_per_dp_rank` 须等于 `max_context_length`。

## 4. 短测

- **V4** 使用 512 条 Alpaca 文本对话样本。[样本文件](../../tests/assets/deepseek_v4/alpaca_samples.jsonl) · [数据来源与许可](../../tests/assets/deepseek_v4/ATTRIBUTION.md)
- **V4.1** 下载并解压 [`docci_samples.zip`](https://raw.gitcode.com/cann/torchtitan-npu/attachment/uploads/3bb0f2e6-754f-4e63-ac67-c0b8ab689bf2/docci_samples.zip)，将 `DATASET_PATH` 指向解包后的 `docci_sft.jsonl`，并保持 `images/` 与数据文件的相对位置。
