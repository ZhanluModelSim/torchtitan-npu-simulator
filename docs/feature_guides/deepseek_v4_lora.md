# DeepSeek-V4 LoRA 微调

DeepSeek-V4 支持对普通线性层、分组线性层和路由专家添加 LoRA 适配器。训练时冻结基座参数，只更新适配器；训练结果可导出为 PEFT 格式，也可离线合并到基座权重。

## 启用方式

完整 Flash 模型使用 `deepseek_v4_flash_lora` 配置，普通层和专家的秩均为 16，缩放参数 `alpha` 为 32。启动脚本沿用对应 CPT 入口的节点、数据、并行和编译配置，并开启检查点保存：

```bash
NODE_IPS="${NODE_IPS}" \
HF_ASSETS_PATH=/path/to/tokenizer \
CKPT_INIT_LOAD_PATH=/path/to/base-weights \
CKPT_SAVE_LOAD_PATH=/path/to/run/checkpoint \
bash examples/deepseek_v4/deepseek_v4_flash_lora_4k_a3.sh \
  --optimizer.name AdamW \
  --training.steps 500 \
  --checkpoint.interval 100
```

A5 使用 `CONFIG=deepseek_v4_flash_lora bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh --checkpoint.no-load-only --checkpoint.save-training-state`，直接调用 A5 CPT 入口，默认启用 block-FP8。运行前按实际设备和数据设置节点、卡数、并行参数及输入路径，详见[训练示例](../../examples/deepseek_v4/readme.md)。命令末尾的普通 CLI 参数可覆盖默认值。

### 模型与优化器配置

自定义配置通过 `deepseek_v4_flash(converters=[DeepSeekV4LoRAConverter.Config(...)])` 启用 LoRA；秩、缩放参数和目标模块在转换器中设置，无需额外启用开关。配置自动选择 LoRA 并行化与检查点管理器。

| 配置项 | 用途 |
| --- | --- |
| `--optimizer.name AdamW` | 使用 AdamW 更新适配器 |
| `--optimizer.name Muon` | 使用 Muon 与 AdamW 混合更新；专家 gate/up 适配器 B 分开由 Muon 更新，要求 TP=PP=1 |
| `--extension.quantization.no-enable-quantized-training` | 关闭量化训练，使用 BF16 基座 |
| `--extension.quantization.recipe` | 选择已有量化方案；A5 入口默认 `all_block_fp8` |

未启用量化时，LoRA 沿用原有基座配置。使用已有 TorchAO-NPU 参数量化方案时，LoRA 配置保留所选量化策略，基座参数沿用原策略选择，适配器保持 BF16。替换整个模块且无法保留适配器前向计算的方案会在配置阶段报错。优化器和并行配置详见 [Muon 优化器](muon_optimizer.md)与[上下文并行](deepseek_v4_cp.md)。

模型并行化后、创建优化器前冻结非 LoRA 参数，并关闭 MoE 负载均衡更新，保留基座路由偏置。需要只训练部分适配器时，可在并行化回调返回后冻结对应参数，例如冻结全部 `lora_a`、仅训练 B。

## 检查点与断点续训

周期检查点使用原生 DCP 格式。`checkpoint.periodic_save_adapter_only=True` 时，只保存适配器和路由缓冲区，不重复保存冻结的基座。启用 `checkpoint.save_training_state=True` 后，还会保存优化器、调度器、数据加载器和训练步数；上述 LoRA 启动脚本已启用此选项。

断点续训时保持相同基座、训练总步数和检查点目录，并通过 `--checkpoint.load-step 100` 指定恢复点。

最后一步默认导出 PEFT 适配器。若需要将最后一步也保存为可续训的 DCP 检查点，设置：

```bash
--checkpoint.no-last-save-in-peft --checkpoint.no-last-save-model-only
```

## 导出与加载适配器

PEFT 导出生成 `adapter_model.safetensors` 和 `adapter_config.json`，只包含适配器权重及配置。导出目录须为本地文件系统路径。

使用 `--checkpoint.peft-base-model-name-or-path` 指定供外部加载器使用的 Hugging Face 基座 ID 或目录。未指定时，优先采用 HF 初始检查点路径，否则采用 HF 资源路径；原生 DCP 路径不会写入基座标识。

加载时使用与训练相同、且 Transformers 能够加载的基座：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("/path/to/hf-base")
model = PeftModel.from_pretrained(base, "/path/to/run/checkpoint/step-500")
```

导出通过 PEFT 的 `target_parameters` 描述分组投影和专家参数，将路由专家的 gate/up 适配器转换为融合参数格式。

## 离线合并

在本仓训练环境中运行以下工具，将最终 PEFT 适配器合并到对应的 safetensors 基座，生成可独立加载的模型权重：

```bash
python -m scripts.lora.merge_adapter \
  --base-model /path/to/base-safetensors \
  --adapter /path/to/checkpoint/step-500 \
  --output /path/to/merged-model
```

输出目录必须不存在，且位于输入目录之外。工具支持官方检查点、Transformers 单文件或分片权重，以及融合专家权重；专家的 gate/up/down 按专家分别合并。

工具保留 FP16/BF16/FP32 基座的数据类型、分片索引与模型配置，仅重写包含合并目标的分片。量化训练输出不能直接离线 merge；输入须包含普通 FP16/BF16/FP32 logical weight。当前仓库不提供通用量化 checkpoint 离线解码命令。

## 支持范围与限制

| 项目 | 约束 |
| --- | --- |
| PEFT 专家导出 | 普通层与专家适配器的秩须相同 |
| MTP 适配器 | 使用原生检查点；需要 PEFT 导出时设置 `include_mtp=False` |
| 完整模型 HF 导出 | `last_save_in_hf=True` 不合并 LoRA 增量，本检查点管理器不接受该选项；使用上述离线合并工具 |
| 离线合并 | 支持标准 LoRA 的 `alpha/r` 缩放，不支持 DoRA、RSLoRA 或逐目标秩与缩放参数 |

集成测试复用现有入口：`deepseek_v4` 中的 `dsv4_lora_ep2_fsdp2` 检查训练与导出，独立 `deepseek_v4_checkpoint` suite 中的 `dsv4_lora_resume_ep2_fsdp2` 在支持分布式 DCP 的运行环境中检查断点续训。运行方式见[集成测试说明](../../tests/integration_tests/README.md)。

合并工具先验证 rank、专家数和量化序列化格式，再按需读取 adapter A/B。结果写入临时目录，全部成功后才发布输出；写入失败会清理临时文件。

支持 block-FP8 的 Ascend 950 设备（CANN 9.2.0 release，2026-09-09 软件包）使用 `deepseek_v4_quantized` suite 运行 `dsv4_lora_block_fp8_ep2_fsdp2`，使用真实 TorchAO-NPU block-FP8 recipe 验证量化基座、浮点 adapter 及训练更新。

专家 gate/up B 参数分别保存为 `w1_lora_b` 和 `w3_lora_b`，共享 `w13_lora_a`，PEFT 导出格式保持不变。普通 `model.load_state_dict` 可拆分旧的 `w13_lora_b` 权重；旧版交错参数的原生 DCP 不能直接续训，因为参数分组及优化器状态已改变。请使用生成该 checkpoint 的版本导出并合并适配器，再从合并后的基座开始新训练。
