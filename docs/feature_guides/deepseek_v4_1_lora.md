# DeepSeek-V4.1 LoRA

V4.1 复用 dense、batched 和 routed-expert LoRA 实现。LoRA 冻结基座权重和 Engram 表，保留 Engram 查表路径。Muon 使用逐头布局和独立的 gate/up 专家因子，AdamW 处理其余匹配的参数。

## 训练

公开配方 `deepseek_v4_1_flash_lora_4k_a3` 默认使用 128 卡、4K 上下文、eager 执行和 Muon。运行前，在 `torchtitan_npu_recipes/models/deepseek_v4_1.py` 中找到该函数，在返回配置前按实际数据和存储设置 checkpoint 路径，并让基座权重、tokenizer 和 Engram token map 与模型匹配。以下为 150 步、每 50 步保存一次的配置示例。

```python
config.training.steps = 150
config.checkpointer.initial_load_path = "/path/to/v41-hf-base"
config.checkpointer.folder = "/path/to/run/checkpoint"
config.checkpointer.interval = 50
```

将 `NODE_IPS` 设为训练节点的地址列表。以下命令以每节点 8 卡启动训练。

```sh
HF_ASSETS_PATH=/path/to/tokenizer \
NODE_IPS="${NODE_IPS}" NGPU=8 \
MODULE=torchtitan_npu_recipes.models.deepseek_v4_1 \
CONFIG=deepseek_v4_1_flash_lora_4k_a3 \
bash scripts/run_train_multinodes.sh
```

切换为 AdamW 时，在 Python 配方中设置 `config.optim.optimizer.optimizers = [AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)]`。LoRA 默认 rank 为 16、alpha 为 32，专家 rank 为 16。

8 卡纯文本配方使用 `MODULE=torchtitan_npu_recipes.tests.models.deepseek_v4_1` 和 `CONFIG=deepseek_v4_1_flash_40layers_16experts_text_lora`。模型内存需求取决于具体模型配置和并行策略。

从 native DCP 基座初始化时，在 Python 配方中设置 `config.checkpointer.initial_load_in_hf = False`。恢复训练 checkpoint 时，保持 checkpoint 路径、EP 度数、Engram 表结构、tokenizer 映射和 LoRA 配置一致，并在启动命令末尾追加 `--resume-step 50` 指定恢复点。

## Checkpoint 与导出

初始化加载基座时保留新建的 adapter。公开配方的周期 checkpoint 默认保存 adapter、持久化模型 buffer 和训练状态。恢复时先加载 `initial_load_path` 指定的固定基座，再恢复这些状态。将 `config.checkpointer.periodic_save_adapter_only` 设为 `False` 可将基座和 Engram 分片一并写入周期 DCP。用于恢复训练的 checkpoint 必须包含 adapter。

PEFT 最终导出保存 `adapter_model.safetensors` 和 `adapter_config.json`，Engram 权重保留在基座中。若配方包含可训练的非 LoRA 参数，例如 24 专家多模态测试配方的视觉参数，将 `config.checkpointer.last_save_in_peft` 和 `config.checkpointer.last_save_model_only` 均设为 `False`，最后一步会保存可续训的 native DCP。普通全模型 HF 导出无法保留 LoRA adapter，LoRA checkpointer 会拒绝 `last_save_in_hf`。

将 adapter 合并到浮点 V4.1 HF 基座：

```sh
python -m scripts.lora.merge_adapter \
  --base-model /path/to/v41-floating-point-base \
  --adapter /path/to/run/checkpoint/step-150 \
  --output /path/to/v41-merged
```

共享合并工具保留基座 dtype、分片索引及配套文件，包括 Engram 映射和模型配置；Engram 权重不参与增量合并。输出目录必须不存在，且不能位于任一输入目录内。量化 checkpoint 需要先通过对应后端解码为普通 FP16/BF16/FP32 逻辑权重。

参数量化 LoRA 可组合已有 TorchAO-NPU recipe：adapter 保持浮点，基座保持冻结。block-FP8 专项测试适用于支持该格式的硬件和运行时；A3 上的 BF16 验证不作为 block-FP8 的执行验证。

## 验证命令

```sh
python -m pytest -q tests/unit_tests/models/deepseek_v4_1/test_v41_lora.py
python -m tests.integration_tests.run_tests /tmp/v41-lora-training \
  --test_suite deepseek_v4_1 --ngpu 2
python -m tests.integration_tests.run_tests /tmp/v41-lora-resume \
  --test_suite deepseek_v4_1_checkpoint --ngpu 2
```

默认 `models` suite 包含纯文本和带图像输入的短程 AdamW 训练及 PEFT 导出。与 DeepSeek-V4 一致，分布式 checkpoint 恢复保留在独立 suite 中，因为 A3 CI 运行时在分布式 checkpoint planning 阶段存在 `libscatter_aicpu_kernel.so` 加载问题。checkpoint suite 对 AdamW 的 adapter-only DCP 和 Muon 的完整 DCP，分别比较连续训练第 1–4 步与从第 2 步恢复后的第 3–4 步，检查 adapter 更新、基座和 routing bias 冻结、Engram 无稀疏梯度、优化器状态恢复、PEFT 张量以及恢复前后 loss 和梯度范数一致性。独立的 `deepseek_v4_1_quantized` suite 对 block-FP8 应用同样的恢复检查。
