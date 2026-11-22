# DeepSeek-V4.1 LoRA

V4.1 复用已有的 dense、batched 和 routed-expert LoRA 实现。训练时冻结基座、Engram 表和视觉编码器，保留 Engram 查表路径。AdamW 和 Muon 仅更新 adapter；Muon 使用逐头布局和独立的 gate/up 专家因子，输出头和 indexer 沿用 AdamW fallback。

## 训练

LoRA 脚本复用现有 CPT 入口，默认选择多模态 `deepseek_v4_1_flash_lora`、参考算子和 eager 执行，并清空 optimizer swap override，与下述 LoRA 集成用例的执行方式一致。基座权重、tokenizer、Engram token map 和模型配置需要匹配，节点数和并行参数按实际资源设置：

```sh
NODE_IPS="${NODE_IPS}" \
HF_ASSETS_PATH=/path/to/tokenizer \
CKPT_INIT_LOAD_PATH=/path/to/v41-hf-base \
CKPT_SAVE_LOAD_PATH=/path/to/run/checkpoint \
USE_GOLDEN=1 STEPS=150 NGPU=16 \
MODULE=torchtitan_npu_recipes.models.deepseek_v4_1 CONFIG=deepseek_v4_1_flash_lora_4k_a3 \
bash scripts/run_train_multinodes.sh
```

上例的 `MODULE` 对应仓库中的 `torchtitan_npu_recipes/models/deepseek_v4_1.py`。在该文件中找到 `CONFIG` 指定的函数，在返回训练配置前设置下文的字段。

上述配方使用 Muon。使用 AdamW 时，在 Python 配方中设置 `config.optim.optimizer.optimizers = [AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)]`。每 50 步保存 checkpoint 时设置 `config.checkpointer.interval = 50`。默认 rank 为 16、alpha 为 32，专家 rank 为 16。通过 Python 配方 `deepseek_v4_1_flash_text_lora` 配置正式纯文本模型，默认使用 `c4_test`，也可用 `DATA_FILES` 指定文本数据。多模态默认数据集为 `cc12m-test`。

`torchtitan_npu_recipes.models.deepseek_v4_1.deepseek_v4_1_debugmodel_lora` 为纯文本调试配置。设备数量应满足模型内存需求，仅缩小并行度不会缩小模型。

从 native DCP 基座初始化时，在 Python 配方中设置 `config.checkpointer.initial_load_in_hf = False`。恢复 native checkpoint 时，保持 EP 度数、Engram 表结构、tokenizer 映射和 LoRA 配置一致。保持相同的 `CKPT_INIT_LOAD_PATH` 和 `CKPT_SAVE_LOAD_PATH`，使用 `--resume-step 50` 指定恢复点。

## Checkpoint 与导出

初始化加载基座时保留新建的 adapter。周期 checkpoint 默认保存 adapter、持久化模型 buffer 和训练状态；恢复时先加载固定基座，再恢复这些状态，因此需要保留 `initial_load_path`。设置 `config.checkpointer.periodic_save_adapter_only = False` 可将基座和 Engram 分片一并写入周期 DCP。恢复路径中的 checkpoint 必须包含 adapter。

最后一步保存 `adapter_model.safetensors` 和 `adapter_config.json`。Engram 权重保留在所引用的基座中，PEFT 仅导出训练的 adapter。将 `config.checkpointer.last_save_in_peft` 和 `config.checkpointer.last_save_model_only` 均设为 `False` 可在最后一步保存可续训的 native checkpoint。`last_save_in_hf` 会被拒绝，因为普通全模型 HF 导出不保留 LoRA adapter。

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
  --test_suite deepseek_v4_1 --test_name dsv41_lora_engram_ep2_fsdp2 --ngpu 2 --no-parallel
python -m tests.integration_tests.run_tests /tmp/v41-lora-resume \
  --test_suite deepseek_v4_1_checkpoint --test_name dsv41_lora_adamw_resume_ep2_fsdp2 --ngpu 2 --no-parallel
```

默认 `models` suite 包含纯文本和带图像输入的短程 AdamW 训练及 PEFT 导出。与 DeepSeek-V4 一致，分布式 checkpoint 恢复保留在独立 suite 中，因为 A3 CI 运行时在分布式 checkpoint planning 阶段存在 `libscatter_aicpu_kernel.so` 加载问题。checkpoint suite 对 AdamW 的 adapter-only DCP 和 Muon 的完整 DCP，分别比较连续训练第 1–4 步与从第 2 步恢复后的第 3–4 步，检查 adapter 更新、基座和 routing bias 冻结、Engram 无稀疏梯度、优化器状态恢复、PEFT 张量以及恢复前后 loss 和梯度范数一致性。独立的 `deepseek_v4_1_quantized` suite 对 block-FP8 应用同样的恢复检查。
