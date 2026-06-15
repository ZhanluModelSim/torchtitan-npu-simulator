# VLM NPU 支持

本文档说明 `torchtitan_npu.models.vlm` 对上游 torchtitan VLM debug model 的 NPU 适配范围。

## 支持内容

- 通过 `torchtitan_npu.models.vlm` 注册 NPU 版本 `vlm` 模型入口。

## 并行化支持

VLM NPU 目前仅支持 FSDP/HSDP 数据并行。

## 运行示例

```bash
PYTHONPATH=/path/to/torchtitan:$PWD:${PYTHONPATH:-} \
NGPU=1 \
MODULE=torchtitan_npu.models.vlm \
CONFIG=vlm_debugmodel_npu \
COMM_MODE=fake_backend \
bash scripts/run_train.sh \
  --training.local_batch_size 1 \
  --training.seq_len 256 \
  --dataloader.max_patches_per_image 64 \
  --dataloader.max_images_per_batch 4
```

## 复用建议

新增多模态模型时，优先复用 `torchtitan_npu.models.multimodal` 中的通用 helper，避免在模型目录内重复实现。
