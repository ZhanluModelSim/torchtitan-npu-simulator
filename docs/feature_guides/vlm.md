# VLM NPU 支持

本文档说明 `torchtitan_npu.models.vlm` 对上游 torchtitan VLM debug model 的 NPU 适配范围。

## 支持内容

- 通过 `torchtitan_npu.models.vlm` 注册 NPU 版本 `vlm` 模型入口。

## 并行化支持

VLM NPU 目前仅支持 FSDP/HSDP 数据并行。

## 运行示例

```bash
PYTHONPATH=/path/to/torchtitan:$PWD:${PYTHONPATH:-} \
NGPU=8 \
MODULE=torchtitan_npu.models.vlm \
CONFIG=vlm_debugmodel_npu \
bash scripts/run_train.sh
```

## 复用建议

新增多模态模型时，优先复用 `torchtitan_npu.models.multimodal` 中的通用 helper，避免在模型目录内重复实现。
