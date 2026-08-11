# 已有图 Pattern

本目录保存可按需启用的图 pattern。新增 pattern 的接入方式参见
[片段融合算子接入](../../../docs/graph_pattern_fusion.md)。

## DeepSeek-V4 Inplace Partial RoPE

该 pattern 将 DeepSeek-V4 中的 interleaved RoPE 小算子片段替换为
`inplace_partial_rotary_mul`。

### 启用

导入 pattern 模块：

```bash
export TORCHTITAN_NPU_PATTERN_IMPORTS=\
torchtitan_npu.compile.patterns.deepseek_v4.inplace_partial_rope
```

训练同时需要开启 Inductor 编译。快速调试时使用：

```text
TORCHINDUCTOR_NPU_EXT_DEBUG=allfallback
--compile.enable
--compile.components model
--compile.backend inductor
```

性能验证时不设置 `TORCHINDUCTOR_NPU_EXT_DEBUG=allfallback`。

模型 override 必须包含：

```text
torchtitan_npu.override.common.rope.workaround
torchtitan_npu.override.deepseek_v4.sparse_attn.cann_metadata=...
torchtitan_npu.override.deepseek_v4.sparse_attn.cann
```

### 约束

- 算子当前只在 A5 上可用；
- 不能同时启用 `torchtitan_npu.override.common.rope.cann_complex`，否则原始小算子
  片段会提前变成 `torch_npu.npu_rotary_mul`，pattern 无法命中；
- 当前不支持 DeepSeek-V4 golden attention，整网验证使用 `sparse_attn.cann`；
- `scripts/run_train.sh` 的 `TEST_OVERRIDES` 默认使用 `rope.cann_complex`，验证该
  pattern 前需要换成 `rope.workaround`。
