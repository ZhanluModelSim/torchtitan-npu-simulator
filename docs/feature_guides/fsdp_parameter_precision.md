<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-->

# FSDP 参数级精度保留

FSDP 的 `MixedPrecisionPolicy` 会按统一的 `param_dtype` 处理模型参数。对于
RMSNorm、router 等对数值精度敏感的参数，可以通过
`parallelism.fsdp_preserve_parameter_patterns` 指定参数名匹配规则，使这些参数在
FSDP 参数分片和聚合时保留原始精度（通常为 FP32）。

## 适用范围

该配置只在 FSDP/HSDP 路径生效。启用 EP 时，专家参数同样通过 FSDP/HSDP 处理，
因此 EP 场景也支持该配置；仅使用 DDP 时配置不会生效，并会输出 warning。

精度标记必须在模型完成 TP/EP 转换后、调用 `apply_fsdp()`（内部执行
`fully_shard()`）前设置。这样既能匹配转换后的完整参数名，也能让 FSDP 在首次分片
时读取精度标记。

## 匹配规则

- 匹配对象是参数的完整 FQN，例如 `layers.0.attention.q_norm.weight`。
- `*` 只匹配一个 FQN 层级中的任意字符，但不会跨越 `.`。
- `**` 匹配零个或多个完整 FQN 层级。
- 未匹配到参数的 pattern 会输出 warning；如果所有 pattern 都没有匹配到参数，
  配置会抛出错误，避免精度策略静默失效。
- 参数已经由 TorchAO wrapper 处理时，不再额外设置该精度标记。

## 配置示例

模型配置通过 `replace` 写入 `ParallelismConfig`：

```python
parallelism=replace(
    base.parallelism,
    fsdp_preserve_parameter_patterns=[
        "layers.*.*norm.weight",
        "layers.*.moe.router.gate.weight",
        "norm.weight",
    ],
)
```

DeepSeek-V4 的默认规则见
`torchtitan_npu/models/deepseek_v4/config_registry.py`。MTP 层也位于
`layers.*` 命名空间时，可以被相应的通配规则覆盖；新增模型或新增参数结构后，
应根据实际 `named_parameters()` 输出检查规则是否仍然命中目标参数。
