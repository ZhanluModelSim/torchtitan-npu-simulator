# 片段融合算子接入

当融合算子只替换 `forward()` 中的一段连续计算时，使用 pre-AOT pattern 可以避免复制
整个 Module 或修改模型代码。

## 什么时候使用

适合：原始片段结构稳定、可被 `torch.compile` 捕获，融合算子位于 Module 内部。

不适合：需要替换完整 Module、包含数据依赖控制流，或融合算子没有 Fake/Meta 实现。训练
算子还必须支持 Autograd。

## 怎么接入

以 DeepSeek-V4 的 `split -> complex RoPE -> cat` 替换为
`inplace_partial_rotary_mul` 为例，核心结构如下。`original_rope_fragment` 和
`fused_partial_rope` 分别表示原始计算片段和融合算子调用：

```python
from torchtitan_npu.compile import PatternReplacement, register_pre_aot_patterns


def make_pattern(*, inverse):
    def search_fn(x, cos, sin):
        return original_rope_fragment(x, cos, sin, inverse=inverse)

    def replacement_fn(x, cos, sin):
        return fused_partial_rope(x, cos, sin, inverse=inverse)

    return PatternReplacement(
        search_fn=search_fn,
        replacement_fn=replacement_fn,
        ignore_literals=True,
    )


register_pre_aot_patterns(
    {
        "dsv4_parent_rope_inverse": make_pattern(inverse=True),
        "dsv4_parent_rope_forward": make_pattern(inverse=False),
    }
)
```

将接入代码放在独立模块中，并在模块导入时完成注册。启动训练前显式设置模块路径：

```bash
export TORCHTITAN_NPU_PATTERN_IMPORTS=\
torchtitan_npu.compile.patterns.deepseek_v4.inplace_partial_rope
```

多个模块使用逗号分隔。该入口只注册编译图 pattern，不占用 `override.imports` 中的
`Config` 节点。

接入时只需注意：

- `search_fn` 必须与模型中的原始片段一致；
- 仅当字面量（如 `split` 尺寸）应作为通配符时设置 `ignore_literals=True`；
- `replacement_fn` 必须保持相同的输出、dtype、layout 和 alias 语义；
- 可复用的 cache 前处理应在 cache 初始化时完成，避免每步重复计算；
- 不希望 FX 展开的算子调用可以使用 `torch.fx.wrap` 包装；
- 算子已注册 Autograd 时直接调用，不要在 override 中重复实现正反向；
- 多个结构变体用一个 factory 生成多个 `PatternReplacement`，然后一次注册。

完整实现参考
`torchtitan_npu/compile/patterns/deepseek_v4/inplace_partial_rope.py`。

## 启用和验证

```bash
# 只开启快速调试，不注册 pattern
TORCHINDUCTOR_NPU_EXT_DEBUG=allfallback \
COMPILE_BACKEND=inductor ./scripts/run_train.sh

# 开启快速调试，并注册 inplace partial RoPE pattern
TORCHINDUCTOR_NPU_EXT_DEBUG=allfallback \
PATTERN_IMPORTS=torchtitan_npu.compile.patterns.deepseek_v4.inplace_partial_rope \
COMPILE_BACKEND=inductor ./scripts/run_train.sh
```

验证时确认 pattern 匹配数量、正反向数值、loss/grad norm，并通过 profiling 检查融合算子
是否生效以及是否新增 clone、copy 或 TensorMove。性能验证时不设置
`TORCHINDUCTOR_NPU_EXT_DEBUG=allfallback`。

> 注：当前 DeepSeek-V4 golden attention 不支持启用 inplace partial RoPE pattern。
