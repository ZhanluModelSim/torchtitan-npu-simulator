<!--
待合入的上游 PR：
- https://github.com/pytorch/torchtitan/pull/3430
- https://github.com/pytorch/torchtitan/pull/3634
- https://github.com/pytorch/torchtitan/pull/3864
-->

# TorchTitan 临时补丁

本目录仅保存已向 TorchTitan 上游提交、但尚未合入当前依赖版本的临时补丁。

固定格式（每个补丁文件必须遵守）：

- 文件**第一行**必须是 PR 链接注释：

  ```python
  # Pending upstream PR: https://github.com/pytorch/torchtitan/pull/NNNN
  ```

- 模块 docstring 需说明补丁内容，并以 "Remove this module after the TorchTitan dependency includes the PR." 结尾。
- 对原模块/类做属性替换的 monkey patch 必须封装在 `def apply() -> None:` 中并在文件末尾调用（`apply()`），补丁逻辑内使用完整模块路径（`import torchtitan.models.common.moe` 后以 `torchtitan.models.common.moe.X = ...` 赋值），不使用短导入别名。纯定义被模型目录引用的 backport 类（如 `BatchedLinear`、`BaseMaskHandler`、`SingleComplexRoPE`、`LoggedAuxLoss`、`VarlenCPMetadata`）不需要 `apply()`。

校验（除 `__init__.py` 等包胶水文件外，每个补丁文件都应命中）：

```bash
grep -L "Pending upstream PR: https://github.com/pytorch/torchtitan/pull/" \
  torchtitan_npu/patches/torchtitan -r --include="*.py"
```

| PR | 说明 |
| --- | --- |
| [#3430](https://github.com/pytorch/torchtitan/pull/3430) | 为变长注意力补充 CP 和 Full DTensor 支持 |
| [#3634](https://github.com/pytorch/torchtitan/pull/3634) | 补充 DeepSeek-V4 所需的公共组件及训练接入 |
| [#3864](https://github.com/pytorch/torchtitan/pull/3864) | 为 torchtitan 补充 LoggedAuxLoss 辅助损失框架 |

对应 PR 合入且 TorchTitan 依赖更新后，应删除相关补丁及导入；全部补丁清理完成后，删除本目录。
