<!--
待合入的上游 PR：
- https://github.com/pytorch/torchtitan/pull/3430
- https://github.com/pytorch/torchtitan/pull/3634
- https://github.com/pytorch/torchtitan/pull/3864
-->

# TorchTitan 临时补丁

本目录仅保存已向 TorchTitan 上游提交、但尚未合入当前依赖版本的临时补丁。每个补丁文件必须在文件头部注释中记录对应 PR 的完整链接。

| PR | 说明 |
| --- | --- |
| [#3430](https://github.com/pytorch/torchtitan/pull/3430) | 为变长注意力补充 CP 和 Full DTensor 支持 |
| [#3634](https://github.com/pytorch/torchtitan/pull/3634) | 补充 DeepSeek-V4 所需的公共组件及训练接入 |
| [#3864](https://github.com/pytorch/torchtitan/pull/3864) | 为 torchtitan 补充 LoggedAuxLoss 辅助损失框架 |

对应 PR 合入且 TorchTitan 依赖更新后，应删除相关补丁及导入；全部补丁清理完成后，删除本目录。
