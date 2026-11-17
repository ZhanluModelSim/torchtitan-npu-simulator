# models 目标规则

`models/` 是上游待合入层的一部分，放置应由 TorchTitan 拥有但当前版本尚未包含的模型定义、配置、并行化和 checkpoint 适配。

- 不以包导入副作用替换上游；通过正常注册、配置或调用路径按需加载。
- **目标状态：** 与 `patches` 一起自包含，只依赖标准库、上游包和本层内部模块；新增代码不得直接依赖 `extensions`、`override` 或 `ops`。
- **当前债务：** DeepSeek-V4/V4.1 的配置、并行化、checkpoint 适配仍直接使用 extension 或 override，Qwen3.5 也直接使用模型专属 override。它们的收敛范围记录在架构 Agent Note，不作为新增依赖的先例。
- 模型需要 NPU kernel 时应经上游或 override 注入，不新增对 `ops` 的直接依赖。
- `train.py` 只导入 `torchtitan_npu` 以激活 patches 与 extensions，不导入其他 `torchtitan_npu` 子模块；真正训练入口来自上游 TorchTitan。
