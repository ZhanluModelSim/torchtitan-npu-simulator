# ops 目标规则

`ops/` 只承载设备后端 kernel、最小启动封装与 `torch.library` 注册。

- 按后端或语言划分目录；每个后端目录声明自己的 backend token，算子名为 `torchtitan_npu::{backend}_{kernel}`。
- 不放上游算子后端替换、TorchTitan 组件封装或一般辅助函数。
- 反向通过 `register_autograd` 注册，不公开 backward 算子；`ops/__init__.py` 不批量导入子模块。
