# examples 规则

`examples/` 放置面向用户的 recipe launcher 和小规模 debug 入口。

- 每个训练 launcher 选择 `torchtitan_npu_recipes.examples.<model>` 中与脚本同名的完整 Python 工厂。数据、训练预算、优化器、并行、checkpoint、profiling、量化和 `override.imports` 均归 Python 工厂，不在 shell 拼接字段参数。
- shell 只保留设备、节点、CANN、网络等启动资源与一条共用 launcher 调用，允许通过 `MODULE` / `CONFIG` 选择自定义工厂。
- 保留用户传入的额外 CLI 参数；不在 wrapper 中隐式改变 checkpoint、数值或并行语义。
- 每个示例的模型、硬件代际、进程数、可选依赖和默认 override 必须可从脚本和相关文档中确认。
- `debug/` 只表示小规模调试配置，不代表数值或性能已验收。
