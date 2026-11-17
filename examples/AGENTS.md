# examples 规则

`examples/` 放置面向用户的 recipe launcher、参数组合和小规模 debug 入口。

- launcher 只组合已实现的 module、config 和 `override.imports`，不承载可复用生产逻辑。
- 保留用户传入的额外 CLI 参数；不在 wrapper 中隐式改变 checkpoint、数值或并行语义。
- 每个示例的模型、硬件代际、进程数、可选依赖和默认 override 必须可从脚本和相关文档中确认。
- `debug/` 只表示小规模调试配置，不代表数值或性能已验收。
