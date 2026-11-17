# extensions/components 目标规则

本目录放带声明名称的自包含集成单元，尤其是可选独立包的适配。

- 组件不直接 monkey-patch，也不定义 `apply()`；它们只通过 `extensions/torchtitan` 的注册接口接线。
- 依赖方向固定为 `extensions/torchtitan → extensions/components`，禁止反向导入。
- **目标状态：** 核心 extension 不在模块顶层导入组件；组件仅在需要时惰性加载。组件禁用后的可用性在禁用开关实现时验证。
- **当前债务：** `TrainerEx` 顶层导入 SDC，checkpoint extension 顶层导入 validation，DeepSeek-V4.1 配置直接导入 gradient clipping。它们是现有接线，不能作为新组件的依赖模式；收敛状态记录在架构 Agent Note。
- `TORCHTITAN_NPU_DISABLE_COMPONENTS` 是未实现的目标开关；落地后，值为 `1` 禁用全部声明组件，逗号分隔的组件名只禁用对应组件。该开关只影响 extension 的组件接线。
