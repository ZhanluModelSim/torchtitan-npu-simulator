# extensions 规则

`extensions/` 是本仓永久额外扩展层。`torchtitan/` 按上游 TorchTitan 的目录归属存放对应扩展；`adaptor/` 存放按需使用的外部集成，不能由包入口自动导入。

- patch 型扩展将变更收敛到 `apply()` 并在模块末尾调用；所属 `__init__.py` 显式导入激活，注册型扩展使用明确注册 API。
- 除所属 `__init__.py` 外，其他模块不得导入带 `apply()` 的模块；只提供定义的无副作用模块可按需复用。直接 monkey-patch 只使用完整点路径赋值。
- 新增代码只依赖标准库、上游包、`extensions/` 内部模块和所需的 `patches/` 定义；不得新增对 `models`、`override` 或 `ops` 的依赖。现有跨层边是迁移债务，不以本规则宣称已经消除。
- extensions 可以导入 `torchtitan_npu.patches.*` 复用 patch 提供的定义，也可以通过 `torchtitan.*`、`torch.*` 或 `torch_npu.*` 使用已替换结果；`patches/` 不得新增反向依赖。
- 新增直接 monkey-patch 必须声明唯一 owner 和激活顺序。现有重复包装链是迁移债务，未经单独设计和行为验证不得合并。
- **当前状态：加载策略尚未实现。** `extensions/torchtitan/` 已按上游目录建立边界，`adaptor/` 已迁入本层；[包入口](__init__.py) 仍会无条件导入现有 extension。统一加载器和下列环境变量尚未实现。
- **目标接口（迁移时实现）：** 在 `extensions/__init__.py` 的唯一加载入口先读取 `TORCHTITAN_NPU_DISABLE_EXTENSIONS`：值为 `1` 时跳过全部 extension 的自动加载，但不影响 `patches`、`models`、`override`、`ops` 的导入。未禁用时，加载器再读取 `TORCHTITAN_NPU_DISABLE_COMPONENTS`：`1` 跳过全部已声明组件；否则按逗号分隔的组件名跳过对应组件。组件必须在加载器的注册表中有唯一名称；未知名称应报清晰错误，不能静默忽略。
- 实现该接口时，新增或迁移的加载器及对应测试必须落到 [包入口](__init__.py)、[components](components/AGENTS.md) 和 [测试规则](../../tests/AGENTS.md) 所指的位置；`torchtitan`、`components`、`compile` 的职责边界见各自规则文件。

| 目录 | 当前职责 | 当前加载方式 |
| --- | --- | --- |
| `torchtitan/` | 镜像上游结构的永久扩展 | 包入口按显式顺序导入其中的激活模块 |
| `adaptor/` | 可选外部 distribution 适配器 | 按量化或导出调用路径导入 |
| `components/`、`compile/`、`experiment/` | 组件、图编译和实验能力 | 由 owner 或调用路径导入 |
| `cpu_offload/`、`distributed/`、`mx_storage_reader/`、`novaswap/` | NPU 训练、状态与 checkpoint 支持 | 按 trainer、override、模型或 checkpoint 调用路径导入 |
