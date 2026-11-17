# torchtitan_npu 目标结构与代码放置

新增和迁移触及的代码遵循本目标；历史代码按 [架构提案](../.agents/notes/proposed/architecture/2026-09-02-plugin-layout-and-injection-boundaries.md) 收敛。正式目标见 [目标架构](../docs/architecture.md)，当前实现状态以代码和该 Agent Note 的迁移矩阵为准。

## 代码放置

生产子包为 `config/`、`patches/`、`extensions/`、`override/`、`models/`、`ops/` 与 `tools/`。外部包适配位于 `extensions/adaptor/`。根目录只保留包入口与训练入口；不要新增其他生产代码落点。

| 目录 | 目标职责 | 允许的直接依赖 |
| --- | --- | --- |
| `config/` | NPU config schema、纯转换器与显式注册 API | 标准库与上游配置 API；当前包导入会安装 ConfigManager loader wrapper，拆除该副作用是后续目标 |
| `patches/` | 上游 `torchtitan` 或 `torch_npu` 的待合入补丁 | 不得新增对 `extensions/` 的依赖 |
| `models/` | 应由上游 TorchTitan 拥有、但尚未合入的模型定义、配置、并行化与 checkpoint 适配 | 现有跨层依赖是迁移债务，不宣称已清零 |
| `extensions/` | 本仓永久额外扩展、组件集成与图编译注册 | 可依赖 `patches/` 定义；不得新增对 `models/`、`override/` 或 `ops/` 的依赖 |
| `extensions/adaptor/` | 可选外部 distribution 与本仓公开配置/模型接口的适配 | 使用公开接口；现有 patch 私有依赖是迁移债务 |
| `override/` | `override.imports` 显式选择的 NPU/CANN 组件替换 | 可使用 `ops/`；现有对 `extensions/` 的依赖是迁移债务 |
| `ops/` | 设备 kernel、最小启动封装与 `torch.library` 注册 | 仅实现算子所需的后端代码 |
| `tools/` | 需要随 Python package 分发的 checkpoint 转换与导出工具 | 上游包与公开的本仓模型/集成 API |

新增代码遵守单向边界：`extensions/` 可以依赖 `patches/`，`patches/` 不得新增对 `extensions/` 的依赖；`extensions/` 不得新增对 `models/`、`override/` 或 `ops/` 的依赖。现有反向边在架构 Agent Note 中登记并逐项收敛。

模型无关能力放在最通用的适用位置。不要把通用 patch、extension、算子或配置逻辑放进模型专属文件；不要为了绕开上述依赖方向增加临时 wrapper。

## 注入与启用

当前 `__init__.py` 的自动导入顺序是：

```text
patches → config → extensions → ops
```

- `patches/` 先导入并完成上游替换；`extensions/` 随后可复用 patch 定义或经上游路径使用其效果。
- `config/` 当前通过导入 `manager` 安装 `ConfigManager._load_config` wrapper；该副作用服务于现有配置转换，尚未迁入 extension 激活入口。
- `override/` 不是自动导入层；它由 `override.imports` 显式选择。
- 收敛为 `patches → extensions` 是后续目标。当前 `config` 和 `ops` 仍由包入口导入，不能把目标顺序写成已实现行为。
- `TORCHTITAN_NPU_DISABLE_EXTENSIONS` 和 `TORCHTITAN_NPU_DISABLE_COMPONENTS` 是未实现的目标加载接口，当前不能作为运行时配置使用；实现状态见 [extensions 规则](extensions/AGENTS.md)。

## patches 与 extensions 的时序边界

以下是目标约束；尚未迁移的历史入口不能据此被视为已经满足。

- `patches/` 是尚未合入上游代码的替身，可在显式激活顺序中替换上游类、函数及 `.Config` schema。凡影响上游 dataclass schema 或默认工厂的替换，必须先于任何会导入上游 `trainer` 的激活模块执行。
- `patches/torchtitan/__init__.py` 的显式导入列表定义当前激活顺序。该列表不能因 import 自动排序、代码整理或自动发现改变；用相应检查保护具体顺序，不使用与当前列表不符的「trainer 必须最后导入」规则。
- 替换上游符号后，必须刷新或明确说明所有已被上游 `from ... import ...` 捕获的绑定。无法确认不存在未知捕获点时，优先使用上游公开注册机制，或将能力改为下述 extension 机制。
- `extensions/` 在 patches 之后运行，默认所有上游模块已导入、所有 dataclass 已生成；新增 extension 不得依赖「导入足够早」，也不得替换已被上游 dataclass 捕获的 `.Config`。
- extension 仅可通过时序无关的机制生效：配置外包装、运行期方法 seam、单目标 hook 注册组件、公开注册 API（如 `torch.library` 或 graph-pass registry）。
- 某项能力既不属于待上游化的 patches，又不能使用上述机制时，不实施 monkey-patch；应提出稳定扩展点，或扩展本仓的配置外包装。

## 迁移守则

- `patches/torchtitan/` 镜像上游模块结构。实际 patch 模块在许可证头后的固定头部区域记录 `Upstream-PR`；`__init__.py` 与 `patches/torch_npu/` runtime compatibility patch 不适用该要求。上游合入并升级 pinned 依赖后删除对应 patch。
- 需要 monkey-patch 的模块将变更收敛到 `apply()` 并在模块末尾调用；所属 `__init__.py` 显式导入激活，禁止自动发现。除所属 `__init__.py` 外，其他模块不得导入带 `apply()` 的 patch/extension 模块；只提供定义的无副作用模块可按需复用。
- monkey-patch 只使用完整点路径赋值，不能以短别名或预先绑定引用掩盖替换目标。linter 对必要属性替换报错时，在赋值行添加窄的抑制注释。
- 新增 monkey-patch 必须声明唯一 owner，其余能力通过该模块公开的注册接口接入。现有重复包装链是迁移债务，未经独立设计与行为验证不得合并。extensions 的固定目标子树及组件/compile 细则见 [torchtitan](extensions/torchtitan/AGENTS.md)、[components](extensions/components/AGENTS.md)、[compile](extensions/compile/AGENTS.md)。
- 目录边界、导入顺序、checkpoint 语义或可选组件行为发生变化时，在同一变更中更新局部规则、根规则、当前架构文档和 Agent Note。
