# torchtitan-npu 架构边界

本文区分当前实现、禁止新增的依赖与后续目标。当前行为以源码为准；迁移状态和历史债务记录在[目录与注入边界提案](../.agents/notes/proposed/architecture/2026-09-02-plugin-layout-and-injection-boundaries.md)。

## Repository layout

```text
torchtitan_npu/  Ascend NPU 插件主体
experiments/     独立实验包；当前包含 torchao-npu distribution
docs/            架构、开发者、功能和用户文档
examples/        可运行 recipe、模型配置与示例 launcher
scripts/         训练、转换、合规检查与辅助脚本
tests/           CPU UT、NPU ST 与 tooling 自测
.agents/         Agent Notes 与可复用 skills
```

根目录只保留工程配置、包/训练入口及上述目录。`third_party/` 的普通内容默认忽略；已跟踪的 `third_party/agent-skills/KPBot` 是 submodule。不新增其他生产代码落点。

## 插件主体

`torchtitan_npu/` 的正式生产子包为 `config/`、`patches/`、`extensions/`、`override/`、`models/`、`ops/` 与 `tools/`。可选外部包适配位于 `extensions/adaptor/`。

| 子包 | 职责 | 直接依赖边界 |
| --- | --- | --- |
| `config/` | NPU config schema、纯转换器和显式注册 API | 当前包导入会安装 ConfigManager loader wrapper；目标是移至 extension 激活入口且不反向导入具体 extension |
| `patches/` | 上游 `torchtitan` 或 `torch_npu` 的待合入补丁 | 不得新增对 `extensions/` 的依赖；现有反向边记录为迁移债务 |
| `models/` | 尚未上游化的模型定义、配置、并行化与 checkpoint 适配 | 现有跨层依赖为迁移债务，不以本文宣称已清零 |
| `extensions/` | 本仓永久额外扩展、组件集成与图编译注册 | 可以依赖 `patches/` 提供的定义；不得新增对 `models/`、`override/` 或 `ops/` 的依赖 |
| `extensions/adaptor/` | 可选外部 distribution 的适配器 | 使用公开配置、模型、算子或 extension API；对 patch 私有符号的现有使用为迁移债务 |
| `override/` | `override.imports` 显式启用的 NPU/CANN 组件替换 | 可使用 `ops/`；现有对 `extensions/` 的依赖为迁移债务 |
| `ops/` | 设备后端 kernel、最小启动封装与 `torch.library` 注册 | 仅保留实现算子所需的后端代码 |
| `tools/` | 随 package 分发的 checkpoint 转换与导出工具 | 依赖公开模型与 integration API，不承载训练期自动注入 |

模型无关能力放在最通用的适用位置。完整的开发约束见 [主包规则](../torchtitan_npu/AGENTS.md) 及其目录级规则。

## 导入与启用

当前 `torchtitan_npu/__init__.py` 的自动导入顺序是：

```text
patches → config → extensions → ops
```

这是当前运行时行为，不是目标顺序。`patches` 先替换上游实现；`extensions` 可从 `torchtitan_npu.patches.*` 复用 patch 提供的定义，也可经 `torchtitan.*`、`torch.*` 或 `torch_npu.*` 使用已替换结果。反向的 `patches → extensions` 依赖不得新增。

`config` 当前导入 `manager` 时会安装 `ConfigManager._load_config` 的转换 wrapper；该副作用是现有配置转换链的一部分。将其改为仅由 extension 激活入口安装是后续目标，不是当前行为。

`override` 不自动加载；一次训练由 `override.imports` 显式选择。将自动导入收敛为下列顺序是后续目标，不能据此描述当前行为：

```text
patches → extensions → selected override
```

> **当前未实现。** `TORCHTITAN_NPU_DISABLE_EXTENSIONS` 与 `TORCHTITAN_NPU_DISABLE_COMPONENTS` 仅是目标加载接口，当前包入口仍会无条件加载已有 extension；不得将它们作为可用运行时配置。

实现后，`TORCHTITAN_NPU_DISABLE_EXTENSIONS=1` 只禁用 extensions。`TORCHTITAN_NPU_DISABLE_COMPONENTS=1` 禁用全部已声明组件，逗号分隔组件名只禁用对应组件；两个开关都不得破坏 patches、models、override 或 ops 的可导入性。

### 两阶段时序语义

`patches` 与 `extensions` 是不同的注入阶段。`patches` 承载尚未合入上游的替身，能够在显式激活顺序中替换类、函数和 `.Config` schema；影响 dataclass schema 或默认工厂的替换必须早于依赖该符号的模块。`patches/torchtitan/__init__.py` 的现有导入顺序是运行时契约，必须通过显式列表和测试维护，不以「trainer 必须最后导入」这类与当前实现不符的绝对规则约束。

patch 替换后必须刷新或说明上游 `from ... import ...` 已捕获的绑定；无法穷尽捕获点时，改用公开注册机制或 extension。extensions 默认上游模块和 dataclass 都已存在，因此不依赖早期导入，也不替换被上游 dataclass 捕获的 `.Config`。它只使用配置外包装、运行期方法 seam、单目标 hook 注册组件或公开注册 API（例如 `torch.library`、graph-pass registry）。不属于 patches 且无法通过这些机制实现的能力，不以 monkey-patch 落地，而应提出稳定扩展点或扩展本仓配置外包装。

## 目标子层

### patches

`patches/torchtitan/` 镜像上游模块路径。新增实际 patch 模块在许可证头后的固定头部区域使用 `Upstream-PR:` 记录上游依据；`__init__.py` 与 `patches/torch_npu/` runtime compatibility patch 不适用该要求。当前 architecture ratchet 同时兼容历史 `Pending upstream PR`/`Upstream PR` 标记，并允许基线中已登记的缺失项，因此它只阻止新增元数据债务。上游合入并升级 pinned 依赖后删除该 patch 和激活入口。强制 eager 的上游行为兼容放 `extensions/torchtitan/`。

带副作用的 monkey-patch 模块把变更收敛至 `apply()` 并在模块末尾调用，由所属 `__init__.py` 显式导入激活。其他模块不得导入带 `apply()` 的模块；定义型无副作用模块可按需复用。替换必须使用完整点路径赋值。

### extensions

`extensions/torchtitan/` 放镜像上游结构的 monkey-patch 激活模块；新增替换必须声明 owner 和顺序。现有 `MetricsProcessor.log`、`CheckpointManager` 与 `ConfigManager._load_config` 的多层包装保留为历史例外，包装顺序由对应测试保护，本轮不合并其业务行为。`extensions/components/` 放带声明名称的自包含集成；当前 TrainerEx、checkpoint extension 和 DeepSeek-V4.1 配置仍存在组件顶层接线或反向使用，这是迁移债务，不是新组件约束。`extensions/compile/` 放 graph-pass 注册基础设施和具体 pattern。

`extensions/` 当前还包含 `adaptor/`、`cpu_offload/`、`distributed/`、`mx_storage_reader/`、`novaswap/` 与 `experiment/`。其中仅包入口显式列出的模块自动加载；其余目录由调用路径按需导入。各目录的职责和加载方式见 [extensions 规则](../torchtitan_npu/extensions/AGENTS.md)。

### override、models 与 ops

override 使用 `@override` 注册并由完整 `module.function` 选择；它保持输入输出、DTensor、sharding、checkpoint 与 `torch.compile` 语义。依赖 extension 的 override 显式声明此前提；extension 禁用时 warn-once 是禁用开关落地后的目标行为。

目标上 models 保持对 NPU 实现细节的零直接依赖，需要 kernel 时经上游或 override 注入。当前 DeepSeek-V4/V4.1 与 Qwen3.5 仍有 extension 或 override 的直接导入，属于已登记的迁移债务；新增模型代码不得复制这种模式。ops 按后端或语言组织，后端目录声明 token，算子命名为 `torchtitan_npu::{backend}_{kernel}`。ops 不放上游替换或组件封装，反向通过 `register_autograd` 注册，且 `ops/__init__.py` 不批量导入子模块。

## 验证

架构迁移阶段同步更新代码、根与受影响目录的 `AGENTS.md`、本文及 Agent Note。使用 CPU UT、NPU ST 和数值验证证明训练入口、override 选择、checkpoint 语义与已验证数值行为不回归。具体测试与数值要求见 [根开发规则](../.agents/AGENTS.md) 和 [测试规则](../tests/AGENTS.md)。
