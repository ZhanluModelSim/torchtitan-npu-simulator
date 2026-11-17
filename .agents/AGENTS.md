# torchtitan-npu 开发指南

`torchtitan-npu` 是 [torchtitan](https://github.com/pytorch/torchtitan) 的 **Ascend NPU 插件仓**。
本仓不直接修改上游 torchtitan checkout，而是通过包导入 patch、配置级 `@override`、模型模块和 NPU 算子将适配能力叠加到上游之上。

## 核心原则

1. **PyTorch 原生训练技术。** torchtitan 核心的训练基础设施和并行代码不依赖非 PyTorch 库。作为插件仓，torchtitan-npu 可使用 `torch_npu` 等外部库，但应尽可能复用 PyTorch 原生接口。

2. **查明根因再修复。** 不做绷带式修补。在提出方案前理解「为什么」出错。如果一个改动看似有效但无法解释原因，需要更深入排查。

3. **复用优于重复。** 新写代码前，检查已有实现是否已覆盖需求。尽量统一跨模型的相似代码路径，不要给每个模型创建独立 wrapper。若上游（torchao、PyTorch）已提供功能，优先使用。

4. **不要将实验泄漏到核心。** 实验性代码应与公共适配逻辑隔离，不要在核心 patch、override 或模型文件中添加 `if experiment_x:` 分支。

5. **保护已验证的代码路径。** 修改已收敛的代码时务必谨慎。标记可能导致现有用户代码或 checkpoint 静默失效的改动。存疑时主动询问。

6. **审计所有调用点。** 修改共享代码（公共模型组件、配置字段、分布式工具）时，检查并更新所有调用点，包括当前维护的 DeepSeek-V3.2、DeepSeek-V4、对应 override 和 patch 注册入口。

## 插件仓专属原则

1. **不修改上游 checkout。** torchtitan-npu 不直接编辑 torchtitan 源码。适配按职责放入以下位置：
   - **Package patch**（`torchtitan_npu/patches/`）：补齐 PyTorch PrivateUse1 backend，或替换当前运行时中缺失的 Python/CANN 符号。`patches/torchtitan/` 只允许有上游依据的临时 patch；`patches/torch_npu/` 用于当前有效的 NPU runtime compatibility patch
   - **配置级 override**（`torchtitan_npu/override/`）：使用 `@override` 声明组件替换，通过 `override.imports` 显式启用
   - **模型模块**（`torchtitan_npu/models/`）：提供 DeepSeek-V3.2 和 DeepSeek-V4 的模型配置、并行化与 checkpoint 适配
   - **NPU 算子**（`torchtitan_npu/ops/`）：封装 CANN 或设备专属算子能力

2. **理解 patch 生效机制。** `torchtitan_npu/__init__.py` 导入 `torchtitan_npu.patches`，后者继续导入 `torch_npu` 和 `torchtitan` 子包，因此导入 `torchtitan_npu` 即会注册或应用其中的入口 patch。只有需要随包导入生效的 patch 才应加入对应的 `__init__.py`；供模型或 override 直接复用的兼容模块按调用路径显式导入。

3. **Override 显式启用。** 每个 override 工厂使用 `@override` 注册，并以完整的 `module.function` 写入 `override.imports`。不要在 `__init__.py` 中批量导入具体 override。

4. **固定上游基线。** `requirements.txt` 声明当前 torchtitan 版本；`.ci/setup_torchtitan.sh` 从该版本推导对应 tag。调整上游版本时检查 patch 目标、函数签名、模型接口和测试是否仍有效。

5. **临时 patch 可删除。** `torchtitan_npu/patches/torchtitan/` 只保存已提交上游但当前依赖版本尚未包含的临时补丁。补丁文件必须记录对应 PR；上游合入并更新依赖后删除相关补丁和导入。

6. **测试目录与执行入口同时维护。** `tests/unit_tests/` 是 CPU 测试的统一执行根；生产语义测试按 `torchtitan_npu/` 镜像，仓库脚本和 skill 自测放在 `tests/unit_tests/tooling/`，执行时不计入产品 UT 覆盖。NPU 模型 ST 只放在 `tests/integration_tests/`，不以 CPU UT、kernel smoke 或 runner 自测替代。详细的目录迁移方案见 [`docs/developer_guides/unit-test-architecture.md`](../docs/developer_guides/unit-test-architecture.md)，低价值测试筛选见 [`docs/developer_guides/low-value-ut-principles.md`](../docs/developer_guides/low-value-ut-principles.md)。

## 代码风格（继承上游 torchtitan）

### 命名

- 名称必须 **准确、描述性、反映实际作用域**。不要在生产代码中使用 `toy`、`test` 或 `temp`；这类上下文放在 docstring 中。
- 遵循上游约定：匹配 torchao 和 PyTorch 的命名。
- 计数使用 `num_` 前缀（如 `num_expert_groups` 而非 `n_expert_groups`）。

### 代码放置

代码放到 **最通用的适用位置**：

| 目录 | 职责 |
| --- | --- |
| `torchtitan_npu/config/` | NPU 配置 schema、转换器和注册 API |
| `torchtitan_npu/patches/torchtitan/` | 包导入时生效的上游临时补丁 |
| `torchtitan_npu/patches/torch_npu/` | 当前有效的 NPU runtime compatibility patch；保留其自动加载和生效时机 |
| `torchtitan_npu/extensions/` | 永久扩展、组件集成和可选外部 adaptor |
| `torchtitan_npu/override/` | 通过 `override.imports` 显式启用的配置级组件替换 |
| `torchtitan_npu/models/` | DeepSeek-V3.2、DeepSeek-V4 模型配置、实现、并行化和 checkpoint 适配 |
| `torchtitan_npu/ops/` | CANN 与 NPU 专属算子封装 |
| `torchtitan_npu/tools/` | 随 package 分发的 checkpoint 转换与导出工具 |
| `tests/unit_tests/` | CPU 单测和静态契约测试 |
| `scripts/` | 训练、合规检查及其他仓库级脚本 |

不要把模型无关的功能放在模型特定文件中。

### 断言与错误处理

- **`ValueError`** 用于用户可见的错误（配置错误、无效输入）。
- **`assert`** 仅用于表示程序错误的内部不变量。
- 分布式代码中显式验证 mesh 维度、tensor placement 和配置值 — 不要假设 1D mesh 或特定 placement。
- 代码路径静默跳过用户配置时，**发出 warning**。

### 参数与配置

- 重要参数放前面，次要参数放后面。
- 首个位置参数之后优先使用 keyword-only 参数。
- 必需配置字段不要用 `None` 默认值。
- `dataclasses.replace()` 是浅拷贝：嵌套 dataclass 和 list/dict 字段共享引用。需要深拷贝时显式处理。

### 注释与文档

- 仅为真正不明显的内容添加注释：维度语义、并行梯度 placement、workaround 存在的原因。
- 使用 TODO 注释标记已知限制并附简要说明。
- 描述放在 docstring 中，不要放在名称里。
- 注释使用英文，文档优先使用中文。

## 标准开发流程

### 1. 获取上下文

- 先确认任务涉及的目录、模型、并行策略和是否影响训练数值。
- 先读取相关源码、配置、测试和脚本；涉及 override 或 patch 时再读取对应目录的说明文档。
- 文档任务使用 `.agents/skills/write-torchtitan-npu-docs/` 中的项目规范。
- 涉及上游同步时，先核对 `requirements.txt` 与 `.ci/setup_torchtitan.sh` 中的固定 commit。

### 2. 实施修改

- 保持改动最小，只改完成目标所需的文件。
- 复用现有 patch、override、模型和 `ops` 实现。
- 新增或修改 patch、override、模型或算子后，同步检查注册入口和所有调用点。
- 对数值、分布式、checkpoint、模型加载路径保持保守；存疑时先给出风险和验证方案。

### 3. 代码检查

先确认 `/tmp/torchtitan` 指向仓库约定的 `torchtitan` checkout，再从仓库根目录运行完整 Lint：

```bash
python -m pre_commit run --all-files --show-diff-on-failure
```

完整 `pre-commit run --all-files` 通过后，才视为 Lint 通过。定位失败时直接依据 hook 输出、相关配置和改动内容分析。

### 4. 测试与数值验证

- Agent 接到精度对齐请求，或准备创建、更新、提交 PR 时，必须先阅读[精度对齐开发指南](../docs/developer_guides/numerical_alignment.md)，依据对齐目标和变更文件列表确定必要的验证项目，并逐项核对完成情况；允许跳过的项目须说明依据。
- 新增、修正、重构或 review 测试时，使用 `.agents/skills/developer-tests-review/` 选择「更新测试」或「review测试」 workflow；前者修改并执行相关测试，后者只做静态审查，不修改或执行测试。
- Python 逻辑改动至少运行相关单元测试；CPU 单测使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -v --tb=short tests/unit_tests -x`。
- 涉及分布式、NPU kernel、override、patch 或模型训练行为时，补充对应 NPU 冒烟或集成测试。
- 数值验证：
  - 非计算性改动（重构、activation checkpointing 调整等）必须保证修改前后 **loss 完全一致**；计算性改动需在代表性数据集（如 C4）上展示 loss 收敛。
  - 对齐验证须加载同一 checkpoint 并固定 NPU 随机性，相同并行策略下两次运行的 loss 和 grad_norm 应一致；**禁止** 使用 `--debug.deterministic_warn_only`。
  - 证明 bit-wise 一致时，应比较 TensorBoard 中未截断的逐 step `loss` 和 `grad_norm`；stdout 的有限显示精度不能作为唯一依据。

### 5. PR 与流水线

- 需要精度验证的改动，必须在 PR 描述中提供对应验证记录：比较双方的代码版本及基线来源、运行环境与关键配置、可复现命令、指标和判定结果，以及原始日志或结果文件的可访问链接。未执行或未通过的项目必须明确状态、原因及待完成项，不得写为验证通过；无需精度验证时，须说明开发指南中的跳过依据。
- PR 描述解释「为什么」而非只写「做了什么」；非简单改动附数值证据，模型变更说明 checkpoint 兼容性。
- 准备、修改或 review PR 时，若改动涉及公开行为、模型支持、配置、训练或测试入口、依赖或软件栈，或用户明确要求检查文档遗漏，使用 `.agents/skills/check-pr-docs/`。默认只读；仅在用户明确要求同步修改文档或依赖时进入修改模式，已有授权继续有效。纯内部重构、普通新增测试且公开契约不变时不自动触发；仅推送已审查且未变化的内容时也不触发。只读模式不编辑文件、不运行可能改文件的检查。
- 已安装 `gitcode-pr` skill 时，创建或推送 PR、读取改动与评论使用该 skill，并按 `.gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md` 填写。标题使用正确的英文类型标签，如 `feat`、`fix`、`refactor`、`docs` 或 `test`；`类型` 和 `Checklist` 只勾选真实完成项，`如何测试` 写实际命令或未执行原因。
