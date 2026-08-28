---
name: developer-tests-review
description: "更新或 review TorchTitan-NPU PR 的 UT、NPU ST 和测试格式。更新测试时修改测试代码、执行相关检查并按规则迭代到通过；review 测试时只读代码并输出报告，不修改或执行测试。"
---

# TorchTitan-NPU 测试 workflow

本 skill 有两个 workflow：`更新测试` 和 `review测试`。两者都使用同一套 UT、NPU ST 和格式规则，但修改权限和执行权限不同。

## 共同范围

开始前完整读取仓库存在的 `.agents/AGENTS.md`；需要统一用语时读取 `references/terminology.md`，输出前读取 `references/report-output.md`。

范围只限本仓库可控的生产代码、测试代码、测试入口和测试资源。PR 若仅修改 torchao、torchft 等第三方仓库的源码或测试，或仅更新第三方依赖的版本/提交引用，且未改变本仓库 `torchtitan_npu/`、`tests/`、`.ci/` 或 `.gitcode/` 中的适配调用、注册、配置、入口或资源，则不纳入本 workflow 的 UT、ST 或格式审查，结论写“不适用”；第三方仓库的测试由其自身仓库和规则负责。只有该依赖变更会影响本仓库的适配接口、注册/配置、训练调用路径或测试入口时，才将受影响的本仓路径纳入审查（即使本仓没有对应源码 diff），并把第三方变更记录为外部前置条件，不以本仓测试替代第三方仓库测试。

UT 和 ST 不是同一条覆盖链路。UT 审查 CPU 上可观察的模块行为、参数传递、路径等价性和独立预期；ST 只审查真实 NPU 集成测试是否进入 PR 改变的训练路径。ST 不读取 CPU UT 的覆盖结论，也不把 CPU UT、CPU oracle 或单元测试计入 ST 覆盖。

测试格式始终使用 [references/format-review.md](references/format-review.md) 独立审查。格式问题不能代替 UT 正向功能或 ST 路径覆盖结论。

## Workflow 1：更新测试

`更新测试` 用于用户要求补充、修正或重构 testcase。此 workflow 可以修改测试代码、测试入口、suite 注册、fixture、golden 或测试配置；除非用户另有授权，不修改生产实现来迁就测试。

### 1. 固定范围并读取规则

记录 PR 改变的生产路径、已有测试、测试入口和资源条件。UT 更新读取 [references/ut-review.md](references/ut-review.md)；ST 更新读取 [references/st-review.md](references/st-review.md)，并按其中要求读取目标分支的 `tests/integration_tests/README.md` 和实际 runner；格式问题读取 [references/format-review.md](references/format-review.md)。

### 2. 设计最小修改

先确认现有测试为什么不能直接提供证据，再修改或新增最小 testcase。UT 必须保护对应正向功能和独立 expected；ST 必须进入实际模型、override、并行和编译组合，并接入现有 `tests/integration_tests` runner、suite 和结果检查方式。

### 3. 执行相关检查

测试代码或测试入口修改后，执行受影响的 testcase 和必要的收集、静态检查或 lint。UT 修改至少执行修改的 unit test；ST 修改至少通过现有 integration runner 执行修改或新增的 case；涉及 golden 时同时执行对应比较。没有真实 NPU 或其他必需环境时，不得把未执行写成通过。

### 4. 按规则 review 并迭代

检查执行结果和测试代码是否满足相应规则，发现失败、覆盖缺口、无效断言、同源 expected、路径未启用、状态泄漏或格式问题时，继续修改并重新执行。循环顺序是“修改 → 执行 → review → 修正”，直到相关检查通过且没有未解决的规则问题。这一步建议使用 subagent 以减小上下文污染。

### 5. 输出更新结果

报告修改的文件、每个 testcase 保护的正向功能或 NPU 路径、执行的命令及结果、golden 是否更新，以及仍受环境限制的检查。只有实际执行并通过的检查才能标记为通过。

## Workflow 2：review测试

`review测试` 用于静态审查已有 testcase 或 PR 中的测试改动。此 workflow 只读，不修改测试代码、生产代码、golden 或 skill，也不导入、收集或执行测试。

### 1. 固定审查对象

记录基线版本、PR 版本、生产代码路径、PR 测试、已有 UT/ST、测试入口和设备预算。只根据实际生产代码和执行链路判断覆盖，不根据测试名称猜测。

### 2. 选择审查规则

在选择规则前，先从 `requirements.txt` 和 `.ci/lint.sh` 确认固定的 TorchTitan commit。凡是需要拆解前向或训练流程的审查，都必须读取该 commit 对应的上游 torchtitan 源码，沿本仓 `patches`、`override`、模型或算子与上游 trainer、model 和 consumer 的真实调用链核对。上游源码只作为调用链和接口基线；上游仓库的测试不计入本仓 UT/ST 覆盖。

UT 读取 [references/ut-review.md](references/ut-review.md)，按正向功能单元检查现有 UT；ST 读取 [references/st-review.md](references/st-review.md)，只对 `tests/integration_tests` 的实际 testcase 做路径覆盖比对；格式读取 [references/format-review.md](references/format-review.md) 独立列出文件、收集、命名和隔离问题。

审查范围限定为本仓 `torchtitan_npu/`、`tests/`、`.ci/` 和 `.gitcode/` 中参与产品运行或测试执行的内容，以及为确认其真实调用链所需读取的上游代码。若 PR 仅修改与本仓适配接口、注册/配置、训练调用路径和测试入口均无关的第三方代码，则不纳入本仓 UT/ST 覆盖，结论写“不适用”。若依赖变更影响本仓适配路径，即使本仓没有对应源码 diff，也要审查受影响的本仓路径，并将该上游变更记录为外部前置条件。

### 3. 输出 review 报告

报告给出已覆盖、部分覆盖、未覆盖、检查无效或不适用的证据，指出准确的测试文件、case 名称、配置组合和最小修改方向。review测试 workflow 不因为发现缺口而直接修改或执行；只提交报告。

## 共同结论

结论只使用：`可以合入`、`补充测试后合入`、`暂停并澄清`、`不适用`。`review测试` 固定写明“测试执行：未执行（仅静态审查）”；`更新测试` 必须列出实际执行结果。
