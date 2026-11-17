# 测试规则

`tests/unit_tests/` 是 CPU 生产语义与静态契约测试根；`tests/smoke_tests/` 是真实 NPU runtime 上的小范围设备语义验证根；`tests/integration_tests/` 是 NPU 集成与训练测试根。测试尽量镜像生产代码；脚本和 skill 自测放 `tests/unit_tests/tooling/`，不作为产品行为覆盖证据。

需要新增、修正或重构 testcase 时，使用 [developer-tests-review 的“更新测试” workflow](../.agents/skills/developer-tests-review/SKILL.md#workflow-1更新测试)：它允许修改测试并要求执行相关 UT、ST、runner 或 golden 验证。仅审查 PR 或既有测试是否覆盖真实路径时，使用同一 skill 的 [“review测试” workflow](../.agents/skills/developer-tests-review/SKILL.md#workflow-2review测试)：它只读，不修改或执行测试。两种 workflow 都先读取本文件；详细的 UT、ST、格式和报告规则只在 skill 中维护，不在此重复。

Python 逻辑至少运行相关 CPU UT；分布式、NPU kernel、override、patch 或训练行为补充 NPU 验证。数值重构比较同 checkpoint、固定随机性的逐 step loss 与 grad_norm，并报告实际命令和未运行原因。
