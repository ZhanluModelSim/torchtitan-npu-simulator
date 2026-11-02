# override 目标规则

`override/` 是 `override.imports` 显式启用的 NPU/CANN 配置级替换层，不是包导入自动 patch 层；纯数值基线 variant 是唯一例外。

- 使用 `@override` 注册，配置写完整 `module.function`；`__init__.py` 不批量导入无关工厂。
- 保持输入输出、DTensor、sharding、checkpoint 与 `torch.compile` 语义。
- 可调用 `ops`，并只按需使用无副作用的上游待合入定义；不得直接依赖 `extensions`。
- 依赖 extension 的 override 必须在注册工厂声明此前提。`TORCHTITAN_NPU_DISABLE_EXTENSIONS` 是未实现的禁用开关目标接口，当前不能作为运行时配置使用，代码中也不存在读取该变量、在 extension 被禁用时 warn-once 的行为；该行为是开关落地后的目标。实现状态见 [extensions 规则](../extensions/AGENTS.md)。
