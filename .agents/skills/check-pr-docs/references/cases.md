# 文档同步判断案例

在区分本 PR 引入的问题、既有漂移和历史修复状态时，按需参考以下案例。案例用于说明
判断方法，不替代对待审 PR 的源码、配置、文档和执行链路的核对。

以下内容整理自 [PR #734 的验证记录](https://gitcode.com/cann/torchtitan-npu/pull/734)：
记录使用的 Skill 提交为 `21b4889dba1032cfad402b2a3ec3da28dd5fa08a`，复核目标分支为
`9b73a33`。该目标分支提交不是各历史 PR 的 merge-base，也不代表最新 master；下述状态
只描述这份记录，不表示已在后续版本重新回放。处理真实任务时，仍按主技能第 1 步取得
相应 base/head、merge-base 和修复基线，重新确认问题是否仍存在。

## 默认行为变更，说明后来已补齐

[PR #717](https://gitcode.com/cann/torchtitan-npu/pull/717) 将默认编译后端改为 AscendC，
当时缺少使用说明。上述复核基线的 `docs/feature_guides/torch_compile.md` 已说明该默认值
及启用方式。

- 历史结论：默认行为改变时需要同步文档。
- 复核结论：后续文档已覆盖该变化，不再重复修改；不能把历史遗漏直接列为当前缺陷。
- 重放时核对启动脚本和对应指南，不能只检查旧 PR 是否改过 Markdown。

## 测试入口后来拆分，依赖可能由环境提供

[PR #711](https://gitcode.com/cann/torchtitan-npu/pull/711) 曾让入口先执行 smoke，再执行
integration，说明没有交代组合入口与直接执行 integration 的区别。上述复核记录中，
两类测试已拆成独立入口，相关说明也已调整。

- 当前修复应追踪仍在使用的 workflow、包装脚本和 `.ci/` 入口，不把旧调用顺序写回文档。
- 新增 `pytest` 命令不等于缺少依赖：先核对实际镜像、`requirements-dev.txt` 和安装步骤。
  镜像内容或 CI 执行版本缺少证据时，标为待核实，不能认定已安装或直接补一份依赖。

## 版本标签过期，不同环境不必使用相同版本

[PR #949](https://gitcode.com/cann/torchtitan-npu/pull/949) 将训练环境的 PyTorch 声明改为
正式版 `2.14.0`，但对应安装说明仍标为 daily；CI 则使用另一套 nightly 安装路径。

- 核对同一用户安装路径中的 requirements 和软件配套表，确认后同步过期标签。
- CI 与用户安装环境用途不同，不能仅因版本不同就统一升级，也不能用 CI 通过证明另一套
  软件栈兼容。上述记录只核对声明和说明，没有实际安装或兼容性验证。
- `torchao` 与 `torchao-npu` 是不同发行包；声明前者不意味着后者已安装或应成为根依赖。

## 删除实现，未改动的文档也可能必须同步

[PR #1006](https://gitcode.com/cann/torchtitan-npu/pull/1006) 删除 reference 路径后，
`docs/feature_guides/deepseek_v4_tnd.md` 仍介绍被删除的实现，虽然该文档不在 PR diff 中。

- 对照 merge-base 中的实现和说明，再核对 head：原先一致、因本 PR 删除实现而失效的
  说明属于「必须同步」，不能归为「既有漂移」。
- 沿被删除入口搜索有效引用，只同步已经失效的说明。已更新的 CP 文档、仍保留的 Golden
  实现和仍需使用的 `attn-gym` 依赖不应跟着删除。

## 内部优化已有说明，关闭未合入的功能不补进当前文档

[PR #725](https://gitcode.com/cann/torchtitan-npu/pull/725) 减少前向计算中的 clone，并已在
README 说明缓冲区复用。上述记录中，该 PR 已关闭且未合入，旧实现也已从复核基线移除。

- 内部优化未改变公开契约、已有说明足够时，判为「无需同步」，不为了清单再补重复文档。
- 历史方案未合入且当前实现已不存在时，不把该功能写回当前指南；功能重新引入时，应按
  新的实现和 PR 范围重新判断。
