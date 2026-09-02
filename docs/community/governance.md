# 社区治理

## 项目定位

`torchtitan-npu` 是 `torchtitan` 的 Ascend NPU 适配插件，使用配置级 `override`、
PyTorch backend 注册机制和 NPU 专属算子扩展上游训练能力。

项目文档和技术结论以源码、配置、测试和脚本为依据。涉及设备、CANN、
`torch_npu`、dtype、并行方式或 checkpoint 格式的结论，应同时说明适用范围和验证状态。

## 协作原则

- **开放协作**：接受 Issue、PR、文档、测试、实验报告和场景反馈等形式的贡献。
- **证据优先**：兼容性、性能和数值结论应有源码、配置、日志、测试或可复现实验支撑。
- **上游友好**：与 NPU 无关的通用能力优先回馈上游 `torchtitan`；本仓聚焦 Ascend 适配。
- **最小改动**：优先复用已有模型、patch、override 和测试入口，避免重复实现。
- **质量优先**：影响训练正确性、checkpoint 兼容性、性能或分布式行为的改动必须提供相应验证。
- **可追溯**：重大设计、上游基线、版本配套和治理决策应通过 Issue、PR 或 SIG 记录留痕。

## 治理边界

`torchtitan-npu` 属于 CANN 开源社区，项目技术治理和社区协作依托 GitCode 仓库及
CANN `framework-adapter` SIG 展开。CANN 社区的正式角色定义、晋升、投票、公告和
非活跃退出机制以
[CANN 社区角色定义及晋升机制](https://gitcode.com/cann/community/blob/master/governance/role-definition-and-promotion-mechanism.md)
为准；本文只说明这些角色在本仓中的协作边界。

### User

User 指使用 `torchtitan-npu` 的用户。User 可以通过 GitCode Issue、讨论或 SIG 会议
反馈安装、训练、精度、性能、文档和易用性问题，也可以提出模型、算子、并行策略或
调试能力需求。

### Contributor

Contributor 指以任何形式参与本项目的人，包括：

- 提交代码、配置、文档、测试或性能数据。
- 参与 PR review、问题定位、版本验证或 SIG 讨论。

Contributor 不要求拥有仓库写权限。一次有效贡献即可成为 Contributor。

### Committer

Committer 是具备本仓写权限的项目成员。典型职责包括：

- 维护负责的代码路径、文档和测试质量。
- Review 并合入满足质量要求的 PR。
- 协助处理缺陷、回归、checkpoint 兼容性和发布前阻塞项。
- 推动 CI、测试、文档和开发流程持续改进。

Committer 应持续提供高质量贡献、负责任的 review 和可追溯的技术判断。

### Maintainer

Maintainer 负责特定模块或 CANN `framework-adapter` SIG 的运营维护。典型职责包括：

- 协调 Roadmap、版本配套、上游同步和关键技术方向。
- 组织重大架构、兼容性、分布式或性能变更的评审。
- 组织 SIG 例会、版本复盘和社区沟通。
- 吸纳并发展 Committer，推动项目长期维护。

Maintainer 应熟悉 `torchtitan`、`torchtitan-npu`、Ascend 软件栈和大模型训练流程，
并持续参与项目建设。

### PMC 与 TSC

项目管理委员会（PMC）和技术指导委员会（TSC）的职责、产生方式和退出机制遵循
CANN 社区统一治理文件。需要跨项目或影响社区方向的事项，应按需提交
`framework-adapter` SIG、PMC 或 TSC 流程。

## 决策机制

项目优先采用基于证据的共识决策：

1. 一般改动通过 PR review 达成共识后合入。
2. 影响架构、兼容性、checkpoint、性能基线、上游同步或版本配套的改动，应在 Issue、
   PR 或 SIG 会议中充分讨论，并提供相应验证依据。
3. 存在分歧时，由相关模块 Maintainer 组织讨论，记录不同方案的取舍和最终处理意见。
4. 对项目方向有重大影响的事项，应同步到 CANN `framework-adapter` SIG 或 CANN 社区
   治理流程。

## 版本基线

不同开发基线使用相应的软件配套。分支、上游 commit 和软件版本配套以
[安装指南中的版本配套表](../user-guides/installation.md#版本配套表) 为准。

涉及上游同步或依赖调整的 PR 应：

- 在描述中注明目标分支和上游 `torchtitan` commit。
- 同步检查 `requirements.txt`、`.ci/lint.sh`、patch 目标和测试入口。
- 需要时更新安装指南、相关功能文档和配置示例。
- 不直接套用其他开发基线的配置、命令或运行时结论。

## 贡献流程

### 提交前准备

1. 明确目标分支、改动范围和非目标，先读取相关源码、配置、测试和脚本。
2. 对共享模型组件、patch、override、分布式工具或 checkpoint 逻辑审计所有调用点。
3. 代码改动补充或更新相关测试；文档改动核对命令、路径、链接和支持范围。
4. 涉及数值、性能、checkpoint 或 NPU 行为时，记录设备、软件版本、并行方式、dtype、
   输入规模和实际验证结果。

### PR 内容

PR 应遵循仓内
[PR 模板](../../.gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md)，至少说明：

- 改动动机和预期行为。
- 受影响的模块、配置或用户路径。
- 实际执行的测试和验证命令；未执行项应说明原因。
- 兼容性、checkpoint、性能或数值行为的变化。

提交前可按 [Lint 指南](../developer_guides/lint_guide.md) 执行文件级或全仓 pre-commit
检查。文档、配置和代码示例应保留机器可读标识符的准确拼写。

### 反馈与协作渠道

- **Issue**：报告缺陷、兼容性问题、文档问题或功能需求。
- **PR review**：围绕代码、配置、测试和实验数据提出可操作意见。
- **SIG 会议**：讨论跨模块设计、版本配套和社区协作事项。
- **讨论区**：交流使用经验和未形成 Issue/PR 的问题。

## 晋升与退出

本仓不单独维护一套与 CANN 社区分离的晋升标准。Committer、Maintainer、PMC 和 TSC
的晋升、投票、公告、权限更新及非活跃退出流程，以
[CANN 社区角色定义及晋升机制](https://gitcode.com/cann/community/blob/master/governance/role-definition-and-promotion-mechanism.md)
为准。

本仓 Committer 申请应按代码仓库维度准备贡献材料，并通过
`framework-adapter` SIG 流程推进。材料应体现对相关代码路径的理解，以及持续的代码
贡献、PR review、问题处理或社区协作记录。

## 行为准则

社区讨论应保持专业、尊重和建设性。不同意见应围绕事实、代码、配置、实验数据和用户
场景展开。对恶意攻击、骚扰、泄露敏感信息或破坏社区协作的行为，维护者可依据 CANN
社区治理流程处理。
