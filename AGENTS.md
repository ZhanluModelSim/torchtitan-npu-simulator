# torchtitan-npu 仓库规则

本文件是跨平台可读的仓库入口。通用开发流程、测试和验证要求见[完整开发指南](.agents/AGENTS.md)；目录级规则优先于本文件。

## 代码归属

| 路径 | 职责 |
| --- | --- |
| `torchtitan_npu/config/` | 配置 schema、转换器和注册 API |
| `torchtitan_npu/patches/` | 上游或 runtime compatibility patch |
| `torchtitan_npu/extensions/` | 永久扩展、可选 adaptor 和组件集成 |
| `torchtitan_npu/override/` | 由 `override.imports` 显式启用的组件替换 |
| `torchtitan_npu/models/` | 模型配置、实现、并行化和 checkpoint 适配 |
| `torchtitan_npu_recipes/` | 完整 Trainer.Config 及模型专属优化器分组、layout、bucket 配置；可复用实现保留在插件包 |
| `torchtitan_npu/ops/` | NPU 专属算子与后端注册 |
| `torchtitan_npu/tools/` | 随 package 分发的转换和导出工具 |
| `scripts/` | 仓库维护、训练和合规脚本 |
| `tests/` | CPU UT、NPU smoke/ST 与 tooling 自测 |

新增或迁移代码先读取[架构边界](docs/architecture.md)及目标目录的 `AGENTS.md`。当前自动导入顺序为 `patches → config → extensions → ops`；`patches → extensions` 是后续收敛目标。extensions 可以依赖 patches；patches 不得新增对 extensions 的依赖。

TorchFT engine 的按需激活、唯一 owner 和顺序见[架构说明](docs/architecture.md#torchft-engine)，不得新增第二个直接绑定入口。

## 上游与验证

`requirements.txt` 声明 torchtitan 版本，`.ci/setup_torchtitan.sh` 从该版本推导对应 tag。更新版本时同步核对 patch 目标、接口和测试。

完整 lint、测试分层和数值验证要求见[完整开发指南](.agents/AGENTS.md)及[测试规则](tests/AGENTS.md)。创建或更新 GitCode PR 使用 `.agents/skills/submit-pr/`；如果同时安装 `gitcode-pr`，以 `.agents/skills/submit-pr/` 的模板和流程为准。`gitcode-pipeline` 是已安装时的可选流水线 workflow。
