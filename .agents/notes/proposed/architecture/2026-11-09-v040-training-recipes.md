# TorchTitan 0.4.0 训练 recipe 归属迁移

基线为 PR [#1211](https://gitcode.com/cann/torchtitan-npu/pull/1211) 的 `hitwdy:upgrade_0.4.0_upstream`。当前目标提交 `1da1222` 使用 PyPI `torchtitan==0.4.0`；对应发布源码为 `45e33cd413b64cb37a7840cfd3c9765d3c639ae1`。

## 本轮范围

完整训练工厂从模型 `config_registry.py` 移至 `torchtitan_npu_recipes/models/`，GraphTrainer 组合移至 `torchtitan_npu_recipes/graph_trainer/`。模型 flavor registry 保留在模型包。调用方、包分发和文档同步切换到新归属。

Muon profile、Engram 优化器 schema、V4.1 trainer/engine、Qwen 数据转换和 GraphTrainer schema 转换保留在模型实现中，插件不得反向导入 recipe。V4.1 工厂补充其已有 `OptimConfig(...)` 调用所需的显式导入。

随后将 `examples/` 下全部 34 个训练脚本的组合移入 `torchtitan_npu_recipes/examples/`，每个脚本选择一个同名工厂：V4 19 个、V4.1 12 个、Qwen3.5 2 个、Qwen3.6 1 个。数据、训练预算、模型与 RoPE、Muon 参数组、并行、checkpoint、profiling、量化和 golden/override 组合均使用真实 typed schema。共用 launcher 只处理资源、进程启动与原生参数透传，支持不启动 worker 的 `--print-config`。

迁移保留原场景的训练选择，不移植旧通用 CLI 的字符串解析，也不引入 callable override adapter。旧文本参数环境变量会明确拒绝；资源路径和各工厂明确支持的环境变量继续有效。GraphTrainer、CP、多模态、SFT、LoRA 以及 Qwen27B 等延期路径保留实际构造或运行时限制，不因配置迁移成为已验收功能。

## 验证边界

新增 CPU 测试通过真实 ConfigLoader 选择 V4 eager 与 V4.1 文本 eager 工厂，检查模型类型、数据对齐和优化器 profile；不使用伪造配置替代产品导入。

示例迁移增加真实 ConfigLoader 的场景语义检查，包括 token budget、Muon materialize、quantization、golden、checkpoint、Qwen 数据与拓扑；启动契约测试覆盖 34 个 wrapper、原生参数分组、节点数、网卡地址、退出码和打印配置。启动替身测试只验证 shell，不表示 NPU 训练通过。

本轮验证：87 个相关检查、36 个 shell 语法检查以及 eager、V4 mini、V4.1 mini 的真实入口 `--print-config` 均通过。格式、架构和合规检查通过；全仓 Pyrefly 的 73 个错误与未修改的 `1da1222` 完全一致。扩大运行的 Qwen adapter 检查另有 9 个失败，在该基线上逐项复现。本轮使用可用的 PTA `c36fa4e` 与 CANN 20261104，未匹配计划声明的精确依赖组合，也未执行 NPU 训练或数值对齐。

训练工厂和模型 helper 的源码搬移须与上述基线比较，静态一致不能替代运行时和数值验证。合入前仍需在匹配 0.4.0 pin 的 torch/torch-npu 环境执行相关 UT、NPU 集成和同 checkpoint 的逐 step loss/grad_norm 比特一致比较。环境受阻或未执行的检查记录在 PR 中，不标记为通过。
