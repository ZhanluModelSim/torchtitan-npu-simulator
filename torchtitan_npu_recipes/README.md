# TorchTitan-NPU Python recipes

本包保存返回完整 `Trainer.Config` 的训练组合。模型、Muon profile、数据转换函数和 trainer/engine 实现保留在 `torchtitan_npu`；recipe 选择并组合这些实现。

| 入口 | 配置归属 |
| --- | --- |
| `torchtitan_npu_recipes.models.deepseek_v3_2` | V3.2 训练配置 |
| `torchtitan_npu_recipes.models.deepseek_v4` | V4 eager、SFT、LoRA 和 TorchFT 配置工厂 |
| `torchtitan_npu_recipes.models.deepseek_v4_1` | V4.1 文本、多模态和 LoRA 配置工厂 |
| `torchtitan_npu_recipes.models.qwen3_5` | Qwen 配置工厂 |
| `torchtitan_npu_recipes.graph_trainer.deepseek_v4` | V4 GraphTrainer 配置工厂 |
| `torchtitan_npu_recipes.examples.deepseek_v4` | 19 个 V4 launcher 的完整配置 |
| `torchtitan_npu_recipes.examples.deepseek_v4_1` | 12 个 V4.1 launcher 的完整配置 |
| `torchtitan_npu_recipes.examples.qwen3_5` | 2 个 Qwen3.5 launcher 的完整配置 |
| `torchtitan_npu_recipes.examples.qwen3_6` | Qwen3.6 launcher 的配置入口（基础工厂仍延期） |

这些是配置的归属，不是功能支持声明。TorchTitan 0.4.0 的当前 NPU 适配优先覆盖 DeepSeek 文本 eager；TP、CP、PP、多模态、SFT、LoRA、图训练与其他延期路径仍受各实现的限制。移动配置不会启用这些功能。

DeepSeek-V4 和 V4.1 示例默认使用非 golden 算子组合，只有显式设置 `USE_GOLDEN=1` 才选择参考组合。基础模型工厂保留各自的配置，示例工厂负责选择实际运行的算子。A3 环境使用 CANN 1104 和 torch-npu 1105.3，下载与安装见[安装指南](../docs/user-guides/installation.md)。

`examples/` 下的训练脚本选择对应工厂，shell 只保留设备、节点、CANN 等启动资源和一条共用 launcher 调用。A3 8P Flash CPT debug 脚本默认选择 `deepseek_v4_flash_8p_cpt_4k_a3_eager`，关闭模型编译。数据、训练预算、优化器、并行、checkpoint 和算子选择均在 Python 中定义。例如查看 mini 示例的完整配置：

```bash
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh --print-config
```

依赖、数据和 HF 资源必须按安装指南准备。训练入口见 [快速上手](../docs/user-guides/quickstart.md)，自定义组合见 [Python 配置指南](../docs/developer_guides/python_recipes.md)，验证入口见 [integration 指南](../tests/integration_tests/README.md)。通用 `--section.option` 参数不属于 TorchTitan 0.4.0 的配置接口；改变训练策略应定义新的 Python 工厂。

`torchtitan_npu/models/*/config_registry.py` 不再提供训练工厂。模型实现继续通过 `torchtitan_npu.models.<model>.model_registry` 构造；训练调用方导入本包。组件替换仍使用现有 `torchtitan_npu.override` 和上游 override pass。
