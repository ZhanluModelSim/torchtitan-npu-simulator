# 快速上手

本文介绍如何在 TorchTitan 0.4.0 中以 eager 模式启动 DeepSeek 文本训练。先按[安装指南](installation.md)准备匹配的 torch、torch-npu、TorchTitan 与 CANN 环境，再从仓库根目录运行启动脚本。

## 数据和模型资源

基础调试配置使用 `tests/assets/c4_test/data.json` 作为训练数据，将 `hf_assets_path` 设为 `tests/assets/deepseek_v3`，使用该目录中的 tokenizer。正式模型需要准备与模型匹配的 Hugging Face 资源；调试用 tokenizer 的训练结果不能代替正式模型的数值验证。

## 单卡 eager

```bash
bash examples/deepseek_v4/debug/deepseek_v4_eager_1p.sh
```

启动脚本调用配置函数 `torchtitan_npu_recipes.examples.deepseek_v4.deepseek_v4_eager_1p`，其中设置了模型、数据、训练步数和算子组合，默认训练 100 步。DeepSeek 示例默认使用 NPU 融合算子，需要参考实现时显式设置 `USE_GOLDEN=1`。在命令末尾加上 `--print-config` 可以只查看配置，不启动训练。

需要缩短试运行步数或更换优化器时，可以编写 Python 配置函数，通过 `MODULE`、`CONFIG` 指定，见[配置组合指南](../developer_guides/python_recipes.md)。

DeepSeek-V4.1 的文本调试配置可以通过以下命令查看：

```bash
MODULE=torchtitan_npu_recipes.models.deepseek_v4_1 \
CONFIG=deepseek_v4_1_debugmodel_text NGPU=1 \
  bash scripts/run_train.sh --print-config
```

去掉 `--print-config` 即可启动训练。该配置默认训练 10000 步；需要减少步数时，在自定义配置函数中设置 `config.training.steps`。

CANN 的 `set_env.sh` 不在默认位置时，显式指定环境脚本路径：

```bash
ASCEND_SET_ENV_PATH=/path/to/ascend-toolkit/set_env.sh \
  bash examples/deepseek_v4/debug/deepseek_v4_eager_1p.sh
```

环境脚本的默认搜索位置见 [`scripts/run_train.sh`](../../scripts/run_train.sh)。启动脚本负责加载环境和启动训练进程，训练参数在 Python 配置中设置。

## 集成和数值验证

集成测试入口及用例定义见[集成测试指南](../../tests/integration_tests/README.md)。训练组合的验证状态以实际执行结果为准；按[数值验证要求](../developer_guides/numerical_alignment.md)比较固定 checkpoint 下未截断的 loss 与 grad_norm。

旧示例中的通用字段 CLI 不再适用于 0.4.0。图训练、CP、PP、多模态、LoRA 和其他延期能力的历史说明不表示当前依赖组合已完成支持；具体边界见[升级计划](../../.agents/notes/proposed/upstream/2026-11-04-upgrade-torchtitan-v0.4.0-eager-plan.md)。
