# Python 训练配置

训练配置由 Python recipe 返回完整的 `Trainer.Config`，启动脚本只负责选择配置和分配进程资源。

## 配置位置

- `torchtitan_npu_recipes.models`：经过验证的完整模型配置。
- `torchtitan_npu_recipes.graph_trainer`：经过验证的完整 GraphTrainer 配置。
- `torchtitan_npu_recipes.tests`：调试、裁减模型和 CI 配置。

模型、数据、优化器、scheduler、checkpoint、并行策略和 `override` 都写在 recipe 中。模型实现和数据处理保留在 `torchtitan_npu`，recipe 只负责组合。资源路径也直接在对应 recipe 中设置，不通过环境变量隐式选择训练策略。

修改训练步数、资源路径、并行度或上下文长度时，直接修改对应 recipe 函数，并同步调整相关模型、数据和 token budget 字段。需要新的优化器、算子或 checkpoint 组合时，新增具名 recipe。

## 启动配置

从仓库根目录使用通用 launcher，`MODULE` 指向 recipe 模块，`CONFIG` 指向配置函数：

```bash
MODULE=torchtitan_npu_recipes.tests.models.deepseek_v4 \
CONFIG=deepseek_v4_flash_1p_2k NGPU=1 \
bash scripts/run_train.sh
```

只查看最终配置时追加 `--print-config`。单机进程数使用 `NGPU`；多机使用 `scripts/run_train_multinodes.sh`，并设置 `NNODES`、`NODE_RANK` 和 `MASTER_ADDR`。CANN 环境通过 `ASCEND_SET_ENV_PATH` 指定。

根目录 `examples/` 已删除，旧示例中的模型、数据和优化器选择已迁移到 recipe。已有配置可直接替换上面命令中的 `MODULE`、`CONFIG`；训练参数和资源路径应先在对应 Python 文件中修改。
