# Python 训练配置

训练配置位于 [`torchtitan_npu_recipes`](../../torchtitan_npu_recipes/README.md)，由 Python 函数返回完整的 `Trainer.Config`。模型、数据、优化器和并行策略在配置函数中定义；启动脚本负责选择配置、设备和节点。

`torchtitan_npu_recipes.models` 提供可复用的基础配置，`torchtitan_npu_recipes.examples` 组合具体训练用例，供 `examples/` 下的脚本调用。

## 编写配置

在仓库根目录创建 `my_recipes.py`，复用基础配置并调整训练步数：

```python
from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_debugmodel


def debug_five_steps():
    config = deepseek_v4_debugmodel(
        seq_len=2048, hf_assets_path="tests/assets/deepseek_v3"
    )
    config.training.steps = 5
    return config
```

`hf_assets_path` 指向 tokenizer 等 Hugging Face 资源所在目录；相对路径从仓库根目录解析。

构造模型前，先确定 `parallelism`、`seq_len` 和资源路径，并传入基础配置函数。模型配置及分片声明确定后，再执行配置变换和 override，由 Trainer 校验并构建模型。构造后修改并行度或序列长度不会自动重建模型配置。

训练参数直接写入 Python 配置，组件替换通过 `config.override.imports` 或 `--override` 指定，详见 [override 说明](../../torchtitan_npu/override/README.md)。

## 启动与查看配置

通过 `MODULE` 指定模块、`CONFIG` 指定配置函数：

```bash
MODULE=my_recipes CONFIG=debug_five_steps \
  bash examples/deepseek_v4/debug/deepseek_v4_eager_1p.sh
```

追加 `--print-config` 可打印配置并退出，不执行训练：

```bash
MODULE=my_recipes CONFIG=debug_five_steps \
  bash examples/deepseek_v4/debug/deepseek_v4_eager_1p.sh --print-config
```

设备、节点和 CANN 环境通过启动脚本配置。单机使用 `scripts/run_train.sh`；多机使用 `scripts/run_train_multinodes.sh`，在各节点设置启动参数并执行。

## 算子与编译设置

DeepSeek-V4 和 V4.1 示例默认使用非 golden 算子组合，设置 `USE_GOLDEN=1` 可选择参考组合。

文本 eager 基础配置使用 `compile=None` 和 `model.local_compile_regions=[]`，并配置 NPU RoPE 兼容实现。自定义 eager 配置需同时关闭整体和局部编译。各训练用例的支持范围见 [recipe 说明](../../torchtitan_npu_recipes/README.md)。
