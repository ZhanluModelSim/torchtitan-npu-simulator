# Python 训练配置

完整训练工厂位于 [`torchtitan_npu_recipes`](../../torchtitan_npu_recipes/README.md)。工厂返回完整训练配置；shell launcher 只选择工厂、设备进程数及启动资源。专用训练策略、golden、优化器与算子组合应定义新 Python 工厂。

`examples/` 下的训练脚本在 `torchtitan_npu_recipes.examples.<model>` 中选择对应工厂。例如 `deepseek_v4_mini_1p_cpt_2k_a3.sh` 选择 `torchtitan_npu_recipes.examples.deepseek_v4.deepseek_v4_mini_1p_cpt_2k_a3`；A3 8P Flash CPT debug 脚本默认选择带 `_eager` 后缀的工厂。工厂包含数据、模型、token budget、优化器、并行、checkpoint、profiling 和算子选择；基础模型工厂仍在 `torchtitan_npu_recipes.models` 中供组合复用。

DeepSeek-V4 和 V4.1 示例默认使用非 golden 算子组合，显式设置 `USE_GOLDEN=1` 才选择参考组合。该开关由示例工厂读取，不改变基础模型工厂的默认配置。

## 文本 eager 兼容性设置

DeepSeek-V4 和 V4.1 文本 eager 基础工厂在 `config.override.imports` 中显式选择 `torchtitan_npu.override.common.rope.workaround`。该现有实现先将复数缓存展开为实数 cos/sin 缓存，再按位置索引，避开 NPU 不支持的 complex64 索引。基础配置同时使用 `compile=None` 和 `model.local_compile_regions=[]`；仅设置前者不能关闭模型组件的局部编译。

这些兼容性设置不构成数值对齐证据；与原实现的数值比较尚未完成。融合或量化示例通过各自工厂选择算子组合，不能将文本 eager 的结果推广到这些配置。

## 组合已有配置

例如在仓库根目录创建 `my_recipes.py`，定义短步数 eager 配置：

```python
from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_debugmodel


def debug_five_steps():
    config = deepseek_v4_debugmodel()
    config.training.steps = 5
    config.hf_assets_path = "tests/assets/deepseek_v3"
    return config
```

使用维护的单卡 launcher 选择该配置：

```bash
MODULE=my_recipes CONFIG=debug_five_steps \
  bash examples/deepseek_v4/debug/deepseek_v4_eager_1p.sh
```

HF 资源目录由 Python 配置的 `hf_assets_path` 字段指定；自定义 tokenizer 或模型资源时修改工厂中的路径，指向包含 tokenizer、`config.json` 等 HF 资源的目录。共用 launcher 会进入仓库根目录，相对路径从该目录解析。示例工厂也保留各自的 `HF_ASSETS_PATH`、数据路径等资源环境变量，具体默认值见对应 Python 工厂。

上游 ConfigLoader 解析 `--module`、`--config`、`--override`、`--comm-backend`、`--output-dir`、`--resume-step` 和 `--print-config`。`--hf-assets-path` 及通用 `--training.*`、`--parallelism.*`、`--optimizer.*` 和 `--override.imports` 字段参数不属于 0.4.0 接口；将这些选择写入 Python 配置。现有 override 仍由配置的 `override.imports` 或 `--override` 选择，本轮不改变其注册机制。

修改序列长度时同时调整模型、RoPE、数据与 token budget 的相应字段。并行与断点续训组合需要定义完整的 Python 工厂，并通过[集成测试入口](../../tests/integration_tests/README.md)验证。

## 查看配置与启动资源

所有示例将原生参数原样传递给共用 launcher。`--print-config` 原样交给上游 ConfigLoader，worker 打印配置后退出，不执行训练：

```bash
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh --print-config
```

`NGPU`、`NNODES`、`NODE_IPS`、`NODE_RANK`、`MASTER_ADDR` 和 CANN 环境等进程启动资源留在 shell。多机示例在提供 `NODE_IPS` 时按地址数推导节点数；节点数以显式 `NNODES` 为准。未提供地址列表时需在每台机器设置 `MASTER_ADDR` 和 `NODE_RANK`。单机脚本 `scripts/run_train.sh` 只启动本机进程；多机示例显式调用 `scripts/run_train_multinodes.sh`，并在各节点执行。`WORLD_SIZE` 由 torchrun 设置，shell 不单独解析 `--print-config`；多机配置打印也经过正常的多机启动流程。

旧 `CLI_OVERRIDES`、`OPTIMIZER_OVERRIDES` 等文本字段组合已移入 Python；传入这些旧入口会明确报错。自定义训练行为应修改或组合工厂，使用 `MODULE` / `CONFIG` 选择。

## 归属与支持范围

模型实现、Muon profile、数据转换和 trainer/engine 保留在 `torchtitan_npu/models/`，recipe 通过普通 Python import 组合这些实现。实现层不依赖 recipe。旧模型 `config_registry.py` 只保留 flavor registry，不再提供训练工厂。

图训练基础工厂位于独立的 `torchtitan_npu_recipes.graph_trainer.deepseek_v4` 模块；示例组合由对应的 `examples` 工厂选择。配置迁移不代表新能力已经可用；当前 0.4.0 适配优先覆盖 DeepSeek 文本 eager，图训练、CP、多模态、SFT、LoRA 等路径仍保留各自的实现限制，并在构造或运行时明确失败。打印配置成功也不能代替 NPU 训练和数值回归。
