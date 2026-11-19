# NPU 配置扩展指南

本指南面向需要为 NPU 训练新增配置的开发者。普通 recipe 可以继续返回
`Trainer.Config`；NPU 训练入口会提供 `TrainerEx.Config` 中定义的扩展字段。

配置字段描述类型、默认值和组件归属，由 Python recipe 赋值。新增字段后，还需要在对应 Trainer 或组件中读取并应用。启动方式见 [Python 配置指南](python_recipes.md)。

## 选择配置位置

根据配置的职责选择位置：

| 场景 | 配置位置 | Python recipe 示例 |
| --- | --- | --- |
| 属于已有 NPU 配置组 | 对应的 extension dataclass | `config.training.extension.allow_hf32 = False` |
| 属于 torchtitan 已有组件 | 继承该组件的上游配置类型 | `DistMuon.Config(enable_hsdp_replica_dedup=True, ...)` |
| 没有合适的上游组件 | `ExtensionConfig` 下的独立分组 | `config.extension.<group>.<field>` |

当前已有配置可作为参考：

```text
TrainingExtensionConfig.allow_hf32
DistMuon.Config.enable_hsdp_replica_dedup
QuantizationExtensionConfig
```

通用配置类型定义在 `torchtitan_npu/config/configs.py`。DistMuon 的 NPU 扩展配置定义在 `torchtitan_npu/extensions/torchtitan/components/optimizer/muon.py`。优化器及其超参数直接配置在 recipe 的 `config.optim.optimizer.optimizers` 列表中。

## 在已有 NPU 配置组中增加字段

直接在对应 dataclass 中增加字段。例如，为 training 增加开关：

```python
@dataclass(kw_only=True, slots=True)
class TrainingExtensionConfig:
    allow_hf32: bool = True
    enable_feature: bool = False
```

在 recipe 中设置 `config.training.extension.enable_feature = True`。`TrainingConfig` 已经接入
`TrainerEx.Config`，因此不需要修改配置类型的注册关系，只需在拥有该行为的组件中读取
`config.training.extension.enable_feature`。

## 为其他 torchtitan 组件增加配置

如果字段属于 checkpoint、optimizer 等已有组件：

1. 在 `torchtitan_npu/config/configs.py` 中定义 NPU 配置类型，并继承对应的上游类型。
2. 在 `TrainerEx.Config` 中将该字段声明为 NPU 配置类型。
3. 在 `torchtitan_npu/extensions/torchtitan/trainer.py` 的 `component_types` 中加入字段名和配置类型。
4. 在对应 Trainer 或组件中读取并应用新增字段。

字段需要独立的 NPU 命名空间时，优先增加 `extension` 子配置；字段本身属于组件的公开
选择时，可以直接放在组件配置中。例如 `TrainingConfig.extension` 保存 NPU 训练开关，
`DistMuon.Config.enable_hsdp_replica_dedup` 保存优化器自身的去重选项。

DeepSeek-V4 的 Muon 配置示例见
[DeepSeek-V4 Muon 优化器](../feature_guides/muon_optimizer.md)。

## 增加根级扩展配置

没有合适上游组件的功能，在 `ExtensionConfig` 下增加独立分组。例如：

```python
from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class RuntimeExtensionConfig:
    enable_feature: bool = False


@dataclass(kw_only=True, slots=True)
class ExtensionConfig:
    quantization: QuantizationExtensionConfig = field(
        default_factory=QuantizationExtensionConfig,
    )
    runtime: RuntimeExtensionConfig = field(
        default_factory=RuntimeExtensionConfig,
    )
```

在 recipe 中设置 `config.extension.runtime.enable_feature = True`。根级扩展不需要加入
`component_types`，但仍需在对应实现中读取 `config.extension.runtime.enable_feature`。
