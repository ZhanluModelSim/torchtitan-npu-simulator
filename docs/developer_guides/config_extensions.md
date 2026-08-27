# NPU 配置扩展指南

torchtitan-npu 复用 torchtitan 的配置工厂和 `override` 机制。标准训练配置由
`torchtitan_npu/config/manager.py` 在 Tyro 创建 CLI schema 前包装为
`torchtitan_npu.config.TrainerConfig`，模型 config_registry 保持上游返回类型即可。

## 配置生命周期

```text
upstream config factory
    -> ConfigManager._load_config NPU patch
    -> TrainerConfig.from_trainer_config()
    -> ConfigManager / tyro.cli()
    -> TrainerConfig.build()
    -> TrainerEx
```

这里的「扩展配置」是指 Tyro 解析前加入的 NPU 配置类型，不是运行时逻辑。配置工厂
仍返回上游 `Trainer.Config`，`ConfigManager` 会在 Tyro 解析前将它包装为
`TrainerConfig`。

## 扩展命名空间

只负责提供配置入口，新增字段仍需要由 `TrainerEx` 或对应实现主动读取和应用。

### 已有组件扩展配置

针对 torchtitan 已有的配置节点扩展字段时，在对应配置类中增加 `extension` 子配置：

```text
training.extension.*
checkpoint.extension.*
optimizer.extension.*
```

以 `training.allow_hf32` 为例，参考 `torchtitan_npu/config/configs.py` 中
`TrainingExtensionConfig`、`TrainingConfig` 和 `TrainerConfig` 的写法：

1. 定义 `TrainingExtensionConfig`，放置 NPU 字段。
2. 在 `TrainingConfig` 中通过 `field(default_factory=...)` 增加 `extension` 字段。
3. 在 `TrainerConfig._CONFIG_EXTENSIONS` 中注册 `training` 到 NPU 版
   `TrainingConfig`。`TrainerConfig.from_trainer_config()` 会自动复制上游配置的公共字段并完成转换。

最终 CLI 参数为：

```text
--training.extension.allow-hf32
```

### 新增组件扩展配置

如果 `Trainer.Config` 中没有对应的配置节点，则在
`TrainerConfig` 根部增加 `extension` 分组：

```text
extension.<config-group>.<field>
```

例如增加一个 runtime 配置组：

```python
from dataclasses import dataclass, field

from torchtitan_npu.extension.trainer import TrainerEx


@dataclass(kw_only=True, slots=True)
class RuntimeExtensionConfig:
    enable_feature: bool = False


@dataclass(kw_only=True, slots=True)
class ExtensionConfig:
    runtime: RuntimeExtensionConfig = field(
        default_factory=RuntimeExtensionConfig,
    )


@dataclass(kw_only=True, slots=True)
class TrainerConfig(TrainerEx.Config):
    extension: ExtensionConfig = field(
        default_factory=ExtensionConfig,
    )
```

对应 CLI 参数为 `--extension.runtime.enable-feature`。
需要注意的是 extension 不需要在 `TrainerConfig._CONFIG_EXTENSIONS` 中注册。

## 特殊 Trainer 类型

标准 `Trainer.Config` 会被包装为继承 `TrainerEx.Config` 的 `TrainerConfig`，因此
标准 NPU 训练最终由 `TrainerEx` 构建。`FluxTrainer.Config`、`GraphTrainer.Config`
等带有额外字段的顶层类型不应套用标准 `TrainerConfig`，否则转换时无法保留专有字段；
这类 Trainer 应定义保留全部字段的专用 Config，并通过自己的入口接入。
