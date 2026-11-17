# 静默数据损坏检测（SDC）

参考 [软件安装](./installation.md) 准备环境后，进入 `torchtitan-npu` 仓库根目录。本文介绍如何在其他训练框架中接入 SDC。

## 接入其他训练框架

训练框架可以导入
`torchtitan_npu.extensions.components.sdc.SDC`，按相同生命周期调用 SDC 接口。
下面以一个最小 PyTorch 训练循环说明各接口的调用位置：

- `SDC.Config`：配置 gradient、checksum 和 HCCL 检测。
- `SDC.Config.build(...)`：模型及编译包装完成后，传入模型和梯度累积步数，一次完成全部初始化。
- `finalize_sdc_step()`：训练步骤成功完成后调用。

```python
import torch
import torch.nn as nn
from torchtitan.trainer import Trainer

from torchtitan_npu.extensions.components.sdc import SDC

# 1. 配置 compiled SDC 的训练约束。
# trainer_config 用于校验当前训练配置满足 compiled SDC 的约束。
trainer_config = Trainer.Config()
trainer_config.compile.enable = True
trainer_config.compile.components = ["model"]
trainer_config.compile.backend = "inductor"
trainer_config.parallelism.pipeline_parallel_degree = 1

sdc_config = SDC.Config(
    gradient_enabled=True,
    # nn.Linear 可能被编译为当前 checksum 尚未覆盖的 aten.addmm，
    # 因此该基础示例只演示 gradient detection 的接入。
    with_checksum=False,
    hccl_mode=0,  # 单卡关闭；多卡启用检测时设置为 1、2 或 3。
)

# 多卡训练先初始化进程组，再构建和包装模型：
# torch.distributed.init_process_group(backend="hccl")

# 2. 创建模型，并通过 torch.compile 启用模型编译。
model = nn.Linear(10, 1, device="npu", dtype=torch.bfloat16)
model = torch.compile(model, backend="inductor")

# 3. 在首次前向计算前一次完成 SDC 初始化。
# 本例每次 backward 都产生一个完整梯度，因此梯度累积步数为 1。
sdc = sdc_config.build(
    trainer_config=trainer_config,
    model_parts=[model],
    gradient_accumulation_steps=1,
)

# 4. 创建损失函数和优化器。
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. 训练循环。
for step in range(10):
    x = torch.randn(32, 10, device="npu", dtype=torch.bfloat16)
    y = torch.randn(32, 1, device="npu", dtype=torch.bfloat16)

    pred = model(x)
    loss = criterion(pred.float(), y.float())

    optimizer.zero_grad()
    loss.backward()

    # 一次完整 backward 成功后、梯度清零前，提交本轮梯度检测状态。
    # 如果使用梯度累积，应在每个 microbatch 的 backward 后调用；SDC 会在
    # gradient_accumulation_steps 指定的累积边界执行检测。
    sdc.finalize_sdc_step()

    optimizer.step()
    print(f"step {step}, loss: {loss.item():.4f}")
```

其他框架无需继承 `TrainerEx`，只需在模型包装后构建 SDC，并在成功的 backward 后调用
`finalize_sdc_step()`。HCCL 检测从 SDC 构造完成后生效，不覆盖此前的初始化通信。

启用 typed SDC 时，原生变量可以不设置，也可以在首次导入 torch-npu 前设为关闭：
`NPU_ASD_CONFIG=enable:false`、`NPU_ASD_ENABLE=0`。原生开关启用或取值非法时会报错。
不要在已经启用 Eager 包装后再修改环境变量尝试切换模式。
SDC 内部在 torch-npu 导入完成后按 CLI 设置 HCCL 开关（可覆盖关闭值 `0`），
且必须早于 native 检测路径首次缓存该开关。

更多配置和检测范围见 [SDC 特性说明](../feature_guides/sdc.md)。
