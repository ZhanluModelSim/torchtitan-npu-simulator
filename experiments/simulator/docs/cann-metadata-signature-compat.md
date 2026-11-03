# CANN metadata Python 包装签名兼容

某些 CANN 包将 metadata Python 函数包装为 `(*args, **kwargs)`，没有暴露具体参数名。旧 simulator 将 `inspect.signature(...).bind(...)` 得到的 `args`、`kwargs` 当作真实算子字段校验，因此在输入准备阶段报错：

```text
RuntimeError: Metadata Python/dispatcher contract differs: lightning_indexer_metadata: {'kwargs', 'args'}
```

这是 Python 包装签名的兼容问题，不是模型训练 OOM 或 C++ 编译失败。用户日志的 CANN 路径为 `cann-9.2.0`，与原始验收的 `cann-9.2.0-beta.2` 包结构不同；仅凭目录名不能确认完整构建号。

## 修正行为

[compat/cann_metadata.py](../torchtitan_simulator/compat/cann_metadata.py) 保留具体 Python 签名的原有路径；签名无法获取或带可变参数时，使用当前 `operator._schema.arguments` 构造绑定签名：

- 使用真实字段顺序、keyword-only 边界、必填字段和默认值，不硬编码四个算子的参数列表。
- 未知关键字、重复传参、缺少必填参数、向 keyword-only 字段传位置参数仍报错，不静默丢弃参数。
- CPU planner tensor 仍转换为 Meta 输入，再调用普通 `operator(**values)`；原生 dispatch、厂商 Meta 和 scope 恢复机制保持。
- 不执行被包装的真实 NPU metadata kernel，不手工推导 shape，不创建替代算子节点。

覆盖四个现有入口：`lightning_indexer_metadata`、`sparse_flash_mla_metadata`、`sparse_flash_mla_grad_metadata`、`sparse_lightning_indexer_kl_loss_grad_metadata`。没有新增 CLI 参数，既有 smoke 命令不变。

## 验证与边界

[tests/test_metadata.py](../tests/test_metadata.py) 覆盖四算子的位置／关键字调用、opaque wrapper、严格参数校验、原生节点与消费者依赖，以及退出后 binding 恢复。环境更新后应重新运行组件探针和真实小模型；包装签名形式相同不证明整套 CANN 软件栈兼容。

## 环境排查经验

- CANN Python 包导入可能触发 `torch.utils.cpp_extension.load()`，出现 C++ 编译与链接日志；这不等于训练启用了 torch.compile。区分冷启动编译、后续缓存与实际捕获耗时。
- NumPy/PyArrow 等二进制依赖需按实际训练栈配套，先检查 import traceback，不依据旧环境版本号盲目降级，也不通过移除依赖校验解决运行问题。
- autograd warning 可能来自 bridge 安装前的能力探针，但必须检查完整运行的梯度清单、保存值与实际 backward 节点；不能仅凭 warning 判断失败或成功。
- Meta 仿真不要求真实设备可用；npu-smi 失败不单独证明 Meta 失败，Meta 成功也不证明真实 NPU/HCCL 可用。
