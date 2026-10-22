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

[tests/test_metadata.py](../tests/test_metadata.py) 的新增 opaque-wrapper 用例在修正前有 12 项失败；修正后四算子的位置／关键字调用、严格参数校验、原生节点与消费者依赖、Python binding 恢复均通过。

真实 `simulate()` 同进程运行正常接口和四个 opaque wrapper 两条路径；两者的 112 个参数梯度清单、保存槽数量、逻辑内存峰值和常驻参数字节一致，图无环且峰值 storage 唯一。此处比较真实模型执行结果，不以 placeholder 图替代。

| 检查 | 实际结果 |
| --- | --- |
| Metadata 专项 | 21 passed，包含 13 项新增用例 |
| 全量 sidecar 回归 | `python -m pytest tests/ -q -x`：444 passed，193.87 秒 |
| Sidecar 类型检查 | 0 errors，11 suppressed；未新增类型忽略 |
| 独立静态复审 | 无必改问题；审查者未执行测试，运行结果由上述实际命令提供 |
| Lint | 常规、Ruff、格式、架构、OAT 通过；主机 Pyrefly 可执行文件缺失，全仓 pre-commit 不能标为全绿；类型检查使用完整依赖容器 |

验证日志位于本机审计目录 `torchtitan-migration-audit-20261020/dispatch-first-remediation/` 的 `metadata-opaque-before.log`、`metadata-opaque-complete.log`、`metadata-opaque-full-tests.log` 和 `metadata-opaque-types.log`。

当前验证环境仍是冻结的 CANN 9.2.0-beta.2，通过 opaque wrapper 重现用户的接口形式；没有完整复验用户的 CANN 9.2.0 软件栈。原 21 项全规模矩阵及哈希保留为原提交的验收证据，不伪装为本补丁重新执行过 21 项矩阵。内部环境更新分支后需重跑同一 smoke 命令；若有新 traceback，应继续核对实际 schema／Meta 能力，不删除校验来掩盖版本差异。

## 安装与启动的已知事项

- 已验收组合为 NumPy 1.26.4 + PyArrow 25.0.0。初始 PyArrow 26.0.0 在该 NumPy 上导入失败；requirements 未锁定该组合，装完依赖后需核对。可保持 NumPy，执行 `python -m pip install --force-reinstall --no-deps "pyarrow==25.0.0"`，再确认两者可导入。
- CANN Python 包导入可触发 `torch.utils.cpp_extension.load()`，出现 `c++`／`.so` 编译和链接日志；与本补丁及 `torch.compile` 关闭无冲突。首次慢，后续可复用编译缓存；持久化 `TORCH_EXTENSIONS_DIR` 可保留缓存。
- autograd warning 可能来自能力探针在安装 bridge 前执行原生 Meta 梯度检查；不能仅凭 warning 判断训练梯度丢失，也不能仅凭 warning 后仍运行就认定数值正确。以完整运行结果、梯度清单与捕获校验为准。
- Meta 仿真不依赖真实 NPU 可用性；`npu-smi info` 不可用不能作为 Meta smoke 失败的唯一依据，也不能据 Meta 成功证明真实 NPU/HCCL 可用。
