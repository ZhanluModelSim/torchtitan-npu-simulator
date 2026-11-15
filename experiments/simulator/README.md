# TorchTitan simulator

独立训练负载发生器：在 Meta 上复用原生模型、并行化、loss 和 optimizer，返回算子图、通信与控制依赖，以及静态显存预算。当前接入 DeepSeek-V4、Kimi K3 与 Qwen3-32B（dense）；性能预测、硬件配置及 DES 归下游平台。本包仅在显式调用时激活，不修改上游安装源码，生产主包不依赖 simulator。

## 从哪里开始

| 目标 | 阅读入口 |
| --- | --- |
| 理解整体执行链、职责和设计取舍 | [整体方案](docs/architecture.md) |
| **从零接入新模型、复用组件并完成验收** | **[新模型接入流程](docs/model-integration.md)** → [常见问题](docs/model-integration-pitfalls.md) |
| 开发、环境核对与测试选择 | [开发导航](docs/developer-guide.md) |
| Python／CLI 调用、只读配置目录与返回对象 | [同进程接入](docs/upstream-integration.md) |
| 判断当前配置是否在支持范围 | [能力矩阵](docs/capability-matrix.md) |
| Qwen3-32B dense 的路线决策、验证边界与未覆盖范围 | [Qwen3 接入说明](docs/qwen3-integration.md) |

新模型接入不能只修改 preset 或模型名白名单。应先核对配置树、runner 接口、并行和 loss 契约，再逐阶段验证；不要从历史测试数量推断新模型已支持。

## 安装与最小运行

先准备配套的 Python >=3.12、PyTorch、torch-npu、CANN、torchtitan 和本仓主包；依赖基线见仓库根 `requirements.txt` 与 `.ci/setup_torchtitan.sh`。独立 sidecar 的依赖声明不能代替完整训练软件栈。加载实际环境的 CANN 设置文件，并在导入 torch 前关闭设备自动加载：

```bash
# 工作目录：experiments/simulator；CANN_ENV_FILE 指向实际存在的文件。
source "$CANN_ENV_FILE"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m pip install --no-deps --no-build-isolation -e .
python -m torchtitan_simulator --config deepseek_v4_smoketest
```

默认执行完整负载捕获但不写文件；CLI 用户若需查看显存文件，显式启用：

```bash
python -m torchtitan_simulator --config kimi_k3_smoketest \
  --training.seq-len 16 --simulation.output-formats mem \
  --simulation.output-dir ./kimi-memory
```

推荐 shell 入口为 `python -m torchtitan_simulator`。`--build-only` 仅构建模型，不能替代 F/B/O、梯度、通信或内存验收。Python 调用示例见[同进程接入](docs/upstream-integration.md)。

运行时会在安装临时 patch 前检查所用 FSDP 私有调用边界；独立诊断及软件栈记录方法见[开发导航](docs/developer-guide.md#先核对环境)。该检查不替代真实负载捕获验证。

## 主要边界

- 单进程承载一个 logical rank，使用逻辑 world；PP 必须为 1，不用 torchrun worker 数替代逻辑设备数。
- 默认同模型／optimizer 执行一次实际 GA 的无捕获结构预热，再捕获一个 microbatch 的 F/B 与一次 O。平台负责按 GA 组织阶段，见[稳态捕获](docs/steady-state-capture.md)。
- DSV4 可用 `--optimizer.name Muon` 选择原生 DistMuon＋AdamW；保留内部计算、all-to-all 和预取依赖；支持常驻缓冲统计和整段聚合状态卸载，范围与限制见 [Muon 捕获](docs/muon.md)。
- Qwen3-32B（dense）attention 走真实 `npu_fusion_attention_v3` 融合算子并经 Meta 桥接 autograd；并行（FSDP/HSDP/DDP/TP×AC）、双卸载与 CPU 数值参考已验证，量化与真机数值边界见 [Qwen3 接入说明](docs/qwen3-integration.md)。
- 返回图保持 FSDP 通信视图，完整源图在 `captured_workload`；显存依据完整原始事件，不依据裁剪图。数据、alias/mutation、stream/event 依赖均保留，零耗时等待也不能删除。
- 默认通过 Python 对象交付；仅显式 `mem` 导出显存汇总和诊断文件。无图折叠或强制图文件中转。
- 激活和优化器卸载是独立开关，训练仓生成传输节点／依赖并修正静态显存，平台计算传输时间；不执行真实主机 copy。
- 完整捕获当前要求 FP32 原始参数，可用 BF16 mixed precision；MXFP8 另需配套 `torchao_npu` 并显式启用 `all_mxfp8`。参数存储 dtype 投影不改变实际计算精度。
- Meta 捕获与逻辑活跃 storage 预算不证明真实设备数值、HCCL 耗时或 allocator 峰值；当前限制和测试入口以能力矩阵为准。

## 专题

[捕获契约](docs/capture-runtime.md) · [张量角色与轴](docs/custom-tensor-semantics.md) · [流控制](docs/stream-control-dependencies.md) · [静态显存](docs/steady-state-memory.md) · [激活卸载](docs/activation-offload-design.md) · [优化器卸载](docs/optimizer-offload.md)

开发验证按 [开发导航](docs/developer-guide.md) 选择组件、真实模型和交付检查。历史迁移、自审及冻结运行数字由 Git 历史保留，当前文档不再以 S0–S5 阶段组织，也不把旧环境日志当作当前验收证明。
