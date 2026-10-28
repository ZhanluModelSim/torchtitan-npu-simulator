# 默认 EP single 交换

simulator 在公共运行时中默认使用 PyTorch 原生 functional `all_to_all_single` 承载 MoE TokenDispatcher 的 EP 交换。DSV4、Kimi 及其已接入 preset 共用同一个适配，不修改模型代码、原生工厂或正常 NPU 训练的默认实现。无需新增 CLI／Python 参数。

## 作用范围

[compat/routing.py](../torchtitan_simulator/compat/routing.py) 在模型并行化后发现实际 TokenDispatcher 类型，通过会话的 `PatchScope` 安装三个 Meta helper：

| helper | functional 调用的输出／输入 splits |
| --- | --- |
| token-count exchange | `None / None`，保留原生 `[EP, local_experts]` 布局 |
| token／router-score dispatch | `output_splits / input_splits` |
| expert-output combine | `input_splits / output_splits`，方向与 dispatch 相反 |

实际通信、输出分配、async wait 和 backward 均由原生 PyTorch 提供；反向按原生 autograd 交换逆向 splits。不手写梯度、不补造算子或通信节点、不通过图折叠删除 buffer。

原生 SPMD list 交换会为每个 peer 分配输出 chunk，然后 cat／stack。single 路径直接产生完整输出；因此 EP 较大时，`empty`／`empty_like` 和拼接节点明显减少。节点数减少不代表通信量减少或硬件训练加速。

此适配只在显式 simulator 会话的 Meta tensor 路径生效。非 Meta tensor 调用原方法；会话成功或异常退出均恢复绑定。CP、FSDP 的交换实现与流控制不由这三个 helper 改写。它不是启用整个模型的 compile，也不把全局 `spmd_backend` 改成其他值。

## 均衡 split、依赖与结果消费

Meta 无法从 tensor 内容读取远端 token count，因此保留原有均衡 round-robin 路由假设及 host split 推导。实际 count collective 及 wait 节点仍在图中；后续 token／score dispatch 显式依赖 received-count tensor。缺少同步 count 依赖时报错。

上游应按规范化 `all_to_all` 类型和通信描述消费，不依赖旧 `c10d.alltoall_` raw 名称。EP 交换现在记录为 `_c10d_functional.all_to_all_single.default`；group、ranks、dtype、payload、F/B 归属及实际 tensor 依赖继续保留。

返回图的 `annotations["runtime_capabilities"]` 包含：

- `ep_token_exchange="functional_all_to_all_single"`：simulator 的默认 EP 交换策略。
- `ep_token_dispatcher_classes`：本次安装适配的 dispatcher 类型数量；EP=1 时可为 0。

该默认策略与原生 eager `spmd_types` 的 list 前端存在实现差异，模拟的是 single 交换负载，不应拿这些 buffer／拼接节点数量直接代表 list 实现。通信量按原始 payload 记录；内存按当前真实捕获的 storage 生命周期估算，不套用旧 list 路径的峰值。

## Selective AC 的联动

EP single 进入原生 SAC `comm` 保存集合。显式选择 `--simulation.selective-ac-save-ops comm` 时，实际交换输出被缓存，重算不会再次发起已保存的前向 EP 通信；原生反向交换继续保留。这会改变保存值集合和驻留区间，不能沿用旧 list 路径「comm 未命中」的结果。命中、消费与反向检查见 [SAC 生命周期测试](../tests/test_s3_native_lifetimes.py)。未实际命中的配置继续发出 warning。

## 验证边界

[test_routing.py](../tests/test_routing.py) 覆盖两种模型的 none/full/selective AC × 量化开关、split 方向、count→dispatch→GMM→combine→unpermute 因果路径、非 Meta 调用以及失败后的绑定／process-group 恢复。

[capture/test_native_collectives.py](../tests/capture/test_native_collectives.py) 通过两个真实 Gloo rank，独立检查不等长交换的输出及远端反向梯度，直接使用同一 functional 入口。完整模型的 Meta 验证覆盖梯度元数据、图及逻辑内存，不能证明 NPU 数值、HCCL 性能或 allocator/workspace 峰值。

预先隔离试验的完整 43 层 Flash 配置为 world=1024、rank=1023、DPR=2、DP shard=256、TP=1、CP=2、EP=128、S=512、AC=none、MTP=0。非量化图从 109,470 节点变为 70,641，MXFP8 从 126,670 变为 87,841；301 次 EP collective 的有序通信签名、1,285 个参数梯度元数据及六项内存预算一致。正式默认实现已在同一配置上独立重新验收，未使用强制分支脚本：两种精度的节点数均与上述 single 结果一致，与原 list 路径的有序通信签名、梯度清单、保存槽数量和内存汇总分别一致；完整 MXFP8 图的 43 层、301 对关键因果关系及 129 次反向 EP 交换复核通过。

正式源码的完整 sidecar 回归为 544 项通过，容器内 sidecar Pyrefly 为 0 错误；宿主全仓 pre-commit 的非类型检查及 OAT 通过，宿主 Pyrefly 仍有 608 项既有缺依赖／导入错误，未把它记为通过。原始日志及逐次源码 SHA256 保存在验收目录 `default-single-20261028`。当前源码的后续变更须重新验证；真实 NPU／HCCL 未执行。
