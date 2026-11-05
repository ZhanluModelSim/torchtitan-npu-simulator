# Simulator 开发导航

本指南是无上下文开发者的阅读与验证入口。先阅读[整体方案](architecture.md)，接入模型按[新模型流程](model-integration.md)逐步执行；修改捕获基础设施则按[捕获契约](capture-runtime.md)核对已有模型。源码、配置和测试是当前行为的依据。

## 文档分工

| 文档 | 回答的问题 |
| --- | --- |
| [整体方案](architecture.md) | 为什么复用原生训练、如何捕获、图与内存分别怎样交付 |
| [新模型流程](model-integration.md)／[常见问题](model-integration-pitfalls.md) | 从模型先验到可验收负载，如何复用、修改、注册和排错 |
| [捕获契约](capture-runtime.md) | 身份、依赖、保存槽、作用域恢复与公共捕获不变量 |
| [同进程接入](upstream-integration.md) | 已支持模型如何调用，参数与返回对象在哪里读取 |
| [能力矩阵](capability-matrix.md)／[未覆盖范围](open-questions.md) | 哪些能力有实际路径与测试，哪些不能宣称支持 |
| [稳态捕获](steady-state-capture.md)／[静态显存](steady-state-memory.md) | 结构预热、GA、优化器状态及逐层 payload 的口径 |
| [语义声明](custom-tensor-semantics.md)／[AdamW 身份](adamw-input-identity.md) | 给下游提供真实角色和轴，不把 shape 猜测当事实 |
| [Muon 捕获](muon.md) | DSV4 混合优化器、Meta 适配、常驻缓冲、聚合卸载与 all-to-all 依赖 |
| [流控制](stream-control-dependencies.md)／[EP 交换](ep-single-transport.md) | 通信与等待如何成为真实依赖 |
| [激活卸载](activation-offload-design.md)／[优化器卸载](optimizer-offload.md) | 传输执行图、静态预算和已知建模近似 |
| [CANN 接口兼容](cann-metadata-signature-compat.md)／[生命周期机制](storage-lifetime-optimization.md)／[chunked loss](chunked-loss-policy.md) | 对应实现为何存在，哪些语义必须保留 |

## 先核对环境

在实际运行的 Python 环境执行；本机是否有源码不能证明容器正在使用它：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python - <<'PYCODE'
import sys
import torch, torchtitan, torchtitan_npu, torchtitan_simulator
print(sys.executable, torch.__version__)
for module in (torchtitan, torchtitan_npu, torchtitan_simulator):
    print(module.__name__, module.__file__)
PYCODE
```

此前须加载实际 CANN 环境。容器工作目录、挂载和 editable 安装路径逐次核对，不复制历史容器名或另一 checkout 的路径。可用 `docker inspect <container>` 检查挂载。主包与 sidecar 都必须指向本次修改；不修改安装源码或用 `sys.path` 拼接替代安装。

## 检查顺序

1. 用最小行为探针区分配置、原生接口、Meta/autograd、图依赖或内存归属问题；保留原始异常。
2. 修改所属层及全部相关调用点，补能够区分错误实现的独立预期；不让新模型专用判断渗入公共捕获。
3. 先验证组件 F/B，再跑真实小模型及相关并行／AC 路径；具体门槛和命令见新模型流程。
4. 核对连续两次会话及异常退出后的清理，保证全局 patch、process group、TLS 和 cache 不污染下一任务。
5. 代码变更完成后检查类型、lint、普通 wheel 安装以及实际消费者；只修改文档时执行链接、命令和必要轻量示例检查，不重复昂贵全规模验收。

### 工具入口

从 `experiments/simulator` 执行，使用已安装 `dev` extra 的完整训练环境：

```bash
# 选择测试子集，或省略 -k 运行整个 sidecar。
TORCHTITAN_SOURCE=/actual/torchtitan-source \
TORCHTITAN_NPU_SOURCE=/actual/torchtitan-npu-source \
bash scripts/validate.sh -k 'custom_tensor_semantics or tensor_semantics'
```

`validate.sh` 先执行 pytest，再对整个 sidecar 执行 pyrefly。源码根必须替换为实际路径；如果上游是普通 wheel，类型检查可能需要其 site-packages 搜索根。`TORCHAO_NPU_SOURCE` 可补充量化包路径。根项目类型检查排除本包，不能拿根目录结果代替 sidecar 检查。

完整 lint 仍遵循仓库根开发指南：核对 `/tmp/torchtitan` 后，从仓库根执行 `python -m pre_commit run --all-files --show-diff-on-failure`。局部 Ruff 通过不能写成全仓 lint 通过。

普通 wheel 安装须离开源码目录确认导入来自 site-packages，并验证关键 API；已有 editable 测试不代替交付测试。模型接入结果必须区分源码核对、CPU 数学、真实 Meta 捕获与真实 NPU 数值／性能验证，不能相互替代。

## 文档维护

当前方案、模型 workflow 和专题各有单一职责；README 与 AGENTS.md 只导航和规定边界。历史审计数字、迁移阶段及中间自审不再单独维护，必要时从 Git 历史追溯。新增能力同步所属专题、能力矩阵、公开入口和相关测试，不再追加另一份重复的「本轮完成」文档。
