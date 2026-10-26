# 捕获运行时与负载契约

本文规定 simulator 基础设施的职责、升级步骤及验收方法。适用于环境和入口、`compat/`、通用 capture、IR、内存与对外契约变更；模型专属实现另见 [模型接入](model-integration.md)。

## 职责与修改位置

| 位置 | 职责 | 修改时需要保护的契约 |
| --- | --- | --- |
| `config.py`、`cli.py`、`runner.py` | 原生工厂、参数白名单、真实 F/B/clip/optimizer 与返回结果 | 配置不静默失效；Python/CLI 结果一致；unsupported 提前拒绝 |
| `runtime.py`、`compat/` | 显式 Meta 会话、能力探针、最小可撤销兼容 | 正常／异常恢复、健康原生实现不被覆盖、跨捕获状态隔离 |
| `capture/` | 原生 dispatch、通信、控制事件、保存值与所有权观察 | 真实输入、tensor/storage 身份、alias/mutation、源事件唯一性 |
| `ir/`、schedule/workload builder | 负载结构、模板实例、数据与控制依赖 | 引用闭合、双向边、DAG、通信与 action 对应关系 |
| `memory/` | 逻辑 storage 驻留、FSDP、AC、offload 和状态预算 | 独立字节、实际释放、去重、分阶段和稳态口径 |
| `cost/` | 当前已有的基础成本注记 | unknown 明确保留；不能反向修改捕获事实 |

端到端 DES、批处理、完整指标平台和独立 cost model 属于后续解耦方向。当前 `build_nodes()` 仍调用内置 `OpCostModel`，IR 仍含 Python dataclass／annotations 对象；不能据此声称跨仓协议已经独立完成。
禁止为后续架构提前把释放时间固定到算子结束，也不在本仓重新引入图折叠与推断补边。

## 固定环境并保护源码

迁移／升级前建立可复现基线：

- TorchTitan、NPU 插件、sidecar、TorchAO-NPU 的版本和实际提交，torch/torch_npu、Python、NumPy/PyArrow、CANN 完整版本与镜像 digest。
- 实际导入路径、editable 安装／`PYTHONPATH` 来源、工作树状态、容器内源码 SHA256。镜像 tag、宿主路径和「master」都不足以唯一定位代码。
- 模型几何、输入、并行布局、AC／精度及逻辑 rank；旧 smoketest 与新 native debugmodel 规模不一定相同，旧峰值不能直接作新版本阈值。

使用包含完整 CANN 的官方镜像作为环境基础，再安装配套训练依赖及 sidecar；不要在干净镜像中通过散装执行包拼出另一套依赖。
[Dockerfile](../Dockerfile) 只安装 sidecar，`BASE_TRAINING_IMAGE` 必须已经配套。开发容器同步源码、构建新镜像与独立安装包验收分别记录，不把三者当作同一步。

不修改上游 checkout 或安装目录来让 smoke 通过；使用独立工作树／副本，保留用户已有变更。核对主包和 sidecar 都加载当前源码，不能仅更新 sidecar 却仍导入旧模型包。

安装阶段曾遇到 NumPy/PyArrow 不配套、CANN 首次导入编译 C++ 扩展、`npu-smi` 不可用和 opaque metadata 包装签名差异。
具体冻结组合与排查方式见 [环境与签名事项](cann-metadata-signature-compat.md#安装与启动的已知事项)。这些是具体环境现象，不是可永久复制的版本判断规则。

## 显式会话与入口

入口为 `simulate(SimulationConfig(...))` 或 `cli.main(argv)`，同进程返回 `WorkloadGraph`；不恢复旧 `SimulationTrainer.train` patch。
独立 CLI 是白名单，不接受任意原生参数树；新增参数要检查 Python 字段、CLI、工厂、实际 consumer 和 [接入文档](upstream-integration.md)。

- 在导入 torch／simulator 前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，按环境加载 CANN；Python API 不替调用者 source shell 脚本。
- 捕获会话拥有 fake process group；宿主已有 process group、嵌套或并发仿真不作为支持路径。当前保留同进程串行调用，PP 捕获入口仍拒绝。
- capture worker、logical rank、logical world、PP stage 不混用。用 fake process group 不等于逻辑 world 为 1，也不证明真实 HCCL 可用。
- `PatchScope` 恢复 Python binding、注册和回调；同时审计 autograd 多线程状态、SPMD TLS、DTensor/RoPE cache、target flags。成功与异常退出都测试，后续换 world/rank 不能继承旧 mesh。
- 恢复局部 patch 不等于撤销厂商模块导入或所有框架缓存。冷／热状态可改变初始化 trace，不能把原始地址或首次运行的节点数当作跨进程恒等 ID。

参考：`test_runtime.py`、`test_patch_scope.py`、`test_config.py`；实现见 `runtime.py`、`compat/patches.py` 和 `compat/distributed.py`。

## 兼容层的决策顺序

坚持「原生 dispatch → 必要的 Meta/autograd 补齐 → 无法表达的部分显式建模」。

1. 用真实 schema、Meta 调用和输入梯度探针确认缺口。Meta 输出 shape 正确，不代表 autograd 已注册或参数梯度正确。
2. 健康原生路径继续走 dispatch；不要为已有 CANN 算子手写第二套节点或 shape 推导。
3. 确认缺少 Meta 或 autograd 后，局部修正分配设备或桥接原生 backward，保存真实输入对象。不能只压 warning，也不能用同 shape 占位 tensor 冒充保存值。
4. 原生运行不可执行且无合适接口表达时，才注册明确的 virtual schema、Meta 和 autograd。记录真实设备不可执行的边界，并保留融合语义。
5. 无法可靠匹配接口时显式失败，不按版本字符串安装一整套旧 patch，不吞掉未知参数。

升级时为旧 patch 建立处置表，逐项记录目标符号、当前调用点和退出条件：原生已具备的删除，仍有行为缺口的保留最小兼容，纯手工补图／推断恢复的优先退役，未支持的能力明确拒绝。
不能只因旧文件移动或 import 成功就认定 patch 继续有效；尤其检查延迟绑定、cached callable 和函数签名。迁移前后的具体台账见 [旧 patch 清单](legacy-patch-inventory.md)。

### CANN metadata 的具体经验

Python wrapper 可能只暴露 `*args/**kwargs`。不能把这两个名称当作实际算子参数；需要从 dispatcher schema 恢复参数顺序、keyword-only、默认值和必填校验。
不能为了兼容而删除校验、静默丢参数或执行真实设备 fallback。

厂商 Meta 本身可能显式分配 NPU buffer；只在已确认缺陷的 metadata shape-inference 作用域内约束为 Meta，继续复用原厂推导。不要全局重定向 `torch.empty` 等 factory。
准备阶段 metadata 能走 dispatch，不等于已经纳入训练 F/B/O 图；必须说明准备阶段的统计范围。详见 [未决成本范围](open-questions.md)。

autograd warning 可能来自 bridge 安装前的能力探针，但不能据此断言无害。最终按参数梯度、实际 backward 节点、保存值和真实运行结果判断。
在主仓 CPU 全套测试中也曾因 mock／真实 CANN 注册交互进入设备 metadata fallback 而停滞；应保留调用栈、隔离测试和环境证据，不将中止运行记为全套通过。

## tensor/storage 与设备事实

- tensor value、Python wrapper、底层 allocation 和 allocation generation 分开记录。view/detach 可能共享 storage，新 wrapper 不一定是新值或新分配。
- mutation 和 alias 仍形成真实数据依赖。不得用最后一个形状相同的 tensor、按值相等或 Python 地址复用来推断身份。
- saved-tensor pack、unpack、slot release、Python 引用和 C++ storage 所有权分别观察。第一次 unpack 或最后读取不自动表示释放。
- 只计真实本地设备 storage。CPU planner／控制元数据不因参与 NPU 调用就变成设备显存；并行原始几何和 local shard 元数据分开。
- 优化释放检测不能降低观测边界频率。候选队列只减少待检查对象，仍在每个原 dispatch 边界确认原生过期；用逐边界参考扫描证明等价，不以峰值碰巧一致代替。

参考：[保存身份测试](../tests/capture/test_saved_tensor_identity.py)、[所有权测试](../tests/capture/test_storage_lifetimes.py)、[逐边界对照](../tests/test_storage_lifetime_capture.py)。

## 通信、流控制与负载结构

通信必须保留真实 process group、rank 成员、primitive、dtype 和 volume。EP／TP／FSDP mesh 可以重叠，不从 world 或模块名猜轴，也不恢复旧 HSDP 轴修正来绕过新版布局。

FSDP 包装可能延迟安装或刷新函数引用。observer 必须覆盖最终生效的 native wrapper，按 transition 关联 AG staging、full allocation、wait、reshard、finalize 和未消费 prefetch 的清理。
不删除看似没有 tensor consumer 的预取，也不以 full parameter Python view 仍存在为由否认显式释放。

同流顺序与 `Event.record`／`wait_event`／`wait_stream` 按实际前沿记录。事件等待的是记录时的源流工作，不自动等待之后的新工作。
零耗时 `stream_wait` 没有 tensor 输入输出，仍必须保留前驱；`record_stream` 是生命周期语义，不是执行屏障。host polling/synchronize 的当前支持边界不能扩大解释。

至少验证 AG 输出实际链接到后续权重使用；延后 AG 会延后投影。HSDP 的 RS/AR 必须按实际 wait/event 约束传播，clip／更新等待其必要归约；无等待的其他流仍可重叠。
不能按捕获序号把全部节点补成串行边，也不能因 FLOPs 为 0 删除控制或 view 节点。

L0 源事件到最终图必须唯一，L1 模板引用、L2 action/DataSlot、L3 iteration 保持闭合和 DAG。当前没有图折叠、节点克隆／重编号及补偿恢复；不要通过重新补图掩盖捕获缺口。
模板复用不等于已经支持完整多 microbatch／PP 训练。当前单 microbatch 的 GBS/GA 元数据不能被下游解释为完整累积峰值。

参考：[流契约](stream-control-dependencies.md)、[FSDP 审计](fsdp-dependency-audit.md)、[精简记录](simulator-cleanup.md)、`validation.py`。

## 内存、成本与对外结果

从真实 optimizer state 按设备与 local storage 去重，再投影稳态驻留；不按权重倍数猜测，不在 optimizer 阶段重复加状态。
首次训练步可能在 optimizer 阶段才懒初始化状态；稳态预算将实际观察到的状态视为从步首常驻，并非额外执行第二步。CPU step scalar 不计设备字节；无可读取 owner 时报告不可用，不将未知状态当作 0。
四个阶段／总峰值包括状态，权重与状态独立大小是说明项，不能再加到峰值上。当前是逻辑活跃 storage，不是 allocator reserved/cache 或真实 workspace。
offload 的保存集合、层外驻留、`unattributed` 和逐层单向 payload 必须遵守 [模型内存规则](model-integration.md#保存值不等于卸载候选)。

`OpNode.peak_mem` 等成本注记不能替代生命周期峰值；unknown cost 不等于零时间。设备、算子成本或未来调度策略的假设应与捕获事实分开。
CLI 和 `simulate()` 成功完成仿真后自动导出 `<output_dir>/memory/memory_summary.json`；`output_formats=["mem"]` 额外写入内存明细。正常调用无需再次执行 `export_result()`；它只用于显式重新导出。build-only 不导出内存结果，捕获或校验失败也不生成新结果。复用目录不会清理旧明细，建议每次使用独立目录。不恢复 JSON/HTML/DOT 等图导出或折叠。
对外 summary 只有六项预算加一个逐层 payload 列表，细节见 [输出契约](steady-state-memory.md)。更改字段或含义时同步 Python、文件、CLI、acceptance 与接入文档。

## 验收与排查入口

在已配置的 sidecar 环境中，从 `experiments/simulator` 运行运行时快速回归：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/test_runtime.py tests/test_patch_scope.py tests/test_metadata.py \
  tests/capture/test_saved_tensor_identity.py \
  tests/capture/test_stream_dependencies.py
```

随后按变更扩大至相关 FSDP／通信、AC、量化、模型及完整几何，并使用 [README](../README.md#验证) 的完整测试／独立类型入口。
sidecar 采用自身 `pyproject.toml`；避免同一模块又以 `experiments.simulator.*` 被根项目扫描。editable 安装的静态分析需要真实 source search path，不能通过把整个依赖替换为 Any 来消除告警。

新失败必须比较相同依赖、命令与搜索根的基线。既有 root 类型错误、主机缺依赖、实际新增错误分别记录；有局部通过不能声称全仓全绿。
性能优化先拆解 import/JIT、准备、dispatch、保存／所有权、graph assembly 和 export，再比较相同几何及冷／热条件。等价性检查包括原始事件、生命周期和控制依赖，性能计时不混入参考审计。
