# 捕获运行时与负载契约

本文约束 `runtime/compat/capture/ir/memory` 的修改，避免新模型兼容破坏已有负载。整体执行顺序见[整体方案](architecture.md)，接入步骤见[模型接入](model-integration.md)。下列规则以当前源码和对应测试为依据；字段不得因下游暂未消费就随意删除。

## 1. 会话、环境与兼容层

`simulate(SimulationConfig(...))` 是完整入口；低层 `run_simulation_step()` 只捕获一次传入的训练动作，不负责结构预热。修改前记录 torch、torch_npu、CANN、torchtitan、主包及 sidecar 的实际版本、导入路径与 Git SHA，不能仅依据镜像名或 editable 安装记录。

- 导入 torch／simulator 前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 并加载配套 CANN 环境；Python API 不替调用者加载 shell 环境。
- `meta_runtime()` 与 `PatchScope` 管理临时兼容绑定、作用域内 kernel 替换和上下文，正常及异常退出都恢复；不修改上游 checkout 或安装目录。`compat/kimi.py` 的 custom op/autograd 定义及 `tensor_semantics` 声明则是模块级静态注册，不随会话卸载，不在每次进入时重复定义。
- `logical_world()` 拥有 fake process group。宿主已有 group、嵌套或并发仿真不属于支持路径；连续串行捕获必须允许换 rank/world。
- 恢复内容包括 autograd 多线程状态、SPMD TLS、mesh／RoPE cache 和 target flags。局部 patch 恢复不等于厂商模块从未导入，不能把原始地址当跨进程稳定 ID。
- 先探测实际 schema、Meta 分配和梯度；健康原生实现不覆盖。Python wrapper 只显示 `*args/**kwargs` 时，从 dispatcher schema 取得参数，不按版本字符串猜签名。
- CANN Meta 显式申请 NPU buffer 时，仅在确认的 metadata shape-inference 范围修正，继续复用原厂推导；禁止全局重定向 tensor factory。

实现：[runtime.py](../torchtitan_simulator/runtime.py)、[compat/patches.py](../torchtitan_simulator/compat/patches.py)、[compat/distributed.py](../torchtitan_simulator/compat/distributed.py)。回归：`tests/test_runtime.py`、`test_patch_scope.py`、`test_metadata.py`、`test_chunked_loss_policy.py`。

## 2. 节点、参数与原始身份

`OpDispatchCapture` 在原生 dispatch 周围记录事件，`build_nodes()` 组装事实，不运行 costmodel。节点保留原始 overload、shape/dtype/stride、FQN、执行身份、依赖和必要参数；不输出训练侧 FLOPs、预测耗时、性能分类或重复峰值。

### 参数 occurrence 与 scalar

一般算子的输入按位置参数、再按关键字插入顺序展开张量，同一张量重复传入保留多个位置。`attrs["input_tensor_bindings"]` 保存 `(参数名, 结构)` 元组：整数引用本节点输入索引，元组保留 list/tuple 边界，`None` 表示缺席。例如：

```python
(("self", 0), ("indices", (None, 1)), ("values", 2))
```

原生 fused AdamW 是明确的例外：六个 TensorList 与尾部可选张量按 schema 顺序规范化；其通用绑定同步采用该顺序，另保存列表长度和可选角色。详见[AdamW 输入身份](adamw-input-identity.md)。

没有 schema 时不生成绑定；张量字典超出当前 v1 的 tuple/list 协议时整组绑定省略，不交付不完整映射。不按 shape、对象相等或 FQN 推断参数角色。

非张量参数按实际值保存，不读取 Meta 数据。合法填充值 `-inf/+inf/nan` 编码为 `("__sim_scalar_float_v1__", "-inf" | "+inf" | "nan")`；该编码是数据字面量，不是硬件参数或耗时。c10d ReduceOp 从实际枚举提取。通用参数中的 `premul_sum` 表示归约身份，乘数由通信事实单独交付。

实现与测试：[operator_arguments.py](../torchtitan_simulator/capture/operator_arguments.py)、[test_operator_arguments.py](../tests/capture/test_operator_arguments.py)。

### 自定义算子语义

本仓自定义融合／包装按完整原始 overload 声明 `attrs["tensor_semantics"]` v1，给出每个输入输出 occurrence 的角色和语义轴。未知轴用 `None`；复合轴表示单一 shape 维度的逻辑展平。保留物理 layout/stride，不根据维度大小猜 query、token 或 head。原生 ATen/CANN 不要求额外声明，详见[自定义张量语义](custom-tensor-semantics.md)。

## 3. 数据、通信与流依赖

| 约束 | 必须保留的事实 | 禁止的替代 |
| --- | --- | --- |
| Tensor 数据流 | producer、实际输入输出、alias 和 storage mutation frontier | 找最后一个同 shape tensor 当 producer |
| 同流顺序 | 该逻辑流的已提交设备工作前沿 | 按全局 seq_idx 把所有流串行化 |
| event 等待 | record 当时源流前沿及目标流先前工作 | 自动等待 record 之后的新工作 |
| 通信 | primitive、实际 process group、group/ranks、本地 tensor 元数据、通信完成依赖 | 从 world 或模块名猜并行域，把 launch 返回当完成 |
| 生命周期 | `record_stream` 等真实所有权信息 | 把 allocator 记录变成执行屏障 |

零耗时 `stream_wait` 无 tensor 输入输出，但前驱必须完成。只改元数据的 view/alias 保留数据边，不推进设备流前沿；不能由预取 buffer 的 view 引入额外同步。`metadata_view` 判定因此是捕获语义，不是应删除的性能分类。

EP 默认通过 `compat/routing.py` 统一适配原生 functional `all_to_all_single`，保留 count→data/score→compute→combine 的完成依赖与原生 autograd。该适配不改 CP/FSDP；非 Meta 调用保留原实现。内部通信 event/slot 的 `volume_bytes` 用于源张量容量、内存和数据传递，不等于下游模型预测的网络流量。

DSV4 Muon 继续执行原生 FlexShard 双向 all-to-all；transfer/caller 两流及 event 快照保留预取重叠和 buffer 复用等待。跨 rank 计划哈希比较在单 logical-rank Meta 会话中明确标记未执行，不伪造 peer 值。混合参数更新和通信依赖验收见 [Muon 捕获](muon.md)。

FSDP wrapper 可能延迟安装或刷新。正式观察器在结构预热后安装，并以 transition 关联 AG staging、full allocation、wait、reshard、finalize 和未消费 prefetch 清理。预热阶段没有 active capture 的原生调用是合法行为，不得要求观察器在那里已有节点。

至少验证：延后 AG/RS/AR 会沿真实等待边延后必要计算与 clip/update；无等待的其他流仍能重叠。参考[流控制依赖](stream-control-dependencies.md)、[EP 交换](ep-single-transport.md)、`tests/test_fsdp_stream_capture.py`、`tests/capture/test_native_collectives.py`。host polling/synchronize 和真实硬件时序不能由这些 Meta 测试推导。

### 通信语义事实 v2

`attrs["collective_facts"]` 为下游通信模型交付独立的语义事实，不计算流量或耗时。首期只覆盖 `c10d._allgather_base_.default`、`c10d._reduce_scatter_base_.default`、`c10d.allreduce_.default` 与 `_c10d_functional.all_to_all_single.default`，不按公共算子名套用其他 overload 的参数角色。

事实包含 `schema_version=2`、精确 primitive、实际 global capture rank、有序组、实际 dense／contiguous 检查、原语的跨 rank 等形保证和逻辑缓冲关系。归约从仍可解释的原生枚举或 TorchBind `op()` 读取，按安装版本的具名枚举映射到 `reduction.operation`，与通用参数绑定并存；不能从 opaque 文本或 FSDP 场景推断 SUM。多 rank 的 AG 本地 alias 不能证明全组 alias，缺全组保证时保守交付 `distinct`；只有单 rank 可凭完整本地 storage／offset 证据交付 `same_region`。原地 AR 由原语契约确定。

Meta 运行时在 `torch.distributed.reduce_scatter_single` 调用边界读取原生 `ReduceOp.factor`，并以输入、输出张量身份关联到对应的 RS dispatch。退出调用时清除关联。标量乘数记录为捕获 rank 的 `reduction.pre_scale`，`post_scale` 为 1，`scale_implemented_by_nodes` 为 false。原生后端负责该次乘法。TorchBind 只暴露归约身份，tensor 乘数、非有限值和绕过观察边界的调用均记录 `reduction.pre_scale` 缺失。v1 的默认值 1.0 对 PREMUL_SUM 无效。

EP 适配在所有 router 的 force-load-balance 已核验、且入口明确交付同 geometry 的 synthetic SPMD 来源时，生成版本化 `torchtitan_balanced_ep_v1` 证明，区分 dispatch／combine；count exchange 交付完整 equal 行矩阵。不从本 rank 的一行 split 或均匀 shape 推断全组矩阵。原生 A2A autograd 节点 metadata 保存对应证明，反向捕获读取实际节点，转置矩阵并交换全组 input／output shape，不替换原生求梯度函数。

缺实际组、归约或完整 split 通过 `unresolved_fields` 明示；Muon 等其他 A2A 来源未交付全组矩阵或证明时同样保留缺失，下游必须拒绝完整通信预测。结构预热、FSDP 通信视图、卸载计划与静态显存口径不因新增事实改变。实现见 [collective_facts.py](../torchtitan_simulator/capture/collective_facts.py)，native FakeProcessGroup／Meta 回归入口见 [test_collective_facts.py](../tests/capture/test_collective_facts.py)；这些测试不证明设备数值或通信性能。

## 4. 阶段与图引用

`comp_type` 组织 F/B/O action，`phase`／`execution_kind` 描述实际 autograd 与重算。chunked loss 的 head backward 可以发生在 F action 内；按 `phase` 全局重新分阶段会破坏已捕获路径。`is_recompute` 与 module path 保留供下游统计。

完整源事件 ID 唯一；模板、action、DataSlot、跨模板依赖及外层 iteration 的引用必须闭合且无环。不得为清除 unresolved 引用而丢边、重编号、折叠节点或推断补出一条全串行链。训练 IR 的层级是组织结构，不等于平台已经实现 PP 或完整 GA 回放。

### FSDP 通信视图

默认返回图根据实际 `framework_scopes` 隐藏 FSDP unshard/wait/reshard/post-backward/state hook/finalize 中的非通信工作，保留模型计算、通信与流控制。禁止按根 FQN 筛除节点，根模块内也可能存在正常计算。

投影沿原有路径将可见前驱传递给可见后继，更新 successors、模板边界和 stream/event 前沿，不给保留节点分配新 ID。通信、slot 与 DataPass 的单值展示锚点同步调整；多分支执行依赖仍以全部正式图边为准，不能用单值锚点替代。

`captured_workload` 保留完整源图。投影初始共用 MemoryPlan；卸载变换随后创建派生预算，保留原始事件/lifetime。内存事件原始 ID 在源图解析，不要求都在交付图可见。`fsdp_communication_view` 注记记录隐藏范围和耗时假设；下游通信视图有意不计 FSDP 打包、copy、cast 等准备开销。

实现：[fsdp_communication_view.py](../torchtitan_simulator/capture/fsdp_communication_view.py)；回归：`tests/capture/test_fsdp_communication_view.py`、`tests/test_fsdp_stream_capture.py`。

## 5. 保存值、storage 与静态内存

- Tensor wrapper、tensor value、storage 和 allocation generation 分开标识；view/detach 可共享 storage，地址复用不表示同一代分配。
- 保存槽 pack/unpack/release、Python 引用和原生所有权分别观察。第一次 unpack、最后读取或 wrapper 销毁均不能单独作为 storage 释放。
- 释放检测在每个原 dispatch 边界检查；弱引用候选只减少待查集合，最终仍由原生过期判定确认。不得降低检查频率或用峰值偶然相等代替逐边界等价。
- 只计本地目标设备 storage；CPU planner 元数据不计设备显存。参数全局几何、本地 shard、通信 tensor、量化 code/scale 分开记录。

保存钩子为了恢复梯度信息引入的 detach 只按精确返回对象身份排除：弱引用匹配后消费标记，保留 alias/mutation 关系；不能按 `aten.detach` 名称或同 storage 一概过滤。排除次数记录在 `capture_artifacts.saved_tensor_unpack_detach_count`，不作为内存预算修正值。回归见 `tests/capture/test_saved_tensor_identity.py`、`test_storage_lifetimes.py` 与 `tests/test_storage_lifetime_capture.py`。

优化器状态从原生 owner 的公开 `state` 获取并按本地 storage 去重。正式捕获初始登记预热后已有状态；初始化算子消失不意味着状态驻留或卸载量消失。未知 owner 报不可用，已知无状态才为 0。无卸载时状态已在 F/B/O 峰值内，不重复相加；开启卸载时保留原始总量，但按聚合近似排除状态设备驻留，暂不计状态 staging buffer。Muon runtime 常驻工作缓冲另行登记为初始内存，覆盖 F/B/O，不计入可卸载状态量，也不随卸载排除。

保存值集合、层归属、head 交错反向及后续 B/recompute 使用共同决定激活卸载。不能靠过滤报表将未归属存储从峰值删除，也不能把传输 payload 当卸载后的驻留值。具体预算和图变换分别见[单卡内存](steady-state-memory.md)、[激活卸载](activation-offload-design.md)、[优化器卸载](optimizer-offload.md)。

## 6. 交付与验收

正式交付为 Python `WorkloadGraph` 和其中的 `memory_plan`／`memory_summary`，默认不落盘。显式 `output_formats=["mem"]` 才导出摘要和诊断明细；`build-only` 没有正式训练图、梯度或内存验收结果。复用目录不会自动清理旧文件，必须避免把旧产物当本次成功。

修改捕获协议时，先验证参数与依赖最小行为，再跑真实模型路径，最后按变化扩大并行／AC／量化组合。快速回归入口如下，需先按 [README](../README.md) 配置完整运行环境；从 `experiments/simulator` 执行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/test_runtime.py tests/test_patch_scope.py tests/test_metadata.py \
  tests/capture/test_operator_arguments.py tests/capture/test_saved_tensor_identity.py \
  tests/capture/test_stream_dependencies.py
```

更多模型验收步骤见[接入工作流](model-integration.md)。区分本轮执行结果、静态检查和旧证据；Meta 结构验证不替代 NPU 数值验证，也不证明真实性能精度。准备阶段是否纳入独立负载见[待定范围](open-questions.md)。
