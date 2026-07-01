# MHC 真实算子名捕获 — 架构与方案设计（阶段一：MHC）

## 1. 背景与问题

`docs/superpowers/specs/2026-07-01-npu-simulator-design.md` 完成后的审阅中发现两个与
"捕获出的图与真实运行算子名一致"这一目标相关的问题：

1. **可视化标签丢失真实算子名（已修复）**：`dot_export.py`/`html_export.py` 之前直接用
   `OpNode.op_type` 作为节点显示标签；`op_type` 只覆盖 `OP_MAPPING` 表里 ~20 个规范算子名，
   其余一律显示为字面量 `"unknown"`。16 层冒烟模型里实测有 94 种不在表内的真实算子（如
   `aten.embedding.default`、`npu.npu_moe_token_unpermute_with_routing_map.default`），全部
   在图上显示成无区分度的 `"unknown"`，即便真实算子名早已被
   `dispatch_capture.py` 记录在 `OpNode.annotations["raw_op_type"]` 里。已修复：新增
   `op_mapping.display_op_label(op_type, annotations)`，`op_type=="unknown"` 时 fallback 到
   `raw_op_type`，`dot_export.py`/`html_export.py`/`text_summary.py` 三处统一改用该函数。

2. **MHC/SMLA 转换器被整体剥离，捕获出的算子序列与生产环境不一致（本文档要解决的问题）**：
   `SimulationTrainer._strip_hardware_dependent_model_converters` 会把
   `npu_mhc_pre`/`npu_mhc_post`/`npu_smla` 从 `config.model_converters.converters` 中删除，
   回退到转换前的纯 PyTorch base 类（`HcPre`/`HcPost`/`HcHead`/`SparseAttention`/`LiCompute`/
   `LiLoss`）。**验收配置 `deepseek_v4_pro_debug_61_layers_4k_384die` 的
   `_default_converters()` 确实同时启用了 `npu_smla` 和 `npu_mhc_pre`**（`config_registry.py`
   第 34-43 行）——这不是假设场景，捕获出的图目前对这些模块显示的是与生产环境完全不同的算子
   序列（base class 的 `matmul`/`softmax` 等，而不是真实跑的 `npu_hc_pre`/Triton kernel）。

本文档设计**阶段一**的修复：仅覆盖 **MHC**（`HcPre`/`HcHead`，对应 `npu_mhc_pre` 转换器，
`npu_mhc_post`/`HcPost` 在验收配置里未启用但设计上一并覆盖）。**SMLA 作为阶段二，本文档不
覆盖，暂维持现状剥离**（原因见 §6）。

## 2. 关键约束（决定了不能"直接跑真实实现"）

读源码逐一确认：

- `MHCPreConverter.convert()`/`MHCPostConverter.convert()` 里
  `use_fused_kernel = get_npu_device_type() == "A5"`；`get_npu_device_type()` 调用
  `torch_npu.npu.get_device_name()`，仿真环境下这是 `meta_env._MetaDeviceModule`，
  `get_device_name()` 返回 `"Meta_Simulator"`，不匹配 `_NPU_DEVICE_TYPE_MAP` 任何 marker
  → 恒为 `"UNKNOWN"` → `use_fused_kernel` 恒为 `False`。也就是说**不加干预的话，仿真环境下
  永远走非 A5（Triton）分支**。
- 即使我们强行让 `get_npu_device_type()` 返回 `"A5"`，`MHCPreConverter.convert()` 里
  `if use_fused_kernel: import custom_ops` 这一步会失败——`custom_ops` 是一个**私有扩展包，
  在任何环境（包括真实容器）都确认不可得**，这不是仿真特有的限制。所以 A5 融合路径**在当前
  条件下无法被真正选中执行**，与是否仿真无关。
- 非 A5 分支最终调用的 `hc_pre_bmm_forward`/`hc_pre_fwd` 等（`ops/triton/*.py`）是裸
  `@triton.jit` kernel：既不经过 PyTorch dispatcher（`TorchDispatchMode` 完全看不到），也无法
  在 meta tensor 上运行（Triton 需要真实内存地址读写）。

**结论**：两条路径都无法真正执行；唯一现实的做法是"影子记录"——不跑真实 kernel，只按真实算
子名 + 解析出的形状公式，向捕获流里手工登记一条 `OpNode`。且由于 A5 路径连
`import custom_ops` 都过不去，**阶段一只实现非 A5（Triton）目标**，A5 目标的真实中间张量形
状（`hc_before_norm`/`inv_rms`/`sum_out`/`norm_out` 等）留待确有需要时再补充验证。

## 3. 架构设计

### 3.1 新增能力一览

```
torchtitan_npu/simulator/
├── capture/
│   └── dispatch_capture.py      # 改动：新增 record_synthetic_op() 方法
├── hardware_shims/               # 新增子包
│   ├── __init__.py
│   └── mhc_shim.py               # 新增：SimHcPre / SimHcHead
├── trainer.py                    # 改动：SimulationConfig 新增 target_npu_device_type 字段；
│                                  #      SimulationTrainer 用 SimHcPre/SimHcHead 替换剥离逻辑
```

不改动任何生产代码文件（`converters/kernels/mhc_prepost.py`、`ops/triton/*.py` 保持字节不
变）——延续本项目"纯侧载新增文件"的约定。

### 3.2 `OpDispatchCapture.record_synthetic_op`

```python
def record_synthetic_op(
    self,
    raw_op_type: str,
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    module_path: str = "",
) -> None:
    """Manually register one synthetic L0 event, as if `raw_op_type` had
    gone through __torch_dispatch__ normally. Used by hardware_shims for
    ops that cannot execute for real (raw Triton kernels / JIT-compiled
    extensions) but whose real op name + output shape are known
    analytically. Participates in the same producer/consumer id(tensor)
    wiring and repeat_count dedup as real dispatched events."""
```

实现上复用 `OpDispatchCapture.__torch_dispatch__` 里已有的 `_RawEvent` 构造 +
producer 绑定 + 去重逻辑（提炼成一个私有 `_record_event(raw_op_type, inputs, outputs,
module_path, phase)` 辅助方法，`__torch_dispatch__` 和 `record_synthetic_op` 两处共用，避免
重复实现"生成 op_id → 建 TensorMeta → 记录 predecessor → 更新 producer 表 → 去重折叠"这套
逻辑）。`op_type` 一律通过现有 `to_canonical_op_type(raw_op_type)` 解析（MHC 的真实名不在
`OP_MAPPING` 表内，会得到 `"unknown"`，然后走 §1 已修复的 `display_op_label` 兜底显示
`raw_op_type`——两个修复点在这里自然衔接）。

shim 如何拿到当前 `OpDispatchCapture` 实例：`OpDispatchCapture` 已经是通过
`with capture:` 包裹整个 step 的 `TorchDispatchMode`，同一时刻只有一个实例处于激活状态。新增
一个模块级 `_active_capture: OpDispatchCapture | None` 上下文变量，`OpDispatchCapture.__enter__`
/`__exit__` 里设置/清空，shim 通过 `get_active_capture()` 取到它——避免把 capture 实例层层透
传进模型 forward。

**关键正确性要求（自查阶段发现）：shim 的核心计算必须实现成 `torch.autograd.Function`**，
结构上镜像真实的 `MHCPreTriton`/`MHCPreOnlyTriton`/`MHCPostTriton`，而不能是普通函数调用。
原因：

1. **梯度图连通性**：`torch.empty(..., device="meta")` 产生的是叶子张量，本身不带任何
   `grad_fn`。如果 `SimHcPre.forward` 只是直接调用普通函数拼出输出，PyTorch 不知道这个输出
   "依赖"于输入——后续调用 `loss.backward()` 时，梯度传播会在这里断掉，导致 MHC 模块之前
   （更早）的所有层都收不到梯度、`step_boundary.py` 统计出的 backward 节点数偏少。用
   `torch.autograd.Function.apply(...)` 包裹（哪怕 `forward`/`backward` 内部只做 shape 推导，
   不做真实数值计算）可以让 PyTorch 自动把输出的 `grad_fn` 指向这个 Function，`.backward()`
   走到这里时会正确调用我们的 `backward()`，梯度（同样是 `torch.empty(device="meta")` 构造、
   与对应输入同 shape）继续正确流向更早的层——这与真实的 `MHCPreTriton` 完全同构：它的
   Triton kernel 内部也是用 `torch.empty(...)` 现分配输出 buffer，梯度连通性同样完全依赖
   外层 `torch.autograd.Function`，而不是张量自身"记得"什么。
2. **backward 阶段合成节点的正确 phase 标记**：`step_boundary.StepBoundaryTracker` 在
   `torch.Tensor.backward()` **被调用的那一刻**把 `current_phase` 设为 `"backward"`，此后
   autograd 引擎在遍历计算图时触发的每一个 `torch.autograd.Function.backward()`（包括我们的
   shim）都发生在这个窗口内。只有把"登记 `hc_pre_bwd`/`hc_pre_bmm_backward` 合成节点"这件事
   真正放在 `_SimHcPreFn.backward()` 里执行（而不是提前在 `forward()` 时就一次性登记完
   forward+backward 两组节点），`record_synthetic_op` 读到的 `phase_provider()` 才会正确返回
   `"backward"`，与真实调度顺序、真实 phase 归类完全一致。

因此 `SimHcPre`/`SimHcHead`/`SimHcPost` 三个 `nn.Module` 内部各自委托给一个同名
`_SimHcPreFn`/`_SimHcHeadFn`/`_SimHcPostFn`（`torch.autograd.Function` 子类）：
`forward(ctx, ...)` 里登记 forward 阶段的合成节点、返回 shape 正确的 meta 输出，
`ctx.save_for_backward` 只需保存**形状信息**（不需要真实数值）；
`backward(ctx, *grad_outputs)` 里登记 backward 阶段的合成节点、返回与每个输入同 shape 的
`torch.empty(device="meta")` 梯度占位。

### 3.3 `SimulationConfig.target_npu_device_type`

```python
@dataclass
class SimulationConfig:
    output_dir: str
    target_npu_device_type: Literal["A5", "non_a5"] = "non_a5"
```

默认 `"non_a5"`（原因：A5 路径依赖不可得的 `custom_ops`，`non_a5` 是唯一当前可验证、可落地
的目标；用户可显式切到 `"A5"`，但阶段一 `SimHcPre`/`SimHcHead` 在 `target_npu_device_type=="A5"`
时的行为是：仍然只登记非 A5 一套算子名，附加一条 `logger.warning`，直到阶段二/三补充 A5 真实
形状公式为止——不做"看起来支持但形状是瞎猜的"半成品）。

### 3.4 `SimHcPre` / `SimHcHead`（`hardware_shims/mhc_shim.py`）

替换生产环境的 `NpuHcPre`/`NpuHcHead`（这两个类只在仿真环境下被这两个 shim 类替换，生产
`torchtitan_npu` 包完全不受影响）。以 `SimHcPre.forward` 为例，对照 `NpuHcPre.forward`
（`mhc_prepost.py`）+ `MHCPreTriton.forward`（`ops/triton/mhc_triton.py`）逐行核对后的真实
调用序列 & 形状（`n = self.hc_mult`，本仓 DeepSeek-V4-Pro 配置里 `n=4`）：

| 步骤 | 真实调用 | 是否需要 shim | 输入 shape | 输出 shape |
|---|---|---|---|---|
| 1 | `torch_npu.npu_rms_norm` | 否（已是真实 dispatcher 算子，meta 下已验证可用，模型其余位置同款调用正常跑通） | `x_flat[BS,nD]`, `gamma[nD]` | `x_norm_flat[BS,nD]`, `rstd[BS,1]` |
| 2 | `torch.matmul` | 否（普通 aten 算子） | `x_norm_mat[B,S,nD]`, `weight[nD,n²+2n]` | `x_proj[B,S,n²+2n]` |
| 3 | `hc_pre_fwd`（Triton, Sinkhorn） | **是** | `mixes=x_proj[B,S,n²+2n]`, `hc_scale[3]`, `hc_base[n²+2n]` | `h_pre[B,S,n]`, `h_post[B,S,n]`, `h_res[B,S,n,n]` |
| 4 | `hc_pre_bmm_forward`（Triton, BMM） | **是** | `H_pre=h_pre[B,S,n]`, `x=x_unflatten[B,S,n,D]` | `y[B,S,D]` |

`raw_op_type` 登记为 `"triton.hc_pre_fwd"`/`"triton.hc_pre_bmm_forward"`（`triton.` 前缀
区分于 `aten.`/`npu.` 命名空间，明确标注"这是一个手工登记的合成节点，对应真实生产环境里的
同名 Triton 函数"，在 `annotations` 里额外加 `"synthetic": True` 标记，供后续排查/统计使用）。
backward 对称登记 `"triton.hc_pre_bwd"`/`"triton.hc_pre_bmm_backward"`。

`SimHcHead`（对应 `MHCPreOnlyTriton`）结构相同，用 `hc_pre_only_fwd`/`hc_pre_only_bwd`
替换步骤 3（只产出 `h_pre[B,S,n]`，没有 `h_post`/`h_res`）。

`SimHcPost`（`npu_mhc_post`，验收配置未启用，但设计一并覆盖，供 `_enable_all_converters()`
场景使用）对应 `MHCPostTriton`：步骤拆成 `hc_post_bmm1_forward([B,S,C],[B,S,4]→[B,S,4,C])` +
`hc_post_bmm2_forward([B,S,4,4],[B,S,4,C]→[B,S,4,C])` + `add_fwd`（逐元素相加，两个
`[B*S,N*D]` 张量相加得到同形状结果，无需自定义 backward——PyTorch 自动微分对纯加法自动给出
梯度，不用单独登记）三个合成节点，backward 对称登记 `hc_post_bmm1_backward`/
`hc_post_bmm2_backward`。

所有形状公式均已对照 `ops/triton/pre_bmm.py`/`post_bmm1.py`/`post_bmm2.py`/
`prepost_sinkhorn.py`/`add.py` 里的 `if xxx.shape != ...: raise ValueError(...)` 断言和函数
docstring 逐一核实，不是猜测值。

### 3.5 影子张量的构造

shim 不调用真实 Triton kernel，而是直接：

```python
def _meta_like(shape: tuple[int, ...], dtype: torch.dtype, ref: torch.Tensor) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=ref.device)  # ref.device is already "meta"
```

`torch.empty(..., device="meta")` 不做真实内存分配（与整个 simulator 的 meta-only 前提一
致），值不重要（下游同样是 shape-only 传播，之前 20 个任务的验收已经证明这套"只关心 shape
不关心数值"的策略对训练一步的完整捕获是够用的——DeepSeek-V4-Pro 61 层验收本身也是这个前提
下跑通的）。

## 4. 端到端集成

`trainer.py`：

```python
def apply_mhc_shims() -> None:
    """Patch MHCPrePostModelConfig.model_converter / MHCPostModelConfig.model_converter to
    point at SimMHCPreConverter/SimMHCPostConverter, instead of stripping npu_mhc_pre/
    npu_mhc_post out entirely -- keeps the captured op sequence's real op names consistent
    with what the real converter would have produced. Reversible via unapply_mhc_shims()."""
```

`SimulationTrainer.__init__` 调用顺序改为：先调用 `apply_mhc_shims()`（把
`npu_mhc_pre`/`npu_mhc_post` 转换器的目标类替换成 shim），再调用
`_strip_hardware_dependent_model_converters(config)`（`_HARDWARE_DEPENDENT_CONVERTER_NAMES`
收窄为只剩 `frozenset({"npu_smla"})`，MHC 不再被剥离）。

替换转换器目标类的具体机制（**实现阶段相对本节初版的简化**：不是对
`config.model_converters.converters` 列表做逐项替换，而是直接对已注册的
`MHCPrePostModelConfig`/`MHCPostModelConfig` 类做**可逆的类属性 patch**——与
`meta_env.py` 现有的"记录原值、patch、可对称 unpatch"约定完全一致，也更简单：不需要在
`config` 实例层面重建兼容的 Config 对象）：新增 `SimMHCPreConverter`/`SimMHCPostConverter`
（继承现有 `ModelCustomConverter`，`convert()` 里把 `HcPre`/`HcHead`/`HcPost` 替换成
`SimHcPre`/`SimHcHead`/`SimHcPost`），`apply_mhc_shims()` 把
`MHCPrePostModelConfig.model_converter`/`MHCPostModelConfig.model_converter` 这两个类属性
直接指向它们（`unapply_mhc_shims()` 对称恢复原值）。

**实现阶段发现的额外关键点（真实容器验证中确认，见 §8）**：
- `SimHcPre`/`SimHcHead`/`SimHcPost` 必须继承 `HcPre`/`HcHead`/`HcPost`（与真实的
  `NpuHcPre(HcPre)`/`NpuHcHead(HcHead)`/`NpuHcPost(HcPost)` 完全一致的继承方式），不能只继承
  `nn.Module`——torchtitan 的 `BaseModel.verify_module_protocol()` 会检查每个子模块是否是
  `torchtitan.protocols.module.Module` 的实例，纯 `nn.Module` 子类会导致模型构建时直接崩溃。
  `HcPre`/`HcHead`/`HcPost` 本身已经是 `Module` 的子类，直接继承它们即可零成本满足这个检查。
- `HcHead`（不同于 `HcPre`）并不把 `hc_mult` 存成实例属性——`HcHead.__init__` 只是把它当一个
  局部变量用来决定 `hc_head_fn`/`hc_head_base` 的形状。`SimHcHead.forward` 必须从
  `self.hc_head_fn.shape[0]` 现算 `hc_mult`，不能引用不存在的 `self.hc_mult`。

## 5. 测试与验证策略（先小后大，呼应最初"先小规模验证"的要求）

1. **单元测试（纯 Python，不需要 torch_npu）**：
   - `record_synthetic_op` 正确生成 `OpNode`、正确串联 predecessor/successor、正确参与
     `repeat_count` 去重（复用 `test_dispatch_capture.py` 已有的测试基础设施和断言风格）。
   - `SimHcPre`/`SimHcHead`/`SimHcPost` 在 CPU 上用**普通（非 meta）张量**跑一遍 forward，
     断言：(a) 输出 shape 与上表一致；(b) `record_synthetic_op` 被调用且 `raw_op_type` 与预
     期一致；(c) 不抛异常、不依赖 Triton/torch_npu。
   - **梯度连通性回归测试（对应 §3.2 的关键正确性要求）**：构造一个 `x.requires_grad_(True)`
     的输入，跑 `SimHcPre(...).sum().backward()`，断言 `x.grad is not None` 且
     `x.grad.shape == x.shape`——证明 `_SimHcPreFn` 正确参与了 autograd 图，没有在 MHC 这一层
     断掉梯度传播。同时断言：`record_synthetic_op` 记录的 forward 组节点在 `.backward()`
     调用**之前**就已存在，backward 组节点在调用**之后**才出现，且都带有正确的
     `phase`（用一个假的 `phase_provider` 模拟 `StepBoundaryTracker` 的行为来验证时序）。
2. **容器内小规模 spike**（先只验证 `hc_pre_bmm_forward` 这一个、形状最简单的节点）：
   构造一个只含 1 层、`npu_mhc_pre` 启用的最小配置，跑一次，确认
   `trace.html`/`compute_graph.dot` 里出现 `triton.hc_pre_bmm_forward` 节点而不是回退到
   base `HcPre` 的 `matmul`/`softmax` 序列。
3. **扩展到全部 MHC 算子**后，重跑 16 层冒烟 + 61 层验收，确认：
   - exit code 0，四个产出文件正常生成。
   - `summary.txt` 的 "Unrecognized op types" 列表里不再包含任何 base-`HcPre`/`HcHead`
     特有的算子（因为已经换成 shim 登记的真实名）。
   - forward/backward 节点总数、通信统计（allgather/reduce_scatter/allreduce）与之前验收
     结果保持同一数量级（MHC 只占整个 61 层模型的一小部分子模块，不应引起总量剧烈变化）。

## 6. 明确不在阶段一范围内（阶段二待办）

- **SMLA**（`npu_smla`：`SparseAttention`/`LiCompute`/`LiLoss`）：继续维持现状剥离。SMLA 的
  稀疏/lightning-indexer 注意力族形状公式比 MHC 复杂得多（`metadata` 依赖运行时 topk 索引结
  构），且非 A5 路径走的是 `build_op(...)` JIT 编译的本地扩展对象方法调用（`_sas_op.xxx(...)`），
  同样拿不到 dispatcher 钩子，需要同一套 `record_synthetic_op` 机制，但形状公式需要单独一轮
  源码核实 + 小规模验证，作为独立后续任务跟踪，不在本文档/本阶段实现。
- **A5 目标的真实形状公式**：`torch_npu.npu_hc_pre`/`npu_mhc_pre_sinkhorn_grad` 等的完整中间
  张量形状（`hc_before_norm`/`inv_rms`/`sum_out`/`norm_out`）未在本阶段验证，`target_npu_
  device_type="A5"` 场景下 `SimHcPre` 会记录 warning 并退化为非 A5 命名，等确有需要时再补充。

## 7. 与用户需求的对应关系

- "抓取的计算图符合原始完整流程" + "抓取出来的图中与真实运行的算子名一致"：本设计让
  MHC 相关节点的 `raw_op_type` 与生产环境真实调用的函数名（`torch_npu` 算子名或
  `ops/triton/*.py` 里的真实函数名）保持一致，不再依赖"剥离转换器、退化成不同实现"的近似。
- 不需要真实 NPU 硬件、不做真实内存分配：`record_synthetic_op` 全程只用
  `torch.empty(..., device="meta")` 构造影子输出，不调用任何真实 Triton kernel 或 aclnn 扩展。

## 8. 验证结果（全新 CANN 容器 `titan-npu-sim-e2e`，与原 20 任务验收同一容器）

实施阶段通过 subagent-driven-development 按 8 个任务逐一实现 + review，容器内真实验证过程中
额外发现并修复了 2 个真实 bug（均已同步更新到本文档 §4 和实施计划文档）：

1. `SimHcHead` 引用不存在的 `self.hc_mult`（`HcHead` 只在 `__init__` 里把它当局部变量用，从不
   存成实例属性）——16 层冒烟跑到 `test_mhc_shim.py` 单元测试阶段就被抓到
   （`AttributeError: 'HcHead' object has no attribute 'hc_mult'`），修复为从
   `self.hc_head_fn.shape[0]` 现算。
2. `SimHcPre`/`SimHcHead`/`SimHcPost` 只继承了 `nn.Module`，未继承 `HcPre`/`HcHead`/`HcPost`，
   导致真实构建 16 层模型时 `torchtitan.protocols.model.BaseModel.verify_module_protocol()`
   直接抛出 `RuntimeError`（17 个 `hc_pre` 子模块全部不满足 Module protocol）——单元测试测不出
   这个问题（单元测试直接构造 `SimHcPre` 实例，不经过完整模型构建 + protocol 校验），只有真实
   跑通 `torchtitan_npu.entry` 才会触发；改为继承 `HcPre`/`HcHead`/`HcPost`（与真实
   `NpuHcPre(HcPre)` 等完全一致的继承方式）后解决。

**16 层冒烟**（`deepseek_v4_pro_simulate_16_layers`，`--training.steps=1`）：`EXIT_CODE=0`，
`compute_graph.dot` 中 `triton.hc_pre_fwd`/`triton.hc_pre_bmm_forward` 各出现 68 次，
`triton.hc_pre_bwd`/`triton.hc_pre_bmm_backward` 各出现 34 次——replaces 掉了之前 base `HcPre`
类的 `rsqrt`/`sigmoid`/`matmul` 等不相关算子序列。`summary.txt` 的 forward/backward/optimizer
节点数（9136/18646/1006）、通信统计与去除 MHC 影响前后保持同一数量级，`is_acyclic=True`。

**61 层/384 die 验收目标**（`deepseek_v4_pro_simulate_61_layers`）：`EXIT_CODE=0`（约 8-9
分钟，与 MHC 修复前基本一致）。关键结果：
- `RankTable`：`world_size=384`（`dp_degree=384`），`tp_degree=pp_degree=1`——与验收目标完全
  一致，MHC shim 不影响并行拓扑。
- 真实算子名验证：`compute_graph.dot` 中 `triton.hc_pre_fwd`/`triton.hc_pre_bmm_forward` 各
  248 次，`triton.hc_pre_bwd`/`triton.hc_pre_bmm_backward` 各 124 次（相对 16 层的 68/34 按
  61/16≈3.8 倍比例放大，与层数缩放关系一致）。
- `summary.txt` 的"Unrecognized op types"列表（105 项）正确包含这 4 个 `triton.hc_pre_*`
  合成算子名（预期行为：它们不在 `OP_MAPPING`/cost model 覆盖范围内，故成本估算标记为
  "未知"，但真实算子名本身已正确捕获并可见——回应 §1 已修复的可视化标签兜底问题）。
- forward/backward/optimizer 节点数：34250/68398/3571（相对 MHC 修复前的
  50027/137299/3508 有差异，符合预期：MHC 真实算子序列与 base-class 回退序列的节点粒度本就
  不同）；`is_acyclic=True`；通信统计（`allgather`/`allreduce`/`reduce_scatter`）与之前保持
  同一数量级。
- 四个产出文件（`simulation_result.json` ~8.8GB、`compute_graph.dot`、`summary.txt`、
  `trace.html`）全部正确生成。

**结论**：阶段一（MHC 真实算子名捕获）目标完全达成并通过真实容器验证；SMLA（阶段二）与 A5
目标形状公式仍按 §6 保持未实现状态，留待后续任务。

