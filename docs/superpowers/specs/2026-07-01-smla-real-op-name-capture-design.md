# SMLA 真实算子名捕获 — 架构与方案设计（阶段二：SMLA）

## 1. 背景与问题

`docs/superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md`（阶段一：MHC）完成并
通过真实容器验收后，用户在审阅 61 层验收结果时发现：**捕获图中 attention 相关计算显示为原始
分解的 aten 小算子**（`matmul`/`softmax`/`einsum`/`topk`/`scatter_`/`cat` 等），而不是融合后的
NPU sparse attention / lightning-indexer 算子。

### 1.1 根因（已用 systematic-debugging 流程确认，非猜测）

- `torchtitan_npu/simulator/trainer.py:77`：`_HARDWARE_DEPENDENT_CONVERTER_NAMES =
  frozenset({"npu_smla"})`——`npu_smla` 转换器目前仍被整体剥离，这是阶段一（MHC）设计文档
  §6 明确记录、有意推迟到"阶段二"的范围外项。
- `npu_smla` 在验收配置的 `_default_converters()` 中确实启用
  （`torchtitan_npu/models/deepseek_v4/config_registry.py:41,52`）。
- 剥离后，`SparseAttention`/`LiCompute` 落回转换前的 base 类（`torchtitan_npu/models/
  deepseek_v4/model.py:455-523`、`526-556`）。这两个 base 类是**手工分解实现**：
  `SparseAttention.forward` 用 `matmul`+`scatter_`+`cat`+`softmax`+`matmul` 手搓 attention；
  `LiCompute.forward` 用 `einsum`+`relu_`+`topk` 手搓 lightning-indexer 打分。完全不调用任何
  融合 kernel。
- 捕获机制本身没有问题——`TorchDispatchMode` 忠实记录了实际执行的每个算子；问题在于实际执行
  的就是这套手工分解实现，而不是真实生产环境会跑的 `torch_npu` 融合算子。

本文档设计**阶段二**的修复：把 `npu_smla` 从"整体剥离"改为"影子记录"（与阶段一 MHC 完全
同构的模式），覆盖 `SparseAttention`、`LiCompute`、`LiLoss` 三个子模块。

## 2. 关键约束（与阶段一相同的结论，但需要重新为 SMLA 逐一确认）

- `get_npu_device_type()` 在仿真环境下恒为 `"UNKNOWN"`（`_MetaDeviceModule.get_device_name()`
  返回 `"Meta_Simulator"`，不匹配任何 marker），所以 `NpuSMLAConverter.convert()` 里
  `use_smla_kernel = get_npu_device_type() == "A5"` 恒为 `False`——非 A5 分支才是仿真环境下
  会被选中的路径。
- 非 A5 分支的三个真实算子（`sparse_attn_sharedkv`、`lightning_indexer`、
  `sparse_lightning_indexer_grad_kl_loss`）都是通过 `build_op(...)` 在**运行时 JIT 编译**的
  本地 ACLNN 自定义算子扩展（`torchtitan_npu/ops/aclnn/*/binding.cpp`，用 `ACLNN_CMD` 宏调用
  真实 NPU 算子库）——已确认这些扩展绑定为 `_sas_op`/`_li_op`/`_kl_op` 三个模块级全局变量
  （`npu_smla.py:41`），既不经过 PyTorch dispatcher，也无法在 meta tensor 上运行（真正调用会
  尝试编译+链接+执行真实 ACLNN 算子，需要真实 NPU 硬件与工具链）。
- A5 融合路径同样不可达：`NpuSMLAConverter.convert_smla_kernel()`（A5 分支）需要
  `importlib.import_module("custom_ops")`——这个私有包在任何环境下都确认不可得（与阶段一
  MHC 的结论完全一致）。

**结论**：与阶段一完全相同——两条路径都无法真正执行，唯一现实的做法是"影子记录"：不跑真实
kernel，只按真实算子名 + 解析出的形状公式，向捕获流里手工登记 `OpNode`。**阶段二只覆盖非 A5
目标**（与阶段一保持一致的范围限定）。

## 3. 命名约定：新增 `aclnn.*` 前缀

阶段一 MHC 的合成算子用 `triton.*` 前缀（真实是 `@triton.jit` 内核）。SMLA 的三个真实算子
是 `ACLNN_CMD` 宏构建的 ACLNN 自定义算子（已读取 `torchtitan_npu/ops/aclnn/*/binding.cpp`
源码确认，非 Triton），用 `aclnn.` 前缀更准确地反映其真实技术形态，同时与 `triton.*`/
`npu.*`（真实 dispatcher 算子）/`aten.*` 保持并列、互不混淆的命名体系。

## 4. 真实算子名 + 形状目录（逐一核实自源码，非推测）

### 4.1 `SparseAttention` → `SimNpuSparseAttention`

固定的公开契约（`SparseAttention.__init__`/`forward`，`model.py:455-523`，任何转换后的类都
必须保持这个 I/O 形状不变）：

```
输入: query_states[B,S,N,D], kv_states[B,S,D], attn_sink[N],
      kv_compress[B,S//R,D]|None, compress_topk_idxs[B,S,K]|None
输出: attn_output[B,S,N,D]
```
（`N`=`n_heads`, `D`=`head_dim`, `R`=`compress_ratios[layer_id]` ∈ {1,4,128}, `K`=`index_topk`）

真实调用链（`NpuSparseAttention.forward` → `npu_sparse_attn_shared_kv` wrapper →
`SparseAttnSharedKV(torch.autograd.Function)`，`npu_smla.py:1330-1359`、`537-588`、
`378-534`）：

| 步骤 | 真实算子名（`raw_op_type`） | 触发时机 | 输入 | 输出 |
|---|---|---|---|---|
| A | `aclnn.npu_sparse_attn_sharedkv_metadata` | forward | 5 个占位张量 + 标量（`num_heads_q=N` 等，仅用于确定 metadata 内容，值不影响下游 shape） | `metadata:[1024]` int32 |
| B | `aclnn.npu_sparse_attn_sharedkv` | forward | `query:[B,S,N,D]` bf16, `ori_kv:[B,S,1,D]` bf16（`kv_states.unsqueeze(2)`）, `cmp_kv:[B,S//R,1,D]`\|None bf16, `cmp_sparse_indices:[B,S,1,K]`\|None int32, `sinks:[N]` fp32, `metadata:[1024]` int32 | `result:[B,S,N,D]` bf16, `softmax_lse:[B,S,N,1]` fp32 |
| C（bwd） | `aclnn.npu_sparse_attn_sharedkv_grad` | backward | `query/ori_kv/cmp_kv/result/softmax_lse`（同 fwd 形状）+ `grad_output:[B,S,N,D]` | `dquery:[B,S,N,D]`, `dori_kv:[B,S,1,D]`, `dcmp_kv:[B,S//R,1,D]`\|空张量（`R==1` 时）, `dsinks:[N]` |

形状来源（已逐行核对 `torchtitan_npu/ops/aclnn/sparse_attn_sharedkv/binding.cpp`）：
`attnOutput = at::empty(query.sizes(), ...)`（第 42 行）、`lse_sizes.back()=1`（第 45-47
行）、`metadata = at::empty(1024, ...)`（第 20 行）、`dQuery/dOriKv/dSinks = at::empty(
query/oriKv/sinks.sizes(), ...)`（第 70-72 行）、`dCmpKv` 仅当 `cmpRatio > 1` 时才
`at::empty(cmpKv.sizes(), ...)`，否则是未定义的空 `at::Tensor()`（第 74-78 行）。

**`R` 的三种取值处理**（`NpuSparseAttention.forward` 第 1355 行：
`cmp_sparse_indices=compress_topk_idxs if self.compress_ratio == 4 else None`）：
- `R=4`：`cmp_kv:[B,S//4,1,D]`, `cmp_sparse_indices:[B,S,1,K]`, `topk=K`。
- `R=128`：`cmp_kv:[B,S//128,1,D]`, `cmp_sparse_indices=None`, `topk=0`。
- `R=1`：`cmp_kv=None`, `cmp_sparse_indices=None`, `topk=0`。

`SparseAttnSharedKV.backward` 的返回值有 24 个位置（对应 24 个 forward 位置参数），shim 只
需要在 `query`/`ori_kv`/`cmp_kv`（视 R 而定）/`sinks` 对应位置返回正确形状的梯度，其余（标量
参数、`cu_seq_lens_*`、`ori_sparse_indices` 等）返回 `None`。

### 4.2 `LiCompute` → `SimNpuLiCompute`

固定的公开契约（`LiCompute.__init__`/`forward`，`model.py:526-556`）：

```
输入: q_indexer[B,S,N_i,D_i], k_indexer[B,S//4,D_i], weights[B,S,N_i], seqlen:int, offset:int
输出: compress_topk_idxs[B,S,K] int32, index_score[B,S,K]
```
（`N_i`=`index_n_heads`, `D_i`=`index_head_dim`；`LiCompute`/`LiLoss` 只在 `R==4` 的层存在，
`InnerAttention.__init__` 里用 `if self.compress_ratio == 4:` 判断，`model.py:682-691`）

真实调用（`NpuLiCompute.forward`，`npu_smla.py:1362-1406`）：

| 步骤 | 真实算子名 | 输入 | 输出 |
|---|---|---|---|
| D | `aclnn.npu_lightning_indexer` | `query:[B,S,N_i,D_i]` bf16, `key:[B,S//4,1,D_i]` bf16（`k_indexer.unsqueeze(2)`）, `weights:[B,S,N_i]` bf16 | `sparse_indices:[B,S,1,K]` int32, `sparse_values:[B,S,1,K]` bf16 → 两者各自 `.squeeze(2)` 得到 `[B,S,K]` |

形状来源（`torchtitan_npu/ops/aclnn/lightning_indexer/binding.cpp:19-26`）：
`sparse_indices/sparse_values = at::empty({B, S1, N2=1, sparse_count=K}, ...)`。

**关键复杂点**：`NpuLiCompute.forward`（非 A5）**直接调用 `_li_op.npu_lightning_indexer(...)`，
没有 `torch.autograd.Function` 包装**（对照 A5 路径的 `LightningIndexer(torch.autograd.
Function)` 有显式 `backward`，`npu_smla.py:1036-1095`）。这不会导致真实训练崩溃，因为：
`compress_topk_idxs` 是 int32（本就无梯度）；`index_score` 的梯度实际经由**独立的** `LiLoss`
通道流动（`InnerAttention.forward` 把 `q`/`kv`/`kv_compress` `.detach()` 后传给
`li_loss`，`model.py:714-726`）。**shim 里必须自己包一层 `torch.autograd.Function`**（真实
实现没有，但仿真需要它来维持梯度图连通性、以及 backward 阶段 phase 标记的一致性——与阶段一
`SimHcHead` 面对"真实实现没有独立反向"时的处理原则相同），backward 对两个输出均返回
`None`（镜像 A5 路径 `LightningIndexer.backward` 返回 `_none_grads(4)` 的行为）。

### 4.3 `LiLoss` → `SimNpuLiLoss`

固定的公开契约（`LiLoss.forward`，`model.py:308-337`；`NpuLiLoss.forward` 参数顺序，
`npu_smla.py:1545`）：

```
输入: q[B,S,N,D](detached), kv[B,S,D](detached), kv_compress[B,S//4,D](detached),
      attn_sink[N](未使用), q_indexer[B,S,N_i,D_i], k_indexer[B,S//4,D_i],
      weights[B,S,N_i], sparse_indices[B,S,K], indexer_score[B,S,K](未使用),
      attention_masks(未使用), offset:int(未使用)
输出: loss（标量 fp32）
```

真实调用链是"延迟计算"模式（`NpuLiLoss.forward` → `npu_sparse_lightning_indexer_grad_kl_loss`
→ `SparseLightningIndexerGradKLLossWrapper(torch.autograd.Function)`，`npu_smla.py:1545-1573`、
`1499-1531`、`1409-1495`）：

| 步骤 | 真实算子名 | 触发时机 | 输入 | 输出 |
|---|---|---|---|---|
| （无） | 无真实调用 | forward | — | `loss = torch.zeros(1,...)[0]`（**forward 本身不发起任何硬件调用**，只是 `ctx.save_for_backward` 保存张量供 backward 用） |
| E | `aclnn.npu_sparse_lightning_indexer_grad_kl_loss` | backward | `query:[B,S,N,D]`, `key:[B,S//4,1,D]`（`kv_compress.unsqueeze(2)`）, `query_index:[B,S,N_i,D_i]`, `key_index:[B,S//4,1,D_i]`（`k_indexer.unsqueeze(2)`）, `weight:[B,S,N_i]`, `sparse_indices:[B,S,1,K]`（`.unsqueeze(2)`） | `d_query_index:[B,S,N_i,D_i]`, `d_key_index:[B,S//4,1,D_i]`, `d_weight:[B,S,N_i]`, `loss:[1]` fp32 |

形状来源（`torchtitan_npu/ops/aclnn/sparse_lightning_indexer_grad_kl_loss/binding.cpp:23-26`）：
`d_query_index/d_key_index/d_weight = at::zeros(query_index/key_index/weight.sizes(), ...)`，
`loss = at::zeros({1}, ...)`。

`SparseLightningIndexerGradKLLossWrapper.backward` 返回 14 个位置的梯度：`query`/`key`
（detached，梯度为 `None`）、`d_query_index`/`d_key_index`/`d_weights`（真实梯度）、
`sparse_indices`（int32，`None`）、其余 8 个标量参数（`None`）。

**与现有 `meta_env._patch_li_loss_to_skip_buggy_einsum` 的关系**：该 patch 打在
`LiLoss.forward`（base 类方法）上，用来绕过 base 类 `_current_selected_attn_dist` 的一个真实
预置 shape bug（从未在真实生产触发，因为生产环境永远用 `NpuLiLoss`）。`SimNpuLiLoss` 继承
`LiLoss` 但**自己定义 `forward`**——Python MRO 规则下，`SimNpuLiLoss` 实例调用 `.forward()`
永远解析到 `SimNpuLiLoss.forward`，`LiLoss.forward` 上的 patch 因此自然变得不可达，
**不需要对 `meta_env.py` 做任何修改**（不需要撤销该 patch，也不会和它冲突——两者面向不同的
类方法，MRO 保证不会同时生效）。

## 5. 架构设计

### 5.1 新增文件（与阶段一 MHC 完全同构的文件组织）

```
torchtitan_npu/simulator/hardware_shims/
├── smla_shim.py        # 新增：SimNpuSparseAttention / SimNpuLiCompute / SimNpuLiLoss
└── smla_converter.py    # 新增：SimSMLAConverter + apply_smla_shims()/unapply_smla_shims()
```

不改动任何生产代码文件（`converters/kernels/npu_smla.py`、`models/deepseek_v4/model.py`、
`ops/aclnn/*/binding.cpp` 保持字节不变），延续本项目"纯侧载新增文件"的约定。

### 5.2 `smla_shim.py` 设计要点

- `SimNpuSparseAttention(SparseAttention)`：`__init__(self, parent)` 用
  `self.__dict__.update(parent.__dict__)`（与阶段一所有 shim 类一致的模式，同时也是
  `torchtitan` `verify_module_protocol()` 要求的继承方式——继承 base 类才能满足
  `isinstance(mod, Module)` 检查，这是阶段一容器验证踩过的坑，这次直接在设计里避免）。
  `forward` 委托给 `_SimSparseAttnFn(torch.autograd.Function)`，登记步骤 A/B（forward）、
  C（backward）三个合成节点。
- `SimNpuLiCompute(LiCompute)`：同样继承 + `__dict__.update`。`forward` 委托给
  `_SimLightningIndexerFn(torch.autograd.Function)`（**阶段二新增的壳，真实实现没有对应的
  autograd.Function**），forward 登记步骤 D，backward 对两个输出返回 `None`（无合成节点可
  登记，因为真实非 A5 路径这里确实没有反向 kernal）。
- `SimNpuLiLoss(LiLoss)`：`forward` 直接返回 `torch.zeros((), device=q.device,
  dtype=torch.float32)`（不经过任何 `torch.autograd.Function`——**镜像真实
  `SparseLightningIndexerGradKLLossWrapper.forward` 本身就不发起硬件调用**这一事实），但用一个
  专门的 `_SimLiLossFn(torch.autograd.Function)` 包装，使得只有当 loss 真正参与
  `.backward()` 时才登记步骤 E 的合成节点（对应真实"延迟计算"语义：真实 kernel 只在反向触发）。
- 所有形状构造统一复用阶段一已验证的 `_record`/`_empty_like_shape` 风格 helper（在
  `smla_shim.py` 内重新定义一份，保持 `hardware_shims` 包内每个 shim 模块自成一体、不产生
  跨模块隐式依赖——与 `mhc_shim.py` 让 `mhc_converter.py`、而非其他 shim 模块，作为唯一的
  跨文件依赖点的组织方式一致）。

### 5.3 `smla_converter.py` 设计要点

```python
class SimSMLAConverter(ModelCustomConverter):
    """Replaces every SparseAttention/LiCompute/LiLoss submodule with the
    corresponding Sim* shim -- never selects the real fused (A5) or
    JIT-compiled (non-A5) implementation (see design doc §2: neither path
    can execute under simulation)."""

    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if isinstance(module, SparseAttention):
                replace_module_with_name(model, name, SimNpuSparseAttention(module))
            if isinstance(module, LiCompute):
                replace_module_with_name(model, name, SimNpuLiCompute(module))
            if isinstance(module, LiLoss):
                replace_module_with_name(model, name, SimNpuLiLoss(module))


def apply_smla_shims() -> None:
    """Patch NpuSMLAModelConfig.model_converter to point at SimSMLAConverter.
    Reversible via unapply_smla_shims(). Mirrors apply_mhc_shims() exactly."""
    ...

def unapply_smla_shims() -> None:
    ...
```

可逆类属性 patch 机制与阶段一 `apply_mhc_shims()`/`unapply_mhc_shims()` 完全同构（记录原值、
patch、对称恢复），patch 目标是已注册的 `NpuSMLAModelConfig.model_converter`（`npu_smla.py`
里 `@register_model_converter("npu_smla")` 装饰的类，`:1639-1640`）。

### 5.4 集成到 `SimulationTrainer`

`trainer.py` 改动：
1. 新增 `from torchtitan_npu.simulator.hardware_shims.smla_converter import apply_smla_shims`
   导入（按字母序插入现有 `hardware_shims`/`ir`/`meta_env` 导入块）。
2. `_HARDWARE_DEPENDENT_CONVERTER_NAMES` 从 `frozenset({"npu_smla"})` 收窄为
   **`frozenset()`（空集）**——不再需要剥离任何转换器。保留 `_strip_hardware_dependent_
   model_converters` 函数本身（不删除，YAGNI 但保留通用机制以备将来需要），只是这次调用时
   传入的名单为空、实际不做任何过滤。
3. `SimulationTrainer.__init__` 里 `apply_mhc_shims()` 调用之后（顺序不敏感，两者互不干扰，
   替换的是不同的转换器名）新增 `apply_smla_shims()` 调用，同样在
   `_strip_hardware_dependent_model_converters(config)` 之前。

## 6. 测试与验证策略（延续阶段一"先小后大"的原则）

1. **单元测试（纯 Python，构造真实 `SparseAttention`/`LiCompute`/`LiLoss` parent 实例）**：
   - 与阶段一相同的已知限制：本沙盒无法导入 `torchtitan_npu.models.deepseek_v4.model`
     （沙盒内 `torchtitan` 非精确 pinned commit，`TokenReorderer` 缺失，与阶段一记录的限制
     同一根因），这几个测试预期在沙盒里失败、在真实 CANN 容器里通过。
   - `SimNpuSparseAttention`：断言 `forward` 输出 shape 与 §4.1 一致；断言
     `record_synthetic_op` 记录了 `aclnn.npu_sparse_attn_sharedkv_metadata`/
     `aclnn.npu_sparse_attn_sharedkv`（forward 阶段）与 `aclnn.npu_sparse_attn_sharedkv_grad`
     （backward 阶段，且仅在 backward 阶段）；断言反向梯度正确传播到 `query_states`/
     `kv_states`/`attn_sink`（`kv_compress`、`compress_topk_idxs` 视 R 而定）。
   - `SimNpuLiCompute`：断言输出 shape；断言记录 `aclnn.npu_lightning_indexer`；断言反向对
     `q_indexer`/`k_indexer`/`weights` 返回 `None` 梯度不报错（即调用
     `.sum().backward()` 时若 `compress_topk_idxs`/`index_score` 参与了计算图，梯度正确
     终止在这里而不传播，同时不抛异常）。
   - `SimNpuLiLoss`：断言 forward 直接返回标量 0、不登记任何合成节点；断言只有真正调用
     `.backward()` 时才登记 `aclnn.npu_sparse_lightning_indexer_grad_kl_loss`，且反向梯度
     正确传播到 `q_indexer`/`k_indexer`/`weights`（`q`/`kv`/`kv_compress` 因为是 detached
     输入，梯度应为 `None`，与真实实现一致）。
2. **容器内小规模 spike**：先只验证 `SimNpuSparseAttention`（形状最简单、独立于 R 分支）跑通
   （构造一个 `R=1` 的最小配置），确认 `trace.html`/`compute_graph.dot` 里出现
   `aclnn.npu_sparse_attn_sharedkv`/`aclnn.npu_sparse_attn_sharedkv_grad` 节点。
3. **扩展到全部三个类** + `R=4`（含 LiCompute/LiLoss）分支后，重跑 16 层冒烟 + 61 层验收，
   确认：
   - exit code 0，四个产出文件正常生成。
   - `compute_graph.dot`/`summary.txt` 中出现全部 5 个 `aclnn.*` 合成算子名，且不再出现
     base `SparseAttention`/`LiCompute` 特有的算子序列（如 `aten.topk.default`——除非其他
     地方也用到 topk，需要具体核实是否消失或只是数量减少）。
   - forward/backward/optimizer 节点总数、通信统计与阶段二修改前保持同一数量级（SMLA 只占
     整个 61 层模型的一部分子模块）。

## 7. 明确不在阶段二范围内

- **A5 目标的真实形状公式**：与阶段一 MHC 相同的推迟理由——A5 路径需要不可得的 `custom_ops`
  私有包，`target_npu_device_type="A5"` 场景下这几个 shim 会记录 warning 并退化为非 A5
  命名，等确有需要时再补充。
- **`SMLAMetadataCache`/`attention_masks` 相关的 A5 专属机制**：非 A5 路径完全不依赖它们
  （已在 §2/§4 确认），本阶段不涉及。

## 8. 与用户需求的对应关系

- "为什么 attention 算子是原始小算子而非融合后的 sparse attention"：本设计让
  `SparseAttention`/`LiCompute`/`LiLoss` 相关节点的 `raw_op_type` 与生产环境真实调用的
  ACLNN 算子名（`aclnn.npu_sparse_attn_sharedkv` 等）保持一致，不再依赖"剥离转换器、退化成
  手工分解实现"的近似。
- 不需要真实 NPU 硬件、不做真实内存分配：`record_synthetic_op` 全程只用
  `torch.empty(..., device="meta")`/`torch.zeros(..., device="meta")` 构造影子输出，不调用
  任何真实 ACLNN 扩展。
