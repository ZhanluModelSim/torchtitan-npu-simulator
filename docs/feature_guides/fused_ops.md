# 融合算子接入指南

本文面向在 `torchtitan-npu` 中接入 AscendC、Triton 或者 Tilelang 等融合算子的开发者。
融合算子应作为 NPU 适配层通过 [override 机制](../../torchtitan_npu/override/README.md) 接入，不修改上游 `torchtitan`；默认训练路径不受影响，只有在 `override.imports` 中显式列出入口后才会替换组件。

## 选择接入入口

| 接入对象 | 接入方式 | 说明 | 算子示例 |
| --- | --- | --- | --- |
| 融合算子对应完整可配置组件 | `override` + `ops` | 整体替换组件；已有设备算子可由 replacement 直接调用 | [`torch_npu.npu_rms_norm`](../../torchtitan_npu/override/common/rms_norm.py) |
| 融合算子是可配置组件的一部分 | `override` + `ops` | 只替换组件中的对应逻辑，其余逻辑沿用上游实现 | [`npu_moe_token_unpermute`](../../torchtitan_npu/ops/ascendc/moe_token_unpermute.py)（接入：[`torchtitan_npu.override.common.token_dispatcher.asc`](../../torchtitan_npu/override/common/token_dispatcher.py)） |

`torch.library.custom_op` 只是 `ops` 实现需要进入 `torch.compile` 图时的兼容封装，
不改变上述两种组件替换方式。

## 目录与命名

### 入口命名

公开入口统一为：

```text
torchtitan_npu.override.<scope>.<target>.<variant>
```

- `scope`：`common` 或模型名（如 `deepseek_v3_2`、`deepseek_v4`）。只依赖 `torchtitan`
  公共组件的实现放在 `common/`；依赖模型专属配置、布局或 metadata 的实现放在模型目录。
- `target`：被替换对象的语义名称，使用 `snake_case`，例如 `rms_norm`、`sparse_attn`、`mhc`。
- `variant`：实现或行为。常用取值如下。

| Variant | 用途 |
| --- | --- |
| `golden` | 标准 PyTorch 算子实现的数值基线 |
| `asc` | AscendC 融合算子 |
| `triton` | Triton 融合算子 |
| `pypto` | 使用 PyPTO 编程模型实现的算子 |
| `tilelang` | TileLang 融合算子 |

注意事项：

- 同一 target 有多个同类实现时，在 variant 后增加职责限定，例如
`rope.asc_complex`、`rope.asc_cossin`、`sparse_attn.asc_metadata`。

- replacement 类采用「variant + target」命名，并保留标准缩写大小写，例如
`AscRMSNorm`、`AscCompressedSparseInnerAttention`、`GoldenCompressedSparseInnerAttention`。

### 文件组织

简单 target 使用单文件；同一 target 包含多个较大的后端实现时使用 package：

```text
torchtitan_npu/override/<scope>/<target>/
├── __init__.py       # @override 注册入口
├── ascendc.py        # AscendC 实现
├── golden.py         # 可选的数值基线
└── triton.py         # 可选的 Triton 实现
```

## 接入流程

### 1. 确定 target 和契约

沿模型调用链找到要替换的 `Configurable.Config`、`build()` 和 `forward()`，记录：

- 输入输出 shape、dtype、layout、stride 和 device；
- 本地 tensor、`DTensor`、mesh、placements、参数 sharding 与 checkpoint 语义；
- 是否有变长 metadata、context parallel 或额外的 host/NPU 数据转换；
- 算子支持的设备代际、CANN 版本、dtype、shape 和并行范围。

选择最小且稳定的组件边界。融合算子覆盖组件全部计算时，整体替换 Module；只覆盖组件部分计算时，
继承并重写对应方法。模型专属 target 不要放入 `common/`。

### 2. 实现算子层

- AscendC/CANN：kernel 由 CANN 或 `torch_npu` 提供接口，仓库不重复实现。`ops/ascendc/` 负责调用
  native op，并在缺少原生反向时补充 `register_autograd`；需要编译入图时，再补充 Fake 和兼容封装。
- Triton/PyPTO：实际 kernel 实现放在 `torchtitan_npu/ops/triton/` 或
  `torchtitan_npu/ops/pypto/`，由 `override` 中的 replacement 调用。
- 只有需要 `torch.compile` 入图时，才使用 `torch.library.custom_op` 注册 schema，并补充
  `register_fake` 和 `register_autograd`。这层是编译兼容封装，不改变「`override` 替换组件、
  `ops` 对接后端算子」的主接入方式。
- `torch.library.custom_op` 使用仓库 namespace，例如
  `torchtitan_npu::npu_moe_token_unpermute`；CANN 原生算子保留其 vendor namespace。
- 算子需要的 TND 或其他 vendor metadata 由该 attention backend 自己的
  `Config.build_attention_metadata` 预计算：decoder 每个 microbatch 只构建一次，同几何的层共享
  同一份（upstream PR #4897 的接缝）。metadata 要把模型公共 metadata 包在里面，而不是把公共
  metadata 改成只适用于某个后端的类型。

当前实现示例：

- DeepSeek-V3.2 的 `AscSparseInnerAttention.Config.build_attention_metadata` 把公共的 varlen
  layout 包成 `AscVarlenMetadata` 并读出 TND 需要的 host 序列长度；`sparse_attn.asc` 调用
  `torch_npu.npu_lightning_indexer` 和 `torch_npu.npu_sparse_flash_attention`。
- DeepSeek-V4.1 的 `AscV41SparseAttention.Config.build_attention_metadata` 建**这个几何自己**的
  entry：参考后端的 builder 给出整行的 document 边界与按自身 `compress_ratio` 折好的 `cu_seq_k`
  （池化轴，折一次，消费者不再除），端口在这两个 grid 上加自己的三个 kernel frame（`q` /
  `swa_k` / `cmp_k`）与 `aicpu_metadata`——按自身几何调用 CANN `sparse_flash_mla_metadata` /
  `sparse_flash_mla_grad_metadata` 与 `lightning_indexer_metadata` /
  `sparse_lightning_indexer_kl_loss_grad_metadata` 得到的四条 metadata（每 microbatch 一次），
  在 `aicpu_metadata` 下按消费它的 kernel 分组、按方向取值：`sparse_attn.fwd` /
  `sparse_attn.bwd` 与 `lightning_indexer.fwd` / `lightning_indexer.bwd`。就这些——一方向一份。
  `sparse_attn.asc` 逐层按 `(端口类, compress_ratio)` 取自己的 entry，直接读
  `aicpu_metadata.sparse_attn.fwd` 等字段：一条 entry 就是一个几何，没有按 topk 或按几何查表的步骤。
  几何（heads/head dim/indexer 三数）是该 config 自己的必填字段、无默认值，由 override 的五个参数
  一次写入，默认值即 V4.1-Flash 的发布形状，因此没有单独的 metadata 入口。这些 metadata 是内核的
  唯一来源——没有 inline 回退：拿着没经过 override 构建的 metadata 前向会直接失败。
  CP 下由 `AscV41Metadata.prepare_cp_metadata` 用 rank-local frames 重建。
  attention override 不声明任何量化：量化后的 attention 是**另一个 backend**，由量化适配层
  （`torchtitan_npu/extensions/adaptor/torchao/converter.py`）安装自己的 config 替换本节点，并由
  `torchao_npu` 的 provider（`quantized_modules/deepseek_v4_1/attention_metadata.py`）建自己的 entry
  ——字段名与上面同一套（`sparse_attn.fwd`/`.bwd`、`lightning_indexer.fwd`/`.bwd`），但 selector
  那一组换成量化 selector 的 QLI schedule 并多出 `fwd_sparse`（QSLI）与 `sparse_length`（逐行候选数）；
  候选池的容量与块大小从模型 config 读取，只作为 launch 参数与候选数推导的输入。因此 `asc` 在这条路径上
  不生效，非量化 entry 的 `fwd_sparse`/`sparse_length` 保持为空，等带 candidate 的非量化算子落地后由端口填。
- DeepSeek-V4 的 `sparse_attn.asc_metadata` 生成 CANN `*_metadata`，
  `sparse_attn.asc` 桥接 LightningIndexer/SparseFlashMLA 的正反向；`sparse_attn.pypto` 用
  PyPTO 实现 LI/LIG，并复用 AscendC 的其余 attention kernel。（V4 尚未迁移到 backend-owned
  接缝，仍由 `metadata_extension` 预计算。）
- MoE token dispatcher 的 `npu_moe_token_unpermute` 与 `npu_moe_re_routing` 在
  `torchtitan_npu/ops/ascendc/` 中以 `torch.library.custom_op` 做编译兼容封装，并提供 Fake 和
  Autograd；
  `npu_moe_token_permute` 直接调用 `torch_npu`。三者由
  `override/common/token_dispatcher.py` 的 `asc` 入口组合，用于替换
  `AllToAllTokenDispatcher` 的部分 `dispatch/combine` 逻辑。

### 3. 实现 replacement Module 和 Config

replacement 通常继承上游组件，保留共有配置字段，并只修改实际差异：

```python
from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm


class AscRMSNorm(RMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


@override(
    target=RMSNorm.Config,
    description="AscendC fused RMSNorm via torch_npu.npu_rms_norm",
)
def asc(cfg: RMSNorm.Config) -> AscRMSNorm.Config:
    return derive(cfg, AscRMSNorm.Config)
```

实现必须保持上游组件的输入输出语义。必要的 layout 转换只应发生在 kernel 边界，并在返回前
恢复模型约定的布局。

`derive(cfg, NewConfig, **deltas)` 会复制同名字段；只有新增字段或明确变化的字段才写入 `deltas`。
replacement 只支持具体 Config 类型时设置 `exact=True`；需要按层或按 FQN 选择实例时使用
`fqns=["..."]`。override 入口函数的可调参数应保持为关键字参数，并通过 CLI 的 JSON 传入。

### 4. 注册和激活

导入模块只负责触发 `@override` 注册，不负责自动替换。启动时使用完整的
`module.function` 路径：

```bash
python -m torchtitan_npu.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --override.imports \
    torchtitan_npu.override.common.rms_norm.asc \
    torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata \
    torchtitan_npu.override.deepseek_v4.sparse_attn.asc
```

多个无参数入口可用空格或逗号分隔。带参数的入口必须将 JSON 作为一个 shell 参数，例如：

```bash
--override.imports \
  'torchtitan_npu.override.deepseek_v4.sparse_attn.asc={"indexer_loss_coeff":2.0}'
```

同一模块包含多个 override 入口时，只会启用命令中点名的入口函数。`torchtitan` 在配置构造完成、组件
`build()` 前依次完成「导入注册 → 解析入口 → 遍历原始配置树收集匹配节点 → 检查冲突 →
替换配置」。匹配在修改前一次性收集，因此入口顺序不会改变匹配结果；启动日志应出现
`[Override]` 和 `Applied N override(s)`。

### 5. 处理配套和冲突

多个 override 是否能同时启用取决于实际声明的配置节点，而不是模块路径是否不同：

- DeepSeek-V3.2 只需启用 `sparse_attn.asc`：attention 与它自己的 TND metadata 同属一个
  backend，metadata 由被替换的 config 直接构建，不再有单独的 metadata 入口。
- DeepSeek-V4.1 的 `sparse_attn.asc` 同时替换 attention 并构建它自己的 metadata，
  没有单独的 metadata 入口；五个几何数是替换后 config 的必填字段，由该入口的参数写入
  （默认值为 V4.1-Flash 形状）。本入口不声明量化：量化 selector 的 metadata 由量化后的 attention
  自己建（另一条 backend，见上文），池的容量与块大小取自模型 config。它与 `lightning_indexer.asc`
  必须同开（教师边只有两侧都在时才存在），见 `torchtitan_npu/override/README.md#deepseek-v41`。
- DeepSeek-V4 的 `sparse_attn.asc_metadata` 必须与 `sparse_attn.asc` 或 `sparse_attn.pypto` 配套；
  `sparse_attn.asc`、`sparse_attn.pypto`、`sparse_attn.golden` 声明同一 attention target，
  三者只能选一。`golden` 是数值参考，不是融合性能路径。
- DeepSeek-V4 MHC 的 AscendC 入口声明 `HcPre.Config`/`HcPost.Config`，Triton 入口还包含
  `HcHead.Config`；同一节点只能选择一个实现。
- 两个 override 声明同一节点，或一个声明另一个的祖先/后代节点时会冲突；可移除互斥入口，
  或使用 `fqns` 缩小范围。

导入任意 `torchtitan_npu.override.*` 子模块会先执行包初始化，因而可能同时加载
`torchtitan_npu` 的 package patch。运行环境应具备匹配的 PyTorch、`torch_npu`、CANN、HCCL；
使用 `cann_ops_transformer` 或 PyPTO 时，还需安装对应 runtime。package patch 只用于无法用
配置级 override 表达的 backend 缺口或临时适配。

## 验证清单

按算子层、组件层和端到端层逐级验证，并在文档或提交说明中写明设备、runtime、dtype、shape
和并行配置：

1. **算子层**：检查 schema、Fake/Meta 输出的 shape/dtype/device；使用
   `torch.library.opcheck`（适用时）覆盖 schema、FakeTensor 和 Autograd。对 forward 与
   backward 分别检查有限值和数值误差。
2. **组件层**：将 replacement 与 eager/reference 组件加载相同参数，比较 forward 输出、
   参数梯度和输入梯度；用 `torch.compile(backend="aot_eager")` 覆盖编译正反向。
3. **模型层**：执行相关 smoke/integration case，确认 override 日志、kernel 调用、
   `DTensor` sharding、context parallel 和 checkpoint 行为。
4. **数值层**：需要基线时使用 `golden` 或上游 eager/reference 实现；比较 TensorBoard 中
   未截断的逐 step `loss` 和 `grad_norm`，不要只使用 stdout 的有限精度。未在目标设备、
   runtime 或并行配置上验证的组合标记为「未验证」。
5. **静态检查**：提交前执行 `git diff --check`，核对入口路径、配置 target、相对链接和
   文档中的支持边界。

## 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 模块导入失败 | 完整入口路径、PyTorch/`torch_npu`/CANN/PyPTO 依赖 |
| `target` 未注册 | 是否使用准确的 `module.function`，以及模块是否被导入 |
| 没有匹配节点 | `target`、`exact`、`fqns`，以及 converter 后的 Config 类型 |
| 同节点或嵌套节点冲突 | 是否同时启用了 fused、PyPTO、Triton 或 golden 替代实现 |
| 训练继续但 replacement 未生效 | 启动日志中的 `[Override]` 和 `Applied N override(s)` |
| 编译反向失败或 FakeTensor 报错 | custom op 是否注册 `register_fake` 和 `register_autograd` |
| kernel dtype/shape 报错 | 调用边界的 dtype、layout、stride、TND metadata 和设备限制 |

## 参考实现

- [`override/README.md`](../../torchtitan_npu/override/README.md)：override 机制、入口和当前清单。
- [`deepseek_v3_2/sparse_attn`](../../torchtitan_npu/override/deepseek_v3_2/sparse_attn)：TND metadata 与 AscendC DSA。
- [`deepseek_v4/sparse_attn`](../../torchtitan_npu/override/deepseek_v4/sparse_attn)：AscendC、PyPTO 与 golden DSA。
- [`deepseek_v4/mhc`](../../torchtitan_npu/override/deepseek_v4/mhc)：AscendC 与 Triton 多后端替换。
- [`torchtitan_npu/override/common/token_dispatcher.py`](../../torchtitan_npu/override/common/token_dispatcher.py)：`token_dispatcher.asc` 在 `AllToAllTokenDispatcher` 中替换局部 dispatch/combine 计算。
- [`torchtitan_npu/override/common/rope.py`](../../torchtitan_npu/override/common/rope.py)：`rope.asc_partial` 整体替换 partial-RoPE 站点，含可选 CANN 依赖的惰性加载。
- [`torchtitan_npu/ops/ascendc/inplace_partial_rotary_mul.py`](../../torchtitan_npu/ops/ascendc/inplace_partial_rotary_mul.py)：原生 mutator 算子的 functional autograd 包装（clone + 前向/反向原地旋转）。
- [`ops/ascendc/moe_token_unpermute.py`](../../torchtitan_npu/ops/ascendc/moe_token_unpermute.py)：custom op、Fake 和 Autograd 注册示例。
