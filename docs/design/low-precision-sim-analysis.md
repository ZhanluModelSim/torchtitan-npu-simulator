# 低精度训练（MXFP8）Simulator 使能分析

> 分支：feat/low-precision-sim
> 日期：2026-07-10

## 1. 目标

在 simulator 中启用 MXFP8 低精度训练特性，使捕获的 IR 中包含 MXFP8 量化算子（`npu_dynamic_mx_quant`、`npu_quant_matmul`、`npu_grouped_matmul`），而非普通的 `aten.mm`/`aten.addmm`。

## 2. 当前状态

### 2.1 MXFP8 在真实训练中的工作方式

```
MXFP8Converter.convert(model)
  → 对 fqns 匹配的 nn.Linear 权重包装为 MXFP8TrainingWeightWrapperTensor
  → forward 时 __torch_function__ 拦截 matmul
    → _to_mxfp8_then_scaled_mm (被 NPU patch 替换为 NpuMXFP8MM.apply)
      → NpuMXFP8MM.forward:
          x_mxfp8 = torch_npu.npu_dynamic_mx_quant(x)     ← 量化
          weight_mxfp8 = torch_npu.npu_dynamic_mx_quant(w)  ← 量化
          output = torch_npu.npu_quant_matmul(x_mxfp8, w_mxfp8)  ← FP8 matmul
      → NpuMXFP8MM.backward:
          (同上，dx 和 dw 都在 FP8 精度下计算)
```

### 2.2 Simulator 中的问题

| 问题 | 原因 | 影响 |
|------|------|------|
| `has_mx_capability` 检查 NPU 硬件 | `get_npu_device_type()` 返回 "UNKNOWN"（无真实 NPU） | MXFP8Converter 初始化时 raise RuntimeError |
| `NpuMXFP8MM.forward` 调用 `npu_dynamic_mx_quant` | meta tensor 无数据，NPU 算子需要真实数据做量化 | 运行时崩溃 |
| `NpuMXFP8MM.forward` 调用 `npu_quant_matmul` | NPU 算子在 meta device 上无 meta kernel | 运行时崩溃 |
| `NpuMXFP8GroupedMM` 同上 | 同上 | 运行时崩溃 |
| torchao 未安装 | 容器中无 torchao | patches 被跳过 |

## 3. 方案

### 3.1 核心思路

与 hardware_shims（MHC/SMLA）的处理方式一致：为 MXFP8 的 NPU 算子创建 **meta-safe 影子实现**（shim），在 meta device 上用标准 PyTorch 算子模拟其 shape 行为，同时记录真实的算子名。

### 3.2 具体改动

#### 3.2.1 绕过硬件检查

在 `meta_env.py` 中 patch `has_mx_capability`，在 `_is_meta_simulation=True` 时直接返回 `True`：

```python
def _patch_mx_capability_check_for_meta():
    from torchtitan_npu.patches.torchao_npu import mx_capability_check
    orig = mx_capability_check.has_mx_capability
    def _meta_safe_has_mx_capability(major, minor):
        if _is_meta_simulation:
            return True
        return orig(major, minor)
    # patch
```

#### 3.2.2 创建 MXFP8 shim

创建 `torchtitan_npu/simulator/hardware_shims/mxfp8_shim.py`，为 `NpuMXFP8MM` 和 `NpuMXFP8GroupedMM` 提供 meta-safe 替换：

**`NpuMXFP8MM` shim**：

```python
class SimMXFP8MM(torch.autograd.Function):
    """Meta-safe shadow of NpuMXFP8MM.
    
    Records the real op names (npu_dynamic_mx_quant, npu_quant_matmul)
    while executing standard matmul on meta tensors for shape inference.
    """
    @staticmethod
    def forward(ctx, x, weight):
        from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
        cap = get_active_capture()
        if cap is not None:
            # Record quant ops
            cap.record_synthetic_op("npu.npu_dynamic_mx_quant.default", [x], [torch.empty_like(x)])
            cap.record_synthetic_op("npu.npu_dynamic_mx_quant.default", [weight], [torch.empty_like(weight)])
            # Record matmul
            out = torch.matmul(x, weight.t())
            cap.record_synthetic_op("npu.npu_quant_matmul.default", [x, weight], [out])
        else:
            out = torch.matmul(x, weight.t())
        ctx.save_for_backward(x, weight)
        return out
    
    @staticmethod
    def backward(ctx, grads):
        x, weight = ctx.saved_tensors
        # Record quant + matmul for dx
        dx = torch.matmul(grads, weight)
        # Record quant + matmul for dw
        dw = torch.matmul(grads.t(), x)
        return dx, dw
```

**`NpuMXFP8GroupedMM` shim**：类似，但使用 `torch_npu.npu_grouped_matmul` 的 shape 推断（或用标准 matmul 模拟）。

#### 3.2.3 替换 NPU MXFP8 patches

在 `meta_env.py` 中，当 `_is_meta_simulation=True` 时，将 `_patched_to_mxfp8_then_scaled_mm` 替换为使用 `SimMXFP8MM`：

```python
def _patch_mxfp8_for_meta():
    if not _is_meta_simulation:
        return
    try:
        import torchao.prototype.mx_formats.mx_linear as mx_linear_mod
        mx_linear_mod._to_mxfp8_then_scaled_mm = lambda *a, **kw: SimMXFP8MM.apply(*a)
    except ImportError:
        pass
    try:
        import torchao.prototype.moe_training.mxfp8_grouped_mm as grouped_mm_mod
        grouped_mm_mod._to_mxfp8_then_scaled_grouped_mm = lambda *a, **kw: SimMXFP8GroupedMM.apply(*a)
    except ImportError:
        pass
```

#### 3.2.4 安装 torchao

容器中需要安装 `torchao`。已安装 `torchao==0.17.0`。

### 3.3 捕获的算子

启用 MXFP8 后，simulator 捕获的 L0 算子将变化：

| 位置 | 不启用 MXFP8 | 启用 MXFP8 |
|------|-------------|-----------|
| Linear forward | `aten.mm.default` | `npu.npu_dynamic_mx_quant.default` × 2 + `npu.npu_quant_matmul.default` |
| Linear backward | `aten.mm.default` (dx, dw) | `npu.npu_dynamic_mx_quant.default` × 2 + `npu.npu_quant_matmul.default` (dx, dw) |
| MoE forward | `npu.npu_grouped_matmul.default` | `npu.npu_dynamic_mx_quant.default` × 2 + `npu.npu_grouped_matmul.default` |
| MoE backward | `npu.npu_grouped_matmul.default` | 同上 |

### 3.4 通信量变化

MXFP8 将 matmul 的输入从 BF16（2 bytes/element）降为 FP8（1 byte/element），通信量减半：
- FSDP allgather 的参数量减半（FP8 vs BF16）
- 但量化 scale 额外传输（每 32 元素 1 byte scale，约 3% 额外开销）

## 4. 实施计划

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `meta_env.py` | patch `has_mx_capability` 在 meta 模式下返回 True |
| 2 | `hardware_shims/mxfp8_shim.py` | 创建 `SimMXFP8MM` 和 `SimMXFP8GroupedMM` |
| 3 | `meta_env.py` | patch `_to_mxfp8_then_scaled_mm` 和 `_to_mxfp8_then_scaled_grouped_mm` 使用 shim |
| 4 | `config_registry.py` | 添加 MXFP8 仿真配置 |
| 5 | 测试 | 验证捕获的算子包含 `npu_dynamic_mx_quant` 和 `npu_quant_matmul` |

## 5. 验证标准

1. **MXFP8Converter 初始化成功**：不报硬件检查错误
2. **捕获的 L0 算子变化**：Linear 层的 `aten.mm` 被替换为 `npu.npu_dynamic_mx_quant` + `npu.npu_quant_matmul`
3. **MoE 专家层**：`npu.npu_grouped_matmul` 前有 `npu.npu_dynamic_mx_quant` 量化算子
4. **shape 正确**：matmul 输出 shape 与不启用 MXFP8 时一致
5. **backward 正确**：dx 和 dw 的 shape 正确
