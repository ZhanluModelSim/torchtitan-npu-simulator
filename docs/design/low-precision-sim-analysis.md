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
| torchao 未安装 | 容器中无 torchao | patches 被跳过 |

## 3. 方案

### 3.1 核心思路

保留真实的 `NpuMXFP8MM` 和 `NpuMXFP8GroupedMM` autograd 实现。当前
`npu_dynamic_mx_quant`、`npu_quant_matmul` 和 `npu_grouped_matmul` 均已提供
meta kernel，可直接完成 shape 推导并被 dispatcher capture、selective AC policy
和 memory tracker 共同观察。Simulator 仅绕过无真实硬件时的 capability 检查。

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

#### 3.2.2 使用真实 meta kernel

`meta_env.py` 不替换 torchao 的 `_to_mxfp8_then_scaled_mm` 或
`_to_mxfp8_then_scaled_grouped_mm`。NPU patch 中的 autograd function 直接在
meta tensor 上执行，因此捕获结果与真实训练使用同一组 dispatcher op。

#### 3.2.3 Selective AC

DeepSeek V4 的 selective AC 扩展保存以下高计算量算子的输出：

- `aten._grouped_mm.default`
- `npu.npu_quant_matmul.default`
- `npu.npu_grouped_matmul.default`

量化算子仍然重计算，避免保存 FP8 tensor 和 scale 带来的额外驻留。

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
| 2 | `models/deepseek_v4/activation_checkpoint.py` | 扩展 selective AC 保存算子 |
| 3 | `config_registry.py` | 添加 MXFP8 仿真配置 |
| 4 | 测试 | 验证真实 NPU op 的捕获和重计算行为 |

## 5. 验证标准

1. **MXFP8Converter 初始化成功**：不报硬件检查错误
2. **捕获的 L0 算子变化**：Linear 层的 `aten.mm` 被替换为 `npu.npu_dynamic_mx_quant` + `npu.npu_quant_matmul`
3. **MoE 专家层**：`npu.npu_grouped_matmul` 前有 `npu.npu_dynamic_mx_quant` 量化算子
4. **shape 正确**：matmul 输出 shape 与不启用 MXFP8 时一致
5. **backward 正确**：dx 和 dw 的 shape 正确
6. **selective AC 正确**：三个高计算量 matmul 不出现在 recompute，量化算子仍在 recompute
