# 优化器算子捕获问题分析

> 分支：feat/optimizer-ops
> 日期：2026-07-13

## 1. 问题

用户反馈：实际 NPU profiling 中看到 `npu_apply_adam_w`（或类似名称如 `applyAdamW_v2`）的 fused 优化器算子，但 simulator 输出的是 `aten._foreach_mul_`、`aten._foreach_add_`、`aten._foreach_lerp_`、`aten._foreach_addcmul_` 等标准 PyTorch foreach 算子。

## 2. 根因分析

### 2.1 真实 NPU 训练中的优化器路径

在真实 NPU 训练中，优化器有三种实现路径：

| 实现方式 | 算子名称 | 触发条件 |
|---------|---------|---------|
| `fused` | `torch._fused_adamw_` → NPU dispatch → `npu.npu_apply_adam_w` | `implementation="fused"` |
| `foreach` | `aten._foreach_mul_` / `aten._foreach_add_` 等 | `implementation="foreach"` |
| `for-loop` | `aten.mul` / `aten.add_` 等（逐参数循环） | `implementation="for-loop"` |

**真实训练使用 `fused` 路径**：`torch._fused_adamw_` 是一个 C++ op，在 NPU 设备上 dispatch 到 `npu.npu_apply_adam_w`（一个 fused kernel，在单次调用中完成整个 AdamW 步进）。

### 2.2 Simulator 中的优化器路径

Simulator 在 `trainer.py` 第 280-289 行强制将 `implementation` 从 `"fused"` 改为 `"foreach"`：

```python
if getattr(config.optimizer, "implementation", None) == "fused":
    # torch.optim's fused implementation validates the parameter
    # device against a hardcoded supported-device list (mps/cuda/
    # xpu/hpu/cpu/mtia/npu) that does not include "meta", raising
    # RuntimeError: fused=True requires all the params to be
    # floating point Tensors of supported devices
    config.optimizer.implementation = "foreach"
```

**原因**：`fused` 实现验证参数设备是否在支持的列表中（`cuda/mps/xpu/hpu/cpu/mtia/npu`），`"meta"` 不在列表中，会 raise `RuntimeError`。

**结果**：Simulator 使用 `foreach` 路径，捕获到的是 `aten._foreach_mul_` 等标准 PyTorch 算子，而不是 `npu.npu_apply_adam_w`。

### 2.3 为什么不能直接用 fused

`torch._fused_adamw_` 在 meta device 上：
1. 验证参数设备 → `"meta"` 不在支持列表 → raise
2. 即使绕过验证，`npu_apply_adam_w` 的 NPU meta kernel 可能未注册
3. `npu_apply_adam_w` 的 schema 需要 `grad` tensor（meta tensor 无数据）

### 2.4 torch_npu 的 fused 优化器

`torch_npu.optim` 提供了 `NpuFusedAdamW`（基于 `NpuFusedOptimizerBase`），但检查发现它**不使用** `npu_apply_adam_w` op。它是一个 Python 层面的 fused 实现，内部仍使用标准 `aten` 算子。

`npu_apply_adam_w` 是一个独立的 NPU C++ op，通过 `torch._fused_adamw_` 的 NPU dispatch key 触发——只有当参数在 NPU 设备上时才会走这个路径。

## 3. 方案

### 3.1 核心思路

与 MHC/SMLA/MXFP8 的处理方式一致：创建 **optimizer shim**，在 meta device 上用标准 PyTorch 算子模拟 AdamW 步进的 shape 行为，同时通过 `record_synthetic_op` 记录真实的 NPU 算子名 `npu.npu_apply_adam_w`。

### 3.2 具体设计

#### 3.2.1 不再强制改为 foreach

移除 `trainer.py` 中 `implementation = "foreach"` 的强制覆盖。改为在 meta 模式下 patch `torch._fused_adamw_` 使用 shim。

#### 3.2.2 创建 optimizer shim

创建 `torchtitan_npu/simulator/hardware_shims/optimizer_shim.py`：

```python
class SimFusedAdamW:
    """Meta-safe shadow of torch._fused_adamw_.
    
    Records npu.npu_apply_adam_w op name while using standard
    foreach ops for shape inference on meta tensors.
    """
    
    @staticmethod
    def apply(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
              state_steps, *, amsgrad, beta1, beta2, lr, 
              weight_decay, eps, maximize):
        from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
        cap = get_active_capture()
        
        if cap is not None:
            # Record the fused NPU op name
            # Use the first param as representative tensor
            rep_tensor = params[0] if params else torch.empty(1)
            cap.record_synthetic_op(
                "npu.npu_apply_adam_w.default",
                [rep_tensor],
                [rep_tensor],  # output shape = input shape
            )
        
        # Execute standard foreach AdamW for shape inference
        # (same as what foreach implementation does, but we record
        #  the fused op name instead of individual foreach ops)
        # ... standard AdamW math using torch._foreach_* ...
```

#### 3.2.3 Patch torch._fused_adamw_

在 `meta_env.py` 中 patch `torch._fused_adamw_`：

```python
def _patch_fused_adamw_for_meta():
    if not _is_meta_simulation:
        return
    import torch
    orig_fused = torch._fused_adamw_
    
    def _meta_safe_fused_adamw(params, grads, exp_avgs, exp_avg_sqs,
                               max_exp_avg_sqs, state_steps, **kwargs):
        if not _is_meta_simulation:
            return orig_fused(params, grads, ...)
        return SimFusedAdamW.apply(params, grads, ...)
    
    torch._fused_adamw_ = _meta_safe_fused_adamw
```

#### 3.2.4 不捕获 foreach 子算子

当使用 shim 时，`record_synthetic_op` 只记录 `npu.npu_apply_adam_w`，不记录内部的 `_foreach_mul_` 等子算子。这通过在 shim 内部设置 `_capture_l0 = False`（跳过子算子捕获），然后 `record_synthetic_op` 记录 fused op，最后恢复 `_capture_l0 = True` 实现。

### 3.3 捕获效果

| 配置 | 不启用 shim | 启用 shim |
|------|------------|-----------|
| optimizer ops | `aten._foreach_mul_` × N + `aten._foreach_add_` × N + ... | `npu.npu_apply_adam_w` × 1 (per param group) |
| 算子数量 | ~266 (大量 foreach 子算子) | ~5 (每个 param group 一个 fused op) |
| 算子名称 | 标准 PyTorch aten | NPU fused kernel |

### 3.4 注意事项

1. **`torch._fused_adamw_` 是 built-in method**，不是 Python 函数，不能直接 monkey-patch。需要通过 `torch.ops` 或 dispatcher 机制 patch。

2. **替代方案**：如果无法 patch `torch._fused_adamw_`，可以保持 `implementation="foreach"` 但在 optimizer step 前后记录 `npu.npu_apply_adam_w` 作为合成算子，同时跳过 foreach 子算子的捕获。

3. **swap_optimizer 和 virtual_optimizer** 路径也使用 `torch._fused_adamw_`（见 `swap_optimizer.py:576` 和 `virtual_optimizer.py:79`），shim 需要覆盖这些路径。

## 4. 实施计划

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `hardware_shims/optimizer_shim.py` | 创建 `SimFusedAdamW`，记录 `npu.npu_apply_adam_w` |
| 2 | `meta_env.py` | patch `torch._fused_adamw_` 使用 shim |
| 3 | `trainer.py` | 移除 `implementation = "foreach"` 强制覆盖 |
| 4 | 测试 | 验证 optimizer ops 为 `npu.npu_apply_adam_w` |

## 5. 验证标准

1. optimizer L0 ops 中出现 `npu.npu_apply_adam_w.default`
2. 不再出现 `aten._foreach_mul_` / `aten._foreach_add_` 等 foreach 子算子
3. 参数 shape 正确（meta tensor 的 shape 不变）
4. 通信量不变（FSDP allgather/reduce_scatter 不受影响）
