# all_to_all 通信捕获缺失分析

> 分支：feat/alltoall-capture
> 日期：2026-07-09

## 1. 问题

用户在实际 NPU profiling 中看到 `all_to_all` 算子，但 simulator 输出中没有。`all_to_all` 出现在 MoE 的 EP（Expert Parallel）token dispatch/combine 过程中。

## 2. all_to_all 在代码中的位置

### 2.1 MoE dispatch（forward，token 发往各 EP rank）

文件：`torchtitan_npu/converters/kernels/moe_dispatch.py`，`NpuExpertParallel._token_dispatch`（第 248 行）

```python
def _token_dispatch(self, mod, inputs, device_mesh):
    routed_input, num_tokens_per_expert, routed_scores = inputs
    ...
    is_fake = is_fake_process_group(device_mesh.get_group())  # ← 第 258 行
    if not is_fake:
        # ↓ 这两个 all_to_all 在 fake PG 下被跳过
        routed_input = all_to_all_single_autograd(routed_input, ...)   # 第 261 行
        routed_scores = all_to_all_single_autograd(routed_scores, ...) # 第 269 行
    ...
```

### 2.2 MoE combine（forward 结束，token 从各 EP rank 收回）

同文件，`NpuExpertParallel._token_combine`（第 314 行）

```python
def _token_combine(self, mod, routed_output, device_mesh):
    routed_output = NPUMoeTokenUnpermute.apply(...)
    if is_fake_process_group(device_mesh.get_group()):  # ← 第 318 行
        return routed_output  # ← fake PG 下直接返回，跳过 all_to_all
    routed_output = all_to_all_single_autograd(routed_output, ...)  # 第 321 行
    return routed_output
```

### 2.3 MoE combine backward（反向，梯度 all_to_all）

`all_to_all_single_autograd` 是一个 autograd Function，其 backward 也会执行 all_to_all。但由于 forward 被跳过，backward 也不会执行。

## 3. 为什么 all_to_all 没被捕获

### 3.1 拦截器已就位

`comm_events.py` 已经拦截了所有 all_to_all 变体：

```python
# comm_events.py 第 313-314 行
orig_all_to_all_single = dist.all_to_all_single
orig_all_to_all = dist.all_to_all

# 第 321 行
orig_funcol_all_to_all_single = funcol.all_to_all_single
# 第 324 行
orig_funcol_all_to_all_single_autograd = funcol.all_to_all_single_autograd
```

拦截器没有问题。

### 3.2 问题：all_to_all 调用被 `is_fake` 检查跳过

MoE 代码在调用 `all_to_all_single_autograd` 之前检查 `is_fake_process_group`。当 simulator 使用 fake PG（`fake_backend` 或 `multi_proc_meta` 模式下的 FakeProcessGroup 子组）时，`is_fake=True`，all_to_all 被跳过。

**执行链**：

```
模型 forward
  → MoE layer forward
    → NpuExpertParallel._token_dispatch (第 248 行)
      → is_fake = is_fake_process_group(group)  ← True
      → if not is_fake:  ← False，跳过
          all_to_all_single_autograd(...)  ← 永远不执行
      → 走 is_fake 分支：本地 permute 代替 all_to_all
```

all_to_all 调用根本没到达拦截器，所以不会被捕获。

### 3.3 三层短路

实际上有三层代码在 fake PG 下跳过 all_to_all：

**第一层**：`meta_env.py` 的 `_patch_moe_dispatch_to_avoid_meta_tensor_value_reads`（第 391 行）

```python
def _meta_safe_token_dispatch(self, mod, inputs, device_mesh):
    group = device_mesh.get_group()
    if not is_fake_process_group(group):       # ← fake PG 时为 True
        return original_token_dispatch(...)   # ← 不走这里
    # ↓ 走这里：解析式计算 split sizes，本地 permute，完全跳过 all_to_all
    ...
```

这个 patch 替换了 `NpuExpertParallel._token_dispatch`，所以原始的 `is_fake` 检查（第 258 行）根本不会被执行——`_meta_safe_token_dispatch` 在更早的层级就短路了。

**第二层**（如果第一层未生效）：`moe_dispatch.py` 原始代码的 `is_fake` 检查（第 258 行）

**第三层**（如果前两层未生效）：`expert_parallel.py` 的 patch（第 138 行）替换了 `ExpertParallel._token_dispatch`

在当前 simulator 中，第一层（`meta_env.py`）生效，因为 `patch_device_type_to_meta()` 在 `SimulationTrainer.__init__` 中调用，先于模型构建。

## 4. 为什么需要跳过 all_to_all

`all_to_all_single_autograd` 的 split sizes（`input_splits`/`output_splits`）依赖 `num_tokens_per_expert` 的**真实数值**——即每个 expert 实际收到了多少 token。在 meta device 上，tensor 没有数据，无法 `.tolist()` 读取 split sizes。

`_compute_all_to_all_splits`（第 213 行）尝试 `.to("cpu").tolist()` 读取 token 计数，这在 meta tensor 上会崩溃（`NotImplementedError: Cannot copy out of meta tensor`）。

因此 `meta_env.py` 的 patch 用 `moe_force_load_balance=True`（路由均匀）的前提，解析式计算 split sizes（`divmod(total_tokens, ep_degree)`），绕过了对真实数据的依赖。

## 5. 方案

### 5.1 核心思路

在 `_meta_safe_token_dispatch` 的 fake 分支中，**在跳过 all_to_all 之前**，调用 `_record_comm` 记录通信事件。这与 CP P2P（`_WindowExchange`）的处理方式完全一致——通信不真实执行，但通信事件被忠实记录。

### 5.2 EP all_to_all 属于 L1

EP 的 all_to_all 是 MoE forward 计算的一部分——token dispatch 和 combine 是 MoE 层的内部步骤，与 `aten.mm`（expert 计算）同级。它不是框架调度层的通信（如 PP P2P 或 FSDP unshard/reshard），而是模型计算图内部的通信。

因此 EP all_to_all 应归为 **L1**（`_comm_layer="L1"`），与 CP P2P/allgather 同级。

### 5.3 具体改动

#### 5.3.1 `meta_env.py`：`_meta_safe_token_dispatch` 中记录 dispatch all_to_all

```python
def _meta_safe_token_dispatch(self, mod, inputs, device_mesh):
    routed_input, num_tokens_per_expert, routed_scores = inputs
    group = device_mesh.get_group()
    if not is_fake_process_group(group):
        return original_token_dispatch(self, mod, inputs, device_mesh)

    # ★ 新增：在跳过 all_to_all 前，记录通信事件
    global _comm_layer
    _comm_layer = "L1"  # EP all_to_all 是 MoE 计算的一部分
    from torchtitan_npu.simulator.capture.comm_events import get_active_recorder, _record_comm
    recorder = get_active_recorder()
    if recorder is not None:
        # dispatch: 2 个 all_to_all（routed_input + routed_scores）
        _record_comm(recorder, "all_to_all", group, routed_input)
        if routed_scores is not None:
            _record_comm(recorder, "all_to_all", group, routed_scores)

    # 以下为现有的 fake 分支逻辑（解析式 split sizes + 本地 permute）
    ep_degree = device_mesh.shape[0]
    ...
```

#### 5.3.2 `meta_env.py`：`_token_combine` 中记录 combine all_to_all

`_token_combine` 目前没有被 `meta_env.py` patch（只有 `_token_dispatch` 被 patch）。需要在 `_meta_safe_token_dispatch` 返回的 result 中处理，或者额外 patch `_token_combine`。

由于 `_token_combine` 在 `NpuExpertParallel` 中有自己的 `is_fake` 检查（第 318 行），需要在该检查的 fake 分支中记录通信：

```python
# moe_dispatch.py 第 318 行，或通过 meta_env.py patch
if is_fake_process_group(device_mesh.get_group()):
    # ★ 新增：记录 combine all_to_all
    _comm_layer = "L1"
    recorder = get_active_recorder()
    if recorder is not None:
        _record_comm(recorder, "all_to_all", device_mesh.get_group(), routed_output)
    return routed_output
```

#### 5.3.3 backward 的 all_to_all

`all_to_all_single_autograd` 的 backward 也会执行 all_to_all。由于 forward 被跳过（没有创建 autograd 图节点），backward 不会执行。但 backward 的 all_to_all 通信也应该被记录。

**方案**：在 `_meta_safe_token_dispatch` 中记录 dispatch all_to_all 时，同时注册一个 pending backward comm event。当 backward 阶段执行到 MoE 层时，记录 combine 的 backward all_to_all。

> 更简单的方案：在 dispatch 和 combine 各记录一次 forward all_to_all，backward all_to_all 通过 autograd hook 在 MoE backward 时记录。

#### 5.3.4 Ulysses CP 的 all_to_all（如有）

如果模型使用 Ulysses CP（`ulysses_cp.py`），其 `AllToAll.forward` 直接调用 `dist.all_to_all`，没有 `is_fake` 检查，会被拦截器正常捕获。Ulysses all_to_all 是 attention 计算的一部分，归为 L1。

DeepSeek-V4 使用 `CompressorAttentionCP`（不是 Ulysses），所以当前不涉及。

## 6. 实施计划

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `meta_env.py` | 在 `_meta_safe_token_dispatch` 的 fake 分支中，跳过 all_to_all 前记录 dispatch all_to_all（`_comm_layer="L1"`） |
| 2 | `moe_dispatch.py` 或 `meta_env.py` | 在 `_token_combine` 的 fake 分支中记录 combine all_to_all |
| 3 | `meta_env.py` | patch MoE backward hook 记录 backward all_to_all |
| 4 | 测试 | 验证 all_to_all 出现在 L1 StepGraph 的 internal_data_passes 中 |

## 7. 验证标准

1. **dispatch all_to_all**：每个 MoE layer 的 forward 中有 2 个 all_to_all（routed_input + routed_scores）
2. **combine all_to_all**：每个 MoE layer 的 forward 结束时有 1 个 all_to_all
3. **backward all_to_all**：每个 MoE layer 的 backward 中有对应的 all_to_all
4. **comm_layer="L1"**：EP all_to_all 归属 L1 StepGraph
5. **tensor_shape 正确**：基于 meta tensor 的 shape
6. **src_exit_op/dst_entry_op 连接到 L1 算子**
