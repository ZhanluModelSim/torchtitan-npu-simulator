# all_to_all 通信捕获缺失分析

> 状态：分析完成，待实现
> 分支：feat/alltoall-capture
> 日期：2026-07-09

## 1. 问题现象

用户在实际 NPU profiling 中观察到 `all_to_all` 算子，但 simulator 的捕获输出中没有 `all_to_all` 通信事件。`all_to_all` 出现在两个位置：

1. **MoE EP（Expert Parallel）**：MoE dispatch/combine 中的 token all-to-all 交换
2. **Attention CP（Context Parallel）**：Ulysses CP 的 head/sequence all-to-all 交换

## 2. 根因分析

### 2.1 拦截器已存在

`comm_events.py` 已经拦截了以下 all_to_all API：

| API | 拦截位置 | 状态 |
|-----|---------|------|
| `dist.all_to_all_single` | `patched_all_to_all_single` | ✅ 已拦截 |
| `dist.all_to_all` | `patched_all_to_all` | ✅ 已拦截 |
| `funcol.all_to_all_single` | `patched_funcol_all_to_all_single` | ✅ 已拦截 |
| `funcol.all_to_all_single_autograd` | `patched_funcol_all_to_all_single_autograd` | ✅ 已拦截 |

拦截器本身没有问题。问题在于 **all_to_all 调用根本没有被执行**——在 simulator 环境下，代码在到达 all_to_all 之前就被短路了。

### 2.2 MoE EP 的 all_to_all 被短路

**调用路径**：`NpuExpertParallel._token_dispatch` → `all_to_all_single_autograd`

**短路位置 1**：`moe_dispatch.py` 第 258-259 行

```python
is_fake = is_fake_process_group(device_mesh.get_group())
if not is_fake:
    # perform all-to-all  ← 只在非 fake PG 时执行
    routed_input = all_to_all_single_autograd(...)
```

`is_fake_process_group` 检查 `str(dist.get_backend(group)).lower() == "fake"`。在 simulator 中：
- `fake_backend` 模式：EP 子组的 PG 是 FakeProcessGroup → `is_fake=True` → **跳过 all_to_all**
- `multi_proc_meta` 模式：EP 子组的 PG 也是 FakeProcessGroup → `is_fake=True` → **跳过 all_to_all**

**短路位置 2**：`expert_parallel.py` 第 138-139 行（torchtitan 原始 ExpertParallel 的 patch）

```python
def _expert_parallel_token_dispatch(self, mod, inputs, device_mesh):
    if not is_fake_process_group(device_mesh.get_group()):
        return _ORIG_EXPERT_TOKEN_DISPATCH(self, mod, inputs, device_mesh)
    # fake PG: 直接做本地 permute，跳过 all_to_all
```

**短路位置 3**：`meta_env.py` 的 `_patch_moe_dispatch_to_avoid_meta_tensor_value_reads`（第 394-395 行）

```python
def _meta_safe_token_dispatch(self, mod, inputs, device_mesh):
    group = device_mesh.get_group()
    if not is_fake_process_group(group):
        return original_token_dispatch(self, mod, inputs, device_mesh)
    # fake PG: 解析式计算 split sizes，跳过 all_to_all
```

三层短路，all_to_all 永远不会被执行。

### 2.3 Attention CP 的 all_to_all 被短路

**调用路径**：`UlyssesCP._pre_hook` → `all_to_all`（`dist.all_to_all`）

**短路原因**：DeepSeek-V4 使用 `CompressorAttentionCP`（不是 Ulysses CP），所以 Ulysses 的 `AllToAll` 不会被调用。但如果模型配置使用 Ulysses CP，`dist.all_to_all` 会被拦截器捕获。

**实际情况**：DeepSeek-V4 的 CP 使用 `CompressorAttentionCP`（P2P + allgather），不使用 Ulysses（all_to_all）。所以 attention 中不会出现 all_to_all。

但如果其他模型使用 Ulysses CP，`dist.all_to_all` 会被 `patched_all_to_all` 拦截器捕获——**前提是它被真实调用**。Ulysses 的 `AllToAll.forward` 直接调用 `dist.all_to_all`，没有 `is_fake` 检查，所以会被拦截。

### 2.4 总结

| all_to_all 来源 | 被短路？ | 原因 |
|-----------------|---------|------|
| MoE dispatch (`all_to_all_single_autograd`) | ✅ 是 | `is_fake_process_group` 检查跳过 |
| MoE combine (`all_to_all_single_autograd`) | ✅ 是 | `is_fake_process_group` 检查跳过 |
| Ulysses CP (`dist.all_to_all`) | ❌ 否 | 无 `is_fake` 检查，会被拦截器捕获 |

**核心问题**：MoE 的 all_to_all 被 `is_fake_process_group` 检查短路，因为 fake PG 下 all_to_all 无法正确执行（split sizes 依赖真实 token 分布数据，meta tensor 无数据）。

## 3. 方案设计

### 3.1 目标

在不破坏 meta simulation 正确性的前提下，捕获 MoE all_to_all 通信事件。

### 3.2 方案：短路前记录通信事件

在 `is_fake` 短路分支中，**在跳过 all_to_all 之前**，调用 `_record_comm` 记录通信事件。这样 all_to_all 虽然不真实执行，但通信事件被忠实记录。

**关键点**：这与 CP P2P 的处理方式一致——`_WindowExchange` 的 `_meta_safe_forward` 也是在短路 P2P 前记录通信事件。

### 3.3 具体改动

#### 3.3.1 `moe_dispatch.py` 的 `_token_dispatch`

在 `is_fake` 分支中，all_to_all 被跳过前记录通信：

```python
is_fake = is_fake_process_group(device_mesh.get_group())
if not is_fake:
    # perform all-to-all
    routed_input = all_to_all_single_autograd(...)
else:
    # Record the all_to_all comm event before short-circuiting
    from torchtitan_npu.simulator.capture.comm_events import get_active_recorder, _record_comm
    recorder = get_active_recorder()
    if recorder is not None:
        _record_comm(recorder, "all_to_all", device_mesh.get_group(), routed_input)
    # ... existing fake-mode logic ...
```

同样在 `_token_combine` 中：

```python
if is_fake_process_group(device_mesh.get_group()):
    # Record all_to_all before short-circuit
    recorder = get_active_recorder()
    if recorder is not None:
        _record_comm(recorder, "all_to_all", device_mesh.get_group(), routed_output)
    return routed_output
```

#### 3.3.2 `expert_parallel.py` 的 patch

在 `_expert_parallel_token_dispatch` 和 `_expert_parallel_token_combine` 的 fake 分支中同样记录通信事件。

#### 3.3.3 `meta_env.py` 的 `_meta_safe_token_dispatch`

在 `_meta_safe_token_dispatch` 的 fake 分支中记录 all_to_all 通信事件。

#### 3.3.4 设置 `_comm_layer`

MoE all_to_all 属于 **L2**（框架调度层，EP 维度的 token 交换），不是 L1（模型计算内部）。需要在记录时设置 `_comm_layer = "L2"`。

但 MoE dispatch 的 `_token_dispatch` 是通过 `ExpertParallel` 的 forward pre_hook 调用的，它在模型 forward 内部执行。从调用路径看，它更像 L1（模型计算的一部分）。

**判定**：MoE all_to_all 是 EP 维度的通信，发生在模型 forward 内部（`ExpertParallel._token_dispatch` 是 forward pre_hook），但它的语义是"跨 EP rank 的 token 交换"——这是并行维度的通信，不是 attention 计算的一部分。

**建议**：将 MoE all_to_all 归为 **L2**，因为它是 EP 并行维度的通信（类似 FSDP 的 unshard/reshard），不是单个 rank 内部的计算。

### 3.4 Ulysses CP 的 all_to_all

如果模型使用 Ulysses CP，`dist.all_to_all` 会被 `patched_all_to_all` 拦截器捕获。Ulysses 的 all_to_all 是 attention 计算的一部分（类似 CP P2P），应归为 **L1**。

需要在 `UlyssesCP` 的 `AllToAll.forward` 中设置 `_comm_layer = "L1"`。

### 3.5 split sizes 的处理

MoE all_to_all 的 split sizes 依赖真实 token 分布。在 meta simulation 下，`moe_force_load_balance=True` 使路由均匀，split sizes 可以解析计算。记录通信事件时，`tensor_shape` 使用 `routed_input.shape`（meta tensor 的 shape 是正确的），`volume_bytes` 基于该 shape 计算。

## 4. 实施计划

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `moe_dispatch.py` | 在 `_token_dispatch` 和 `_token_combine` 的 `is_fake` 分支中，跳过 all_to_all 前记录通信事件 |
| 2 | `expert_parallel.py` | 在 `_expert_parallel_token_dispatch` 和 `_expert_parallel_token_combine` 的 fake 分支中记录通信事件 |
| 3 | `meta_env.py` | 在 `_meta_safe_token_dispatch` 中记录 all_to_all 通信事件；设置 `_comm_layer="L2"` |
| 4 | `ulysses_cp.py` | 在 `AllToAll.forward` 中设置 `_comm_layer="L1"` |
| 5 | 测试 | 验证 MoE all_to_all 出现在 L2 DataPass 中 |

## 5. 验证标准

1. **MoE all_to_all 出现在 L2**：`data_passes` 中有 `all_to_all` 类型的 DataPass
2. **MoE all_to_all 的 comm_layer="L2"**：属于 EP 维度的框架通信
3. **Ulysses all_to_all 出现在 L1**（如果使用 Ulysses CP）：属于 attention 计算
4. **all_to_all 的 tensor_shape 正确**：基于 meta tensor 的 shape
5. **all_to_all 的 src_exit_op/dst_entry_op 连接到 L1 算子**
