# EP 通信域分离

本特性为 Ascend NPU 上的 MoE 专家并行（Expert Parallel，EP）创建独立的
HCCL process group，同时保留 dense、`dp_replicate` 和 `efsdp` 使用的原有通信组。
特性在导入 `torchtitan_npu` 时自动生效，不增加配置字段；仅对 EP size 大于 1
且 process group 不是 `fake` backend 的运行生效。

## 动机

torchtitan 的 `ParallelDims.build_mesh()` 在同一个 sparse mesh 构建流程中创建
`dp_replicate`、`efsdp` 和 `ep` 轴。在部分并行拓扑下，FSDP 与 EP 会复用同一个
process group。FSDP 的 all-gather 和 reduce-scatter 会阻塞 EP 的 all-to-all，使两类通信无法充分重叠，
通信争用可能影响整网训练性能。

## 适用范围

`build_mesh()` 的完整 sparse mesh 轴为
`("pp", "dp_replicate", "efsdp", "ep")`。`spmd_types` 注册时使用其子 mesh
`("dp_replicate", "efsdp", "ep")`，`pp` 由 pipeline parallel 路径单独管理。
EP 通信域分离只替换 `ep` 轴对应的 process group，其他轴、mesh layout、rank map
和 process group 的 ranks 保持不变。

| 运行路径 | EP size 大于 1 | EP size 为 1 或 `fake` backend |
| --- | --- | --- |
| `spmd_types` | 注册 sparse mesh 时替换 `ep` 轴；`maybe_set_sparse_mesh()` 继续使用真实 `DeviceMesh` | 返回原 mesh，不创建新 process group |
| `default` / `full_dtensor` | 通过 `BaseEPTokenDispatcher.wire_meshes()` 将独立 EP mesh 传给 dispatcher | 返回原 mesh，不创建新 process group |
| DeepEP | 继承上述 dispatcher wiring；按 `ElasticBuffer` 容量配置，显式调用 `get_deepep_group()` | 不创建独立 group |

## 通信域关系

| 状态 | FSDP/EFSDP | EP | PG 数量 |
| --- | --- | --- | --- |
| 分离前 | 原 process group | 与 FSDP/EFSDP 共用 | 1 |
| 分离后 | 原 process group | 独立 process group | 2 |

独立 EP process group 与原 process group 使用相同 ranks，因此不改变专家分片或 token 路由，
只改变 EP collective 和通信 buffer 使用的 process group。

## 实现原理

### 1. 替换 sparse mesh 的 EP 轴

`_patch_sparse_mesh_registration()` 包装
`torchtitan.distributed.spmd_types.set_spmd_meshes()`。注册 sparse mesh 时：

1. `_replace_sparse_ep_mesh()` 检查 mesh 是否包含 `ep` 轴，并取出 `sparse_mesh["ep"]`。
2. `_separate_ep_mesh()` 对 EP size、`fake` backend 和缓存进行检查。
3. `_create_separate_ep_mesh()` 调用 `DeviceMesh._unflatten()` 创建一维
   `("ep",)` mesh，并通过 `ProcessGroupHCCL.Options()` 指定 HCCL backend。
4. 浅拷贝原 sparse mesh，仅替换 `_dim_group_names` 中 `ep` 轴的 process group 名称。
5. 通过 `_SPARSE_MESHES` 缓存替换后的 sparse mesh，通过 `_EP_MESHES` 缓存独立 EP mesh。

`_warm_spmd_mesh_axis()` 会对新 process group 调用 `spmd.MeshAxis.of()`，使它在
`torch.compile` 观察到前完成 `spmd_types` 的 mesh axis 注册。

### 2. 适配 dispatcher wiring

`_patch_base_dispatcher_wiring()` 包装
`BaseEPTokenDispatcher.wire_meshes()`。所有继承该基类并直接读取
`self.ep_mesh` 的 dispatcher 都会收到独立 EP mesh，包括标准 all-to-all、DeepEP、
HybridEP 和 MinimalAsyncEP 路径。

若 dispatcher 设置 `_uses_custom_ep_process_group = True`，wrapper 会保留该
dispatcher 自己的 EP mesh，不再创建通用的独立 mesh。

### 3. DeepEP 专用 group

`get_deepep_group()` 为需要按 buffer 容量配置通信组的 NPU DeepEP 调用方提供显式
接口：

1. 使用 `ElasticBuffer.get_moe_ep_ccl_buffer_size()` 根据 EP world size、
   `num_max_tokens_per_rank`、hidden dim、expert 数量和 `top_k` 计算容量。
2. 使用 `ProcessGroupHCCL.Options()` 设置
   `{"hccl_buffer_size": <capacity_mb>}`。
3. 通过 `dist.new_group(..., backend="hccl", group_desc="npu_deepep")` 创建
   HCCL process group。
4. 按源 EP group 的 `group_name` 缓存 group。后续调用复用同一 group；如果请求的
   buffer 容量更大，则报错，要求共享同一 EP group 的 dispatcher 使用共同的最大配置。

当前上游 `DeepEPTokenDispatcher` 通过 `BaseEPTokenDispatcher.wire_meshes()` 获取
独立 EP mesh；`get_deepep_group()` 是按 `ElasticBuffer` 需求创建专用 group 的显式
API，不会额外 patch dispatcher 的底层算子。
