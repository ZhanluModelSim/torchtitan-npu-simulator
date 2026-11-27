# FSDP 通信缓冲的计算流内存复用

将 FSDP 通信缓冲切换到计算流复用缓存，减少多流各自保留大块空闲显存导致的 Reserved 增长。默认关闭。

## 启用方式

在 worker 启动、首次导入 `torchtitan_npu` 前设置：

```bash
export TORCHTITAN_NPU_FSDP_ALLOC_ON_COMPUTE_STREAM=1
```

- 仅值 `1` 表示开启，在首次导入时读取；修改配置需重启进程。
- 开启后若 `MULTI_STREAM_MEMORY_REUSE` 未设置或为空，会自动默认设为 `2`（与 `MULTI_STREAM_MEMORY_RESERVE` 不同）；用户已显式设置的非空值会被保留。

## 实现概要

- **AllGather（AG）**：为每个 FSDP 参数组启用 PyTorch 原生异步解分片（`FSDPParamGroup.unshard_async_op`，对应上游 [PyTorch #202343](https://github.com/pytorch/pytorch/pull/202343)），缓冲申请、类型转换与输入拷贝在当前计算流执行。
- **ReduceScatter（RS）**：将 `foreach_reduce` 输出缓冲的申请切至计算流，保留原生归约流，并通过 `wait_stream` 与 `record_stream` 维护跨流依赖（对应上游 [PyTorch #206728](https://github.com/pytorch/pytorch/issues/206728) / [PyTorch #206795](https://github.com/pytorch/pytorch/pull/206795)）；目标设备为 CPU 或 HSDP 梯度累积未同步步（`all_reduce_grads=False`）时跳过包装，透传原生实现。

## 注意事项与限制

- **前向预取**：开关不自动构建预取链。基础 DeepSeek-V4/V4.1 的 EP>1 路径已有显式预取；EP=1 或自定义模型需通过 `set_modules_to_forward_prefetch` 配置下一层以实现通信计算重叠。
- **混合精度 cast 缓冲**：当前插件 patch 仅切换 `reduce_scatter_comm.allocate` 申请的 `reduce_output`；当 `reduce_dtype != sharded_grad_dtype` 时，post-reduce 阶段的类型转换缓冲仍留在归约流上，待上游 [PyTorch #206795](https://github.com/pytorch/pytorch/pull/206795) 合入并升级依赖后完整覆盖。
- **CPU offload**：走 `grad_accum` CPU-canonical 归约路径的参数组不经过 RS 计算流分配器（AG 仍生效）；完整 CPU offload 组合未纳入验证。
