# Storage 生命周期观测

捕获器使用弱引用候选队列减少无效扫描，但保留每个原生 dispatch 边界的释放检查。Python 弱引用回调只报告候选，最终是否释放仍由原生 storage weak reference 判断；不能降低检查频率来加速，否则会改变生命周期与峰值。

## 保持的正确性边界

当前记录的是逻辑 storage 生命周期，不是 allocator reserved、缓存或设备 workspace。原实现每次 dispatch 前检查所有活跃 `StorageWeakRef.expired()`，有释放时使用同一个序号边界记录 `release_seq`。

优化仍在每个原检查边界执行，不降低采样频率，也不以最后一次使用、tensor wrapper 销毁或 autograd 保存槽释放代替 storage 释放。

候选通知来自实际 untyped storage wrapper 的 Python 弱引用。该 wrapper 持有原生 storage 的所有权；当前固定 PyTorch 构建的实现见 [Storage.cpp](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/torch/csrc/Storage.cpp#L55-L72) 和 [PyObjectPreservation](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/torch/csrc/utils/pyobject_preservation.h)。通知只用于缩小待查集合，最终仍由原来的 C++ weak reference 判定是否过期。

## 正式机制

1. 每次注册新 storage generation 时，保留原生弱引用，同时注册 storage wrapper 的 Python 弱引用回调。
2. 回调只将 generation key 加入待查集合，不分配 Tensor、不记录释放序号、不修改计算图，也不持有 storage 的强引用。
3. 在原来的 dispatch 前检查点，仅对待查 key 调用原生 `expired()`。
4. 确认过期后，按原规则分配释放序号。若通知早于原生释放，保留候选，在后续原检查边界继续确认。
5. capture 关闭前执行最终检查，随后清理回调和候选引用。不同 capture 不共享队列。

## 验证方式与兼容边界

[所有权测试](../tests/capture/test_storage_lifetimes.py) 覆盖 CPU／Meta 的 view、显式 storage 持有、autograd、C++ 持有、GC、提前通知及 generation 复用。[逐边界参考测试](../tests/test_storage_lifetime_capture.py) 在每个检查点比较全量原生过期集合与候选队列结果，覆盖 HSDP、AC、MXFP8 及 TP／CP／EP；参考模式用于正确性审计，不用于性能计时。

跨进程比较使用 shape/dtype/storage 字节、birth/death、consumer 序号等生命周期多重集，不用 Python 地址或 storage 地址作为稳定 ID。还须比较图依赖、梯度清单及内存摘要，不能只看最终峰值一致。

storage 不支持 Python 弱引用时，将其常驻待查集合，每个原边界检查原生引用并发出一次 warning。提前通知继续等待原生过期；回调不强持有 tensor/storage。不同 capture 不共享候选队列。升级 PyTorch/Python 后重跑上述测试，不沿用历史耗时或检查数量作为正确性证据。
