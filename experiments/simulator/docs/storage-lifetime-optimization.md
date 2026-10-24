# Storage 生命周期检测优化

正式捕获默认使用「弱引用通知候选＋原生过期确认」，替代每次 dispatch 前的全量 storage 扫描。完整 Flash 配置实测从原实现的 71.2 秒降到正式版 20.7 秒；保留全部检查边界和原生过期判定，不新增配置、扩展编译或 CANN kernel。

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

这样，仍存活的权重、优化器状态和保存激活无需在每个算子前反复确认。完整 Flash 中，原来约 3.38 亿次过期检查减少为 27,893 次候选检查；所有 74,938 个检查边界仍然保留。

正式实现的唯一注册调用点直接传入已有的 `storage`，不使用 `_new_with_weak_ptr` 私有 API。前期原型通过进程内替换验证算法，以下保留其参考扫描与所有权探针记录。

## 性能与对照

环境为 4 vCPU、约 3.8 GiB 的 ARM64 虚拟机；Python 3.12.14、torch 2.14.0+cpu，已有 CANN 扩展编译缓存。使用完整 `deepseek_v4_flash_baseline_bf16`、S=512、MTP=1、world=16／logical rank=15、DP replicate/shard=2/2、TP/CP/EP=2/2/2、AC none；不写导出文件。

| 项目 | 原实现 | 候选队列原型 |
| --- | ---: | ---: |
| 完整 `simulate()` | 71.20 秒 | 21.32 秒 |
| 前反向及捕获 | 50.51 秒 | 5.34 秒 |
| 优化器及捕获 | 13.97 秒 | 9.75 秒 |
| 最终图节点数 | 75,968 | 75,968 |
| 梯度参数数量 | 1,316 | 1,316 |

计时是本轮单次配对测量，不代表所有模型或机器的固定加速比。

正式版复验：完整 `simulate()` 20.69 秒，前反向 5.09 秒、优化器 9.51 秒。节点数、原始 tensor 事件摘要、79,763 条生命周期记录、六项内存汇总与原实现一致。正式实现的完整 Flash 逐边界参考扫描执行 74,938 次、检查 337,756,038 个原生活跃引用，差异为 0；参考扫描耗时不计入上述性能结果。机器结果见 [正式验证快照](storage-lifetime-optimization-results.json)。

独立原实现与原型之间检查了：

- 原始 tensor 事件的算子、阶段、输入输出 shape／dtype／storage bytes 摘要一致。
- 79,763 条生命周期记录的 birth／death、原始与建模字节、kind、shape、dtype、consumer 序号多重集逐项一致。
- 六项显存汇总、节点数量和完整捕获校验结果一致。

比较不使用 Python 地址或 storage 地址作为跨进程 ID。最初按地址首次出现顺序重新编号的摘要不一致，因此改为直接比较上述生命周期多重集；原始地址摘要不作为等价性依据。

## 逐边界参考检查

参考模式在同一次执行的每个原检查边界运行旧全量扫描，将其实际过期集合与候选队列的结果直接比较，再按同一序号规则记录释放。此模式仅用于正确性审计，不用于性能计时。

| 配置 | 检查边界数 | 全量过期检查数 | 候选过期检查数 | 集合差异 |
| --- | ---: | ---: | ---: | ---: |
| smoke | 4,501 | 1,896,816 | 1,845 | 0 |
| HSDP | 7,220 | 3,829,280 | 2,258 | 0 |
| TP/CP/EP 组合并行 | 9,334 | 5,444,972 | 3,582 | 0 |
| HSDP＋Selective AC＋MXFP8 | 11,180 | 4,046,149 | 3,926 | 0 |
| 完整 Flash | 74,938 | 337,756,038 | 27,893 | 0 |

另有 20 项所有权探针，覆盖 CPU／Meta 的 view、显式 storage 持有、C++ autograd 保存值、DLPack capsule 的 C++ 持有、GC 循环、提前通知和跨线程释放。1,000 次 generation 注册中实际出现 998 次地址复用，释放记录仍正确。

## 回归与兼容边界

正式替换只修改 storage 注册和释放观测层，保持 tensor 身份、依赖图、saved-tensor hooks 和内存估算公式。新增 [storage 所有权单测](../tests/capture/test_storage_lifetimes.py) 与 [四配置逐边界参考测试](../tests/test_storage_lifetime_capture.py)，覆盖 HSDP、Full AC、Selective AC＋MXFP8，以及 TP／CP／EP 组合并行。

完整 simulator 套件 455 passed，独立 Pyrefly 0 errors，增量静态审查未发现必改问题。改动不参与训练数值计算，按精度对齐指南第 3 节的非数值改动边界验证图、梯度清单与内存事实；没有将 Meta 结果作为真实 NPU loss／grad_norm 比特一致证明。

storage 不支持 Python 弱引用时，正式版将该 storage 常驻待查集合，在每个原边界检查原生引用，并发出一次 warning。提前通知继续等待原生过期，回调不强引用 tensor／storage。目标 PyTorch／Python 升级后需重跑所有权与参考测试。

降低扫描频率会改变释放边界，不采用。缓存 Python 属性查询仍保留全量扫描的规模问题。C++ 批量扫描可以作为后备方向，但本轮未实现；候选队列已获得明显收益，且不增加编译和扩展维护成本。

完整原型、脚本、日志与机器结果位于本机 `torchtitan-migration-audit-20261020/storage-lifetime-exploration-20261023/`：`weakqueue.py`、`experiment.py`、`ownership_probes.py`、`results.json` 和各配置结果。以上为固定软件栈下的 CPU／Meta 逻辑验证，不是实际 NPU 数值或 allocator 峰值证明。

正式实现的复验记录位于 `torchtitan-migration-audit-20261020/storage-lifetime-production-20261024/`。完整 pre-commit 的结果与根仓未改代码的既有类型告警单独记录，不将 simulator 的 0 errors 表述为全仓类型检查通过。

2026-10-24，用户明确批准「允许本次提交，保留告警记录」，仅豁免本次根仓既有类型检查失败。该检查在 24 个本次未修改的文件中报告 65 条告警；simulator 的 455 项回归和独立类型检查均已通过。未关闭检查或修改其通过条件。
