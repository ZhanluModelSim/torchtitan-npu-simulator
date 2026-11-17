# 结构预热后的稳态捕获

`simulate()` 和 CLI 默认在同一个 Meta 模型及 optimizer 实例上完成一次结构预热，再捕获 F/B/O，避免把优化器状态的首次初始化当作每次训练更新的成本。没有新增用户配置；低层 `run_simulation_step()` 继续只捕获调用者提供的一步。

## 执行顺序

1. 构建原生模型、并行、loss、optimizer、scheduler 和输入，保留必要的 Meta runtime、逻辑进程组及均衡路由兼容。
2. 清梯度，按实际 GA 执行 F/B，使用含 GA 的原 loss 分母，然后 clip 和 optimizer 更新。预热使用同一份合成输入，结束后再次清梯度。
3. 预热不推进 scheduler，不改变用户 `training.steps` 或 LR 配置。optimizer 的状态及 step 计数正常推进。这是结构预热，不声称数值等价的第二训练步。
4. 安装 FSDP 驻留、checkpoint、dispatch、通信和保存张量观察器，建立 buffer 校验基线，正式捕获一个 microbatch 的 F/B 和一次 O；正式 scheduler 正常推进一次。
5. 仅对正式图执行 FSDP 通信视图投影、激活与优化器卸载变换，再交付和导出。

实现位于 `runner.py` 与 `warmup.py`。预热不保留事件、loss 或激活列表，正常和异常退出均清梯度及恢复阶段上下文；异常继续向上抛出，由外层 runtime 恢复会话。原生模型及 optimizer 的更新不回滚。预热后不清空原生缓存或重建模型，以免恢复首用行为。

## 交付与内存

原 `annotations["capture"]` 字段保留，并新增：

- `mode="post_warmup"`：正式图来自结构预热之后。
- `warmup_iterations=1`：执行一次结构预热。
- `warmup_microbatches`：预热实际 GA。
- `warmup_scheduler_advanced=false`：预热未推进 LR scheduler。

`batch_configuration.memory_scope` 标明预热后的单 microbatch 静态预算。预热执行完整 GA 不表示正式图或显存预算包含多 microbatch 回放。

正式捕获开始时登记真实 optimizer state；结束时按实际本地 storage 清单重分类为 optimizer_state，并按 storage 身份去重。已有状态不会因初始输入和最终清单各出现一次而重复计数。 Muon runtime 在 warmup 后持有的 local／双通信 slot 工作缓冲也登记为初始驻留，覆盖 F/B/O；它们与 optimizer state 分开，不随状态卸载排除。预热消除初始化计算，初始登记保留已分配容量，二者不能互相替代。卸载仍从正式静态预算排除状态，原始状态量保留；逐层激活不混入预热保存值。

不按 `zeros`、`empty` 或 `fill` 名称删除节点：正常每轮的 MoE 计数清理必须保留。Meta 捕获和逻辑存储预算不证明真实设备数值或 allocator 峰值。

## 验证入口

`tests/test_warmup.py` 验证预热顺序、scheduler 与异常清理。`tests/test_optimizer_contract.py` 和 `tests/memory/test_optimizer_state.py` 检查状态清单；真实模型与显存交付见 `tests/test_memory_delivery.py`、`tests/test_kimi_capture.py` 和 `tests/test_dsv4_capture.py`。预热观察器必须区分无捕获阶段，不能假定 get_active_capture() 始终存在。

## 预热开销的测量方法

在固定软件栈和模型配置下，以 `time.perf_counter` 分别测量 `warmup_training_iteration`、完整 `simulate` 以及含导入的进程墙钟。独立进程串行重复运行，记录中位数及逐次占比，区分冷启动与编译缓存。默认 simulate 不导出文件；显式 mem 的 I/O 开销应单独说明。

这些时间是负载生成器自身的 CPU 开销，不是仿真训练硬件时延。预热直接耗时也不等于相对无预热实现的净增量：首次状态初始化从正式捕获移到了预热。比较净开销必须使用相同模型、输入、依赖和导出设置做 A/B，不能把历史小模型结果推广到完整模型或其他机器。
