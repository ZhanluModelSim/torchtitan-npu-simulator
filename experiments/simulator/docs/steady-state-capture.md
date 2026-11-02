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

正式捕获开始时登记真实 optimizer state；结束时按实际本地 storage 清单重分类为 optimizer_state，并按 storage 身份去重。已有状态不会因初始输入和最终清单各出现一次而重复计数。卸载仍从正式静态预算排除状态，原始状态量保留；逐层激活不混入预热保存值。

不按 `zeros`、`empty` 或 `fill` 名称删除节点：正常每轮的 MoE 计数清理必须保留。Meta 捕获和逻辑存储预算不证明真实设备数值或 allocator 峰值。

## 验证入口

`tests/test_warmup.py` 覆盖实际 GA 次数、异常清理、同进程捕获、原生 AdamW 状态初始化后的字节去重，以及一轮和两轮预热后图与峰值一致。`tests/test_step.py` 保留低层首次捕获及状态投影回归。完整模型、并行、AC 和卸载组合仍须通过真实 Meta 系统验证；不以小模型单测替代这些验证。

本轮在 `ttsim` 的工作区 editable 环境验证：warmup、step、runtime 与 memory_delivery 完整组最终 38 项通过；旧冷启动峰值断言已更新为「raw 已含状态」，同时保留状态字节的独立手算和 CSV 时间线回放校验。Kimi 捕获文件 13 项通过，包含单卡、FSDP 首末 rank、HSDP、full/selective AC 和精度配置。Ruff 修改文件检查及 sidecar pyrefly 通过。

交付验证将主包和 sidecar wheel 普通安装至独立 `/tmp/steady-wheel-env-20261102`，两包实际导入均来自该环境 `site-packages`；在源码目录外运行 warmup 与 loss 核心测试 8 项通过，并完成 Kimi `world_size=2`、`seq_len=16` 的真实 Meta CLI 捕获及显存导出。未执行真实 NPU 数值或性能验证；全仓 pre-commit 未运行，不能将局部 Ruff 写为全仓 Lint 通过。


## 预热耗时测量（2026-11-02）

在ttsim容器、Intel Core i7-14700/WSL2、torch 2.14.0+cpu环境中，四组配置各以独立进程串行运行3次，使用time.perf_counter测量实际墙钟时间。共同配置为逻辑world_size=4、DP replicate2/shard2、EP2、MBS1、seq_len128、selective AC，无MTP或卸载；GBS=4×GA。下表各列为三次测量的中位数，占比先逐次计算再取中位数。

| 模型预设 | GA | 预热秒数 | simulate总秒数 | 预热占比 | 整个进程秒数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| deepseek_v4_smoketest | 1 | 0.810 | 5.599 | 14.5% | 7.944 |
| deepseek_v4_smoketest | 2 | 0.984 | 5.772 | 17.0% | 8.136 |
| deepseek_v4_smoketest | 4 | 1.309 | 6.087 | 21.5% | 8.422 |
| kimi_k3_smoketest | 2 | 0.855 | 4.872 | 17.7% | 7.221 |

simulate总耗时包括构建、预热、正式捕获、内存/通信视图处理和默认原生显存摘要导出，不含模块导入及平台预测/DES/结果导出。整个进程额外包含Python启动、导入和退出，此分母下预热占比约10%～15%。DSV4 GA2预热内部F/B约0.348秒、clip约0.169秒、optimizer及首次状态初始化约0.467秒。

这是负载生成器自身的CPU墙钟成本，不是被仿真训练的硬件耗时。预热直接耗时也不等于相对旧实现的净增量，因为首次初始化由正式捕获移到了预热；本次没有对旧实现做净开销A/B比较。结果仅覆盖所列小模型与环境，不作为完整模型、其他GA或机器的性能承诺。

测量通过外部包装runner.warmup_training_iteration、run_simulation_step、export_result计时，未增加生产计时开关。原始12份结果及measure.py/run.py/summarize.py保存在工作区runs/warmup-timing-20261102/，大体量运行产物不入库。复测时固定相同配置，每次新建进程，串行运行，分别记录预热/完整simulate/进程墙钟，保留实际导入路径与组件版本。
