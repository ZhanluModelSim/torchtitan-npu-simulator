# Loss spike 检测与路由重放（Anticipatory Routing）

Anticipatory Routing 用于在 MoE 训练发生 loss spike 时，恢复异常起点之前的 checkpoint，
并通过预先录制、延迟重放专家 ID 执行一段恢复训练。该能力默认关闭，通过
`TrainerEx.Config.anticipatory` 对应的 CLI 参数启用，适用于 TorchTitan 0.3.0 的标准 NPU Trainer。

恢复过程复用上游 `train()` 和 `train_step()`，梯度累积、梯度裁剪、optimizer 和
LR scheduler 更新仍由上游完成。路由重放只固定专家 ID，gate scores 和梯度根据当前参数重新计算。

## 启动入口

使用 `torchrun -m torchtitan_npu.train` 启动，使 NPU 配置转换构造 `TrainerEx`。
`scripts/run_train.sh` 和 `scripts/run_train_multinodes.sh` 默认使用 NPU 入口。

使用单卡 DeepSeek-V4 调试脚本的示例：

```bash
HF_ASSETS_PATH="$PWD/tests/assets/deepseek_v3" \
CKPT_SAVE_LOAD_PATH="$PWD/outputs/anticipatory_run/checkpoint" \
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh \
  --training.steps 1000 \
  --checkpoint.enable --checkpoint.no-load-only --checkpoint.interval 50 \
  --anticipatory.enable \
  --anticipatory.delay-steps 8 \
  --anticipatory.active-steps 32
```

从仓库根目录运行；`tests/assets/deepseek_v3` 需包含已准备好的 tokenizer.json 和 tokenizer_config.json，
数据使用 `tests/assets/c4_test`。checkpoint 目录需可写，独立实验应使用新的目录。
该示例通过训练 loss 的趋势和波动自动检测异常。
启动脚本可能恢复现有 checkpoint；进行独立实验时应使用独立的输出目录。

## 配置参数


| CLI 参数                                    |  默认值 | 约束与作用                                                                     |
| ------------------------------------------- | ------: | ------------------------------------------------------------------------------ |
| `--anticipatory.enable`                     | `false` | 开启自动检测和回滚恢复。                                                       |
| `--anticipatory.delay-steps`                |    `16` | 非负整数；路由预取队列的目标深度，也是 WARMUP 的 step 数据量。                 |
| `--anticipatory.active-steps`               |   `500` | 正整数；ACTIVE 的 optimizer 更新次数，不包括 DRAIN。                           |
| `--anticipatory.max-rollbacks`              |     `3` | 非负整数；本次运行的回滚次数上限，`0` 表示仅检测、不执行回滚，跳过回滚相关的 checkpoint 校验。                         |
| `--anticipatory.index-store-dtype`          |  `auto` | 专家 ID 的存储类型：`auto`、`int16`、`int32`、`int64`；必须能表示全部专家 ID。 |
| `--anticipatory.detector.warmup-steps`      |   `100` | 至少为`2`；检测器开始判定异常前积累的统计观测次数。                            |
| `--anticipatory.detector.z-threshold`       |   `6.0` | 大于`0`；触发 spike 的标准化残差阈值。                                         |
| `--anticipatory.detector.level-decay`       |  `0.97` | 在`(0, 1)` 内；loss 水平和残差方差的衰减系数。                                 |
| `--anticipatory.detector.trend-decay`       |   `0.9` | 在`[0, 1)` 内；loss 趋势的衰减系数。                                           |
| `--anticipatory.detector.onset-z-threshold` |   `2.0` | 大于`0` 且不超过触发阈值；用于回溯异常起点。                                   |
| `--anticipatory.detector.onset-lookback`    |    `64` | 正整数；异常起点回溯的最大记录数，也是连续冻结统计更新的上限。                 |
| `--anticipatory.detector.cooldown-steps`    |   `500` | 非负整数；两次触发之间实际完成的 optimizer 更新次数必须严格大于该值，包括 ACTIVE/DRAIN；回滚不倒退该计数。                             |

`detector.warmup-steps` 是检测器的统计预热，与回滚后的路由 WARMUP 不同。
`delay-steps=0` 时不预填队列，ACTIVE 每步先录制当前数据，再立即重放训练，通常不经过 DRAIN。

队列每个元素保存一个 optimizer step 的全部 microbatch 及其专家 ID。microbatch 保留在 CPU，
专家 ID 保留在训练 device；内存占用随 `delay-steps` 增长。过大的 WARMUP 可能耗尽主机内存、
pinned memory 或 NPU 显存而发生 OOM。ACTIVE 先入队再出队，会暂时多持有一个 step；
仅前向工作副本和 buffer 快照也会增加峰值内存。

## 检测与回滚目标

每个 microbatch 结束后仅累加 detached loss；完整 optimizer step 结束后归约为全局 loss。
NORMAL 阶段每步调用检测器，ACTIVE 和队列未排空的 DRAIN 不触发新回滚。

检测器使用 Holt 线性趋势预测。设历史水平为 `level`、趋势为 `trend`、残差方差为 `var`：

```text
predicted_loss = level + trend
sigma = sqrt(max(var, 0))
sigma_floor = max(1e-12, 1e-6 * max(abs(loss), abs(predicted_loss)))
z = (loss - predicted_loss) / max(sigma, sigma_floor)
```

统计预热完成且超过冷却期后，`z > z_threshold` 触发检测。检测器从当前记录向前回溯，
直到遇到 `z <= onset_z_threshold`，得到连续异常区间的起点 `onset`。
超过起点阈值的观测会暂时冻结统计更新，避免异常抬高预测基线；冻结受 `onset-lookback` 限制；达到上限后持续更新统计，只有观测回到 onset 阈值以内才重置冻结计数。
相对标准差下限避免恒定 loss 后的微小数值噪声触发回滚。冷却计数独立于逻辑 step，检测器重置也不清零。

回滚目标 `target` 是满足 `0 < target < onset` 的最新完整 checkpoint，而不是固定回退若干步。
候选必须包含模型参数、optimizer、LR scheduler、dataloader 和训练进度；开启 EMA 时也需保留其恢复状态。
通过原生 DCP 元数据检查 checkpoint 的完整性，选择包含完整训练状态的恢复点。
分布式运行要求 checkpoint 目录对相关 rank 可见，并同步选中的目标 step。

没有有效目标或回滚预算耗尽时，记录 warning 并继续训练。回滚计数属于当前运行状态，
不作为本特性的 checkpoint 状态保存。检测器在回滚和恢复 NORMAL 时重置统计历史，但保留冷却记录。

## 生命周期

```text
TrainerEx.Config.__post_init__()
-> validate_anticipatory_config()

Trainer 标准初始化完成
-> AnticipatorySchedule(trainer)
   -> 关闭时不创建 schedule，trainer.anticipatory_schedule 为 None
   -> 创建 engine：包装 dataloader，安装 checkpoint 保存保护
   -> schedule 绑定 router cache，创建 detector 和 FIFO 队列

TrainerEx.train_step()
-> schedule.training_step_context()
   -> 准备 microbatch 数据和路由模式
   -> super().train_step(batches)
      -> forward_backward_step() 为当前 microbatch 选择路由 slot
      -> 前向、反向、loss 累加
      -> 梯度裁剪、optimizer 和 scheduler 更新
   -> 汇总 step loss、推进阶段、检测 spike
   -> 必要时恢复 checkpoint，并在当前调用内部执行 WARMUP
```


| 阶段   | 数据读取                                  | 路由行为                          | 参数更新 |
| ------ | ----------------------------------------- | --------------------------------- | -------- |
| NORMAL | 从 dataloader 读取当前 step               | 当前参数正常选择专家              | 是       |
| WARMUP | 预取最多`delay_steps` 个 step             | 仅前向录制专家 ID，数据和路由入队 | 否       |
| ACTIVE | 每步预取并录制一个新 step，再取出队头训练 | 重放队头数据对应的专家 ID         | 是       |
| DRAIN  | 停止预取，消费队列中剩余数据              | 继续重放对应的缓存专家 ID         | 是       |

WARMUP 在回滚处理内部完成，不是外层训练循环的独立 step。恢复 checkpoint 后，
engine 丢弃旧的 dataloader iterator，并清理梯度、辅助统计和 FSDP 临时状态。
下一次读取基于恢复后的 dataloader 状态重建 iterator。

ACTIVE 预算归零后，队列非空则进入 DRAIN；队列为空则恢复 NORMAL。
数据和剩余训练步数充足时，恢复阶段执行 `active_steps + delay_steps` 次 optimizer 更新。
临近训练结束或数据耗尽时会提前停止预取、排空已有队列。不完整 step 的读取会撤销，避免把尾部数据误记为已消费。

例如恢复到 step 50，`delay_steps=2`、`active_steps=3`：

```text
WARMUP：录制 B51、B52，step 保持 50
ACTIVE 51：录制 B53，重放训练 B51
ACTIVE 52：录制 B54，重放训练 B52
ACTIVE 53：录制 B55，重放训练 B53
DRAIN  54：重放训练 B54
DRAIN  55：重放训练 B55，队列排空
NORMAL 56：读取 B56，恢复实时路由
```

最后一个 DRAIN step 完成后已恢复 NORMAL，其 loss 可以作为新统计窗口的观测。

## 数据、路由与状态隔离

每个队列元素的 `microbatches[i]` 与 `slots[i]` 对应；slot 内使用 router 的模块路径区分各层。
所有参与录制的动态 router 共享一个 cache，由调度器设置 `OFF`、`CAPTURE`、`REPLAY` 模式。
训练每个 microbatch 前选择对应 slot，并保持到该 microbatch 的 backward 完成，包括 activation checkpoint 重计算。

录制时从 CPU batch 制作 device 工作副本；训练时由上游先在 CPU 统计有效 token，再搬运数据到 device。
直接保留 CPU batch 引用也保留其已有 pinned 属性，不额外执行 CPU 缓存复制。

仅前向使用 `no_grad()`，不调用 `model.eval()`。录制前后保存并恢复注册 buffer、辅助 loss 统计、
token 计数及 Python、NumPy、PyTorch CPU/device RNG，随后重置 FSDP 迭代状态。

engine 在初始化时包装当前 checkpointer 实例的 `save()`。恢复阶段暂停保存，训练失败时也不保存；
允许保存时仍调用原始方法。队列未排空时 dataloader 进度领先参数更新，且 checkpoint 不保存预取队列，
因此此时保存会在恢复后跳过未训练的数据。回到 NORMAL 后解除限制，由上游按原有策略决定保存时机。

## torch.compile 与路由重放

开启 model 编译时，训练继续使用上游的 block 编译流程。每个动态 router 持有一个本地专家 ID Tensor，
所有 router 共享一个标量 bool Tensor，表示是否重放。图内使用 `torch.where()` 在实时 top-k ID 和本地
缓存 ID 之间选择；NORMAL 与 REPLAY 执行相同的 Tensor 运算，不根据 Python 路由模式生成不同的训练图。
REPLAY 仍计算实时 top-k，但选择结果使用缓存 ID，路由分数继续从当前参数的 gate 输出中 gather。

首次训练前，使用已经取出的第一个 microbatch 执行一次隔离的 eager 仅前向，按实际本地路由形状初始化
运行时 Tensor，不额外推进 dataloader、训练 step 或训练统计。所有实际训练 microbatch 均使用上游编译流程。
训练每个重放 microbatch 前，在图外根据 router 路径查找对应 slot，
校验形状并将专家 ID 写入已有 Tensor。该 Tensor 保持到 backward 结束，包括 activation checkpoint
重计算。这样图内不访问 slot 字典和层名字符串，不会因为各层路径或 microbatch slot 不同而逐层重编译。

WARMUP 和 ACTIVE 的仅前向录制保持 training 状态并使用 `no_grad()`，直接调用模型的原有入口。
已经由上游编译的 block 沿用其编译 callable、backend 和选项，未编译的 block 保持 eager。
录制不替换模块的 `forward` 或 `_compiled_call_impl`，FSDP hooks 和 AC 包装沿用上游调用流程。
由于录制与训练的梯度状态和路由操作不同，同一个上游入口可能缓存不同的编译变体。
本特性不修改 `recompile_limit`；多层模型的实际编译变体数量由上游编译入口及 Dynamo guards 决定。

eager 和 compile 使用相同的 `RoutingReplayBuffer` 流程。每个 router 持有两个独立的 int64 Tensor：
`indices` 供重放读取，`capture_indices` 供录制写入。通过显式方法设置 `INITIALIZE`、`CAPTURE`、
`TRAIN` 和 `OFF` 操作，不使用梯度开关、模块的 training 属性或是否正在编译来推断录制与重放。
`TRAIN` 操作在 NORMAL 与 REPLAY 之间保持不变，仅通过 Tensor 开关选择实时或缓存 ID；
`CAPTURE` 操作将实时 top-k 写入 `capture_indices`。录制和训练分别使用明确的操作分支，
局部 `no_grad()` 或 `enable_grad()` 不改变路由操作。`begin_training()` 设置的状态覆盖完整训练 step
的前向和反向，包括 AC 重计算；结束后 `reset()` 切回 `OFF`，直接返回实时 ID，验证输入不受运行时 buffer 形状约束。
图内不写 Python slot 字典。`begin_capture()` 设置录制状态，前向成功后调用 `save_captured_routes()`，
在图外将每层录制结果复制、转换为 `index_store_dtype`，写入当前 microbatch 的 slot。
队列项拥有独立存储，后续录制覆盖运行时 Tensor 不会改变历史路由；录制 Tensor 也不覆盖重放 Tensor。
这两个运行时 Tensor 不注册为模型 buffer，不写入 checkpoint，也不会被仅前向状态恢复覆盖。

首次缓冲区形状初始化先调用 `begin_initialization()`，再执行隔离的 eager 前向。
初始化、录制和训练依次执行，调用方均通过 `try/finally` 保证正常结束或异常时调用 `reset()`，
清除路由状态与 slot 引用，但保留已分配的 Tensor。trainer 使用的 `training_step_context()` 保持不变。
已编译模块与 eager 模块使用同一套路由 buffer 和操作状态，无需安装额外的录制编译入口。
首次遇到录制图时发生编译，后续满足相同 guards 的执行复用缓存。
eager 重放也会计算实时 top-k，再通过 Tensor 开关选择缓存 ID；每层额外持有两个固定 ID buffer。
`_isolated_forward()` 仍在图外保存、恢复训练 buffer、辅助统计、RNG 和
token 计数，并重置 FSDP 临时状态；编译本身不负责撤销额外前向的副作用。

## 模块职责


| 路径                                                         | 职责                                                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `torchtitan_npu/extensions/torchtitan/trainer.py`            | 配置接入、一次初始化及训练上下文编排。                                               |
| `torchtitan_npu/extensions/experiment/anticipatory_routing/config.py`   | 声明恢复配置、参数范围及配置兼容性校验。                                                             |
| `torchtitan_npu/extensions/experiment/anticipatory_routing/engine.py`   | 组件初始化、loss 汇总、仅前向隔离、checkpoint 选择及恢复、保存保护。 |
| `torchtitan_npu/extensions/experiment/anticipatory_routing/schedule.py` | 管理 FIFO 队列、阶段转换、训练数据供应及 spike 响应。                                |
| `torchtitan_npu/extensions/experiment/anticipatory_routing/cache.py`    | 管理路由模式、microbatch slot 和专家 ID 存储。                               |
| `torchtitan_npu/extensions/experiment/anticipatory_routing/data.py`     | 管理可恢复 iterator、完整 step 读取及仅前向工作副本。                                |
| `torchtitan_npu/extensions/experiment/anticipatory_routing/detector.py` | 维护 loss 趋势统计，判定 spike 并回溯 onset。                                        |
| `torchtitan_npu/extensions/experiment/anticipatory_routing/router.py`     | 通过 HashRouter 配置 override 接入录制与重放，查找 router 并创建、绑定共享缓存。                                            |

## 验证范围

`tests/unit_tests/extensions/experiment/anticipatory_routing/test_anticipatory_routing.py` 覆盖以下场景：

- 回滚后模型、optimizer 动量、LR scheduler、训练 step、token 计数和 dataloader 位置恢复，重建 iterator 后的数据及参数更新与恢复点一致。
- 多 microbatch、多层 router 的专家 ID 正确配对；实时 top-k 与录制结果不同时仍使用缓存 ID，路由分数保持可求梯度。
- CPU 场景下的 WARMUP → ACTIVE → DRAIN → NORMAL 完整恢复周期，包括数据顺序、队列深度、参数更新次数及数据和路由的存储设备。
- WARMUP 不推进参数、scheduler、token 计数或注册 buffer；ACTIVE 和 DRAIN 均消费对应的缓存路由，队列排空后恢复正常路由。
- 恢复期间暂停 checkpoint 保存，恢复 NORMAL 后允许保存；实际 REPLAY 前向异常原样传播、清理队列和路由上下文，并禁止后续保存。
- 新 checkpoint 缺少 optimizer 元数据前缀时选择旧的完整 checkpoint；拒绝其他 dataloader 和可广播但形状不匹配的路由缓存。
- eager 与编译 router 在不同梯度开关和 train/eval 属性下按内部上下文录制、重放，NORMAL 不覆盖录制结果。
- 40 层 router 全部编译或交替使用 eager/compile，并分别开启、关闭 AC 时，输出及梯度与 eager 路径对齐，重复录制和重放复用编译缓存。
- 40 层动态 router 在默认重编译预算下复用训练图，覆盖普通训练、仅前向录制、多 microbatch 重放、再次恢复普通训练和 checkpoint 加载。
- 无 AC 和 FullAC 场景下，编译路径与 eager 路径的输出、梯度和录制 ID 一致；录制图与训练图分别缓存，重复录制、模式切换及 checkpoint 加载不新增编译图，重放前检查路由缺失及形状错误。
- 多个录制 slot 不共享运行时存储，后续录制不覆盖历史专家 ID；录制编译失败时保留原有编译入口并恢复路由上下文，模型 buffer、RNG 和 token 计数保持不变。
- 录制沿用上游模块编译入口及 forward hooks；录制前后保持 RNG、模型 buffer 和 token 计数不变。

`tests/unit_tests/extensions/experiment/anticipatory_routing/test_detector.py` 覆盖持续水平漂移、恒定 loss 后的微小噪声、
跨回滚及 ACTIVE/DRAIN 更新的冷却计数、非有限阈值校验、零回滚预算的 checkpoint 校验、PP 拒绝和 ID 存储类型边界。

`tests/smoke_tests/anticipatory_routing/test_anticipatory_routing_smoke.py` 使用小型 NPU 路由模型和上游训练步，覆盖以下场景：

- 一次性扰动 labels 引起真实 loss 升高，由自动检测器识别 spike 并触发回滚。
- 选择并加载磁盘 DCP checkpoint，恢复模型、optimizer、LR scheduler、训练进度及 dataloader 状态。
- 执行 WARMUP → ACTIVE → DRAIN → NORMAL 完整恢复流程，检查阶段切换、队列深度和训练完成状态。
- 校验录制与重放时的输入及各层专家 ID 一致；实时 top-k 改变时，训练仍使用缓存的专家 ID。
- 检查数据消费顺序，以及 ACTIVE、DRAIN 的路由重放和恢复 NORMAL 后的实时路由计算。
- 检查 WARMUP 的 buffer、token 计数和 scheduler 状态隔离，以及 CPU microbatch、NPU 专家 ID 的存储位置。
- 检查恢复期间暂停 checkpoint 保存，恢复 NORMAL 后按训练流程继续保存。

## 恢复边界与执行时序

回滚恢复模型、optimizer、LR scheduler、dataloader 和训练进度。随机流继续推进，
不回退到 checkpoint 时刻，与普通续训一致；这不承诺逐位重现历史随机计算。
WARMUP 的 RNG 隔离只保证额外录制前向不推进进入录制时的随机流。

稳态每个外层训练迭代完成一次 optimizer 更新。触发回滚的那一轮会在 train_step 返回前
回退逻辑 step：checkpoint 保存被暂停，validator 仍按恢复后的 step 判断，可能重复验证该 step；
profiler 仍按实际执行的外层迭代推进，结构化日志的 step 标签保留触发步，直到下一轮重新设置。
因此回滚轮次的验证、分析和日志步号不等同于无回滚训练。

全 Hash 路由模型没有动态专家选择，固定表的路由天然一致。此时启动会给出 warning，
特性退化为仅回滚；现有调度仍会预取和排空数据，但不缓存动态专家 ID。

真实入口集成用例 `dsv4_anticipatory_recovery` 注册到 models/deepseek_v4 suite，
使用单卡真实 DeepSeek-V4 recipe、CLI 配置和原生 checkpointer，训练目标仅在测试入口中
一次性放大以触发真实检测器，验证 override 生效及 WARMUP → ACTIVE → DRAIN → NORMAL。
生产配置不增加手动触发参数。执行命令：

```bash
python -m tests.integration_tests.run_tests /tmp/anticipatory_integration \
  --test_suite deepseek_v4 --test_name dsv4_anticipatory_recovery --ngpu 1 --no-parallel
```
