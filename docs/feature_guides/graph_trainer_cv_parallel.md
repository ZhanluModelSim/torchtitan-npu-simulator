# DeepSeek-V4 batch 双流调度

该实验将 local batch 等分为两个 chunk，分别放在主流和辅助流。启动时先测量 free 双流图，再通过同流局部重排和跨流 event 增加重叠。搜索优先平衡提高「Cube 被 Vector 掩盖」和「Vector 被 Cube 掩盖」两种覆盖率，再提高纯 Cube/Vector（CV）重叠；每次修改都不允许 CC 或 VV 重叠增加。只有每个 rank 的成本模型都预测两种覆盖率严格提高时才选择 scheduled 图；随后的设备实测用于报告和告警，不再触发 `dependency_only` 回退。

## 运行

当前解析契约针对 CANN 9.2 和 `requirements.txt` 固定的 torch-npu 版本，需要匹配的 torchtitan、CANN/HCCL 环境和空闲 NPU。在仓库根目录执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5_cv_parallel.sh \
  --dump-folder outputs/dual_chunk_cv > dual_chunk_cv.log 2>&1
```

默认复用 `graph_trainer_deepseek_v4_flash_43layers_16experts` 配置，使用 8 卡、43 层、EP8/FSDP8、MBS=2；GBS=64、seq4096、100 步等通用默认值继承 A5 launcher。可通过 `CONFIG` 和 `NGPU` 选择配置与卡数，通过追加 `--training.steps`、`--training.global-batch-size`、`--training.seq-len` 和 `--dump-folder` 调整训练参数。tokenizer、数据和 checkpoint 路径沿用 A5 launcher 的 `HF_ASSETS_PATH`、`DATASET_PATH`、`CKPT_SAVE_LOAD_PATH` 设置；运行前应配置有效路径。卡数须满足模型的 EP/FSDP 拓扑要求。

后四卡的小模型验证使用仓库已有的 `deepseek_v4_debugmodel` 的 GraphTrainer 入口：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
CONFIG=graph_trainer_deepseek_v4_debugmodel \
NGPU=4 \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5_cv_parallel.sh \
  --training.global-batch-size 8 --training.seq-len 512 --training.steps 10 \
  --dump-folder outputs/dual_chunk_debug_cv > dual_chunk_debug_cv.log 2>&1
```

脚本从仓库根目录运行，使用当前 Python 环境，并返回训练退出码。日志输出到标准输出/错误，示例命令将其重定向到文件。配置入口为：

```text
module = torchtitan_npu.models.deepseek_v4
config = graph_trainer_deepseek_v4_flash_43layers_16experts
```

双 chunk 的启用参数由脚本传入：`--training.local-batch-size 2`、`--compile.ep-overlap.enabled`、`--compile.ep-overlap.chunk-dim batch`、`--compile.ep-overlap.strategy graph`、`--compile.ep-overlap.module-fqn 'layers.*'` 和 `--compile.pass-pipeline cv_parallel`。实际调用的 A5 launcher 默认启用 `all_block_fp8`、LI FP8 和 KV norm MXFP8；GraphTrainer 配置名不会关闭这些 CLI 参数。需要 BF16 训练时，在命令末尾追加 `--extension.quantization.no-enable-quantized-training`。

如果通过 CLI 整体替换 `--override.imports`，须保留以下实验依赖：

```text
torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata
torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li
torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk_metadata.asc_metadata
torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk_token_dispatcher.asc_dispatcher
```

实验 `batch_chunk_metadata.asc_metadata` 在启用时注册 `cv_parallel` pipeline。实验 metadata、dispatcher 分别替换普通 `sparse_attn.asc_metadata`、`common.token_dispatcher.asc`，同一配置节点不能同时启用两种 override。RoPE 沿用公共实现；dispatcher 的包装只适配符号 token 数，实际前后向仍调用原生 NPU kernel。普通配置不会选中实验实现。

完整 batch 在追踪时约束为 `B >= 2`，图切分后每个 chunk 可以只有一个样本。运行时输入和追踪适配通过 `cv_parallel` 的 post-init hook 绑定到当前 trainer 实例，不替换公共 GraphTrainer 方法或原有 `ep_overlap` 注册项。

## 实现流程

1. 在 eager 阶段为两个半 batch 分别构建 sparse-attention metadata。文档边界必须与等分点对齐；压缩 plan、AscendC metadata 和 host 字段不能按 tensor 长度机械切半。
2. 复用 torchtitan 的 batch graph chunk pass，将两个 chunk 固定到两条流，并把 duplicated layer 的 metadata 输入重映射到预构建的局部值。
3. `dependency_only_schedule` 保留数据依赖、流内 FIFO、通信、主机观察和共享状态屏障；跨流消费和最终输出汇合仍会产生必要 event。
4. 对同一个 batch 重放 free 图，采集各 rank 的节点设备耗时和 kernel 资源类型。校准会克隆输入、恢复可变 buffer，并验证参数和梯度状态未改变。
5. 搜索器按双向 CV 掩盖率、纯 CV 时间、同类重叠和 MIX 重叠依次比较局部重排或 event gate。候选不得降低任一方向的掩盖率，不得增加 CC/VV，且全图预测 span 不得超过 free。通信、RNG、原地写、未知调用及强制屏障不会被跨越。
6. 每个 rank 的最终预测必须同时提高两种 CV 掩盖率，且不增加 CC/VV；否则回退到已校准的 `dependency_only` 图。选中 scheduled 图后再次计时并采集 profile，记录纯 CV、双向掩盖率、CC/VV 和同步 F/B 的验收结果。实测未提高时保留 scheduled 图继续训练，并在报告中写入 `validation_warning`。

free 校准预热 3 次、计时 3 次；scheduled 预热 1 次、计时 3 次；两者各采集一次详细 profile。同步计时覆盖完整前后向，但不含输入克隆、状态检查、profiler 和优化器。

### 搜索计算开销

自动搜索以通信、RNG、原地写、未知调用等不可跨越的强屏障划分区域。节点数不超过 512 的区域完整枚举可重排节点的全部合法目标位置、所有独立 Cube/Vector phase 组合及其可用等待源，并持续迭代到该区域不再产生改进。超过 512 个节点的区域使用自适应有界搜索：最多执行 8 轮，每类最多选择 16 个高收益重排节点，每个节点最多尝试 24 个近邻、指数间隔和全局目标位置；每轮最多检查 2048 个 CV 候选，每个候选最多检查 32 个等待源。搜索仍不生成 VV gate。

区域彼此不存在可调度的跨界候选，因此按屏障拆分不会缩小合法搜索空间；区域候选仍与区域外已累计的覆盖量合并后，按全图双向覆盖率打分。重叠查询缓存区间前缀覆盖值；新增等待只传播发生延后的父节点完成时间。窗口评分复用相同节点和开始时间的覆盖值，重排依赖集合按目标位置增量构建。小区域不裁剪候选；大区域的上述预算用于限制启动搜索成本。

搜索还缓存三类可复用结果：按 `(node, start_time)` 缓存平移后的 kernel phase；按完整窗口顺序和移动节点缓存所有依赖安全的重排模板；按调度状态版本缓存 event/reorder trial。前两类只依赖不可变 phase 或精确窗口顺序，可跨 `refresh()` 复用；trial 依赖 clocks、开始时间、gate 和覆盖量，每次 `refresh()` 必须失效。缓存均有容量上限。区域数、最大区域节点数、搜索模式、预算、完整 trial 数、缓存命中、候选跳过数量和 trial 失效数量写入 `schedule_feedback.json`。

最后删除不降低双向 CV 掩盖率、不增加 CC/VV、跨度或计算并集的冗余 gate。依赖当前 clocks、时间线或 coverage 的缓存随调度状态重建而失效；只有不可变 phase 和精确窗口顺序对应的模板允许跨状态复用。搜索是区域内贪心迭代，不等价于组合意义上的全局穷举；小区域迭代到收敛，大区域达到收敛或预算上限后停止。

MIX 分类沿用 `a95f74f` 的逐 launch 计数器规则：优先读取 `op_summary*.csv`，没有该文件时读取 `kernel_details.csv`，按名称、开始时间、时长及可用的 stream/task ID 匹配。主核时间占比至少 75%、次核累计时间不超过 0.1 ms 时，细分为 `mix_aiv` 或 `mix_aic`；有效的 MAC/Vector 执行单元时间优先于整核时间。缺少有效计数器、匹配不唯一或不满足阈值时保留 `mix`，不凭 MIX 名称推断主核。

节点成本、kernel phase 与重叠统计使用同一分类规则。该分类只是调度近似，不保证次核空闲，MIX 不计入纯 CV。统计另列 Cube×`mix_aiv` 和 Vector×`mix_aic`；搜索在双向纯 CV 掩盖率、纯 CV 及 CC/VV 目标之后比较 MIX 重叠。DeepEP 屏障保持不变。恢复分类不代表恢复旧环境的全部成本输入或 A5 实测收益。

## 接入已有 graph pipeline

已有 pipeline 完成切分并为节点写入 `chunk_id=0/1` 后，可把调度 pass 放在最后：

```python
from functools import partial
from pathlib import Path

from torchtitan_npu.extensions.experiment.cv_parallel.cv_parallel import schedule_pass

passes.append(partial(
    schedule_pass,
    profile_root=Path(config.dump_folder) / "profiling",
    runtime_context=runtime_context,
))
```

`runtime_context` 须显式提供 `traced_result`、`module`、本次追踪参数 `args` 和 `train_context`；`cv_parallel` pipeline 从实例 hook 获取这些对象。输入必须是可重放的完整前后向 FX 图，并保留 split、合并、梯度累加、通信和共享状态依赖。该入口只负责 profile、调度和验收，不提供 seq 切分语义，也不能与另一套跨流调度叠加。

## 输出与判读

校准结果默认位于 `--dump-folder` 指定目录下的 `profiling/schedule_feedback.json`；设置 `CV_PARALLEL_PROFILING_DIR` 时使用该目录：

- `status=applied` 且 `selected=profile_guided_cv`：逐卡预测满足双向 CV 掩盖率提高且 CC/VV 不增加，scheduled 图用于训练。
- `status=fallback` 且 `selected=dependency_only`：至少一个 rank 没有有效预测收益；原因记录在 `fallback_reason`。设备实测结果不会产生该状态。
- `prediction`：free 成本模型的双向掩盖率、CV/CC/VV、compute union 和 span 预测，不等于设备提速。
- `schedule.cv_validation_by_rank`：逐卡实测 CV 时间、双向掩盖率及 CC/VV 差值。
- `validation_passed` 与 `validation_warning`：记录 scheduled 实测是否满足全部验收项及首个告警原因。验收失败只告警，不切换执行图。
- `schedule.forward_backward_delta_ms`：scheduled 相对 free 的前后向中位数变化，正值表示变慢；此时 `forward_backward_guard_passed=false`，但仍保留 scheduled 图。

rank 0 的原生 profile 保存在 `profiling/dependency_only/rank0/` 和 `profiling/profile_guided_cv_validation/rank0/`。可用以下命令查看关键阶段：

```bash
rg 'CV parallel regional search|CV parallel predicted gains:|CV parallel waits:|CV parallel scheduled measured:|CV parallel CV guard:' dual_chunk_cv.log
```

### 正常训练输出

调度实现按职责分为三个模块：`schedule_search.py` 生成候选并控制区域搜索；`schedule_simulation.py` 维护依赖、时间线和预测缓存；`schedule_scoring.py` 提供指标公式与候选接受规则。搜索器持有独立的模拟对象。`schedule_calibration.py` 在本地搜索返回后汇总各 rank 的预测并检查通信顺序，再进入候选图实测。预测使用固定 profile 时长，不代表实际争用或完整 step 加速。

`profile_overlap.py` 是启动校准的指标计算模块：`profile_whole_graph_costs` 调用它，从实际 profile 中计算前后向计算并集及 C/V、CC/VV、MIX 重叠，供基础图与候选图的实测验收和告警使用。它不生成多模式比较报表，也不直接执行调度搜索；搜索使用另行提取的节点成本和 kernel phase。当前校准流程仍需要该模块。

入口直接启用 C/V 双流调度，不额外生成 manifest、Git 快照或状态 JSON。训练输出由 `--dump-folder` 控制；需要 TensorBoard 时显式追加 `--metrics.enable-tensorboard`。

启动校准仍会生成调度所需的 profile 和验收报告。未找到预测收益时使用保留依赖和屏障的 free 双流图继续训练。校准的前后向耗时和重叠率不能证明完整训练 step 提速。

## 性能与内存测试报告

### 测试配置

测试配置采用仓库内的启动脚本 [deepseek_v4_flash_8p_cpt_4k_a5_cv_parallel.sh](../../examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5_cv_parallel.sh)，测试设备为 **A5 服务器**。对比 baseline 的 `graphtrainer` 与开启图级 CV 并行的 `cv_parallel`。

### 训练耗时与内存

| 指标 | baseline | cv_parallel |
| --- | ---: | ---: |
| 状态 | 完成 | 完成 |
| 稳态 E2E 中位数（ms/step） | 24536.326 | 24360.393 |
| F/B scope（ms） | 23750.771 | 23664.367 |
| 计算跨度（ms） | 23792.064 | 23739.603 |
| 计算并集（ms） | 17027.062 | 16650.671 |
| 计算加速比（按计算并集） | 1.000× | 1.023× |
| E2E 加速比（按稳态中位数） | 1.000× | 1.007× |
| 内存（GB） | 62.95 | 77.61 |

计算并集减少 **376.391 ms（2.21%）**，稳态 E2E 中位数减少 **175.933 ms（0.72%）**。E2E 包含通信、同步及其他开销，因此计算并集的改善不会按同样比例转化成完整 step 的改善。

内存增加 **14.66 GB（23.29%）**。双流执行与重排可能延长张量存活时间，但现有数据不足以把增量归因到某一环节。

### 资源重叠

| 指标 | baseline | cv_parallel |
| --- | ---: | ---: |
| 未重叠计算（ms） | 17027.062 | 3434.019 |
| 总重叠（ms） | 0.000 | 13216.652 |
| 纯 CV 重叠（ms） | 0.000 | 3072.006 |
| Cube × MIX_AIV 重叠（ms） | 0.000 | 16.253 |
| Vector × MIX_AIC 重叠（ms） | 0.000 | 971.115 |
| CC 重叠（ms） | 0.000 | 3719.612 |
| VV 重叠（ms） | 0.000 | 1631.421 |
| MIX/不明确重叠（ms） | 0.000 | 4793.613 |
| 总掩盖率（%） | 0.000 | 79.376 |
| 纯 CV / 计算并集（%） | 0.000 | 18.450 |
| Cube 被 Vector 覆盖（%） | 0.000 | 31.959 |
| Vector 被 Cube 覆盖（%） | 0.000 | 34.645 |

纯 CV 重叠由 0 增至 **3072.006 ms**，说明本次 profile 中 Cube 与 Vector 的执行区间形成了并发。Cube 被 Vector 覆盖 **31.959%**，Vector 被 Cube 覆盖 **34.645%**，分别从两类资源自身的执行时间衡量重叠程度。总重叠还包含 CC、VV 和 MIX/不明确重叠，因此 **79.376%** 的总掩盖率不能解释为纯 CV 协同收益。

### 统计口径与结果边界

- **E2E** 是完整训练 step 的端到端耗时。稳态统计排除预热、性能采样与调度搜索阶段；加速比采用 baseline 中位数除以 cv_parallel 中位数。
- **F/B scope** 是前向与反向测量范围的耗时；**计算跨度**是首个计算 kernel 开始到最后一个计算 kernel 结束的时间，包含其中的间隙。两者与完整 step 的测量边界不同。
- **计算并集**是计算 kernel 执行区间的并集：重叠部分只计一次，不含纯通信与空闲区间。计算加速比采用 baseline 计算并集除以 cv_parallel 计算并集。
- **总重叠**按本次数据为计算并集中至少两个计算 kernel 同时执行的时间；未重叠计算与总重叠之和等于计算并集。总掩盖率以计算并集为分母。
- **纯 CV 重叠**不含 MIX 算子，占比以计算并集为分母。两项覆盖比例分别以 Cube 和 Vector 各自的执行时间为分母，不能相加。

## 限制

- 当前只支持整层栈的等分 batch chunk，不支持该 metadata 路径上的 context parallel；文档不能跨半 batch 边界。
- profiler 依赖 CANN 9.2 导出的 `trace_view.json`、`kernel_details.csv` 及 torch-npu `analyse()` 接口；升级 CANN 或 torch-npu 后须先回归 trace 命名、CSV 列和 launch 关联测试。
- event 只约束先后，不能保证 kernel 同时开始。预测 span 不增也不代表实测加速，因此仍采集设备验收数据，但验收结果不触发执行图回退。
- 搜索优先平衡两种方向的 CV 掩盖率，并限制 CC/VV 不增加；小区域完整枚举并收敛，大区域使用固定预算，两者都是贪心搜索，不承诺组合意义上的全局最优或端到端提速。通信、host 下发、event 开销和新的资源竞争可能抵消收益。
- 校准 batch 通过不保证后续每步都获得相同收益；重排还可能延长张量存活时间。
