# 原生 dispatch 收敛与能力复验

> 历史验收记录：本文命令和输出对应当时源码。当前已移除 mem 以外的图导出、CSV rank 参数和 view 折叠；当前入口见 [接入指南](upstream-integration.md)，当前验证见 [mem 与精度审计](mem-only-precision.md)。

本轮将 simulator 的计算事实采集收敛到原生 dispatcher，保留必要的 Meta/autograd 能力修正，以及 storage 驻留、通信归属等独立语义。TorchTitan、torchtitan-npu、PyTorch、TorchAO-NPU 的安装源码未编辑。

## 实现与直接验证

| 审查项 | 当前实现 | 直接证据 |
| --- | --- | --- |
| FSDP 整段屏蔽 | unshard/wait/reshard/post_backward 改为标注作用域，转换、累加、packing 和 alias 均保留原始图与内存事件 | HSDP 模型测试检查 `_to_copy`、`add_`、`chunk_cat` 与转换输出 storage 的 lifetime |
| AdamW 手工节点 | 使用原生 `aten._fused_adamw_`；仅保留 Meta 设备资格检查和参数组注解 | 实际分桶、unused 参数、参数/梯度/双状态 mutation identity、真实 storage 和非默认超参数断言 |
| 通信手工前反向 | 删除三套自定义 autograd、Python collective API 替换、按值别名扫描和输出占位；保留 native functional、c10d、DTensor transport | Meta 轴变换、原生 SAC、两 rank Gloo 不均匀 all-to-all、通信前后精确 producer/consumer 与既有 alias |
| 列表通信 | c10d `alltoall_` 的全部输入输出进入图；payload 根据实际所有 chunk 求和 | 不均匀 chunk 的 128-byte 独立先验与输出 identity 检查；不只按首个 chunk 计费 |
| CANN metadata | 顶层 `torch.ops` 走 dispatch；能力探针只修正厂商 Meta 内错误的 NPU 分配，复用原厂推导 | 四个算子的 shape/dtype/consumer；正常、异常和健康 Meta 注册恢复 |
| 遗留显式模型 | 删除无生产调用者的 synthetic SAC、缺梯度人工生命周期与当前禁用的 FP8 AG transport 投影 | 原生 cache/保存槽仍被观察；缺梯度诊断不制造 storage，标准模型入口缺梯度即失败 |

RoPE、RMSNorm、token permute、SwiGLU 的桥接仍由真实梯度能力探针选择，并调用原生前后向算子。均衡 EP split、Meta device/stream protocol、实际 saved-tensor/alias/storage 观察与 FSDP 驻留仍保留。

## 图与通信的验收约束

原始 trace 保留 alias 和 mutation 事实；L0 计算图只对 alias-only metadata 路径做依赖收缩。恢复原始事件不以复制合成节点替代。

- 每个捕获的原生计算/通信 source ID 在最终图中恰好出现一次；禁止恢复 FSDP 整段 dispatch 屏蔽。
- 检查全局 DAG、跨模板双向边、模板/实例、SchedulePlan/DataSlot、通信引用及全部导出消费者。
- 通信描述从真实 operands 提取 group、所有 chunk 的形状和字节；按实际 RS/AR process group 标注 DP 归约，TP all-reduce 不冒充 DP 归约。
- 实际 DP 归约数与 L1/L2 归属计数相等，EP 启用时必须存在实际 A2A；归约、clip、optimizer 保持数据依赖。
- 原地通信在写入前读取实际输入 producer，按输出 storage 找到首次消费者；独立控制依赖不会覆盖数据来源。

## 内存验收约束

内存使用实际 local Tensor/storage、alias generation、保存槽、原生 cache 与 FSDP 驻留事件。逻辑优化器注解不会替代真实参数、梯度和状态 operands。

每项完整捕获检查保存值真实来源、pack/unpack/release 顺序、生命周期有效区间、峰值 storage 去重、参数 shard 字节及 FSDP allocation/release。独立消费者从导出的生命周期区间重新求总峰值，按 raw-op phase 坐标另算前向／反向／optimizer 与 optimizer 前峰值，并复核 CSV/JSON/DOT 的图引用与边。

恢复此前隐藏的转换和临时分配可能合理改变峰值；不强制复现 S5 的旧峰值。参数存储 dtype 与 offload 仍是显式内存投影。指标不含 allocator reserved/cache、碎片、未建模 kernel workspace 或真实 offload 时延。

## 当前复验

完整矩阵的精确字节、source hash、独立导出审计及 S5 差值见 [本轮结果](dispatch-first-results.json)。S5 的 21 项历史结果保留在 `s5-results.json`，不作为本轮通过证据。

21/21 项完整 Meta F/B/clip/optimizer 和独立导出／分阶段内存审计通过。所有配置的保存槽数量、总峰值、反向及 optimizer 峰值与 S5 相同；恢复的 FSDP 临时操作改变了部分 BF16 前向峰值，精确差值保留在 JSON。

| Preset | none：节点 / 峰值 GiB | full：节点 / 峰值 GiB | selective：节点 / 峰值 GiB | 参数梯度数 | 三模式最大 RSS GiB |
| --- | --- | --- | --- | --- | --- |
| deepseek_v4_flash_baseline_bf16 | 37,738 / 543.577 | 47,567 / 543.577 | 47,229 / 543.577 | 1316 | 1.595 |
| deepseek_v4_flash_baseline_mxfp8 | 41,039 / 543.577 | 51,577 / 543.577 | 51,454 / 543.577 | 1316 | 1.711 |
| deepseek_v4_pro_20t_baseline_bf16 | 82,018 / 37720.882 | 104,137 / 37720.882 | 103,328 / 37720.882 | 2907 | 2.190 |
| deepseek_v4_pro_20t_baseline_mxfp8 | 89,290 / 37720.882 | 112,992 / 37720.882 | 112,663 / 37720.882 | 2907 | 2.478 |
| deepseek_v4_pro_baseline_bf16 | 53,043 / 2982.633 | 67,140 / 2982.633 | 66,626 / 2982.633 | 1864 | 1.772 |
| deepseek_v4_pro_baseline_mxfp8 | 57,694 / 2982.633 | 72,797 / 2982.633 | 72,588 / 2982.633 | 1864 | 1.980 |
| deepseek_v4_smoketest | 5,025 / 0.340 | 5,735 / 0.275 | 5,708 / 0.280 | 143 | 1.315 |

GiB 为捕获 logical rank 的逻辑活跃 storage；RSS 为捕获进程的实际主机内存，不含 JIT 子进程。大模型总峰值位于 optimizer，不能用总峰值相同推断 AC 对激活无影响。


标准全规模矩阵入口仍为：

```bash
source /usr/local/Ascend/cann/set_env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/acceptance.py --output-dir /audit/dispatch-first-remediation/matrix-release
```

七个 preset 各运行 none/full/selective，保留原生几何；B=1、S=512、MTP=1、world=16、logical rank=15，组合 HSDP+EP+TP+CP。smoketest 导出全部六种格式，大模型导出 text/csv/mem。另有单卡、DDP/FSDP/HSDP/EFSDP、EP、TP/SP、CP、非零 rank、MXFP8、dtype/offload 投影的模型级测试。

## 测试与审查

- 全量测试 431 passed；独立包类型检查 0 errors（11 suppressed）；全仓同环境原始基线／候选均为 64 个既有类型错误，新增 0。
- 完整 pre-commit 已运行，常规、Ruff、format、architecture、OAT 通过；macOS 的 Pyrefly launcher 找不到可执行文件。因此不声明全仓 pre-commit 全绿；类型结果取自完整依赖 Linux 容器。
- 独立代码审查的图／依赖／内存必改项已修复；最终复审未发现新的必改项。最终能力状态以本轮矩阵和独立导出审计为准。

| 层次 | Oracle 与覆盖 | 结论边界 |
| --- | --- | --- |
| 算子与 tensor 契约 | 实际 identity/storage、非默认 AdamW 参数、128-byte 多 chunk 通信、注册恢复、准确 producer/consumer | 验证真实 operands 和写入关系；不以相同 shape 冒充相同对象 |
| 反向与 AC | 两 rank Gloo 远端梯度、不均匀 A2A；原生 SAC/full AC 保存、unpack 和释放 | CPU 通信／autograd 契约与 Meta 图；不证明 HCCL 或 fused NPU 数值 |
| 集成图 | 完整 DSV4/MTP、并行组合、原生节点唯一保留、DP/TP process group 区分、归约→clip→optimizer | 实际 Meta F/B/O 和所有导出消费者 |
| 内存 | 独立 storage 区间峰值、alias 去重、FSDP alloc/free、保存槽来源及释放、dtype/offload 投影 | 逻辑活跃 storage；不包含硬件 allocator/workspace |

代表性新增测试为 `test_native_fused_buckets_keep_actual_updates_and_mutations`、`test_native_all_to_all_gloo_preserves_uneven_rows_and_remote_gradients`、`test_native_inplace_collective_descriptors_follow_data_and_preexisting_alias`、`test_native_list_all_to_all_counts_every_chunk_and_keeps_output_storages` 与 `test_vendor_metadata_dispatch_preserves_consumer_dependency_and_restores_scope`。golden 未更新；独立字节、identity、远端梯度和区间求和均直接断言。

执行环境为固定 CANN 容器内 `/workspace` sidecar；全量命令 `TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/audit-venv/bin/python -m pytest tests/ -q`，另以 `python -m pyrefly check --config pyproject.toml` 加入固定 TorchTitan/NPU/TorchAO-NPU search-path 完成类型检查。日志为 `full-tests-8.log`、`types-final.log`、`root-types-comparison.json`。

| 变更范围 | NPU ST 判断 | 实际执行与限制 |
| --- | --- | --- |
| 独立实验包的 Meta 兼容、capture/IR/memory/export 和测试 | 未修改生产 `torchtitan_npu/` 的 NPU 训练路径，生产训练 ST 不适用本次 sidecar 图验收 | 执行 Meta 矩阵及 CPU/Gloo；未执行真实 NPU ST，不将 CPU UT 计为 NPU ST |

BF16 的前向峰值较 S5 可增加：例如 Pro none 的峰值增加 4,565,217,236 bytes。新峰值处恢复捕获到独立的 `aten.empty` BF16 packing buffer（6,341,787,648 bytes），经 `_foreach_copy_`、`fsdp.all_gather_copy_in` 消费，与 AG 输出 storage 分开。旧、新峰值发生位置不同，因此 buffer 字节不等于净峰值差；独立生命周期求和与当前结果一致。

## 镜像与复现

镜像：`torchtitan-simulator:titan030-dispatch-first`（Docker context `colima-titan-migration-audit`）。沿用已含完整官方 CANN 的固定环境，不运行 CANN 安装包。安装包的 78 个 Python 源文件与矩阵捕获源 SHA256 一致；TorchTitan/NPU 663 个、TorchAO-NPU 50 个 Python 源文件与冻结基线一致。

独立安装镜像以正常 CLI 验证 MXFP8 selective、world16/rank15、六种导出格式；另验 BF16 selective、world2048/rank2047、CSV 限制一份。镜像加载 `site-packages`，不依赖工作区 editable import。精确镜像 ID、命令、源码一致性和导出检查保存在审计目录 `dispatch-first-remediation/image-*-integrity.json`、`image-*-validation.json` 及 `final-manifest.json`。

```bash
docker --context colima-titan-migration-audit run --rm \
  -v /absolute/audit/output:/audit \
  torchtitan-simulator:titan030-dispatch-first \
  --simulation.world-size 16 --simulation.logical-rank 15 \
  --parallelism.data-parallel-replicate-degree 2 \
  --parallelism.tensor-parallel-degree 2 \
  --parallelism.context-parallel-degree 2 \
  --parallelism.expert-parallel-degree 2 \
  --training.num-mtp-modules 1 --training.seq-len 512 \
  --activation-checkpoint.mode selective \
  --extension.quantization.enable-quantized-training \
  --simulation.csv-max-ranks 1 --simulation.output-dir /audit/run \
  --simulation.output-formats text html dot json csv mem
```

## 兼容与边界

- 未建成本公式的算子保留原始节点与内存事实，成本仍显式标记 unknown（逐项计数见结果 JSON）；图完整不等于耗时／FLOPs 估算完整。
- 当前范围是 DSV4 Meta simulator。Gloo 两 rank 验证 CPU collective 契约；真实 NPU 数值、HCCL 和硬件性能未验证。
- PP、FP8 通信与 A5 仍在公开入口明确拒绝。其他模型和完整多 microbatch 梯度累积峰值未由本轮扩大支持。
- metadata 当前在 host mask/CP planning 阶段生成，随后才开始训练捕获；准备阶段成本范围见 [遗留问题](open-questions.md)。
- 低层 `run_simulation_step`/memory API 删除了 synthetic AC 和 FP8 transport 投影参数；包内调用点已同步。正常 CLI 的已支持能力保持。
- 当前目录是独立实验 distribution，不会因导入生产包而自动激活。旧分支 simulator 保留，未将其其他模型 shim 整批带回。
