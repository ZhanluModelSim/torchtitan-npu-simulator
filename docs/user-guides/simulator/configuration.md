# Simulator 配置与建模开关

> 现状核对：2026-10-07，`feat/npu-simulator`，实现基线 `ecde2f0`。依据源码与 Git 历史静态核对，未执行测试或仿真；后续行为以当前源码为准。

返回[使用入口](../simulator.md)。字段来源为 [SimulationConfig](../../../torchtitan_npu/simulator/trainer.py) 和 [preset registry](../../../torchtitan_npu/simulator/config_registry.py)。

## 核心字段

| 配置 | 当前默认 | 语义/约束 |
|---|---|---|
| `simulation.world_size` | None | 最终 logical device 数量，先于 mesh 和 spawn 解析 |
| `simulation.simulated_parallel_degrees` | 空字典 | 最终并行配置的兼容快照，运行时重建，不应手工维护第二套参数 |
| `simulation.output_dir` | 通用为 `./simulator_output`，preset 指定子目录 | PP 每 worker 导出 `rank_N/`，不自动 merge |
| `simulation.output_formats` | `[]` | `text/html/dot/json/csv/mem` 显式选择；默认不会导出全套图文件 |
| `simulation.enable_memory_tracking` | True | 控制 raw memory/saved capture 和估算；开启即独立导出 summary |
| `simulation.memory_parameter_storage_dtype` | 空字符串 | 仅覆盖本地持久参数 shard 驻留假设，不改变 compute/grad/state/full param dtype |
| `simulation.memory_offload_ac_saved_tensors` | False | none/full/selective 的保存激活统一建模为 0 设备驻留；仍保留逻辑字节和回捞统计 |
| `simulation.enable_fsdp_allgather_fp8` | False | 修改 FSDP AG wire metadata/bytes 和短期 staging 驻留；full params/compute dtype 保持捕获值 |
| `simulation.enable_ep_dispatch_fp8` | False | NpuExpertParallel dispatch 模型：E4M3 payload + 每 32 元素 1 byte E8M0 scale；combine 仍 BF16 |
| `simulation.selective_ac_save_ops` | None | 仅 selective 模式允许；None 保留原 policy，显式值覆盖保存集合 |
| `simulation.csv_max_ranks` | None | 限制 CSV logical rank 展开；不改变 world/mesh/捕获 |
| `simulation.target_npu_device_type` | `non_a5` | 当前检索到字段及默认值测试，未发现生产消费点；不能视作已生效 A5 路径选择 |
| `simulation.replicate_embedding_and_first_layer` | False，DeepSeek 专属 | embedding/global layer 0 在自身 DP/EDP mesh 复制；routed expert 仍按 EP 分区 |
| `model_overrides.*` | 来自模型 preset | 模型规模、attention/LoRA/compression/MoE/mHC 或 KDA/MLA/LatentMoE 参数与联动校验 |
| `mxfp8_fqns` | None | 需要恰好一个 MXFP8 converter；显式设置其转换范围 |
| `parallelism.*` / `training.*` | 来自真实训练 preset | world/batch、PP schedule/microbatch、TP/CP/EP、mixed precision 等真实训练选项被复用 |

SAC choices：`full`、`none`、`default`、`compute-intensive`、`attention`、`linear`、`mm`、`gmm`、`quant-mm`、`comm`、`all-to-all`、`max`。

- `none` 必须单独选择；`full` 是类别并集别名，**不含**显式 opt-in 的 `all-to-all`。
- `mm/linear` 保留上游交替重计算规则，所以 save-ops 的 `full` 不意味着逐次全部保存。
- fake `comm.all_to_all` 和上游 `_c10d_functional.all_to_all_single` 是不同目标；`default/comm/full` 不隐式开启前者。
- DeepSeek 要保留默认及模型 GMM/quant-MM 扩展，同时缓存 fake A2A，可显式选择 `default gmm quant-mm all-to-all`。
- 缓存 A2A 删去对应 recompute 通信，正常 backward 梯度 A2A 仍存在，并增加保存激活大小。

## 启动与并行参数

`world_size = PP × DP_replicate × DP_shard × CP × TP`。EP/ETP 是 dense world 的重解释，不是额外乘数：

```text
EFSDP = DP_shard × CP × TP / (EP × ETP)
EDP = DP_replicate × EFSDP
```

`EP × ETP` 必须整除 `DP_shard × CP × TP`，ETP 必须为 1 或 TP；模型自身还有[额外限制](model-matrix.md)。`DP_shard=-1` 由最终 world_size 推导。world_size 来源依次是显式 `--simulation.world-size`、`NGPU`、spawn 内部的 `TORCHTITAN_SIM_WORLD_SIZE`、配置或可推导值。正式 baseline 没有固定 384 卡默认值。

PP=1 自动选择 `fake_backend`，PP>1 自动选择 `multi_proc_meta` 并启动 PP 个真实进程。推荐使用 launcher，使 CLI 覆盖先于进程数量确定。外部 `torchrun` 须自行保证 `nproc_per_node=PP`、真实进程 `RANK=0..PP-1`、`WORLD_SIZE=PP`；不要将完整逻辑 mesh 的代表 rank 当作 Gloo 的 RANK。

PP 的 `pipeline_parallel_microbatch_size` 与梯度累积是不同概念：

```text
pipeline_microbatches = local_batch_size / pipeline_parallel_microbatch_size
global_batch_size = local_batch_size × DP_replicate × DP_shard × gradient_accumulation_steps
```

非 PP 路径按 `gradient_accumulation_steps` 读取真实的多个输入批次并依次执行前反向，以所有有效 token 的总数归一化 loss；一次训练 step 只清梯度一次、裁剪一次、更新优化器和学习率一次。独立调用 `run_simulation_step` 时 GA>1 必须提供对应数量的 `microbatches`。若只有 meta labels，必须显式提供 `local_valid_tokens` 或 `global_valid_tokens`，否则报错；不能由 shape 推断 IGNORE_INDEX 的数量。当前 PP 与 GA>1 的组合显式拒绝，PP 内部的 microbatch 切分仍由 schedule 执行。

当前 PP 多 microbatch 且包含 optimizer 的内存回放尚未实现跨 MB 的梯度累积绑定，会显式报错。可关闭 `simulation.enable_memory_tracking` 检查计算模板，但这不代表已捕获完整的累积梯度图；单 microbatch 内存回放和无 optimizer 的激活回放不受此限制。

PP 开启时 local batch 必须能被 PP microbatch size 整除。microbatch 数少于 stage 数不应直接写成非法；具体 schedule 的要求和效率需分别判断。并行参数检查脚本为 `scripts/validate_parallel_config.py`，运行命令仍需满足模型约束。

## 常用 CLI 覆盖

```bash
--activation-checkpoint.mode none
--activation-checkpoint.mode full
--activation-checkpoint.mode selective
--simulation.selective-ac-save-ops default gmm quant-mm all-to-all
--simulation.memory-parameter-storage-dtype bfloat16
--simulation.memory-offload-ac-saved-tensors
--simulation.enable-fsdp-allgather-fp8
--simulation.enable-ep-dispatch-fp8
--simulation.replicate-embedding-and-first-layer
--simulation.csv-max-ranks 4
--simulation.output-formats text html csv mem
```

以上是独立选项示例，不能一次组合成所有模型都可用的命令：SAC save-ops 只适用于 selective；最后的 replica 选项只属于 DeepSeek；EP FP8 和 synthetic AllToAll 缓存接入范围见矩阵。布尔字段关闭时用 `--simulation.no-<字段名>`，例如 `--simulation.no-enable-memory-tracking`。

模型规模通过 `--model-overrides.<字段名>` 覆盖，CLI 将 schema 的下划线转换为连字符。布尔字段用正/负开关，可空字段传 `None`，列表字段逐项传值。详细字段与联动校验见[DeepSeek V4 参数目录](../deepseek_v4_model_parameters.md)和[Kimi K3 参数目录](../kimi_k3_model_parameters.md)。

```bash
NGPU=8 python3 scripts/run_simulator_spawn.py \
    --config kimi_k3_smoketest \
    --model-overrides.n-layers 3 \
    --model-overrides.dim 192 \
    --model-overrides.kda-layers 0 1 \
    --model-overrides.num-experts 16 \
    --model-overrides.router-top-k 4 \
    --simulation.output-formats text
```

新增仿真 preset 应复用真实训练配置，参考当前 registry 的 `_simulation_config` / `_kimi_k3_simulation_config`。不要继续调用旧说明中的 `_to_simulation_config`。`dataclasses.replace()` 是浅拷贝，嵌套配置若需隔离必须单独复制；通信模式和 `simulated_parallel_degrees` 由最终 runtime 生成。
