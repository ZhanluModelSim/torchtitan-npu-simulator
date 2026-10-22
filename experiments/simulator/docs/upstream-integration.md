# 上游同进程接入指南

本文供上游平台及其接入 AI 使用：将旧的 `torchtitan_npu.entry → SimulationTrainer.train → trainer.workload_graph` 调用迁移为同进程函数调用，直接取得 `WorkloadGraph`，交给现有消费者。当前不需要子进程；PP 的执行与接入方案后续另行设计。

接口基线为 `codex/simulator-titan030` 的提交 `848c32a`。运行时固定 TorchTitan v0.3.0 `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`、NPU master 快照 `7eb02313a7fb5ac764133f0c4d909a7e8ce44da1`。不据此假定其他 master 提交兼容。
参数的源码快照另见 [integration-parameters.json](integration-parameters.json)，它列出实际 argparse 注册、Python 字段和默认值；运行限制及字段间约束以本文和源码为准。

## 1. 接入方式与运行前提

优先使用 `simulate(SimulationConfig(...))`。若上游已有 CLI 字符串数组，可调用 `cli.main(argv)`，两者均在当前进程同步返回同一个类型的图。

| 旧接入契约 | 当前接口与接入动作 |
| --- | --- |
| 通过 `torchtitan_npu.entry` 启动 simulator | 改调 `torchtitan_simulator.runner.simulate` 或 `torchtitan_simulator.cli.main`；当前没有旧入口兼容层 |
| monkey-patch `SimulationTrainer.train()`，执行原方法后取成员 | 移除该 patch；当前 sidecar 没有 `SimulationTrainer`，直接接收函数返回值 |
| 导入 `torchtitan_npu.simulator.*` 的 IR 类 | 改为 `torchtitan_simulator.ir.*`；避免旧、新 distribution 类型混用 |
| 将全部原生训练参数直接透传 | 建立下文参数映射；新 CLI 是独立 argparse 白名单，不接受完整原生 ConfigManager 参数树 |
| 启动 torchrun／设置 NGPU 推导逻辑 world | 同进程直接传 `world_size`、`logical_rank`；不启动 torchrun、不自行初始化 process group |
| 取得图后由上游继续处理 | 保留此流程；读取返回对象，无须先导出再反序列化 |

运行条件：

- Python ≥3.12；PyTorch、torch_npu、完整 CANN、TorchTitan、torchtitan-npu 已配套安装。MXFP8 另需配套 TorchAO-NPU。sidecar 的安装依赖声明只有 `torch`，不能依靠一次 pip 安装自动补齐整套训练环境。
- 已验证环境为 Linux arm64、torch 2.14.0+cpu、torch_npu 2.14.0.dev20260918、完整官方 CANN 9.2.0-beta.2；无真实 NPU 的 Meta 仿真已验证。其他环境需要重新验收。
- 在 Python 进程导入 torch／simulator 前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，并按部署环境加载 CANN 环境。Python API 不会读取 `CANN_ENV_FILE` 并 source shell 脚本；该变量只由镜像入口脚本使用。
- 宿主当前不得已有初始化的 `torch.distributed` process group；否则报 `Simulator logical world must own its process group`。simulator 自行创建、销毁 fake process group。
- 单进程串行调用；禁止嵌套仿真，也不要并发运行多个仿真或与真实训练混用。多次顺序调用已有恢复测试，但没有并发安全承诺。原生常量缓存的冷／热状态会影响实际捕获到的初始化算子数量，不能要求重复调用得到完全相同的 trace。局部兼容 patch 退出时恢复，不代表厂商模块导入、DTensor cache 清理等所有进程级副作用都被撤销。

当前本机已验证镜像为 `torchtitan-simulator:titan030-dispatch-first`，位于 Docker context `colima-titan-migration-audit`。源码路径为仓库内 `experiments/simulator/`，从该目录安装：

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

镜像标签和 context 是本机验收信息；交付其他机器时需提供镜像或按 [Dockerfile](../Dockerfile) 构建，不能假定远端 registry 存在同名镜像。

## 2. 推荐的 Python 接入

以下示例在已准备好的运行环境中执行。环境变量应由宿主启动环境提前设置。

```python
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.ir.workload_graph import WorkloadGraph
from torchtitan_simulator.runner import export_result, simulate

options = SimulationConfig(
    config="deepseek_v4_smoketest",
    world_size=16,
    logical_rank=15,
    data_parallel_replicate_degree=2,
    tensor_parallel_degree=2,
    context_parallel_degree=2,
    expert_parallel_degree=2,
    seq_len=512,
    local_batch_size=1,
    num_mtp_modules=1,
    activation_checkpoint_mode="selective",
    output_dir="/audit/upstream-python",
    output_formats=["json", "csv", "mem"],
)
options.validate()
workload_graph = simulate(options)
assert isinstance(workload_graph, WorkloadGraph)
assert workload_graph.schedule_plan is not None
assert workload_graph.step_templates
assert "capture_validation" in workload_graph.iteration.schedule.annotations

# Pass workload_graph directly to the existing upstream consumer here.

# Optional: remove this call when the upstream only needs the in-memory graph.
export_result(workload_graph, options)
```

`simulate()` 不导出文件，但可产生日志；`output_dir`、`output_formats` 只有调用 `export_result()` 时才控制落盘。配置在准备阶段深拷贝，不会把派生后的运行配置写回传入对象；例如 `_mxfp8` preset 会启用量化，调用者应从返回图的 `quantization_configuration` 读取实际选择。

`SimulationConfig` 是 keyword-only dataclass，不是原生 Trainer.Config。不能传入旧 Trainer config、任意 dict 或额外训练字段。需要模型几何复写时：

```python
from torchtitan_simulator.model_overrides import ModelOverrides, MoEOverrides

geometry = ModelOverrides(
    n_layers=4,
    compress_ratios=(0, 0, 4, 128),
    moe_args=MoEOverrides(route_norm=False),
)
# Supply geometry as SimulationConfig(model_overrides=geometry, ...).
```

## 3. 保留字符串参数数组的接入

```python
from torchtitan_simulator.cli import main
from torchtitan_simulator.ir.workload_graph import WorkloadGraph

argv = [
    "--config", "deepseek_v4_smoketest",
    "--simulation.world-size", "16",
    "--simulation.logical-rank", "15",
    "--parallelism.data-parallel-replicate-degree", "2",
    "--parallelism.tensor-parallel-degree", "2",
    "--parallelism.context-parallel-degree", "2",
    "--parallelism.expert-parallel-degree", "2",
    "--training.seq-len", "512",
    "--training.local-batch-size", "1",
    "--training.num-mtp-modules", "1",
    "--activation-checkpoint.mode", "selective",
    "--simulation.output-dir", "/audit/upstream-argv",
    "--simulation.output-formats", "json", "csv", "mem",
]
workload_graph = main(argv)
assert isinstance(workload_graph, WorkloadGraph)
# Pass workload_graph directly to the existing upstream consumer here.
```

传入的是参数 token 列表，不包含 `python`、`-m` 或模块名，不应把整条 shell 命令当作一个字符串。始终显式传 `argv`；`main()` 无参数时会读取宿主的 `sys.argv`。该路径会自动调用 `export_result()`，适合确实需要标准输出文件的接入。

参数语法：

- 表中的点号和连字符是准确拼写；不要假定支持下划线别名或任意原生配置路径。选项名写全，不依赖 argparse 的缩写匹配。
- 普通数值／字符串用 `--option VALUE` 或 `--option=VALUE`；列表使用多个 token，不传 JSON 数组或逗号连接字符串。
- `store_true` 开关出现即为 true，关闭时省略；例如不能用 `--simulation.memory-offload-ac-saved-tensors false`。
- 几何字段 `route_norm` 有显式负向开关，见第 6 节。
- `-h`／`--help` 输出帮助并抛出 `SystemExit(0)`；未知参数、缺少参数值等解析错误抛出 `SystemExit(2)`。API 宿主若保留此调用方式，须按需捕获 `SystemExit`，不要把它当成图。
- `--build-only` 返回诊断 dict，不返回 `WorkloadGraph`，不得用于正式接入。

需要单独手工诊断时可运行 `python -m torchtitan_simulator --help`。不要在本次接入中使用安装的 `torchtitan-simulate` console script：当前 setuptools wrapper 调用 `sys.exit(main())`，而 `main()` 返回图／dict，构建成功也可退出为 1。该问题已由 `--build-only` 实测确认；`python -m` 与同进程函数调用不经过此 wrapper。本指南未修改该实现。

## 4. 全部常规参数

下表列出当前全部常规 CLI 参数及 Python 字段。默认值指 sidecar 配置默认值，不是所有 preset 最终模型字段的默认值。布尔项默认 false，除非另有说明。

| CLI 参数 | Python 字段 | 类型／默认值 | 可选值、行为与限制 |
| --- | --- | --- | --- |
| `--config` | `config` | str；`deepseek_v4_smoketest` | 七个 preset 或 `module.factory`；详见第 5 节 |
| `--simulation.world-size` | `world_size` | int；1 | ≥1；逻辑 world，不是捕获进程数 |
| `--simulation.logical-rank` | `logical_rank` | int；0 | `0 <= rank < world_size`；选择被捕获的逻辑 rank |
| `--simulation.target-npu-device-type` | `target_npu_device_type` | str；`non_a5` | parser 接受 `non_a5`／`a5`，当前运行时只支持 `non_a5`，`a5` 报错 |
| `--training.seq-len` | `seq_len` | int；128 | ≥1；原生模型、CP、压缩、量化还有 shape 约束；完整矩阵使用 512 |
| `--training.num-mtp-modules` | `num_mtp_modules` | int；0 | ≥0；已验证 0／1，其他值不等于已验收 |
| `--training.local-batch-size` | `local_batch_size` | int 或 None；None | 显式值 ≥1；None 沿用 preset；完整矩阵使用 1 |
| `--training.global-batch-size` | `global_batch_size` | int 或 None；None | 显式值 ≥1；None 沿用 preset；GBS／GA 约束见下文 |
| `--parallelism.data-parallel-replicate-degree` | `data_parallel_replicate_degree` | int；1 | ≥1 |
| `--parallelism.data-parallel-shard-degree` | `data_parallel_shard_degree` | int；-1 | -1 自动推导，否则 ≥1 |
| `--parallelism.tensor-parallel-degree` | `tensor_parallel_degree` | int；1 | ≥1；并行化仍受模型维度整除约束 |
| `--parallelism.context-parallel-degree` | `context_parallel_degree` | int；1 | ≥1；仍受原生 CP 输入布局约束 |
| `--parallelism.expert-parallel-degree` | `expert_parallel_degree` | int；1 | ≥1；不得乘入下面的 world 公式 |
| `--parallelism.pipeline-parallel-degree` | `pipeline_parallel_degree` | int；1 | 当前只允许 1，其他值在构图前拒绝 |
| `--activation-checkpoint.mode` | `activation_checkpoint_mode` | str；`none` | `none`／`full`／`selective` |
| `--simulation.selective-ac-save-ops` | `selective_ac_save_ops` | list[str] 或 None；None | CLI 至少一个 token；仅 selective 模式可设置；全部选项见第 7 节 |
| `--extension.quantization.enable-quantized-training` | `enable_quantized_training` | bool；false | `store_true`；使用原生 TorchAO-NPU recipe；`_mxfp8` preset 自动启用 |
| `--extension.quantization.recipe` | `quantization_recipe` | str；`all_mxfp8` | 唯一支持值 `all_mxfp8`；只设 recipe 不会使 BF16 preset 自动开启量化 |
| `--simulation.memory-parameter-storage-dtype` | `memory_parameter_storage_dtype` | str；空字符串 | 空值保留捕获 dtype；全部可选值见第 8 节；仅内存投影 |
| `--simulation.memory-offload-ac-saved-tensors` | `memory_offload_ac_saved_tensors` | bool；false | `store_true`；保存激活驻留投影，适用于 none/full/selective；不执行实际 offload |
| `--simulation.enable-fsdp-allgather-fp8` | `enable_fsdp_allgather_fp8` | bool；false | parser 保留，设置 true 会明确拒绝；不是已支持的优化 |
| `--simulation.enable-ep-dispatch-fp8` | `enable_ep_dispatch_fp8` | bool；false | parser 保留，设置 true 会明确拒绝 |
| `--simulation.output-dir` | `output_dir` | str；`./simulator_output` | 相对路径按宿主 cwd 解析；接入建议显式绝对路径且每次调用独立目录 |
| `--simulation.output-formats` | `output_formats` | list[str]；[] | `text html dot json csv mem` 的任意组合；CLI 可传零个值；详见第 10 节 |
| `--simulation.csv-max-ranks` | `csv_max_ranks` | int；1 | ≥1；仅限制 CSV 展示，不改变 world、mesh 或捕获 rank |
| `--build-only` | 无；CLI 专用 | bool；false | 仅诊断模型构建，返回 dict；不执行完整 F/B/O |
| `-h`／`--help` | 无；CLI 专用 | 无值 | 帮助后退出，不执行仿真 |

仅 Python API 支持的字段：

| Python 字段 | 默认值 | 用法与边界 |
| --- | --- | --- |
| `model_overrides` | None | `ModelOverrides` 实例，内含 `MoEOverrides`；CLI 对应第 6 节多个展开选项 |
| `override_imports` | None | `list[str]`；None 使用下列九个默认 override；显式列表完整替换默认列表，不是追加；空列表会移除默认选择，不能据此假定仍支持标准路径。自定义组合未由 21 项默认矩阵证明；CLI 无 `--override.imports` 参数 |

默认 `override_imports`：

```text
torchtitan_npu.override.common.rms_norm.asc
torchtitan_npu.override.common.token_dispatcher.asc
torchtitan_npu.override.common.rope.asc_complex
torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata
torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li
torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata
torchtitan_npu.override.deepseek_v4.sparse_attn.asc
torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre
torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post
```

并行与批约束：

```text
world_size = DP_replicate × DP_shard × TP × CP
DP_shard = world_size / (DP_replicate × TP × CP)  # when shard == -1
(DP_shard × TP × CP) % EP == 0
DP_degree = DP_replicate × DP_shard
GBS % (local_batch_size × DP_degree) == 0
GA = GBS / (local_batch_size × DP_degree)
```

不能整除或维度不合法时报错。EP 是原生并行 mesh 的重叠维度，不单独乘入 world。若 preset 的 GBS 为负数则推导为 `local_batch_size × DP_degree`；显式 CLI GBS 不能为负数。
即便配置 GA>1，当前仍只实际捕获一个 microbatch 和一次 optimizer 更新；token 分母包含配置的 GA。读取图的 `batch_configuration`，不得宣称完成多个 microbatch 的真实累积图／内存回放。

## 5. 模型 preset 与实际量化选择

| `--config`／`config` | 原生工厂／几何 | 量化行为 |
| --- | --- | --- |
| `deepseek_v4_smoketest` | `torchtitan_npu.models.deepseek_v4.config_registry.deepseek_v4_debugmodel`；原生 4 层／dim256 | 默认不开启；可显式开 MXFP8；不是旧版 dim128 smoketest |
| `deepseek_v4_flash_baseline_bf16` | 同包 `deepseek_v4_flash`；43 层／dim4096 | 默认不开启；可显式开 MXFP8 |
| `deepseek_v4_flash_baseline_mxfp8` | 同上 | 自动启用 MXFP8 |
| `deepseek_v4_pro_baseline_bf16` | 同包 `deepseek_v4_pro`；61 层／dim7168 | 默认不开启；可显式开 MXFP8 |
| `deepseek_v4_pro_baseline_mxfp8` | 同上 | 自动启用 MXFP8 |
| `deepseek_v4_pro_20t_baseline_bf16` | `torchtitan_simulator.presets.deepseek_v4_pro_20t`；96 层／dim12288／2048 专家 | 默认不开启；保留旧几何的适配 preset，不是上游新官方 preset |
| `deepseek_v4_pro_20t_baseline_mxfp8` | 同上 | 自动启用 MXFP8 |

不应仅根据 preset 名称判断实际精度；读取 `annotations["quantization_configuration"]` 的 `enabled`、`recipe` 和实际 `quantized_parameters`。

自定义 `module.factory` 入口会调用 `factory(num_mtp_layers=...)`，返回对象必须符合当前 native Trainer config 结构，且 `model_spec.name == "deepseek_v4"`。这不是任意模型插件接口；自定义工厂还不能使用本 sidecar 的平铺 `model_overrides`。工厂返回的配置仍会被 sidecar 的并行、AC、seq_len、override、量化设置覆盖；compile/CUDA graphs 被关闭，EP 使用均衡路由假设。需要不同能力时先扩展并验证实现，不通过私有对象随意修改字段。

## 6. 全部模型几何复写参数

以下字段均默认 None，表示沿用 preset 对应值；不是把模型字段设为 None。通过原生 builder 重建完整配置树，不直接赋值单个 layer。Python 顶层字段属于 `ModelOverrides`；MoE 字段属于 `ModelOverrides(moe_args=MoEOverrides(...))`。

| CLI 参数 | Python 字段 | 类型／允许值与含义 |
| --- | --- | --- |
| `--model-overrides.dim` | `dim` | int ≥1；hidden dim |
| `--model-overrides.n-layers` | `n_layers` | int ≥1；主干层数 |
| `--model-overrides.vocab-size` | `vocab_size` | int ≥1；词表大小，loss 配置同步重建 |
| `--model-overrides.n-heads` | `n_heads` | int ≥1；attention heads |
| `--model-overrides.head-dim` | `head_dim` | int ≥1 |
| `--model-overrides.rope-head-dim` | `rope_head_dim` | int ≥1；不得大于 head_dim 和 index_head_dim |
| `--model-overrides.q-lora-rank` | `q_lora_rank` | int ≥1 |
| `--model-overrides.o-lora-rank` | `o_lora_rank` | int ≥1 |
| `--model-overrides.o-groups` | `o_groups` | int ≥1；映射到原生 n_groups；须整除 n_heads |
| `--model-overrides.window-size` | `window_size` | int ≥1 |
| `--model-overrides.norm-eps` | `norm_eps` | 有限 float >0 |
| `--model-overrides.index-n-heads` | `index_n_heads` | int ≥1 |
| `--model-overrides.index-head-dim` | `index_head_dim` | int ≥1，且为 2 的幂 |
| `--model-overrides.index-topk` | `index_topk` | int ≥1 |
| `--model-overrides.moe-inter-dim` | `moe_inter_dim` | int ≥1；专家中间维度 |
| `--model-overrides.hc-mult` | `hc_mult` | int ≥1 |
| `--model-overrides.hc-sinkhorn-iters` | `hc_sinkhorn_iters` | int ≥1；映射到原生 sinkhorn_iters |
| `--model-overrides.hc-eps` | `hc_eps` | 有限 float >0 |
| `--model-overrides.compress-rope-theta` | `compress_rope_theta` | 有限 float >0 |
| `--model-overrides.original-seq-len` | `original_seq_len` | int ≥1 |
| `--model-overrides.rope-theta` | `rope_theta` | 有限 float >0 |
| `--model-overrides.rope-factor` | `rope_factor` | 有限 float >0 |
| `--model-overrides.load-balance-coeff` | `load_balance_coeff` | 有限 float >0；当前不能通过该复写字段设为 0 |
| `--model-overrides.compress-ratios` | `compress_ratios` | CLI 为一个或多个 int；Python 为 tuple[int,...]；元素为 0／1／4／128，0 映射为原生 1 |
| `--model-overrides.moe-args.num-experts` | `moe_args.num_experts` | int ≥1 |
| `--model-overrides.moe-args.num-shared-experts` | `moe_args.num_shared_experts` | int ≥0 |
| `--model-overrides.moe-args.top-k` | `moe_args.top_k` | int ≥1；≤ num_experts |
| `--model-overrides.moe-args.n-hash-layers` | `moe_args.n_hash_layers` | int ≥0 |
| `--model-overrides.moe-args.route-norm` | `moe_args.route_norm` | bool；出现即 true；省略沿用 preset |
| `--model-overrides.moe-args.route-scale` | `moe_args.route_scale` | 有限 float >0 |

`route_norm=False` 使用 `--model-overrides.moe-args.no-route-norm`。argparse 另生成别名 `--no-model-overrides.moe-args.route-norm`，效果相同；不要传字符串 `false`。一次调用只选一种正／负开关。

`compress_ratios` 必须至少覆盖复写后的全部主干层，超出部分按 n_layers 截取。增加 n_layers 时不能假定旧压缩列表会自动扩展。以上只是显式校验条件；原生 builder、TP/EP、kernel Meta 与量化分块还有联动约束，不保证任意合法标量的笛卡尔积都可执行。

## 7. Selective AC 的全部 save-set 选项

未设置 `selective_ac_save_ops`（None）时使用原生 policy 默认集合；设置后只有包含 `default` 才保留原生默认集合。多个组按集合合并，`none` 必须单独使用。它不是 `activation_checkpoint_mode="full"` 的另一种写法。

| token | 当前含义／限制 |
| --- | --- |
| `default` | 加入原生 policy 默认 save 集合 |
| `compute-intensive` | 加入当前 PyTorch partitioner 提供的 compute-intensive 集合 |
| `attention` | 注册的 attention 算子组，包括 `cann_ops_transformer.sparse_flash_mla.default` |
| `linear` | `aten.linear.default`；保留原生交替 MM/linear 重算策略 |
| `mm` | `aten.mm.default`；保留原生交替策略 |
| `gmm` | 原生 grouped MM 与 NPU grouped matmul |
| `quant-mm` | `npu.npu_quant_matmul.default`；不等于保存所有 MXFP8 算子 |
| `comm` | functional reduce-scatter／all-to-all-single；不代表所有 c10d 通信都可缓存 |
| `max` | `aten.max.default` |
| `full` | 展开为 default、compute-intensive、attention、linear、mm、gmm、quant-mm、comm、max 的并集；不是 Full AC 模式 |
| `none` | 不选择这些 SAC save 算子；必须独用 |
| `all-to-all` | 类型字面量中保留，但当前显式拒绝：旧 synthetic SAC 扩展未支持 |

CLI 示例：`--activation-checkpoint.mode selective --simulation.selective-ac-save-ops default gmm comm`。某组在当前环境无注册算子会报错；通信组选择后未实际命中时可产生 warning。读取真实保存／重算事件，不能仅据配置字符串推断缓存行为。

## 8. 参数存储 dtype 的全部选项

`memory_parameter_storage_dtype` 空字符串表示不投影；非空支持：

```text
float32 float64 float16 bfloat16 float8_e4m3fn float8_e5m2
int64 int32 int16 int8 uint8 bool
```

这些名称可带小写 `torch.` 前缀，内部去掉前缀后转小写；例如 `torch.bfloat16` 可用。**不支持** `bf16`、`fp16`、`fp8` 等缩写。建议始终使用表中的标准小写名称。

此字段只按目标每元素字节数投影常驻参数 shard，不改变 native 参数、梯度、AdamW 状态、量化 code/scale 或实际通信 dtype。整数／bool 代表字节预算假设，不表示训练支持整数／bool 参数。参数 dtype 投影与激活 offload 均不证明真实硬件能取得对应峰值。

## 9. 返回图的消费契约

```python
schedule = workload_graph.iteration.schedule
annotations = schedule.annotations
plan = workload_graph.schedule_plan
nodes = {
    op_id: node
    for template in workload_graph.step_templates.values()
    for op_id, node in template.nodes.items()
}
memory = annotations["memory_plan"]
peak_bytes = memory.peak_active_bytes
phase_peaks = {
    "forward": memory.forward_peak_active_bytes,
    "backward": memory.backward_peak_active_bytes,
    "optimizer": memory.optimizer_peak_active_bytes,
}
```

| 路径／字段 | 对象与消费规则 |
| --- | --- |
| `workload_graph` | `torchtitan_simulator.ir.workload_graph.WorkloadGraph`；完整执行和 validation 后返回；不是 Trainer、dict 或 JSON 字符串 |
| `step_templates` | `dict[str, StepGraph]`；当前 PP=1 标准路径为 `s0_F`、`s0_B`、`s0_OPTIMIZER`；key 等于 StepGraph.step_id |
| `template.nodes` | `dict[int, OpNode]`；predecessors/successors 可跨模板，先建立全局索引再解析依赖；不要逐模板丢弃外部边 |
| `node.op_type`／`node.annotations["raw_op_type"]` | 规范化类别／原始 dispatcher 名称；未知类别的原始节点仍保留；不得只消费已知 op_type 而丢弃 unknown |
| `node.inputs`／`outputs`／`attrs` | TensorMeta 列表和算子属性；包括原地写入 operands；不是可供真实运算的 tensor 数据 |
| `schedule_plan` | 权威 SchedulePlan；消费 actions、data_slots、template_ref 与实际 op 引用；非 PP 当前可能没有 DataSlot，不要硬性要求非空 |
| `iteration.schedule` | 保留 ScheduleGraph、实例、数据流与 annotations；不能只按 L0 CSV 行顺序另造调度 |
| `annotations["memory_plan"]` | Python 内存中为 MemoryPlan 对象，不是 dict；含实际 storage 生命周期、保存槽和各阶段峰值；导出 JSON 后才是序列化结构 |
| `annotations["communication_events"]` | 通信记录列表；保留实际 group、ranks、形状／payload 和图引用；原生名称不再固定为旧 `comm.*` |
| `annotations["capture_coordinates"]` | logical_rank/world、capture worker 和 pipeline_stage；当前单 worker 的 capture_process_rank=0，不等于被模拟 rank=0 |
| `annotations["batch_configuration"]` | GBS、local batch、DP、GA 与 captured_microbatches；当前 captured_microbatches=1 |
| `annotations["quantization_configuration"]` | 实际量化开关、recipe 和参数选择 |
| `annotations["model_configuration"]` | preset、原生工厂、几何复写与实际模型规模 |
| `annotations["gradient_validation"]` | 实际 trainable 参数梯度清单和 shape/dtype/placement |
| `annotations["capture_validation"]` | 本次图／梯度／通信／内存完整性检查结果；构建失败不会返回一张「占位成功图」 |
| `annotations["cost_unknown_counts"]` | 未知成本算子统计；图存在不代表已知硬件时延 |

算子名变化是接入重点：通信现在保留 `c10d.*`、`_c10d_functional.*`、`_dtensor.*` 的原生节点；optimizer 为 `aten._fused_adamw_.default`，不再生成旧 `npu.npu_apply_adam_w` 手工节点。不要按旧 raw name 写死识别；可参考 [communication_primitive](../torchtitan_simulator/capture/op_mapping.py) 的映射。

原始 alias／mutation 事实保留用于身份和内存分析；规范化图会收缩 alias-only metadata 路径。节点 ID 仅用于当前返回图内引用，不应跨运行比较 Python object identity、storage 地址或假定 op_id 稳定。L0 edges 与 L2 action/DataSlot 依赖分别消费，不靠节点列表顺序替代数据依赖。

逻辑内存指标为 bytes；GiB 换算除以 `2**30`。保存槽数量、去重 storage 数、offload inventory 的累计字节都不等于峰值。当前指标不含 allocator reserved/cache、碎片、kernel workspace；metadata 准备在训练捕获前完成，其准备成本不计入训练图。

## 10. 导出行为与可选格式

纯内存接入只调用 `simulate()` 即可；需要标准文件时显式调用 `export_result(workload, options)`。`cli.main(argv)` 自动导出，**即使 output_formats=[] 也会创建 output_dir，写 export_manifest.json 和 memory/memory_summary.json**。

| 格式 token | 主要输出 |
| --- | --- |
| `text` | `simulation_result.txt` |
| `html` | `simulation_result.html` |
| `dot` | `simulation_result.dot`，包含跨模板边 |
| `json` | `simulation_result.json`，序列化 WorkloadGraph |
| `csv` | `kernel_summary/rank_*.csv`、`s0_*_l0.csv`、`schedule_plan.csv` |
| `mem` | `memory/` 下的事件、tensor 生命周期、保存值和 FSDP 驻留明细 |

格式列表未知值报错。CSV 的 rank 行为是同一代表性捕获模板的展示投影，不是每个 rank 独立捕获；`csv_max_ranks` 不改变实际模拟 world。`schedule_plan.csv` 包含 Actions 与 `# DataSlots` 分隔的两段，不能把后半部分当成 action 行。export_manifest schema_version=2；跨机器消费者如需序列化应消费该版本契约，而非依赖 dataclass 的私有实现。

导出函数不是原子写入事务，失败时可能留下部分文件；不同请求应使用不同目录。`simulate()` 与导出分离可让上游在导出失败后仍持有已生成的图。

## 11. 旧参数迁移与不支持项

| 旧字段／做法 | 当前处置 |
| --- | --- |
| 旧 `--model.*` 平铺几何 | 映射到第 6 节 `--model-overrides.*`；不要盲目去掉前缀，字段层次和命名可能改变 |
| `--training.steps` | 无 CLI／SimulationConfig 字段；标准路径固定一次 microbatch + optimizer 捕获；不据 workload 的迭代计数推断执行了多步 |
| `--comm.mode`、`--simulation.simulated-parallel-degrees` | 无公开参数；由显式 world／rank／parallelism 建立逻辑 fake world |
| `NGPU`、`TORCHTITAN_SIM_WORLD_SIZE`、`WORLD_SIZE`／`RANK` | 新 CLI 不据此推导 simulation world／rank；调用者显式传值，不把 torchrun worker 数当作逻辑 world |
| `--simulation.enable-memory-tracking` | 不再提供开关；当前标准运行总会采集内存 |
| 旧 HF assets／真实 dataloader 参数 | 新 CLI 不接收；当前使用合成单文档 next-token 输入、均衡 EP 路由 |
| compile、CUDA graphs、旧 model converter 列表 | 无通用参数透传；dispatch capture 关闭 compile/CUDA graphs，使用默认原生 override 配置 |
| 旧 MXFP8 FQN 覆写、任意 recipe | 不支持；沿用原生 all_mxfp8 的实际选择 |
| PP>1、FP8 AG/A2A、A5 | 当前拒绝；不移除报错或静默回退；PP 后续单独设计 |
| 其他模型、任意 optimizer／checkpoint／真实训练参数 | 不属于当前公开配置范围；不得以当前参数字典透传给 SimulationConfig |

移除未知旧字段时应在接入层显式列出转换规则或报出「当前不支持」，不要使用 parse_known_args、静默过滤任意字段，或在失败时退回 build-only。不能把无效旧参数「忽略后成功」视为等价迁移。

## 12. 交给接入 AI 的执行清单

1. 找到旧入口调用、`SimulationTrainer.train` patch、旧 IR imports、参数序列化和图消费者；保留消费者入口，替换生成图的调用。
2. 选择 `simulate(SimulationConfig)`（推荐）或 `cli.main(argv)`；当前只做同进程串行接入，不增加子进程／torchrun／PP 实现。
3. 依据第 4～8 节建立显式字段映射。未知字段及未支持项返回可定位错误；布尔、列表和 None 按各自接口正确编码。
4. 从返回的 WorkloadGraph 读取权威计划和全局节点；更新旧通信／optimizer raw name 识别；保留跨模板边、unknown 节点、dtype 和实际 tensor/storage 来源信息。
5. 完整仿真异常不得包装成空图。CLI 解析的 SystemExit 与仿真 ValueError/RuntimeError、导出 OSError 分开处理；记录原始异常并交给调用方。
6. 用小模型做同进程完整 F/B/O 接入验收；检查返回类型、非空模板、schedule_plan、capture_validation、梯度清单、memory_plan 及消费者实际处理成功。不要仅凭 build-only 通过。
7. 在同一进程连续运行至少两次（可采用下方验收脚本），确认前后 process group 未残留、两张图都可由消费者读取；不在本次接入引入并发运行。
8. 按平台实际使用增加 selective/MXFP8、world/rank 与关键并行组合；GBS>单 MB 时检查 captured_microbatches 的边界。保留失败信息，不扩大成「全部参数笛卡尔积均支持」。

以下检查基于第 2 节已创建的 `options`，再连续执行两次；也可将 `simulate(options)` 替换成第 3 节的 `main(argv)`：

```python
import torch.distributed as dist
from torchtitan_simulator.runner import simulate

for _ in range(2):
    assert not dist.is_initialized()
    workload_graph = simulate(options)
    assert not dist.is_initialized()
    assert workload_graph.schedule_plan is not None
    assert workload_graph.step_templates
    annotations = workload_graph.iteration.schedule.annotations
    assert "capture_validation" in annotations
    assert "memory_plan" in annotations
    coordinates = annotations["capture_coordinates"]
    assert coordinates["logical_world_size"] == 16
    assert coordinates["logical_rank"] == 15
    assert all(key == step.step_id for key, step in workload_graph.step_templates.items())
    # Invoke the actual upstream consumer here as part of integration acceptance.
```

## 13. 事实依据与文档验证

源码依据：[CLI](../torchtitan_simulator/cli.py)、[配置及校验](../torchtitan_simulator/config.py)、[执行与导出](../torchtitan_simulator/runner.py)、[几何复写](../torchtitan_simulator/model_overrides.py)、[SAC 选择](../torchtitan_simulator/selective_ac.py)、[dtype 口径](../torchtitan_simulator/capture/tensor_utils.py)、[逻辑 world 生命周期](../torchtitan_simulator/compat/distributed.py)、[WorkloadGraph](../torchtitan_simulator/ir/workload_graph.py)。已验证能力的完整范围见 [能力矩阵](capability-matrix.md) 和 [原生 dispatch 复验](dispatch-first-remediation.md)。

| 文档检查／示例 | 状态 | 结果与边界 |
| --- | --- | --- |
| 全部参数与可选值 | passed | 对照当前 parser 的 58 个 action（含 help、负向别名）及 Python 配置字段；逐项确认所有选项拼写都在本文中，机器快照与源码一致 |
| 第 2／3 节原样代码块，在同一进程依次执行 | passed | world16/rank15、BF16、MTP1、selective；两次均返回 WorkloadGraph、143 个参数梯度、357 个通信事件、266 个保存槽；调用前后 process group 均未初始化，第一张图在第二次调用后仍可读取 |
| 两次调用的逻辑内存峰值 | passed | 均为 300,400,844 bytes；不以此推断所有几何和配置在重复调用时均相等 |
| 参数 help、Markdown 链接与 Python 示例语法 | passed | 固定环境执行模块 help；相对链接与代码语法静态核对 |
| 文档 lint | passed | `python3 -m pre_commit run --files experiments/simulator/README.md experiments/simulator/docs/upstream-integration.md experiments/simulator/docs/integration-parameters.json` 通过；未涉及的 Python hooks 跳过 |
| 全仓 lint | failed | `python3 -m pre_commit run --all-files` 的常规／Ruff／架构／OAT 通过；主机 Pyrefly 找不到可执行文件，不能声明全仓 lint 全绿；本次未改运行代码 |
| 重新安装 sidecar | not-run | 固定环境已安装相同源码，未重复执行 pip 安装；安装命令从当前 pyproject 核对 |
| console script 的退出码 | failed | `torchtitan-simulate --build-only` 构建成功但退出为 1；本接入明确不使用该 wrapper |

两次示例的图节点分别为 5,708／5,679，差值为首轮 indexer 内 `_hadamard` 的 1 个 ones、21 个 cat、7 个 neg；目标原生 `torchtitan_npu/models/deepseek_v4/compressor.py::_hadamard` 使用 `@cache`，后续调用复用常量。因此本轮证明同进程顺序取图、验证和清理可用，不承诺冷启动与热启动的算子计数完全相同，也不在本次接入中人为补造缓存命中的初始化节点。

本轮验证证据保存在本机审计目录 `torchtitan-migration-audit-20261020/dispatch-first-remediation/` 的 `integration-contract.json`、`integration-example-validation.json`、`integration-examples.log`、`integration-console-exit.txt` 和 `integration-document-validation.json`。本次只修改文档和导航，运行源码未修改；未重新执行昂贵的 21 项矩阵，历史实现验收结果见上述能力矩阵。

真实 NPU 数值／HCCL、并发宿主混用及 PP 均未由本接入文档验证。本文仅说明现有接口，不补充旧入口兼容层或扩展运行能力。
