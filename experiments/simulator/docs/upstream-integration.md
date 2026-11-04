# 同进程调用与结果交付

本指南供平台或脚本调用已经接入的训练模型；新模型开发见[模型接入流程](model-integration.md)。使用普通安装或 editable 安装后的 `torchtitan_simulator`，不把源码目录加入 `sys.path`，不安装第二套捕获器。

## 环境与入口

导入前加载实际 CANN 环境并设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。主包、sidecar、torchtitan 和可选 `torchao_npu` 必须配套；实际路径检查见[开发导航](developer-guide.md)。同进程允许串行调用，不支持宿主已初始化 process group、嵌套或并发会话。

```python
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate

options = SimulationConfig(
    config="kimi_k3_smoketest",
    seq_len=16,
    world_size=4,
    expert_parallel_degree=2,
    activation_checkpoint_mode="selective",
)
workload = simulate(options)
annotations = workload.iteration.schedule.annotations
summary = annotations["memory_plan"].to_summary_dict()
assert summary == annotations["memory_summary"]
assert workload.step_templates and workload.schedule_plan is not None
```

返回 `WorkloadGraph`，不是文件路径或性能结果。失败保留原异常，不返回空图，不退回 build-only。默认不创建目录；显式 `output_formats=["mem"]` 才导出显存文件。

需要兼容已有字符串参数数组的调用方，可直接调用同一个 CLI 函数：

```python
from torchtitan_simulator.cli import main

workload = main([
    "--config", "kimi_k3_smoketest",
    "--training.seq-len", "16",
    "--simulation.world-size", "4",
    "--parallelism.expert-parallel-degree", "2",
])
```

`main(argv)` 完整模式返回图，`--build-only` 返回构建报告。shell 使用 `python -m torchtitan_simulator`；当前 setuptools 的 `torchtitan-simulate` 直接绑定返回对象的 `cli.main`，可能把返回对象传给 `sys.exit` 而退出为非零，不将其作为推荐 shell 入口。该已知入口问题不影响模块入口或 Python API。

## 配置位置与常用映射

参数真源为 [SimulationConfig](../torchtitan_simulator/config.py) 和 [CLI](../torchtitan_simulator/cli.py)，可运行 `python -m torchtitan_simulator --help` 查全部 CLI 选项，不维护另一份冻结 parser JSON。

| Python 字段 | CLI | 含义 |
| --- | --- | --- |
| `config` | `--config` | 已支持 preset，或满足当前模型族契约的 `module.factory` |
| `world_size`／`logical_rank` | `--simulation.world-size`／`--simulation.logical-rank` | 逻辑 world 及本次捕获 rank |
| `seq_len`／`num_mtp_modules` | `--training.seq-len`／`--training.num-mtp-modules` | 序列长度及 MTP 数量，仍受模型结构限制 |
| `local_batch_size`／`global_batch_size` | `--training.local-batch-size`／`--training.global-batch-size` | 原生 batch 配置，推导 GA；不把正式模板扩成 GA 份 |
| `data_parallel_replicate_degree` 等并行字段 | `--parallelism.data-parallel-replicate-degree` 等 | DP replicate/shard、TP、CP、EP；PP 只能为 1 |
| `activation_checkpoint_mode` | `--activation-checkpoint.mode` | `none`／`full`／`selective` |
| `selective_ac_save_ops` | `--simulation.selective-ac-save-ops` | selective 保存集合，未传使用默认，空列表与未传不等价 |
| `training_dtype` | `--training.dtype` | 原始参数 dtype；当前完整捕获使用 float32 |
| `mixed_precision_param`／`mixed_precision_reduce` | `--training.mixed-precision-param`／`--training.mixed-precision-reduce` | 实际计算参数与归约 dtype，不是纯内存假设 |
| `optimizer_name`／`muon_ns_steps` | `--optimizer.name`／`--optimizer.muon-ns-steps` | 保留 recipe 默认或显式选择 AdamW／DSV4 Muon；可选覆盖 Muon 迭代次数；见 [Muon 范围与依赖](muon.md) |
| `enable_quantized_training` | `--extension.quantization.enable-quantized-training` | 显式启用原生 MXFP8，默认关闭 |
| `quantization_recipe` | `--extension.quantization.recipe` | 当前仅 `all_mxfp8` |
| `memory_parameter_storage_dtype` | `--simulation.memory-parameter-storage-dtype` | 仅投影常驻本地参数 shard 字节 |
| `memory_offload_ac_saved_tensors` | `--simulation.memory-offload-ac-saved-tensors` | 激活卸载执行图及静态预算 |
| `memory_offload_optimizer_states` | `--simulation.memory-offload-optimizer-states` | AdamW 独立优化器状态卸载；Muon 当前拒绝 |
| `output_dir`／`output_formats` | `--simulation.output-dir`／`--simulation.output-formats mem` | 可选显存导出位置／格式 |
| `model_overrides` | `--model-overrides.*` | 类型化 DSV4 几何覆盖，重建完整配置树 |
| `override_imports` | 无对应通用 CLI | Python 显式替换默认 override 列表；并非追加，慎用空列表 |

默认参数和约束应读当前代码，不根据环境 `WORLD_SIZE`／`RANK`／`NGPU` 自动推导。`world_size = DP replicate × DP shard × TP × CP`，shard=-1 自动推导；EP 须整除 DP shard × TP × CP。PP、FP8 通信及 A5 当前明确拒绝，不能删除校验制造假支持。

### 模型几何与 selective AC

`ModelOverrides`、`MoEOverrides` 的字段、CLI 注册和重建逻辑归 [model_overrides.py](../torchtitan_simulator/model_overrides.py)。当前仅内置 DSV4 预设支持该平铺覆盖；Kimi 或自定义 factory 自己维护完整原生配置树，不能直接修改最终对象的几个维度而留下旧投影矩阵配置。`compress_ratios` 在 Python 中为 tuple，CLI 为多值列表；几何完整性由重建后的原生配置校验。

Selective AC 的合法集合与组合规则归 [selective_ac.py](../torchtitan_simulator/selective_ac.py)。当前 token 包括 `full`、`none`、`default`、`compute-intensive`、`attention`、`linear`、`mm`、`gmm`、`quant-mm`、`comm`、`all-to-all`、`max`，但合法名字不意味着所有组合都支持；`none` 独占，额外 `all-to-all` 扩展当前拒绝。默认原生 collective 策略不等于额外扩展。只有 selective 模式接受 save-set。

MXFP8 由开关选择，不使用旧 `_baseline_bf16`／`_baseline_mxfp8` preset，也不按模型名推断量化。完整精度和模型支持范围见[能力矩阵](capability-matrix.md)。

## 返回对象的消费边界

| 位置 | 内容和约束 |
| --- | --- |
| `workload.step_templates` | F/B/O 等原生捕获模板，节点保留数据／控制依赖；遍历全体模板建立全局 ID 索引 |
| `workload.schedule_plan` | 原生 action／slot 计划；不能与平台 L1/L2 命名直接等同 |
| `workload.iteration.schedule.annotations` | 捕获事实、配置、梯度验证、通信和静态内存等注记 |
| 同级 `capture` | 实际预热／捕获边界，见稳态捕获专题；旧结果缺失不补造 |
| 同级 `memory_plan`／`memory_summary` | 正式静态内存对象和摘要，单位 bytes，不依赖文件 |
| `workload.captured_workload` | 完整源图，用于审计；默认下游消费返回的 FSDP 通信视图 |
| `node.attrs["tensor_semantics"]` | 自定义算子的真实角色及轴，见[语义说明](custom-tensor-semantics.md) |

F 内可能交错 lm_head/loss backward，不能只按 `phase` 切 F/B。原始 ID 不重编号，也不承诺跨次调用恒定。通信组、成员顺序、owner 和 stream/event 依赖保留；`stream_wait` 零耗时但必须等待前驱，不能作为缺模型算子丢弃。详见[捕获契约](capture-runtime.md)及[流控制](stream-control-dependencies.md)。

本仓不交付 FLOPs 或预测时间；未知算子的内存诊断不是性能 miss。四项峰值是本 rank、单 microbatch 捕获下的静态逻辑 storage 预算，不乘 GA，不加 allocator reserved/cache。优化器状态与激活驻留的处理见[静态显存](steady-state-memory.md)。

## 导出与接入验收

`output_formats=[]` 不写文件；`["mem"]` 写 `memory/memory_summary.json` 和事件／timeline／storage／保存值／FSDP 等诊断明细。`export_result()` 也按该选项控制。无 JSON/HTML/DOT 图导出；平台自行保存其公共图快照。导出非原子事务，I/O 失败可能留下部分文件，每次使用独立目录，旧文件不自动清理。

接入至少运行一次完整 F/B/O，再在同一进程顺序执行第二次，检查返回图可消费、梯度齐全、图无环、状态不泄漏，且第一张图仍可读取。验证缺失与无状态零值的区别，不能只比较输出文件是否存在。真实模型测试入口见 `tests/test_dsv4_capture.py`、`tests/test_kimi_capture.py`、`tests/test_memory_delivery.py` 和 `tests/test_runtime.py`；新模型步骤见模型接入流程。
