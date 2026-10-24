# mem 导出收敛与原生精度边界

后续新增的稳态优化器状态驻留汇总见 [稳态内存说明](steady-state-memory.md)；本文结果与 SHA256 保留本阶段冻结快照，不代表后续源码复验。

当前 Simulator 只保留 `mem` 文件导出，同进程仍返回完整 WorkloadGraph。参数存储投影继续支持；实际训练精度沿用原生配置与算子，不通过后处理重写 dtype。

## 接入变化

- 删除完整图的 text、HTML、DOT、JSON、kernel/schedule CSV、manifest，以及 IR 上的文件导出方法。
- `output_formats=[]` 为默认值，不创建目录、不写文件。显式 `--simulation.output-formats mem` 才导出内存资料。
- 删除 `--simulation.csv-max-ranks`；旧格式 token 明确报错。更新上游已保存的 argv。
- mem 目录中的 JSON/CSV、memory trace 保留，它们用于内存分析，不是完整图序列化。旧目录的历史文件不会自动删除。
- 删除 metadata view 折叠，原始 view/alias 节点保留在图中，FLOPs 仍为零。节点增多不意味着分配增加；storage 去重与生命周期仍按原始捕获计算。
- 保留 L0–L3 构建、通信归属与 action/DataSlot 引用校验。这些负责表达真实执行依赖，不是展示折叠。

Python `simulate()` 不落盘，`cli.main(argv)` 自动调用按选项控制的 `export_result()`。返回图结构仍是上游消费者的输入。

## 实际精度与可选范围

| 含义 | CLI | Python 字段 | 可选值与验证边界 |
| --- | --- | --- | --- |
| 基础参数 dtype | `--training.dtype` | `training_dtype` | `float32`、`bfloat16`；DSV4 完整捕获仅支持 FP32，BF16 仅模型构建诊断 |
| FSDP 混合精度参数 | `--training.mixed-precision-param` | `mixed_precision_param` | `bfloat16`、`float32`；FP32 base 下完整 Meta F/B/O 已验证；DSV4 有原生 FP32 模块例外 |
| FSDP 梯度归约 | `--training.mixed-precision-reduce` | `mixed_precision_reduce` | 仅 `float32`；当前固定上游 TrainingConfig 的公开范围 |
| MXFP8 计算 | `--extension.quantization.enable-quantized-training` | `enable_quantized_training` | 原生 `all_mxfp8` recipe 选中的算子，保留实际 code/scale；不代表全部 tensor 都变成 FP8 |

三个 dtype 字段默认均为 `None`，沿用原生 factory；当前内置 preset 对应 `float32 / bfloat16 / float32`。
使用规范名称；不支持 `16`、`bf16`、`fp16` 等别名。`TORCH_DTYPE_MAP` 包含 float16 不等于原生 TrainingConfig 公开支持 FP16；当前不开放 FP16 训练或 BF16/FP16 归约。

推荐保持：

```bash
python -m torchtitan_simulator \
  --config deepseek_v4_smoketest \
  --training.dtype float32 \
  --training.mixed-precision-param bfloat16 \
  --training.mixed-precision-reduce float32 \
  --simulation.output-formats mem \
  --simulation.output-dir /tmp/simulator-mem
```

这里 BF16 是混合精度参数／计算路径。常驻参数及其梯度保持 FP32，DSV4 路由等特定组件仍按原生策略使用 FP32；优化器状态、cast buffer、通信 dtype 和字节均由实际原生执行捕获。
原生 FSDP 会将混合 BF16/FP32 参数组按 `uint8` 字节视图打包传输。因此部分 all-gather 节点显示 uint8 是正常的原生行为，不是 8bit 量化：payload 字节数仍来自原始 BF16/FP32 数据，并没有按元素压缩。

选择 `mixed_precision_param=float32` 会改变原生执行，而不是仅将结果中的 dtype 字符串改名。实际配置保存在 `annotations["precision_configuration"]`，参数和梯度 dtype 分别在 `parameter_layouts` 与 `gradient_validation`。

### 以前 base BF16 为什么会失败

构建 BF16 模型时，DSV4 的 `attn_sink`、compressor `ape` 等仍显式为 FP32。同一个原生 FSDP 参数组内出现两种原始 dtype，PyTorch 在 `_init_mp_dtypes` 报错：

```text
AssertionError: FSDP expects uniform original parameter dtype but got {torch.bfloat16, torch.float32}
```

本轮在 world4、EP2、MTP1 的实际模型上复现该错误。原生 per-FQN mixed precision policy 只指定计算 dtype，不能消除原始参数 dtype 混合。
因此完整捕获在模型／分布式构建前拒绝 base BF16 并解释原因，不改写参数、不绕过 FSDP 检查。`--build-only --training.dtype bfloat16` 仍可检查原生模型构建，但不能作为完整捕获成功的依据。

同时修复已发现的 Simulator 缺口：此前 `build_meta_model()` 没有读取 `training.dtype`，还受宿主进程默认 dtype 影响。现在复用原生 `set_default_dtype` 上下文，构建后恢复宿主默认 dtype；自定义 factory 不再被静默忽略。

## 存储精度假设仍需要独立投影

`--simulation.memory-parameter-storage-dtype` 仅改变常驻本地 `parameter_shard` 的 `modeled_dtype / modeled_num_bytes`；原始 dtype、num_bytes、计算图、梯度、优化器、全参数驻留与通信不变。
因此它可以用于「假设常驻权重以某 dtype 存储」的局部内存预算，不能代替完整低精度训练，也不能用于估算 FP8 all-gather。
这个统一投影也会作用于原生刻意 FP32 的常驻参数，使用者应明确接受这一假设。

支持的投影名称为：`float32`、`float64`、`float16`、`bfloat16`、`float8_e4m3fn`、`float8_e5m2`、`int64`、`int32`、`int16`、`int8`、`uint8`、`bool`。允许可选的 `torch.` 前缀；整数／bool 只是字节假设，不代表模型可按该 dtype 训练。

`memory_offload_ac_saved_tensors` 也继续作为显式驻留投影保留，不执行传输。AG/A2A 的 8bit 特殊假设仍未支持，入口继续拒绝。
原生已提供的 BF16 混合精度与 MXFP8 计算，不需要额外做图 dtype 后处理。

## 验证与证据

基线为 `c990d47`，固定 TorchTitan v0.3.0 与配套 NPU 主线源码；使用既有官方完整 CANN 9.2.0-beta.2 容器，无真实 NPU。
测试入口为 [test_precision.py](../tests/test_precision.py)、[test_memory_delivery.py](../tests/test_memory_delivery.py)、完整 [tests](../tests/) 与更新后的 [acceptance.py](../scripts/acceptance.py)。

验证包含：真实模型构建与默认 dtype 恢复；HSDP 的 all-gather 输入 dtype、reduce-scatter FP32 输入、完整梯度与 AdamW；原始 view 保留、依赖校验和 mem 文件峰值核对；MXFP8 × 三种 AC 的存储/offload 投影隔离。

当前已完成：

| 检查 | 结果与范围 |
| --- | --- |
| Sidecar 全套回归 | `python -m pytest tests/ -q`：436 passed；包含图、依赖、保存值、内存 oracle 与原有投影测试 |
| 扩展精度矩阵 | 随后扩大参数化并执行 `python -m pytest tests/test_precision.py -q`：18 passed；其中 12 项为 BF16/FP32 param × dense/MXFP8 × none/full/selective，world4、DP replicate2、DP shard2、MTP1；其余覆盖构建 dtype 恢复和无效配置 |
| Sidecar 类型检查 | 固定容器执行 Pyrefly：0 errors，10 suppressed |
| 静态审查 | 精度生效、字节打包、view 保留、mem-only 与测试独立断言复核后无必改问题；该审查不执行测试 |
| 全仓与改动文件 pre-commit | 常规检查、Ruff、codespell、架构检查通过；OAT 因无 staged 文件跳过扫描；主机 Pyrefly 缺少可执行文件，整体不是全绿 |

完整 Flash BF16 原生几何（43 层、dim4096）、world16/rank15、HSDP+EP+TP+CP、MTP1、AC none 已通过更新后的 `scripts/acceptance.py`：74,937 图节点、1,316 个参数梯度；峰值 583,661,755,528 bytes，mem 独立时间线核对通过。该值是逻辑活跃内存，不是真实 NPU allocator 峰值。

对比 `c990d47` 与当前实现的 BF16/MXFP8 × none/full/selective 六组（world16/rank15、HSDP+EP+TP+CP、MTP1），原始算子事件、去掉进程 ID 排序差异后的生命周期多重集合、参数梯度、保存槽数量、常驻参数字节、总峰值及各阶段峰值均一致。图节点数增加来自保留下来的 view：例如 BF16 none 从 5,025 增至 9,333，峰值仍为 364,947,236 bytes。

本文推荐的模块 CLI 原样执行通过。参数快照共 60 个 parser action、29 个 SimulationConfig 字段，文档列举、相对链接和 Python 示例语法核对通过。

机器结果及本阶段冻结的 71 个运行模块的 SHA256 见 [验证结果](mem-only-precision-results.json)。完整日志与对比脚本保存在本机 `torchtitan-migration-audit-20261020/mem-only-precision/`。开发容器已同步当前源码；本轮未重建发布镜像。历史 21 项全规模验收仍绑定原来的源码哈希；本次不将历史数据写成新源码的完整矩阵结果。Meta 测试不能证明 NPU 数值、HCCL 性能或 allocator 峰值。
