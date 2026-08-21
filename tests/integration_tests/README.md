# DeepSeek-V4 集成测试基础设施

本目录遵循 Torchtitan 的 `tests/integration_tests` 布局，负责维护 DeepSeek-V4
集成测试定义、测试入口以及可选的 loss 精确比较。基础架构代码由 torchtitan 迁移而来。

## 测试矩阵

| Case 名称 | 模型 | 并行配置 | Rank 数 | 编译配置 | Check Loss | 不检查 Loss 原因 |
|---|---|---|---|---:|---|---|
| `dsv4_golden_1rank` | DeepSeek-V4 | 1 Rank 参考配置 | 1 | - | 是 | - |
| `dsv4_golden_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | - | 是 | - |
| `dsv4_smla_1rank_aot_eager` | DeepSeek-V4 | 1 Rank | 1 | `aot_eager` | 否 | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_smla_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | - | 否 | SMLA 暂不支持 `--debug.deterministic`；原先 `aot_eager` 触发 `NpuMoeTokenUnpermuteBackward0` SymInt 解包失败，当前不编译规避，见 [#106](https://gitcode.com/cann/torchtitan-npu/issues/106) |
| `dsv4_smla_cp2_ep2_fsdp2` | DeepSeek-V4 | CP2 + EP2 + FSDP2 | 4 | - | 否 | 与 EP2 同源：`NpuMoeTokenUnpermuteBackward0` SymInt 问题；当前不编译规避，见 [#106](https://gitcode.com/cann/torchtitan-npu/issues/106) |
| `dsv4_smla_cp2` | DeepSeek-V4 | CP2 | 2 | - | 否 | SMLA 暂不支持 `--debug.deterministic` |

`use_golden` 与 `check_loss` 是两个独立维度：`use_golden` 仅决定使用 Golden 参考算子
还是 SMLA/NPU override；`check_loss` 决定是否启用 deterministic、读取参考 loss 并执行
精确数值比较。

当前两个 Golden case 设置 `check_loss=True`，使用固定随机种子和 deterministic 模式，
比较 TensorBoard 标量 `loss_metrics/global_avg_loss`，要求 step 集合和每个浮点值均精确相等。

四个 SMLA case 都设置 `check_loss=False`，因此不会启用 `--debug.deterministic`，也不会
读取 golden loss。它们用于覆盖 SMLA/NPU override 在单卡、EP+FSDP、CP+EP+FSDP 以及
CP 场景下的实际构图、编译和训练执行路径；其中 1P 场景使用 `aot_eager`，EP2/CP2 两个
场景当前以 eager 方式执行以规避 [#106](https://gitcode.com/cann/torchtitan-npu/issues/106)。

这里的 integration recipe 聚焦 sparse-attention / MHC 回归边界。端到端 example 脚本
额外启用 Virtual Optimizer / checkpoint override；这些 storage/checkpoint override 不属于
当前 integration loss regression 的覆盖范围。

## 已知问题 / 待解决

- **`NpuMoeTokenUnpermuteBackward0` SymInt 解包失败**：2P EP=2 SMLA + `aot_eager` 时，
  `torch_npu.npu_moe_token_unpermute` 的自动生成 backward 在 AOTAutograd 中报
  `RuntimeError: when unpacking SymInt, expected int but got u12 + u13`。
- 影响用例：`dsv4_smla_ep2_fsdp2`、`dsv4_smla_cp2_ep2_fsdp2`。
- 当前规避：EP2/CP2 两个用例不启用 `--compile.enable`，以 eager 方式覆盖
  SMLA/EP/CP 路径；1P AOT 用例保留用于单卡编译回归。1P 编译路径当前另有
  `torch_npu.npu.get_device_name()` 触发 Dynamo Unsupported 的环境/编译限制。
- 跟踪 issue：[#106](https://gitcode.com/cann/torchtitan-npu/issues/106)。

## 入口

CI 通过以下脚本启动测试：

```bash
.ci/smoke_test.sh
```

或直接运行 Python 入口：

```bash
python -m tests.integration_tests.run_tests
```
