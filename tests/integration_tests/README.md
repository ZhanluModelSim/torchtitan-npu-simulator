# DeepSeek-V4 集成测试基础设施

本目录遵循 Torchtitan 的 `tests/integration_tests` 布局，负责维护测试
定义、测试入口以及记录 loss 的精确比较工具。基础架构代码由torchtitan迁移而来。

## 覆盖矩阵

### 当前覆盖
| Case 名称 | 模型 | 并行配置 | Rank 数 | 编译配置 | 环境变量 |
|---|---|---|---:|---|---|
| `dsv4_golden_1rank` | DeepSeek-V4 | 1 Rank 参考配置 | 1 | - | `USE_GOLDEN=1` |
| `dsv4_golden_ep2_fsdp2` | DeepSeek-V4 | EP2 + FSDP2 | 2 | - | `USE_GOLDEN=1` |

### 计划覆盖
| Case 名称 | 模型 | 并行配置 | Rank 数 | 编译配置 | 环境变量 | 当前未覆盖原因 |
|---|---|---|---:|---|---|---|
| `dsv4_smla_1rank_aot_eager` | DeepSeek-V4 | 1 Rank 参考配置 | 1 | `aot_eager` | - | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_smla_ep2_fsdp2_aot_eager` | DeepSeek-V4 | EP2 + FSDP2 | 2 | `aot_eager` | - | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_smla_cp2_ep2_fsdp2_aot_eager` | DeepSeek-V4 | EP2 + FSDP2 + CP2 | 4 | `aot_eager` | - | SMLA 暂不支持 `--debug.deterministic` |
| `dsv4_golden_cp2` | DeepSeek-V4 | CP2 | 2 | - | `USE_GOLDEN=1`, `CP_DEGREE=2` | Golden 路径暂不支持 CP |

每个测试用例都使用匹配的模型、并行配置、编译配置、随机种子和输入设置，读取
对应的 golden 文件并执行被测实现。比较器读取 TensorBoard 标量
`loss_metrics/global_avg_loss`，要求实际输出和 golden 的 step 集合完全相同，
并对 event 文件中读回的每个浮点值执行精确相等比较。

## 入口

CI 通过以下脚本启动测试：

```bash
.ci/smoke_test.sh
```

或直接运行 Python 入口：

```bash
python -m tests.integration_tests.run_tests
```
