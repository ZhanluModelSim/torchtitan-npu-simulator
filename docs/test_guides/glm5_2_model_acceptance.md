# GLM-5.2 训练建模验收方案

本方案沿用 `docs/test_guides/deepseek_v32_model_acceptance.md` 的验收口径；本次实测结果见仓库根目录 `glm5_2_validation_report.md`。

## 验收范围

| 类别 | 必验内容 | 通过标准 |
| --- | --- | --- |
| 配置 | 官方尺寸、78 主层、3 dense + 75 MoE、256 experts、8 top-k | 配置字段与模型结构一致 |
| IndexShare | full/shared schedule、shared 层无 indexer、top-k 跨层传递 | full 层产生 top-k，shared/MTP 层消费相同形状 top-k |
| MTP | 1 个 MTP 输出 | 输出列表长度为 2，logit shape 为 `[B, S-1, V]` |
| meta forward | smoke config 在 meta 上前向 | 不触发硬件；输出 shape 正确 |
| TTNS IR | forward/backward/optimizer 捕获 | 至少生成 `s0_F`、`s0_B`，DSA/MoE/linear/norm 等算子有记录；模板图无环 |
| TP/EP/CP/PP/FSDP | 配置与 parallelize plan | 不注册 shared 层不存在的 indexer FQN；并行规划可构建；PP 确认 top-k/MTP metadata 的 stage 边界传输 |
| checkpoint | HF key mapping | MLA、indexer、MoE、MTP key mapping 可实例化；shared 层不要求 indexer 参数 |
| regressions | DSV3.2 原有单测 | 原有 DSV3.2 配置、adapter、RMSNorm、DSA 测试不回退 |

## 本地验证命令

```powershell
# 配置/模型单测
& 'D:\HW_project\.conda-dsv32\python.exe' -m pytest -q `
  tests/unit_tests/models/test_glm5_2_config.py

# DSV3.2 回归
& 'D:\HW_project\.conda-dsv32\python.exe' -m pytest -q `
  tests/unit_tests/models/test_deepseek_v32_config.py `
  tests/unit_tests/models/test_deepseek_v32_state_dict_adapter.py `
  tests/unit_tests/models/test_deepseek_v32_rmsnorm.py `
  tests/unit_tests/models/test_dsa_indexer_loss.py

# 单卡 TTNS meta 仿真
& 'D:\HW_project\.conda-dsv32\python.exe' scripts/run_simulator_spawn.py `
  --config glm5_2_smoketest `
  --simulation.world-size 1 `
  --simulation.no-enable-memory-tracking

# TP2：使用 ATen MoE recipe；仅 TP 不使用 npu_gmm
& 'D:\HW_project\.conda-dsv32\python.exe' scripts/run_simulator_spawn.py `
  --config glm5_2_tp_smoketest `
  --simulation.world-size 2 `
  --simulation.no-enable-memory-tracking

# FSDP2 / CP2 / 核心组合 / PP2 及最终组合均沿用 DSV3.2 验收口径，
# 使用 --training.steps 1；PP 组合需要 local-batch-size >= PP degree。
```

## 解释边界

没有真实 `torch_npu`/CANN 时，可以在 CPU meta simulator 中完成配置、meta forward、TTNS IR、算子 shape 和主要并行图验证；本仓库已对 CPU-only fake backend、pipeline CPU RNG、FSDP prefetch 环和 GLM PP metadata 做兼容处理。当前实测：单卡、TP2、EP2、FSDP2、CP2、TP2+EP2、核心 dp2+tp2+ep2+cp2、PP2、最终 PP2+dp2+tp2+ep2+cp2、AC none/selective 均通过；ETP2 为预期不支持负例。不能在 CPU 虚拟环境中声称完成真实 NPU kernel 数值、性能和真实 checkpoint 训练验收；真实 NPU 验收仍需补跑生产 DSA、MoE dispatch/GMM、RoPE/RMSNorm converter 及至少一个短训练 step。
