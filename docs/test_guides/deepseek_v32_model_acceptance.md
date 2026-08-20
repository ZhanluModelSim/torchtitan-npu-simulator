# DeepSeek-V3.2 Simulator 建模验收记录

本文记录 DeepSeek-V3.2 接入的可重复验收矩阵。被测 revision 为包含本文的
commit，开发基线为 `15ea194bbfa46ef226e1ba6c0c3340a8ef27d758`，分支为
`codex/deepseek-v32-modeling`。环境使用
`torchtitan-npu-simulator-memory-env:latest`，其中 torch 为 `2.12.0+cpu`、
torch_npu 为 `2.12.0.rc1`、CANN 为 `9.1.0-beta.1`。

## 支持范围

| 能力 | 状态 | 说明 |
|---|---|---|
| 单卡、FSDP/eFSDP、TP、EP、CP、PP | 支持 | 均完成一个完整训练 step 的 meta 捕获 |
| AC none/full/selective | 支持 | 分别验证无重算、完整重算和选择性重算 |
| DSA 融合算子 | 支持 | shape-only 前反向保持真实 NPU raw op 名称 |
| ETP | 不支持 | V3.2 `parallelize.py` 明确抛出 `NotImplementedError` |
| TP-only + NPU GMM | 不支持 | 使用专用 ATen MoE TP recipe，默认融合 recipe 快速失败 |

## 必测矩阵结果

所有命令均在容器内、仓库根目录执行，先加载
`source /usr/local/Ascend/ascend-toolkit/set_env.sh`。

| 场景 | 关键覆盖 | world size | 结果与证据 |
|---|---|---:|---|
| 单卡 full AC | 默认 `deepseek_v32_smoketest` | 1 | 614 ops；DSA 四类前/反向 raw op 均存在 |
| FSDP2 | `dp_shard=2` | 2 | 791 ops / 12 comm；常驻参数 69,157,504 B |
| TP2 | `deepseek_v32_tp_smoketest` | 2 | 677 ops / 6 comm；6 个 TP gradient reductions |
| EP2 | `ep=2` | 2 | 867 ops / 21 comm；local expert 数从 8 降为 4 |
| CP2 | `cp=2` | 2 | 867 ops / 30 comm；DSV32 SDPA CP 策略命中 |
| TP2+EP2 | `tp=2, ep=2` | 4 | 918 ops / 33 comm |
| 核心组合 | `dp_shard=2,tp=2,ep=2,cp=2` | 8 | 994 ops / 51 comm；常驻参数 17,852,736 B |
| PP2 | `pp=2`，batch=2 | 2 | 两 stage 均成功；send/recv forward/backward 模板成对 |
| 最终组合 | `pp=2,dp_shard=2,tp=2,ep=2,cp=2` | 16 | 两 stage 均成功；rank 0/1 分别捕获 42/92 个通信事件 |
| AC none | `activation_checkpoint.mode=none` | 1 | 435 ops；无 recompute execution kind |
| AC selective | `mode=selective,memory_budget=0.5` | 1 | 598 ops；162 recompute nodes |
| ETP2 负例 | `tp=2,ep=2,etp=2` | 4 | 预期快速失败：`ETP is not supported currently` |

输出保存在 `simulator_output/deepseek_v32_smoketest/`、
`simulator_output/deepseek_v32_tp_smoketest/` 和
`simulator_output/deepseek_v32_matrix/`。这些运行产物不纳入 Git。

## 代表命令

```bash
# 单卡基线
python3 scripts/run_simulator_spawn.py \
  --config deepseek_v32_smoketest \
  --training.steps=1 \
  --simulation.world-size=1 \
  --simulation.output-formats mem

# 纯 TP
python3 scripts/run_simulator_spawn.py \
  --config deepseek_v32_tp_smoketest \
  --training.steps=1 \
  --simulation.world-size=2 \
  --simulation.output-formats mem

# FSDP + TP + EP + CP 核心组合
python3 scripts/run_simulator_spawn.py \
  --config deepseek_v32_smoketest \
  --training.steps=1 \
  --parallelism.data-parallel-shard-degree=2 \
  --parallelism.tensor-parallel-degree=2 \
  --parallelism.expert-parallel-degree=2 \
  --parallelism.context-parallel-degree=2 \
  --simulation.world-size=8 \
  --simulation.output-formats mem

# PP2：两层分别放入两个 stage，两个 microbatch
python3 scripts/run_simulator_spawn.py \
  --config deepseek_v32_smoketest \
  --training.steps=1 \
  --training.local-batch-size=2 \
  --parallelism.pipeline-parallel-degree=2 \
  --parallelism.pipeline-parallel-first-stage-less-layers=0 \
  --parallelism.pipeline-parallel-last-stage-less-layers=0 \
  --simulation.world-size=2 \
  --simulation.output-formats mem
```

## 判定说明

- 单卡常驻参数为 138,315,008 B；FSDP2 为其二分之一，核心组合继续按实际
  placement 降至 17,852,736 B，符合参数切分趋势。
- DSA shim 捕获 `npu_lightning_indexer`、`npu_sparse_flash_attention`、
  `npu_sparse_lightning_indexer_grad_kl_loss` 和
  `npu_sparse_flash_attention_grad`，且 backward 为所有可微输入返回同 shape 梯度。
- PP rank 0 生成 `PP_SEND_F`/`PP_RECV_B`，rank 1 生成
  `PP_RECV_F`/`PP_SEND_B`，stage 边界和方向闭合。
- state-dict 单测覆盖 V3.2 拆分 attention/indexer 映射、grouped expert 合并拆分
  和 MTP 映射开关。
- Meta 模拟不验证数值精度；真实 NPU loss/grad 对比属于独立精度验收。
