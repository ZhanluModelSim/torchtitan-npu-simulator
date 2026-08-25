# GLM-5.2 训练建模验证报告

验证日期：2026-08-25
代码分支：`codex/glm5-2-modeling`
基线：DeepSeek-V3.2 建模实现（commit `4734cb8`）

## 1. 结论

GLM-5.2 已完成基于 DSV3.2 的训练建模接入，并在 Windows + Miniforge CPU 虚拟环境中完成了 meta/TTNS 结构验证和正式 simulator launcher 的单卡、TP、EP、FSDP、CP、PP 及组合并行矩阵验证。官方结构依据为 [GLM-5.2 官方 config.json](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json)。

本轮已修复两类阻断：FSDP prefetch anchor/通信前驱回填导致的 L1 模板环，以及 PP+MTP+IndexShare 在 CP/TP/DTensor 组合下的 payload/positions 绑定和 shape-only autograd 桥问题。当前 CPU meta simulator 验收项全部通过；ETP2 仍按 DSV3.2 口径保留为预期不支持负例。

正式 launcher 使用的 `torch_npu` 仅用于 CPU meta shape capture 的临时测试桩已在验证完成后删除。该桩不执行真实 NPU kernel，因此本报告不宣称真实 NPU 数值、性能或 checkpoint 训练通过。

## 2. 已实现内容

| 模块 | 实现 | 状态 |
| --- | --- | --- |
| GLM-5.2 配置 | 154880 vocab、6144 hidden、78 主层、3 dense + 75 MoE、256 experts、top-8、1 MTP | 已实现 |
| MLA/DSA | 2048/512 LoRA rank、192/64 QK split、32×128 indexer、top-k 2048 | 已实现 |
| IndexShare | full/shared schedule、shared 层不创建 indexer、跨层传递 top-k、MTP 复用 | 已实现 |
| RoPE | theta 8000000、interleave、1M context 参数 | 已实现 |
| 并行化 | 复用 DSV3.2 TP/EP/CP/PP/FSDP plan，并跳过 shared 层不存在的 indexer FQN | 已实现 |
| checkpoint | 复用 DSV3.2 HF key 映射，补充 GLM adapter 类型 | 已实现 |
| simulator | DSA full/shared shape-only shim 支持 `glm5_2` | 已实现 |
| CPU meta 兼容 | fake backend 按可用 device 注册；MTP 数据集白名单支持 `glm5_2`；pipeline CPU RNG fallback | 已实现 |
| CLI 配置 | `glm5_2_smoketest`、`glm5_2_tp_smoketest`、`glm5_2_78layers_1mtp` 和模型 override | 已实现 |

主要代码位置：

- `torchtitan_npu/models/glm5_2/`
- `torchtitan_npu/models/deepseek_v32/model.py`
- `torchtitan_npu/models/deepseek_v32/parallelize.py`
- `torchtitan_npu/converters/kernels/dsa.py`
- `torchtitan_npu/simulator/hardware_shims/dsa_shim.py`
- `torchtitan_npu/simulator/config_registry.py`

## 3. DSV3.2 同口径验收结果

| 验收项 | 结果 | 实际结果/阻断点 |
| --- | --- | --- |
| 官方 78+1 拓扑 | PASS | `len(layers)=79`；3 dense、75 MoE、1 MTP；IndexShare schedule 与配置一致 |
| IndexShare schedule | PASS | 前 7 层为 `full, full, full, shared, shared, shared, full`；MTP 为 shared |
| meta forward | PASS | main/MTP 两组 logits，smoke shape 均为 `(1, 6, 320)` |
| 单卡 full AC TTNS | PASS | 1633 ops；forward 472 nodes、backward 460、recompute 452、optimizer 249；0 comm |
| TP2 | PASS | 使用 `glm5_2_tp_smoketest`；2016 ops、21 comm；TP plan、反向和 optimizer 完成 |
| FSDP2 / dp-shard2 | PASS | 2304 个图节点；`s0_F=737`、`s0_B=1293`、optimizer=274，所有模板 `is_acyclic=True`；prefetch anchor 前驱采用延迟回填和逐边环检测 |
| EP2 | PASS | 2361 ops、54 comm；MoE/GMM、EP 路由、FSDP 收尾和 optimizer 完成 |
| CP2 | PASS | 2533 ops、80 comm events；CP 策略命中，F/B/optimizer 模板生成且无环 |
| TP2+EP2 | PASS | 2574 ops、90 comm；TP、EP、FSDP、MoE 通信和 optimizer 完成 |
| 核心组合 dp2+tp2+ep2+cp2 | PASS | world size 8；3158 ops、157 comm events；调度图无环 |
| PP2 | PASS | world size 2；rank0=1031 ops/12 comm，rank1=937 ops/12 comm；MTP 与 IndexShare top-k metadata 跨 stage 传递完成 |
| 最终 PP2+dp2+tp2+ep2+cp2 | PASS | world size 16、local batch 2；rank0=1582 ops/166 comm，rank1=1578 ops/198 comm；两 rank 均完成 8 个模板且全图无环 |
| AC none | PASS | 1180 ops；无 recompute execution kind |
| AC selective | PASS | 1585 ops；404 recompute nodes |
| ETP2 负例 | PASS（预期失败） | 快速得到 `NotImplementedError: ETP is not supported currently` |
| DSV3.2/GLM 配置与模型单测 | PASS | 目标集合 28 项，28 passed |
| simulator capture 回归单测 | PASS | `test_single_stage_trace_assembler.py`：29 passed |
| compileall | PASS | GLM、DSV3.2、simulator、CPU pipeline patch 均通过编译检查 |
| 真实 NPU kernel/数值/性能 | NOT RUN | 当前环境无 CANN、驱动和真实 `torch_npu` kernel |

## 4. 关键修复与验证命令

为使 CPU meta simulator 能进入正式 launcher，并完成组合并行验收，本轮补充了以下兼容/拓扑逻辑：

1. CPU PyTorch 不识别 `npu` 时，fake process-group 仅注册 CPU device backend。
2. `glm5_2` 加入 MTP 数据集入口白名单。
3. pipeline 的 `fork_rng` 在无 NPU device 时回退到 CPU RNG backend。
4. FSDP prefetch anchor 的全部前驱延迟到克隆通信节点完成后逐边回填，环边自动跳过并保留诊断信息。
5. GLM PP payload 显式携带 IndexShare top-k/MTP 输入；`forward` 同时兼容位置参数 payload 和关键字 `positions`，并对 DTensor/普通 Tensor 的 shape-only 依赖桥做类型保护。

TP2 使用单独的 `glm5_2_tp_smoketest`，因为 DSV3.2 的 `npu_gmm` 路径明确不支持“仅 TP、不含 EP”；这与模型失败无关，是并行 recipe 选择约束。

代表性命令：

```powershell
$env:PYTHONPATH = 'D:\HW_project'
& 'D:\HW_project\.conda-dsv32\python.exe' scripts/run_simulator_spawn.py `
  --config glm5_2_smoketest `
  --simulation.world-size 1 `
  --simulation.no-enable-memory-tracking `
  --simulation.output-formats text

& 'D:\HW_project\.conda-dsv32\python.exe' scripts/run_simulator_spawn.py `
  --config glm5_2_tp_smoketest `
  --simulation.world-size 2 `
  --simulation.no-enable-memory-tracking `
  --simulation.output-formats text
```

## 5. 遗留项

1. 在匹配版本的 CANN、驱动和真实 `torch_npu` 环境补跑生产 DSA、MoE dispatch/GMM、RoPE/RMSNorm converter，以及至少一个短训练 step 的 loss/grad/checkpoint 验证。
2. 完整 78+1 模型已完成配置和结构构造验证；完整参数初始化、真实 checkpoint 映射和长序列内存峰值仍需在目标 NPU simulator/runtime 中执行。
