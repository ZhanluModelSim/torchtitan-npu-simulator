# 新模型接入对话模板 + 环境运行命令（qwen3_next 实例）

> 用法：接入下一个新模型时，把本文档模板中 `<>` 字段换成新模型的真实信息即可。
> 每个字段标注了本次（qwen3_next）的真实取值；当时缺失的信息明确写"本次无"，
> 不要临场编造。文档分两部分：① 给 AI 的对话模板；② 开发者手动进环境跑模型
> 的命令。规则文档见 `torchtitan_training_modeling_guide.md` /
> `docs/feature_guides/new_model_onboarding.md` / `docs/test_guides/model_acceptance.md`。

## 1. 对话模板（qwen3_next 实际填充版）

```text
参考仓库根目录 torchtitan_training_modeling_guide.md（工作流索引）、
docs/feature_guides/new_model_onboarding.md（接入规则）、
docs/test_guides/model_acceptance.md（验收标准），
接入 qwen3_next（Qwen3-Next-80B-A3B，hybrid 3/4 线性注意力 + 1/4 门控 GQA，全 MoE）。

- 结构来源（sketch，定规模和结构）：
  MindSpeed-LLM 仓内建模代码（管理方指令：模型代码严格按该仓库逐行镜像）
  /home/l30079348/MindSpeed-LLM/mindspeed_llm/fsdp2/models/qwen3_next/modeling_qwen3_next.py
  （背景结构：HF Qwen3-Next-80B-A3B 官方架构；GDN 线性注意力内核参考
  /tmp/opencode/MindSpeed/mindspeed/ops/triton/，注意该参考版与服务器安装版可能有差异）

- 部署参考（只取部署/通信/算子语义）：
  /home/l30079348/MindSpeed-LLM（commit 6082e0de，mindspeed_llm/fsdp2/），
  实测口径配置 pretrain_qwen3_next_16k_fsdp2_A3.yaml
  （FSDP=8, EP=2, eFSDP=4, recompute off, reduce bf16, seq 16384）

- 实测算子流：本次无。管理方未提供实测算子流 CSV，实测对齐口径通过管理方
  6 轮逐条反馈给到（router 反向链 / 非EP融合permute / GDN包装层eager算子 /
  f32内核输出约定 / dtypecast / ep=2 口径等，全程记录在
  docs/user-guides/simulator/qwen3-next-op-stream-alignment.md）。
  接入下个模型时若有实测 kernel_summary CSV，在此填路径。

- 支持矩阵：
  必须支持 FSDP/eFSDP + EP + TP(+序列并行)（组合 fsdp+ep+tp）+ DP replicate
  + AC full/selective；量化 mxfp8 必须支持；
  CP / PP / ETP fail fast 并写明原因（v1 不实现）。

- 测试口径（写死，后续不改）：
  ep=2 实测对齐（管理方指令）；验证主配置 qwen3_next_80b_a3b_2layer
  （80B 维度裁 2 层：1 GQA + 1 GDN，均 MoE，seq 16384，AC=none 基线 +
  full 对照）；smoketest 4 层调试配置（ep=2）；量化验证 qwen3_next_2l_mxfp8；
  AC 双覆盖 + 并行矩阵变体（_tp2/_ep2/_fsdp2/_tp2ep2/_fsdp2ep2/_tp2ep2fsdp2）
  按验收规范 §3 落地。

- 范围声明：
  本环境为蓝区：完成建模 + 契约 + 算子账本 + 单测 + 验收矩阵；
  不做湛卢（黄区）接入，黄区交接用模型包内 ZHANLU_HANDOVER.md；
  修改完成后只本地 commit，不推送，我确认后再推。

先分析模型特点、列 sketch 与 reference 的差异表给我逐项确认，再动手。
```

### 字段核对清单（接入新模型时逐项替换）

| 字段 | qwen3_next 本次取值 | 缺失时 |
| --- | --- | --- |
| 模型名/架构一句话 | qwen3_next，hybrid 3/4 GDN + 1/4 门控 GQA，全 MoE | 必填 |
| 结构来源 | MindSpeed-LLM 建模代码（逐行镜像指令） | 必填 |
| 部署参考 + commit | MindSpeed-LLM @6082e0de fsdp2 + 实测 yaml | 必填 |
| **实测算子流 CSV** | **本次无**（以管理方逐轮反馈为口径） | 有则填路径，无则写"本次无"+口径替代 |
| 支持矩阵 | fsdp+ep+tp / mxfp8 必须支持；CP/PP/ETP fail fast | 必填 |
| 测试口径 | ep=2、2layer 主配置、AC 双覆盖、验收矩阵 | 必填（写死防止中途漂移） |
| 范围声明 | 蓝区；不推送等确认 | 必填 |

## 2. 开发者手动进环境运行命令

### 2.1 进入环境

```bash
# 工作副本 bind-mount 关系：宿主机 /home/l30079348/titan-qwen3.5
#                = 容器内 /opt/torchtitan-npu-simulator（git 分支 qwen3.5）
# 宿主机上改代码，容器内立即可见；git 操作在宿主机副本做。
docker exec -it titan-llf bash
```

```bash
# 容器内每次新 shell 先 source CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /opt/torchtitan-npu-simulator
```

### 2.2 运行模拟器（meta 仿真，无需真实 NPU）

```bash
# 基线：实测对齐主配置（world=2 由 NGPU 指定，优先级高于配置默认值）
NGPU=2 python3 scripts/run_simulator_spawn.py \
  --config qwen3_next_80b_a3b_2layer \
  --simulation.output-formats text,csv

# 并行矩阵变体（验收 §3；world 跟随 NGPU）
NGPU=2 python3 scripts/run_simulator_spawn.py --config qwen3_next_2l_tp2  --simulation.output-formats text,csv
NGPU=2 python3 scripts/run_simulator_spawn.py --config qwen3_next_2l_ep2  --simulation.output-formats text,csv
NGPU=2 python3 scripts/run_simulator_spawn.py --config qwen3_next_2l_fsdp2 --simulation.output-formats text,csv
NGPU=4 python3 scripts/run_simulator_spawn.py --config qwen3_next_2l_tp2ep2 --simulation.output-formats text,csv
NGPU=4 python3 scripts/run_simulator_spawn.py --config qwen3_next_2l_fsdp2ep2 --simulation.output-formats text,csv
NGPU=8 python3 scripts/run_simulator_spawn.py --config qwen3_next_2l_tp2ep2fsdp2 --simulation.output-formats text,csv

# AC 开启对照（与上面同配置追加一个参数）
NGPU=2 python3 scripts/run_simulator_spawn.py --config qwen3_next_2l_fsdp2 \
  --simulation.output-formats text,csv --activation-checkpoint.mode full

# 量化变体（单卡）
NGPU=1 python3 scripts/run_simulator_spawn.py --config qwen3_next_2l_mxfp8 \
  --simulation.output-formats text,csv

# 调试小配置（快速冒烟，~1 分钟）
NGPU=2 python3 scripts/run_simulator_spawn.py --config qwen3_next_smoketest \
  --simulation.output-formats text,csv
```

配置名解析自 `torchtitan_npu/simulator/config_registry.py`（聚合注册表，
新增配置必须在此登记，否则报 "Config function not found"）。

### 2.3 输出物位置

```text
simulator_output/<config名>/
├── summary.txt                     # 一步训练汇总（loss/吞吐/时间线）
├── kernel_summary/rank_*.csv       # 算子账本（raw_op_type/module_path/phase/shape）
├── memory/memory_summary.json      # 显存分解（persistent/active peak/分阶段）
├── comm/                           # 通信事件（类型/字节/依赖）
└── ir_export/                      # 依赖 DAG 导出
```

### 2.4 单元测试

```bash
cd /opt/torchtitan-npu-simulator
python3 -m pytest tests/unit_tests/models/test_qwen3_next.py -q   # 29 项
```

### 2.5 注意事项

- **GPU 资源**：模拟器走 meta 设备，NGPU 只是逻辑并行度，不需要真实 NPU；
  CPU 环境跑仓库通用验证时加 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
- **推送规则**：宿主机副本 commit 后不自动推送；确认后由用户明确指示推送
  （远程 `qwen3.5:qwen3.5`）。
- **黄区**：湛卢接入见模型包内 `ZHANLU_HANDOVER.md`，蓝区环境不执行。
- 仓内既有 5 个单测失败（kimi overrides ×4、multimodal 资产 ×1）与
  qwen3_next 无关，接入前即失败。
