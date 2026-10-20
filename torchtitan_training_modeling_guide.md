# 新模型接入工作流（实操手册）

> 本文是**操作手册**：按阶段给出「做什么、参考哪份文档、提示词怎么写」。
> 规则与验收标准以两份权威文档为准，本文只做索引和实操补充：
>
> - 接入规则：`docs/feature_guides/new_model_onboarding.md`
> - 最终验收：`docs/test_guides/model_acceptance.md`
>
> 下游（湛卢）流程见 §8。

## 0. 总览

| 阶段 | 产出 | 参考文档 |
| --- | --- | --- |
| 1. 模型分析 + 契约 | 模型特点总结、`MODEL_CONTRACT.md` | onboarding §1 |
| 2. 单卡骨架 | `core.py` / `attention.py` / `feed_forward.py` / `model.py` / `config_registry.py`（debug/reduced/正式三档） | onboarding §2 |
| 3. 注册 + 权重 | `__init__.py`（ModelSpec）、module 注入、state-dict adapter | onboarding §3 |
| 4. 融合算子 shape-only | `simulator/hardware_shims/<model>_shim.py` + `OP_REFERENCE.md` | onboarding §4、`hardware_shims/OP_INTERFACE_REFERENCE.md` |
| 5. 并行 | `parallelize.py`（按 TP→EP→CP→AC→FSDP→PP 顺序） | onboarding §5 |
| 6. 量化（可选） | `<model>_mxfp8.py` converter（FQN 控制） | onboarding §4、kimi/deepseek 配置先例 |
| 7. 验证 | 模拟器矩阵 + 单测 + 显存对账 | acceptance 全文 |
| 8. 下游湛卢 | 注册文件 + example_args + 未命中算子闭环 | 湛卢 `docs/torchtitan.md`、§10 |

环境要点：容器内跑（本仓无 NPU 也无妨，模拟器走 meta）；先 source CANN
`set_env.sh`；CPU 验证加 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；模拟器并行度用
`NGPU` 环境变量（优先级高于配置默认 world_size）。进环境与运行命令见 §12。

## 0.1 开发者上手路径：读什么、怎么驱动 AI

**阅读顺序（开发者开工前按序读完，AI 同样按此顺序建上下文）**：

1. 本指南（工作流总索引 + 每阶段提示词模板 + §11 对话模板实例）；
2. `docs/feature_guides/new_model_onboarding.md`（各阶段硬性规则）；
3. `docs/test_guides/model_acceptance.md`（验收判据——决定"什么才算完成"）；
4. `torchtitan_npu/simulator/hardware_shims/OP_INTERFACE_REFERENCE.md`
   （阶段 4 前必读：算子建模硬性规则——真名、输入个数判变体、rank 硬解包、
   meta 安全、autograd 契约、并行化后绑定）；
5. 先例模型包（kimi_k3 / deepseek_v4 / qwen3_next）：parallelize 写法（TP/EP/
   FSDP/mxfp8）、state_dict_adapter、单测 `tests/unit_tests/models/test_<model>.py`；
6. 先例模型的交付物：模型包内 `MODEL_CONTRACT.md` / `OP_REFERENCE.md` /
   `ZHANLU_HANDOVER.md` + `docs/user-guides/simulator/<model>-*-alignment.md`
   的"适配全程回顾"章节（先读再开工，避免重踩）。

**如何用 skill 驱动 AI**：

- AI 侧技能名 `torchtitan-npu-model-adapt`，位于
  `~/.config/opencode/skills/torchtitan-npu-model-adapt/`（换机器需要拷贝该目录）。
  对话中出现"适配 xx 模型到 titan 模拟器 / 参照 MindSpeed 移植模型 / 新增模型
  preset"即自动加载；也可以在提示词里显式要求"先读 SKILL.md 的坑清单再动手"。
- skill 内容 = 工作流程 + 28 条已知坑（实测对齐教训沉淀）+ 实例参考；它不替代
  上面 6 步阅读，而是把踩过的坑前置到 AI 上下文。
- 开发者要做的事：按 §11 模板把字段填全发给 AI（尤其"实测算子流"字段——
  有就给路径，没有就写明，不许 AI 自行臆测口径），然后逐项裁决 AI 提出的
  sketch/reference 差异表；验证口径（ep/AC/配置名）在开工时写死。

**每个模型的固定产出物（阶段 7 结束时必须齐全）**：

| 产出 | 位置 |
| --- | --- |
| 模型契约（参数量公式/支持矩阵/偏差/验证记录） | `torchtitan_npu/models/<model>/MODEL_CONTRACT.md` |
| 算子账本（真名/FQN/分 phase 次数/融合边界） | `torchtitan_npu/models/<model>/OP_REFERENCE.md` |
| 黄区交接说明 | `torchtitan_npu/models/<model>/ZHANLU_HANDOVER.md` |
| 单测 | `tests/unit_tests/models/test_<model>.py` |
| 模拟配置（含并行/量化变体） | 模型包 `config_registry.py` + `torchtitan_npu/simulator/config_registry.py` **聚合注册** |
| 算子流对齐记录（实测对齐类模型） | `docs/user-guides/simulator/<model>-op-stream-alignment.md` |

## 1. 阶段 1：先分析，后动手

提示词要点（实测有效模板）：

```text
参考 docs/feature_guides/new_model_onboarding.md 和 docs/test_guides/model_acceptance.md
接入一个模型。先帮我分析下模型特点 <sketch 代码目录>；
XX 部分（如 MoE 部署/通信逻辑）参考 <reference 代码目录>，
训练部署方法在博客 <url>，先总结看看。
```

关键动作：
- **区分「结构来源」与「部署参考」**：sketch 定规模和结构，reference 只取部署/通信
  语义——两者冲突时让用户逐项裁决（如：规模用放大版还是官方版、哪些特性排除）。
- 把 sketch 与 reference 的**差异表**列出来请用户确认（打分函数、bias、归一化等）。
- 产出 `MODEL_CONTRACT.md`：来源、结构、参数量公式（独立于代码手工推导）、
  支持矩阵、已知风险。参数量公式是后续显存对账的独立基线。

## 2. 阶段 2：单卡骨架

提示词要点：

```text
不要按 reference 的规模来，按 <sketch> 的放大版；只接 <某变体>；
reference 只参考 <部署逻辑/通信>；<差异项> 可以补齐；
另外开始之前先给我讲清楚 <存疑组件> 是什么。
```

关键动作：
- 参考实现接口 **1:1 落地**（类名、签名、函数拆分一致），生产 kernel 缺失处留
  可选 hook（`try: from .kernel import ...`），并给数学等价的 eager 回退。
- 结构校验 fail fast（整除关系、层布局、非法配置）。
- 存疑组件（如稀疏注意力）先给用户讲清语义再动手。
- 单卡 CPU 验证：前向/反向/loss、参数量与公式逐字节对账、state-dict 回路、
  meta 构造（`torch.device("meta")`）。

## 3. 阶段 3：注册

- `__init__.py` 提供分层 builder + `model_registry(flavor) -> ModelSpec`。
- `torchtitan_npu/__init__.py`：import、`_supported_models`、`_inject_module`。
- `simulator/config_registry.py` 加模拟配置函数。
- state-dict adapter：无外部权重时做 identity 直通并 fail fast HF 路径。

## 4. 阶段 4：模拟器融合算子

提示词要点：

```text
关于 <自定义算子> 我给你参考实现（贴代码），你按照这个接口来写 core.py，
然后开始建模。
```

关键动作：
- 先读 `hardware_shims/OP_INTERFACE_REFERENCE.md` 的硬性规则（真名、输入个数
  判变体、rank 硬解包、有效 shape、meta 安全、autograd 契约、并行化后绑定）。
- 有真实 ACLNN/triton kernel 的用已建模真名；全新 kernel 用**约定生产名**并在
  契约声明。
- 产出 `OP_REFERENCE.md`：算子个数、抓到的 kernel 名、逐位输入契约、实现代码、
  实测捕获次数（模型特有算子不能摊成小算子——验收 4.3 红线）。

## 5. 阶段 5：并行（按 onboarding §5 顺序）

TP → EP → CP → AC → FSDP/eFSDP → DP replicate，PP 最后；未实现的维度 **fail
fast 并写明原因**（不准静默降级）。MoE 模型先想清楚 EP 切什么维度（头还是
专家——以 reference 部署结论为准），通信契约（次数/字节数/top-k 相关性）写进
契约，模拟器跑通后用 `memory_events.csv` 核对 comm 事件。

## 6. 阶段 6：量化（可选）

- 复用上游 `MXFP8Converter` 体系（继承它保住 `find_pad_multiple` 与模拟器
  meta patch），按 **module-path FQN** 控制范围。
- 路由类参数（MoE gate、稀疏注意力 router）**强制 fp32 排除**：配置命中即
  fail fast，convert 后二次守卫。
- 配置加 `_mxfp8` 变体；模拟器验证量化算子名（`npu_quant_matmul` /
  `npu_grouped_matmul` 等）入账。

## 7. 阶段 7：验证（最终验收见 acceptance 文档）

最小闭环（acceptance §3 必测矩阵的落地版）：

| 配置 | world | 覆盖 |
| --- | --- | --- |
| `<model>_smoketest` | 1 | 单卡基线 + 融合算子捕获数 |
| `_fsdp2` / `_tp2` / `_ep2` / `_cp2` | 2 | 单维度并行 + comm 事件核对 |
| `_tp2ep2` / `_fsdp2ep2` | 4 | 组合正交性 |
| `_mxfp8`（可选） | 1 | 量化算子入账 |

逐项核对：参数量对账、显存分解（persistent/active peak）、comm 类型/字节数
与契约一致、AC 开关对照、单测（构造/参数量/shape/state-dict/并行 fail-fast）
+ 相邻模型回归。汇总进 `MODEL_CONTRACT.md` §验证记录。

## 8. 阶段 8：下游湛卢接入

上游建模验收后，把代码带进湛卢平台做算子账本比对：

1. **代码过墙**：上游仓建模好的代码上传并导入黄区环境；黄区内拉取湛卢代码
   （`Zhanlu-Model-Platform`），上游仓以 `third_party/torchtitan-npu-simulator`
   子模块/目录形式存在。
2. **湛卢注册**：参考 `docs/torchtitan.md`（运行逻辑）与 dsv4 的接入方式——
   通常只有几个注册文件（模型注册 + 配置注册 + example_args）。
3. **造一个单配置 example**：`bin/example_args/torchtitan_batch.example.json`，
   一个最小可跑配置。
4. **调试闭环**：`ZHANLU_DISABLE_PRUNE_OP=1` 运行，把所有**未命中建模的算子**
   打印出来 → 对照上游 `OP_REFERENCE.md` 补缺失算子建模或修正算子名 → 重跑
   直至无未命中、无明知不合理项。
5. 把上述湛卢步骤沉淀为该模型的接入说明（环境已备好时，提示词模板见下）。

## 9. 提示词模板（湛卢阶段）

```text
<上游建模验收完成后>
docker 环境 <env_name>，帮我将
<path>/third_party/torchtitan-npu-simulator/torchtitan_npu/models/<model>
模型接入当前的湛卢，指定 ZHANLU_DISABLE_PRUNE_OP=1
用来把所有未命中建模的算子打印出来。
运行逻辑参考 <zhanlu>/docs/torchtitan.md（环境我已准备好）。
接入逻辑参考 dsv4 是怎么注册的，应该就几个文件的注册。
另外生成一个只有一个配置的
<zhanlu>/bin/example_args/torchtitan_batch.example.json。
跑下验证看看结果，看缺哪些算子建模、是否有明显不合理的地方，整理个流程。
```

## 10. 常见问题（高频翻车点）

**内存对不上——先理论后建模。**
建模前先在契约里写好**独立的理论估算**（参数量公式、逐层权重 shape、激活
大头如 `[B,H,T,T]` 分数矩阵、FSDP full-parameter 驻留），模拟器跑完后与
`memory/memory_summary.json` 逐项对账：persistent_param_bytes = 参数量 ×
storage dtype 字节数、active peak 的阶段与构成（forward/backward/optimizer）。
对不上先查：tied weight 重复计数、dtype 当成 storage dtype、AC 开关没对齐、
只算 shard 漏了 full-parameter 驻留（验收 §4.5）。

**算子名在湛卢 miss——优先复用，不要新造。**
湛卢侧的 cost model 按算子名建账，本仓**感知不到**湛卢有哪些已建模算子；
自造名字（哪怕只差一个词）下游 registry 直接 miss。两条铁律：

1. **先复用**：翻 `hardware_shims/OP_INTERFACE_REFERENCE.md` 和其他模型的
   shim（smla/kda/mhc/ar_llm 等），结构同构的算子直接用**已建模的真名**和
   同款输入布局（必要时按约定补占位输入）；
2. **确属新算子才新造**：用约定的生产 kernel 名，并且**必须把名字和接口
   主动同步给湛卢侧登记**（提示词里明确要求"缺哪些算子建模"的输出就是
   这个闭环的入口）。

配套自检：`ZHANLU_DISABLE_PRUNE_OP=1` 跑一次 → 未命中清单对照本模型的
`OP_REFERENCE.md` → 逐条处理（改名复用 / 补建模 / 确认登记），直到清零。

## 11. 对话模板实例与字段核对清单（qwen3_next 实际填充版）

> 用法：接入下一个新模型时，把本节模板中 `<>` 字段换成新模型的真实信息即可。
> 每个字段标注了 qwen3_next 适配时的真实取值；当时缺失的信息明确写"本次无"，
> 不要临场编造。

```text
参考仓库根目录 torchtitan_training_modeling_guide.md（工作流索引）、
docs/feature_guides/new_model_onboarding.md（接入规则）、
docs/test_guides/model_acceptance.md（验收标准），
接入 <模型名>（<架构一句话>）。

- 结构来源（sketch，定规模和结构）：
  MindSpeed-LLM 仓内建模代码（管理方指令：模型代码严格按该仓库逐行镜像）
  <参考 modeling 路径>
  （背景结构：<HF 官方架构 / 内核参考路径，注明参考版与服务器版可能有差异>）

- 部署参考（只取部署/通信/算子语义）：
  <部署参考仓 + commit + 实测口径 yaml 及关键并行参数>

- 实测算子流：<有则填 CSV 路径；没有则写"本次无"并说明口径替代来源>
  接入下个模型时若有实测 kernel_summary CSV，在此填路径。

- 支持矩阵：
  <必须支持的并行/量化组合；v1 不实现的维度 fail fast 并写明原因>

- 测试口径（写死，后续不改）：
  <实测对齐口径、验证主配置、smoketest、量化验证、AC 双覆盖与并行矩阵变体>

- 范围声明：
  <蓝/黄区分工；commit/推送纪律>

先分析模型特点、列 sketch 与 reference 的差异表给我逐项确认，再动手。
```

### 字段核对清单（接入新模型时逐项替换）

| 字段 | qwen3_next 本次取值 | 缺失时 |
| --- | --- | --- |
| 模型名/架构一句话 | qwen3_next，hybrid 3/4 GDN + 1/4 门控 GQA，全 MoE | 必填 |
| 结构来源 | MindSpeed-LLM 建模代码（逐行镜像指令） | 必填 |
| 部署参考 + commit | MindSpeed-LLM @6082e0de fsdp2 + 实测 yaml | 必填 |
| **实测算子流 CSV** | **本次无**（以管理方逐轮反馈为口径，记录于 `docs/user-guides/simulator/qwen3-next-op-stream-alignment.md`） | 有则填路径，无则写"本次无"+口径替代 |
| 支持矩阵 | fsdp+ep+tp / mxfp8 必须支持；CP/PP/ETP fail fast | 必填 |
| 测试口径 | ep=2、`qwen3_next_80b_a3b_2layer` 主配置（AC=none 基线 + full 对照）、smoketest 4 层 ep=2、量化 `qwen3_next_2l_mxfp8`、并行矩阵按验收 §3 | 必填（写死防止中途漂移） |
| 范围声明 | 蓝区（建模+契约+账本+单测+验收矩阵）；黄区交接用 `ZHANLU_HANDOVER.md`；只本地 commit，确认后再推 | 必填 |

## 12. 开发者手动进环境与模型运行命令

### 12.1 进入环境

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

### 12.2 运行模拟器（meta 仿真，无需真实 NPU）

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

### 12.3 输出物位置

```text
simulator_output/<config名>/
├── summary.txt                     # 一步训练汇总（loss/吞吐/时间线）
├── kernel_summary/rank_*.csv       # 算子账本（raw_op_type/module_path/phase/shape）
├── memory/memory_summary.json      # 显存分解（persistent/active peak/分阶段）
├── comm/                           # 通信事件（类型/字节/依赖）
└── ir_export/                      # 依赖 DAG 导出
```

字段语义详见 [输出与内存](docs/user-guides/simulator/outputs.md)。

### 12.4 单元测试

```bash
cd /opt/torchtitan-npu-simulator
python3 -m pytest tests/unit_tests/models/test_qwen3_next.py -q   # 29 项
```

### 12.5 注意事项

- **GPU 资源**：模拟器走 meta 设备，NGPU 只是逻辑并行度，不需要真实 NPU；
  CPU 环境跑仓库通用验证时加 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
- **推送规则**：宿主机副本 commit 后不自动推送；确认后由用户明确指示推送
  （远程 `qwen3.5:qwen3.5`）。
- **黄区**：湛卢接入见模型包内 `ZHANLU_HANDOVER.md`，蓝区环境不执行。
- 仓内既有 5 个单测失败（kimi overrides ×4、multimodal 资产 ×1）与
  qwen3_next 无关，接入前即失败。
