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
`NGPU` 环境变量（优先级高于配置默认 world_size）。

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
