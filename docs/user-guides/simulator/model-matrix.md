# Simulator 模型与特性矩阵

> 现状核对：2026-10-07，`feat/npu-simulator`，实现基线 `ecde2f0`。依据源码与 Git 历史静态核对，未执行测试或仿真；后续行为以当前源码为准。

返回[使用入口](../simulator.md)。以下“有实现”表示注册与代码路径存在，不能替代模型最终验收或所有配置组合的运行验证。

正式 simulator registry 一共 10 个公共工厂：DeepSeek V4 7 个，Kimi K3 3 个。

| 模型 preset | 主干层 / hidden / heads | routed experts / top-k | 精度配置 | 默认 EP | 默认 MTP / AC |
|---|---|---|---|---|---|
| DeepSeek V4 Flash | 43 / 4096 / 64 | 256 / 6 | BF16、MXFP8 | 32 | 1 / full |
| DeepSeek V4 Pro | 61 / 7168 / 128 | 384 / 6 | BF16、MXFP8 | 64 | 1 / full |
| DeepSeek V4 Pro 20T | 96 / 12288 / 192 | 2048 / 23 | BF16、MXFP8 | 256 | 1 / full |
| DeepSeek V4 smoketest | 4 / 128 / 4 | 8 / 2 | BF16 | 1 | 0 / full |
| Kimi K3 full | 93 / 7168 / 96 | 896 / 16 | BF16、MXFP8 | 128 | 无专门 MTP 路径 / full |
| Kimi K3 smoketest | 4 / 256 / 8 | 8 / 3 | BF16 | 1 | 无专门 MTP 路径 / selective |

上表层数是主干层数；DeepSeek baseline 的默认 MTP 另加一层。所有精度与 parallel degree 是 preset 默认，可经 typed overrides/CLI 调整，但需符合模型限制。

```text
deepseek_v4_flash_baseline_bf16
deepseek_v4_flash_baseline_mxfp8
deepseek_v4_pro_baseline_bf16
deepseek_v4_pro_baseline_mxfp8
deepseek_v4_pro_20t_baseline_bf16
deepseek_v4_pro_20t_baseline_mxfp8
deepseek_v4_smoketest
kimi_k3_baseline_bf16
kimi_k3_baseline_mxfp8
kimi_k3_smoketest
```

证据：[simulator registry](../../../torchtitan_npu/simulator/config_registry.py)、[DeepSeek 模型默认值](../../../torchtitan_npu/models/deepseek_v4/__init__.py)、[Kimi 模型默认值](../../../torchtitan_npu/models/kimi_k3/__init__.py)。

## 模型特性矩阵

| 特性 | DeepSeek V4 家族 | Kimi K3 |
|---|---|---|
| 核心架构 | shared-KV sparse attention、compressor、Lightning Indexer/indexer loss、mHC、hash/普通 MoE、SwiGLU | KDA + Gated MLA、ShortConv、AttnRes、dense 前缀、LatentMoE、SiTU-GLU |
| Attention 层分布 | 由 `compress_ratios` 等参数控制 | full：69 KDA + 24 MLA；debug：3 KDA + 1 MLA |
| DP/FSDP/EFSDP/HSDP | 有代码路径，含 replica 模块 capture 修复 | 有代码路径，复用通用 FSDP/HSDP |
| TP/Sequence Parallel | 有代码路径；sequence/head 等整除约束 | 有代码路径；含 KDA conv/head、MLA 和 AttnRes 的专门方案 |
| EP | NpuExpertParallel fake bridge | 通用 ExpertParallel fake bridge |
| ETP | 未支持，模型代码显式抛错 | 有 EP+ETP 实现及 mesh/维度校验；本次未运行组合验证 |
| CP | 需要 `npu_smla`，使用 compressor/window exchange 等路径 | 全序列 all-gather 的保守正确方案；要求 contiguous shards，load balancer 为 None |
| PP | 有实现，含复杂 schedule/virtual stages | 未支持，parallelize 明确拒绝 |
| MTP | 有实现；MTP+PP 明确拒绝，PP 必须设置 `training.num_mtp_modules=0` | 没有与 DeepSeek 等价的专门 MTP 接入 |
| AC none/full/selective | 有实现和重计算归属 | 有实现和 attention synthetic SAC |
| AC memory_budget | 未支持，需要 compile，与 eager 捕获冲突 | 同左 |
| Reentrant AC | 当前 execution tracker 拒绝 | 同左 |
| MXFP8 默认范围 | 指定 attention/indexer/output projection、MoE/shared experts、MTP e/h projection 等 FQN | 默认仅 `moe.experts`、`moe.shared_experts` |
| `model_overrides` / `mxfp8_fqns` | 有 typed schema、校验和 CLI | 有稳定 schema，重建逐层 KDA/MLA/dense/MoE 配置 |
| 参数存储 dtype 假设 / saved activation offload / FSDP AG FP8 | 公共内存/通信建模路径 | 公共内存/通信建模路径 |
| EP dispatch FP8 假设 | 默认 NpuExpertParallel 路径已接入 | 当前默认 converter/ExpertParallel 路径未接入该开关 |
| fake AllToAll 输出 SAC 保存扩展 | NpuExpertParallel synthetic bridge 已接入，须显式 `all-to-all` | 通用 funcol A2A shim 未接入同一 synthetic cache；不能直接标为支持 |
| embedding + 第 0 层 replica 实验 | 专属选项；保留 EP/TP，改用组内 replica + grad AR；不支持 tied embedding | 没有该专属选项 |

限制证据：[DeepSeek parallelize](../../../torchtitan_npu/models/deepseek_v4/parallelize.py)、[MTP+PP 检查](../../../torchtitan_npu/models/deepseek_v4/model.py)、[Kimi PP 检查](../../../torchtitan_npu/models/kimi_k3/parallelize.py)。

## 代码已存在但不属于正式 simulator 支持矩阵

- Kimi `16layer_reduced`：模型 registry 和真实训练 recipe 存在，16 层、32 专家；当前 simulator registry 没有对应 wrapper。可通过已接入 preset 的 overrides 配置近似规模，但不等于已有同名仿真 preset。
- Llama3/Llama4、Qwen3、DeepSeek V3/V3.2、VLM 等：基础 NPU 插件仓的模型/训练能力不等于 simulator 已正式接入；当前 simulator registry 未提供这些模型工厂。
- `raw_model/ar_llm`：CSA/HCA/KDA、LatentMoE、Mixture of Recursions、Engram 等超大 AR 模型原始设计。
- `raw_model/world_model`：视频 DiT、VAE、文本编码器、相机/深度组件及交互 world model 原始结构。
- `raw_model/unified_mm`：GLM5-Next 风格统一多模态原始结构。
- `raw_model/mm_gc`：SLA2 + MoE / Multi-Head MoE 原始结构。
- `raw_model/block_diff`：Block Diffusion 原始结构。

以上 raw_model 未发现正式 ModelSpec + SimulationTrainer registry + 并行/shim 接入闭环；部分原始文件还引用未随目录提供的 companion module，不能将“文件已加入”写成“可以完整仿真”。

模型规模字段详见[DeepSeek V4 参数目录](../deepseek_v4_model_parameters.md)和[Kimi K3 参数目录](../kimi_k3_model_parameters.md)。
