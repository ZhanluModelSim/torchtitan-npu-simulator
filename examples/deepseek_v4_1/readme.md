# DeepSeek-V4.1 training examples

每个 shell 入口只选择与文件名一致的完整 Python 工厂，配置位于 [`torchtitan_npu_recipes.examples.deepseek_v4_1`](../../torchtitan_npu_recipes/examples/deepseek_v4_1.py)。数据、上下文、token 预算、优化器、量化、checkpoint、profiling 与 override 列表均由 recipe 拥有，shell 只保留设备进程数和运行环境。

## 选择和查看配置

例如查看单卡裁剪模型的完整配置：

```bash
bash examples/deepseek_v4_1/debug/deepseek_v4_1_mini_1p_cpt_a5.sh --print-config
```

该入口默认使用 2048 token 上下文、5 步训练和 block-FP8。`--print-config` 不启动分布式进程，也不构建模型权重；需要的 Python 依赖仍必须先安装。多模态、SFT 和 LoRA 的已有依赖缺口可能在构造配置时直接报告。

同一规则适用于 40 层多模态 debug、完整 Flash、QAT、LoRA、SFT 和 1M 文本 CP 入口。`USE_GOLDEN`、`USE_TILELANG` 及支持的路径环境变量由 Python 工厂读取。示例默认使用非 golden 算子组合，参考组合需显式设置 `USE_GOLDEN=1`；A3/A5 算子选择保留在对应 recipe 中。A3 环境版本见[安装指南](../../docs/user-guides/installation.md)，配置存在不等于训练路径已完成适配。

## 自定义配置

在仓库根目录创建 `my_recipes.py`，从所需入口组合配置：

```python
from torchtitan_npu_recipes.examples.deepseek_v4_1 import deepseek_v4_1_mini_1p_cpt_a5


def five_steps():
    config = deepseek_v4_1_mini_1p_cpt_a5()
    config.training.steps = 5
    config.hf_assets_path = "tests/assets/deepseek_v3"
    config.extension.quantization.enable_quantized_training = False
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
```

通过原 launcher 选择该配置：

```bash
MODULE=my_recipes CONFIG=five_steps \
  bash examples/deepseek_v4_1/debug/deepseek_v4_1_mini_1p_cpt_a5.sh --print-config
```

训练设置不再使用通用字段 CLI。token 预算包括 padding，每步预算需匹配有效 DP 度和 microbatch；修改上下文时还需同步模型、RoPE 和数据对齐。完整接口见 [Python 配置指南](../../docs/developer_guides/python_recipes.md)。

## 资源与 checkpoint

`HF_ASSETS_PATH` 指定 HF 资源目录；该路径本身不要求加载初始权重。`DATASET`、`DATASET_PATH`、`DATA_FILES` 和 SFT 专用环境变量由所选 recipe 构造数据图。完整 Flash 的 `CKPT_SAVE_LOAD_PATH`、`CKPT_INIT_LOAD_PATH` 决定恢复目录和 HF 初始权重来源。debug 默认关闭 checkpoint；需要保存或加载时在工厂中设置具体 checkpointer。

多机启动使用 `NODE_IPS` 或 `NNODES`、`MASTER_ADDR` 和 `NODE_RANK`，每节点进程数由 `NGPU` 指定。A3 完整 Flash 默认 16 进程 × 8 节点，A5 默认 8 进程 × 16 节点。A5 affinity 和 HCCL 超时由 shell 的运行环境提供。`--print-config` 可以在未配置分布式网络时检查所请求的逻辑拓扑。

旧的 `CLI_OVERRIDES`、`OPTIMIZER_OVERRIDES`、`DATALOADER_OVERRIDES` 和 `MHC_POST_OVERRIDE` 配置文本不再解析；设置这些变量时会给出迁移提示。完整替换列表使用 `config.override.imports`，追加一个注册的 override 使用原生 `--override`。

## 当前验证边界

1M 文本 recipe 保留 CP/EP、Engram 关闭和数据 padding 的声明；当前模型的 CP 限制仍会在准备训练时报告。多模态、SFT、LoRA 和编译也保留各自的未完成路径，不回退到其他模式。CPU 配置检查不能证明融合 NPU 算子、完整训练、保存恢复或数值对齐通过；历史训练记录不能代替当前配置的 NPU 验证。

### 融合算子逐项说明

融合栈由 A3/A5 脚本默认启用，reference 使用同一入口的 `USE_GOLDEN=1`。逐项语义：

| override | 替换范围 | 实现 |
|---|---|---|
| `common.rms_norm.asc` | decoder、compressor、indexer 与视觉塔的公共 RMSNorm（保留各位置原有 epsilon） | Ascend RMSNorm |
| `common.rope.asc_complex`（A3 文本） | attention/compressor/indexer 的 split-aware 文本旋转（公共 `ComplexRoPE.Config`，保留 split/theta/YaRN） | Ascend rotary mul（interleave） |
| `common.rope.asc_partial`（A5 文本） | 同上三处文本位点；单个 `inplace_partial_rotary_mul` 只旋转尾部 `dim` 通道 | AscendC partial rotary |
| `common.rope.asc_half_rotation` | 视觉塔 2D 位置表的 half 旋转（半宽 cos/sin，融合前复制为全宽表，逐 batch 折叠保位置） | 同上 |
| `sparse_attn.asc`（仅 A5，须与下一行同开） | 替换完整 attention forward；反向除自身梯度外还在 `topk_scores` 梯度上产出 indexer 教师。该 config 按**自己的几何**（每个 `(端口, compress_ratio)` 一个 map entry）每微批预计算 SMLA 与 indexer 元数据（`sparse_flash_mla_metadata` / `sparse_flash_mla_grad_metadata` / `lightning_indexer_metadata` / `sparse_lightning_indexer_kl_loss_grad_metadata`），随 entry 的 `aicpu_metadata` 字段下发，前向/重算/反向直接读字段（没有查表）。几何即该 config 自己的必填字段（无默认值），由该入口的五个参数一次写入，默认值即 V4.1-Flash 的发布形状；记录是内核 metadata 的唯一来源：拿着没经过端口构建的 metadata 前向会直接失败，不会在层里补一次算子调用（没有 inline 回退） | Ascend sparse flash MLA + CANN `*_metadata` |
| `lightning_indexer.asc`（仅 A5，须与上一行同开） | 替换 selector 节点 | CANN `lightning_indexer` + `sparse_lightning_indexer_kl_loss_grad` |
| `common.swiglu_group.asc`（仅 A5） | `*.moe.routed_experts.activation_fn` 的 SwiGLU | cann_ops_nn.swiglu_group |
| `common.swiglu_group.asc_shared_experts`（仅 A5） | `*.moe.shared_experts` 的 SwiGLU（FQN 限定，不替换视觉 MLP） | 同上 |
| `common.token_dispatcher.asc` | 公共 MoE 的 permute / re-routing / unpermute；reference 不启用 | Ascend token dispatcher |
| `mhc.asc_hc_post` | mHC post 变换 | Ascend mHC post |
| `mhc.asc_sinkhorn`（仅 A5） | `HcPre._split_sinkhorn`（新接口，`hc_eps` 字段） | npu_mhc_sinkhorn |

量化路径不是这两个 override 的包装，而是另一个 backend：开启 `config.extension.quantization.enable_sparse_attention_quantization = True` 后，量化后的 sparse attention（`torchao_npu` 里 fused attention 的拷贝）由 converter 安装，它自己构建 entry 的四份 metadata（含量化 selector 的两条 schedule），因此 `sparse_attn.asc` 在该路径上不生效；`config.extension.quantization.li_quantization = "mxfp4"` 必须与该开关同时开启，否则非量化 attention 的 BF16 schedule 会被量化 kernel 读到（垃圾索引，kernel 可能崩），converter 会直接报错。非量化路径的 `lightning_indexer.fwd_sparse` / `.sparse_length` 当前留空，等带 candidate 的非量化算子落地后由 port 填。

routed experts 使用独立的 `GroupedLinear.Config`：`w13` 保存 gate/up 投影，`w2` 保存 down 投影。融合 override 仅替换 `activation_fn`，保留分组矩阵乘和 dispatcher；路由分数在激活后、`w2` 前单独乘入。参数名现为 `routed_experts.w13.weight` 和 `routed_experts.w2.weight`。融合输出、输入/权重梯度与训练轨迹的误差需与迁移后同配置 reference 对照验收；仅运行成功不代表数值或性能验收完成。

### RoPE seam 说明

文本旋转由主干构造公共 `ComplexRoPE.Config`（带 split 前缀宽度）：reference 通过 `common.rope.workaround` 用展开实数表；A3 融合选 `asc_complex`，A5 融合选 `asc_partial`，三者精确匹配同一配置类，同一列表只能出现一个（override 框架对同节点双 claim 直接报错）。视觉塔的二维位置表使用 `HalfRotation` / `asc_half_rotation`，与文本 interleaved 布局不同，不能互换。inverse（attention o_rope 路径）经 `-sin` 支持。

CPU 测试覆盖两类 seam 的输出/梯度、partial 非连续输入、逐 batch 位置与配置隔离；CPU mock 不代表 NPU kernel 验证通过。

### Muon 优化器注意事项

Python 工厂通过 `config.optim.optimizer.name = "Muon"` 选择 DistMuon，并调用 `materialize()` 构造 `[DistMuon(pattern), AdamW(.*)]`。自定义 Muon 超参应先设置 `config.optim.optimizer.lr`、`weight_decay`、`muon_momentum` 等 schema 字段，再调用 `materialize()`；直接改旧的 optimizer 子组字段会在重新 materialize 时被替换。indexer 仍属于 AdamW fallback 组。

### 验证范围

CPU 单测覆盖模型与融合配置选择、RoPE 输出/梯度等核心行为；不保留专用的 V4.1 两卡手动冒烟 suite。真实训练通过上述 A3/A5 入口执行。历史 A3 的 50 步对照见 PR 845；该证据不代表当前 HEAD 或 A5 新 CANN 已完成重验。

## LoRA

通过 [LoRA 脚本](deepseek_v4_1_flash_lora_4k_a3.sh) 选择同名工厂 `torchtitan_npu_recipes.examples.deepseek_v4_1.deepseek_v4_1_flash_lora_4k_a3`。
纯文本基础工厂为 `torchtitan_npu_recipes.models.deepseek_v4_1.deepseek_v4_1_flash_text_lora`，通过通用 `scripts/run_train.sh` 选择；两者仍受当前 LoRA 迁移限制。
AdamW/Muon、Engram 冻结、native DCP 恢复和 PEFT 导出/合并方式见 [V4.1 LoRA 指南](../../docs/feature_guides/deepseek_v4_1_lora.md)。
