# DeepSeek-V4.1 GraphTrainer 编译路径适配

GraphTrainer 将模型前向、loss 和反向捕获为联合 FX 图，使分片通信、激活重计算与 MoE 通信重叠能够在图内优化。本仓基于 TorchTitan v0.4.0，为 DeepSeek-V4.1（DSV4.1）的纯文本与单图多模态模型提供 GraphTrainer recipe。

## 启动训练

### 公共 launcher

安装配套依赖并准备 tokenizer 后，从仓库根目录启动 A3 八卡纯文本配置：

```bash
export ASCEND_SET_ENV_PATH=/path/to/cann/set_env.sh
export HF_ASSETS_PATH=/path/to/DeepSeekV41_tokenizer
export TORCHTITAN_DIR=/path/to/torchtitan
export PYTHONPATH="${TORCHTITAN_DIR}:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODULE=torchtitan_npu_recipes.tests.graph_trainer.deepseek_v4_1 \
CONFIG=graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a3 \
bash scripts/run_train.sh
```

`ASCEND_SET_ENV_PATH` 用于指定 CANN 环境脚本，需导出给 launcher；未设置时优先查找 `/usr/local/Ascend/cann/set_env.sh`，再尝试其他标准安装路径。`HF_ASSETS_PATH` 用于 tokenizer 等模型资产，其压缩词表必须与 Host Engram 配置匹配；设置该路径本身不等于加载模型权重。

调试 recipe 位于 [tests/graph_trainer/deepseek_v4_1.py](../../torchtitan_npu_recipes/tests/graph_trainer/deepseek_v4_1.py)，共用同一个工厂：

| 配置函数 | 模态 | 默认量化训练 |
| --- | --- | --- |
| `graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a3` | 纯文本 | 关闭 |
| `graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a3_multimodal` | 单图多模态 | 关闭 |
| `graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a5` | 纯文本 | `all_block_fp8` |
| `graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a5_multimodal` | 单图多模态 | `all_block_fp8` |

默认 8 卡、4K、40 步、FSDP8/EP8，A3 使用 40 层 / 16 experts，A5 使用 40 层 / 24 experts。`num_tokens_per_train_step=-1` 由框架推导 token budget，保持每步一个 microbatch。层数、步数和其他训练字段在 Python recipe 中修改，配置组织见 [recipe 指南](../../torchtitan_npu_recipes/README.md)。

多模态使用同一模块，更换 `CONFIG` 和数据目录：

```bash
DATASET=cc12m-test \
DATASET_PATH=tests/assets/cc12m_test \
PACKING_BUFFER_SIZE=0 \
MODULE=torchtitan_npu_recipes.tests.graph_trainer.deepseek_v4_1 \
CONFIG=graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a3_multimodal \
bash scripts/run_train.sh
```

### 三种图执行模式

在返回配置前修改 `config.compile`：

| 场景 | `ep_overlap.enabled` | `pass_pipeline` |
| --- | --- | --- |
| 基础 GraphTrainer | `False` | `mutation-functionalization` |
| EP overlap | `True` | `mutation-functionalization` |
| Auto Overlap（默认） | `True` | `mutation-functionalization+npu_auto_overlap` |

三种模式保留相同的模型、Host Engram 和优化器配置。Auto Overlap 在正式训练前执行候选图校准，首步耗时包含抓图、校准及编译。

## 上游 GraphTrainer 能力

DSV4.1 复用上游图训练设施，而不是另建训练循环：

- **联合图捕获**：在 FakeTensor 模式下捕获前向、loss 与反向，反向计算显式进入 FX 图。
- **SimpleFSDP**：以可追踪的 DTensor 操作表达参数 AllGather 与梯度 ReduceScatter，集合通信参与 graph pass 调度。
- **图级激活重计算**：由 memory policy 决定激活保存和重算；本仓 DSV4.1 recipe 选择 `memory_policy="full"`。
- **EP overlap 与 pass pipeline**：通过 MoE chunk 边界标注识别通信和计算区域，本仓 Auto Overlap 在此基础上测量候选图并调整调度。

NPU recipe 显式设置 `disable_passes=["cudagraph_pass"]`，不启用上游 CUDA Graph capture。DSV4.1 的并行支持范围不等同于上游 GraphTrainer 的所有组合。

## DSV4.1 支持

图训练 recipe 复用 DSV4.1 的模型和优化器配置，接入上游 GraphTrainingEngine，并提供以下能力：

- **文本与单图多模态**：保留跨层稀疏注意力和 Lightning Indexer 训练逻辑，多模态路径支持视觉编码、图像特征对齐和图像 token 路由。
- **MoE 通信重叠**：支持基础 GraphTrainer、EP overlap 和 Auto Overlap，默认开启 Auto Overlap；多模态的图像 mask 与 token 一起切分。
- **Torch Host Engram**：与 FSDP2 一样默认开启，CPU 表使用独立 SparseAdam，稠密参数使用 Muon / AdamW。候选图校准与正式训练的稀疏梯度状态隔离。

调试配置默认将每个 hash head 的目标表行数缩为 `100000`，可通过 `TORCHTITAN_ENGRAM_TABLE_ROWS` 调整。该设置保留 N-gram、head 数、embedding width 和 tokenizer compression 规则；完整表及优化器状态需要另行评估主机内存。

配置入口见 [GraphTrainer recipe](../../torchtitan_npu_recipes/graph_trainer/deepseek_v4_1.py)。通信重叠原理见 [EP overlap](graph_trainer_ep_overlap.md) 和 [Auto Overlap](graph_trainer_auto_overlap.md)。

## 支持范围与限制

| 能力 | 当前范围与限制 |
| --- | --- |
| 并行拓扑 | 当前仅验证 `FSDP = EP`、`TP = CP = PP = 1`。`CP` 场景未适配。 |
| 梯度累积 | 默认只使用一个 microbatch，TorchTitan v0.4.0 的 SPMD 路径不支持将梯度累积与 EP overlap 或自定义 pass pipeline 组合使用。 |
| 多模态 | 每个 microbatch 仅一张图。符号 patch 维已有实现，不代表所有分辨率组合与连续变分辨率回放都已验证。 |
| Host Engram | 默认开启原生 Torch Host lookup，CPU 表使用 SparseAdam；暂不支持 ElasticBuffer 后端，发布规模表及图路径 checkpoint 保存恢复尚未验证。 |

当前已验证 A3 八卡裁层模型的短步功能；A5 FP8、完整模型精度和长期稳定性仍需分别验收。

## 相关文档

- [GraphTrainer EP overlap](graph_trainer_ep_overlap.md)。
- [GraphTrainer Auto Overlap](graph_trainer_auto_overlap.md)。
- [Recipe 组织与公共 launcher](../../torchtitan_npu_recipes/README.md)。
