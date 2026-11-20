# TorchTitan-NPU Recipes

## 1. 目录结构与规则

本目录通过 Python 函数返回完整训练配置，统一组织模型、数据、优化器、并行策略、checkpoint 和算子选择。

| 目录 | 规则 |
| --- | --- |
| `models/` | 存放经过验证的完整模型运行配置 |
| `graph_trainer/` | 存放经过验证的完整模型 GraphTrainer 配置，函数名使用 `graph_trainer_` 前缀 |
| `tests/` | 存放调试、裁减模型和 CI 配置；按 `models/`、`graph_trainer/`、`ci/`、`torchft/` 分类 |

## 2. 当前完整模型配置

以下模块均以 `torchtitan_npu_recipes.` 为前缀；不列调试和 CI 配方。同系列用 `{a3,a5}` 表示两个独立函数。

| 模块 | 配置函数 | 用途 |
| --- | --- | --- |
| `models.deepseek_v4` | `deepseek_v4_flash_cpt_4k_a3` | 128 卡 Flash CPT，4K、BF16、Muon + swap |
| 同上 | `deepseek_v4_flash_cpt_4k_a5` | 128 卡 Flash CPT，4K、DeepEP、block-FP8 |
| 同上 | `deepseek_v4_flash_qat_4k_a5` | A5 CPT 基础上启用 MXFP4 QAT |
| 同上 | `deepseek_v4_flash_sft_4k_a3` | 128 卡对话 SFT，允许保存 checkpoint |
| 同上 | `deepseek_v4_flash_lora_4k_a3` | 128 卡 LoRA，rank 16，PEFT checkpoint |
| 同上 | `deepseek_v4_flash_cpt_1024k_a5` | 1M 上下文、CP128；待迁移 |
| `graph_trainer.deepseek_v4` | `graph_trainer_deepseek_v4_flash_cpt_4k_a5` | 128 卡 A5 图训练、DeepEP、block-FP8；当前配置 schema 不兼容 |

上述完整模型组合尚未完成当前分支的统一验证；编译路径仍受当前 trainer 限制。

## 3. 运行配置

安装依赖并准备 CANN、tokenizer 和数据后，先修改原 recipe，再从仓库根目录直接启动。

以 `deepseek_v4_flash_43layers_16experts_eager_bf16` 为例：

1. 在 [tests/models/deepseek_v4.py](tests/models/deepseek_v4.py) 的对应函数中，将 `hf_assets_path`、`data_files` 的默认值改为实际路径；如需加载或保存权重，在配方中设置 `checkpointer`。
2. 该配方默认 8 卡 EP8 / FSDP8、4K 上下文、100 步，使用 BF16、Muon + swap 和融合算子。改变卡数时同步修改配方中的并行布局。
3. 选择模块和配置函数，直接运行：

```bash
ASCEND_SET_ENV_PATH=/path/to/cann/set_env.sh \
NGPU=8 \
MODULE=torchtitan_npu_recipes.tests.models.deepseek_v4 \
CONFIG=deepseek_v4_flash_43layers_16experts_eager_bf16 \
bash scripts/run_train.sh
```

追加 `--print-config` 可只打印配置。运行其他配方时替换 `MODULE` / `CONFIG`，或直接在 `scripts/run_train.sh` 中修改。资产路径和训练参数直接在对应 recipe 中修改，进程数须与并行布局一致。多机使用 `scripts/run_train_multinodes.sh`，各节点设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `NGPU`。
