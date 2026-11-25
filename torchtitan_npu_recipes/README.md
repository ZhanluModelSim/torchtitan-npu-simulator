# TorchTitan-NPU Recipes

## 1. 目录结构与规则

本目录通过 Python 函数返回完整训练配置，统一组织模型、数据、优化器、并行策略、checkpoint 和算子选择。

| 目录 | 规则 |
| --- | --- |
| `models/` | 存放经过验证的完整模型运行配置 |
| `graph_trainer/` | 存放经过验证的完整模型 GraphTrainer 配置，函数名使用 `graph_trainer_` 前缀 |
| `torchft/` | 只存放完整模型的 TorchFT 配方；裁剪模型放在 `tests/torchft/`，不按卡数区分 |
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
| `torchft.deepseek_v4` | `deepseek_v4_flash_torchft` | 两个 64 卡 replica，43 层、256 专家，FSDP64/EP64 |

上述完整模型组合尚未完成当前分支的统一验证；编译路径仍受当前 trainer 限制。

## 3. 运行配置

安装依赖并准备 CANN、tokenizer 和数据后，先修改原 recipe，再从仓库根目录直接启动。

### 3.1 模型运行示例

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

### 3.2 TorchFT 运行示例

这三个 FT 配方使用 Muon + AdamW，并复用 DeepSeek-V4 模型配方的 reference indexer 蒸馏损失，使 selector 参数产生梯度。比较 FT 开关精度时，普通基线也须启用同一损失配置。

安装 TorchFT 可选依赖后，从仓库根目录启动。八卡配方使用仓内测试 tokenizer 和数据。以下命令分别在三个终端运行；两个训练终端使用相同且可写的输出根目录，新作业请选空目录。若容器已完整激活 CANN 且重复加载环境脚本会报错，可在两个训练终端设置 `ASCEND_SET_ENV_PATH=/dev/null`；未激活 CANN 时不要跳过初始化。

```bash
# 终端一：单机共用一个 Lighthouse
torchft_lighthouse --bind 127.0.0.1:29830 --min_replicas 2
```

```bash
# 两个训练终端都执行
export MODULE=torchtitan_npu_recipes.tests.torchft.deepseek_v4 CONFIG=deepseek_v4_flash_8p_torchft
export TORCHFT_LIGHTHOUSE=http://127.0.0.1:29830 NGPU=4

# 终端二
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TORCHFT_REPLICA_ID=0 LOG_PREFIX=ft_r0 \
    bash scripts/run_train.sh --output-dir ./outputs/ft_8p
# 终端三
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 TORCHFT_REPLICA_ID=1 LOG_PREFIX=ft_r1 \
    bash scripts/run_train.sh --output-dir ./outputs/ft_8p
```

验证 `2→1→2` 时，请另起一轮：Lighthouse 使用 `--min_replicas 1`，两个训练终端均设置 `TORCHFT_MIN_REPLICA_SIZE=1`。确认两个 replica 已共同训练再停止一个；待存活组继续更新后，使用停止组原来的 ID 和输出根目录重新启动。不要把一次八进程启动当作两个 replica。

从旧的 `torchtitan_npu_recipes.tests.ci.deepseek_v4` 迁移时，配置函数名不变。8 卡配置和 `deepseek_v4_flash_43layers_16experts_torchft` 均使用上面的测试模块：后者保留 43 层，但将专家数裁剪到 16，每个 replica 为 FSDP64/EP16。只有完整的 256 专家配置 `deepseek_v4_flash_torchft` 使用 `MODULE=torchtitan_npu_recipes.torchft.deepseek_v4`。

两个 128 卡配置均只验证了配置解析，未实测多机训练；启动时需按每个 replica 的拓扑设置节点和卡数，不能照搬上述单机命令。
