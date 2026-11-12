# Qwen3.5 NPU training examples

本目录的两个脚本分别选择 `torchtitan_npu_recipes.examples.qwen3_5` 中的同名 Python 工厂。
模型、数据、训练预算、优化器和 override 由工厂组合；shell 只选择配置与启动资源：

- `qwen3_5_debugmodel_1p_cpt_512.sh`：Qwen3.5 多模态 debugmodel 1 卡冒烟入口，
  复用 `scripts/run_train.sh`，默认 cc12m-test 数据、512 token 上下文、10 步。
- `qwen3_5_debugmodel_moe_8p_cpt_512.sh`：Qwen3.5 MoE debugmodel 8 卡入口，
  默认 EP=8、DP_SHARD=8，每个 microbatch 为 1024 token，训练 10 步。

纯文本 SFT 的 `qwen35_27b_4k_sft` 和 `qwen35_27b_long_text_sft` recipe 声明使用
`GrainDataLoader`、Hugging Face JSON/JSONL source、上游 `ChatProcessor` 和 first-fit packing。
`DATA_FILES` 指定文件，长文本 recipe 支持逗号分隔的多个文件；样本由各自的消息转换函数映射为
单轮 `[user, assistant]`，原始 `messages` 也须满足上游单轮约束。训练长度与 microbatch 均使用 token 预算字段。
多模态 debug recipe 直接使用上游 `GrainDataLoader`、multimodal processor 和 collator，
通过 `DATASET` / `DATASET_PATH` 选择数据源。输入 tokens 和 positions 为 `[T]`，MRoPE 为 `[T,3]`；
视频的连续时间坐标由当前固定上游构建，不再安装旧的 batched MRoPE patch。
27B 模型迁移尚未完成，SFT 入口目前会明确拒绝构造；保留配置不代表训练支持已完成。
当前 pin 的完整 NPU 冒烟仍会在 GDN 的 CUDA-only short-convolution 路径失败；
数据管线和 MRoPE 的 CPU 验证通过不代表该训练后端已完成适配。

## 多模态依赖

Qwen3.5 复用上游的图像与视频预处理。先按[安装指南](../../docs/user-guides/installation.md)
准备匹配的 Torch、torch_npu 与 torchvision，再安装 `av`、`einops`、`pillow`。
已有匹配的软件栈时，可使用以下命令安装源码，避免重新解析容器中的定制运行时依赖：

```bash
python -m pip install av einops pillow
python -m pip install --no-deps -e ../torchtitan
python -m pip install --no-deps -e .
```

## Qwen3.5 NPU override 开关

示例工厂显式选择 Triton GDN override，并组合上游 debug 模型和原生 Grain 数据源。
视觉 mask 和 FLA 兼容补丁由 `torchtitan_npu.models.qwen3_5`
导入时按上游接口自动安装。

MoE token dispatcher 是性能选项（`ENABLE_NPU_MOE_DISPATCHER` 缺省值为 `0`），
默认关闭。可通过环境变量启用：

```bash
ENABLE_NPU_MOE_DISPATCHER=1 \
  bash examples/qwen3_5/qwen3_5_debugmodel_moe_8p_cpt_512.sh --print-config
```

原生 `--override` 向工厂已有的列表追加条目，例如：

```bash
bash examples/qwen3_5/qwen3_5_debugmodel_moe_8p_cpt_512.sh \
  --override torchtitan_npu.override.common.token_dispatcher.asc --print-config
```

需要替换或清空列表时，在 Python 工厂中设置 `config.override.imports`。

### 启动参数与路径

从仓库根目录运行脚本。`MODULE`、`CONFIG` 可选择自定义 Python 工厂，额外参数只使用
TorchTitan 0.4.0 的原生启动选项，例如 `--print-config`、`--output-dir`、`--override`。
训练步数、token 预算和完整 override 列表由 Python 配置修改。工厂按以下优先级选择上游资源目录：

1. `TORCHTITAN_REPO`；
2. `TORCHTITAN_DIR`（`.ci/common.sh` 会导出此变量指向 third_party 克隆）；
3. 仓库同级的 `../torchtitan`。

`HF_ASSETS_PATH` 可指定 tokenizer 目录；否则使用所选上游目录的 `tests/assets/tokenizer`。
数据路径优先读取 `QWEN_DATASET_PATH`，其次读取 `DATASET_PATH`；均未指定时使用上游目录的
`tests/assets/cc12m_test`。编译策略也由 Python 工厂声明，当前 NPU 后端限制仍需单独验证。

例如在仓库根目录保存 `my_qwen_recipes.py`，定义两步、清空默认 override 的 MoE 配置：

```python
from torchtitan_npu_recipes.examples.qwen3_5 import qwen3_5_debugmodel_moe_8p_cpt_512


def moe_two_steps():
    config = qwen3_5_debugmodel_moe_8p_cpt_512()
    config.training.steps = 2
    config.override.imports = []
    return config
```

```bash
TORCHTITAN_DIR=/path/to/torchtitan \
MODULE=my_qwen_recipes CONFIG=moe_two_steps LOG_RANK=5 \
  bash examples/qwen3_5/qwen3_5_debugmodel_moe_8p_cpt_512.sh --print-config
```

此命令只检查配置。清空 override 用于消融或定位问题，不会补齐 CUDA-only 路径的 NPU 支持。

### Qwen3.5-VL 适配链路

入口关系如下；后续训练仍受上述依赖与后端限制：

```text
qwen3_5_debugmodel_1p_cpt_512.sh
  └─ 选择 MODULE、CONFIG、NGPU -> scripts/run_train.sh
       └─ torchtitan_npu.train -> ConfigLoader
            └─ torchtitan_npu_recipes.examples.qwen3_5 中的同名工厂
                 ├─ 组合上游模型、Grain 数据源、tokenizer 与训练预算
                 ├─ 根据资源环境变量设置路径，显式选择 override.imports
                 └─ Trainer 应用 override、构建模型和数据管线，再执行训练
```

固定的 TorchTitan 版本仅使用 `spmd_types`，launcher 不再接受 backend 选择。
Qwen3.5 的长文本 CP 配方保留 CP=4 及 typed Ulysses 变换配置；当前固定上游的 Ulysses
实现要求 CUDA，NPU runtime 仍不受支持，启动时会明确拒绝。多模态 CP 的视觉元数据适配
也尚未完成。配方构造和 CLI 解析通过不代表这些 CP 训练路径可运行。

已有 Qwen3.5 单卡 NPU 冒烟检查入口如下；执行仍受上述后端限制：

```bash
python -m tests.integration_tests.run_tests /tmp/qwen3_5-tests \
  --test_suite qwen3_5 --test_name all --ngpu 1
```

## Qwen3.5 视觉 mask 边界

当前 NPU eager Flex 兼容层只处理 Qwen3.5 vision patch 显式标记的、等长且
各 attention head 共享的 self-attention mask；其他模型和其他 Flex 语义继续走
原生路径。dense bool mask 在一次 vision forward 内缓存并由所有 vision block
复用；aot_eager 等编译追踪期间，满足同样条件的 NPU Flex mask（即使未被
vision patch 显式标记）也会走该有界 dense SDPA 路径，eager 执行仍以显式
标记为界。

为避免合法的大图或长视频直接触发 NPU OOM，物化上限为 `16,777,216` 个 bool
元素（单样本等长输入约为 4096 个 merged vision token）。超过上限会在算子下发
前 fail-fast；请通过数据配置降低图片最大像素、视频帧数或视觉序列长度。
