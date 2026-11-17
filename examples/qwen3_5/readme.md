# Qwen3.5 NPU training examples

本目录提供 Qwen3.5 在 NPU 上的训练入口：

- `qwen3_5_debugmodel_1p_cpt_512.sh`：Qwen3.5 多模态 debugmodel 1 卡冒烟入口，
  复用 `scripts/run_train.sh`，默认 `qwen35_debugmodel` 配置、cc12m-test 数据、
  512 序列 10 步。`ENABLE_NPU_MOE_DISPATCHER=1` 可选启用 ASC MoE dispatcher。
- `qwen3_5_debugmodel_moe_8p_cpt_512.sh`：Qwen3.5 MoE debugmodel 8 卡入口，
  默认 `qwen35_debugmodel_moe` 配置，EP=8/DP_SHARD=8（spmd_types 后端）。

## Qwen3.5 multimodal dependencies

The Qwen3.5 adapter reuses TorchTitan's upstream image and video preprocessing.
Install the torchvision nightly that declares compatibility with the container's
Torch build, then install this repository without resolving the container's
custom Torch/torch_npu pair:

```bash
python -m pip install av einops pillow
python -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/nightly/cpu \
  torchvision==0.29.0.dev20260814+cpu
python -m pip install --no-deps -e ../torchtitan
python -m pip install --no-deps -e .
```

## Qwen3.5 NPU override 开关

默认启用 Triton GDN override。配置工厂保持与上游完全一致，不会静默注入 NPU
实现；视觉 mask、视频 MRoPE 和 FLA 兼容补丁由 `torchtitan_npu.models.qwen3_5`
导入时按上游接口自动安装。

MoE token dispatcher 是性能选项（`ENABLE_NPU_MOE_DISPATCHER` 缺省值为 `0`），
默认关闭。可通过以下任一方式显式启用：

```bash
ENABLE_NPU_MOE_DISPATCHER=1 bash examples/qwen3_5/qwen3_5_debugmodel_1p_cpt_512.sh

OVERRIDE_IMPORTS="torchtitan_npu.override.qwen3_5.gated_delta.npu,torchtitan_npu.override.common.token_dispatcher.asc" \
  bash examples/qwen3_5/qwen3_5_debugmodel_1p_cpt_512.sh
```

用户设置的 `OVERRIDE_IMPORTS` 始终优先；显式设置为空字符串可禁用全部默认
override，用于消融或定位问题。

### 启动参数与路径

本目录的入口脚本与仓库通用的 `scripts/run_train.sh` 使用相同的参数透传方式：
脚本只设置模型、数据和多模态参数默认值，所有训练参数继续交给
`torchtitan.train`。上游目录按以下优先级解析：

1. `TORCHTITAN_REPO`；
2. `TORCHTITAN_DIR`（`.ci/common.sh` 会导出此变量指向 third_party 克隆）；
3. 仓库同级的 `../torchtitan`。

数据和路径也可以覆盖：`HF_ASSETS_PATH` 默认为
`${TORCHTITAN_REPO}/tests/assets/tokenizer`，`DATASET_PATH` 默认为
`${TORCHTITAN_REPO}/tests/assets/cc12m_test`。如需使用 torch.compile，可在
脚本中修改 `COMPILE_ARGS` 或通过 `COMPILE_BACKEND` 环境变量控制。

示例（单机 8 卡 MoE）：

```bash
TORCHTITAN_DIR=/path/to/torchtitan \
NGPU=8 CONFIG=qwen35_debugmodel_moe LOG_RANK=5 \
  bash examples/qwen3_5/qwen3_5_debugmodel_moe_8p_cpt_512.sh --training.steps 2
```

### Qwen3.5-VL 适配链路

运行时调用关系如下，NPU 适配只替换必要边界，上游 recipe 和模型主体仍由
TorchTitan 提供：

```text
qwen3_5_debugmodel_1p_cpt_512.sh
  ÚÄ 组装 tokenizer、cc12m-test、override.imports 和用户参数
  ÚÄ scripts/run_train.sh -> torchrun -m torchtitan_npu.train
       ÚÄ torchtitan_npu.models.qwen3_5 导入时安装视觉/MRoPE/FLA patch
       ÚÄ 上游 config_registry 构造 qwen35_debugmodel(_moe)
       ÚÄ GDN override 替换 delta kernel；可选 dispatcher 替换 MoE dispatch
       ÚÄ parallelize_qwen3_5_npu 建立 TP/EP/FSDP；CP 使用序列元数据与 mesh resolver
       ÚÄ upstream multimodal collator 读取文本、图像/视频并生成 MRoPE
       ÚÄ vision encoder + language decoder 前向，执行 GDN/MoE/CP 数据交换
       ÚÄ backward、优化器更新与 TensorBoard/JSONL 日志
       ä 进程组销毁
```

`spmd_types` 使用 TorchTitan v0.3 的 `resolve_fsdp_mesh`/
`resolve_sparse_fsdp_mesh`，而旧 backend 保留 `fsdp/efsdp` 名称；因此同一套
parallelizer 可兼容当前主线和旧 NPU 运行环境。当前 Qwen3.5-VL 多模态视觉 mask
尚未完成 CP 序列元数据适配；VL 配置启用 CP 时会在并行化入口明确失败，避免
运行到错误的 dict-mask 路径。纯文本长上下文配置仍可使用专用 CP recipe。

集成测试通过以下命令运行 Qwen3.5 的 1-rank 冒烟用例：

```bash
python -m tests.integration_tests.run_tests /tmp/qwen3_5-tests \
  --test_suite qwen3_5 --test_name all --ngpu 2
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
