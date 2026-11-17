### V4.1 独立训练基线与融合栈

V4.1 的模型、图像路由、压缩 attention、metadata 和并行化均由 `torchtitan_npu/models/deepseek_v4_1` 持有，不依赖 V4 模型或其专属 override。模型默认算子是 Attention Gym 的 eager `gather_attn`（`CompressedSparseInnerAttention2`）与公共 MoE 工厂；模型目录不含 indexer 训练目标，eager 路径也没有：它只在 A5 的两个融合 override 同开时由两者的反向传递，见下文。

当前支持 **FSDP + EP、TP1 / CP1 / PP1、eager 执行与 torch.compile**，保留 FullAC 与图文输入。A3 脚本组织公共实验参数与融合列表，A5 调用 A3 并追加 CPU 亲和性、partial 文本 RoPE、两项 SwiGLUGroup、sparse attention 与 mHC Sinkhorn；两种入口均通过 `USE_GOLDEN=1` 选择 reference。A3 默认融合包含 RMSNorm、文本（`asc_complex`）与视觉 RoPE、MoE token dispatcher 和 mHC post；A5 将文本 RoPE 换为 `asc_partial` 并追加 routed/shared SwiGLUGroup、sparse 与 Sinkhorn。routed experts 的 grouped GEMM 是 reference 与融合共用的公共路径，不再作为独立融合开关。已接入量化入口及 Host Engram；不支持 MTP、DSpark、GraphTrainer；不支持的 TP、CP、PP 和 compile 配置在入口拒绝。

**torch.compile 已支持**（每个 TransformerBlock 整图编译，跨层状态为显式前向参数），通过脚本 `"$@"` 透传超参开启：

```sh
# aot_eager 整图（fullgraph=True）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --compile.enable --compile.components model --compile.backend aot_eager

# inductor
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --compile.enable --compile.components model --compile.backend inductor
```

编译态按 MoE 路由形态二分：负载均衡路由（A3/A5 脚本默认）下 inductor 保持整图（含 MoE 通信）；数据依赖路由（显式 `--debug.no-moe-force-load-balance`）时 MoE 通信在编译图外执行。

单机 8 卡入口（默认读取仓内上游 CC12M 测试 tar；数据源可通过标准 CLI 覆盖）：

```sh
# A3 融合栈（默认）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir>

# A5（A3 融合栈 + A5 专属算子）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh \
    --hf-assets-path <v41_tokenizer_dir>

# 纯 reference 对照（同一数据与并行形状）
USE_GOLDEN=1 \
    bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir>

# 短跑（如 3 步）：显式同步步数与 LR 调度
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir> \
    --training.steps 3 --lr-scheduler.total-steps 3 --lr-scheduler.warmup-steps 2
```

默认实验为 40 层 / 16 专家、seq4096（4k）、local/global batch 1/8、FSDP8/EP8、Muon（NS steps=10）、eager、FullAC，训练与调度均为 40 步。Python 配方保留模型结构、数据协议和优化器参数布局；实验参数由脚本组织，末尾 CLI 参数覆盖脚本默认值。`CONFIG` 可选择 `deepseek_v4_1_debugmodel_multimodal` 调试宽度，硬件选择仍由脚本负责。

关闭 Engram 使用 `--no-engram-enabled`；优化器由脚本显式选择，末尾 `--optimizer.name AdamW` 仍可覆盖。A3 单机使用 `--training.seq-len 2048`，脚本保留 4k 默认配置。A5 多机保留与单机一致的 swap override；可通过 `OPTIMIZER_OVERRIDES` 显式覆盖。

A5 的 `CPU_AFFINITY_CONF` 应按主机拓扑覆盖。通过 `CLI_OVERRIDES` 扩展融合列表：A3 基础列表之外，文本 RoPE 经该通道按硬件选择（A3 默认 `asc_complex`，A5 换为 `asc_partial`），A5 默认追加 SwiGLUGroup、sparse attention、融合 LI 选分与 Sinkhorn，不重复传入 `--override.imports`。显式传 `--override.imports` 会替换整个集合，需自行包含所需的 RoPE 与 swap optimizer。最终选择随 Trainer Config 打印。

普通运行不强制随机种子和确定性；精度对照须在双方命令中追加 `--debug.seed 42 --debug.deterministic`。默认关闭 checkpoint 且设置 `load_only=True`；保存时同时传 `--checkpoint.enable --checkpoint.no-load-only`。`load_only` 表示禁止保存，与 model-only 加载不同。tokenizer 统一经 `--hf-assets-path` 提供（测试可用仓内 `tests/assets/deepseek_v3` mini tokenizer），该参数本身不加载模型权重。

indexer 的训练目标没有独立的损失项，它只走 A5 两个融合算子的反向夹具：`sparse_attn.asc` 的 SMLAG 反向按每个选中条目产出该层 attention 自身完整 softmax（窗口 + 压缩条目 + sink）的边际质量 `p`，写在 `topk_scores` 的梯度上；`lightning_indexer.asc` 的 SLIKG 反向把它当作教师，由 kernel 施加 `dI = Z·Y - p` 并产出 indexer 的 `q`/`k`/`w` 梯度。`topk_scores` 不参与任务损失，除这条边之外没有消费者，因此**A5 的两个 override 必须同开**：模型构造时按配置类标记检查这一对，只开一个直接报 `ValueError`，不会进入训练。单开 LI 时该梯度为 `None`、SLIKG 反向返回零梯度，indexer 全程不更新；单开 sparse attention 时 reference selector 会把 attention 级别的边际质量加到按 head 求和的 logits 上，训练信号是错的（该组合不做修复，仅由构造期检查拦住）。未使用的 top-k 槽位（`-1`）在教师上被置零。完整契约与 CPU 验证范围见 [override 说明](../../torchtitan_npu/override/README.md#deepseek-v41)。

### Flash 完整主干多机训练

`CONFIG=deepseek_v4_1_flash`，参数对齐 [HF 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)：40 层、隐藏维度 5120、384 个路由专家、1 个共享专家、top-6、MoE 中间维度 2304；attention 64 heads × 512、Q/O LoRA 1280/1024；ViT 32 层、维度 1024、16 heads、MLP 2816。Engram 位于第 1/14 层，逻辑表行数 384006168/384016682，沿用仓内 Host 存储和 2048 行对齐。训练长度默认为 4096；模型配置保留官方的最大位置范围 1048576，运行时按训练长度准备 RoPE。

这里的“完整”指非裁剪的文本主干、ViT 和 Engram。HF 额外的 3 层 MTP 和 DSpark 尚未实现，本入口不代表完整 HF 训练目标均已支持；HF 的 FP8/FP4 权重发布格式也不等于本入口的训练量化配方。

每台机器使用相同的 `NODE_IPS` 列表执行：

```sh
NODE_IPS=<node0_ip>,<node1_ip>,...,<node15_ip> \
HF_ASSETS_PATH=/path/to/DeepSeek-V4.1-Flash \
DATASET_PATH=/path/to/cc12m_tar_dir \
bash examples/deepseek_v4_1/deepseek_v4_1_flash_cpt_4k_a5.sh
```

A5 默认每机 8 卡、EP128/FSDP128、TP1/CP1/PP1、MBS1/GBS1024、100 步、Muon、eager/FullAC、强制负载均衡，默认开启 SMLA/LI 及 block-FP8 训练。A3 对应 `deepseek_v4_1_flash_cpt_4k_a3.sh`，默认每机 16 卡、其余并行度相同，使用 A3 融合列表且不默认量化。

`NGPU`、`EP`、`DP_SHARD`、`GBS`、`STEPS` 可通过环境变量设置；`NODE_IPS` 必填。总卡数须能被 EP 和 DP_SHARD 整除，EP 须兼容 384 专家；MBS 必须为 1。CLI 参数放在最后，可覆盖训练配置；修改拓扑请优先使用上述环境变量，保证派生的 DP_REPLICATE 一致。Host Engram 需要正确的 tokenizer 压缩映射与足够主机内存。

初始权重不会因设置 `HF_ASSETS_PATH` 自动加载：需要时设置 `CKPT_INIT_LOAD_PATH` 指向 HF checkpoint；保存 checkpoint 另加 `--checkpoint.no-load-only`。默认不启用参数/梯度 CPU offload，保留 `swap_optimizer` 对 Engram 的适配。`USE_GOLDEN=1` 切换 reference override；A5 若还要关闭量化，追加 `--extension.quantization.no-enable-quantized-training`。

### Checkpoint 加载范围

以下为追加到训练命令末尾的标准 CLI 参数：

- 原生 DCP 完整续训：`--checkpoint.enable --checkpoint.initial-load-path /path/to/dcp/step-N --checkpoint.no-initial-load-in-hf --checkpoint.no-initial-load-in-hf-quantized --checkpoint.no-initial-load-model-only`。恢复模型、优化器和训练状态；并行布局与原 checkpoint 保持兼容。
- 普通 HF 权重初始化：`--checkpoint.enable --checkpoint.initial-load-path /path/to/hf --checkpoint.initial-load-in-hf --checkpoint.no-initial-load-in-hf-quantized --checkpoint.initial-load-model-only`。只加载模型权重，优化器重新初始化。
- 输入包含官方格式的 Engram MXFP8 权重时，在 HF 初始化参数上追加 `--checkpoint.initial-load-in-hf-quantized`，按 row/block scale 反量化到训练参数。该选项描述输入 checkpoint 格式，与训练时是否启用 Engram MXFP8 算子相互独立。

Engram HF 表以全局逻辑行保存，不包含本地 padding，支持 EP 分片和 EP × EFSDP 联合分片的权重导入导出；离线 DCP 转 HF 同样支持这两种表布局。这不包含原生优化器状态在不同分片布局之间的自动迁移。

当前补齐的是 Engram 表及 gate 的 HF 映射和 MXFP8 读取；V4.1 主线尚不能直接读取官方整包的所有非 Engram 量化权重（例如 Attention），因此开启 quantized 选项不意味着官方整包可以直接开始训练。HF 导出遵循 `export_dtype`，不导出优化器状态，也不生成官方 MXFP8 发布格式。

### 真实 CC12M 数据入口（图像条件 caption 预测）

数据入口与 GitHub TorchTitan 的 Qwen/Kimi 多模态训练一致，使用 CC12M WebDataset tar（同名 `.jpg` / `.txt` 样本对）。默认 `dataset=cc12m-test`，读取仓内 `tests/assets/cc12m_test/`，无需传入 dataloader 参数；其他本地 tar 目录通过 `--dataloader.dataset-path <cc12m_tar_dir>` 指定。不提供 manifest、离线准备或 digest 专用脚本。

数据准备沿用上游的两种方式：

- 在线：选择 `--dataloader.dataset cc12m`，不指定 `dataset-path`，从 Hugging Face 流式读取，无需预先下载完整数据集。
- 本地：下载包含图片和 caption 的 WebDataset tar 分片，使用 `--dataloader.dataset-path <cc12m_tar_dir>`。不解压、不生成 manifest，不预先 tokenize 或计算 ViT 特征；图片预处理在数据加载时执行，ViT 在模型前向中执行。

`cc12m-train-0000.tar` 原样复用 [GitHub TorchTitan v0.3.0 测试资产](https://github.com/pytorch/torchtitan/blob/v0.3.0/tests/assets/cc12m_test/cc12m-train-0000.tar)，包含 32 条图文样本，仅用于测试。完整在线数据源通过 `--dataloader.dataset cc12m` 选择；指定本地目录后，读取的是该目录的数据，而非仓内测试资产。

直接复用上游 `HuggingFaceMultiModalDataset` 的 DP 分片、`MMSamplePacker` 和状态恢复，以及 `ParallelAwareDataloader`。针对 TorchTitan 0.3.0 的 HF 恢复起点跨 epoch 重放问题，按上游文本 loader 的方式补齐 `set_epoch`；packing 缓冲满时调用上游 `flush()`。DSV4.1 只适配协议层：`BOS + 完整图片协议 + caption + EOS`；只有 caption 与 EOS 参与监督，图片特征索引按 pack 中的图片顺序连续编号，每篇文档的位置从零开始。每篇文档按 dataloader 的 `per_doc_alignment`（recipe 由模型压缩比取 LCM 得到，当前为 2）在 EOS 后补至整数个池化组：pad 的 label 为 `-100`，位置继续累加，因此它留在本文档内、既不被监督也不改变下一篇文档的起点。注意 pad 会顶掉奇数长度文档 EOS 的目标（label 在文档内右移，EOS 的下一个位置变成了 pad，而 pad 的 label 是 `-100`），即每篇奇数长度文档少一个监督目标；对齐不覆盖行切分，长于 `seq_len` 的文档仍按行切开，续行另起一套池化组。

与上游一致，`--dataloader.packing-buffer-size 0` 默认关闭 packing；设置正值（例如 `128`）启用。一个 rank 的 local batch 仍为 1，但一个 packed 序列可以包含多篇图文样本。装箱和合并仍调用上游实现；缓冲量还包括待输出的 packed 队列及 loader 预取，不是进程内存的硬上限。

A/B 必须固定 tar 内容及顺序、tokenizer、序列长度、packing buffer、DP/全局 batch、seed 和初始权重。旧模型结构的 manifest/合成轨迹与新入口、新 LI 训练目标的轨迹互不可比；迁移后的对照统一使用新 reference 起点或同一份完整 checkpoint。精确续训使用完整 checkpoint 恢复 loader 状态；model-only 加载只复用权重（新旧结构需按参数名对账后映射，不做静默兼容）。

上游多模态模块需要与 PyTorch 匹配的 `torchvision`（CPU 图像预处理）；按运行环境安装对应版本。数据路径支持 `DATASET_PATH` 环境变量或标准 CLI；不指定路径时使用所选数据集的上游注册源。

### Vision-language SFT（LLaVA / VQA）

SFT 复用同目录的 CPT launcher，只替换 dataloader；不另设 checkpoint 或并行策略。输入可以是 JSON、JSONL 或 Parquet，支持：

- LLaVA/ShareGPT 的 `conversations`（`human` / `gpt` 与 `<image>` 标记）；
- OpenAI message schema 的 `messages` 与有序 text/image content blocks；
- 纯文本 Alpaca `instruction` / `input` / `output`。

`developer` 映射为 V4.1 的 `system`，`last_reminder` 映射为官方 `latest_reminder`。其他 role 会明确报错。训练策略由 override 统一配置，数据行中的同名字段不会覆盖 `thinking_mode`、`drop_thinking`、BOS 或 reasoning effort。

最小 LLaVA VQA 样例：

```json
{"image":"train2014/example.jpg","conversations":[{"from":"human","value":"<image>\nWhat is shown?"},{"from":"gpt","value":"A red bus."}]}
```

图片相对路径默认以数据文件所在目录为根；也可以通过 override 的 `image_root` 指定统一根目录。普通 HTTP(S) URL 不在训练 worker 中下载，需先落盘；data URI 可以直接使用。无图片的纯文本样本沿用同一官方 chat encoding。

默认 `chat` 模式对每个 assistant 回复计算监督，system/user/image/padding label 均为 `-100`，assistant 回复及 EOS 参与 loss。超过 `training.seq_len` 的样本被跳过。与 CPT 一样，每个数据并行 rank 处理一条序列（`training.local_batch_size=1`）；这不限制节点数，global batch 由有效 DP 度和梯度累积扩展。`dataloader.num_workers` 等 worker 配置与 tensor batch 独立并正常透传，DP shard 和恢复状态沿用公共 V4.1 数据生命周期。

```sh
DATASET_PATH=/path/to/vqa.jsonl \
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_sft_4k_a3.sh \
    --hf-assets-path /path/to/DeepSeek-V4.1-Flash
```

A5 使用同样的调用方式，将 launcher 换为同目录
`deepseek_v4_1_flash_8p_sft_4k_a5.sh`；它复用 A5 CPT wrapper，由后者继续负责量化与融合算子配置。

需要覆盖官方 encoding 选项时，将参数放在同一个 override JSON 中，例如：

```sh
'torchtitan_npu.override.deepseek_v4_1.vision_language_dataloader.sft={"thinking_mode":"thinking","drop_thinking":false,"reasoning_effort":"high"}'
```

encoder 只调用模型目录 `encoding/encoding.py` 的 `encode_messages` 公共入口，并通过官方前缀编码求 assistant span；`drop_thinking` 使历史前缀变化时，改用完整对话的公开字段探针求边界。它不复刻工具合并、thinking 清理或逐消息渲染规则；若未来官方版本不再保持上述公共行为，loader 会在生成 loss mask 前明确失败，避免静默监督错位。

### 相同初始权重（生成与加载）

baseline 与融合组要从同一份显式权重出发时，先生成再加载（dcp 格式，model-only）：

```sh
# 生成：跑 3 步保存训练 checkpoint（后续仅加载其中的模型权重）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir> \
    --training.steps 3 --lr-scheduler.total-steps 3 --lr-scheduler.warmup-steps 2 \
    --checkpoint.enable --checkpoint.no-load-only \
    --checkpoint.folder <ckpt_dir>

# 加载同一份权重起跑（--checkpoint.load-only 禁用后续保存）
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
    --hf-assets-path <v41_tokenizer_dir> \
    --checkpoint.enable \
    --checkpoint.initial-load-path <ckpt_dir>/step-3 \
    --checkpoint.initial-load-model-only
```

两个要点：`--checkpoint.enable` 是加载生效的前提（不传时 CheckpointManager 未激活，会静默 fresh start，日志中只有一条 warning）；`initial-load-path` 指向 `step-N` 子目录。输出 `checkpoint.folder` 中已有有效 checkpoint 时会优先恢复它；从 `initial-load-path` 起跑应使用新的输出目录。加载后应核对 checkpoint 日志与实际数据配置，再比较训练轨迹；单凭首步 loss 接近不能证明权重和数据一致。

### 融合算子逐项说明

融合栈由 A3/A5 脚本默认启用，reference 使用同一入口的 `USE_GOLDEN=1`。逐项语义：

| override | 替换范围 | 实现 |
|---|---|---|
| `common.rms_norm.asc` | decoder、compressor、indexer 与视觉塔的公共 RMSNorm（保留各位置原有 epsilon） | Ascend RMSNorm |
| `common.rope.asc_complex`（A3 文本） | attention/compressor/indexer 的 split-aware 文本旋转（公共 `ComplexRoPE.Config`，保留 split/theta/YaRN） | Ascend rotary mul（interleave） |
| `common.rope.asc_partial`（A5 文本） | 同上三处文本位点；单个 `inplace_partial_rotary_mul` 只旋转尾部 `dim` 通道 | AscendC partial rotary |
| `common.rope.asc_half_rotation` | 视觉塔 2D 位置表的 half 旋转（半宽 cos/sin，融合前复制为全宽表，逐 batch 折叠保位置） | 同上 |
| `sparse_attn.asc`（仅 A5，须与下一行同开） | 替换完整 attention forward；反向除自身梯度外还在 `topk_scores` 梯度上产出 indexer 教师。该 config 按**自己的几何**（每个 `(端口, compress_ratio)` 一个 map entry）每微批预计算 SMLA 与 indexer 元数据（`sparse_flash_mla_metadata` / `sparse_flash_mla_grad_metadata` / `lightning_indexer_metadata` / `sparse_lightning_indexer_kl_loss_grad_metadata`），随 entry 的 `aicpu_metadata` 字段下发，前向/重算/反向直接读字段（没有查表）。几何即该 config 自己的必填字段（无默认值），由该入口的五个参数一次写入，默认值即 V4.1-Flash 的发布形状；记录是内核 metadata 的唯一来源：拿着没经过端口构建的 metadata 前向会直接失败，不会在层里补一次算子调用（没有 inline 回退） | Ascend sparse flash MLA + CANN `*_metadata` |
| `lightning_indexer.asc`（仅 A5，须与上一行同开） | 替换 selector 节点 |

量化路径不是这两个 override 的包装，而是另一个 backend：开启 `--extension.quantization.enable-sparse-attention-quantization` 后，量化后的 sparse attention（`torchao_npu` 里 fused attention 的拷贝）由 converter 安装，它自己构建 entry 的四份 metadata（含量化 selector 的两条 schedule），因此 `sparse_attn.asc` 在该路径上不生效；`--extension.quantization.li-quantization mxfp4` 必须与该开关同时开启，否则非量化 attention 的 BF16 schedule 会被量化 kernel 读到（垃圾索引，kernel 可能崩），converter 会直接报错。非量化路径的 `lightning_indexer.fwd_sparse` / `.sparse_length` 当前留空，等带 candidate 的非量化算子落地后由 port 填。
| `common.swiglu_group.asc`（仅 A5） | `*.moe.routed_experts.inner_experts` 的 grouped SwiGLU（FQN 限定） | cann_ops_nn.swiglu_group |
| `common.swiglu_group.asc_shared_experts`（仅 A5） | `*.moe.shared_experts` 的 SwiGLU（FQN 限定，不替换视觉 MLP） | 同上 |
| `common.token_dispatcher.asc` | 公共 MoE 的 permute / re-routing / unpermute；reference 不启用 | Ascend token dispatcher |
| `mhc.asc_hc_post` | mHC post 变换 | Ascend mHC post |
| `mhc.asc_sinkhorn`（仅 A5） | `HcPre._split_sinkhorn`（新接口，`hc_eps` 字段） | npu_mhc_sinkhorn |

routed experts 已走公共 grouped GEMM（`_ClampGroupedExperts` → 仓内 NPU `_grouped_mm`），reference 与融合共用，不改变模型参数名称、FSDP/EP 或 checkpoint 格式。融合输出、输入/权重梯度与训练轨迹的误差需与迁移后同配置 reference 对照验收；仅运行成功不代表数值或性能验收完成。

### RoPE seam 说明

文本旋转由主干构造公共 `ComplexRoPE.Config`（带 split 前缀宽度）：reference 通过 `common.rope.workaround` 用展开实数表；A3 融合选 `asc_complex`，A5 融合选 `asc_partial`，三者精确匹配同一配置类，同一列表只能出现一个（override 框架对同节点双 claim 直接报错）。视觉塔的二维位置表使用 `HalfRotation` / `asc_half_rotation`，与文本 interleaved 布局不同，不能互换。inverse（attention o_rope 路径）经 `-sin` 支持。

CPU 测试覆盖两类 seam 的输出/梯度、partial 非连续输入、逐 batch 位置与配置隔离；CPU mock 不代表 NPU kernel 验证通过。

### Muon 优化器注意事项

`--optimizer.name Muon` 启用 DistMuon 时，`materialize()` 会整体替换 `param_groups` 为 `[DistMuon(pattern), AdamW(.*)]`，launcher 传入的 `--optimizer.param-groups.0.*` 参数静默失效。学习率等超参请使用顶层字段：`--optimizer.lr`、`--optimizer.weight-decay`、`--optimizer.muon-momentum` 等。indexer 参数在 LI 下有梯度，但仍留在 AdamW fallback 组，不因迁移改变参数策略。

### 验证范围

CPU 单测覆盖模型与融合配置选择、RoPE 输出/梯度等核心行为；不保留专用的 V4.1 两卡手动冒烟 suite。真实训练通过上述 A3/A5 入口执行。历史 A3 的 50 步对照见 PR 845；该证据不代表当前 HEAD 或 A5 新 CANN 已完成重验。

## LoRA

通过 [LoRA 脚本](deepseek_v4_1_flash_lora_4k_a3.sh) 复用 CPT 入口。默认配置为多模态 `deepseek_v4_1_flash_lora`，设置 `CONFIG=deepseek_v4_1_flash_text_lora` 可选择纯文本模型。AdamW/Muon、Engram 冻结、native DCP 恢复和 PEFT 导出/合并方式见 [V4.1 LoRA 指南](../../docs/feature_guides/deepseek_v4_1_lora.md)。
