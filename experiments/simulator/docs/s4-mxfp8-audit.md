# S4：原生 MXFP8 接入与保存值生命周期

2026-10-21。使用 S3 相同的 TorchTitan 0.3.0 / NPU master 固定快照与完整官方 CANN 镜像。
本轮接入原生 TorchAO-NPU `all_mxfp8` recipe；删除迁移计划中的 simulator FQN CLI 复写需求，
不移植旧 `mxfp8_fqns` 字段、旧 converter singleton 或额外默认 FQN 名单。

## 启用与原生选择范围

从 sidecar 目录，在已有训练依赖与独立 `torchao_npu` 包的环境运行：

```bash
python -m torchtitan_simulator \
  --extension.quantization.enable-quantized-training \
  --extension.quantization.recipe all_mxfp8 \
  --training.num-mtp-modules 1 \
  --simulation.world-size 8 \
  --parallelism.data-parallel-replicate-degree 2 \
  --parallelism.expert-parallel-degree 2 \
  --activation-checkpoint.mode selective \
  --simulation.selective-ac-save-ops quant-mm \
  --simulation.output-formats json mem \
  --simulation.output-dir ./s4-mxfp8
```

调用目标 `TrainerEx` 使用的 `apply_quantization_converter()`，在模型构建前转换配置树，随后应用原生 overrides、
构建模型、并行化并创建 optimizer。`MXTrainingWeightWrapperTensor` 与量化 Function 来自目标独立 TorchAO-NPU 包。
配置解析不接受 `--mxfp8-fqns` 或 `--simulation.mxfp8-fqns`；不通过另一组参数重建相同复写能力。

| 原生 recipe 范围 | 本轮行为 |
|---|---|
| attention 的 wq_a/wq_b/wkv/wo_a/wo_b | 原生 MX wrapper；BatchedLinear 的 wo_a 进入量化 BMM |
| indexer 的 wq_b | 按原生 config-tree suffix 选择 |
| routed experts 三个权重、shared experts 三个 Linear | 原生 MX grouped MM / MM |
| MTP 内相同 attention/MoE 子模块 | 同一原生 suffix 规则覆盖 |
| MTP e_proj/h_proj、compressor、indexer weights_proj、mHC、lm_head | 不属于当前原生 all_mxfp8 的默认选择，不恢复旧额外名单 |
| LI Q/K、KV norm 或 sparse-attention kernel 的额外量化 | 与投影权重量化分开；本轮未开放其独立开关，不据 all_mxfp8 名称声称已启用 |

debugmodel + MTP=1 实际有 56 个 MX wrapper 参数，143 个预期参数梯度。
导出的 `quantization_configuration` 记录 recipe、选择来源及实际包装后的参数 FQN。
这是已构建对象的清单，不是仅复述请求配置。

## 真实算子、保存值与独立先验

实际执行原生 `npu_dynamic_mx_quant_with_dual_axis`、`npu_dynamic_mx_quant`、`npu_grouped_dynamic_mx_quant`，
以及 `npu_quant_matmul` / `npu_grouped_matmul` 的前反向；dual-axis 量化已归类为 `quantize`。
不以 BF16 matmul 或同形状新 tensor 替代这些算子或其保存值。

| 先验 | 独立预期及直接检查 |
|---|---|
| MM：64×64、BF16 输入 | backward 保存 A_q2/A_s2/B_q1/B_s1；两份 FP8 code 各 4096 bytes，两份成对 E8M0 scale 各 128 bytes，合计 8448 bytes |
| BMM：batch=2、64×64 | 同四项各增加 batch 维，合计 16896 bytes |
| grouped MM：M=128、E=2、K=N=64 | A code 8192 + grouped scale 512 + B code 8192 + B scale 256 = 17152 bytes；按原生 Meta 的 scale 分配布局计量 |
| full/selective AC：64×64 MM 的前反向间隙 | 原始量化 code/scale 已释放；HP 输入、master weight、HP 输出各 8192 bytes，合计 24576 bytes |
| 反向重算 | 重算量化输出有新 storage/generation，活至实际 backward 消费；不能沿用原始前向的出生/释放时点 |

测试逐项对照原生量化输出的 storage key、真实保存槽、实际反向 consumer 与释放记录；MM/BMM/grouped MM 均包含
forward、dgrad、wgrad 三次对应低精度 matmul，输入与权重梯度的 shape/dtype 保持正确。
AC 的独立用例还检查原始量化值在间隙中不驻留，重算后的低精度值不早于消费者释放。

non-reentrant checkpoint 的内部保存 hook 会覆盖外层 hook，因此 `autograd_saved_tensors` 不等于所有内存持有者。
checkpoint 内部值按弱 storage ownership 与实际消费追踪；native SAC 缓存另有写入/消费记录。
`quant-mm` 缓存的是实际 matmul 输出（通常为 BF16），不能假设它自动缓存所有 FP8 code/scale。

## 环境缺口与作用域

当前官方 CANN 9.2 beta.2 的 `swiglu_group` schema 有三个位置参数，`clamp_limit` 为 keyword-only，
其注册 backward 却返回四项梯度，首次原生 MXFP8 运行因此失败，证据为 `s4-first-native.log`。
兼容层先做真实 Meta 梯度探针；仅在该 arity 缺口存在时，为 Meta 调用接上原生 `swiglu_group_backward`。
保存实际 x/weight/group_index，按原生逻辑重建必要的 y_origin；恢复输入和 routed-score 梯度。
正常退出后还原调用绑定，不改 CANN、TorchAO-NPU 或生产训练源码。未来原生探针通过时直接使用原生 autograd。

MXFP8 计算仍有高精度 master weight，code/scale 是按需分配的临时量或 autograd 保存值；不能直接把参数常驻字节除以 2/4。
FSDP full parameter、AG staging 继续使用 S3 的实际分配/释放事件。FP8 通信与 FSDP prequantization 未启用。

## 模型配置与验证范围

- 原生 debugmodel + MTP=1：none/full/selective × HSDP+EP、CP+EP、world16/rank15 的 HSDP+EP+TP+CP。
- 核验量化输出到保存槽的来源、释放下界、全部梯度、归约到 optimizer 依赖、跨模板 DAG、通信唯一归属和峰值 storage 唯一。
- Flash/Pro/Pro 20T 的 BF16/MXFP8 共六个旧 preset 名称可构建 Meta 模型；smoketest 保持原入口，并可用原生量化开关启用 MXFP8。
- Flash/Pro 使用新主线原生工厂。主线没有 Pro 20T 工厂，sidecar 用原生配置树构造器保留旧 96 层 / dim12288 /
  2048 experts / top23 几何，明确标记 `legacy_pro_20t_geometry`；不宣称旧新模型数值等价。
- 大模型配置仅验构建、尺寸与实际 wrapper 选择；完整规模 F/B/O、宿主开销及全下游格式消费仍属 S5。
- `model_overrides` 的类型化几何字段已迁移：调用原生 builder 重建完整配置树，并同步 loss 的词表大小及 optimizer profile；不修改孤立顶层字段。
- 原始值、量化 code、scale、参数 shard、梯度和 optimizer state 通过各自的实际 storage 计量。存储 dtype/offload 仍是独立投影。

无真实 NPU，本轮证明原生 Meta 图、身份、保存/重算及逻辑内存合同；不证明量化数值误差、收敛、硬件耗时或 allocator 峰值。
内存不含 allocator reserved/cache、碎片和未建模 workspace；单 microbatch 捕获保留 GBS/GA，不能冒充完整累积训练峰值。

## 类型化模型几何配置

几何复写与被删除的 MXFP8 FQN 复写是不同接口。`--model-overrides.*` 保留类型检查，并在量化转换前
重建 attention、compressor/indexer、MoE、mHC、embedding/lm_head 和 MTP 的原生配置。
四个 preset 的空复写与原生配置树结构对照一致；三种 AC 的小模型 CLI 用例验证到量化、loss、backward 和 optimizer。

| 设置 | 本轮接口与处理 |
|---|---|
| dim/n-layers/vocab-size/n-heads、head/LoRA 尺寸 | `--model-overrides.<name>`；全部相关子模块重新构建 |
| o-groups、hc-sinkhorn-iters | 分别映射到原生 n_groups、sinkhorn_iters |
| indexer、MoE 中间尺寸、窗口、压缩比及 norm/RoPE 参数 | 可用字段见 CLI help；压缩比支持原生 0/1/4/128 |
| MoE expert/shared/top-k/hash/route 参数 | `--model-overrides.moe-args.<name>`；保留 `no-route-norm` 布尔别名 |
| MTP 数、序列长度、batch | 使用已有 `--training.num-mtp-modules`、`--training.seq-len` 和 batch 参数，跟随原生接口 |
| 自定义原生工厂 | 可通过 `--config module.factory` 自行构造配置树；不叠加 flat 几何复写，避免覆盖工厂的模型特例 |
| 其余旧 flat 字段 | 未提供相同旧 CLI；未知旗标明确拒绝，不声称旧 Config 的每个字段均已兼容 |

`model_configuration` 导出工厂、preset、请求复写及实际 dim/vocab/layer/MTP 配置。
增大层数时必须提供足够的压缩比；heads/groups、rope/head、indexer 幂次、expert/top-k 及正数/有限值均显式检查。
Pro 在 MTP=0 且无非压缩主层时，plain RoPE theta 取原生 builder 默认，不能误用压缩层 theta。

安装镜像另验 Flash MXFP8 + dim128/vocab512/三层/八 heads/四 experts/MTP1/full AC，
实际有 45 个量化参数和 116 个预期梯度，2923 节点图无环，38996628-byte 峰值 storage 唯一。
证据为 `s4-image-geometry-validation.json`；首次并行冷启动的 C++ 编译被杀日志保留为
`s4-image-geometry-resource-failure.log`，已在同一镜像串行重跑通过。

## 最终验收证据

- `s4-validate-final.log`：410 项测试通过，独立包 0 type errors（14 项局部忽略）；Ruff check/format 通过。
- 常规 pre-commit、architecture 与 Linux OAT 通过，OAT 扫描 152 个 staged 文件，无无效文件类型或许可头。
  主机 Pyrefly hook 缺可执行文件，不能称全仓 pre-commit 全绿。
- 全仓在完整环境中，分别使用各自 checkout 的 TorchAO-NPU 源码根：基线与候选均 64 个相同既有错误，新增 0。
  S3 的 44 个错误来自未展开独立 TorchAO-NPU 的搜索根口径，不能将这两个数量直接解释为回归或修复。

固定 debugmodel、MTP=1、world8、HSDP replicate2 + EP2、S128、GBS32/GA4：

| AC | 逻辑峰值 MiB | forward MiB | backward MiB | 外层可见低精度保存槽 | native SAC 缓存槽 |
|---|---:|---:|---:|---:|---:|

| none | 590.622 | 539.325 | 590.622 | 224 | 0 |
| full | 527.842 | 476.545 | 527.842 | 44 | 0 |
| selective | 528.978 | 477.681 | 528.978 | 44 | 9 |
| quant-mm | 537.967 | 486.670 | 537.967 | 44 | 33 |

四次运行均有 143 个预期参数梯度、56 个实际 MX wrapper 参数；保存槽全部释放；28 个 FSDP full alloc/free 区间、
30 个 staging release 闭合。原始事件、保存槽与峰值明细见 `s4-final/`；保存槽数量与累计字节不等于峰值。

镜像 `torchtitan-simulator:titan030-s4`，ID 为
`sha256:0391b3d85402988dd63622842ccee334ec681fb6e4cdf5ea5808c667fce12b79`。
安装后的 81 个 Python 文件与工作区逐项 SHA256 相同。镜像入口不挂载开发代码，执行 world16/rank15、
HSDP+EP+TP+CP、MTP=1、S512、selective/quant-mm、GBS32/GA8：
56 个实际量化参数、143 个预期梯度、300 个量化 kernel 事件、361 个通信源事件；跨模板 DAG 无环，
287 个保存槽均释放，307346636-byte 峰值独立求和一致且底层 storage 唯一。
镜像还独立构建 Pro 20T/MXFP8/MTP=1；它属于构建检查，未据此声称完整规模执行。

证据包括 `s4-image-validation.json`、`s4-image-20t-build.log`、`s4-pyrefly-baseline-comparison.json`、
`s4-oat-linux.log`；完整入口、源码/测试哈希与逐项出口索引见 `s4-final-manifest.json`。
S3 镜像和证据保留为历史快照，S4 未修改其镜像 tag 或上游源码。
