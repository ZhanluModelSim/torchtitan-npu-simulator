# Kimi K3.1 模型与捕获

Kimi K3.1 Pro 复用现有 K3 decoder、KDA、SiTU、LatentMoE 和 Attention Residual，仅为原有的 24 个 Gated MLA 层增加独立 DSA 索引器。模型独立放在 `torchtitan_npu/models/kimi_k3_1/`，使用 `kimi_k3_1` family；公共组件单向复用 `kimi_k3/`，K3 不导入 K3.1。

## 依据与几何

模型配置固定为 [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json)，revision 为 `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`。该 checkpoint 没有发布 modeling 文件；索引公式参考 [NeMo KimiDSAIndexer](https://github.com/NVIDIA-NeMo/Automodel/blob/c581829ae0709eabafa333bd0c5f33486ae439e9/nemo_automodel/components/models/kimi_k3/model.py)，训练边界见其 [K3.1 说明](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/moonshotai/Kimi-K3.1-Pro)。本实现不等于复制 NeMo 的 cuDNN／FlashMLA 后端。

完整几何保留 K3 的 93 层、hidden 7168、96 个注意力头、896 个专家与 top-16 路由。69 层为 KDA；从 1 开始计数的第 4、8、…、92、93 层为 DSA。索引器有 32 个头，每头 128 维，`topk=2048`，无索引 RoPE，key 使用带 bias 的 LayerNorm。

## 入口

| preset | 几何与用途 |
| --- | --- |
| `kimi_k3_1_smoketest` | 现有 K3 的小几何，4 层、3 KDA＋1 DSA；索引器缩为 2 头、每头 16 维、top-k 8，使短序列也覆盖真实选取路径 |
| `kimi_k3_1_16layer_reduced` | 复用 K3 的 16 层 reduced 几何；索引器保持 32 头、128 维、top-k 2048 |
| `kimi_k3_1_full` | 完整 K3.1 文本几何；索引器 top-k 2048 |

在已经安装配套主包、sidecar 和训练栈并加载 CANN 的环境运行：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m torchtitan_simulator --config kimi_k3_1_smoketest --training.seq-len 16
```

主包对应工厂为 `torchtitan_npu.models.kimi_k3_1.config_registry.kimi_k3_1_debugmodel`、`kimi_k3_1_16layer_reduced` 和 `kimi_k3_1_full`。每次调用创建独立配置树。

## DSA 执行与交付

`kimi_k3/attention.py` 提供共享投影与 attention 扩展点；`kimi_k3_1/attention.py` 实现 DSA 差异，保留原有 Q/K/V、输出门控和投影。`dsa.py` 新增两步计算：

1. 从归一化 Q latent 投影索引 Q，从 hidden 投影共享索引 K 与各头权重。打分为 `sum_head(ReLU(Q·K) * weight)`；FP32 累加并施加因果 mask，选出每行最多 top-k 个 key。可见 key 不足 top-k 的槽位为 `-1`。
2. 按索引读取 K/V，只对这些位置计算 QK、FP32 softmax 与 AV。查询分块限制单次 gather 的大小，但 autograd 仍保存各块所需的 K/V，不能据此认为总训练保存量有界；原生 autograd 保留真实 K/V 保存值和散加梯度。不会先算完整 attention 再用 mask 假装稀疏。

索引打分仍遍历候选 key，不能将整层 DSA 视为线性复杂度。序列长度不超过 top-k 时所有因果 key 都可见，直接复用 K3 的 dense attention 边界并跳过索引器计算。

Meta runtime 自动安装 K3.1 的三个融合捕获边界：

- `fused.dsa_indexer.default`：输入投影后的 Q、归一化 K 和已缩放的 FP32 weights，输出 int64 选取索引。索引器冻结，没有反向节点。
- `fused.dsa_attention.default`：输入展开的 Q/K/V 和索引，输出 attention 结果和 FP32 logsumexp。Q/K 为 192 维、V 为 128 维时，输出仍为 128 维。
- `fused.dsa_attention_grad.default`：读取前向实际保存的 Q/K/V、索引、输出、logsumexp 以及输出梯度，产生 Q/K/V 三项梯度。

投影、LayerNorm、weights 缩放和输出门控继续保留在融合边界外。三个节点都有明确的 `tensor_semantics`；不向原始节点填入 FLOPs 或耗时。算子仓为这些训练入口解析逻辑语义，复用现有索引器和 FlashAttention 正反向公式；它们不是吸收投影后的推理 DSA 的原始名别名。

融合不再暴露 eager 分块实现内部的 gather、score 和 softmax 临时张量；静态内存按该融合保存契约记录，而非声称保持 eager 临时显存不变。模型对索引打分采用全候选对代理，对 attention 使用有效因果 top-k 对数，并假设跳过早期 `-1` 槽位。这些是融合性能建模边界，不等于原 ATen 分块执行量或真实 NPU 内核测量。

普通设备模型仍执行上述 PyTorch 稀疏参考实现。融合定义只用于显式 Meta 会话，在正常或异常退出时恢复方法；不会冒充原生 CANN kernel。KDA、SiTU 和短序列 dense attention 继续采用已有 Meta 兼容。

## 训练、状态与验证边界

- 当前任务为单文档文本训练捕获；不接入视觉、推理 KV cache、文档 packing 或索引器蒸馏 loss。
- 索引器遵循 NeMo 的冻结训练策略：其参数保留在模型与静态参数内存中，但不进入语言模型梯度及 AdamW 状态。硬 top-k 不向索引器传梯度；这不代表索引器蒸馏训练没有反向。
- checkpoint adapter 保留五类索引器状态：`wq_b.weight`、`wk.weight`、`weights_proj.weight`、`k_norm.weight`、`k_norm.bias`。映射针对普通权重；不增加 MXFP4 checkpoint 解包能力。
- 索引器在 TP 上复制，MLA 主头按原有策略分片；CP 复用原有全上下文 gather／本地 slice。因而图反映该参考实现的重复计算和通信，不宣称为最优并行方案。
- 小几何已覆盖单卡、FSDP、TP＋DP、CP、EP 及 none／full／selective AC。`world_size=2, TP=2` 会在公共通信 rank 解析中递归，原有 K3 也可复现；当前 TP 验证采用 `world_size=4, TP=2`，不将前者写成可用组合。
- 原始参数 FP32；计算参数可用原有 BF16 mixed precision。当前拒绝 K3.1 量化训练；MTP、compile 和 PP 的既有拒绝边界保持生效。
- 普通训练是显式 PyTorch 稀疏参考实现；Meta 捕获采用上述融合契约，不模拟未知内部 scratch，不能将其图耗时或静态内存当作融合 NPU kernel 的性能实测。真实 KDA provider 仍需显式提供。

CPU 数学、梯度与状态映射验证见 [test_kimi_dsa.py](../../../tests/unit_tests/models/kimi_k3_1/test_kimi_dsa.py)；真实入口的 Meta 捕获、重计算、冻结内存和并行验证见 [test_kimi_k3_1_capture.py](../tests/test_kimi_k3_1_capture.py)。这两类验证均不证明真实 NPU 训练数值或性能。

融合捕获验证使用 wheel 普通安装，覆盖小模型单卡、FSDP、TP、CP、EP 及 none/full/selective AC，并回归原 K3 捕获和 CPU 稀疏数学测试。完整 `kimi_k3_1_full` 另以 S=4224、64 逻辑卡、TP=CP=PP=1、DP replicate=4、DP shard=16、EP=16、selective AC 且保存 gmm 验证：三个模板共 99,111 个节点，索引器 48 个、稀疏 attention 前向 48 个、反向 24 个；2642 个参数有梯度，indexer 无梯度。全部 120 个 DSA 节点经平台适配后命中索引器／FlashAttention 正反向模型并返回正耗时。用于联通验证的硬件为 synthetic；本验证不证明真实 NPU 数值、内部 workspace 或耗时精度。
