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

K3.1 的模拟训练在长序列把 `W_UK` 吸收到 Q，在注意力后用 `W_UV` 恢复每头的 V 宽度，
中间传一份共享 latent KV 和独立位置通道。Meta 抓图用 `npu_sparse_flash_attention` /
`npu_sparse_flash_attention_grad` 表达前反向，保留实际 max、sum 和输出的保存关系；
投影、归一化、输出门控和共享 KV 梯度加和位于融合边界外。学习型索引器仍使用
`kimi_sim.dsa_indexer.default`，语言模型损失不会对硬 top-k 与索引器参数回传。

`seq_len <= topk` 默认跳过索引器，使用稠密因果 `npu_fusion_attention_v3` 和对应反向。
要显式比较短序列稀疏路径，可在模拟器 CLI 加 `--simulation.kimi-k31-short-sparse`，
或在 sim-platform TOML 的 `[training]` 中写 `kimi_k31_short_sparse = true`。
该选项构造全覆盖因果索引，用 `-1` 填充到原生 ABI 的索引宽度。
Scale 从原始 QK head dim 算，Cube/Vec 的有效配对仍按因果三角累计，物理索引读按填充形状计。
Meta 兼容也支持 debug 小模型形状，其定位是抓图和公式回归。

## 训练、状态与验证边界

- 当前任务为单文档文本训练捕获；不接入视觉、推理 KV cache、文档 packing 或索引器蒸馏 loss。
- 索引器遵循 NeMo 的冻结训练策略：其参数保留在模型与静态参数内存中，但不进入语言模型梯度及 AdamW 状态。硬 top-k 不向索引器传梯度；这不代表索引器蒸馏训练没有反向。
- checkpoint adapter 保留五类索引器状态：`wq_b.weight`、`wk.weight`、`weights_proj.weight`、`k_norm.weight`、`k_norm.bias`。映射针对普通权重；不增加 MXFP4 checkpoint 解包能力。
- 索引器在 TP 上复制，MLA 主头按原有策略分片；CP 复用原有全上下文 gather／本地 slice。因而图反映该参考实现的重复计算和通信，不宣称为最优并行方案。
- 小几何已覆盖单卡、FSDP、TP＋DP、CP、EP 及 none／full／selective AC。`world_size=2, TP=2` 会在公共通信 rank 解析中递归，原有 K3 也可复现；当前 TP 验证采用 `world_size=4, TP=2`，不将前者写成可用组合。
- 原始参数 FP32；计算参数可用原有 BF16 mixed precision。当前拒绝 K3.1 量化训练；MTP、compile 和 PP 的既有拒绝边界保持生效。
- 普通训练是显式 PyTorch 稀疏参考实现；Meta 捕获采用上述融合契约，不模拟未知内部 scratch，不能将其图耗时或静态内存当作融合 NPU kernel 的性能实测。真实 KDA provider 仍需显式提供。

CPU 数学、梯度与状态映射验证见 [test_kimi_dsa.py](../../../tests/unit_tests/models/kimi_k3_1/test_kimi_dsa.py)；真实入口的 Meta 捕获、重计算、冻结内存和并行验证见 [test_kimi_k3_1_capture.py](../tests/test_kimi_k3_1_capture.py)。这两类验证均不证明真实 NPU 训练数值或性能。

早期虚拟 DSA 路径以 wheel 普通安装验证了小模型单卡、FSDP、TP、CP、EP 及 none/full/selective AC，并回归原 K3 捕获和 CPU 稀疏数学测试。该路径的完整 `kimi_k3_1_full` 另以 S=4224、64 逻辑卡、TP=CP=PP=1、DP replicate=4、DP shard=16、EP=16、selective AC 且保存 gmm 验证：三个模板共 99,111 个节点，索引器 48 个、稀疏 attention 前向 48 个、反向 24 个；2642 个参数有梯度，indexer 无梯度。全部 120 个 DSA 节点经平台适配后命中索引器／FlashAttention 正反向模型并返回正耗时。用于联通验证的硬件为 synthetic；本验证不证明真实 NPU 数值、内部 workspace 或耗时精度。
