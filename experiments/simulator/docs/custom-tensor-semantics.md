# 自定义算子的张量语义

训练负载为本仓自定义融合与包装算子提供输入／输出角色和轴含义，使下游无需根据 shape 猜测 query、token、head 等语义。信息存放在 `OpNode.attrs["tensor_semantics"]`；原有 shape、dtype、stride、物理 layout、参数绑定及依赖保持不变。本轮只扩展训练仓输出，平台和算子仓尚未增加读取逻辑。

## 数据结构

以 `kimi_sim.fusion_attention.default` 为例：

```python
{
    "version": 1,
    "inputs": (
        {"index": 0, "role": "query", "axes": ("batch", "head", "q_len", "qk_head_dim")},
        {"index": 1, "role": "key", "axes": ("batch", "head", "kv_len", "qk_head_dim")},
        {"index": 2, "role": "value", "axes": ("batch", "head", "kv_len", "padded_value_head_dim")},
    ),
    "outputs": (
        {"index": 0, "role": "attention_output", "axes": ("batch", "head", "q_len", "padded_value_head_dim")},
    ),
}
```

- `index` 引用本节点 `inputs` 或 `outputs` 中的零基位置，不是原生参数列表位置；同一张量多次出现仍保留各自角色。
- `axes` 每一项严格对应 shape 的一个维度。字符串表示已知含义；`None` 表示此边界未定义该轴的更具体语义，不根据 rank 或大小推测。
- 轴中的元组表示多个逻辑轴展平到一个 shape 维度，例如 KDA `dt_bias` 的 `axes=(("head", "key_head_dim"),)`。不额外推测各分量大小。
- 语义轴不代表物理存储布局、连续性或 stride，也不提供模型成本。这里的维度值读取节点实际 shape，沿用现有本地张量范围。
- 省略的可选输入不占 `inputs` 位置；真实空张量仍有角色。JSON 序列化时元组变为数组，`None` 变为 `null`。

输入角色通过原生 `input_tensor_bindings` 的 occurrence 索引定位，允许关键字重排及同一张量承担多个角色。声明按完整原始 overload 精确匹配，不使用下游简化名称。捕获时校验输入／输出覆盖、索引与声明轴数；声明失配明确报错，不能静默写入错误语义。

## 已覆盖边界

| 自定义原始名称 | 输入与轴的依据 |
| --- | --- |
| `kimi_sim.chunk_kda.default`、`chunk_kda_grad.default` | `KimiDeltaAttention.forward` 明确 reshape 为 B/S/H/D；beta 为 B/S/H，A_log 为 H，dt_bias 为 H×D 的展平维 |
| `kimi_sim.fusion_attention.default`、`fusion_attention_grad.default` | `KimiGatedMLA.forward` 已转置为 B/H/S/D；value 已 padding，不能把轴大小当作未填充的 V head dimension |
| `kimi_sim.situ_glu.default`、`situ_glu_backward.default` | 仅末轴 `feature` 在边界明确；前轴保持 `None`，兼容 dense、共享专家和 routed expert 输入 |
| `torchtitan_npu.npu_moe_re_routing.default` | rank-major rows 转为 expert-major rows；计数矩阵为 EP rank × local expert；恢复索引为逆置换，缺 scales 时第二个输出仍为真实空张量 |
| `torchtitan_npu.npu_moe_token_permute.default`、`npu_moe_token_permute_backward.default` | tokens 为 token × feature；indices 的公开包装契约支持 token 向量或 token × topk；输出为 routed rows |
| `torchtitan_npu.npu_moe_token_unpermute.default`、`npu_moe_token_unpermute_grad_unweighted.default` | probs 存在时按 token/topk 合并；probs 为 None 时只恢复 routed rows，空 probs 仍属于前者；feature 也可为标量尺度展开后的大小 1 |

Kimi 反向节点同时声明保存输入角色、`output_gradient` 和输出梯度角色。原生 ATen、CANN、HCCL 算子不因注册本功能而新增语义轴；原生参数名绑定照常保留。控制节点及 offload payload 使用已有专属事实，不给它们猜测模型轴。

本表覆盖当前 simulator 路径的六个 Kimi 虚拟融合与五个主包 MoE 包装 overload，不宣称其他未接入的 Triton、TileLang 或 DeepEP 自定义路径均已覆盖。

## 维护位置与验证

Kimi 声明与定义同在 [`compat/kimi.py`](../torchtitan_simulator/compat/kimi.py)。主包包装算子的声明在 [`compat/moe_tensor_semantics.py`](../torchtitan_simulator/compat/moe_tensor_semantics.py)，依据主包 `torchtitan_npu/ops/ascendc/moe_*.py`；通过 Meta runtime 加载，主包无需依赖 simulator。注册是模块级静态声明，重复串行 runtime 不重复执行注册。

[`capture/tensor_semantics.py`](../torchtitan_simulator/capture/tensor_semantics.py) 负责精确匹配和绑定。新增自定义边界时，应在定义处或对应兼容模块声明真实角色与轴，不能从 shape 经验分类。只有参数存在性或接口明确支持的 rank 变体可选择声明；其他未知轴保留 `None`。手工 `record_synthetic_op` 若注册此声明，调用者必须在 attrs 中显式提供准确的 `input_tensor_bindings`；当前没有需要补充的手工计算调用。

回归入口为 `tests/capture/test_tensor_semantics.py` 、`tests/test_kimi_capture.py` 和 `tests/test_custom_tensor_semantics_delivery.py`。测试覆盖 Meta 前后向、关键字顺序、重复张量、空张量／可选输入、两种 token indices 结构、声明失配，以及真实 Kimi 配置捕获。验证只针对捕获事实，不证明真实设备数值或性能。

2026-11-03 工作区验证：`ttsim` 容器加载 `/workspace/SimWorkspace/torchtitan-npu-simulator` 的主包和 sidecar editable 源码，使用 `/tmp/simplatform-field-audit-20261030/bin/python`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 和已加载的 CANN 环境。局部语义测试及既有 Kimi 捕获共 28 项通过；真实交付测试的 DSV4（S=128、EP=2）、Kimi（S=16、EP=2）及 Kimi（S=16、TP=2、EP=2）三项通过。DSV4 的 S=16 因原生压缩块最小长度 128 被拒绝，验收配置已改为满足该限制。变更 Python 文件 Ruff 通过，sidecar pyrefly 为 0 errors；未执行完整仓库 Lint 或真实 NPU 数值验证。
