# Qwen3.5 文本模型接入与验证

记录 Qwen3.5（上游 TorchTitan 0.3.0 自带实现）在 simulator 的文本捕获接入。按[模型接入工作流](model-integration.md)的路线执行：模型定义、配置工厂与 parallelize 归上游 TorchTitan 与主包 `torchtitan_npu`，本包只做捕获兼容；排错经验见[常见问题](model-integration-pitfalls.md)。

## 模型与 preset 来源（工作流 §1–§2）

- 模型参考 revision：上游 TorchTitan 0.3.0（容器内 site-packages 安装，`torchtitan.models.qwen3_5`）；VL 结构——GatedDeltaNet 线性注意力与 Qwen35Attention 全注意力混合栈，可选 vision tower。
- NPU 生产绑定：主包 `torchtitan_npu/models/qwen3_5/` 是薄包装（FLA 导入 shim、多模态 patch、NPU parallelizer 切换）；GDN 核心走 `torchtitan_npu.override.qwen3_5.gated_delta.npu`（Triton-Ascend kernel），卷积用纯 torch 参考实现。
- 复用路线：同结构直接复用上游 config builder 与 NPU parallelizer，未复制任何模型代码。preset `qwen3_5_debugmodel`、`qwen3_5_debugmodel_moe` 为上游 `qwen35_debugmodel(_moe)` 原生配方，经 `torchtitan_simulator.presets` 包装：拒绝 MTP（`num_mtp_layers` 非零显式报错）、parallelize_fn 换为文本捕获版。上游 model.py 顶层绑定 fla 符号，preset 内先 `ensure_qwen3_5_importable()` 再导入（导入顺序敏感，ruff 重排会破坏 shim 窗口）。
- 几何先验固化在 `tests/test_qwen3_5_capture.py`：文本捕获参数 132,412,336（embed/head 2×248320×256 + 6 GDN 层 660,040 + 2 全注意力层 656,000 + final norm），张量数 133；8 层（layer 3/7 全注意力），GDN key_dim=128、value_dim=256、conv kernel=4。

## 显式建模边界（工作流 §3–§4）

| 组件 | 生产绑定 | simulator 处理 |
| --- | --- | --- |
| GDN 核心 | `torchtitan_npu.ops.triton.gdn`（Triton-Ascend，Meta 不可执行） | `qwen3_5_sim.gated_delta_rule(_grad)` 虚拟融合，1 前向 + 1 反向算子；头扩展等 eager 算子保留在图中 |
| FlexAttention 核心 | NPU compile 路径的 flex lowering | `qwen3_5_sim.flex_attention(_grad)` 虚拟融合；原生路径在 Meta 上执行 mask_mod 会读到未初始化 Meta 数据（`aten.index` 越界），不能按原生 dispatch 捕获 |
| causal conv | override 的纯 torch 实现（F.pad/conv1d/silu） | 原生 dispatch，无桥接 |
| MoE dispatcher | `token_dispatcher.asc`（npu permute/unpermute） | 与 DSV4/Kimi 共用 `_COMMON_OVERRIDES` 与 permute 兼容层；语义声明复用主包包装算子表 |
| vision tower | VL 训练参与，text-only 配方 detach | preset 的 `_parallelize_text_capture` 按 `qwen35_27b_long_text_sft` 同款方式置 None；文本捕获图不含 vision 算子 |
| 输入路径 | runner 过渡契约 `build_attention_masks` | `compat/qwen3_5.py` 适配器：调上游 `Qwen35Model.get_attention_masks` 构造混合栈 per-consumer mask dict，注入 text-only `special_tokens`；CP 显式拒绝（NotImplementedError） |

两个虚拟融合按工作流 §4 完成注册顺序：`custom_op` schema → `register_fake` → `register_autograd(setup_context)` → `register_tensor_semantics`（GDN 为 B/S/H/D 边界布局，flex 为 B/H/S/D；GDN `reset` 为可选 B/S bool 掩码——固定长度路径由 cu_seqlens 物化成真实张量，直接调用可为 None，语义按 optional 声明）。真实设备调用显式 RuntimeError。语义表见[自定义语义](custom-tensor-semantics.md)。

支撑性修改：`metadata.py` 对 torch 2.14 非 dataclass 的 `BlockMask` 按字段名单独 lift 张量字段；`runner.py` 模型注记 `num_layers` 对无 `n_layers` 字段的 Config 取 `len(layers)`（Kimi 的 property 与列表长度一致，DSV4 走原路径）；`compat/routing.py` 稠密模型无 router 时返回 `ep_token_exchange: none_no_routers` 注记而非报错。

## 验收记录（工作流 §6，2026-11-03）

环境：容器 `cann:9.2.0-910b-ubuntu22.04-py3.12-devel`，torch 2.14.0+cpu，torch_npu 2.14.0.dev20260918，torchtitan 0.3.0，triton_ascend 3.2.1，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`。

| 关卡 | 结果 |
| --- | --- |
| A 配置与构建 | 参数公式与张量数断言通过；MTP 拒绝；override 到位（GatedDeltaKernel 全部为 TritonGatedDeltaKernel，EP=2 派生配置树 dispatcher 为 AscAllToAllTokenDispatcher.Config——该 MoE 的 dispatcher 在 routed experts 内部构造，不是具名子模块）；配置遍历 FQN 与实际 Linear 模块集合一致（含 attention/delta_net→attn 结构重命名） |
| C 单任务 F/B/O | `simulate()`：四个虚拟融合前反向成对出现；梯度清单 133 项；DAG／引用闭合；`persistent_param_bytes`／`optimizer_state_bytes`（2×FP32 动量 + 每参数 4B step）按独立公式断言 |
| D 并行与 AC | single / FSDP(2, 首末 rank) / HSDP(4, replicate=2) / full AC / selective AC；MoE EP=2+FSDP=2 CLI 冒烟；EP=1 但显式 EP 参数非法组合被 validate 拒绝 |
| 语义交付 | 捕获图四个虚拟融合节点携带 `tensor_semantics`（角色/轴/索引与声明一致，axes 长度等于 rank），aten/npu 节点不带语义；语义与源图一致 |
| build-only | `--build-only` 报告正常（capability `qwen3_5_gdn_meta_bridge`） |

平台链路：sim-platform `evaluate_training` 消费 qwen3_5 捕获（synthetic 后端）产出 F/B/O 三图与迭代时延，融合算子归一为 `Qwen35SimGatedDeltaRule`/`Qwen35SimFlexAttention`。

## 回归

`test_dsv4_capture/test_kimi_capture/test_config/test_runtime/test_routing/test_memory_delivery/test_step/test_warmup`：107 过；12 个失败全部为 mxfp8 用例的 `torchao_npu` 环境缺失（公开 wheel 0.1.0.dev0 缺 `DeepSeekV4QuantLightningIndexerConfig`），与本次改动无关。

## 未覆盖与交接

- CP（varlen/asc_cp 配方与 QwenCPMetadata）、量化（MXFP8，依赖 torchao_npu 配套版本）、vision tower 前向图、`fla_fused_recurrent` 后端、生产规模几何（27B–397B）、PP。
- Flex 虚拟融合只覆盖 out_transform=None 路径（Qwen3.5 捕获路径）；若上游启用 lse 输出需扩展。
- Roofline 预测覆盖由平台算子仓决定；qwen3.5 特有算子依赖平台预测后端扩展。
- 复用与差异：模型/工厂/parallelize 复用上游与主包；差异仅为本包 `compat/qwen3_5.py`（两个虚拟融合 + 输入适配）、preset 包装与支撑性小改。
