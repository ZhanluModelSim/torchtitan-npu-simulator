# Qwen3-32B Dense 接入说明

本文是 dense Qwen3-32B 接入的专题说明：模型事实、attention 路线决策、已验证边界与明确未覆盖范围。通用流程见[新模型接入流程](model-integration.md)，当前能力总表见[能力矩阵](capability-matrix.md)。

## 1. 定位与入口

| preset | 几何 | 用途 |
| --- | --- | --- |
| `qwen3_32b_smoketest` | 完整 32B 几何的 4 层切片（其余维度不变） | 快速回归；`tests/test_qwen3_capture.py` |
| `qwen3_32b` | 完整 64 层 Qwen3-32B | 完整几何捕获 |

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0   # 先加载实际 CANN 环境
python -m torchtitan_simulator --config qwen3_32b_smoketest --training.seq-len 128 --build-only
python -m torchtitan_simulator --config qwen3_32b_smoketest --training.seq-len 128
python -m pytest -q tests/test_qwen3_capture.py
```

## 2. 模型事实与复用边界

模型依据 revision：HuggingFace `Qwen/Qwen3-32B` config（commit `9216db5`，2026-11-09 获取）、transformers v4.51.0 参考实现、上游 torchtitan 0.3.0。

几何（与 HF config 逐项核对一致）：64 层，dim 5120，GQA 64Q:8KV，head_dim 128，SwiGLU 25600，vocab 151936，RMSNorm eps 1e-6，RoPE θ=1e6（max_seq 40960），不 tie embedding，attention_bias=false，无 sliding window。

| 内容 | 归属 | 说明 |
| --- | --- | --- |
| 模型数学、几何、parallelize、sharding、HF checkpoint 适配 | 上游 `torchtitan.models.qwen3` | 本仓不复制几何 builder；`as_npu_config` 按字段重绑 Config，跟随上游字段演进 |
| `build_attention_masks` 协议 | `torchtitan_npu/models/qwen3/model.py` | runner 要求的 repo-local 方法：原生 `get_attention_masks` 产 VarlenMetadata，CP 显式拒绝 |
| 工厂签名与精度 | `torchtitan_npu/models/qwen3/config_registry.py` | `num_mtp_layers` 非零显式拒绝；FP32 原始参数 + BF16 计算（与 DSV4/Kimi 约定一致） |
| dense GQA 融合 attention | `torchtitan_npu/override/common/varlen_attention.py` | `AscVarlenAttention`：真实 NPU kernel（见下节） |
| Meta autograd 桥 | `torchtitan_simulator/compat/attention.py` | 复刻 [norm.py](../torchtitan_simulator/compat/norm.py) 模式：能力探针 → autograd.Function 桥 |

独立预期（验收参考，不以实现自证）：完整几何 32,762,118,144 参数（≈32.76B）；可训练参数 64×9+3=579（4 层切片为 39），全部 FP32 梯度。

## 3. Attention 路线：真实融合算子直捕

三条候选路线的探针证据（2026-11-09，容器 CANN 9.2.0-beta.2 + torch_npu 2.14.0.dev）：

| 候选 | 证据 | 结论 |
| --- | --- | --- |
| varlen 原生（`aten::_flash_attention_forward`） | 有 Meta kernel，**无 NPU kernel** | 与真实 NPU 执行不对齐，不采纳 |
| FlexAttention | 明确不支持 Meta（"only supported on CUDA, CPU, HPU, or MPS"） | 无法仿真，出局 |
| Kimi 式合成算子 | 可行但无必要 | 仅当无原生算子时才合成（Kimi KDA/SiTU 的情况） |
| **`npu_fusion_attention_v3` 直捕 + Meta 桥** | v3/`npu_fusion_attention_grad_v3` 均有 NPU+Meta kernel；Meta 上厂商公式不向输入传播梯度 | **采纳** |

采纳结果：

- **对齐原有实现**：与 qwen3.5 真实 NPU 训练路径（`override/qwen3_5/varlen_attention.py`）同一算子、同一 TND varlen 语义（`actual_seq_qlen` 为**不含前导 0** 的累加和，即 `cu_seq_q[1:]`）与 `sparse_mode=7` + 共享 causal mask 约定；区别仅是 dense 无 CP head 交换。
- **合理融合粒度**：捕获图中每层前向恰好 1 个 `npu.npu_fusion_attention_v3` 节点、反向 1 个 `npu.npu_fusion_attention_grad_v3` 节点；无 naive 分解、无 `_flash_attention_forward` 残留。GQA（64:8）由算子原生处理，q/k/v 先投影后按原生 VarlenAttention 契约转 BF16。
- **桥接最小化**：厂商 Meta 前向直接复用（零兼容代码）；仅反向经 autograd.Function 桥——前向调用真实算子（节点进图），backward 调用真实 `grad_v3`（含前向返回的 softmax 统计量），保存值全部为真实张量。`install_meta_fusion_attention_autograd` 先探测原生能力，未来 torch_npu 补齐 Meta autograd 后自动回退 `native`。
- rope 保留原生 CosSinRoPE 分解 + 通用 rotary 桥（qwen3.5 真实配方同样未启用 rope 融合 override）；若后续确认真实训练使用 `npu_rotary_mul`，再按 override 机制对齐。

## 4. 已验证边界（2026-11-09）

| 关卡 | 内容 | 结果与证据 |
| --- | --- | --- |
| A 配置与构建 | build-only：13 处 RMSNorm override（含每层 qk_norm）+ 4 处 AscVarlenAttention override 全命中；非法 `num_mtp_layers`、CP、load balancer 显式拒绝 | 通过（容器实跑） |
| C 单任务 F/B/O（smoke） | `test_qwen3_capture.py`：39/39 梯度精确 FQN 集合且全 FP32；4+4 v3/grad_v3；17+17 `npu_rms_norm`/`_backward`；fused AdamW 更新数与 shape 契约；`memory_plan`/`memory_summary`/`fsdp_communication_view` 齐备；全模板无环、前驱闭合 | 1 passed |
| C 单任务 F/B/O（完整几何） | 手工运行 `qwen3_32b`（64 层）：F 5242 / B 6676 / OPT 1171 节点全无环；579/579 梯度 FP32；64+64 v3/grad_v3；257+257 rms_norm；内存计划齐备 | 通过 |
| 共享层回归 | `test_routing.py`+`test_config.py`（42 passed）；`test_kimi_capture.py`（13 passed）；`test_dsv4_capture.py`（15 passed）——dense 无 router 的 `install_balanced_dispatch` 早退不改变 MoE 模型行为 | 全部通过 |

## 5. 明确不支持与未覆盖

**明确拒绝**（有显式报错）：`num_mtp_layers>0`（无 MTP）；`cp_mesh`（`build_attention_masks` 显式 NotImplementedError；上游 varlen 后端对 CP 同样拒绝）；`load_balancer_type` 非 None；`AscVarlenAttention` 的非 causal window_size；EP（dense 无路由，`install_balanced_dispatch` 返回 `not-applicable-dense`）。

**未验证/后续工作**（不因本文声称已支持）：

1. **并行组合（关卡 D）**：当前回归仅 world=1（FSDP mesh size=1 路径）；world=2/4 的 FSDP/HSDP、TP/SP 组合、首末 rank 与非均匀 shard 尚未补测试。
2. **内存与卸载（关卡 E）**：激活/优化器卸载对 dense 的专项验证未做（开关本身为公共机制）。
3. **数值验证（关卡 B 的 attention 部分）**：`npu_fusion_attention_v3` 无 CPU kernel；CPU 参考需经 flex 前向或手写 GQA+QK-Norm 对照。Meta 捕获不证明真实 NPU 数值、HCCL 行为或 kernel 性能。
4. **量化（关卡 F）**：MXFP8 未接入验证（全 Linear 理论可复用，按流程后置）。
5. **真实 NPU 数值/性能**：本仓不证明；`AscVarlenAttention` 的真机行为需真机验证后回填。
6. **acceptance.py**：仍只覆盖 DSV4 四结构，Qwen3 需要各自验收脚本。

## 6. 跨仓提示

交付图中的 `npu.npu_fusion_attention_v3`/`_grad_v3`、`npu.npu_rms_norm`/`_backward` 是算子仓需要建模的真实 NPU kernel 边界；`aten._to_copy`（BF16 cast）与 `aten.slice`（`cu_seq_q[1:]`）为伴随的真实数据流。捕获的静态显存含全部 storage 事实，动态峰值与耗时归平台 DES。
