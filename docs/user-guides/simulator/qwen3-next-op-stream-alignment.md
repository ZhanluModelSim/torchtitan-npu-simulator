# Qwen3-Next 仿真算子口径对齐说明

与 NPU 实测负载对比后发现的算子流偏差及修正记录。参考准绳：MindSpeed-LLM fsdp2 版
`mindspeed_llm/fsdp2/models/qwen3_next/`（commit `6082e0de`）。修正范围 commit
`4f1dee9` → `d8b9536`。

## 修正清单

| Commit | 修正 | 对照实测问题 |
|---|---|---|
| `4f1dee9` | GatedAttention 抽 `_attention_core` 缝，仿真态记录 `fusion_attention` 融合虚拟算子 | flash attention 未融合 |
| `fa1ed3d` | MoE 严格 mirror MindSpeed `Qwen3NextMoeExperts`（见下） | 缺 index_copy、gather/index_add 偏少、scatter 漏捕获 |
| `1f4cb6c` | `RMSNormGated` 绑定融合 `npu_rms_norm` 边界（`run_meta_rms_norm`） | 搬运/copy 类算子差距大 |
| `eefe481` | router topk 归一化去掉 `+1e-20`（参考无 epsilon） | 严格逐算子对齐 |
| `d8b9536` | GDN 核心解除合成算子屏蔽，eager 分解开放捕获（实测走 MindSpeed 文件内 `torch_chunk_gated_delta_rule` torch eager 路径） | elementwise 个数明显少于实测 |

### MoE 生产算子流（`fa1ed3d`）

- 专家权重融合：`gate_up_proj` `[E*h, 2i]` + `down_proj` `[E*i, h]`，每层 GMM 3→2 次
  （EP 分组下扁平权重 Shard(0) 切分后，本地专家数按 `gate_up.shape[0] // hidden_size` 推导）
- token 分发改为生产调用序：`npu_moe_token_permute` → GMM → `npu_swiglu` → GMM →
  `npu_moe_token_unpermute`，前反向 + grad 全套虚拟算子（`qwen3_next_converter.py`）
- router：去掉 mm 前 float 强转，`F.softmax(dtype=torch.float32)`，topk 归一化无 epsilon
- `op_mapping.py` 补 index/scatter 族（index/gather/index_add_/index_put/scatter*/
  index_copy/sort/histc/cumsum/topk）——此前 eager scatter 算子已捕获但被归一化成
  `unknown`，呈现为"漏捕获"

### GDN eager 分解（`d8b9536`）

实测参考跑的是 MindSpeed 建模文件内的 pure-torch fallback（逐 chunk for 循环 + in-place
写）。此前该核心被记成 1 个合成算子（`fla.ops.gated_delta_rule.chunk_gated_delta_rule`），
内部全部 suspend，导致 elementwise/copy 计数缩水、flops=0 定价缺失。现已删除 shape-only
shim，逐行移植的分解实现开放跑在 meta 张量上（前反向 autograd 对逐 chunk 循环安全，已
回归验证）。

## 效果（80B a3b，world-size 8，rank_0）

| 指标 | 对齐前 | 对齐后 |
|---|---|---|
| 总节点数 | ~25,000 | **321,879** |
| elementwise | ≈0（被合成算子隐藏） | **155,072**（GDN 内 152,844：mul 54K / add 53K / bmm 46K / exp 14K） |
| copy 类 | 4,084 | 46,708（GDN chunk 循环 in-place 写 / contiguous / float cast） |

MoE 区（逐算子 = MindSpeed 生产口径）：`moe_token_permute` 94+grad47、`npu_swiglu`
94+47、`moe_token_unpermute` 94+47、`grouped_mm` 376、`fusion_attention` 24；手写
index/index_add/gather/sort 全部消失。

## loss / softmax 对齐状态

- loss：两侧均为 `F.cross_entropy` 分解（`_log_softmax`×1 + `nll_loss`×1 + 反向各 1 +
  logits float cast），逐算子一致；唯一差异是 reduction 属性（`sum` vs HF 默认 `mean`），
  不影响算子个数/类型
- router softmax：`F.softmax(dim=1, dtype=float32)` / topk / 归一化 / `.to(dtype)` 与
  MindSpeed 逐行一致（`eefe481` 后）
- attention softmax：融合在 `fusion_attention` 内，与实测 `npu_fusion_attention` 同口径

## 给成本模型的口径建议

1. `view` / `_unsafe_view` 是零搬运元数据操作，不应计入搬运类（全局 1151 次）
2. `_to_copy`（dtype cast）与 `clone`/`copy_` 应分开，按 tensor 字节数走带宽模型，
   而非统一 "copy" 单价
3. 融合算子（`moe_token_permute`/`moe_token_unpermute`/`npu_swiglu`/`fusion_attention`/
   `npu_rms_norm`）必须有条目，否则对应区块贡献零耗时
4. GDN eager 分解算子按常规 eager 算子计价即可（shape 元数据齐全）

## 验证

smoketest EP=1/2/8、`kimi_k3_smoketest` 回归、80B 全量捕获均通过；与实测的对表方法：
按 `module_path` 分解 `kernel_summary/rank_0.csv` 算子计数，逐模块比对。
