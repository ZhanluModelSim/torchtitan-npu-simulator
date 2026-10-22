# Qwen3-Next 黄区（湛卢）交接文档

> 读者：黄区 AI 工具 / 工程师。蓝区（torchtitan-npu-simulator）建模已验收，
> 本文档交接湛卢接入所需的全部上下文。上游规则见
> `torchtitan_training_modeling_guide.md` §8/§9/§10。

## 1. 蓝区已交付（git 分支 qwen3.5，commit `4da194a` 起）

| 交付物 | 位置 |
| --- | --- |
| 模型包（model/attention/gdn/feed_forward/loss/parallelize） | `torchtitan_npu/models/qwen3_next/` |
| 模型契约（参数量公式/支持矩阵/已知偏差/验证记录） | 同目录 `MODEL_CONTRACT.md` |
| **算子账本（黄区比对基准）** | 同目录 `OP_REFERENCE.md` |
| 融合算子 shim（GDN triton 12 内核 + EP dispatcher 流） | `torchtitan_npu/simulator/hardware_shims/{gdn_shim.py,qwen3_next_converter.py}` |
| 对齐全程记录（6 轮实测反馈的"现象→修改→根因"） | `docs/user-guides/simulator/qwen3-next-op-stream-alignment.md` |
| 实测标准树 | `/home/l30079348/MindSpeed-LLM`（commit `6082e0de`，mindspeed_llm/fsdp2/） |
| 模拟器验证 | 8 配置矩阵 + AC 双覆盖全过，单测 29 项；证据路径见 MODEL_CONTRACT §7 |

## 2. 黄区工作步骤（guide §8）

1. **代码过墙**：上游仓以 `third_party/torchtitan-npu-simulator` 子模块/目录
   形式导入黄区环境。
2. **湛卢注册**：参考 `Zhanlu-Model-Platform` 的 `docs/torchtitan.md` 运行逻辑，
   按 **dsv4 的接入方式**——通常只有三个注册文件：
   - 模型注册（model 包路径）
   - 配置注册（复用模拟器 preset 名：`qwen3_next_smoketest`、
     `qwen3_next_80b_a3b_2layer`、`qwen3_next_2l_*` 变体）
   - `bin/example_args/torchtitan_batch.example.json`（单配置最小 example）
3. **未命中闭环**：
   ```bash
   ZHANLU_DISABLE_PRUNE_OP=1 <湛卢运行命令>
   ```
   把所有未命中建模的算子打印出来 → 逐条对照本模型 `OP_REFERENCE.md`：
   - **改名复用**：湛卢已有同构算子 → 用已建模真名（优先）；
   - **补建模**：确属新算子 → 用约定生产 kernel 名，**主动同步湛卢侧登记**；
   - **确认登记**：蓝区契约声明的偏差（见 §3）不算缺失。
   重跑直至未命中清零、无明知不合理项。
4. **沉淀接入说明**：黄区步骤跑通后写该模型的湛卢接入说明。

## 3. 已知事项（判定"缺失"前先看）

| 事项 | 状态 | 处置 |
| --- | --- | --- |
| `divinplace` / `catoutout` / 显式 `zero_` | 钉住的 core 参考版不存在；管理方要求严格按服务器版 | 需服务器 `mindspeed/ops/triton` 源码或实测算子行（raw_op_type+module_path+shape）落位后再建模 |
| TP 下 lm_head 复制（入口 1 次 AG） | 蓝区契约 §5.1 声明的 v1 决策 | MindSpeed 实测 tp=1 无 vocab-parallel 证据；不算缺失 |
| debug load-balance 分支 | 契约 §5.3 | 仅测试口径；权重走真实反向图 |
| EP 显存记账（persistent 不随 EP 再减） | 公共模拟器观察项（契约 §5.2） | 非模型特例，黄区不处理 |
| view 类算子折叠（aclnnSplitV/TransData/TensorMove 不建模） | 蓝区已结论：正确设计（映射表见 alignment.md"已结论"节） | `aten.clone/copy_` 已承担对应搬运；SplitV 有意不建模，不算缺失 |
| swiglu 两路径 | 已核实入流（EP eager / 非 EP npu_swiglu 融合对） | 不算缺失 |

## 4. 算子真名速查（最高频核对项）

模型特有算子全部使用已建模真名，无自造名：

- GDN triton 序列：`mindspeed.ops.triton.{l2norm_fwd,l2norm_bwd,chunk_local_cumsum,chunk_scaled_dot_kkt_fwd,solve_tril,recompute_w_u_fwd,chunk_gated_delta_rule_fwd_h,chunk_fwd_o,chunk_bwd_dv_local,chunk_gated_delta_rule_bwd_dhu,chunk_bwd_dqkwg,prepare_wy_repr_bwd}`
- 注意力：`fusion_attention` / `fusion_attention_grad`
- 通用 NPU：`npu.npu_rms_norm(.backward)`、`npu.npu_rotary_mul(.backward)`、`npu.npu_grouped_matmul`
- 量化：`npu.npu_dynamic_mx_quant`、`npu.npu_quant_matmul`
- 逐次实测数据见 `OP_REFERENCE.md` §1-§5（含 2 层 ep=2 口径的精确计数）

## 5. 运行参考（蓝区侧口径，黄区命令按湛卢文档替换）

```bash
# 模拟器（meta，蓝区验证口径）
NGPU=2 python3 scripts/run_simulator_spawn.py --config qwen3_next_80b_a3b_2layer \
  --simulation.output-formats text,csv
# 测试口径：ep=2 实测对齐（管理方指令）；AC 对照 --activation-checkpoint.mode full
```
