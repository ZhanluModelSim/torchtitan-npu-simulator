# glm5_next（GLM-5.3-Flash / unified_mm）Meta 模拟器验收证据包

依据：`docs/test_guides/model_acceptance.md` 第 6 节模板。
结论：**Conditionally ready**（声明范围见第 2 节与 MODEL_CONTRACT.md §10；所有未支持项 fail fast，无静默降级）。

## 1. 环境与基线（所有配置共用）

| 字段 | 内容 |
| --- | --- |
| Commit/base | 被测 commit `8a06a8e79436bab00c952d69a37e2d739c82cae7`（工作区含 glm5_next 新增文件的未提交状态）；上游 torchtitan 基线 `ac13e536c84e7f6647b14fa9375c3c8a8a2b8578`（requirements.txt 固定） |
| Environment | 容器 `titan-sim`，镜像 `torchtitan-npu-simulator:v1.0`（id `2e54c5764927…`）；Python 3.12.13；torch 2.12.0+cpu；torch_npu 2.12.0.rc1；CANN cann-9.1.0-beta.1 |
| 模拟模式 | meta 设备 + fake_backend 进程组（pp=1），`run_simulator_spawn.py` 每配置单 capture rank（rank0） |
| dtype | 参数/激活 float32（meta 形状捕获；`memory_parameter_storage_dtype` 默认；fp8 checkpoint 存储格式不参与建模，见契约 §1） |
| 模型契约 | `torchtitan_npu/models/glm5_next/MODEL_CONTRACT.md`（loop 固定步数、indexer 冻结、gather 稀疏 DSA 等全部偏差决策） |
| tokenizer/数据 | 文本：`tests/assets/tokenizer/deepseekv3_tokenizer` + c4_test；多模态：`tests/assets/tokenizer/vlm_tokenizer`（`<|image|>`=1998）+ cc12m-test（uniform 方形网格，image_size=56） |

## 2. 支持矩阵与必测配置汇总

| # | 配置 | 并行 | AC | 结果 | 证据目录 |
| --- | --- | --- | --- | --- | --- |
| 1 | 单卡基线（debug） | world=1 | none | PASS | `glm5_next_ev_debug_w1_none` |
| 2 | AC 对照（debug） | world=1 | full | PASS | `glm5_next_ev_debug_w1_full` |
| 3 | AC 对照（debug） | world=1 | selective | PASS | `glm5_next_ev_debug_w1_selective` |
| 4 | 单卡基线（reduced） | world=1 | none | PASS | `glm5_next_ev_reduced_w1_none` |
| 5 | AC 对照（reduced） | world=1 | full | PASS | `glm5_next_ev_reduced_w1_full` |
| 6 | EP | world=2, ep=2 | full（recipe 默认） | PASS | `glm5_next_ev_reduced_ep2` |
| 7 | TP | world=2, tp=2 | full | PASS | `glm5_next_ev_reduced_tp2` |
| 8 | CP | world=2, cp=2 | full | PASS | `glm5_next_ev_reduced_cp2` |
| 9 | CP + AC full 显式 | world=2, cp=2 | full | PASS | `glm5_next_ev_reduced_cp2_full` |
| 10 | 核心组合 FSDP+TP+EP | world=4, tp=2, ep=2 | full | PASS | `glm5_next_ev_reduced_combo` |
| 11 | 多模态通路（vision 进 meta capture） | world=2（FSDP） | selective | PASS | `glm5_next_ev_mm_fsdp2` |
| 12 | 容量级 full（96 层名义/41 唯一块） | world=8, ep=8, seq=2048 | full | PASS | `glm5_next_ev_full_capacity` |
| 13 | MXFP8（attention/MoE 量化） | world=2（FSDP, target A5） | selective | PASS | `simulator_output/glm5_next_debug_mxfp8` |
| 14 | MXFP8 | world=2（FSDP, target A5） | full | PASS | `glm5_next_ev_reduced_mxfp8` |

单测（`tests/unit_tests/models/test_glm5_next.py` 22 项 + `tests/unit_tests/simulator/hardware_shims/test_glm5_next_shim.py` 4 项，全过）：flavor 注册/round-trip、层布局、参数量公式逐字节对账（debug/reduced）、前反向、多模态前向、DSA 输出宽度、indexer 冻结、state-dict round-trip、HF 命名。

**fail fast 探针**（均以明确错误退出，指向 MODEL_CONTRACT.md）：

| 不支持组合 | 报错 |
| --- | --- |
| PP=2 | `NotImplementedError: glm5_next pipeline parallelism is deferred until the loop-region and hc_head cross-stage contract is defined`（pipelining_fn 入口即抛） |
| loop_train_steps=5 | 配置解析期 `ValueError: loop_train_steps=5 must be within train_min/max_steps [1, 4]; adaptive halting is not modeled` |
| share_loop_weights=False | `ValueError: glm5_next v1 only models the weight-shared loop region (share_loop_weights=True)`（校验层+单测覆盖） |
| ETP=2（tp=1） | 框架层 `ValueError: ETP must be 1 or equal TP (1), got 2` |
| TP2+EP2+ETP2 | 框架层 `ValueError: EP*ETP (4) must divide DP-shard*CP*TP (2)` |
| CP2+TP2（world=2） | 框架层 `ValueError: world_size 2 must be divisible by PP*DP-replicate*CP*TP (4)`；tp>1 时模型层另有 `NotImplementedError`（cp 仅支持无 TP 组合） |

## 3. 参数量与切分对照（验收 4.1/4.2）

独立公式（MODEL_CONTRACT.md §8）与模型实参 `sum(p.numel())` 由单测对账（debug/reduced 逐字节相等）：

| 规格 | 总参数量 | 其中 vision | 其中 MoE | 公式=模型 |
| --- | --- | --- | --- | --- |
| debug | 8,291,862 | 449,600 | 4,290,616 | ✓ |
| reduced | 226,652,082 | 8,029,952 | 135,954,720 | ✓ |
| full（仅公式口径，meta 容量验证） | 17,304,210,090,342（17.30T） | 6,578,937,856 | 17,194,675,152,896 | 容量级对账见下 |

每 rank 常驻参数字节（fp32 storage，`persistent_param_bytes`）：

| 配置 | 每 rank 字节 | 对照解释 |
| --- | --- | --- |
| debug 单卡 | 33,167,448 | = 8,291,862 × 4B，精确 |
| reduced 单卡 | 906,608,328 | = 226,652,082 × 4B，精确 |
| reduced EP=2 | 453,304,208 | = 单卡/2 精确（routed experts EP 切半，其余 FSDP/2） |
| reduced CP=2 | 453,304,208 | = 单卡/2 精确（CP 不改变静态权重，减半来自 FSDP） |
| reduced TP=2 | 479,821,256 | = 单卡 × 0.529（head/inter/词表维被切；indexer 保持 Replicate、mHC fp32 参数与 RMSNorm replicated，符合契约 §6/§4.2） |
| reduced 组合 tp2+ep2 | 239,910,672 | = 单卡 × 0.265（TP/EP/FSDP 正交切分，无重复除、无漏切） |
| full world=8（ep=8） | 8,652,105,045,376 | ×8 = 69,216,840,363,008 B ≈ 17.30T × 4B（差 16.6KB 为非持久化对齐项）；ep=8 与 dp_shard=8 重合，全体参数按 8 均匀切分 |

## 4. 核心算子账本（验收 4.3）

融合算子以真实 raw op 名捕获（F=original_forward；AC full 经 checkpoint 边界重算，未产生独立 recompute op 记录；B=backward 对称记录，如 `*_grad`）。理论次数公式：

```
KDA 执行次数（conv/chunk_kda） = pre_KDA + T(loop) + post_KDA      # T = loop_train_steps
DSA（sparse_attn / lightning_indexer） = pre_DSA + post_DSA        # loop 区无 DSA（契约 §3）
GMM = (pre_sparse + T + post) × 3（w1/w3/w2）
mHC（hc_prepost 族） = (块数 × 2 站点) + (T-1) × 2
```

| op | debug 理论/实测 | reduced 理论/实测 | full(ep=8, T=4) 理论/实测 |
| --- | --- | --- | --- |
| `triton_ascend_kernels.chunk_kda[_grad]` | 8 / 8 ✓ | 11 / 11 ✓ | 34 / 34 ✓ |
| `aten.convolution[_backward].default`（KDA fused qkv conv，不 shim，与 kimi_k3 一致） | 8 / 8 ✓ | 11 / 11 ✓ | 34 / 34 ✓ |
| `aclnn.npu_lightning_indexer` | 2 / 2 ✓ | 3 / 3 ✓ | 10 / 10 ✓ |
| `aclnn.npu_sparse_attn_sharedkv[_grad]` | 2 / 2 ✓ | 3 / 3 ✓ | 10 / 10 ✓ |
| `aten._grouped_mm.default` | 24 / 24 ✓（8×3） | 36 / 36 ✓（12×3） | 120 / 120 ✓（40×3） |
| `triton._triton_hc_prepost_fwd_kernel` 族（DSv4 shim） | 20 / 20 ✓ | 28 / 28 ✓ | 88 / 88 ✓ |
| `aten.convolution.default`（vision patch_embed+downsample，仅 mm 配置） | — | — | —（mm_fsdp2：2 / 2 ✓） |

backward 侧 `*_grad` 与 forward 同数（debug 实测 conv_grad=8、chunk_kda_grad=8、sharedkv_grad=2 ✓）；冻结 indexer 无反向算子（`lightning_indexer` 仅前向，契约 §4.2）。

**接口对齐自检**（`hardware_shims/OP_INTERFACE_REFERENCE.md` §0.9）：12 个证据运行逐一解析 `memory_events.csv`，断言模型特有 op 的输入/输出个数与 rank 全部符合 §1–§3 表格——`chunk_kda_grad` 6 进（v 4-D 于位 2）5 出、`lightning_indexer` 3 进（4D/4D/3D）2 出（4-D int32 indices）、`sparse_attn_sharedkv` 先 metadata 后 6 输入主 op（ori_kv 4-D 头折叠 `nh×(k_dim+v_dim)`，softmax_lse 4-D）、`sharedkv_grad` 7 进 4 出；无自造 op 名。此前的记录签名（grad 单输入、indexer 2 进、缺 softmax_lse/sinks 占位）不满足下游 cost model 解析契约，已按参考文档修正并以 `test_glm5_next_shim.py` 4 项单测固化。MoE 路由直方图由 `debug_force_load_balance` round-robin 保证静态（每 expert `m·S·topk/E`），无空 expert padding。

## 5. 显存与 AC 对照（验收 4.5）

saved activations（logical bytes）随 AC 模式变化，peak 由 optimizer 阶段主导（各对照间一致）：

| 配置 | AC | saved tensors (MB) | ckpt 边界 (MB) | recompute saved (MB) | peak (MB) |
| --- | --- | --- | --- | --- | --- |
| debug w1 | none | 87.41 | 0 | 0 | 128.8 |
| debug w1 | full | 7.08 | 2.62(10 实例=9 块+loop) | 0 | 128.8 |
| debug w1 | selective | 7.08 | 2.62 | 8.39(88 张量) | 128.8 |
| reduced w1 | none | 2,819.92 | 0 | 0 | 3,557.2 |
| reduced w1 | full | 255.87 | 117.44(14=12 块+loop×3) | 0 | 3,557.2 |
| reduced tp2 | full | 196.10 | 58.72 | 0 | 1,883.6 |
| reduced ep2 | full | 161.49 | 58.72 | 0 | 1,342.2 |
| reduced combo | full | 131.61 | 29.36 | 0 | 702.0 |
| full ep8 | full | 28,068 | 17,716.7(44 实例=41 块+loop×3) | 0 | 25,952,443.6 |

趋势核对：full AC 使 saved activations 降一个量级（87→7、2820→256）✓；selective 额外引入 recompute saved 张量 ✓；TP/EP/组合按语义维切分 local tensor 而非整体除 world size ✓；loop block 的 ckpt 实例数 = T（debug 2、reduced/full 4）✓ 执行期语义正确进入显存模型。

## 6. MXFP8 量化通路（验收补充）

`MXFP8Converter`（mxfp8_rceil，FQN 清单见契约 §12）端到端跑通（能力门由 simulator `meta_env` 的 meta-safe 补丁放行，真机仍按 Ascend950 把关）：

- capture 记录 `npu.npu_dynamic_mx_quant`（debug 188 / reduced 274 次）、`npu.npu_quant_matmul`（70 / 101 次）——module_path 确认命中 KDA q/k/v/o、DSA q_a/q_b/kv_a/kv_b、indexer wq_b/weights_proj、shared experts gate/up/down，**未命中** router、dense MLP、embedding/output、vision；
- MoE grouped GMM 替换为 `npu.npu_grouped_matmul`（debug 24 / reduced 36 次）；
- 单测 `test_mxfp8_fqn_targets_match_module_paths` 固化 FQN 子串匹配语义（只命中 attention/moe 域）。

## 7. 多模态通路（验收补充）

`glm5_next_debug_mm`（world=2，FSDP）：vision tower（patch_embed conv → 32→2 blocks SDPA → downsample conv → merger）进入 meta capture（`aten.convolution.default` ×2 前向 + backward ✓）；离线数据管线核对：cc12m-test 4 样本批 `pixel_values [2,16,588]`、grid 全有效，**有效 merged patch 数 8 == 文本 image token 数 8**（token id 1998），scatter 顺序一致。早期融合改用 cumsum-order gather（`masked_select` 无 meta kernel，数学等价，MODEL_CONTRACT.md §7 已记录）。

## 8. 复现命令

```bash
# 矩阵（<name>/并行/AC 见第 2 节表；每个配置）
NGPU=<n> python3 scripts/run_simulator_spawn.py --config <config> --training.steps 1 \
  --simulation.output_formats mem [--parallelism.* ...] [--activation-checkpoint.mode ...] \
  --simulation.output-dir ./simulator_output/glm5_next_ev_<name>
# 单测
python3 -m pytest tests/unit_tests/models/test_glm5_next.py -q
# fail fast 探针见第 2 节表（PP/ETP/CP+TP/loop overrides）
```

## 9. 首版限制（Conditionally ready 声明范围）

- loop 固定步数建模（T=4 默认）；adaptive halting / kv_mirror / 非共享 loop fail fast（契约 §3）。
- MXFP8 真实 NPU 训练路径依赖 Ascend950（`has_mx_capability`）；模拟器以 `target_npu_device_type="A5"` 声明后走 torchao meta 包装，真机数值未实测。模拟器捕获经 shim 单测（`test_glm5_next_shim.py` 4 项：绑定保持 FQN/hook、chunk_kda/causal_conv1d/lightning_indexer/sparse_attn 融合名记录、TP 局部切片）与端到端账本双重验证。
- 真实 NPU 数值训练与硬件 profiler 不在本验收目标内（验收规范 §1）。
- video 通路、非均匀 vision 网格、MTP、fp8 参数存储、CP+TP、ETP、PP、offload、compile 均声明不支持并 fail fast。
- 多模态通道仅 image（video token 出现即报错）；vision 仅随 FSDP 分片，不支持 vision TP/CP。
