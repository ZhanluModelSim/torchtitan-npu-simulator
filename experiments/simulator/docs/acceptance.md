# 分阶段出口核验

当前开发按 [统一指南](developer-guide.md) 分为 [捕获运行时与负载契约](capture-runtime.md) 和 [模型适配与验收](model-integration.md)。运行时升级与新模型接入均按对应规则建立新证据。
历史阶段的通过项和测试数量不替代新模型的保存值来源、驻留／传输及独立峰值核验。

当前原生 dispatch 收敛及复验见 [本轮验收](dispatch-first-remediation.md) 与 [机器可读结果](dispatch-first-results.json)。下文 S0–S5 按当时的验证口径保留。

## S0–S2 出口核验

2026-10-21。范围为固定 TorchTitan v0.3.0 / NPU master 快照、DSV4 debugmodel、BF16、AC none、PP=1。

| 计划要求 | 直接证据 |
|---|---|
| S0 固定源码、完整镜像、输入与旧基线 | migration-status 的 SHA/digest；旧 baseline manifest；合成单文档和均衡路由假设随输出记录 |
| S0 可选包、新配置入口与明确拒绝 | pyproject/cli/config；MTP 0/1 build-only、plain-import、A5/PP/FP8 拒绝测试 |
| S0 目标/host/capture/logical rank 分开 | capture_coordinates；宿主 A5 flag 干扰测试；rank 1/8/15；compile/CUDA graphs 关闭 |
| S1 身份、alias、mutation、generation | CPU/Meta/DTensor saved identity、parameter provenance、storage witness 释放测试 |
| S1 真实组件及 autograd | RoPE/RMSNorm/permute connectivity 探针与 native backward；完整 attention/compressor/LI/mHC/MoE 模型梯度 |
| S1 保存专家实际前向值 | native expert saved slot 的 tensor ID/storage/shape/dtype 对照 raw events，不创建同形占位值 |
| S1 主/indexer/MTP loss、F/B/clip/O | MTP 0/1 完整运行；梯度清单、逐参数 clip、native fused AdamW 与 scheduler |
| S2 各并行轴和关键组合 | 11 个 rank-0 用例、3 个非零 rank；native SPMD backend；CP 原生 host planner |
| S2 通信唯一归属与完整依赖 | 每个真实 comm 源 ID 恰好出现一次；跨模板 DAG 无环；count A2A 到 data A2A 的依赖保留 |
| S2 参数/AG/wait/reshard/buffer 释放 | FSDP observer 安装在 native 延迟增强后；按 transition 关联 staging；实际 finalize 清理；无 marker imbalance |
| S2 归约到 clip 再到 optimizer | 所有真实梯度归约都属于 clip 和 optimizer 的祖先；完整参数组与本次更新子集分开 |
| S2 全局工作量与 local 内存分开 | fused operand shape 对照并行化前原始参数；local 参数/梯度/双状态/step 字节 oracle；分桶不重复计费 |
| S2 旧 workaround 退役依据 | 38 项逐项台账；native HSDP dense/expert、FakePG 和 dtype 模型级用例；不复制旧轴修正 |

`scripts/validate.sh` 执行完整 sidecar 测试和类型检查。镜像入口还独立执行 rank 15 的
HSDP+EP+TP+CP、MTP=1、S=512 捕获；此用例不依赖开发容器的 editable sidecar 安装。

全仓类型基线的 64 个既有错误、Linux OAT 结果以及无真实 NPU 的边界均保留在
[migration-status.md](migration-status.md)，不能将本阶段交付解释为全仓 CI 或完整 DSV4 迁移通过。

## S3 出口核验

| 计划要求 | 直接证据 |
|---|---|
| 原生 Full/Selective 配置与 get_save_ops | `test_s3_native_lifetimes.py` 的配置构建及实际保存值用例；未修改原生 getter |
| pack/unpack/release 与重算 | CPU full AC 外部 loss、未使用 SAC 输出、Node 延迟释放、quant-mm Meta 缓存先验与 native 模型矩阵 |
| FSDP 临时驻留与 staging 生命周期 | 160-byte 独立区间、延迟 staging、根模块内部清理回归；所有 AG 逐项对应 release，full alloc/free 闭合 |
| storage 身份与不重复计费 | 弱 ownership、原始 storage key、完整 FSDP 分配成员；各配置独立重算峰值且峰值 storage 唯一 |
| 参数/buffer/梯度/state/激活分类 | 初始化 buffer / 1568-byte 预分配输入 oracle、146 个实际完整梯度 storage 审计、optimizer/gradient 依赖、saved-slot 与缓存来源；CPU 元数据排除先验 |
| none/full/selective 的保存/重算/通信比较 | `s3-final/comparison.json` 与各模式 `peak_inventory.json`；functional A2A/RS SAC；eager SPMD 行为明确报告 |
| dtype/offload 与逻辑回捞明细 | 三种 AC 下图/梯度清单不变；模型投影与 native resident capture 分开标注 |
| GA>1 保留而不冒充完整累积峰值 | GBS32/GA4、组合 rank15 的 GA8 元数据及 token 分母；captured_microbatches=1 |
| 质量与独立安装包 | 371 tests、sidecar 0 type errors、全仓同环境 44→44；S3 镜像 78 文件哈希与 361 通信源事件复核 |

完整证据、镜像 digest、自审与未验证边界见 [S3 生命周期审计](s3-lifetime-audit.md)。

## S4 出口核验

| 计划要求 | 直接证据 |
|---|---|
| 原生量化入口，无 FQN CLI 复写 | `test_s4_mxfp8.py` 的 CLI 拒绝、实际包装后的参数清单；原生 `apply_quantization_converter` |
| 真实量化算子与前反向保存值 | MM/BMM/grouped MM 三个独立字节先验；storage 对照实际原生量化输出，输入/权重梯度与低精度反向 matmul |
| AC 下释放与重算正确 | 24576-byte 间隙先验；原始 code/scale 释放，重算 storage 保留至实际消费 |
| 并行与 optimizer 的完整链路 | 三种 AC × HSDP+EP/CP+EP/组合 rank15；全部梯度、归约→optimizer、DAG、通信与峰值 storage 唯一 |
| typed 模型几何复写 | 四个 preset 的空复写结构对照，三种 AC 的 CLI 量化 F/B/O，安装镜像几何与参数 shape 直接核验 |
| 新版配套缺口可撤销适配 | CANN SwiGLU backward arity 最小复现；Meta bridge 保存真实 x/score 并恢复绑定 |
| Flash/Pro/20T 的 BF16/MXFP8 配置 | 六个真实尺寸 Meta 构建，MTP=1；20T 为保留旧几何的原生配置树适配 |

完整规模执行及所有下游消费留待 S5；不据配置构建声称完整规模 F/B/O 或数值验收。
运行结果、镜像与源文件证据见 [S4 审计](s4-mxfp8-audit.md)。

## S5 出口核验

| 要求 | 直接证据 |
| --- | --- |
| 七个 preset、完整几何及三种 AC | `s5-results.json` 的 21 项 native-size F/B/clip/optimizer；MTP=1、B1/S512、组合并行 |
| L0–L3 引用、DAG、DataSlot、通信归属 | 正常入口自动 validation；原生模型级负向破坏测试；独立读取导出 CSV 逐边复核 |
| 内存与实际保存值 | 每次执行核验保存值来源与释放；独立读取生命周期 CSV 求峰值、比对 summary 与报告 |
| 六种输出与有限 rank 导出 | smoketest 全格式；大模型 text/csv/mem；world2048 的 CSV=1 而 world/mesh 保持不变 |
| 20T 旧配置语义 | 压缩比 0→1 别名对照；完整 96 层/12288 dim/2048 专家 BF16/MXFP8 三种 AC |
| 软件包与环境 | 安装镜像 82 文件哈希；独立完整 20T MXFP8 selective；713 个原生包文件未修改 |
| 质量、迁移说明与范围 | 424 tests、独立类型0、全仓64→64；完整 pre-commit 的主机 Pyrefly 限制如实记录；能力矩阵及命令可复现 |

NPU 数值／性能、PP、FP8 通信、完整多 microbatch 内存回放和 metadata 准备成本不由本阶段证明。
S5 当时的 CANN 普通 dispatcher 入口问题已在本轮收敛中修复；准备阶段成本仍单列于 `open-questions.md`。
