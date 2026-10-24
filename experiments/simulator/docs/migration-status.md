# Simulator 开发与验收状态

mem 汇总已精简为四个含优化器状态的稳态峰值，以及单卡权重／优化器状态大小，旧统计从结果中移除；口径与验证见 [稳态内存](steady-state-memory.md)。

当前新增收敛：只保留 mem 文件导出，返回图不折叠 metadata view；原生精度参数与 DSV4 base BF16 限制见 [本轮审计](mem-only-precision.md)。此前 21 项全规模结果仍绑定原源码哈希，不代表新源码重跑。

2026-10-22。原生 dispatch 收敛后，七个 DSV4 preset × none/full/selective 共 21 项完整原生几何 Meta 仿真重新通过；独立导出与各阶段内存峰值复核通过。
范围为 BF16/MXFP8、MTP=1、非 PP、B=1/S=512 与组合并行；真实 NPU 数值、耗时及 allocator 峰值未验证。

## 原生 dispatch 收敛

2026-10-22 起的当前实现恢复 FSDP 原始事件，使用原生 AdamW、functional/c10d 和 DTensor transport；
metadata 走普通 dispatcher，缺失梯度与旧 synthetic SAC 不再人工补齐。
431 项测试通过；sidecar 类型 0 errors，全仓同环境基线／候选 64→64、新增 0；最终镜像源与矩阵源哈希一致。
当前轮完整复验的状态与证据见 [收敛记录](dispatch-first-remediation.md)；下文 S0–S5 保留历史验收口径。

## S5 历史交付

- 原生 Flash 43 层、Pro 61 层、旧 20T 几何 96 层均执行完整 F/B/clip/optimizer；未以 build-only 或缩小几何代替。
- 模板 ID 统一、跨模板双向依赖恢复、DOT 跨模板边保留；SchedulePlan 是权威调度，CSV 不另推调度。
- 图引用/DAG、通信唯一归属、实际保存张量来源、梯度/clip/optimizer 集合、FSDP 释放与逻辑内存区间峰值均核验。
- 20T 的旧「不压缩」标记 0 映射到原生 1；BF16/MXFP8 的别名对照测试及完整 20T 复测通过。
- 424 tests passed；独立包 0 type errors、原有 14 suppressed；全仓同环境 64→64，新增 0。
- 完整 pre-commit 已执行，常规/Ruff/architecture/OAT 通过，主机 Pyrefly 缺失，故不标全仓全绿。
- 独立安装镜像的完整 20T MXFP8 selective 与 world2048/rank2047 用例通过；82 个包源文件哈希匹配。
- 原生 TorchTitan/NPU 663 个与 TorchAO-NPU 50 个 Python 文件未修改；原工作区用户变更保留。

逐项能力与边界见 [能力矩阵](capability-matrix.md)，执行和迁移命令见 [交付说明](s5-delivery.md)，
精确结果与来源哈希见 [S5 结果](s5-results.json)，自审见 [S5 审查](s5-review.md)。
以下 S0–S4 各节保留历史阶段的验收口径，不代表当前仍缺少上述 S5 能力。

## S4 最新实现与证据

- 原生 `extension.quantization.enable_quantized_training` + `all_mxfp8`；不移植旧 FQN CLI 复写及额外默认名单。
- debugmodel/MTP=1 实际选择 56 个 MX wrapper 参数，143 个预期参数梯度；量化 code/scale、保存槽、重算与释放均有真实来源。
- MM/BMM/grouped MM 的 8448/16896/17152-byte 先验，以及 full/selective AC 的 24576-byte 间隙先验通过。
- 九个 MXFP8 的 AC/并行组合与六个 BF16/MXFP8 配置构建通过；大模型完整执行仍属 S5。
- 类型化模型几何 CLI 已通过原生配置树重建及三种 AC 的量化 F/B/O；不恢复 MXFP8 FQN 复写。
- 410 项测试通过；独立包 0 type errors；全仓同环境基线/候选 64→64，新增 0。
- 完整复核与镜像、环境限制见 [S4 审计](s4-mxfp8-audit.md)，设计核对见 [S4 自审](s4-review.md)。


## S3 最新结果

- 371 项测试通过；独立 distribution 类型检查 0 errors，原有 13 项局部忽略未增加；Ruff check/format 通过。
- 新增可手算的 160-byte FSDP shard/full/staging 区间、608-byte Python 局部持有、48-byte 设备/CPU 分离等先验；
  full AC 外部 loss、SAC 未使用输出、Node 在 unpack 后继续保存值均有修复前失败证据与 CPU loss/grad 对照。
- native DSV4 + MTP=1：三种 AC × HSDP+EP、CP+EP、world16/rank15 HSDP+EP+TP+CP；所有预期参数梯度与跨模板 DAG 保持闭合。
- 校验每次 AG 都有 staging 释放；full alloc/free 闭合；峰值按独立区间求和复核，实际 storage 在峰值处不重复计费。
- `source/release_seq/storage_key` 与 FSDP 分配明细随 memory 导出；保存槽读取与销毁分开，弱 storage ownership 不因最后读取结束而提前释放。
- 参数 shard dtype / 保存激活 offload 投影不改变图或梯度清单；GBS/GA 及 token 分母保留。eager SPMD 交换按原生 SAC 行为重算，显式 comm 未命中时报告。
- 全仓 pre-commit 的常规、Ruff、architecture、OAT hooks 通过；主机 Pyrefly hook 因缺少可执行文件未通过；完整依赖容器中的检查结果如下。
  在完整固定环境、统一包搜索根后，原始主线与候选均为 44 个相同既有错误，新增 0；与下文历史 64 的命令口径不同，不代表修复了生产类型错误。
  sidecar 由自有配置检查，避免根项目把同一类同时解释成 `experiments.simulator.*` 和 `torchtitan_simulator.*`。
- 原工作区 tracked diff SHA256 仍为 `cfc1085feacba48b89f803f0265f72fa49be0a946093ff34ff60445ca58c12ab`，未覆盖用户 PP 修改。
- S3 镜像 `torchtitan-simulator:titan030-s3` 已由安装包独立复验 rank15 的组合并行与 selective AC；78 个 Python 文件与工作区 SHA256 一致。
  镜像 digest 和峰值明细见 S3 审计报告。上游 TorchTitan/NPU 的 663 个 Python 文件与冻结源码哈希一致。

细节、边界和证据入口见 [s3-lifetime-audit.md](s3-lifetime-audit.md)，自审见 [s3-review.md](s3-review.md)。
以下各节保留 S0–S2 的历史验收口径。

## 固定验证基线

| 项目 | 固定值 |
|---|---|
| 旧 simulator | da283eb674af2962695e120cad1d193a138d51c7 |
| TorchTitan v0.3.0 | 086bf6c166ec85c1298eb5596fa9bf95f6a2d840 |
| NPU master 快照 | 7eb02313a7fb5ac764133f0c4d909a7e8ce44da1 |
| torch / torch_npu | 2.14.0+cpu / 2.14.0.dev20260918 |
| 官方完整 CANN 镜像 | swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-beta.2-950-ubuntu22.04-py3.12 |
| 镜像 index digest | sha256:6cd0c0ba13f9d6f85667c6e27b578c341446fa5af80be3d6f15a624ecd6a90c8 |
| 验证隔离环境 | Python 3.12，arm64，PyArrow 25.0.0 |

官方镜像包含 CANN，未向干净镜像安装 `.run` 包。beta.2 与目标文档 CANN 9.2 的配套差异、
目标 CI torch 2.15 dev 与 requirements torch 2.14 的差异仍需在真实 NPU 发布验收时明确。
上述 SHA 是本次固定快照，不声称始终等于最新远端 master。

## 已实现与证据

- 独立 sidecar，配置/入口与生产包分离；运行时适配由 PatchScope 和捕获 context 恢复。
- 原生 DSV4 debugmodel + MTP=1 的前向、主/indexer/MTP loss、反向、clip、fused AdamW 和 scheduler。
- 11 个 rank-0 用例：单逻辑卡、纯 DDP、FSDP、HSDP、EP+EFSDP、TP/SP、CP、HSDP+EP、EP+TP、EP+CP、HSDP+EP+TP+CP。
- rank 1/8/15 的组合并行，独立核验 dense/expert mesh、local shape、通信域与参数字节；另验 MTP=0。
- 全部预期训练参数有梯度，验证 shape/dtype/placement，以及 mesh 拓扑与实际 process group。
- 实际 RMSNorm、token-permute 及其原生 backward 算子进入捕获；组件测试检查真实保存对象身份。
- 通信源事件完整保留；梯度归约的实际图节点属于 optimizer 祖先；未消费 prefetch 不再被归一化删除。
- 按原生 finalize_backward 的实际清理采集 pending AG buffer 释放，按 transition 关联 staging 驻留。
- 原生 chunked loss 在 forward 中交错执行 head backward：外层计算块与逐 op 的实际 phase 分别记录；阶段内存峰值按真实时间区间计算。
- text/html/dot/json/csv/mem 已从完整入口生成。格式的下游全面回归仍属于 S5。
- `scripts/validate.sh`：334 tests passed；sidecar Pyrefly 当前配置 0 errors。Ruff check/format 通过。
- optimizer 完整声明组与实际 fused 更新集合分开；分桶/unused 参数测试验证不扩组、不重复计费。
- 所有实际梯度归约都是 clip 和 optimizer 的祖先；每个训练参数均有 clip 更新。逻辑 operand 对照原始参数 shape，内存 operand 对照 local 参数、梯度、双状态和 step 字节。
- plain import 不激活 simulator；异常退出恢复绑定、autograd TLS、通信/metadata/FSDP 状态。non-A5 选择不继承宿主 A5 flag；capture worker 0 与 logical rank 独立。

证据文件保存在本机独立审计目录 `torchtitan-migration-audit-20261020/evidence/`：
`s0-s2-validate.log`、`s2-native-identity.log`、`s2-image-hybrid-rank15.log`、`s2-final-exports.log`。
旧版代表性图与内存保存在 `old-baseline/`；配置命令、源码状态、哈希见 `s0-old-baseline-manifest.json`。
旧 smoketest dim=128，新原生 debugmodel dim=256；模型实现、规模和输入来源发生变化，旧峰值不能直接作为新版阈值。
后续运行会更新这些记录，不把历史通过直接当作新修改后的验收。

## 旧 patch 处置与升级耦合

38 项旧 helper 的迁移前清单保留在 [legacy-patch-inventory.md](legacy-patch-inventory.md)。
当前只恢复本阶段真实路径需要的能力，不整批安装旧全局 patch：

| 能力 | 当前实现/处置 |
|---|---|
| 旧 converter singleton / 模型注入 | 改用原生 Configurable 和显式 override.imports |
| 全局 NPU 设备重定向 | 不安装；仅元数据 helper 的局部 factory 转 Meta，以及 Meta unit norm buffer 初始化 |
| Meta RoPE/RMSNorm/token-permute 断梯度 | 先探测原生梯度传播；确有缺口时使用原生 backward op 桥接并保存实际输入 |
| MoE `.tolist()` split | 明确均衡路由，静态推导 host split，保留原生 count/data 交换和控制依赖 |
| FSDP device/Meta validation | 仅 Meta 路径适配，保留原生 mesh/placement/sharding 校验 |
| 旧 HSDP 轴修正 | 未安装旧修正；原生 dense/expert mesh 通过组合模型与非零 rank 的独立 shape/字节 oracle |
| 旧参数 dtype backport | 未搬回生产；使用目标原生参数与归约 dtype policy |
| FSDP 驻留/collective 按值别名 | 观测最终生效的 native wrapper，按函数身份刷新缓存绑定，退出恢复 |
| DTensor/SPMD/RoPE 跨捕获状态 | 隔离 Python DTensor cache、RoPE pool、SPMD TLS；清理原生 DTensor cache，避免旧 world mesh 泄漏 |
| fused optimizer | 保留原生配置；仅 Meta bypass 硬件探测，分别记录全局工作量与本地内存 operand |
| PP / 多 capture worker / MXFP8 特例 | 不恢复为隐式兼容；对应入口和支持状态明确后置 |

私有接口集中于 `compat/distributed.py` 和 capture 观测层；升级时优先检查这些 seam。
原生 DTensor cache 清理会丢弃缓存加速结果，不修改模型/梯度数据。无可靠能力匹配时显式失败。

## 质量检查与边界

- 新增代码的单测、模型矩阵、Ruff 和独立 distribution 类型检查通过。
- 全仓 pre-commit 常规 hooks 通过。macOS OAT 脚本存在 GNU sed 兼容问题；已在 Linux 对最终 staged overlay 运行，135 files、0 invalid file type、0 invalid license header。
- 全仓 Pyrefly 在完整目标环境下仍报 64 个错误。原始 NPU master 同条件也报 64 个；归一化消息、文件和行号后新增为 0。证据：`pyrefly-baseline-native.log`、`pyrefly-full-native.log`、`pyrefly-baseline-comparison.txt`。因此不声称全仓 pre-commit 全绿；本次没有修改生产代码处理基线问题，未提交或推送。
- Pyrefly 的 13 个局部/已有忽略包括 Function 固定 backward arity 对变参基类 stub、可选 orjson 和动态 hook typing；未整体关闭类型错误。
- 同步 collective 的保守 host control edge 与真实 tensor data edge分别标注；async all-reduce 不冒充同步完成。FSDP async 顺序依据实际 transition/residency 事件。
- 原有 PP 调度单测已修复并通过，但 PP 训练捕获仍明确拒绝，不能据此宣称 PP 已迁移。
- 无真实 NPU，不证明 fused kernel 数值、实际 offload 或硬件耗时。内存是逻辑活跃 storage，不含 allocator reserved、碎片和未建模 workspace。

历史 S0–S2 未包含 AC。AC 的 BF16 状态见 S3，MXFP8 状态见 S4；完整规模/下游导出、FP8 通信及 PP 仍按后续里程碑处理。

逐项出口证据见 [acceptance.md](acceptance.md)，38 个旧 helper 的实际处置见对应台账末节。
