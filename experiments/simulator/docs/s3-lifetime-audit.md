# S3：AC 与 tensor/storage 生命周期审计

2026-10-21。固定 TorchTitan 0.3.0 / NPU master 快照，与 S0–S2 相同。
S3 代码验收通过：371 项测试、独立类型检查 0 errors。此为 S3 验收快照；后续 MXFP8 状态见 [S4 审计](s4-mxfp8-audit.md)，完整迁移仍需 S5。镜像的最终复核记录见文末。

## 内存计量契约

- 捕获前已存在的输入、labels 和嵌套 metadata 从初始化计量；view/重复引用按 storage 去重，不能等到首次使用才计入。
- 正常入口只计 Meta tensor 对应的逻辑设备字节。CPU 控制元数据保留在原始图中，不混入设备内存。
- 本地参数 shard 是常驻项；FSDP full parameter、AG staging 是独立临时项。
  初始化的注册 buffer 单独常驻计量，并检查其存储/布局在捕获前后不变；buffer 不受参数存储 dtype 假设影响。
  full parameter 由实际 unshard/reshard 事件决定驻留；保存一个 Python 参数 view 不会阻止原生 FSDP 缩零其 storage。
  同一个 storage 对象可跨越多次缩零/重建，必须以不同 allocation interval 计量，不能因句柄相同贯穿整个前反向。
- 原生同步前向 AG 可将 staging 留至下一次 copy-out 或前向根模块清理；异步预取/反向通常在当前 copy-out 后释放。
  下一次 wait 中的局部变量也会持有上一份结果，故不能在 `comm_ctx.all_gather_state = None` 的赋值时立即计为释放。
- 观测 shared comm context 的清理动作，仅暂存整数/字符串元数据，在 copy-out 返回、当前 full parameter 已入账后记录释放。
  根模块在 `_post_forward` 内创建并清理的最后一份 staging 也必须被捕获。
- 通用 tensor 按捕获的 storage/generation 合并别名。弱 storage 引用在 dispatcher 边界检查实际 ownership，
  Python/C++ 引用尚存时不能仅因最后一次算子读取结束而释放。观测器不持有 tensor 或强 storage 引用。
- 实际 autograd 保存槽分别记录 pack、unpack、release。`unpack` 只是读取，不能当作 Node 已释放保存值。
  对 packed handle 用弱 finalizer 观测销毁；保留图或截断捕获中未观测到释放的值持续到捕获末尾。
- native SAC 的缓存字典只增加写入/消费观测，不改变策略、缓存内容、version check 或返回对象。
  即使缓存输出没有普通算子消费者，也按实际缓存持有与消费计费。
- 峰值按同一时间点活跃分配求和，不累计 saved slots、storage inventory 或 offload inventory。
  分配与最终使用同序号时先计入分配再释放。阶段峰值沿实际 F/B 交错时间线求得。

观测粒度是 dispatcher 与框架语义边界，不包含 allocator reserved/cache、碎片、未建模 kernel workspace，
也不能据 Meta 结果宣称真实 NPU kernel 数值或硬件峰值已经验证。

## 独立先验与已复现问题

| 先验 | 预期与验证方式 | 修复前证据 |
|---|---|---|
| FSDP shard/full/staging | 16-byte shard 常驻；64-byte full 仅驻留于 [3,6]、[11,16]；含重叠 staging 的手算峰值为 160 bytes | `s3-oracles-before.log` 中 staging view 过早释放 |
| 延迟 staging | 明确把前向 staging 延至 seq=9；seq=8 仍有 shard 16 + activation 16 + staging 64，full 已释放 | 新增独立区间 oracle；原生源码与 root cleanup 回归共同核验 |
| full AC 外部 loss | `sin` 输出被外部 square loss 保存，必须活至真实 backward 使用；CPU loss/grad 与直接计算完全一致 | `s3-sac-oracle-before.log` |
| SAC 未使用输出 | MUST_SAVE 的未使用 sin 输出仍被 native cache 持有；重算消费后才释放；CPU loss/grad 完全一致 | `s3-sac-oracle-before.log` |
| unpack 后仍驻留 | 自定义 backward 读完 saved value 后再分配 temporary，保存槽仍被 Node 持有，二者必须重叠 | `s3-node-release-before.log` |
| Python 局部持有 | x/sin/cos 各 32 bytes，与 512-byte temporary 共存，手算峰值 608 bytes；弱引用确认局部对象仍在 | `s3-python-storage-before.log` |
| 标准 collective SAC | canonical functional A2A/RS 保存后不得重复前向通信；其 backward collective 及输入梯度仍在 | A2A 的 `s3-comm-sac-before.log`；A2A/RS 回归 |
| CPU/设备分离 | Meta 输入 16 + 输出 32 = 48 bytes；128-byte CPU metadata 不计设备内存且原始依赖保留 | 独立混合设备 oracle |
| 预分配输入 | 32-byte 与 512-byte 输入从开始驻留，与 1024-byte temporary 共存，手算峰值 1568 bytes；嵌套 dataclass/view 不重复计费 | `s3-initial-inputs-before.log`；native 模型入口回归 |
| 初始化 buffer | 参数 shard dtype 投影为 8 bytes，独立 buffer 仍为 16 bytes；参数别名 buffer 不增加分配，未使用前已驻留 | `s3-buffer-before.log` |

oracle 按具体 tensor/storage 身份匹配，不用相同出生序号或 shape 猜测对象。修复前日志保存在审计目录的 `evidence/`。

## AC 策略与配置

`--activation-checkpoint.mode none|full|selective` 使用原生 FullAC / SelectiveAC 配置与 non-reentrant wrapper。
`SimulatorSelectiveAC` 仅覆盖原生扩展点 `get_save_ops()`；保留原生 alternating mm、强制重算 FQN、RNG、early-stop 等规则。
通过 `--simulation.selective-ac-save-ops` 选择 default/gmm/quant-mm/comm 等组；非法组、无已注册算子的组明确报错。

functional A2A/RS 使用目标原生 canonical Meta kernel，SAC 能看到其真实输出；捕获层将实际执行转换为一个语义通信节点，避免重复计费。
原生 `spmd_types` 的 eager ProcessGroup 交换不经过该 canonical save-set，因此仍按原生规则重算。
显式请求 comm/full 而未命中可缓存通信时会 warning，并写入导出的 assumptions。
特殊 synthetic A2A cache 和 8bit 通信仍后置，不假称它们已经恢复。

量化 selector 有独立原生 `npu_quant_matmul` Meta 缓存用例；完整 MXFP8 参数、scale、反向与模型组合仍属于 S4。

```bash
python -m torchtitan_simulator \
  --simulation.world-size 8 \
  --parallelism.data-parallel-replicate-degree 2 \
  --parallelism.expert-parallel-degree 2 \
  --training.num-mtp-modules 1 \
  --training.global-batch-size 32 \
  --activation-checkpoint.mode selective \
  --simulation.output-formats json mem \
  --simulation.output-dir ./s3-selective
```

GBS/GA 由原生 batch degree 与 batch 配置推导，不能改成 1。上述 GA=4，但只捕获一个 microbatch 模板和一次 optimizer 更新；
导出保留实际 GBS/GA、有效 token 分母及捕获范围，不能将该内存峰值称为完整多 MB 累积训练峰值。

`--simulation.memory-parameter-storage-dtype` 是 shard 存储字节假设，不改变实际图或梯度 dtype。
`--simulation.memory-offload-ac-saved-tensors` 是理想保存值驻留投影，并提供逻辑回捞 inventory；
不实际执行 offload，也未将异步传输、临时回捞重叠与带宽建模成真实执行峰值。两项结果须与默认 resident 模式区别解释。

## 证据与出口

- 独立 CPU / Meta oracle；CPU 可表达部分比较实际 loss/grad，其他 fused/collective 仅验证目标原生 Meta 契约。
- native DSV4 + MTP=1：none/full/selective × HSDP+EP、CP+EP，以及 world16/rank15 的 HSDP+EP+TP+CP。
- 每次运行检查梯度清单、所有 AG→staging release 对应、full alloc/free 闭合、真实 consumer 下界、独立区间峰值重算。
- 同三种 AC 验 shard dtype / offload 投影不改变原始图及梯度清单；验证 GBS/GA 不丢失。
- `memory/autograd_saved_tensors.csv` 增加 source/release_tracking/release_seq；新导出 `fsdp_residency_events.json`、
  `storage_lifetimes.json`，与 tensor lifetimes/timeline 共同支持定位峰值来源。
  `tensor_lifetimes.csv` 增加 storage_key；FSDP marker 的 allocation_storage_bytes 区分实际分配成员与仅用于别名追溯的 storage_keys。
- `s3_audit.py` 在固定环境重跑三种 AC，导出 `s3-final/{none,full,selective}`、每项峰值 inventory 与 `comparison.json`。

完整固定环境：`s3-validate.log` 为 371 passed / 0 type errors；`s3-pyrefly-baseline-comparison.json` 为全仓基线 44 / 候选 44 / 新增 0。
常规 pre-commit hooks、Ruff、architecture、OAT 均通过，见 `s3-pre-commit-all.log`；Linux OAT 的 145 文件扫描无问题。
主机的全仓 Pyrefly hook 找不到可执行文件；已在完整固定容器环境分别运行全仓与独立包检查，不能写成全仓 pre-commit 全绿。独立包的自有配置始终单独检查，不整体忽略类型错误。

固定 debugmodel、MTP=1、world8、HSDP replicate2 + EP2、S128、GBS32/GA4 的 resident 模式：

| AC | pre-optimizer peak MiB | forward peak MiB | backward peak MiB | 实际保存槽 |
|---|---:|---:|---:|---|
| none | 586.061 | 534.764 | 586.061 | autograd 771 |
| full | 526.971 | 475.674 | 526.971 | autograd 264 |
| selective | 529.482 | 478.185 | 529.482 | autograd 264 + native cache 29 |

直接观测 FSDP 实际接收的 146 个完整梯度 storage，146 个均有内存区间，缺失与晚于观测才入账的数量均为 0；
证据见 `s3-gradient-storage-audit.json`。

三次运行均有 143 个预期参数梯度，保存槽均观测到释放；28 个 full alloc/free 区间及 30 个 AG/staging release 对应。
这些数字是当前小模型配置的结果，不能套用到旧模型或全规模模型，也不能将 saved slot 数量当作峰值。
`peak_inventory.json` 可追溯每笔活跃分配，并检查底层 storage 不重复；grouped FSDP full allocation 有成员 storage/字节明细。

最终镜像为 `torchtitan-simulator:titan030-s3`，digest：
`sha256:ecaf9a1f36980117c9cd836c6f294b30758f53a89695d77a2f14d86f6b59400c`。
安装后的 78 个 Python 文件与工作区逐文件 SHA256 一致，未依赖开发目录挂载。
镜像独立执行 world16/rank15、HSDP+EP+TP+CP、MTP=1、S512、selective、GBS32/GA8：
143 个参数梯度、361 个通信源事件恰好保留一次、30 个 staging 释放、266 个保存槽释放；
跨模板图无环，300400844-byte 峰值的活跃集合求和一致且 storage 不重复。
证据为 `s3-image-final.log`、`s3-image-validation.json`、`s3-image-source-verification.txt`。

旧 S0–S2 镜像保留原快照；S3 镜像未更改官方完整 CANN 基础环境。

最终逐项证据与文件哈希索引见审计目录 `evidence/s3-final-manifest.json`；
安装镜像导出结果由独立脚本 `s3_verify_release.py` 读取并复核。
