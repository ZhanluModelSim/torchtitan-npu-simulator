# Chunked loss 的 FSDP 策略恢复

模拟器在 `meta_runtime()` 中通过 `PatchScope` 包装原生
`torchtitan.components.loss.ChunkedLossWrapper.__call__`。离开 runtime 后恢复原方法，
普通 `torchtitan_npu` 导入和原生训练不激活此修复。

## 修复范围

当前依赖中的 chunked loss 临时关闭 head 的前向、反向 reshard 和梯度同步，
最后直接全部设为 `True`，未恢复调用方原值。Kimi 的 head 与 norm 共享 FSDP 参数组，
其原始 `reshard_after_forward=False` 因此在第一次 loss 后变成 `True`，导致后续
microbatch 多一次 head allgather。

`compat/loss.py` 保留原生分块执行、梯度累积、最后一个 chunk 的同步和显式 reshard，
只在调用前保存、调用结束时恢复它触碰的策略：

- 递归 FSDP state 的自动 reshard 开关和各参数组的前向 mesh、反向 reshard 策略。
- head 自身各参数组的 `reduce_grads`、`all_reduce_grads`，分别保存，保留原始 `False`。
- 使用 `finally` 覆盖成功和异常退出；不拦截原生异常、不恢复执行中间状态，不承诺失败后可继续训练。

支持同一 state 多参数组；保存原 mesh 对象，避免将部分 reshard 策略简化为布尔值。
这不是新增 no-sync 或 GA 同步策略：chunk 内仍执行原生同步逻辑。

## 验证证据

`tests/test_chunked_loss_policy.py` 覆盖多组、递归范围、不同初始同步开关、正常／异常
恢复、scope 退出还原，以及非 FSDP 原生 loss 的输出和梯度一致性。

2026-11-02 在 ttsim 中运行平台 `scripts/audit_training_ga.py`，配置为 DP replicate=2、
shard=2、EP=2、selective AC、seq_len=128，连续捕获 4 次真实 Meta F/B：

| 预设 | 前向 allgather 次数 | 完整通信事件顺序 |
| --- | --- | --- |
| `kimi_k3_smoketest` | 7 / 7 / 7 / 7（修复前 7 / 8 / 8 / 8） | 四次相同 |
| `deepseek_v4_smoketest` | 8 / 8 / 8 / 8 | 四次相同，与原核对一致 |

本地报告位于工作区 `runs/kimi-policy-fixed-20261102/kimi-policy-fixed/report.json`
和 `runs/dsv4-policy-fixed-20261102/dsv4-policy-fixed/report.json`；这些是未纳入版本管理的运行产物。
上述验证证明 Meta 通信结构和策略恢复，不代表 NPU 数值或硬件性能验收。
