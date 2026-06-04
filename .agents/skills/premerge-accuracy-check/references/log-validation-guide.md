# 训练日志校验指南

运行 `loss_compare.py` 之前，先校验两边日志的完整性。`loss_compare.py` 使用 `training-log-visualization` 技能的 `read_training_metrics` 解析日志，并内置了 NaN/Inf 预扫描和解析警告拒绝机制。

## 1. 可视化检查（推荐）

先调用 `training-log-visualization` 技能绘制两条 loss/grad_norm 曲线，目视确认曲线正常下降、无异常跳跃。

## 2. Step 数量校验

```bash
grep -c "step:" ./numerical_report/<branch_a>_<config>_train.log
grep -c "step:" ./numerical_report/<branch_b>_<config>_train.log
```

若数量不一致 → 某次训练提前崩溃/OOM/被终止，不要继续对比。

## 3. 搜索异常关键字

```bash
rg -i -n "error\|exception\|traceback\|OOM\|out.of.memory\|killed\|SIGTERM\|SIGKILL\|hang\|timeout" \
    ./numerical_report/<branch_a>_<config>_train.log \
    ./numerical_report/<branch_b>_<config>_train.log
```

## 4. NaN/Inf 检测（`loss_compare.py` 自动完成）

`loss_compare.py` 在解析前自动扫描日志中的 NaN/Inf 指标值，并在发现时直接退出（exit 1）。此外，若 `read_training_metrics`（来自 `training-log-visualization`）返回任何解析警告，`loss_compare.py` 同样拒绝对比。无需手动 grep。

## 5. 快速扫描首尾 step

```bash
grep "step:" ./numerical_report/<branch_a>_<config>_train.log | head -5
grep "step:" ./numerical_report/<branch_a>_<config>_train.log | tail -5
grep "step:" ./numerical_report/<branch_b>_<config>_train.log | head -5
grep "step:" ./numerical_report/<branch_b>_<config>_train.log | tail -5
```

## 检查要点

> [!IMPORTANT]
> - **Step 数量不一致** → 训练崩溃/OOM/被终止，不要对比
> - **日志中有 OOM/error/exception/traceback** → 训练异常退出，即使 step 数相同也不可信
> - **loss/grad_norm 出现 NaN/Inf** → `loss_compare.py` 自动检测并拒绝，进入 `accuracy-debug` 排查
> - **loss 恒为常数或为 0** → 模型未正常训练（可能是 checkpoint 加载失败或梯度未更新）
> - **loss 首步差异巨大（>10x）** → 可能两边用了不同的初始化或 checkpoint，检查 config 一致性

只有两边日志都**干净**（无异常关键字、step 数一致、loss 正常下降）时，才继续运行 `loss_compare.py`。

## stdout 精度限制

> [!WARNING]
> 训练日志中 loss 仅打印 5 位小数（`8.5f`），grad_norm 仅打印 4 位小数（`7.4f`）。若差异小于 stdout 的舍入误差，可能无法从日志中检出。对于需要更高精度的验证（如 bit-wise 一致性），可临时修改 `torchtitan_npu/patches/tools/metrics.py` 中的格式字符串（例如 `8.6f` / `7.6f`）增加打印精度，如果认为用户可能需要更高的精度验证，可以和用户询问。
