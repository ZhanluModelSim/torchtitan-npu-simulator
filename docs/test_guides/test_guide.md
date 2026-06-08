# 测试使用指南

## 常用命令
### 单元测试
```bash
# 运行全部单元测试（带 NPU patch 的 torchtitan 上游 UT + 本仓 torchtitan-npu UT）
bash .ci/unit_test.sh

# 只运行本仓 `torchtitan-npu` 的单元测试（直接用 pytest）
python3 -m pytest -v --tb=short tests/unit_tests
```

### 冒烟测试
```bash
# 运行 smoke 套件（torchtitan-npu 集成 smoke + tests/smoke_tests）
bash .ci/smoke_test.sh
```

> 注意：`.ci/smoke_test.sh` 中 `run_torchtitan_smoke`（上游集成 smoke 路径）当前被注释掉，
> 因此默认运行只执行 torchtitan-npu 集成 smoke 和 `pytest tests/smoke_tests`。

### 集成测试 (Integration Test)

`tests/smoke_tests/integration_test.py` 是端到端集成测试入口，用于验证：
- 新增模型功能支持情况
- 特性兼容性
- 并行策略兼容性

#### 运行方式

```bash
# 通过 .ci/smoke_test.sh 运行（集成 smoke 是套件的一部分）
bash .ci/smoke_test.sh

# 独立运行 integration_test.py
python tests/smoke_tests/integration_test.py output_dir \
    --test_name all \
    --ngpu 2
```

#### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | 无（必填） | 测试输出目录 |
| `--test_name` | `all` | 指定测试用例名称 |
| `--ngpu` | `2` | 最大 GPU 数 |

#### OverrideDefinitions 使用说明

`OverrideDefinitions` 是定义集成测试用例的配置类：

```python
OverrideDefinitions(
    override_args=[[...]],  # 必填：命令行参数列表
    test_descr="...",        # 必填：测试描述
    test_name="...",         # 必填：测试名称
    ngpu=2,                  # 可选：所需 GPU 数
    disabled=False,          # 可选：是否禁用
)
```

#### 新增测试用例步骤

1. 打开 `tests/smoke_tests/integration_test.py`
2. 在 `generate_smoke_tests()` 中，按用例类型将新配置添加到对应的分组函数（`_fake_backend_tests()` / `_base_tests()` / `_tp_tests()` / `_ep_tests()` / `_checkpoint_tests()`）的返回列表中（例如下面的 TP 用例加到 `_tp_tests()`）：
```python
OverrideDefinitions(
    [
        [
            "--module your_model",
            "--config your_config",
            "--parallelism.tensor_parallel_degree 2",
        ],
    ],
    "Your Model TP Test",
    "your_model_tp",
    ngpu=2,
)
```
3. 运行测试验证：
```bash
python tests/smoke_tests/integration_test.py ./outputs --test_name your_model_tp
```

#### Config Registry 配置

集成测试每个测试用例会向`scripts/run_train.sh` 传入 `--module` 和 `--config`，再追加 `tyro` 嵌套覆盖参数，例如
`--training.steps 2` 或 `--parallelism.tensor_parallel_degree 2`。

### 模型并行专项命令
```bash
# 基础模型并行冒烟测试
python3 -m pytest -v tests/smoke_tests/model_parallel/

# 多进程模型并行冒烟测试
RUN_MODEL_PARALLEL_MULTI_RANK=true torchrun --nproc_per_node=4 -m pytest -v tests/smoke_tests/model_parallel/
```

## 什么时候用哪个命令
| 命令 | 适用场景 |
|---|---|
| `.ci/unit_test.sh` | 修改的是硬件无关逻辑，比如 converter、config、helper、patch |
| `.ci/smoke_test.sh` | 修改的是真实 NPU 执行链路或 wrapper 行为，需要跑集成 smoke |
| `python tests/smoke_tests/integration_test.py ... --test_name <name>` | 需要单独跑某个端到端集成测试用例 |
| `python3 -m pytest tests/smoke_tests/model_parallel/` | 修改了模型并行行为 |

## 快速判断
- 只改了硬件无关逻辑：先跑 `.ci/unit_test.sh`
- 改了 NPU 特性链路或 wrapper：跑 `.ci/smoke_test.sh`
- 改了训练主链路接线：用 `integration_test.py --test_name <name>` 定向跑对应用例
- 改了模型并行行为：跑 `pytest tests/smoke_tests/model_parallel/`

## 测试报告
- 输出目录：`test_reports/`
- 常见产物：
  - `smoke_test.log`：torchtitan-npu 集成 smoke 日志
  - `integration_tests/`：集成测试结果目录

## 使用建议
1. 先跑和改动最匹配的最小命令。
2. 不依赖 NPU 的改动，优先跑 `.ci/unit_test.sh`。
3. 能用 `integration_test.py --test_name <name>` 定向跑单个用例时，就不要默认全量跑 smoke。
4. 如果测试布局或执行方式变了，记得同步更新文档。
