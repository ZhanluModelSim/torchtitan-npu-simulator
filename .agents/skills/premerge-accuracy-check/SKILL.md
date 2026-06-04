---
name: premerge-accuracy-check
description: "用于验证代码变更的数值精度一致性，生成 PDF 对比报告。当用户说 numerical stability report / 精度验证 / compare precision / verify numerical stability / 数值精度对比 / 精度对比报告 时触发。与 accuracy-debug 互补：本技能在变更前正向验证精度，accuracy-debug 在问题发生后诊断根因。"
---

# premerge-accuracy-check 技能

用于验证代码变更前后的数值精度一致性，生成包含 loss 曲线叠加图、grad_norm 对比和差异统计的 PDF 报告。

## 参考资料

- 对比脚本：[scripts/loss_compare.py](scripts/loss_compare.py)（本仓版，解析 stdout 日志）
- 上游参考：[torchtitan/scripts/loss_compare.py](https://github.com/pytorch/torchtitan/blob/main/scripts/loss_compare.py)（Meta 官方版，编排训练+TensorBoard+CI，本仓脚本灵感来源）
- 报告 HTML 模板：[templates/premerge-accuracy-check.html](templates/premerge-accuracy-check.html)
- Python 依赖：[requirements.txt](requirements.txt)
- Config 矩阵选择：[references/config-matrix-guide.md](references/config-matrix-guide.md)
- 训练日志校验：[references/log-validation-guide.md](references/log-validation-guide.md)
- 日志解析/可视化：`training-log-visualization` 技能（`loss_compare.py` 内部调用 `read_training_metrics`）
- reproduce.json 编写：[references/reproduce-json-guide.md](references/reproduce-json-guide.md)
- 报告格式与邮件：[references/report-format-guide.md](references/report-format-guide.md)

## 适用场景

- 提交 PR 前，验证代码变更未引入数值精度回退。
- 对比两个分支/commit 在相同配置下的训练精度。
- 重构（activation checkpointing 调整、代码重排等）后的 bit-wise 一致性验证。
- 切换 CANN/torch_npu 版本前后的精度对比。

## 不适用场景

- 已出现 NaN/Inf 或 loss 偏离基线 → 使用 `accuracy-debug` 技能。
- 纯 OOM 问题 → 使用 `oom-analysis` 技能。
- 上游 torchtitan 同步 → 使用 `torchtitan-sync` 技能。
- 纯性能回退（吞吐/TFLOPs 下降，loss 正常）。
- 多机训练（多节点分布式），后续迭代再支持。

## 错误上报原则

> [!IMPORTANT]
> **工作流中任何环节出现报错，必须立即上报用户，不得静默降级或绕过。** 包括但不限于：
> - 命令行非零退出码（训练崩溃、OOM、SIGKILL）
> - 日志中的 ERROR/Exception/Traceback
> - Config/CLI 参数覆盖不生效（如 `None` 被解析为字符串）
> - Checkpoint 加载/保存失败（DCP metadata 缺失、路径不存在）
> - 解析器警告（`read_training_metrics` 返回 warnings）
> - NaN/Inf 检出
> - 两次运行 step 数不一致
> - 任何「预期外」的行为
>
> 上报格式：错误类型 + 完整报错内容 + 该环节的上下文（命令/日志片段）。由用户决定降级策略、修复方案、或向上游报 issue。

## 确认清单

在开始前，与用户逐项确认以下内容。**所有项必须明确后**才进入工作流。

- [ ] **分支 A（基线）**：基准分支/commit，如 `cann/master`、`origin/master`
- [ ] **分支 B（待验证）**：待验证分支/commit
- [ ] **模型列表**：代码库 `models/` 目录下的所有模型，由用户选择 `all` 或任意多选
- [ ] **Config 矩阵**：待验证的 config 组合（在工作流 Step 2 中根据 diff 范围逐轴确认）
- [ ] **训练步数**：默认 100 步，至少 100 步, 如果有必要，用 --steps 覆盖默认配置。
- [ ] **Checkpoint**：工作流会自动生成初始 checkpoint（1 步 + HF 格式），两侧从同一 checkpoint 加载训练。若遇 CLI 覆盖/DCP metadata 等问题 → 停止，上报用户
- [ ] **Tokenizer 路径**：tokenizer 文件路径
- [ ] **确定性选项**：`--debug.seed=42 --debug.deterministic`（默认开启，禁止使用 `--debug.deterministic_warn_only`）
- [ ] **报告接收邮箱**：PDF 报告发送到哪个邮箱（不提供则仅保存本地，不发送）
- [ ] **训练日志已存在？**：若用户已分别运行两个分支的训练且有 stdout 日志文件，可直接用已有日志对比，跳过工作流 Step 0-4

> [!IMPORTANT]
> 相同并行策略和 debug 选项下，两次运行应产生 **bit-wise 一致** 的 loss 和 grad_norm。
> 这是本仓的核心精度要求，参见 `AGENTS.md` 数值验证章节。

## 工作流

> **开篇声明**（技能激活后首先告知用户）：
>
> 我将按照 `premerge-accuracy-check` 技能的工作流执行精度验证。该流程分为 9 个步骤：
>
> - Step 0：工作空间审计与准备（git worktree 隔离，不修改用户当前工作区）
> - Step 1：环境确认（NPU/CANN/torch 版本）
> - Step 2：Config 矩阵确认（根据 diff 范围逐轴选择验证配置）
> - Step 3-4：训练运行（每个 config × 每个分支，在隔离 worktree 中执行）
> - Step 5：精度对比（loss_compare.py）
> - Step 6：生成 PDF 报告（含复现步骤）
> - Step 7：邮件发送（可选）
> - Step 8：清理 worktree
>
> **核心原则：全程使用 git worktree 隔离，你的当前工作区零修改**（不 checkout、不 stash、不切换分支）。训练期间可继续在当前分支正常开发。

### 0. 工作空间审计与准备（必做，第一步）

审计工作空间，创建隔离 worktree：

- 收集 git remote、branch tracking、未提交修改、未跟踪文件、stash 列表
- **关键：检测多 remote 下的陈旧 fork。** 若 `origin/master` 和 `cann/master` 同时存在，必须向用户确认用哪个作为基线
- 获取 `git diff <branch_a>..<branch_b> --stat` 确认测试对象
- 将所有变更分为三类：A（已提交测试对象）、B（Infra，两边都需要）、C（仅候选分支）。**逐文件与用户确认分类。**
- 创建两个隔离 git worktree（`../torchtitan-npu-baseline`、`../torchtitan-npu-candidate`），将类别 B 文件复制到两个 worktree

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE_DIR=$(dirname "$REPO_ROOT")
git worktree add "$WORKTREE_DIR/torchtitan-npu-baseline" <branch_a>
git worktree add "$WORKTREE_DIR/torchtitan-npu-candidate" <branch_b>
for wt in "$WORKTREE_DIR/torchtitan-npu-baseline" "$WORKTREE_DIR/torchtitan-npu-candidate"; do
    cp <infra_file> "$wt/<infra_file>"
done
```

> [!WARNING]
> **小心陈旧 fork。** 用户当前工作区**零修改**——不 checkout、不 stash、不切换分支。

### 1. 环境确认

自动搜索 Ascend 安装位置并收集环境信息，覆盖 CANNLab（`/home/developer/`）、个人用户（`/home/xxx/`）、标准安装（`/usr/local/`）等场景。

```bash
# 搜索 CANN 安装位置（覆盖多种安装路径）
find /usr/local/Ascend /home -maxdepth 4 -name "version.cfg" -path "*/ascend-toolkit/*" 2>/dev/null

# NPU 状态
npu-smi info -t board -c 0 2>/dev/null || ls /dev/davinci[0-9]* 2>/dev/null

# torch / torch_npu 版本
python -c "import torch; print(f'torch={torch.__version__}'); import torch_npu; print(f'torch_npu={torch_npu.__version__}')"

# 关键环境变量（简单确认）
env | grep -E '^ASCEND_HOME=|^ASCEND_TOOLKIT_HOME=' || echo "ASCEND_HOME/ASCEND_TOOLKIT_HOME not set"
```

> [!WARNING]
> 搜索结果中任何异常必须**立即上报用户**，附带已收集到的全部输出：
> - 找不到 `version.cfg`（所有路径均无结果）
> - `npu-smi` 和 `/dev/davinci*` 均不可用
> - `import torch_npu` 失败
> - `ASCEND_HOME` / `ASCEND_TOOLKIT_HOME` 均未设置
>
> 由用户判断环境是否就绪，不可自行降级或绕过。

### 2. Config 矩阵确认

根据 diff 范围逐轴确认验证配置。详见 [references/config-matrix-guide.md](references/config-matrix-guide.md)。

关键轴：TP degree、torch.compile、Activation Checkpointing、Optimizer、Expert Parallelism、Context Parallelism、ModelConverters。最小矩阵：1 个 TP=1 + 1 个 TP>1。

> [!IMPORTANT]
> Config 修改必须逐条列出并征得用户同意。

### 3-4. 训练运行

生成初始 checkpoint，两侧从同一 checkpoint 加载训练。

> [!WARNING]
> 任何环节出错（训练非零退出、配置覆盖不生效、checkpoint 加载失败等）**必须立即上报用户**，附带完整报错和上下文。不可自行降级或绕过。参见上方「错误上报原则」。
>
> 已知问题：
> - `--checkpoint.initial_load_path None` 可能被解析为字符串 `"None"` 而非 Python `None`。若遇此问题 → 上报用户
> - DCP metadata 可能缺失导致 `AssertionError: metadata is None` → 上报用户

**B1. 生成初始 checkpoint：**

```bash
cd ../torchtitan-npu-candidate
bash scripts/run_train.sh --module <model> --config <config> \
    --training.steps 1 --debug.seed=42 --debug.deterministic \
    --training.dataset_path=<data_path> \
    --checkpoint.enable --checkpoint.last_save_in_hf --checkpoint.last_save_model_only \
    2>&1 | tee <host_repo>/numerical_report/gen_ckpt_<config>_train.log

mkdir -p <host_repo>/numerical_report/initial_ckpt_<config>
mv outputs/checkpoint/step-1 <host_repo>/numerical_report/initial_ckpt_<config>/
```

**B2. 两侧加载相同 checkpoint：**

```bash
# 基线
cd ../torchtitan-npu-baseline
rm -rf outputs/checkpoint
bash scripts/run_train.sh --module <model> --config <config> \
    --training.steps=<steps> --debug.seed=42 --debug.deterministic \
    --training.dataset_path=<data_path> \
    --checkpoint.enable --checkpoint.load_only \
    --checkpoint.initial_load_path <host_repo>/numerical_report/initial_ckpt_<config> \
    --checkpoint.initial_load_in_hf \
    2>&1 | tee <host_repo>/numerical_report/baseline_<config>_train.log

# 候选
cd ../torchtitan-npu-candidate
rm -rf outputs/checkpoint
bash scripts/run_train.sh --module <model> --config <config> \
    --training.steps=<steps> --debug.seed=42 --debug.deterministic \
    --training.dataset_path=<data_path> \
    --checkpoint.enable --checkpoint.load_only \
    --checkpoint.initial_load_path <host_repo>/numerical_report/initial_ckpt_<config> \
    --checkpoint.initial_load_in_hf \
    2>&1 | tee <host_repo>/numerical_report/candidate_<config>_train.log
```

对 Config 矩阵中的**每个 config** 重复以上步骤。两个 worktree 使用完全相同的 config 修改。

### 5. 精度对比

**5a. 可视化检查**：调用 `training-log-visualization` 技能绘制两条 loss/grad_norm 曲线，目视确认曲线正常下降、无异常跳跃。

**5b. 日志完整性校验**：详见 [references/log-validation-guide.md](references/log-validation-guide.md)。`loss_compare.py` 内置了 NaN/Inf 预扫描和解析警告拒绝，无需手动 grep NaN/Inf。

**5c. 运行对比**：

```bash
python .agents/skills/premerge-accuracy-check/scripts/loss_compare.py \
    --baseline ./numerical_report/<branch_a>_<config>_train.log \
    --candidate ./numerical_report/<branch_b>_<config>_train.log \
    --output ./numerical_report/<case>/
```

### 6. 生成报告

1. 打包类别 B 文件：`tar czf ./numerical_report/infra_files.tar.gz -C "$REPO_ROOT" <files>`
2. 写入 `reproduce.json`（所有字段填实际值，禁止占位符）。详见 [references/reproduce-json-guide.md](references/reproduce-json-guide.md)。
3. 执行 **Stranger Test** 验证复现指南可用性（同 reference）。
4. 生成 HTML/PDF：

```bash
python .agents/skills/premerge-accuracy-check/scripts/generate_report.py \
    --report ./numerical_report/fsdp/report.json \
    --report ./numerical_report/tp/report.json \
    --reproduce ./numerical_report/reproduce.json \
    --template .agents/skills/premerge-accuracy-check/templates/premerge-accuracy-check.html \
    --output ./numerical_report/numerical_stability_report.pdf
```

报告格式与内容要求详见 [references/report-format-guide.md](references/report-format-guide.md)。

### 7. 邮件发送（可选）

HTML 为邮件正文（图片用 CID 嵌入，不是文件路径引用）。附件：PNG（带 Content-ID）、infra_files.tar.gz、diff_summary.csv、train log、numerical_stability_report.pdf、pr_summary.pdf。详见 [references/report-format-guide.md](references/report-format-guide.md)。

### 8. 清理 worktree

```bash
git worktree remove "$WORKTREE_DIR/torchtitan-npu-baseline"
git worktree remove "$WORKTREE_DIR/torchtitan-npu-candidate"
```

> [!TIP]
> - 用户当前工作区从始至终**未被修改**。
> - 训练日志和报告文件（`./numerical_report/`）保留在 host 工作区，供后续查阅。

## 输出要求

最终输出必须包含：

- [ ] 环境信息（NPU 数量、CANN 版本、torch/torch_npu 版本）
- [ ] 分支 A 和分支 B 的 commit hash
- [ ] 使用的模型、config 名称、训练步数、确定性选项
- [ ] **复现步骤**（从 clone repo 开始，包含 infra 变更列表、每条训练命令、compare 命令）
- [ ] loss 曲线叠加图（PNG，嵌入 PDF）
- [ ] grad_norm 曲线叠加图（PNG，嵌入 PDF）
- [ ] 差异统计表（max/mean absolute diff、max/mean relative diff）
- [ ] 逐 step 差异明细（CSV，附在报告末尾或作为附件）
- [ ] 每次训练的 stdout/stderr 日志（`.log` 文件，作为报告附件）
- [ ] 结论：通过（所有差异 ≤ 1e-5）或 未通过（指出超标 step 和差值）
- [ ] HTML 报告文件路径（主要报告格式，浏览器打开即可查看）
- [ ] PDF 报告文件路径（可选，需中文字体支持，乱码时用 HTML 替代）

## 成功标准

| 指标 | 达标标准 |
| --- | --- |
| loss_metrics/global_avg_loss | max absolute diff ≤ 1e-5 |
| grad_norm | max absolute diff ≤ 1e-5 |
| NaN/Inf 数量 | 基线 = 候选（通常均为 0） |

- 若 **通过**：代码变更未引入数值精度回退，可以合并。
- 若 **未通过**：存在超过 1e-5 的差异，需要调查根因。
  - 优先检查：dtype 变化、算子替换、并行策略变更、浮点运算顺序变化。
  - 进一步诊断使用 `accuracy-debug` 技能。

> [!IMPORTANT]
> 两个分支从**同一个 checkpoint** 加载训练，起始权重完全一致。若 checkpoint 生成/加载失败 → 停止，上报用户。不可降级到种子+确定性。

## 关键路径

| 类别 | 路径 |
| --- | --- |
| 对比脚本 | `.agents/skills/premerge-accuracy-check/scripts/loss_compare.py` |
| 训练入口 | `torchtitan_npu/entry.py` |
| 训练启动脚本 | `scripts/run_train.sh` |
| Config 文件 | `torchtitan_npu/models/*/train_configs/*.toml` |
| 报告模板 | `.agents/skills/premerge-accuracy-check/templates/premerge-accuracy-check.html` |
| 精度规范（bit-wise 一致性） | `.agents/AGENTS.md`（数值验证章节） |
