# 报告格式与邮件发送指南

## HTML/PDF 生成

```bash
# 将所有 case 的 report.json 传给同一个 generate_report.py（一个 --report = 一个 case）
python .agents/skills/premerge-accuracy-check/scripts/generate_report.py \
    --report ./numerical_report/fsdp/report.json \
    --report ./numerical_report/tp/report.json \
    --reproduce ./numerical_report/reproduce.json \
    --template .agents/skills/premerge-accuracy-check/templates/premerge-accuracy-check.html \
    --output ./numerical_report/numerical_stability_report.pdf
```

该脚本同时生成：
- HTML：`./numerical_report/numerical_stability_report.html`（始终可用）
- PDF：`./numerical_report/numerical_stability_report.pdf`（需中文字体）

## 中文字体检查

生成 PDF 前，先检查中文字体是否可用：

```bash
fc-list :lang=zh 2>/dev/null | head -3
```

- **有输出** → 字体可用，PDF 正常生成。
- **无输出** → 向用户报告："系统未安装中文字体，PDF 会乱码。HTML 报告不受影响，可直接用浏览器打开。是否需要安装字体后重试？"

> [!TIP]
> 中文字体安装：
> ```bash
> # CentOS/RHEL: yum install -y google-noto-sans-cjk-fonts
> # Ubuntu/Debian: apt-get install -y fonts-noto-cjk
> fc-cache -fv
> ```

## 报告内容要求

- 环境信息（NPU 数量、CANN 版本、torch 版本）
- 运行参数（分支A/B commit hash、模型、config、步数、并行策略）
- 分支变更摘要（分支关系 + 代码改动总结）
- **复现步骤**（从 clone 开始到 compare 命令的完整操作序列）
- 每个 case 的 loss 曲线叠加图 + grad_norm 曲线叠加图
- 差异汇总表（max/mean absolute/relative diff）
- 结论（通过/未通过 + 建议）

## 复现步骤格式

每条命令自包含，可独立复制粘贴执行。实际步骤数量取决于 Config 矩阵。

> [!NOTE]
> 以下为格式示例。每个 config 组合产生三步：先生成初始 checkpoint，再分别运行基线+候选（均从同一 checkpoint 加载），最后对比。

> ## 复现步骤
>
> ### 1. 准备代码
> ```bash
> git clone <repo_url> && cd torchtitan-npu
> git checkout <candidate_commit>  # 先切到候选分支生成 checkpoint
> tar xzf ./infra_files.tar.gz
> ```
>
> ### 2. 生成初始 checkpoint — <config_label_1> (branch: <candidate_branch>)
> ```bash
> git checkout <candidate_commit>
> tar xzf ./infra_files.tar.gz
> bash scripts/run_train.sh --module <model> --config <config_1> \
>     --training.steps 1 --debug.seed=42 --debug.deterministic \
>     --training.dataset_path=<data_path> \
>     --checkpoint.enable --checkpoint.no_load_only \
>     --checkpoint.initial_load_path None \
>     --checkpoint.last_save_in_hf --checkpoint.last_save_model_only \
>     2>&1 | tee ./numerical_report/gen_ckpt_<case>_train.log
> mkdir -p ./numerical_report/initial_ckpt_<case>
> mv outputs/checkpoint/step-1 ./numerical_report/initial_ckpt_<case>/
> ```
>
> ### 3. 基线训练 — <config_label_1> (branch: <baseline_branch>)
> ```bash
> git checkout <baseline_commit>
> tar xzf ./infra_files.tar.gz
> rm -rf outputs/checkpoint
> bash scripts/run_train.sh --module <model> --config <config_1> \
>     --training.steps=<steps> --debug.seed=42 --debug.deterministic \
>     --training.dataset_path=<data_path> \
>     --checkpoint.enable --checkpoint.load_only \
>     --checkpoint.initial_load_path ./numerical_report/initial_ckpt_<case> \
>     --checkpoint.initial_load_in_hf \
>     2>&1 | tee ./numerical_report/baseline_<case>_train.log
> ```
>
> ### 4. 候选训练 — <config_label_1> (branch: <candidate_branch>)
> ```bash
> git checkout <candidate_commit>
> tar xzf ./infra_files.tar.gz
> rm -rf outputs/checkpoint
> bash scripts/run_train.sh --module <model> --config <config_1> \
>     --training.steps=<steps> --debug.seed=42 --debug.deterministic \
>     --training.dataset_path=<data_path> \
>     --checkpoint.enable --checkpoint.load_only \
>     --checkpoint.initial_load_path ./numerical_report/initial_ckpt_<case> \
>     --checkpoint.initial_load_in_hf \
>     2>&1 | tee ./numerical_report/candidate_<case>_train.log
> ```
>
> <!-- 对 Config 矩阵中的每个组合，重复步骤 2–4 … -->
>
> ### N. 精度对比
> ```bash
> python .agents/skills/premerge-accuracy-check/scripts/loss_compare.py \
>     --baseline ./numerical_report/baseline_<case>_train.log \
>     --candidate ./numerical_report/candidate_<case>_train.log \
>     --output ./numerical_report/<case>/
> ```
>
> ### N+1. 结果文件
> 所有输出位于 `./numerical_report/`

## 邮件格式

- **正文（HTML 内联，图片用 CID 嵌入）**：
  1. 读取 `numerical_stability_report.html` 内容。
  2. 将 HTML 中的 `<img src="fsdp/loss_comparison.png">` 替换为 `<img src="cid:loss_comparison">`，grad_norm 同理（每个 case 的图片需要唯一 CID）。
  3. 以 `MIMEText(html, "html", "utf-8")` 作为邮件正文。**HTML 只放在正文，不作为附件。**
- **附件（数据文件）**：
  - `loss_comparison.png`、`grad_norm_comparison.png` — 附加 `Content-ID` 头，正文中 `cid:xxx` 引用
  - `infra_files.tar.gz` — 复现所需的全部 infra 文件
  - `diff_summary.csv` — 逐 step 差异明细
  - `*_train.log` — 每次训练的完整 stdout/stderr 日志
  - `numerical_stability_report.pdf` — PDF 报告（若生成成功）
  - `pr_summary.pdf` — PR 摘要页面（单页紧凑版，便于截图贴到 PR 描述）

> [!IMPORTANT]
> HTML 报告**不是**附件，是邮件正文。图片通过 CID 内联到正文中渲染（不是文件路径引用），其余数据文件作为附件供下载。
