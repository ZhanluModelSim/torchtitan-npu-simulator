# reproduce.json 编写指南

在生成 PDF 前，将本次验证的完整复现信息写入 `./numerical_report/reproduce.json`。**所有字段必须填入实际值，禁止使用占位符**。读者应能逐行复制粘贴执行。

同时，将 Step 0 中识别的类别 B（Infra）文件打包为 `infra_files.tar.gz`，放在 `./numerical_report/` 下：

```bash
tar czf ./numerical_report/infra_files.tar.gz -C "$REPO_ROOT" \
    <infra_file_1> <infra_file_2> ...
```

## JSON 格式示例

> [!NOTE]
> 以下仅为**格式示例**。实际操作时 `training_runs` 和 `compare_commands` 的内容取决于 Config 矩阵（可能包含 FSDP/TP/EP/CP/compile 等任意组合）。每条命令、每个路径都必须来自本次实际运行。

```json
{
  "timestamp": "2026-05-30T15:30:00+08:00",
  "baseline_branch": "<baseline_branch>",
  "baseline_commit": "<baseline_commit_hash>",
  "candidate_branch": "<candidate_branch>",
  "candidate_commit": "<candidate_commit_hash>",
  "model_name": "torchtitan_npu.models.deepseek_v32",
  "config_name": "deepseek_v32_smoketest_debug",
  "training_steps": 100,
  "parallelism_summary": "TP=1/EP=1 (FSDP), TP=2/EP=1 (TP)",
  "branch_diff": "分支关系 + 改动总结（见下方规则）",
  "infra_steps": [
    "# loss_compare.py 来自 premerge-accuracy-check 技能，用于精度对比",
    "# smoketest config，关闭 npu_rms_norm converter 以验证原生路径"
  ],
  "training_runs": [
    {
      "label": "Generate initial checkpoint (FSDP)",
      "branch": "<candidate_branch>",
      "command": "bash scripts/run_train.sh --module <model> --config <config> --training.steps 1 --debug.seed=42 --debug.deterministic --training.dataset_path=<data_path> --checkpoint.enable --checkpoint.no_load_only --checkpoint.initial_load_path None --checkpoint.last_save_in_hf --checkpoint.last_save_model_only 2>&1 | tee ./numerical_report/gen_ckpt_fsdp_train.log\nmkdir -p ./numerical_report/initial_ckpt_fsdp\nmv outputs/checkpoint/step-1 ./numerical_report/initial_ckpt_fsdp/",
      "train_log": "./numerical_report/gen_ckpt_fsdp_train.log"
    },
    {
      "label": "Baseline FSDP (load from checkpoint)",
      "branch": "<baseline_branch>",
      "command": "rm -rf outputs/checkpoint && bash scripts/run_train.sh --module <model> --config <config> --training.steps=<steps> --debug.seed=42 --debug.deterministic --training.dataset_path=<data_path> --checkpoint.enable --checkpoint.load_only --checkpoint.initial_load_path ./numerical_report/initial_ckpt_fsdp --checkpoint.initial_load_in_hf 2>&1 | tee ./numerical_report/baseline_fsdp_train.log",
      "train_log": "./numerical_report/baseline_fsdp_train.log"
    },
    {
      "label": "Candidate FSDP (load from checkpoint)",
      "branch": "<candidate_branch>",
      "command": "rm -rf outputs/checkpoint && bash scripts/run_train.sh --module <model> --config <config> --training.steps=<steps> --debug.seed=42 --debug.deterministic --training.dataset_path=<data_path> --checkpoint.enable --checkpoint.load_only --checkpoint.initial_load_path ./numerical_report/initial_ckpt_fsdp --checkpoint.initial_load_in_hf 2>&1 | tee ./numerical_report/candidate_fsdp_train.log",
      "train_log": "./numerical_report/candidate_fsdp_train.log"
    }
  ],
  "compare_commands": [
    {
      "label": "FSDP comparison",
      "command": "python .agents/skills/premerge-accuracy-check/scripts/loss_compare.py --baseline ... --candidate ... --output ./numerical_report/fsdp/"
    }
  ],
  "environment": {
    "npu_count": "16",
    "cann_version": "8.2.RC1",
    "torch_version": "2.7.0",
    "torch_npu_version": "2.7.0"
  }
}
```

## 写入规则

> [!IMPORTANT]
> - **禁止占位符。** 不允许出现 `<branch_a>`、`<config_name>`、`<log_file>` 等模板变量。
> - **`branch_diff` 必须包含两部分：分支关系 + 改动总结。** 不是 `git diff --stat` 原文。
>   1. **分支关系**：从 Step 0 审计结果中总结两个分支的拓扑关系。例如：
>      - "基线是候选的直接祖先，候选领先基线 3 commits"（线性）
>      - "两分支从 `<merge_base>` 分叉，基线领先 2 commits，候选领先 1 commit"（分叉）
>      - "基线落后 upstream 15 commits → 需要注意基线可能不是最新"（陈旧基线警告）
>   2. **改动总结**：用一两句话说明哪些文件改了、改了什么功能、影响范围。
> - **类别 B 文件通过 `infra_files.tar.gz` 分发，不手动复制。** `infra_steps` 中不再包含逐文件的 `cp` 命令。读者将邮件附件 `infra_files.tar.gz` 放到仓库根目录后 `tar xzf ./infra_files.tar.gz` 即可。
> - **`infra_steps` 改为解释性注释。** 每一条说明 `infra_files.tar.gz` 中包含的文件的**来源和意图**（如 `# loss_compare.py 来自 premerge-accuracy-check 技能`），而不再是 shell 命令。
> - **每条训练命令自包含。** 不写 "same args as step 3" 或 "apply same infra changes"。
> - **每个 config 的首个 training_run 是 checkpoint 生成步骤。** 训练流程为：先在一侧分支生成初始 checkpoint（`--training.steps 1 --checkpoint.last_save_in_hf --checkpoint.last_save_model_only`），再让两侧分支从同一 checkpoint 加载训练（`--checkpoint.load_only --checkpoint.initial_load_path ... --checkpoint.initial_load_in_hf`）。这确保两个分支起始权重完全一致。
> - **`compare_commands` 中的路径必须是实际的训练日志文件。** 不要写占位符。

## 写入后验证（Stranger Test）

写完 `reproduce.json` 后，**逐条检查**：

1. `./numerical_report/infra_files.tar.gz` 是否已创建？`tar tzf` 查看其内容，确认所有类别 B 文件都在其中。
2. 假设你是一个刚 clone 了仓库的新人，将 `infra_files.tar.gz` 放到仓库根目录后执行 `git checkout <commit> && tar xzf ./infra_files.tar.gz`，能否直接开始训练？
3. `training_runs` 中每个 config 的首条是否为 checkpoint 生成步骤？是否包含 `mv outputs/checkpoint/step-1`？
4. 后续基线/候选训练是否都加载了同一个 checkpoint（`--checkpoint.load_only --checkpoint.initial_load_path ./numerical_report/initial_ckpt_... --checkpoint.initial_load_in_hf`）？
5. `infra_steps` 中的每条注释是否说明了文件的来源和意图（而非无意义的 shell 命令）？
6. `branch_diff` 是否同时包含**分支关系**（拓扑关系、是否陈旧、merge base）和**改动总结**？是否中文可读？

如果任何一项未通过，修正后重新检查。
