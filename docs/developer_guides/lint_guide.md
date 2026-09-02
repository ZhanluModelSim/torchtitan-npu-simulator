# Lint 指南

本指南只覆盖代码静态检查、格式检查和开源合规检查，不包含模型训练、模型加载、NPU 算子验证或功能测试。
Lint 可以在 CPU 环境执行，不需要 CANN、HCCL、可用的 Ascend NPU 或完整的运行时依赖。

## 检查范围

仓库通过 pre-commit 统一执行检查，具体配置见
[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml)。不同文件类型会触发不同的 hook：

| 修改内容 | 主要检查 | 是否需要 NPU |
| --- | --- | --- |
| Markdown、YAML、JSON 和配置文件 | 文件格式、尾随空格、冲突标记、拼写和 OAT 合规性 | 否 |
| Python | Ruff 格式与规则检查、Pyrefly 类型检查 | 否 |
| C/C++ | clang-format 格式检查 | 否 |

## 环境与依赖

本地建议使用 Python 3.12，与仓库 CI 的要求保持一致。Windows 环境需要使用 WSL 发行版或
Git Bash，以便执行 OAT hook 使用的 `bash scripts/oat_check.sh`。

### 最小安装

只检查 Markdown、配置文件或其他非 Python 文件时，安装 pre-commit 即可：

```bash
python -m pip install pre-commit
```

修改 Python 文件时，还需要 Pyrefly。可以只安装相关工具：

```bash
python -m pip install pre-commit pyrefly
```

也可以安装仓库提供的开发依赖。该依赖集合用于 lint 和开发测试，不包含 `torch`、`torch_npu`、
`torchtitan` 或 Triton 等训练运行时依赖：

```bash
python -m pip install -r requirements-dev.txt
```

`scripts/oat_check.sh` 在检测到本地没有 `oat-py` 时会尝试自动安装 `oat-py>=1.0.1`。
`requirements.txt` 只在需要进行模型运行、训练或 NPU 功能验证时安装，本指南不要求安装它。

## 执行检查

### 检查本次修改

提交 PR 前，优先对本次修改的文件执行增量检查：

```bash
python -m pre_commit run --files <file1> <file2> --show-diff-on-failure
```

例如：

```bash
python -m pre_commit run --files docs/developer_guides/lint_guide.md --show-diff-on-failure
```

部分 hook（例如 Ruff 和 clang-format）可能会自动修复文件。检查完成后，应查看工作区差异并重新运行命令。

### 检查整个仓库

需要复现仓库级 lint 时执行：

```bash
python -m pre_commit run --all-files --show-diff-on-failure
```

该命令仍然只进行静态检查，不会启动训练。

### 安装 Git hook

提交前必须安装 Git hook。安装后，每次提交会自动对增量文件执行检查：

```bash
python -m pre_commit install
```

## 检查失败时

- 根据 hook 输出修复格式、拼写、类型或合规问题，然后重新运行同一条命令。
- 如果提示 `pyrefly` 不存在，安装 `pyrefly` 或完整的 `requirements-dev.txt`。
- 如果 Windows shell 找不到 `bash`，切换到 Git Bash 或已安装 Linux 发行版的 WSL。
- Lint 无法替代 NPU 功能验证。涉及 `torch_npu`、NPU 算子、分布式并行、checkpoint 或模型训练行为的改动，
  仍需在具备相应设备和软件栈的环境中另行验证。

相关配置和开发依赖如下：

- [`requirements-dev.txt`](../../requirements-dev.txt)
- [`pyproject.toml`](../../pyproject.toml)
