---
name: check-pr-docs
description: "检查 torchtitan-npu PR 涉及的文档、示例命令、依赖声明、版本兼容信息和 CI 安装路径。准备、修改或 review PR 且改动涉及公开行为、模型支持、配置、训练或测试入口、依赖或软件栈，或用户明确要求检查文档遗漏时使用。默认只读；用户明确要求同步修改时才编辑文件。纯内部重构、普通新增测试且公开契约不变，或仅推送已审查且未变化的内容时不自动触发。不负责通用代码审查、精度性能结论、提交推送或远端 PR 操作。"
---

# 检查 PR 文档与依赖

根据目标 PR 的实际变化，找出需要同步的用户文档、依赖声明、安装说明和兼容性信息。
默认只读，先报告需要同步的内容和依据。只有用户明确要求同步修改文档或依赖时，才对
证据充分的必需项做最小修改；已有授权继续有效，不逐项重复询问。不要按路径表重写文档，
也不要顺手修复无关的旧问题。只读模式不编辑文件、不运行可能改文件的检查；仅准备、
修改或 review PR 不等于授权修改文档或依赖。仅推送已审查且未变化的内容时，不重新触发
同步或额外检查。

与 `write-torchtitan-npu-docs` 配合时，本技能确定文档/依赖影响范围和当前执行模式，写作
技能负责具体文档的事实核对与表达规范，两者都不扩大用户授权。只读审阅或单篇文档任务
发现范围外的下游影响时，先报告，不顺带修改。用户已要求同步相关文档或依赖时，先说明
受影响范围，再完成授权范围内的最小修改；「下游先报告」不表示要逐篇重新申请许可。

## 开始前要读的文件

1. 先读取适用的 `AGENTS.md`，再读取 `README.md`、
   `.gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md` 和 `pyproject.toml`，
   以及本次改动涉及的 docs、requirements 和 CI 文件。撰写文档时遵循
   [write-torchtitan-npu-docs](../write-torchtitan-npu-docs/SKILL.md) 的写作约定和
   `docs/AGENTS.md` 的文档归属。
2. 明确 diff 范围后读取 [references/repo-impact-map.md](references/repo-impact-map.md)，
   只使用与本次变更相关的映射。候选路径可能过期；使用前重读对应版本的文件内容和
   调用关系，不能仅凭路径存在下结论。
3. 根据用户提供的 PR 元数据，或通过当前 Git 分支、上游分支和合并基点确认 PR 的基准
   提交与当前提交；文件不存在时跳过，不从其他分支补齐规则或 Skill。

## 职责边界

- 检查并同步 PR 涉及的用户文档、依赖声明和兼容性说明。若 PR 修改了依赖
  版本或上游基线，还要核对安装文档和 CI 脚本中的对应版本、tag 和 hash。
- 识别并按任务范围执行文档/依赖同步所需的最小相关检查；代码正确性审查、数值/性能
  结论、超出当前任务要求的行为测试和远端 PR 操作不属于本技能。完成同步后把最终
  diff、测试需求和 PR 模板素材交给相应流程。
- 不发布评论、不更新远端 PR，也不 stage、commit 或 push。
- 不修改生产代码来迎合文档。如果源码行为与 PR 意图冲突，报告冲突并转代码修复，
  不把当前偶然实现写成承诺。
- 不自动选择「最新」的 CANN、driver、PyTorch、torch_npu、torchtitan、Triton、
  Docker 镜像或发布版本。

## 工作流

### 1. 确定 PR 范围和修复基线

先确认目标仓库、目标分支、PR 状态和 base/head；已安装 `gitcode-pr` 时可用其读取元数据，
否则使用用户给出的信息或 GitCode 页面/API。不要默认 fork 的 `origin/master` 就是目标仓
最新代码。用户明确指定修复仓库或基线时按其要求，并记录与 PR 目标仓的差异。

对仍待合并的 GitCode PR，可替换下面三个值后获取目标分支和 PR head；每次 fetch 成功后
立即保存 SHA，任一步失败就停止这组命令，不能继续使用旧的 `FETCH_HEAD`：

```bash
pr_remote='<目标仓 remote>'
pr_target='<目标分支>'
pr_number='<PR 编号>'
git fetch --no-tags "$pr_remote" "$pr_target" &&
target_sha=$(git rev-parse FETCH_HEAD) &&
git fetch --no-tags "$pr_remote" "refs/merge-requests/$pr_number/head" &&
head_sha=$(git rev-parse FETCH_HEAD) &&
git rev-parse "$target_sha" "$head_sha" &&
git merge-base "$target_sha" "$head_sha" &&
git diff --name-status -M "$target_sha...$head_sha" &&
git diff --stat -M "$target_sha...$head_sha"
```

没有远端 PR 时，结合当前分支、目标分支和 merge-base 确认范围；禁止默认用 `HEAD~1`
代表整个 PR。历史文件用 `git show <commit>:<path>`、`git grep <pattern> <commit>` 读取。

历史 PR 分开处理取证与修复：

- 用当时的 base/head、merge-base 和最终 diff 判断遗漏；已合入 PR 不用最新目标分支
  与旧 head 的三点 diff 重建历史，也不以后来的实现替代当时的行为。
- 修复前先 fetch 目标仓最新 `master`（用户指定其他分支时用指定分支），记录修复基线
  SHA。在该版本重新确认问题是否仍存在、路径是否迁移、实现是否被替换。
- 只把仍存在且证据充分的问题修到包含该基线的工作区；已经修复的不重复改，已关闭且
  未合入的功能不补进当前文档。无法刷新或缺少历史证据时，明确标记未核实。

开始前记录：

- 目标仓、PR 状态、历史或当前 base/head、merge-base，以及修复基线 SHA；
- `git status --short` 中 staged、unstaged 和 untracked 文件；
- PR 已有 commits 与尚未提交但明确属于该 PR 的本地改动；
- 用户现有的无关 dirty 文件。

查看 name-status、stat 和完整 diff，并启用 rename 检测。只把确认属于目标 PR 的内容
放入审计范围，保留所有无关工作区改动。

### 2. 看清行为上改了什么

从审计范围对应的源码、测试和最终配置确认行为，不只复述 PR 描述。逐项识别：

- 公开 CLI/config/env 名称、默认值、错误语义和启动命令；
- 模型、特性、并行、硬件、checkpoint 和数据格式的支持范围；
- 安装步骤、训练/测试入口、输出路径和用户排障流程；
- 新增、删除或改变用途的第三方包、外部工具与软件栈；
- Python/CANN/PyTorch/torch_npu/torchtitan/Triton、包版本和 CI pin。

把文档承诺与 PR 意图、实际调用顺序对照；行为与意图冲突时报告冲突，不把错误行为固化
成文档承诺。本仓搜不到配置字段时，沿固定版本的上游基类、配置构建和参数透传路径核对，
不要直接判定字段无效。

按第 8 步的表头建立影响清单。仅内部重构且对外行为不变时，记「无需同步」及理由，
不要为了勾选 PR 清单制造改动。搜索引用时使用 `rg --hidden`；隐藏目录、忽略规则和
import 名与发行包名的区别见影响地图。

### 3. 区分本 PR 问题与既有漂移

对每个疑似不一致同时查看目标分支与 PR diff，按下表定级：

| 结论 | 典型情况 | 动作 |
| --- | --- | --- |
| 必须同步 | PR 改名 flag/config、改变默认命令或路径、增加必需依赖、更新已确定的版本；删除或下线用户可见入口或实现；新增或修改行含失效引用 | 同步仍引用该事实的有效文档、声明或安装命令，包括未出现在 diff 中的说明；是否写入按当前模式决定 |
| 条件性同步 | README 支持矩阵、硬件/checkpoint 兼容、性能数据、依赖删除或大版本升级 | 取得与具体声明匹配的证据后修改，否则列为待决 |
| 无需同步 | override 内部重构、等价 kernel 优化、普通新增 unit test 且公开契约不变 | 记录理由，不制造文档 diff |
| 既有漂移 | 不一致在 merge-base 已存在，与本 PR 无关且不在其新增或修改行上 | 单独报告，不顺手修复；用户要求全量审计时才纳入 |
| 并行冲突 | 最新目标分支与 PR 各自有效，合并后出现重复声明或配置冲突 | 记录双方 SHA 和失败位置；属于同步范围的修复后重新检查，其余交代码修复 |

本 PR 新增或修改行中的过期命令、错误默认值和失效引用，即使根因来自历史改动，也按
「必须同步」处理。文档在 merge-base 与实现一致、因本 PR 行为变化而失效，也属于本 PR
必须同步的内容，不是「既有漂移」，不能因该文档不在 diff 中而跳过。历史发现是否需要
修复，仍以第 1 步对最新基线的复核为准。需要区分历史遗漏、当前修复状态或不同依赖环境
时，参照 [references/cases.md](references/cases.md) 中的案例，不将旧结论直接套用到当前版本。

不要修改 README 新闻、Roadmap、性能表、硬件支持承诺、Docker image tag 或外部 URL，
除非目标 PR 本身提供相应来源和验证。

### 4. 同步用户文档

以下同步项在只读模式下作为建议输出，在已授权的修改模式下写入文件。

按影响地图找到负责该事实的正式指南，先更新该指南，再更新引用同一事实的概览或示例：

- 名称、flag、默认值、路径或命令变化：同步所有会被用户复制执行的有效示例；
- 新公开能力或支持范围：更新已有的 feature guide 或实现目录 README。源码可以证明
  入口、默认值与显式限制；硬件、并行、checkpoint 兼容或「已验证」声明需要相应验证
  依据，精度和性能需要对应运行记录。注册或测试文件存在不等于运行通过；
- 安装或全局默认入口变化：检查 installation、quickstart 和 README；
- 测试 runner、层级、参数或产物变化：核对 `.ci/*.sh`，同步测试调用说明，必要时更新
  PR 模板；新增普通 test node 本身不需要文档。只修有直接证据的安装命令或声明，
  测试调度、退出码和执行逻辑问题交对应代码修复流程；
- 上游或软件栈兼容变化：更新 `docs/user-guides/installation.md` 的软件配套表、实际
  CI pin 及必要镜像说明。

新文档按项目约定优先中文；维护现有文档时沿用其语言，不做无关的整篇翻译。命令、字段名
和路径必须来自修复基线及本次待提交改动。优先链接正式章节，不在 README、quickstart
和 feature guide 重复完整说明。普通功能 PR 不添加新闻、发布日期或路线图。

### 5. 同步依赖与兼容声明

先确认依赖的使用方式，再提出或按授权修改已有声明体系：

- 生产路径的无条件 import：检查 `requirements.txt` 与
  `pyproject.toml` 中 `[project]` 的 `dependencies` 安装承诺；
- 可选特性：沿用已有 optional/feature 安装方式。改成根依赖时核对实际使用范围和安装
  文档；无直接 import 不足以判定依赖多余，还要检查动态加载、插件、外部命令及传递依赖。
  区分不同发行包，不能把 `torchao` 已声明当作 `torchao-npu` 已安装；
- 测试或开发工具：检查 `requirements-dev.txt`，并核对 `.ci/setup_torchtitan.sh`、
  `.ci/common.sh`、各 `.ci/*_test.sh` entrypoint、`.ci/gitcode.dockerfile`，以及
  `.gitcode/` 下 workflow 与包装脚本的实际安装和调用路径；
- 文档构建 plugin：只有构建配置或插件真的变化时才改
  `.github/mkdocs/requirements_gh_pages.txt`，普通 Markdown 修改不碰；
- Skill 私有工具：放在该 Skill 自己的依赖中，不污染根依赖；
- CANN、driver、编译器或源码安装扩展：更新安装/feature guide 和已存在的 CI pin，
  不把系统依赖伪装成普通 PyPI 包。

`requirements.txt` 的环境 pin、`pyproject.toml` 的包元数据、CI 镜像和文档验证版本
可以有意不同。核对各自承诺，不要求版本号逐字一致。正式版与 daily/nightly、主次版本
变化时复核软件配套表的对应行和脚注；判定依据见影响地图。

安装命令使用 `--no-deps` 时，核对该包所需依赖是否由镜像、其他安装步骤或声明文件提供。
依据目标版本的包元数据确认缺口，不因该参数存在就判错，也不凭历史评论添加传递依赖。

新增依赖前确认真实执行路径和最小必要版本；删除依赖前搜索生产、scripts、tests、docs、
CI、动态 import 和插件注册。版本升级、ABI/硬件兼容或新 index 缺少测试与来源证据时，
把该项标为待决，不猜版本。

软件包版本、徽章和发布记录只在明确的发布 PR 中修改；普通功能 PR 只报告已有差异。

### 6. 只改必须改的地方

只读模式仅报告影响清单和最小修改建议。用户已明确授权同步修改时，对「必须同步」且
证据充分的项目直接修改，不再停下来询问是否要修每个文档。
保持现有结构、措辞和依赖组织方式，不新建 lock/constraint、依赖管理框架或重复指南。

如果某项需要产品支持承诺、兼容矩阵选择、外部镜像更新或无法从仓库确认的版本决策，
继续完成其他确定项，并把该项作为具体待决问题报告。

### 7. 自查

按实际改动选择最小验证。只读审阅时仅记录会修改文件、安装依赖或执行行为测试的命令与
前置条件，不在原工作区运行；无副作用的 diff、路径与语法检查可按任务范围执行。

按实际改动叠加相关检查，不要求每项改动都跑构建或行为测试：

| 改动 | 检查 |
| --- | --- |
| 所有文档/依赖同步 | 最终 diff、旧/新名称和版本的剩余引用；相对链接的文件及标题锚点 |
| TOML/YAML 声明、依赖或安装配置 | 解析改动文件；仍待合并的 PR 还要检查与最新目标分支合并后的相关配置 |
| shell 安装或 CI 命令 | `bash -n`；沿 workflow、包装脚本和安装步骤核对调用关系 |
| 文档构建配置或插件 | 依赖已就绪时运行 `mkdocs build --config-file .github/mkdocs/mkdocs.yml --site-dir "$(mktemp -d)"`，避免把产物留在仓库；不执行 `gh-deploy` |

**合并结果检查：**使用第 1 步记录的最新目标分支 SHA 与 head SHA，在不修改原工作区和
索引的前提下生成合并树，再读取其中的配置。不能把三参数 `git merge-tree` 的文本输出
当成文件树，也不能解析原工作区文件后声称合并结果通过。预览 merge ref 只有在确认对应
当前双方 SHA 后才能使用。具体命令如下（需 Git 支持 `--write-tree`、Python 3.11+）：

```bash
merge_tree=$(git merge-tree --write-tree --no-messages "$target_sha" "$head_sha")
```

仅在该命令退出码为 0 时，读取合并树中的实际文件。例如检查 `pyproject.toml`：

```bash
python - "$merge_tree" <<'PY'
import subprocess
import sys
import tomllib

content = subprocess.check_output(["git", "show", f"{sys.argv[1]}:pyproject.toml"])
tomllib.loads(content.decode("utf-8"))
print("merged pyproject.toml: OK")
PY
```

同样读取受影响的其他 TOML/YAML，使用对应解析器检查；YAML 还要拒绝重复键，不能只依赖
默认 `yaml.safe_load`。合并命令退出码 1 表示合并冲突，其他非零值表示命令或环境失败，
分别报告，不把失败后的输出当作有效合并；工具不支持时在临时副本检查或标记未验证。
SHA 预览不包含本地未提交修改；有此类修改时须在临时副本纳入后重查，不能声称最终内容
已验证。历史 PR 已合入的，按当前修复基线检查，无需再次合入旧 head。

按任务范围和适用 `AGENTS.md` 执行 Lint；用户要求只读或明确跳过时记录未执行。只有
固定上游就绪后 `python -m pre_commit run --all-files --show-diff-on-failure` 完整通过
才算 Lint 通过，文件级结果不能替代。`--all-files` 不覆盖 untracked 文件，可补充
`--files <paths>`，不要为了覆盖新文件而 stage。

依赖解析、wheel/Docker 构建、网络安装、外部 URL 和行为测试按任务与环境选择；未执行项
写清原因。CI 结论还须核对该次运行实际加载的 workflow 版本及执行脚本的提交，不能由
某次流水线通过推断新 workflow 或另一套依赖配置已验证。

验证后再次检查 `git status`，确认只新增了本次必要的文档/依赖修改，没有覆盖用户工作区
内容，也没有 stage 任何文件。

### 8. 输出结论

先给出审计范围与基线，再给出影响清单；没有发现时直接说明，无需填空表：

| 变更证据 | 受影响的文档或 CI | 事实来源 | 结论 | 理由 |
| --- | --- | --- | --- | --- |

结论使用第 3 步的分类；证据不足时填「待决」并说明缺少什么。历史 PR 另行标明
「历史发现／最新基线仍存在／已修复／不适用／未核实」，不要把历史问题直接列成当前缺陷。

按实际情况补充：

- 目标仓、base/head、merge-base、修复基线 SHA 和包含的本地改动；
- 主动修改的文件，以及每项对应的代码/配置/依赖证据；
- 明确判定无需修改的文档或依赖及理由；
- 单独列出的既有漂移与待决兼容选择；
- 实际运行的验证、结果和未验证项；
- 应交给最终代码审查和 PR 发布流程的 checklist/测试说明素材。

只有证据完整时才说明「相关文档已更新」或「依赖已同步」。后续代码修改若再次改变公开
契约，重新运行本技能。
