# torchtitan-npu PR 影响地图

本文件列出当前仓库中需要检查的文档、依赖和 CI 路径，判断流程见 SKILL.md。路径只帮助
定位，具体行为须重读对应版本的文件内容和调用关系，不能仅凭路径存在下结论。

历史 PR 的 base/head 用于判断当时的改动和遗漏；修复前按主技能锁定最新目标分支提交，
再检查遗漏是否仍存在、入口是否改名或迁移。修复以该最新提交为基线，不把历史说明直接
复制到当前文档，也不以后来的实现替代历史取证。命中路径不等于必须修改其中的全部文档。

本仓的 CI、workflow 与 Agent 规则位于 `.ci/`、`.github/`、`.gitcode/`、`.agents/` 等隐藏
目录。从仓库根目录搜索时使用 `rg --hidden`，避免漏掉这些目录；该选项仍遵守 ignore
规则，被忽略但与任务相关的文件需单独核对，不能把搜索无结果当作不存在引用的证明。

## 文档归属

文档归属按 `docs/AGENTS.md` 的职责划分确定。下表补充源码变化对应的文档入口。

| 变更语义 | 定位起点 | 更新条件 |
| --- | --- | --- |
| 顶层公开能力、模型/并行/算子支持范围 | `README.md` 的支持概览与项目结构 | 公开能力确有变化，并按下述声明类型核对依据；内部重构不扩写支持范围 |
| 安装步骤与软件栈兼容 | `docs/user-guides/installation.md` | 用户安装命令、软件配套表或兼容承诺变化 |
| 启动入口、`MODULE`/`CONFIG`/`NGPU`、数据、tokenizer 与环境变量 | `docs/user-guides/quickstart.md`、`scripts/run_train.sh`、`scripts/run_train_multinodes.sh`、`examples/AGENTS.md` 与 `examples/<model>/` 下的 wrapper 及 `readme.md` | 区分通用启动脚本、模型配置与包装脚本的默认值和参数透传，同步引用该参数或入口的说明 |
| 某项特性的行为、启用方式与限制 | `docs/feature_guides/` 中对应指南，用 `rg -l <特性关键字> docs/feature_guides` 定位 | 启用方式、配置键、支持范围、shape/dtype/并行约束或验证结论变化 |
| 开发者流程：配置扩展、checkpoint、指标与调试、lint、UT 架构、数值对齐 | `docs/developer_guides/` 中对应指南 | 开发者可见的 schema、流程、入口或判定标准变化 |
| 目录职责、自动导入顺序与依赖方向 | `docs/architecture.md`，以及目标目录的 `AGENTS.md` | 包布局、导入顺序或分层边界变化 |
| `torchtitan_npu/override/**` 的注册与启用契约 | `torchtitan_npu/override/README.md`、README 项目结构 | 对 override 作者或用户可见的注册/启用方式变化 |
| 临时上游 patch 的适用条件和生命周期 | `torchtitan_npu/patches/torchtitan/README.md` | patch 目标、上游状态或移除条件变化 |
| 测试分层、marker、runner、CI 命令或产物 | `tests/AGENTS.md`、`tests/integration_tests/README.md`、`docs/developer_guides/unit-test-architecture.md`、`.ci/*.sh`、`.gitcode/workflows/llt_action.yml`（UT）、`.gitcode/workflows/codecheck_action.yml`（代码检查） | 开发者或用户的执行方式变化；普通新增 test node 不写文档 |
| PR 类型、checklist 和验证要求 | `.gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md` | PR 工作流本身变化 |
| 新增仓库 Skill | `.agents/skills/<name>/`，以及 `.agents/AGENTS.md` 的「标准开发流程」 | Skill 成为标准流程的一环时，在 `.agents/AGENTS.md` 写明触发时机 |

入口、默认值和显式约束以注册、配置和实现为依据；注册或测试文件存在不代表运行验证
通过。硬件、dtype、并行和 checkpoint 支持声明须说明依据及适用范围；数值、性能或
「已验证」结论须有对应环境的运行记录，不能由某一个入口或测试推广到全部组合。

优先更新按文档归属确定的主要指南，再更新引用同一参数、版本或使用方式的概览与示例。
没有对应指南且新行为确实需要稳定的用户说明时，才按 `docs/AGENTS.md` 的归属新增文档。

## 依赖与兼容事实源

比较声明时使用规范化 distribution 名，不要假设它等于 import 名：`triton` 对应
`triton-ascend`，`yaml` 对应 `PyYAML`。

| 文件 | 当前职责 | 一致性要求 |
| --- | --- | --- |
| `requirements.txt` | 训练环境的精确 pin、extra index / find-links 与 `torchtitan==<version>` 基线 | 保留实际存在的 index、marker 与安装顺序；关键 pin 变化时检查使用该版本的安装命令、CI 和文档 |
| `pyproject.toml` | 包元数据、`requires-python >=3.12`、`[project]` 中的 `dependencies`、optional extras、pytest marker 与 pyrefly 配置 | 按 `pip install -e .` 的公开承诺判断是否同步，不把环境精确 pin 复制进来 |
| `requirements-dev.txt` | unit test、lint 与类型检查的开发依赖 | 新增 test/dev import 必须能由 `.ci` 的实际安装路径获得 |
| `.ci/common.sh` | smoke 与 integration 共享的 CANN 环境、解释器 shim 和 torchtitan checkout | 改 torchtitan 基线或 CI 环境变量时先看这里，再看各 entrypoint |
| `.ci/unit_test.sh`、`.ci/smoke_test.sh`、`.ci/integration_test.sh` | UT、smoke 与 integration 三个 CI job 对应的依赖安装与测试命令 | `.ci/unit_test.sh` 自带安装逻辑并按名硬编码部分 wheel；smoke 与 integration 经由 `.ci/common.sh`。改这些脚本是否真的改变流水线行为，见下方入口说明 |
| `.gitcode/scripts/torchtitan-npu_ut.sh`、`.gitcode/scripts/torchtitan-npu_smoke.sh` | 仓内跟踪的 UT 与 smoke 包装脚本，分别调用 `.ci/unit_test.sh` 和 `.ci/smoke_test.sh` | 改测试入口、环境变量或退出码处理时一并核对；流水线实际执行的是 OBS 同名副本，见下方入口说明 |
| `.ci/gitcode.dockerfile` | CI 镜像的 Python 3.12 与 CANN 基础环境 | 先确认镜像与脚本的分工；镜像已固定的 wheel 不在脚本里重装 |
| `.gitcode/workflows/llt_action.yml` | PR 流水线调用的 UT job 定义 | 核对传入镜像、checkout ref 与 OBS 包装脚本入口；实际执行副本见下方入口说明 |
| `.gitcode/workflows/codecheck_action.yml` | 调用 `.ci/setup_torchtitan.sh` 在 `/tmp/torchtitan` 准备固定上游，再执行 precommit | 核对 setup 调用与 `pyproject.toml` 中 pyrefly `search-path` 的 `/tmp/torchtitan` 一致；无需 NPU 设备不等于无需 Python 依赖 |
| `experiments/torchao-npu/pyproject.toml`、`torchtitan_npu/extensions/adaptor/torchao/converter.py` | 可选 TorchAO-NPU 包依赖与训练侧集成入口 | 区分根依赖 `torchao` 和可选适配包 `torchao-npu`，核对安装指南、quickstart 与量化包装脚本的实际启用方式 |
| `.github/mkdocs/mkdocs.yml`、`.github/mkdocs/requirements_gh_pages.txt`、`.github/workflows/deploy-docs.yml` | 文档构建配置、构建依赖与 master 发布入口 | `docs_dir` 指向仓库 `docs/`；验证时只做本地 build，不执行 `gh-deploy` |
| `docs/user-guides/installation.md` | 当前分支的软件配套表与可复制安装顺序 | 核对配套表、版本声明和安装命令；可选包变为必需依赖时核对可选安装章节；兼容或已验证结论须有对应依据，不能由声明变化推断验证完成 |
| Skill 内 `requirements.txt` | 仅该 Skill 脚本需要的第三方包 | 没有私有脚本就不要创建，也不污染根依赖 |

`.gitcode/workflows/PR-pipeline_torchtitan-npu.yml` 的 integration job 直接执行仓内
`.ci/integration_test.sh`，codecheck 直接 source `.ci/setup_torchtitan.sh`。UT 与 smoke job
先把 OBS 上的 `torchtitan-npu_ut.sh`、`torchtitan-npu_smoke.sh` 下载到 workspace 根目录，
再执行根目录的那份副本，而不是 `.gitcode/scripts/` 下的同名脚本。

审查 CI 时分别核对平台加载的 workflow 定义版本和 job 检出的代码版本，再沿执行命令与
包装脚本追踪到最终入口。当前测试 job 声明检出 PR 的 `merge_commit_sha`；这不能证明
平台加载了该提交中的 workflow 定义。改 `.gitcode/workflows/` 时，须从平台配置或该次
运行记录确认定义来源；若确认读取目标分支版本，该次运行不能验证 PR 中的新编排，需在
合入后或明确使用新定义的运行中确认，不能一概认定所有 PR 都无法验证。

仓内包装脚本调用 `.ci/` 的链路可由源码确认；OBS 副本是否同步需结合该次 CI 记录核对。
同时记录实际 checkout SHA、镜像与安装版本。缺少证据时只标记对应链路未验证，不能
仅凭仓内修改断言流水线已覆盖新依赖。

根 `requirements.txt` 已声明 `torchao`；`experiments/torchao-npu` 是另行安装或通过源码
路径加载的可选 `torchao-npu` 包，导入名为 `torchao_npu`。
`torchtitan_npu/extensions/torchtitan/trainer.py` 仅在启用量化训练时导入上述 converter。
分别核对两者的版本声明、安装方式和启用路径，不因可选适配包尚未安装就重复添加根依赖。
声明存在只证明依赖已列出，不证明已安装或可运行；验证可选路径时再按任务与环境执行，
不要求无条件导入可选包。

仓库当前没有统一 lock/constraint、`assets/version.txt` 或独立 versioning policy 文档。
不要从其他分支复制不存在的体系来完成普通 PR，也不要修改 CI 临时目录或生成物。

## 关键版本信息的同步范围

### torchtitan 基线

当前 `requirements.txt` 的 `torchtitan==<version>` 是唯一基线声明，不是 VCS/commit 依赖。
`.ci/setup_torchtitan.sh`、`.ci/common.sh` 和 `.ci/unit_test.sh` 都解析该行并推导 tag
`v<version>`；UT、smoke 与 integration 还允许 `TORCHTITAN_COMMIT` 覆盖。tag 与该环境变量
默认值是脚本派生值，不是需要手工同步的第二份 pin。

改动基线时核对：

- `requirements.txt` 的 `torchtitan==` 行；
- `docs/user-guides/installation.md` 中适用的软件配套行；
- `.ci/setup_torchtitan.sh`（checkout 到 `/tmp/torchtitan`，供 codecheck workflow 使用）
  与 `.ci/common.sh`、`.ci/unit_test.sh`（checkout 到 `third_party/torchtitan`）；
- `torchtitan_npu/patches/torchtitan/README.md` 和 `torchtitan_npu/override/README.md`
  中确实绑定该上游 API 的说明；
- `docs/community/governance.md` 的「版本基线」要求，及其指向 installation 版本配套表的
  锚点；根 `AGENTS.md` 与 `.agents/AGENTS.md` 中「从版本推导 tag」的机制描述。

### PyTorch、torch_npu、Python、CANN 和 Triton

核对 PyTorch、`torch_npu`、Python ABI、CANN 和 `triton-ascend` 的配套关系，包括
`requirements.txt` 的版本与 index、`pyproject.toml` 的 Python 下限、安装文档、CI shell
和 Docker 基础环境。软件配套表当前还包含 torchtitan-npu 版本和 torchtitan 分支；检查
受影响版本行的全部列及脚注，核实分支、tag 或版本号的实际含义，不按固定列数复制表格。
正式版与 daily/nightly 互换、主版本或次版本变化，至少列入配套表的条件性同步检查。
若 `(daily)` 等标注与同一安装路径中的正式版声明冲突，核实该行适用的软件栈；无法确定
时标为待决，不能靠改标签宣称兼容性已验证。

CI 与 `requirements.txt` 的部分版本和 CANN 路径有意不同：`.ci/unit_test.sh` 与
`.ci/common.sh` 都改装 nightly 的 `torchvision` 覆盖 requirements 中的正式版，两者的 CANN
入口也不同（前者 source Jenkins 目录下的 CANN，后者用
`/usr/local/Ascend/cann/set_env.sh`）。

CI 通过只能证明其实际软件栈下的检查结果，不能替代另一套 requirements 配置的验证。

仅当当前声明使用架构专属 wheel URL/hash 时才分别核对并保留，不预设存在两套 wheel/sha256，
也不自动选择最新构建或镜像。新增 index 时，index、package 声明和可复制安装命令必须一起
审计。

### package version

包版本有两处声明：`pyproject.toml` 的 `version`，以及 `README.md` 顶部写死版本号的 pypi
badge；两者当前并不一致，该差异在本 PR 之前已存在。仓库没有 CHANGELOG 或发布说明文件。
