# 提交前代码自审清单

按阻塞、重要、建议三个级别输出；每条给出文件与行号、证据、影响和最小修正。

## 1. 规则和范围

- 读取根 `AGENTS.md`、`.agents/AGENTS.md` 和改动路径上所有 `AGENTS.md`，最近的
  目录规则优先。
- 读取 PR 模板、`check-pr-docs`、`developer-tests-review`、`numerical_alignment.md`
  和 `metrics_and_debugging.md` 的相关部分。
- diff 基于 merge-base，不包含无关提交、生成物、日志、密钥、`_remote` 或远程
  skill 软链；更新已有 PR 时先读取实际 base/head，不用仓库默认分支。
- 代码增删合计不超过 1000 行（排除纯文档、生成数据和测试数据）；超出时已拆分，
  或已取得明确同意。
- 没有混入无关的格式化、重命名或重构。

## 2. AGENTS.md 合规

| 改动路径 | 重点 |
| --- | --- |
| 根目录、`.agents/**` | 仓库入口、代码归属、skill、测试和文档规则 |
| `torchtitan_npu/**` | 目标结构、代码放置、依赖方向、注入顺序 |
| `patches/**` | 临时补丁、`Upstream-PR`、`apply()`、显式导入 |
| `override/**` | `override.imports` 显式启用，不自动导入 |
| `models/**`、`ops/**` | 模型和算子边界、checkpoint 适配、注册 |
| `tests/**` | UT 和 ST 分层、runner、格式、数值验证 |
| `docs/**` | 文档归属、命令依据、支持范围、验证状态 |
| `scripts/**`、`experiments/**`、`examples/**` | 脚本入口、实验隔离、示例依赖 |

- 代码放在最通用的适用位置；模型无关的能力不要放进模型专属文件。
- `patches/` 不得新增对 `extensions/` 的依赖；`extensions/` 不得新增对
  `models/`、`override/` 或 `ops/` 的依赖。
- `__init__.py` 的自动导入顺序为 `patches → config → extensions → ops`。
- 临时 patch 要记录 `Upstream-PR`；monkey-patch 使用完整点路径，并收敛到 `apply()`。
- 命名要准确、有描述性；生产代码不要使用 `toy`、`test`、`temp`；计数使用 `num_`
  前缀。
- 用户可见的错误使用 `ValueError`；内部不变量使用 `assert`；分布式代码中显式验证
  mesh、placement 和配置。
- 修改共享代码时，检查所有调用点，包括 DeepSeek-V4、override 和
  patch 注册入口。

## 3. 精简和注释

- 复用现有实现，不新增重复的 wrapper、工具函数或抽象。
- 删除未使用的导入和变量、死代码、注释掉的代码、调试输出、临时开关和兼容垫片。
- 保持最小改动，不顺手修复无关问题，也不为单次使用增加配置、泛化或依赖。
- 检查是否有重复计算、重复 IO、重复拷贝和不必要的同步或等待；同时检查异常路径和
  资源清理。
- 注释只保留维度语义、并行梯度 placement、workaround 原因、上游 PR 引用和已知限制。
- 删除复述代码、历史过程和个人备注的注释；TODO 要说明限制；注释用英文，文档用
  中文。

## 4. 正确性和兼容性

- 检查空输入、最小 shape、单 rank 和多 rank、不同 mesh 和 placement 等边界。
- 检查配置字段名、默认值、命令行、公开 API 和序列化兼容性。
- 检查 checkpoint 和 state_dict 的键名、shape、dtype、layout、加载和保存路径。
- 检查随机性、确定性、归约顺序、混合精度和 HF32 设置。
- 检查是否引入额外拷贝、同步、通信、编译失败或显存峰值。
- 检查已验证的代码路径是否可能静默失效；存疑时先询问。
- 创建新 PR 前，按 `numerical_alignment.md` 第 4 节和 diff 判断并分类，不询问用户；
  更新已有 PR 只更新本次修改涉及的描述部分，补充后重新压缩整段描述，删除过期或
  重复内容，不改标题、draft 或 review 状态。
- 新建 PR 时，无影响直接创建可审查 PR，描述仍写跳过依据；有影响只创建草稿 PR，在
  `精度和性能验证结果` 段标注“精度验证待补充”，不请求 review，由用户自行补充验证
  后转为可审查 PR。未验证前不得创建可审查 PR；判断不清时按有影响处理。
- 修改前后精度对比优先用 `scripts/numerical_alignment/loss_compare.py`（两个 Git
  ref）；只有结果不一致或需要定位到算子/模块时，才用 `numerics-debugging` 的激活
  捕获与 diff（两次运行同 dtype/seed，完成后还原临时 patch）。

## 5. 测试

按 `developer-tests-review` 检查：

- UT 应保护正向功能和独立 expected，不要复制实现作为 expected。
- ST 应进入真实模型、override、并行和编译组合；CPU UT 不计入 ST 覆盖。
- 测试格式按 `references/format-review.md` 检查；格式检查不能代替覆盖结论。
- `review测试` 保持只读，不导入、收集或执行测试，并写明“测试执行：未执行
  （仅静态审查）”。
- `更新测试` 要实际执行相关用例并迭代到通过；未执行不得写成通过。
- 结论只使用：`可以合入`、`补充测试后合入`、`暂停并澄清`、`不适用`。
- 除非用户授权，不要修改生产实现来迁就测试；第三方改动不影响本仓适配时，结论写
  “不适用”。

## 6. 文档和依赖

按 `check-pr-docs` 检查：

- 检查公开行为、模型支持、配置、训练和测试入口、依赖或软件栈的变化是否影响
  README、docs、示例、安装说明或 CI。
- 影响分类包括：必须同步、条件性同步、无需同步、既有漂移、并行冲突。
- 依赖变更时，核对 `requirements.txt`、`pyproject.toml`、`requirements-dev.txt`、
  `.ci/` 和 `.gitcode/`。
- 默认只读；未获授权不修改文档或依赖。需要修改时，遵循
  `write-torchtitan-npu-docs`，并确保命令、路径、配置键和版本都有来源。
- 不要宣称未验证的结论，不要自动选择最新版本；文档与实现冲突时先报告，不要修改
  生产代码来迁就文档。

## 7. 隐私和敏感信息

- 检查标题、描述、评论、commit message、日志、trace、截图和证据链接。
- 不得包含个人隐私信息、token 或密钥、内网 IP、主机名、内部 URL、客户数据、
  未公开数据，以及带用户名的本地路径。
- 使用 `<user>`、`<host>`、`<token>` 等占位符或脱敏链接；测试数据使用合成或
  脱敏数据。
- 必须提供敏感原文时，改用仓库认可的私下渠道；同时检查临时 payload、日志和截图
  没有被提交。
- 用 `git log --format='%h %an <%ae>' <merge_base>..HEAD` 检查提交作者信息。

## 8. 输出

- **阻塞**：正确性、兼容性、违反规则、测试失败、精度验证缺失（可审查 PR）、隐私泄露
  或未授权改动；必须修复。
- **重要**：其他验证缺口、文档或依赖待同步、性能回退风险；修复或明确说明。
- **建议**：可以简化、可读性不足或注释冗余；不阻塞。

最后给出自审结论、已修复项、未解决风险和待补验证。不要只写“已 review”。
