---
name: torchtitan-npu-code-reviewer
description: "用于 torchtitan-npu 仓库的关键风险代码审查：本地 diff、指定 commit/分支区间或 PR/MR diff，以及 GitCode/OpenLibing CodeCheck 告警的修复与屏蔽取舍判断；当用户要求 review/检视/审查代码、检查本地修改、审查某几个 commit 或某个 PR、分析 CodeCheck 告警是否需要在 openlibing 屏蔽时使用。"
---

# torchtitan-npu 代码审查

对 torchtitan-npu 的改动做关键风险与代码质量审查，覆盖五类维度——专项规则、代码坏味道、设计与架构、上游交互、训练 infra 与 NPU 性能——并按「必须改 / 建议级」分级输出；同时给出 CodeCheck 告警的修复或屏蔽取舍。只报告明确、可验证的问题，不堆主观风格建议。

## 一、审查流程

1. 读取 `.agents/AGENTS.md`，确认本仓核心原则和开发 pipeline——**必须遵守**。
2. 确定审查范围：
   - 本地改动：`git diff --stat`、`git diff --cached --stat`，必要时查看具体 diff；
   - 指定 commit / 分支区间：`git diff <base>..<head>`；范围内含 merge commit 时，分别对照两个 parent 确认冲突解决没有丢失任一侧内容；
   - PR/MR：通过 `gitcode-pr` 获取文件列表和 diff。若缺少该 skill，先使用 `default-skills` 安装。
3. 按「二、审查维度」逐项审查：专项规则 → 代码坏味道 → 设计与架构 → 上游交互 → 训练 infra 与 NPU 性能。
4. 所有发现按「三、审查准则与分级」判断报不报、定严重级别，再按「六、输出格式」输出。

## 二、审查维度

五类维度逐项核对；每条发现都要按「三、审查准则与分级」过滤后才进报告。**判据用于定位问题；修复方向仅在非显然时标注**。

### （一）专项规则审查

按改动路径加载 `.agents/rules/*.md` 中匹配的规则文件（frontmatter 的 `paths` 字段是适用范围）并据此审查。命中多个规则全部加载交叉检查；不要一次性加载无关规则；没有匹配时明确说明"未匹配到专项规则"。

| 改动路径 | 必读规则 |
| --- | --- |
| `torchtitan_npu/config/**/*.py` | `.agents/rules/config.md` |
| `torchtitan_npu/converters/**/*.py` | `.agents/rules/converters.md` |
| `torchtitan_npu/distributed/**/*.py`、`torchtitan_npu/patches/distributed/**/*.py` | `.agents/rules/distributed.md` |
| `torchtitan_npu/models/**/*.py` | `.agents/rules/models.md` |
| `torchtitan_npu/patches/**/*.py` | `.agents/rules/patches.md` |

### （二）代码坏味道审查

1. **命名不当**（AGENTS.md 命名规范 / 自文档化）
   - 判据：名称不准确、不反映实际作用域；计数未用 `num_` 前缀；生产代码出现 toy/test/temp/tmp/foo 等占位名；晦涩缩写；与 torchao / PyTorch 上游命名不一致；布尔量无 is_/has_ 语义；把描述塞进名字而非 docstring。
   - 注意：**改公共名 = 改签名**——改 config 字段名、函数/参数名或 state_dict key 需审计所有调用点，并评估旧 checkpoint / 配置兼容（可能破坏加载 → 至少 S2 并显式提示风险）。

2. **注释与文档不当**（AGENTS.md 注释与文档规范）
   - 判据：注释解释"显而易见"的代码，而非真正不明显处（维度语义、并行梯度 placement、workaround 存在原因）；**明显 AI 生成痕迹的注释**（逐行复述代码、冗长、千篇一律）；描述塞进名称而非 docstring；已知限制无 TODO 标记；注释用了中文（注释应英文、文档优先中文）；diff 改了行为但相邻注释 / docstring 仍描述旧行为（面向用户的行为 / 接口变化还需同步更新对应 `docs/`）。

3. **硬编码（把数值/字符串直接写死在代码里）**（KISS / 显式优于隐式）
   - 判据：维度、并行度、dtype、阈值、设备号、路径、模型名等直接写死在逻辑里（如 `hidden_dim = 7168`、散落的 rank/world_size 字面量判断、`num_heads // 8`、设备/路径字符串、超参字面量），且本应由 config / model_args / 具名常量提供。

4. **配置浅拷贝 / 共享引用陷阱**（AGENTS.md 参数与配置）
   - 判据：`dataclasses.replace()` 后修改嵌套 dataclass 或 list/dict 字段，误以为是独立副本（需要深拷贝处显式处理）。

5. **边界校验缺失，用静默回退（silent fallback）掩盖未对齐前提**（断言与错误处理 / Fail-Fast）
   - 判据：静默跳过用户配置却不发 warning；用 `assert` 兜用户可见错误（应改用 `ValueError`）；用静默回退掩盖未校验的前提假设（如假设 1D mesh / 固定 placement / dtype）；**空值 / 除零未保护**。
   - 注意：把"静默截断"改成显式 `raise` 是对的，但要确认所有调用方满足新前提（如 seqlen 能整除），否则原本能跑的路径会变报错。

6. **shell / CI 脚本健壮性**
   - 判据：变量名 typo 或未定义即用，导致 NaN/Inf 检测等逻辑**静默失效**；依赖未创建的目录等。
   - 修复：脚本统一 `set -eo pipefail`。

### （三）设计与架构审查

> **大改动强制项**：单个 PR / 改动净增改超过约 **500 行**，或新增 ≥3 个结构性文件 / 模块时，**必须完整走查本节（设计与架构）与（五）训练 infra 维度**，并在报告中明确给出"设计是否合理"的判断与改进建议。改动越大，越要先问：分层对不对、有没有更简洁的结构、是否为后续模型/并行扩展留了正确的扩展点。

1. **跨模型重复逻辑 / 每模型独立 wrapper**（原则3 复用优于重复 / DRY）
   - 判据：同一适配逻辑在多个模型各复制一份；或新增只做透传、不提升清晰度的薄封装层（thin wrapper）。

2. **实验 / 模型特例分支泄漏进核心 patch/converter**（原则4 / SRP 单一职责 / 关注点分离）
   - 判据：核心 patch/converter 文件里出现 `if experiment_x:`、模型名特判、临时开关散落在通用流程中（应隔离到独立路径 / 独立 converter）。

3. **逻辑放错层 / 模型无关功能塞进模型特定文件**（代码放置表 / SRP）
   - 判据：对照 AGENTS.md 目录职责表，distributed/config/converter 逻辑写进 models 文件，或通用能力硬编码在某个模型里。
   - **本仓红线**：`model.py` 必须保持**单卡训练语义**，严禁出现分布式策略代码（`Replicate` / `Shard` / CP / mesh / DTensor placement 等）——这些一律放 `parallelize.py` 或并行化函数。审到 model.py 里写分布式就按 S1 报。

4. **共享抽象的契约被单点修改而未审查调用点**（原则6）
   - 判据：改了公共模型组件 / 配置字段 / 分布式工具 / patch 签名，但未同步所有变体调用点。
   - 注意：这通常已构成正确性问题（按 S1 处理）；此处从设计角度强调共享契约的完整性。

5. **可扩展性 / 开闭**（OCP 开闭 / 组合优于继承，受 YAGNI 约束）
   - 判据：新增模型 / 并行策略 / 算子时在核心里加 `if model == ...` 分支，而不走 converter 注册表、模型注入或 config 扩展点；用继承堆叠变体而组合更合适；扩展点被绕过，导致下次加模型仍要改同一处核心代码。
   - 修复：通过本仓既有扩展机制接入新变体，让核心对扩展开放、对修改关闭。但遵守 YAGNI——只为已存在或明确在路线图上的需求预留扩展点，不为臆想需求过度抽象。

6. **超大文件 / 超长函数**
   - 通用参考：普通逻辑文件 200-400 行典型、800 行偏大。
   - **本仓例外**：`torchtitan_npu/models/**/__init__.py`、`config_registry.py` 及模型注入/注册文件天然集中声明大量模型、配置、参数，**大体量属正常，不据此报坏味道、不要求拆分**。

7. **重构未删除旧代码 / 残留死代码**
   - 判据：新实现替代旧机制后，旧类 / 旧函数 / 旧配置路径已无人引用却保留；"调用后又恢复 patch"等无意义操作；被新版取代的兼容分支未清理。

8. **PR 范围不纯净 / 耦合无关改动**
   - 判据：一个 PR 里混入与主题无关的改动；尤其无关配置值改动混进功能/重构 PR。

9. **新公共类型的不变量设计**
   - 按封装 / 不变量表达能力 / 有用性 / 强制校验 四个角度各给一句简评，提示"让非法状态不可表示、在构造处校验"。

#### 设计原则速查

**仅在能指向具体改动时引用，不空谈教条**：

- **SRP 单一职责**：一个 patch/converter/函数只做一件事，特例不混进通用流程。
- **OCP 开闭**：加模型/算子靠注册表与注入扩展，不改核心加分支。
- **LSP 里氏替换**：覆盖上游类/函数时保持契约（签名、返回语义、placement），否则破坏 patch 对齐。
- **ISP / DIP 接口隔离与依赖倒置**：依赖 PyTorch 原生抽象，不把 NPU 细节硬绑进通用接口（原则1）。
- **DRY**：复用优于重复（原则3）。
- **KISS**：简单直接，消除写死的常量与无谓抽象。
- **YAGNI**：扩展点够用即可，不为臆想需求过度设计。
- **迪米特法则**：减少 `a.b.c.d` 长链耦合，避免越层访问内部状态。
- **组合优于继承**：模型 / 策略变体优先组合与配置。
- **Fail-Fast / 显式优于隐式**：边界显式校验，silent fallback 要么补 warning 要么改显式报错。

### （四）上游交互核对

本仓 reviewer 最有价值的发现往往不在 diff 文本里，而在 diff 与上游 torchtitan 行为的交互中，逐项核对：

- **共享代码调用点**：改动公共组件 / 配置字段 / 分布式工具 / patch 签名时，审查所有模型变体（llama3/llama4/qwen3/deepseek_v3/deepseek_v32 等）的调用点是否同步。
- **patch 目标对齐**：patch 的目标函数 / 类在当前上游版本仍存在、签名未变；新增 / 修改 patch 后 `_apply_patches()` 注册正确，不与现有 patch 冲突；converter 通过 registry 正确注册。
- **文档与行为矛盾**：行为开关或支持范围变化时，检查 `docs/` 对应指南是否同步，避免用户按旧文档操作直接报错。

### （五）训练 infra 与 NPU 性能审查

面向模型训练基础设施与昇腾 NPU 的深度审查，这是本仓最核心、最易被通用 review 漏掉的维度。逐项核对。NPU 算子接口与硬件亲和约束以**昇腾社区文档（hiascend.com）**为准。

1. **并行与 DTensor 正确性**
   - 判据：硬编码 1D mesh 或固定 placement；新并行组合（如 TP+EP+CP）未校验 DTensor spec（placement / `tensor_meta.dtype`）一致；PP schedule（如 1F1B）改动未说明对 bubble / 峰值显存的影响；CP 开启时未同步开启依赖算子（如 `npu_dsa` 同步校验）。
   - 建议：显式校验 mesh 维度与 placement；多并行组合补 DTensor 一致性说明；改并行策略必须附 loss / grad_norm 对齐验证。

2. **NPU 算子选择、正确性与硬件亲和**
   - 判据：能用 NPU 原生 / 融合算子（`npu_permute`、`grouped_mm`、fused attention、smla/li/mhc）却用低效 eager 拼接；新增对非 aclnn 封装或 torch_npu 原生的外部算子库的依赖；关键张量维度未对齐昇腾 cube（16×16，常见对齐到 256）导致算力浪费；引入无谓的 host-device 同步或 NPU copy。
   - **算子反向正确性**：自定义 `autograd.Function` / `torch.library` 算子改 forward 时，**backward 必须同步实现、且返回的梯度数量与 forward 输入一一对齐**。
   - **维度要兼容多模型规格**：算子 / kernel 的分块、padding、维度上限不要为单一模型写死，用条件表达式按规格适配，避免为一个规格优化却劣化另一个。

3. **显存 / 内存与重计算**
   - 判据：重计算（activation checkpoint、kl loss recompute）路径持有多余引用导致泄漏；重复计算本可缓存的中间量；optimizer state（swap / virtual optimizer）保存加载漏掉空分片或分布式 state；大改动未评估峰值显存。
   - 建议：重计算 / 显存相关改动给"显存 + loss"双重验证。

4. **通信与计算掩盖**
   - 判据：把可异步的集合通信（AlltoAll / Allgather / async collectives）改成同步、打断计算-通信 overlap；在热点路径引入额外全局 barrier / 同步。

5. **优化器与 checkpoint 完整性**
   - 判据：改 muon / swap / virtual optimizer 或 checkpoint 逻辑未覆盖空分片、多并行度、各模型兼容；旧 checkpoint 无法加载；权重转换只挂在 `state_dict_adapter.from_hf/to_hf`，而 `initial_load_in_hf=False` 的 DCP 加载路径不过 adapter → 旧 DCP checkpoint key 不匹配（需补 DCP 迁移路径或提供转换脚本）。
   - 建议：补 save / load 往返验证；改 state_dict 结构显式提示兼容风险（参见「（二）命名不当」中改 state_dict key 的提醒）。

6. **精度与数值稳定**
   - 建议：按 AGENTS.md 数值验证，可用 `premerge-accuracy-check` skill 给 loss / grad_norm 对比。

7. **torch.compile / inductor 兼容**
   - 判据：改动破坏已验证的 compile 路径；自定义算子编译对所有模型生效而非按需限定。
   - 建议：compile 相关改动跑 `--compile.enable` 验证。

8. **converter / 模块注入包装正确性**
   - 判据：用 `self.__dict__.update(parent.__dict__)`（或不调 `super().__init__()`）来"继承"父模块属性，存在参数追踪与状态共享隐患；覆盖上游 `forward` 时新增的入参（如 `enable_gqa` / `is_causal`）声明却未使用；converter 接口与上游 `ModelConverter` 完全一致即无意义包装。
   - 建议：包装上游模块时 `super().__init__()` 后再选择性复制 `_modules` / `_parameters` / `_buffers`；删掉只是同名透传、不增价值的 converter。

## 三、审查准则与分级

### 报告门槛

只报告明确、可验证的问题：

- 代码无法编译、无法导入、类型或名称必然错误；
- diff 内逻辑必然错误，或会让配置、patch、converter、distributed、模型注入机制失效；
- 明确违反 `.agents/rules/` 或 `.agents/AGENTS.md` 的规则；
- 有 OpenLibing/CI 证据且属于真实代码问题的 CodeCheck 新增告警；
- 数值、checkpoint、并行 placement、patch 注册等会静默破坏训练结果的问题；
- 有**明确可维护性 / 可扩展性收益**的设计或坏味道改进（命名、硬编码、重复、扩展点等）——作为"建议级"提出（见下方「分级与底线」），给出收益与成本，不阻塞合入。

不要报告：

- 无明确收益、纯主观偏好的"可以更优雅"建议（有明确可维护性 / 扩展性收益的，按下方「建议级」处理，不在此列）；
- 风格细节，除非它明确违反本仓规则；
- diff 之外早已存在的问题；
- 需要大量猜测才能成立的问题。

### 分级与底线

适用于「二、审查维度」的全部发现：

- **正确性 / 原则违反 → 必须改**：违反 AGENTS.md 核心原则、或会导致正确性 / 数值 / 兼容问题的缺陷，按 S1/S2 正常报。
- **优雅 / 重构 / 扩展性 → 建议级**：行为不变、但能明确提升可维护性或可扩展性的改进，作为 **S3（标注"建议级"）** 提出，给出收益与成本，不阻塞合入，由作者权衡。命名、硬编码、扩展点这类即便不阻塞也要指出，不要因为"能跑"就放过。
- **底线（不可逾越）**：任何重构 / 优雅化建议都必须评估对**已验证路径、训练数值、checkpoint 兼容、patch 目标对齐**的影响：
  - 非计算性重构必须保证 loss/grad_norm 完全一致（AGENTS.md 数值验证），建议中要带上这条验证要求；
  - 改 patch/converter / 公共签名 / 配置字段 / state_dict key 时，先说明对所有调用点、上游对齐和旧 checkpoint 的影响；
  - 大范围重构先给方案与验证计划，经确认后再改。
- **用 YAGNI 平衡扩展性**：为合理的未来扩展预留扩展点是对的，但不要为臆想需求引入当下无用的抽象层。优先复用本仓**既有扩展机制**（converter 注册表、ModelConverter、模型注入、config 扩展），不另造框架。

严重级别：

- `S1`（必须修复）：必然导致训练出错、数值错误、导入失败，或破坏 patch / converter / 模型注入等关键机制。
- `S2`（应当修复）：明确违反规则，或影响重要场景，但不一定阻塞所有用户。
- `S3`（建议修复）：低风险但可验证的问题，含"建议级"改进；没有把握就不要报。

## 四、CodeCheck 告警处理

CodeCheck 失败定位、日志抓取和流水线触发由 `gitcode-pipeline` 负责；本 skill 只在拿到明确告警证据（规则编号/规则名、严重级别、文件路径、行号、失败信息）后判断处理策略：

1. 结合本仓架构和 `.agents/rules/` 判断是否真实代码问题；真实问题给最小修复建议。
2. 已知误报或与本仓适配模式冲突的告警，不作为必须修复的 finding；给出"建议在 openlibing 发起屏蔽申请"的结论，并说明为什么直接重构会降低可维护性或破坏适配模式。
3. 没有日志或 OpenLibing 证据时，只能给风险提示，不把公开规则集的泛化要求当作 finding。
4. 简单低级且机械性的真实问题可自行最小修复，例如明显未使用导入、拼写错误、尾随空格、缺失文件头或局部格式问题；若告警需要较大范围改动，或涉及拆分函数、调整配置/注册结构、修改公共接口、改变模型/patch/converter 行为、影响数值或兼容性，必须先向用户说明原因和方案，确认后再修改。

### 当前已知建议屏蔽场景：

- `torchtitan_npu/models/**/__init__.py` 和 `torchtitan_npu/models/**/config_registry.py`：模型注入和配置注册表天然会集中声明大量模型、配置和参数。若 CodeCheck 仅报告超大函数、函数参数超过 5 个、复杂度/长度类告警，优先建议 openlibing 屏蔽，不要求拆分或改签名；但若存在真实重复逻辑、导入错误或注册遗漏，仍按实际问题处理。
- `torchtitan_npu/patches/**/*.py`：patch 文件常需要访问上游 torchtitan、PyTorch 或 torch_npu 的受保护成员以完成 monkey-patch。若 CodeCheck 仅报告类外访问类内受保护成员（protected-access），优先建议 openlibing 屏蔽；但必须确认访问目标存在、上游版本兼容、patch 注册正确，不能用屏蔽掩盖真实兼容性问题。

### 常见 CodeCheck 修复指导

#### G.CLS.06 类内方法排序

修复 Python CodeCheck 规则 `G.CLS.06 类的方法建议统一按照一种规则进行排列` 时，核心原则是：**只移动完整方法块，不改函数体、不改调用、不顺手重构**。

适用场景：

- CodeCheck/openLiBingCI 报告包含 `G.CLS.06`。
- 问题描述包含 `A should be after B` 形式，例如 `DistributedMuon._get_param_type should be after DistributedMuon._update_momentum_single...` 或 `SomeClass.normal_method should be after SomeClass._sort_pairs_by_numel...`。
- 用户要求“仅移动函数顺序”“不要改逻辑”。

修复步骤：

1. 先定位报告中的类和方法，确认二者都在同一个 class 内。
2. 只移动完整方法块；若方法带装饰器、紧邻注释或 docstring，一并随方法移动。
3. 优先满足报告中的显式约束：`A should be after B` 表示最终行号必须满足 `line(A) > line(B)`。
4. 移动后检查相邻方法的装饰器、注释、空行没有被拆散；不要借机重命名、抽函数或修改逻辑。
5. 用 `pre-commit run --files <changed file>` 或对应 CodeCheck 复跑确认。

方法排序参考：

1. `__new__`
2. `__init__`
3. `__post_init__`
4. 其它魔法函数
5. `@property`
6. `@staticmethod`
7. `@classmethod`
8. 普通方法
9. 保护方法或私有方法

## 五、推荐验证

按改动范围做最小有效验证，并在结果中如实说明执行情况：

```bash
# 改动文件的 lint / 类型检查（本仓 pre-commit 含 pyrefly）
pre-commit run --files <changed files>
# 相关单元测试；影响共享逻辑时扩大到全量
pytest tests/ -x
```

## 六、输出格式

审查报告必须先给问题清单，再给摘要。若没有问题，直接说明"未发现明确、可验证的问题"。

```markdown
### 代码检视检查结果
- [x] 项目规范：已读取 .agents/AGENTS.md
- [x] 专项规则：已读取 .agents/rules/<rule>.md（或：未匹配到专项规则）
- [x] 审查维度：已核对代码坏味道 / 设计与架构 / 上游交互 / 训练 infra 与 NPU 性能
- [x] CodeCheck 告警：已判断修复/屏蔽取舍（或：本次未涉及）
- [x] 验证范围：已运行 <command>（或说明未运行原因）

### 审查结论
> 严重级别：S1 = 必须修复（必然出错或破坏关键机制）；S2 = 应当修复（违反规则或影响重要场景）；S3 = 建议修复（低风险或建议级改进）。

- [S1] <一句话问题> — `<file>:<line>`
  原因：<为什么这是确定问题，引用具体规则或失败条件>
  建议：<最小修复方向>
```

当发现 CodeCheck 问题时，在 finding 中附加：

```markdown
  - 建议：包装上游模块时 `super().__init__()` 后再选择性复制 `_modules` / `_parameters` / `_buffers`；删掉只是同名透传、不增价值的 converter。
  CodeCheck 证据：<job/rule/severity/log line>
```

## 七、GitCode PR 评论

只有用户明确要求发布评论时才调用 GitCode API 写评论。

- 发布前确认问题已验证，且每个唯一问题只发一条行内评论。
- 行内评论必须落在变更行上；先用 PR diff 定位，再用 raw 文件或本地 checkout 确认准确行号。
- 评论内容保持简洁：问题、规则/证据、最小修复建议。
- 可提交建议只用于小型自包含修复；跨文件或需要额外验证的修复不要给 suggestion block。
