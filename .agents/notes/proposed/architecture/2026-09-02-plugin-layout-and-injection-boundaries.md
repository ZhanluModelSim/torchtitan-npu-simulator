# Agent Note: Plugin layout and injection boundaries

Status: proposed
Created: 2026-09-02
Last updated: 2026-10-12
Tracking: PR #688

## Context

当前包导入依次加载 `patches`、`config`、`extensions` 和 `ops`。临时上游补丁、永久扩展、NPU override、模型代码和算子代码的职责与导入边界尚未完全收敛，重构时容易产生互相导入、重复 monkey-patch 或语义回归。

截至 2026-10-12，顶层 `compile/` 已迁入 `extensions/compile/`，主包入口仍自动导入 `config` 和 `ops`，尚未收敛为目标的 `patches → extensions`。多个层级之间仍存在历史反向依赖，extension loader、组件注册表及禁用开关也尚未实现。因此本 Note 同时记录当前迁移状态；`docs/architecture.md` 明确区分当前实现、禁止新增与后续目标。

## Proposal

目标结构由根和目录级 `AGENTS.md` 规定：

- `patches/` 与 `models/` 是上游待合入层。新增代码不得建立对 `extensions/`、`override/` 或 `ops/` 的反向依赖；现有跨层边按迁移债务逐项收敛。
- `extensions/` 是本仓永久扩展层，可以依赖 `patches/` 提供的符号，不得新增对 `models/`、`override/` 或 `ops/` 的依赖。
- patch 先替换 `torchtitan`，extension 随后可以通过 `torchtitan_npu.patches.*` 复用 patch 提供的定义，也可以从正常的 `torchtitan.*`、`torch.*` 或 `torch_npu.*` 导入路径使用替换结果。依赖方向保持单向：patch 不得新增对 extension 的依赖。
- `override/` 经 `override.imports` 显式替换 NPU/CANN 组件；`ops/` 只承载 kernel 与 `torch.library` 注册。
- 自动注入的目标顺序是 `patches → extensions`；selected override 随后在组件构建前生效。`models` 与 `ops` 按需导入。
- patches 是可早期激活的上游替身层；影响 dataclass schema 或默认工厂的 patch 早于依赖该符号的模块。`patches/torchtitan/__init__.py` 的显式导入列表定义当前激活顺序；该顺序由检查保护，不以「trainer 最后导入」这类与当前实现不符的绝对规则表达。extensions 在 patches 完成激活后加载，可以复用 patch 定义，并通过配置外包装、运行期 seam、单目标 hook 或公开注册 API 生效。
- `config/` 作为公开配置 schema 与转换基础层保留，但其包导入不得隐式安装 patch。
- `torchtitan_npu/compile/`、`torchtitan_npu/distributed/`、`torchtitan_npu/interfaces/` 和 `torchtitan_npu/patches/workaround/` 已迁移或删除；可安装的 checkpoint 转换工具放在 `torchtitan_npu/tools/`，与根目录仓库维护脚本区分。
- `experiments/torchao-npu/` 在它仍以独立实验包发布时保留现位置；不将独立 Python distribution 嵌入 `torchtitan_npu/extensions/experiment/`。

### Migration baseline

| 原路径或当前路径 | 职责 | 决策 | 当前或目标路径 | 验证 | 状态 |
| --- | --- | --- | --- | --- | --- |
| `torchtitan_npu/config/` | NPU config schema、Tyro 转换与 loader patch | KEEP；当前导入 `manager` 会安装 ConfigManager loader wrapper，后续拆除包导入副作用 | `torchtitan_npu/config/` | config 解析 UT、隔离导入测试 | planned |
| `torchtitan_npu/compile/` | SDC checksum graph pass | MOVE | `torchtitan_npu/extensions/compile/` | SDC UT 与 smoke test | implemented; NPU smoke not run |
| `torchtitan_npu/distributed/` | HF32 backend 开关 | DELETE，行为归 trainer extension | `torchtitan_npu/extensions/torchtitan/trainer.py` | TrainerEx/config CPU UT | implemented |
| `torchtitan_npu/scripts/` | 可通过 package 调用的 checkpoint 转换工具 | MOVE；删除无调用者的 patch 转发 shim | `torchtitan_npu/tools/` | CLI/import UT | implemented; 37 targeted CPU UT passed; distributed roundtrip blocked by sandbox networking |
| `torchtitan_npu/interfaces/torchao_converter.py` | TorchAO-NPU 与模型 config tree 的适配 | MOVE，禁止继续增加顶层 interface | `torchtitan_npu/extensions/adaptor/torchao/converter.py` | converter UT、可选依赖导入测试 | implemented; targeted CPU UT passed; NPU validation not run |
| `torchtitan_npu/extensions/trainer.py`、`profiler.py`、`tools/utils.py` | 对上游 Trainer 和工具模块的永久扩展 | MOVE，按上游 owner 镜像 | `torchtitan_npu/extensions/torchtitan/trainer.py`、`torchtitan_npu/extensions/torchtitan/tools/` | Trainer/profiler targeted UT | implemented; 7 passed |
| `torchtitan_npu/extensions/components/{checkpoint,metrics,lora_compat,optimizer}.py` | 对上游 components 的永久扩展 | MOVE，按上游 owner 镜像 | `torchtitan_npu/extensions/torchtitan/components/` | 路径、import 与调用点检查 | implemented |
| `torchtitan_npu/extensions/distributed/fsdp.py` | 对上游 FSDP 的永久扩展 | MOVE，按上游 owner 镜像 | `torchtitan_npu/extensions/torchtitan/distributed/fsdp.py` | import 与调用点检查 | implemented |
| `torchtitan_npu/extensions/graph_trainer/` | GraphTrainer 扩展 | MOVE，按上游 experiments owner 镜像 | `torchtitan_npu/extensions/torchtitan/experiments/graph_trainer/` | collection 与路径检查 | implemented |
| `torchtitan_npu/extensions/experiment/torchft/` | TorchFT 容错训练扩展 | MOVE，按上游 experiments owner 镜像 | `torchtitan_npu/extensions/torchtitan/experiments/torchft/` | 路径与 import 引用检查 | implemented; optional dependency required for execution |
| `torchtitan_npu/extensions/ep_process_group.py` | EP mesh 与 DeepEP process group 接线 | MOVE，先归入 TorchTitan 镜像层，后续再拆 owner | `torchtitan_npu/extensions/torchtitan/ep_process_group.py` | isolated CPU UT | implemented; 11 passed |
| `experiments/torchao-npu/` | 独立发布的实验 TorchAO-NPU 包 | KEEP，达到稳定性门槛后再提升 | `experiments/torchao-npu/` | 该 distribution 自有测试 | current |
| `torchtitan_npu/patches/workaround/` | 历史 runtime 与 eager workaround | MOVE 后 DELETE | `torchtitan_npu/patches/torch_npu/` 或 `torchtitan_npu/extensions/torchtitan/` | patch/extension UT 与 NPU smoke | implemented; device-copy/tooling UT 14 passed; Qwen adapter test interrupted; NPU smoke not run |
| `tests/smoke_tests/` | 需要 NPU 语义的小范围验证 | KEEP，纳入正式测试分层 | `tests/smoke_tests/` | smoke runner | current |
| `torchtitan_npu/extensions/experiment/` | 与 TrainerEx 紧密集成的试验性训练能力 | KEEP，不向核心模块泄漏分支 | `torchtitan_npu/extensions/experiment/` | 功能 UT/NPU smoke | current |

ownership debt manifest、目录布局检查和 extension ownership 检查脚本当前只保存在代码仓外的本地阶段工具目录，不属于本 PR，也不是仓库或 CI 的有效门禁。后续如需引入，必须重新设计、补充自测并作为独立变更评审。

### Current dependency debt

以下边不符合目标单向依赖，但当前代码仍依赖它们。`scripts/check_architecture_boundaries.py` 以 `.ci/architecture_import_baseline.json` 记录直接 import 基线并阻止新增；它不是债务已经清零的证明。

| 来源 | 当前直接依赖 | 收敛方向 |
| --- | --- | --- |
| `config/manager.py` | 包导入时安装 `ConfigManager._load_config` wrapper | 将安装时机移至 extension 激活入口，保持 converter 行为和 CLI 兼容 |
| DeepSeek-V4/V4.1 模型配置、并行化与 checkpoint 适配 | `extensions/` 与 `override/` | 提炼公开模型/配置接口后逐项替换，不改变训练、checkpoint 或数值语义 |
| Qwen3.5 模型模块 | 模型专属 `override/` | 通过明确注册或调用边界消除反向依赖 |
| `extensions/torchtitan/trainer.py`、checkpoint extension | `extensions/components/` | 在 loader 和组件注册表稳定后改为按需接线 |
| TorchAO adaptor、CV Parallel 与 TorchFT extension | `models/`、`override/`、`ops/` 或 patch 私有符号 | 用公开 adaptor/extension 接口分阶段替换 |
| `patches/torchtitan/models/qwen3_5/vision_encoder.py` | `extensions/torchtitan/eager_flex.py` | 将 eager 兼容逻辑移到单向可激活层，消除 patch 反向依赖 |

`patches/torchtitan/` 的新增文件使用 `Upstream-PR:` 头部。现有文件含历史 `Pending upstream PR`/`Upstream PR` 标记和基线中已登记的缺失项；当前门禁只禁止新增元数据债务，统一整理另行完成。

### Migration order

1. 已建立当前态矩阵和目录级规则；本地 ownership 工具未纳入仓库。
2. 已删除名不副实的 `torchtitan_npu/compile/`、`torchtitan_npu/distributed/`、`torchtitan_npu/scripts/`、`torchtitan_npu/interfaces/` 和 `torchtitan_npu/patches/workaround/`，并保持原有行为与调用时机；仓库根 `scripts/` 继续承载训练与维护脚本。
3. 已建立 `torchtitan_npu/extensions/torchtitan/` 镜像目录，将具有明确上游 owner 的模块、测试和引用迁入对应路径。
4. TorchAO converter 已迁入 `torchtitan_npu/extensions/adaptor/torchao/`；它对 patch 定义的依赖符合单向依赖规则，对 model 和 override 实现的依赖保留为后续架构债务，本阶段不重构，避免改变业务逻辑。
5. 下一阶段实现 extension loader、组件注册表和禁用开关，再收敛主包自动导入为 `patches → extensions`。在此之前，禁用开关仅是设计草案，不作为当前行为约束。
6. loader 边界稳定后，按实际依赖逐项清理跨层导入和重复 monkey-patch owner，并消除 `patches/torchtitan/models/qwen3_5/vision_encoder.py → extensions/torchtitan/eager_flex.py` 的反向依赖。

## Alternatives considered

### 保持现有顶层目录和导入副作用

这无法提供稳定归属和依赖方向，隐式耦合会继续累积。

### 让 patch 反向导入 extension

这会形成与 `patches → extensions` 激活顺序相反的依赖，使 patch 无法独立加载，并可能产生循环导入。extension 可以依赖 patch，patch 不得依赖 extension。

### 将所有 NPU 实现放进 override

override 是配置级显式替换层，不适合承载必须随包导入生效的上游补丁、永久扩展或 kernel 注册。

## Acceptance criteria

- 迁移文件符合所属目录的职责和依赖规则。
- 当前自动导入顺序是 `patches → config → extensions → ops`；收敛为 `patches → extensions` 是后续目标。override 仍仅由 `override.imports` 选择。
- extensions 可以依赖 patches；patches 不得新增对 extensions 的依赖。`patches/torchtitan/models/qwen3_5/vision_encoder.py → extensions/torchtitan/eager_flex.py` 是当前已登记的反向依赖债务。
- extension 禁用不影响 patches、models、override 或 ops 的可导入性。该禁用能力当前尚未实现，不能作为可用运行时配置。
- 若后续引入 ownership checker，其规则、例外和自测必须随独立变更提交；当前分支不存在该仓库门禁。
- 迁移矩阵中每个路径只有一个决策和一个目标 owner；状态与代码同步。
- 每阶段保持训练入口、override 选择、checkpoint 语义和已验证数值行为，并提供相称的 CPU UT、NPU ST 或数值证据。
- 同一变更更新根规则、受影响目录规则和 `docs/architecture.md`；后者定义目标架构，阶段完成状态记录在本 Agent Note。

## Consequences

目录 owner 和新增代码落点更加明确，但现有调用者必须使用迁移后的 import 路径。本轮不为以下已删除路径保留 re-export shim：

```text
torchtitan_npu.scripts.checkpoint_conversion
interfaces.torchao_converter
torchtitan_npu.compile.sdc_checksum
torchtitan_npu.distributed.utils
```

它们是目录迁移带来的显式 breaking change，PR 与发布说明必须逐项列出。允许 extension 复用 patch 定义可以避免重复实现，同时要求 patch 保持对 extension 的零依赖。主包自动导入、extension 禁用能力和历史跨层依赖仍未收敛，本 Note 在这些验收条件完成前保持 `proposed`。

## Risks

上游 API 漂移、导入顺序变化和错误的依赖判断都可能导致注册时机、checkpoint、分布式路径、性能或数值回归。
