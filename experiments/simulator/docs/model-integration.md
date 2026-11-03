# 新模型接入工作流

本指南面向首次接触本仓的开发者和 agent：将外部模型资料转成原生模型配置，再接入 simulator，最后验证图、梯度元数据、并行依赖和静态内存。先读[整体方案](architecture.md)，运行时细节查[捕获契约](capture-runtime.md)，排错查[常见问题](model-integration-pitfalls.md)。

当前正式入口只接受 DeepSeek-V4 和 Kimi K3；增加一个 preset 或放开 family 校验不等于完成新模型接入。模型定义与真实训练能力归主包，simulator 只负责必要的 Meta 兼容和负载交付。Meta 成功不代表真实设备数值或性能已经验证。

## 1. 固定环境和模型依据

先完成三件事，再写模型：

1. 阅读仓库根、`torchtitan_npu/models/` 和 `experiments/simulator/` 的 `AGENTS.md`，核对 `requirements.txt`、当前 TorchTitan 与主包／sidecar 导入路径。不要默认已有容器使用当前 checkout。
2. 固定模型 config、参考实现和 checkpoint 格式的 revision。列出数学与执行边界差异，不能只凭论文名或相同类名复用。
3. 选择最接近的已有模型作为工程参考，建立小配置和目标配置。小配置必须保留所有不同路径，例如混合 attention 层型、dense／routed／shared experts、MTP、低秩 gate；不能缩到只剩一个普通 Transformer block。

在已安装当前主包及 sidecar、加载配套 CANN 的环境执行：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python - <<'PY'
import torch, torchtitan, torchtitan_npu, torchtitan_simulator
print(torch.__version__)
for module in (torchtitan, torchtitan_npu, torchtitan_simulator):
    print(module.__name__, module.__file__)
PY
```

把以下先验写入模型说明或测试预期，而不是散落在聊天中：

| 先验 | 必须明确的内容 |
| --- | --- |
| 几何 | 层数／层型顺序、hidden/head/rank dimension、专家数／top-k、词表、共享专家、MTP、压缩比例与序列限制 |
| 数学 | norm epsilon、gate／激活、router score 相乘位置、mask／位置规则、dtype 转换、初始化、冻结与共享参数 |
| 融合 | 前后向边界、真实保存输入、输入输出角色与轴；不能将同名 CANN 算子直接视为等价 |
| 状态 | checkpoint key、矩阵方向、专家堆叠、共享权重关系、优化器需要更新的参数 |

本机容器路径、过去某个峰值和测试数量不是模型契约；每次验收保留当次实际配置与环境。

## 2. 先选复用路线，再确定文件

优先复用公式和接口相同的原生组件；只实现新模型真正不同的部分。

- **同结构、不同几何**：先复用已有配置 builder，不复制整套模型。需要改变几何时重建完整配置树，并同步 loss 词表等依赖字段。
- **大体相同、局部数学不同**：复用已有 decoder／block、Linear、Embedding、norm、router、RoutedExperts、GroupedExperts，仅替换差异组件。Kimi 的工程参考是 `feed_forward.py`：沿用原生专家与 dispatcher，保留自身 SiTU 和 latent MoE 顺序。
- **执行协议不同**：先确认与 simulator 的输入、loss、层边界、优化器和内存识别是否兼容，再决定扩展位置；不要将非 decoder 模型伪装成已有 family 绕过校验。

复制已有 block 前逐项核对残差、norm、激活、路由权重位置和并行归约顺序。复用工程组织不意味着复制所有默认值。主包模型层已有的跨层依赖债务不作为新增代码的先例。

| 内容 | 修改位置／参考 |
| --- | --- |
| 模型／block／组件配置和构建 | `torchtitan_npu/models/<family>/`；参考 [Kimi model](../../../torchtitan_npu/models/kimi_k3/model.py)、[配置 builder](../../../torchtitan_npu/models/kimi_k3/__init__.py) |
| 原生训练工厂 | `config_registry.py`；参考 [Kimi 工厂](../../../torchtitan_npu/models/kimi_k3/config_registry.py) |
| 参数与激活分片／FSDP | `sharding.py`、`parallelize.py`；参考 [Kimi 分片](../../../torchtitan_npu/models/kimi_k3/sharding.py)、[并行化](../../../torchtitan_npu/models/kimi_k3/parallelize.py) |
| checkpoint 映射 | 模型自己的 `state_dict_adapter.py`，按实际 key 和方向验证 |
| 可用于真实训练的 NPU 组件选择 | 主包既有 `override/` 配置机制或 provider 注入；模型层不直接依赖 `ops/override/extensions` |
| 仅模拟需要的兼容／虚拟融合 | sidecar `compat/`，由 `runtime.py` 显式安装并恢复 |
| simulator 公开入口与事实 | `config.py`、必要时 `cli.py`、`runner.py`；不要复制一套新 runner |
| 验证 | 主包模型数学／状态测试 + sidecar 模型捕获测试；按真实职责分别证明 |

## 3. 建立原生模型和训练配置

### 3.1 可遍历的配置树

实现或复用 `Module.Config`／`BaseModel.Config`，让每个投影、norm、attention、专家和子模块通过配置构建。constructor 中临时创建、不暴露配置的模块可能绕过 override、量化和 sharding。验收时比较配置遍历选中的 FQN 与实际构建模块，而不是只看顶层配置值。

层命名也参与内存归属：当前 `memory/layer_scope.py` 识别 `layers.<数字>` 与 `mtp_layers.<数字>`。优先保持已有 decoder 的 ModuleDict/block 组织；新路径不符合时应明确适配并验证归属，不能把未识别值假标为某层。

配置工厂每次返回独立对象；不能靠浅拷贝隔离嵌套 list/dataclass。保留原模型初始化策略、权重共享和 `requires_grad`。确实不参与训练的参数应有结构依据并冻结，不能添加假梯度来让清单完整。

`update_from_config()` 应把 parallelism 等设置应用到完整配置树。`build_meta_model()` 使用训练 dtype 上下文构建并执行 `init_states(buffer_device=meta)`；新增初始化不能偷分配真实设备 tensor，也不能修改进程默认 dtype 后不恢复。

### 3.2 完整的 ModelSpec 与工厂

当前 `prepare_config()` 调用 `factory(num_mtp_layers=...)`，工厂返回原生 `Trainer.Config`。无 MTP 的模型也应接收该关键字并对非零值明确拒绝。

工厂至少核对：

- `model_spec.name/flavor/model` 与独立配置树。
- `parallelize_fn` 的当前原生关键字接口；PP 仍未接入，不能仅填 `pipelining_fn` 就声明支持。
- loss、optimizer、scheduler，以及适用的 `post_optimizer_build_fn`。MoE 的 load-balancing hook 不能因为不是参数算子而漏掉。
- checkpoint adapter 的实际能力；没有验证过的 checkpoint 路径标明限制。

`ModelSpec` 正确并不保证 runner 可用。当前 [`runner.py`](../torchtitan_simulator/runner.py) 依赖以下 decoder 协议，新增 family 必须逐项核对：

| 当前调用 | 新模型要满足的事实 |
| --- | --- |
| `model.lm_head`、`loss_fn.set_lm_head(...)`、`model._skip_lm_head=True` | head 交由实际 loss 路径执行，不能主干与 loss 各算一遍 |
| `model.build_attention_masks(tokens, labels, {"positions": ...}, cp_mesh=..., load_balancer_type=...)` | 返回 tokens、labels、kwargs 三项；kwargs 必须保留 `positions`，CP 不能只切 shape |
| `model(**input_dict)` 与 `loss_fn(..., positions=...)` | loss 返回 `(loss, aux)`；当前合成输入为单文档 next-token 文本，多模态／额外状态需明确扩展协议 |
| 原生 `parallelize_fn`、SPMD context、clip、optimizer、scheduler | 使用实际训练调用与依赖，不能旁路为手工图模板 |
| 参数／buffer／梯度清单 | 并行前后名称可对应，本地 shard、dtype、placement 和冻结参数可解释 |
| CLI build-only 报告 | 读取 `model.layers` 与 `model.tok_embeddings.weight`；不同结构需明确调整报告，不只改 `simulate()` |

### 3.3 接入 simulator 的检查清单

按实际需要修改，不为新 family 无条件复制模型专属逻辑：

1. `config.prepare_config()`：新增 preset 别名、family 白名单、默认 override 选择；`module.factory` 同样受 family 校验。
2. 检查公共 override 是否真实适用。默认应复用已有公共 RMSNorm／dispatcher 等路径，模型特有数学另行实现；不能照抄 DSV4 sparse attention／mHC override 到另一模型。
3. `runtime.RuntimeSession.configure_model()`：只有确需模拟兼容时增加有作用域的安装；没有缺口则无需新分支。
4. `runner.py`：核对上述协议、dtype 约束和事实交付。模型不满足既有协议时先收敛窄接口，不扩散按模型名判断的分支。
5. `model_overrides.py`：当前平铺几何覆盖只支持已列出的 DSV4 preset，不自动支持新模型。新模型可先由 `module.factory` 返回完整配置树，不得静默接受后忽略覆盖。
6. 若新增用户参数，同步 `SimulationConfig`、`validate()`、CLI 白名单、实际 consumer 与接入文档。只添加参数声明不算支持。

**本阶段通过条件**：工厂能独立创建配置，build-only 使用正确模块和 dtype；非法 MTP、并行、精度或几何在明确边界拒绝。此时还不能声称已完成负载捕获。

## 4. 接入算子：先原生，后必要兼容

按以下顺序决定实现，避免把模型仿真变成另一套训练实现：

1. **原生 torch／CANN 接口健康**：直接执行并由 dispatch 捕获，不再手工补一份同名节点。
2. **原生接口只有 Meta／autograd 缺口**：用最小探针确认缺失，复用真实 schema 与保存对象，局部补齐。只在有缺口时安装，正常与异常退出都恢复原绑定。
3. **确实需要保留自定义融合边界**：在 `compat/` 定义 dispatcher custom op、fake/Meta 实现和 autograd。参考 [`compat/kimi.py`](../torchtitan_simulator/compat/kimi.py)，不在模型 forward 中新增 simulator 开关。

具体注册顺序为 `@torch.library.custom_op` 定义 schema → `register_fake` 定义 Meta 输出 → `register_autograd(..., setup_context=...)` 绑定前反向 → `register_tensor_semantics` 声明原始 overload 的角色与轴。以上在模块加载时静态注册一次；每次 runtime 的 `PatchScope` 只安装／恢复实际调用点，不重复注册 dispatcher 或语义表。

每个自定义融合的完成条件：

- raw forward/backward 名称和 schema 明确，输出 shape/dtype 与真实边界一致；Meta 输出不是内部 workspace 的承诺。
- `setup_context` 保存真正参与前向的输入；backward 使用实际 `dy`，返回数量与 schema 一致，保留所需参数梯度。不要用同 shape 新 tensor 冒充保存值。
- backward 独立 dispatch 可见，producer、保存／恢复身份和依赖可追溯；不以只出现一个前向节点为完成标准。
- 提供 [`tensor_semantics`](custom-tensor-semantics.md)：每个输入输出 occurrence 的角色、逐轴含义，反向亦声明。未知轴用 `None`，复合展平轴不猜分解大小；shape／dtype 和物理 stride/layout 仍是独立事实。
- 激活调用点通过 `PatchScope` 替换并恢复。纯 Meta 虚拟算子遇真实设备调用明确拒绝；若主包需要真实 provider，由主包配置注入，不暗中使用近似实现。
- 不添加 FLOPs、耗时、cube/vec 分类或硬件信息；特征语义归负载，性能建模归下游。

声明时复用 `input_tensor_bindings` 绑定输入角色，不假定 kwargs 的出现顺序；保留重复 tensor 和可选输入的缺席状态。手工 `record_synthetic_op()` 不是新计算算子的首选接入方式。

参考验证：[融合保存与反向身份](../tests/test_kimi_capture.py)、[语义轴单测](../tests/capture/test_tensor_semantics.py)、[真实语义交付](../tests/test_custom_tensor_semantics_delivery.py)。

## 5. 并行与优化按顺序使能

先建立单 logical rank 的正确 F/B/O，再一次增加一条实际执行路径。当前全局边界仍为 PP=1、非 A5、无 FP8 通信；不能因新模型自身支持就绕过 simulator 的限制。

### 原生分片 → AC → FSDP

Kimi 的可复用工程顺序为：配置树设置 sharding → `model.parallelize(parallel_dims)` → 原生 AC apply → 解析 dense/expert mesh → `apply_fsdp_to_decoder()`。FSDP 使用实际 `dp_mesh_dims/edp_mesh_dims`、参数／归约 dtype 和 reshard policy；不能把旧 plain-Tensor 包装代码加两个参数后视为完成迁移。

复用 decoder FSDP helper 前，检查它实际遍历的 `layers`、block `moe_enabled`、routed experts、embedding/norm/head 等结构与新模型一致。保持一致的 block 组织有利于复用 AC、FSDP 和模块归属；字段拼写相同但行为不同仍需适配。

- **TP/SP**：验证每种参数及激活布局、本地 shape 和 partial reduction。非线性 norm/gate 前需要完成的归约不能延后。
- **EP**：使用已有原生 dispatcher 与 [`compat/routing.py`](../torchtitan_simulator/compat/routing.py) 的均衡交换路径；验证 count→数据交换→GMM→combine 依赖及真实通信组。无需给新模型复制通信实现。
- **CP**：因果卷积、递归状态或全上下文 attention 不能仅按序列维切开独立执行。保守 full-context gather/local slice 也是建模选择，必须保留真实通信并验证输出和梯度。
- **FSDP/HSDP**：测试首末 logical rank、非均匀／空 shard、dense/expert mesh 和 AG/RS/AR 完成依赖。不能按 `numel/world_size` 一律计算本地字节。

### AC、量化与 offload

AC 通过原生 `none/full/selective` 策略启用；selective save set 必须命中实际执行的原语，不能从配置名推断已经缓存。重算节点仍应有 FQN、`execution_kind` 和实际依赖。

量化默认关闭，由 `enable_quantized_training` 与已有 recipe 显式启用。检查实际参数 wrapper、codes/scales、反向及子类是否被保留；SiTU 等模型专属激活不能被公共转换器悄悄替换。当前仅 `all_mxfp8`，模型名不应包含精度开关。

激活与优化器卸载独立打开，分别复用[激活卸载](activation-offload-design.md)和[优化器卸载](optimizer-offload.md)。训练仓交付传输节点、等待依赖及静态显存；不要在新模型里计算传输耗时。head/MTP、层外保存值和 chunked loss 必须核对归属，详见[常见问题](model-integration-pitfalls.md)。

## 6. 从构建到交付的验收关卡

每一关都要在新模型的真实入口上验证；既有模型测试通过只能证明没有明显回归。

| 关卡 | 操作和通过条件 | 参考 |
| --- | --- | --- |
| A 配置与构建 | 独立配置、关键层型齐全、实际 override/量化选择正确、非法组合拒绝；校对参数清单与独立几何计数 | [test_config.py](../tests/test_config.py)、[Kimi 配置选择](../tests/test_kimi_capture.py) |
| B 核心组件 | CPU 数学／状态映射与独立参考对照；Meta F/B 的实际保存输入、dy、梯度与 raw 名称正确 | 主包 `tests/unit_tests/models/`、[融合捕获](../tests/test_kimi_capture.py) |
| C 单任务 F/B/O | 调用 `simulate()`；所有预期 trainable 参数有正确梯度、图 DAG／引用闭合、buffer 不漂移、状态清单完整 | [DSV4 捕获](../tests/test_dsv4_capture.py)、[Kimi 捕获](../tests/test_kimi_capture.py)、[优化器契约](../tests/test_optimizer_contract.py) |
| D 并行与 AC | 每条支持路径覆盖 local shape/placement、非零 rank、独立字节及通信完成依赖；对不同层型测 none/full/selective | [Kimi 并行](../tests/test_kimi_parallel.py)、[FSDP 流依赖](../tests/test_fsdp_stream_capture.py) |
| E 内存与卸载 | 层内／层外、共享 storage、chunk-local、recompute 归属可解释；两种卸载分别及共同启用均保护状态与 payload | [内存交付](../tests/test_memory_delivery.py)、[交错 loss](../tests/memory/test_interleaved_loss_offload.py)、[逐层汇总](../tests/memory/test_layer_offload_summary.py) |
| F 量化与交付 | 只验证承诺支持的 recipe 组合；实际 code/scale、梯度和自定义语义完整；串行两次和异常退出不泄漏状态 | [Kimi 量化](../tests/test_kimi_quantization.py)、[runtime 恢复](../tests/test_runtime.py)、[语义交付](../tests/test_custom_tensor_semantics_delivery.py) |

最小组合应含 world=1、FSDP/HSDP、适用的 TP／EP／CP、首末 rank、AC 三种模式；再补实际会改变执行的交叉组合和量化路径，不机械做完整笛卡尔积。明确不支持的组合应有拒绝测试与说明。

独立预期至少包括：按模块几何计算的权重元素数、冻结／共享参数处理、按真实分片规则计算的本卡字节、优化器状态项与 step 标量；用小 storage 例子验证别名去重、同时驻留和 offload payload。不能以新实现生成的同一份 summary 反过来证明自身正确，也不固定后端端到端耗时作为模型接入验收值。

`simulate()` 默认先做实际 GA 次 F/B 的一次结构预热，再捕获单 microbatch F/B 与 O；预热不推进 scheduler。测试须区分首用状态初始化与稳态更新，保留既存 optimizer state 的内存计量，不能按 zeros/empty/fill 名称过滤。捕获结果不是完整 GA 内存回放，详见[稳态捕获](steady-state-capture.md)。

交付图继续使用 FSDP 通信视图；内存从完整捕获事实计算。返回的 `memory_plan`／`memory_summary` 是正式内存对象；默认不要求文件存在，显式 `output_formats=["mem"]` 才校验导出与对象一致。相关结构见[同进程接入](upstream-integration.md)和[稳态内存](steady-state-memory.md)。

## 7. 可复用验证入口与交接

以下命令用于先跑通已有参考模型；新模型应增加自己的薄测试，再以其真实工厂与参数运行。使用已核对的 Python，工作目录为 `experiments/simulator`，先加载配套 CANN 并设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。

```bash
python -m torchtitan_simulator --config kimi_k3_smoketest --training.seq-len 16 --build-only
python -m pytest -q tests/test_kimi_capture.py tests/test_custom_tensor_semantics_delivery.py
```

完整 sidecar 回归和类型检查用 `bash scripts/validate.sh`；editable 安装时按脚本说明设置 `TORCHTITAN_SOURCE`／`TORCHTITAN_NPU_SOURCE`／可选 `TORCHAO_NPU_SOURCE`。根仓完整 Lint 与真实训练数值验证遵循主仓规则，不能用 sidecar 类型检查或 Meta 梯度存在替代。普通 wheel 安装须从源码目录外复验导入和小案例，不能靠 `PYTHONPATH` 修正包边界。

交接说明至少保留以下项目，供下一位 agent 继续开发：

- 模型参考 revision、依赖版本、源码提交／dirty 状态和实际导入路径。
- 支持的 preset/factory、输入与并行／AC／量化／offload 组合，以及明确拒绝的组合。
- 哪些模块复用了谁，哪些差异单独实现；自定义算子的 schema、角色／轴和真实 provider 边界。
- 每个验收关卡的命令、独立预期、结果与材料路径；区分 CPU 数学、Meta 结构、逻辑内存及真实设备验证。
- 未解释的保存值、未覆盖路径与后续动作。无证据的能力不标为完成。

上述工作流不要求预先实现该模型所有硬件 kernel 或性能公式；必须清楚标明虚拟融合、均衡路由、静态内存等假设，使后续平台与算子维护者能正确消费负载。
