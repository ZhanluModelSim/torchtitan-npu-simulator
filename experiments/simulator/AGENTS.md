# Simulator 开发规则

`experiments/simulator` 是独立 distribution `torchtitan-simulator`，在 Meta 上复用 torchtitan-npu 的原生模型、配置、并行化、loss 和 optimizer，捕获训练负载并估算逻辑活跃 storage 内存。主包导入不得依赖或自动激活本包；不编辑上游 checkout 或安装源码。

继承[仓库开发指南](../../.agents/AGENTS.md)和 [experiments 规则](../AGENTS.md)。本文件用于定位入口和约束修改；详细语义以当前源码及下面的领域指南为准，历史报告不作为当前版本的自动验收证明。

## 开始工作

- 先读[统一开发指南](docs/developer-guide.md)。运行时、兼容、capture/IR/通用内存和公开接口修改按[捕获运行时与负载契约](docs/capture-runtime.md)；模型结构、配置、组件、并行、量化及 checkpoint 修改按[模型适配与验收](docs/model-integration.md)。同时涉及两个领域时，两套规则都适用。
- 安装与最小运行见 [README](README.md)；平台同进程接入与参数见[接入指南](docs/upstream-integration.md)；支持范围查[能力矩阵](docs/capability-matrix.md)，并对照源码。主包支持某模型或某设备不表示 simulator 已支持。
- 修改前确认实际 Python、torch/torch_npu、CANN、torchtitan、torchtitan-npu、simulator 及可选 torchao-npu 的版本和导入路径。容器镜像、宿主源码与容器安装包可能不同，必须核实加载的是当前工作树。

## 入口与执行链路

| 入口 | 实现与用途 |
| --- | --- |
| `python -m torchtitan_simulator` | `torchtitan_simulator/__main__.py` → `cli.main()` |
| `torchtitan-simulate` | `pyproject.toml` 注册的同一个 `cli.main()`；需要安装 distribution |
| Python 同进程调用 | 从 `torchtitan_simulator.config` 导入 `SimulationConfig`，从 `torchtitan_simulator.runner` 导入 `simulate`；`simulate(options)` 返回 `WorkloadGraph` |
| `--build-only` | `prepare_config()` → `meta_runtime()` → `build_meta_model()`，打印模型构建报告；不捕获训练步、不验证梯度、不导出内存结果 |
| 容器入口 | `scripts/container-entry.sh` 加载 `CANN_ENV_FILE` 后执行模块 CLI；`Dockerfile` 只安装 sidecar，基础镜像必须已有完整训练依赖 |

正式捕获沿以下顺序定位，不恢复旧 `SimulationTrainer.train` patch：

1. `config.prepare_config()` 校验 `SimulationConfig`、解析 preset 或 `module.factory`，构造原生配置树，选择 override、AC 和量化。
2. `runner.simulate()` 进入 `runtime.meta_runtime()` 与 `compat.distributed.logical_world()`，构建 Meta 模型并执行原生 parallelize，创建 loss、optimizer、scheduler 和合成输入。
3. `step.run_simulation_step()` 捕获 forward/backward、clip、optimizer、scheduler 和 zero-grad 路径，将 dispatch 事件组装成 step、schedule 和 workload，并计算内存计划。
4. `runner` 检查参数梯度与 buffer 契约，附加配置、rank、量化与假设注记，经 `validation.validate_workload()` 后导出并返回结果。

CLI 是参数白名单，不是任意原生训练参数的透传。新增参数须同步检查 `SimulationConfig`、`cli.py`、`prepare_config()`、实际 consumer、测试及接入文档；几何参数还需检查 `model_overrides.py`。不接受配置后静默忽略。

## 源码归属

以下路径相对于 `torchtitan_simulator/`。

| 位置 | 职责 |
| --- | --- |
| `config.py`、`presets.py`、`model_overrides.py`、`selective_ac.py` | 原生配置组合、保留几何、类型化结构覆盖和原生 selective AC 策略 |
| `runtime.py`、`compat/` | 显式会话、能力探针、可恢复的 Meta/autograd/分布式兼容及模型虚拟融合 |
| `meta_env.py`、`rank_context.py`、`rank_table.py`、`metadata.py` | 捕获上下文、逻辑 rank/mesh、CPU 控制元数据与 Meta 元数据转换 |
| `runner.py`、`step.py` | 原生训练步接线、捕获编排及结果交付 |
| `capture/` | dispatch、通信与流依赖、保存槽、storage 生命周期、FSDP 驻留和图组装 |
| `ir/` | 算子、tensor 元数据、step、schedule plan/graph 和 workload 结构 |
| `memory/` | storage 去重、生命周期回放、AC/offload 投影、状态预算和文件输出 |
| `cost/` | 基础算子成本注记；未知项显式标记，不反向修改捕获事实 |
| `validation.py` | workload 引用、依赖、DAG 与捕获一致性校验 |

`tests/`、`scripts/` 和 `docs/` 位于 distribution 根目录。模型定义、配置工厂、parallelize/sharding 和 checkpoint adapter 归属主包 `torchtitan_npu/models/`；仅模拟所需的兼容留在本包。不得把实验分支泄漏到主包训练路径。`capture/`、`ir/`、`memory/`、`cost/` 的迁移须延续语义与测试，移动文件不等于完成新版验收。

## 当前运行边界

- `prepare_config()` 当前接受 DeepSeek-V4 和 Kimi K3。preset 映射以该函数为准；默认 `deepseek_v4_smoketest` 使用原生 debugmodel。`deepseek_v4_pro_20t` 在 `presets.py` 保留旧几何，不是新增的上游官方 preset。
- 单进程承载选定的 `logical_rank`，一次捕获一个 microbatch 和一次优化器更新。GBS/GA 配置不表示已执行完整梯度累积。输入是合成单文档 next-token 序列，EP 使用均衡路由假设。
- 当前 PP 必须为 1，FP8 通信和 A5 目标明确拒绝；compile/CUDA graphs 为 dispatch 捕获关闭。逻辑 world、capture worker 和 PP stage 必须区分。
- 当前 DSV4/Kimi 完整捕获要求 FP32 原始参数，可用 BF16 `mixed_precision_param` 计算；build-only 接受的 dtype 不代表完整捕获可用。`memory_parameter_storage_dtype` 是存储预算投影，不改变真实计算精度。
- 量化默认关闭，以 `enable_quantized_training` 显式启用原生 `all_mxfp8` recipe，需要配套 `torchao_npu`。preset 只选择结构，不从名称推断精度，不恢复旧精度后缀 preset 或 FQN 名单覆盖。
- 在导入 torch/simulator 前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 并加载对应 CANN 环境。Meta 捕获仍依赖配套训练软件栈，不能理解为仅安装 CPU torch 即可运行全部模型。
- 会话拥有 fake process group；不支持宿主已有 process group、嵌套或并发仿真。同进程串行调用须恢复 patch、TLS、mesh/cache 和全局状态，覆盖正常与异常退出。

## 捕获与内存必须保持的语义

- 原生 dispatch 优先。`compat/` 按真实接口/schema 与最小行为、梯度探针选择兼容，不按版本号分支；健康原生实现不得被无条件替换。patch 集中于可撤销作用域，禁止全局伪造设备工厂或用占位 tensor 冒充真实保存值。
- EP dispatcher 共用 `compat/routing.py` 的 functional `all_to_all_single` 适配，保留 count 到数据交换的依赖；不为模型增加另一套通信变体，不据此改写 CP/FSDP。
- 保留原始 view/alias/mutation、真实 process group 和流控制依赖。零耗时 `stream_wait` 仍参与调度；不得恢复图折叠、按序号全串行补边或推断恢复来掩盖捕获缺口。
- 默认返回 FSDP 通信视图，以实际 framework scope 隐藏 FSDP 非通信节点；完整源图保存在 `captured_workload`，所有内存事件照常参与估计。投影只传递已捕获的路径，不重编号或推断依赖；原始内存事件的 ID 到源图解析，详见[通信视图契约](docs/capture-runtime.md#fsdp-通信视图)。
- tensor、wrapper、storage 与 allocation generation 分开；一次 unpack、最后读取或 Python 对象销毁不等于设备 storage 释放。结合实际所有权、保存槽销毁、SAC 缓存消费及 FSDP alloc/free 判断生命周期。
- 保存钩子引入的 unpack 恢复用 detach 按精确对象身份排除，并保留别名依赖；不按算子名全局过滤 detach。模型／框架原有元数据节点仍保留，不能当作真实 Vector kernel 计数，详见[运行时契约](docs/capture-runtime.md#保存钩子引入的-detach)。
- 保存槽不等于激活或卸载候选。区分 F action 内交错反向和后续 B/recompute，核查同一 storage 的全部保存及使用。AC 先确认层／checkpoint 绑定再清零，覆盖无保存槽、无绑定的非层重算引用。
- `unattributed` 的目标是没有未解释项。已知 head/norm/loss/梯度桥接不伪装成 decoder 层；默认逐层 offload 仅选归属明确的层或显式 checkpoint 保存值。层外和未确认归属的值继续参与设备生命周期及峰值，禁止删除记录或只过滤报表得到零未知项。
- 使用实际本地 storage 去重和独立字节先验。optimizer state 来自实际状态清单；稳态峰值已含状态，不能重复相加。逐层 payload 表示单向传输量，不取卸载后为零的驻留字节。
- Meta 图、梯度元数据与逻辑 storage 预算不能证明真实 NPU 数值、HCCL 耗时或 allocator 峰值。未知 cost 不等于零耗时；allocator reserved/cache、碎片和未建模 workspace 不在预算内。

## 输出契约

成功的 CLI 和 `simulate()` 默认写入 `<output_dir>/memory/memory_summary.json`，`output_formats=["mem"]`／`--simulation.output-formats mem` 额外导出内存明细。普通调用无需再次调用 `export_result()`；该函数供重新导出使用。每次验证使用独立输出目录，旧目录中的结果和明细不会自动清理。

summary 保持六项预算（四个含状态的稳态峰值、权重大小、优化器状态大小）及逐层 offload payload 列表。字段与口径见[稳态内存说明](docs/steady-state-memory.md)。修改时同步 `memory/records.py`、`memory/export.py`、实际消费者、验收脚本及接入文档；不恢复 JSON/HTML/DOT 图导出。

## 开发与验证入口

以下命令均从 `experiments/simulator` 执行，使用已配置完整训练栈的 Python。安装命令见 [README](README.md)，测试工具来自 `pyproject.toml` 的 `dev` extra。

```bash
# 小模型完整捕获；默认同时校验图、梯度元数据并写入内存汇总。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torchtitan_simulator \
  --config deepseek_v4_smoketest \
  --simulation.output-dir /tmp/simulator-smoke

# 本包测试与独立 pyrefly；源码路径按实际安装位置填写。
TORCHTITAN_SOURCE=/path/to/torchtitan \
TORCHTITAN_NPU_SOURCE=/path/to/torchtitan-npu \
bash scripts/validate.sh
```

`validate.sh` 先运行本包 pytest，再运行本包 pyrefly；参数透传给 pytest。可用 `TORCHAO_NPU_SOURCE` 补充类型搜索路径。根项目类型检查刻意排除本包，不能用根目录检查代替 sidecar 检查，也不能以 `experiments.simulator.*` 重复扫描同一模块。

### 当前工作区的 Docker 环境

本工作区已有 `ttsim` 容器，宿主 `/home/andy/Projects` 挂载到 `/workspace`，当前仓库对应 `/workspace/torchtitan-npu-simulator`。优先复用并核对其状态及挂载，不因宿主缺少 Python/CANN 就判定无法运行，也不直接重建环境。这是本机环境记录，不是其他机器必须采用的容器名或路径。

从宿主执行下面的命令可直接使用挂载的 simulator 源码；容器停止时先执行 `docker start ttsim`。`docker exec` 不会重新执行容器 entrypoint，需显式加载所需 CANN 环境：

```bash
docker exec -w /workspace/torchtitan-npu-simulator/experiments/simulator \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 ttsim bash -lc \
  'source /usr/local/Ascend/ascend-toolkit/set_env.sh && python -m torchtitan_simulator --config deepseek_v4_smoketest --simulation.output-dir /tmp/simulator-smoke'
```

这里的 `/tmp/simulator-smoke` 在容器内。运行模块无需本包的 console script 已安装，但主包及其训练依赖必须可导入；验证前仍需核对主包 editable 路径。容器具备运行依赖也不表示安装了 `dev` extra 或 `pre-commit`。

### 按变更选择检查

- 运行时恢复和配置：优先定位 `tests/test_runtime.py`、`test_patch_scope.py`、`test_config.py`；捕获／IR／内存／cost 分别查对应测试子目录。
- 模型端到端：查 `test_dsv4_capture.py`、`test_kimi_capture.py`、`test_kimi_parallel.py`、`test_kimi_quantization.py`、`test_s4_mxfp8.py`。
- 内存归属：重点检查 `tests/memory/test_interleaved_loss_offload.py`、`test_layer_offload_summary.py`、`tests/test_memory_delivery.py`；控制依赖查 `tests/test_fsdp_stream_capture.py`。
- 修改或 review 测试时遵循仓库 `developer-tests-review` 技能。新配置／并行组合必须从真实入口验证到图、梯度及内存消费者；模型、loss、AC、量化、并行和 offload 变更须检查真实配置／数学／保存／使用／释放链路，不能仅凭 build-only 或测试数量声明完成。
- `scripts/acceptance.py` 是独立完整几何验收入口，默认覆盖四个 DSV4 结构和三种 AC；量化由 `--enable-quantized-training` 单独开启。按改动选择 `--presets`／`--modes`，Kimi 覆盖看其专门测试和[接入记录](docs/kimi-k3-migration.md)，不能将此脚本当作全部模型验收。
- 仓库完整 Lint 仍按根开发指南执行：先核对 `/tmp/torchtitan`，再从仓库根运行 `python -m pre_commit run --all-files --show-diff-on-failure`。本包测试／类型检查通过不等于完整 Lint 通过。
- 记录实际源码、依赖、命令、配置、逻辑 rank、输出和未验证范围。纯文档修改至少静态检查相对链接、参数和入口；运行条件具备时执行相关轻量验证，不仅为文档校对触发完整几何验收或真实 NPU 训练。
