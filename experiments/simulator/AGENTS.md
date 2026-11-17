# Simulator 开发规则

`experiments/simulator` 是独立 distribution `torchtitan-simulator`，在 Meta 上复用原生训练生成负载与静态显存。继承[仓库规则](../../.agents/AGENTS.md)和 [experiments 规则](../AGENTS.md)。主包不得依赖本包；不编辑上游 checkout 或安装源码。

## 无上下文任务的阅读顺序

1. [README](README.md)：定位、环境和最小运行。
2. [整体方案](docs/architecture.md)：执行链、职责、图与内存交付边界。
3. **新模型必须按[模型接入流程](docs/model-integration.md)及[常见问题](docs/model-integration-pitfalls.md)推进**，不能只注册 preset 或放宽 family 校验。
4. [开发导航](docs/developer-guide.md)：按任务选择专题、测试和交付检查。
5. 改捕获／运行时时读[捕获契约](docs/capture-runtime.md)；查支持范围读[能力矩阵](docs/capability-matrix.md)，不要把历史迁移报告当成当前能力。

## 修改归属

- 主包 `torchtitan_npu/models/` 定义模型结构、配置工厂、parallelize 与 checkpoint；优先复用同构 block、公共组件和现有配置级 override，模型差异留在所属模型。
- `config.py`／`presets.py`／`model_overrides.py`／`selective_ac.py` 负责模拟参数与原生配置组合。CLI 是白名单，不是任意训练配置透传。
- `runtime.py`／`compat/` 仅提供显式、可恢复的 Meta/autograd/分布式适配及必要虚拟融合；正常设备训练不受影响。
- `capture/`／`ir/` 保留数据、通信、流控制、保存值和身份事实；`memory/` 计算逻辑 storage 预算及卸载驻留；`runner.py`／`step.py` 编排单次交付。
- 性能预测、硬件、cube/vec 分类、DES、GA 阶段展开与最终指标归下游，不在训练仓恢复 cost 或编造耗时。

## 必须保持的边界

- 原生 dispatch 优先。按实际 schema 与最小 Meta／梯度探针判断兼容缺口，不按版本号装整套 patch，不用同 shape 占位 tensor 冒充真实保存值。
- 自定义融合必须具有明确 fake/Meta、autograd、真实保存输入和 `tensor_semantics` 声明；未知轴为 None，不从 shape 猜角色。主包定义的算子不得反向依赖 simulator，声明放在对应兼容层。详见[语义契约](docs/custom-tensor-semantics.md)。
- 默认同模型／optimizer 执行实际 GA 的一次无捕获结构预热，再捕获单 microbatch F/B 和 O。低层 `run_simulation_step` 不自动预热；预热不是数值意义上的真实第二训练步。F 内可交错 head/loss backward，不能只按 phase 判断 F/B。见[稳态捕获](docs/steady-state-capture.md)。
- 默认交付 FSDP 通信视图，完整源图在 `captured_workload`；静态显存读取完整原始事件。隐藏 FSDP 准备／搬移是约定，不为所谓完整性擅自打开。
- 保留 stream/event、producer、alias/mutation、跨模板依赖；零耗时控制节点也参与等待。逻辑 stream 不等于硬件资源，不按序号全串行补边，不折叠或重编号掩盖缺口。
- tensor、wrapper、storage、allocation generation 和保存槽分开。一次 unpack、最后读取或 Python wrapper 销毁不等于释放设备 storage；只精确排除保存钩子引入的 detach，不全局过滤原生 detach/view。
- head/norm/loss 与无法确认层归属的值保留设备驻留，不伪装成 decoder 层，也不为消除 unattributed 删除记录。AC/offload 按真实保存、生产、消费和写入关系处理。
- 既存 optimizer state 与 Muon runtime 常驻工作缓冲按实际本地 storage 去重进入初始内存。未卸载 F/B/O 含状态；聚合卸载排除状态并暂忽略状态 staging buffer，但状态总量保留。原生工作缓冲和计算临时量不能随状态卸载减掉，不能按权重倍数推断状态量。激活与优化器卸载独立。
- 卸载在训练执行图生成传输及等待，静态预算与同一计划一致；平台只定时与 DES，不重复补拖尾。激活阶段内预取、晚 producer 的 F 末保守收尾、共享 storage 和复制后写入限制见[激活卸载](docs/activation-offload-design.md)。优化器双向并行近似见[优化器卸载](docs/optimizer-offload.md)。
- 原生参数绑定与 AdamW 六组列表／可选角色不能丢失。源张量容量 `volume_bytes` 服务内存／调度，不是网络模型流量；不恢复 `communication_input_bytes` 或旧成本字段。

## 环境、入口与交付

使用 `python -m torchtitan_simulator` 或 `runner.simulate(SimulationConfig(...))`；具体参数与返回对象见[同进程接入](docs/upstream-integration.md)。`--build-only` 不证明图、梯度或内存正确。

运行前加载实际 CANN 环境，并在导入 torch 前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。核对 Python、依赖版本及主包／sidecar 实际导入路径。容器与宿主 checkout 不一定相同，不把旧挂载路径写成安装契约。

当前单进程只承载选定 logical rank；已有 process group、嵌套或并发模拟不支持，正常及异常退出都须恢复会话状态。PP>1、FP8 通信和 A5 当前拒绝。模型限定、精度与量化范围以能力矩阵和实际校验为准。

默认 `output_formats=[]`，仅返回 WorkloadGraph 和静态内存，不创建目录或打印导出预算；显式 `["mem"]` 才导出 summary／明细。`export_result` 空格式无操作，不恢复强制文件交付。旧输出目录不自动清理。

## 验证与文档

- 修改／review 测试使用仓库 `developer-tests-review` skill；文档使用 `write-torchtitan-npu-docs`。新模型按流程的分阶段出口验收，不用 build-only、测试数量或 Meta 成功替代数值／性能证明。
- 组件、真实小模型、相关并行／AC／量化、内存独立字节预期、连续会话和 wheel 普通安装分层验证。只扩大实际受影响的路径，不机械执行全部组合。
- `scripts/validate.sh` 是 sidecar pytest + pyrefly 入口；完整 lint 按根仓要求执行。`scripts/acceptance.py` 当前只覆盖 DSV4 的四种结构，Kimi 或新模型需要各自测试，不能自动外推。
- 文档只维护当前方案、workflow 和必要专题。接口、能力或命令变化同步 README 导航、对应专题及能力矩阵；不再增加阶段迁移／自审文档，历史证据由 Git 保留。
