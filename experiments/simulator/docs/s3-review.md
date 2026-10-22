# S3 自审记录

按 `torchtitan-npu-code-reviewer` 的五个维度复核。范围为 S3 相对冻结 S0–S2 镜像源码的 24 个 Python 文件，
另含测试、文档与根项目的独立 distribution 类型检查边界。具体 diff 保存在审计目录 `evidence/s3-source-vs-s2.diff`。

## 结论

当前验证范围内没有未解决的 S1/S2 发现。设计合理：真实保存/释放发生在 capture 层，字节与驻留策略在 memory 层，
原生 AC 扩展点与设备兼容放在独立 sidecar 中；未修改上游 checkout 或生产训练 Python 文件。
未匹配到 `torchtitan_npu/` 路径专项规则；适用 `experiments/AGENTS.md` 与 sidecar `AGENTS.md`。

## 关键核对

- 专项/注册：所有观测由显式入口安装；plain import 不激活适配；异常与正常退出恢复公共绑定、FSDP class setter 与 TLS。
- 设计/引用：弱 storage 引用和 packed-handle finalizer 不持有额外 Tensor；SAC observer 保留 native cache 的值、版本检查与策略。
  FSDP deferred release 只暂存元数据，实际分配成员与追溯别名分开；注册 buffer、预分配输入与嵌套 metadata 不能因首次使用较晚而漏算初始驻留。
- 上游交互：检查 native FSDP wait/post-forward/finalize 的清理与局部引用，复现根模块内部创建并释放最后 staging 的情况。
  AC 使用原生 Configurable/get_save_ops，未恢复旧 `_get_save_ops` 全局覆盖；逻辑参数 FQN 通过 wrapper 公共 named_parameters 方法读取。
- 训练基础设施：三种 AC 的 HSDP/EP/TP/CP 组合保留全部梯度、实际归约→optimizer 依赖和跨模板无环图；每次 AG 与释放一一对应。
  canonical functional collective 的 SAC 与 eager SPMD 原生行为分别核验，不强行让后者获得目标训练路径没有的缓存。
- 兼容/输出：新增字段与 JSON 明细为增量；旧手工/历史 capture 可保留缺少 storage provenance 的兼容路径。
  新 native 入口要求实际 provenance 完整。PP 的 observed ownership replay 明确拒绝，未宣称 PP 已迁移。
- 类型/静态检查：独立包 0 errors，原有 13 项局部忽略未增加。根项目排除另有类型配置的 sidecar，消除同源码的重复模块身份；
  完整环境下生产基线与候选 44 个相同既有错误，新增为 0，未以 Any cast 掩盖重复身份错误。

## 验证与边界

371 项测试包含独立手算、CPU loss/grad、native Meta 模型、dtype/offload 投影、GA 元数据、通信与生命周期验收；
保存槽 / 缓存 / FSDP / Python 引用均有独立先验或修复前失败证据。峰值活跃集合独立求和且检查 storage 唯一。

无真实 NPU，未声称 fused kernel 数值或实际硬件峰值验证。offload 为理想驻留投影，缺少真实传输与回捞重叠模型；
大规模捕获时 weak-storage 轮询的主机开销须在 S5 评估。全规模、MXFP8 模型组合、所有下游消费与 PP 不属于本次通过范围。
