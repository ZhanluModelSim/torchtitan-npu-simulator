# CV parallel 迁移状态

目标 TorchTitan 版本为 `ebdfc8c2a24d89d89c99c0ef96ed8ac149864175`。该版本通过提交 `43f73649f195a7790752daf11548d5523e9306b3` 删除了基于 Graph 的 EP 分块 producer 及其符号工具，并移除了 CV 使用的旧 trainer hooks。

CV 调度器仍可通过真实 ConfigManager 选择，配置 override 会注册目标版本的 pass pipeline 和 post-init hook。CV 自用的符号分析算法保留在 `chunk_symbols.py`，来源为删除前的上游提交 `c04d1bd3b6fd9f6f829bca6be148d9872a563c7a`，不会修改或补造上游模块。

CV Graph 运行路径尚未完成迁移。post-init hook、pipeline 和分块 pass 入口会在访问 runtime 状态前抛出明确的 `NotImplementedError`。配置检查通过仅代表配置与注册可用，不能据此启动 CV 训练。目标版本的 eager 分块使用不同的 metadata 协议，后续迁移需要重新对齐 producer、attention metadata 和调度输入，并验证实际执行语义。
