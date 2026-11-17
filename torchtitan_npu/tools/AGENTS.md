# package tools 规则

`torchtitan_npu/tools/` 放置需要随 Python package 分发、并依赖包内模型或 checkpoint 实现的离线工具。

- 与根 `scripts/` 区分：根目录是仓库维护和运行脚本，本目录是可安装、可通过 `python -m` 调用的正式离线工具。
- 参数解析与可复用转换逻辑分离；可复用部分使用公开函数，CLI 入口只组织 I/O、参数和错误报告。
- 依赖公开 API；可选 adaptor 按需导入，并给出明确的缺失依赖错误。
- checkpoint 转换修改必须检查 DCP/HF 键映射、EMA、CPU shard、并发读取和可选量化权重。
