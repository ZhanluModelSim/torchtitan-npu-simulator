# torchao-npu 实验包规则

`torchao-npu` 是独立安装的实验性 Python distribution，包含 NPU 量化 config、wrapper tensor、算子和 serialization 能力。

- 保持 `torchao_npu` 包对 `torchtitan_npu` 的独立性；反向适配由 `torchtitan_npu/extensions/adaptor/torchao/` 承担。
- 不将该 distribution 整体移入 `torchtitan_npu/extensions/experiment/`，否则会混淆包发布边界与主包激活边界。
- 新能力在本目录的 `tests/` 中验证；涉及 torchtitan-npu 模型集成时，主仓同时补充 integration 测试。
- 文档必须明确该包需要单独安装，不将仓内源码存在写成默认可用。
- 通用量化配置、原语、权重 wrapper、量化 tensor 和 serialization 保留在公共目录；`torchao_npu.configs` 继续统一导出量化配置，各模型子包不自动导入具体算子或模块。
- `quantization/quant_primitives/mx_bf16.py` 和 `ops/kv_cache_fake_quant.py` 按 KV 缓存格式提供通用能力，保留在公共目录。
