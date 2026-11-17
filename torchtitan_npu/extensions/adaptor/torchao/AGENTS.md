# TorchAO-NPU adaptor 规则

`torchao/` 连接独立安装的 `torchao_npu` distribution 与 torchtitan-npu model config tree。

- `converter.py` 只在量化训练或量化导出路径中按需导入；不由 package 入口自动加载。
- 现有 model、override 和 patch 私有依赖属于 Agent Note 已记录的迁移债务；新增代码使用公开接口。
- 私有依赖通过等价公开接口逐步替换，保持 recipe、filter、config-tree 替换顺序、默认值和数值行为不变。
- 相关 CPU 测试位于 `tests/unit_tests/extensions/adaptor/torchao/`；可选包未安装时保持现有 skip 语义。
