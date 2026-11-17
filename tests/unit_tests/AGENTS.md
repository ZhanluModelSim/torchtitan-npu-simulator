# CPU 单测规则

本目录验证 CPU 可运行的生产语义和静态契约，不以 CPU mock 替代需要 NPU 语义的集成验证。脚本和 skill 自测放 `tooling/`。

通常从仓库根运行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -v --tb=short tests/unit_tests -x
```

优先运行与变更相关的测试。
