# experiments 规则

`experiments/` 放置具有独立打包、依赖和测试边界的实验性 distribution，不作为 `torchtitan_npu` 主包的自动导入层。

- 每个子项目保持独立 `pyproject.toml`、可选安装方式和自有测试。
- 主包不得因导入 `torchtitan_npu` 而要求实验包已安装；集成入口必须按需导入并在缺少依赖时给出清晰错误。
- 实验包达到稳定 API、版本、发布和验证要求后，再通过 Agent Note 决定是否提升为正式顶层 distribution。
