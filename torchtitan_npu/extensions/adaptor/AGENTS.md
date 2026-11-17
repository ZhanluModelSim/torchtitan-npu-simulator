# adaptor 规则

`adaptor/` 放置可选外部 Python distribution 与 torchtitan-npu 公开配置、模型和运行时接口之间的适配器。

- 包入口保持无副作用；不在 `adaptor/__init__.py` 中批量导入可选 adaptor。
- adaptor 由其 owner 在需要时按需导入，缺少可选依赖时给出清晰错误。
- 新代码只依赖各层公开 API，不新增对 patch 或 override 私有符号的依赖。历史迁移债务在局部规则和 Agent Note 中显式记录。
- 测试镜像放在 `tests/unit_tests/extensions/adaptor/`；真实设备语义仍按 tests 分层放入 smoke 或 integration tests。
