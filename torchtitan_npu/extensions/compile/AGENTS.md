# extensions/compile 目标规则

本目录放图 pattern 基础设施与具体 pattern。

- graph pass 通过公开注册 API 接入，属于注册，不是 monkey-patch。
- 具体 pattern 按需显式导入；`__init__.py` 不批量加载 pattern。
- pattern 的环境变量、注册时机和验证路径必须在其 owning 文档或测试中明确。
