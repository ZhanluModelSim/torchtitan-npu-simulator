# config 规则

`config/` 是 NPU 配置 schema、纯 config 转换器与显式注册 API 的公共基础层。

- schema 只描述类型、默认值、校验和组件归属，不在 dataclass 中执行设备初始化或训练副作用。
- converter 保持纯 config-tree 转换；不导入具体模型、override 或 extension 实现。
- **当前行为：** `config/__init__.py` 导入 `manager`，后者安装 `ConfigManager._load_config` 的转换 wrapper。这是现有配置转换链的一部分。
- **目标状态：** `config/__init__.py` 只导出 schema 和注册 API；ConfigManager wrapper 由 extension 激活入口显式安装。完成前，不得把「导入 config 无副作用」写成当前事实。
- 新配置字段按 owner 放入上游组件子配置或 `ExtensionConfig` 分组；不为了 CLI 便利复制另一个已有字段。
- 修改 schema 时同时检查 Tyro CLI、`derive`/converter、recipe builder、checkpoint 兼容性和 config UT。
