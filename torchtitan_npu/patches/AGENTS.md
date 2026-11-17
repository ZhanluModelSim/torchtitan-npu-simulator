# patches 目标规则

`patches/` 是上游待合入层的一部分，放置 `torchtitan` 或 `torch_npu` 补丁。

- `patches/torchtitan/` 镜像上游路径；新增临时补丁在许可证头后的固定区域使用 `Upstream-PR:` 记录上游依据。
- **当前 ratchet：** `scripts/check_architecture_boundaries.py` 同时识别历史 `Pending upstream PR`/`Upstream PR` 标记，并允许基线中已登记的缺失项；它阻止新增缺失项，不表示所有现有 patch 已完成统一头部整理。
- `patches/torch_npu/` 放 NPU runtime compatibility patch；它可以按运行时需要组织，但激活方式仍遵循 `apply()` 与显式导入约定。
- NPU runtime compatibility patch 放入 `patches/torch_npu/`；强制 eager 的上游行为兼容放入 `extensions/torchtitan/`。不要重新创建通用 `patches/workaround/` 目录。
- monkey-patch 模块定义并调用 `apply()`；所属 `__init__.py` 显式导入激活，禁止自动发现。
- 除所属 `__init__.py` 外，其他模块不得导入带 `apply()` 的模块；只提供定义的无副作用模块可按需复用。替换必须使用完整点路径赋值，不用短别名掩盖目标。
- 与 `models` 一起自包含：只依赖标准库、上游包和本层内部模块；不得直接依赖 `extensions`、`override` 或 `ops`。
- 上游合入并升级 pinned 依赖后，删除已吸收补丁和入口。
