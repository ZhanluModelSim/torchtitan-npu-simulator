# torch.compile 支持
torch.compile 是 PyTorch 2.0 的核心特性。通过 JIT （即时编译），将 PyTorch 代码转化为高度优化的融合算子，在几乎不改动原有代码的前提下显著提升性能。作为 PyTorch 原生的分布式训练框架，torchtitan 的一大优势便是可以便捷、充分地发挥 torch.compile 的性能收益。在此基础上，torchtitan-npu 结合 CANN 生态的编译能力，在 NPU 平台上的分布式训练任务中为 torch.compile 提供支持。

## NPU 上的 torch.compile

在 torch.compile 的工作流程中，PyTorch 代码依次经过 Dynamo 成图， Inductor 图编译优化、Codegen，生成在硬件 runtime 上执行的优化 DSL 代码。

<p align="center">
<img src="../assets/include_npu_ext.png" style="width:80%; max-width: 1200px" >
</p>

为了在 NPU 平台上充分利用 `torch.compile` 原生的编译能力，`torchtitan_npu` 在保留 Dynamo 与 Inductor 既有编译流程的基础上，使用 [`torch_npu` 内置的 AscendC Codegen 后端](https://gitcode.com/Ascend/torchair/blob/master/experimental/_inductor_npu_ext/README.md)。该后端借助 [AutoFuse](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/graph/graphguide/autofuse_1_0001.html) 的自动融合能力，从 Inductor IR 生成 AscendC 融合 Kernel。

## 支持范围
torchtitan-npu 当前支持 `DeepSeek-V3、DeepSeek-V3.2、DeepSeek-V4` 模型的全流程编译。

## torch.compile 示例

### 1. 使用 AscendC Codegen 后端

AscendC Codegen 后端已随兼容版本的 `torch_npu` 打包在 `torch_npu/_inductor/ascendc` 中，无需单独安装。通过 `torchtitan_npu.entry` 启动训练并开启 `torch.compile` 时，本仓会自动选择该后端。

在独立代码中直接调用 `torch.compile` 时，可通过 `options` 显式选择：

```python
import torch
import torch_npu  # noqa: F401

compiled_fn = torch.compile(
    fn,
    options={"npu_backend": "ascendc"},
)
```

### 2. 配置 compile

方式一：在模型的 `config_registry.py` 中配置 `CompileConfig`：

```python
from torchtitan.config import CompileConfig

compile = CompileConfig(
    enable=True,
    # 编译完整模型
    components=["model", "loss", "muon"],
    # Dynamo 使用 Inductor；AscendC 是 Inductor 内部的 NPU Codegen 后端
    backend="inductor",
)
```

`components` 可以按需选择编译范围：

| component | 说明 |
|-----------|------|
| `"model"` | 编译模型 forward/backward 主图 |
| `"loss"` | 编译 loss 计算 |
| `"muon"` | 编译 Muon optimizer 中的 Newton-Schulz 张量函数，详见 [Muon 优化器特性](./muon_optimizer.md) |


方式二：启动训练时通过命令行开启：

```bash
export TORCHINDUCTOR_SIZE_ASSERTS=0
bash scripts/run_train.sh --compile.enable
```

## 注意事项

### `NameError: name '_world' is not defined`

如果编译时报错 `NameError: name '_world' is not defined`，训练前需要关闭 Inductor 的
FxGraph / AOTAutograd 缓存：

```bash
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
```

关闭上述两个缓存后，Inductor 不再走 `FxGraphCache.load_with_key` 路径，可以规避该问题，代价是每次启动都会重新走
完整编译流程，一次性 warmup 时间会增加，稳态步长不受影响。

### 清理修改模型结构后的编译产物

当模型结构发生变化（如修改代码、切换分支、更新算子实现等）后，旧的编译产物可能导致编译失败或运行异常。若怀疑命中了旧产物，可以清理以下目录后重新编译：

```bash
rm -rf /root/.cache
rm -rf /tmp/* /tmp/.[!.]*
```
