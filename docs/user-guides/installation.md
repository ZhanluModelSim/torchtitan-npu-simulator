# 软件安装

## 版本配套表

torchtitan-npu 支持 Atlas 800T A3 等昇腾训练硬件。软件版本配套表如下：

| torchtitan-npu 版本            | torchtitan 分支  | PyTorch 版本    | torch_npu 版本 | CANN 版本  | Python 版本                               |      Triton Ascend        |
|------------------------|-------------|--------------|-------------|---------|----------------------------------------|--------------|
| master | main | 2.12.0 | 2.12.0rc1 | 9.0.0 | Python 3.11.x | 3.2.1 |
| v0.2.2-dev | v0.2.2 | 2.10.0 | 2.10.0 | 9.0.0 |  Python 3.11.x |   3.2.1  |
| override-refactor | main | 2.14.0(daily) | 2.14.0(daily) | 9.2.0 |  Python 3.12.x |   3.2.1  |

> [!NOTE]
> 安装所需的 Python 包依赖及版本见 [requirements.txt](../../requirements.txt)

## 源码安装

### 1. 安装依赖的软件

在安装 torchtitan-npu 之前，请参考版本配套表，安装配套的昇腾软件栈，软件列表如下：

<table>
  <thead>
    <tr>
      <th>依赖软件</th>
      <th>软件安装指南</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>昇腾 NPU 驱动</td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit">驱动固件安装指南</a>》</td>
    </tr>
    <tr>
      <td>昇腾 NPU 固件</td>
    </tr>
    <tr>
      <td>Toolkit（开发套件）</td>
      <td rowspan="3">《<a href="https://www.hiascend.com/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0604%2Ch0703">CANN 软件安装指南</a>》</td>
    </tr>
    <tr>
      <td>Kernel（算子包）</td>
    </tr>
    <tr>
      <td>NNAL（Ascend Transformer Boost 加速库）</td>
    </tr>
    <tr>
      <td>PyTorch</td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a>》</td>
    </tr>
    <tr>
      <td>torch_npu 插件</td>
    </tr>
  </tbody>
</table>

> 注：安装 NNAL（Ascend Transformer Boost 加速库）前，请先执行 `source /usr/local/Ascend/cann/set_env.sh` 配置 CANN 环境变量。

### 2. 下载 torchtitan-npu 源码


 ```shell
git clone https://gitcode.com/cann/torchtitan-npu.git
 ```

### 3. 安装 torchtitan-npu

```shell
cd torchtitan-npu
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

### 4. 安装 torchao-npu（可选）

仓库已在 `torchtitan_npu/experiments/torchao_npu` 中内置 TorchAO-NPU 适配代码，
默认无需单独克隆 `torchao-npu` 仓库。从 torchtitan-npu 仓库根目录执行：

```shell
python3 -m pip install -e ./torchtitan_npu/experiments
```

该命令以 editable 模式安装仓内 `torchao_npu` 包，并安装其声明的
`torchao==0.17.0` 依赖。

不安装适配包时，也可以将源码的父目录加入 `PYTHONPATH`：

```shell
python3 -m pip install torchao==0.17.0
export PYTHONPATH="/path/to/custom/parent${PYTHONPATH:+:${PYTHONPATH}}"
```

`/path/to/custom/parent` 必须直接包含 `torchao_npu/__init__.py`。使用仓内源码时，可将其
替换为 `<torchtitan-npu>/torchtitan_npu/experiments`。通用训练脚本只透传量化 CLI，
不会自动修改可选依赖路径。具体命令见
[快速上手](./quickstart.md#deepseek-v4-torchao-npu-低精度训练)。

## PyPI 安装

> 主线暂未提供此安装方式，待 torchtitan 发布稳定版本后提供

## 卸载

```shell
pip uninstall torchtitan_npu
```
