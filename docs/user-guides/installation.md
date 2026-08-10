# 软件安装

## 版本配套表

torchtitan-npu 支持 Atlas 800T A3 等昇腾训练硬件。软件版本配套表如下：

| torchtitan-npu版本            | torchtitan版本  | PyTorch版本    | torch_npu版本 | CANN版本  | Python版本                               |      Triton Ascend        |
|------------------------|-------------|--------------|-------------|---------|----------------------------------------|--------------|
| master | main `ac13e536c84e7f6647b14fa9375c3c8a8a2b8578` | 2.12.0 | 2.12.0rc1 | 9.0.0 | Python3.11.x | 3.2.1 |
| v0.2.2-dev | v0.2.2 `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce` | 2.10.0 | 2.10.0       | 9.0.0    |  Python3.11.x        |   3.2.1  |
| override-refactor | main `cc286a63599e42480a07928cc362e514ae448a85` | 2.14.0 | 2.14.0       | 9.2.0    |  Python3.12.x        |   3.2.1  |

## 源码安装

### 1. 安装依赖的软件

安装 torchtitan-npu 前，需参考版本配套表安装配套的昇腾软件栈。依赖软件如下：

<table>
  <thead>
    <tr>
      <th>依赖软件</th>
      <th>下载与安装资料</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>昇腾NPU驱动</td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/latest/softwareinst/instg/instg_0005.html?OS=Debian&InstallType=local">驱动固件安装指南</a>》</td>
    </tr>
    <tr>
      <td>昇腾 NPU 固件</td>
    </tr>
    <tr>
      <td>Toolkit（开发套件）</td>
      <td rowspan="3"><a href="https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260805101249091/">CANN 主线取包链接</a></td>
    </tr>
    <tr>
      <td>ops/Kernel（算子包）</td>
    </tr>
    <tr>
      <td>NNAL（Ascend Transformer Boost 加速库）</td>
    </tr>
    <tr>
      <td>PyTorch</td>
      <td><a href="https://gitcode.com/hitwdy/torchtitan-npu-ci-wheels/releases">PyTorch Nightly WHL 取包链接</a></td>
    </tr>
    <tr>
      <td>torch_npu 插件</td>
      <td><a href="https://gitcode.com/hitwdy/torchtitan-npu-ci-wheels/releases">torch_npu 自编译 WHL 取包链接</a></td>
    </tr>
  </tbody>
</table>

> 注：安装 NNAL（Ascend Transformer Boost 加速库）前，需执行 `source /usr/local/Ascend/cann/set_env.sh` 配置 CANN 环境变量。CANN 安装在其他目录时，需使用实际的 `set_env.sh` 路径。

> 注：单独安装 `torch_npu` 时，需同时安装 `numpy` 和 `PyYAML`，避免导入 `torch_npu` 时缺少依赖。

### 2. 下载 torchtitan-npu 源码

下载 `override-refactor` 分支，请注意命令中的字母大小写：

```bash
git clone --branch override-refactor --single-branch \
  https://gitcode.com/cann/torchtitan-npu.git
```

### 3. 安装 torchtitan-npu

当前自构建 `torch_npu` wheel 的元数据依赖 `torch==2.14.0`，而 `override-refactor` 分支使用配套的 PyTorch nightly。为避免依赖解析将 nightly 替换为正式版，先安装其他依赖，再使用 `--no-deps` 安装 `torch_npu`：

```bash
cd torchtitan-npu

sed '/^torch_npu @ /d' requirements.txt \
  > /tmp/torchtitan-npu-requirements.txt
grep '^torch_npu @ ' requirements.txt \
  > /tmp/torchtitan-npu-torch-npu-requirements.txt

python -m pip install \
  -r /tmp/torchtitan-npu-requirements.txt
python -m pip install --no-deps \
  -r /tmp/torchtitan-npu-torch-npu-requirements.txt
python -m pip install --no-deps -e .
```

## PyPI 安装

[PyPI](https://pypi.org/project/torchtitan-npu/) 提供已发布版本。该方式不包含 `override-refactor` 分支的未发布改动。
