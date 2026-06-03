# 软件安装

## 版本配套表

torchtitan-npu支持Atlas 800T A3等昇腾训练硬件形态。软件版本配套表如下：

| torchtitan-npu版本            | torchtitan版本  | PyTorch版本    | torch_npu版本 | CANN版本  | Python版本                               |      Triton Ascend        |
|------------------------|-------------|--------------|-------------|---------|----------------------------------------|--------------|
| master | main `ac13e536c84e7f6647b14fa9375c3c8a8a2b8578` | 2.12.0 | 2.12.0rc1 | 9.0.0 | Python3.11.x | 3.2.1 |
| v0.2.2-dev | v0.2.2 `73a0e6979dd10b6b1904098eb3c8f62c18ab87ce` | 2.10.0 | 2.10.0       | 9.0.0    |  Python3.11.x        |   3.2.1

## 源码安装

### 1.安装依赖的软件

在安装torchtitan-npu之前，请参考版本配套表，安装配套的昇腾软件栈，软件列表如下：

<table>
  <thead>
    <tr>
      <th>依赖软件</th>
      <th>软件安装指南</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>昇腾NPU驱动</td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit">驱动固件安装指南</a>》</td>
    </tr>
    <tr>
      <td>昇腾NPU固件</td>
    </tr>
    <tr>
      <td>Toolkit（开发套件）</td>
      <td rowspan="3">《<a href="https://www.hiascend.com/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0604%2Ch0703">CANN 软件安装指南</a>》</td>
    </tr>
    <tr>
      <td>Kernel（算子包）</td>
    </tr>
    <tr>
      <td>NNAL（Ascend Transformer Boost加速库）</td>
    </tr>
    <tr>
      <td>PyTorch</td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a>》</td>
    </tr>
    <tr>
      <td>torch_npu插件</td>
    </tr>
  </tbody>
</table>

> 注：安装 NNAL（Ascend Transformer Boost加速库）前，请先执行 `source /usr/local/Ascend/cann/set_env.sh` 配置 CANN 环境变量。

> 注：单独安装 torch_npu 2.12.0rc1 时，请同时安装 numpy 和 PyYAML，避免导入 torch_npu 时缺少依赖。

### 2. 下载torchtitan-npu源码master分支（请注意下列命令的大小写）


 ```shell
git clone https://gitcode.com/cann/torchtitan-npu.git
 ```

### 3. 安装torchtitan-npu

```shell
cd torchtitan-npu
pip install -r requirements.txt
pip install -e .
```

> 注：如有旧版本Torchtitan-npu，请先[卸载](#卸载)，再进行安装

> 注：`requirements.txt` 已为 `triton-ascend==3.2.1` 配置 `--extra-index-url https://triton-ascend.osinfra.cn/pypi/simple`；默认 PyPI 源当前不提供该版本，单独安装时请使用 `pip install --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple triton-ascend==3.2.1`。`triton-ascend` 是安装包名，实际 Python 模块名为 `triton`，代码中请使用 `import triton`，不要使用 `import triton_ascend`。


### 4. 安装inductor-npu-ext（可选，使能torch.compile时需要）

为了在 NPU 平台上充分利用 `torch.compile` 原生的编译能力，`torchtitan_npu` 在保留 Dynamo 与 Inductor 既有编译流程的基础上，接入了 Codegen 后端 [`inductor-npu-ext`](https://gitcode.com/Ascend/torchair/blob/master/experimental/_inductor_npu_ext/README.md)。该后端借助 [AutoFuse](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/graph/graphguide/autofuse_1_0001.html) 的自动融合能力，从 Inductor IR 生成 AscendC 融合 Kernel。

inductor_npu_ext 需要从源码安装。在运行环境内执行以下命令：
```bash
git clone https://gitcode.com/Ascend/torchair.git
cd torchair/experimental/_inductor_npu_ext/
pip3 install -e ./python/
cd -
```


> 注：具体torch.compile特性文档请参考 [说明文档](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/feature_guides/torch_compile.md)


## PyPI安装

> 主线暂未提供此安装方式，待torchtitan发布稳定版本后提供

## 卸载

```shell
pip uninstall torchtitan_npu
```
