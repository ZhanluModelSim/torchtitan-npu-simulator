---
name: torchtitan-npu-env-setup
description: 在 Ascend NPU 上为 torchtitan-npu 项目搭建训练/开发环境，覆盖 local、conda、docker 三种安装方式（具体版本与 docker 镜像均以 installation.md 为准）。只要用户提到以下任意一项就必须触发本技能：配置 conda 环境、配 docker 环境、装 torchtitan-npu、装 CANN（Toolkit/Kernel/NNAL）、装 torch_npu、新机器搭环境、排查环境问题等。
---

# torchtitan-npu-env-setup

## 概述

本技能严格对齐项目内的 `docs/user-guides/installation.md` 的安装顺序与内容执行
其中 CANN 环境安装与环境变量配置不在本技能内重复实现，统一复用外部技能：
`cann-operator-env-config`（来源：<https://github.com/Ascend/agent-skills/blob/master/skills/cann-operator-env-config/>）。

该外部技能不随本仓提交，需按需安装（要求能访问 github.com）。若本地 `.agents/skills/cann-operator-env-config` 缺失，先执行：

```bash
bash .agents/skills/default-skills/scripts/install-default-skills.sh cann-operator-env-config
```

## 何时使用

- 新机器首次安装 `torchtitan-npu`。
- 现有环境可用性异常，需要按官方安装文档重走一遍安装流程。
- 需要给开发者提供标准化、可复现的安装步骤。

## 标准流程（严格参考 installation.md）

### Step 0：先询问安装方式（local / conda / docker）

开始执行前，先确认用户希望：

- 本地安装（local）
- conda 安装（推荐）
- docker 部署

约束如下：

- 默认推荐 `conda` 安装。
- 若用户选择 `conda`，必须先创建并激活 conda 环境，再继续后续步骤。
- 若用户选择 `docker`，执行下文「Docker 分支」。
  - docker 分支支持的模型/镜像以 `installation.md` 提供的镜像为准。
  - 若 `installation.md` 未列出该模型对应的镜像，则告知用户当前文档暂未提供对应镜像，引导其改用 `conda/local`，或由用户直接提供镜像下载地址后继续。

conda 分支前置命令（先读取 `installation.md` 版本配套表中对应分支的 Python 版本，替换下方 `<PYTHON_VERSION>`）：

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda create -n torchtitan-npu python=<PYTHON_VERSION> -y
conda activate torchtitan-npu
```

### Docker 分支

镜像信息（下载地址、加载后的镜像名:tag）按以下优先级确定：

1. **优先从 `installation.md` 读取**：若文档的「Docker 安装」或相关章节提供了对应模型的镜像下载地址与镜像名:tag，直接采用。
2. **文档未提供则询问用户**：若 `installation.md` 未提供，则向用户询问镜像下载地址（或本地镜像包路径）与加载后的镜像名:tag。

确定镜像信息后执行以下流程，命令中的 `<...>` 占位符按上一步取得的实际值替换。

#### 1) 获取 Docker 镜像

将镜像包上传到每个节点后，在每个节点加载（`<IMAGE_PKG>` 为实际镜像包文件名）：

```bash
gunzip -c <IMAGE_PKG> | docker load
```

加载完成后用 `docker images` 确认镜像名:tag，记为后续使用的 `<IMAGE>`。

#### 2) 启动 Docker 容器

容器名由用户指定，镜像用上一步确认的 `<IMAGE>`。在每个训练节点执行：

```bash
IMAGE="<IMAGE>"                 # 上一步 docker images 确认的镜像名:tag
CONTAINER_NAME="<自定义容器名>"  # 由用户指定

docker run -u root -itd --name "${CONTAINER_NAME}" --ulimit nproc=65535:65535 --ipc=host \
    --device=/dev/davinci0     --device=/dev/davinci1 \
    --device=/dev/davinci2     --device=/dev/davinci3 \
    --device=/dev/davinci4     --device=/dev/davinci5 \
    --device=/dev/davinci6     --device=/dev/davinci7 \
    --device=/dev/davinci8     --device=/dev/davinci9 \
    --device=/dev/davinci10    --device=/dev/davinci11 \
    --device=/dev/davinci12    --device=/dev/davinci13 \
    --device=/dev/davinci14    --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /home:/home \
    -v /data:/data \
    -v /mnt:/mnt \
    -v /etc/localtime:/etc/localtime \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip -v /etc/hosts:/etc/hosts \
    -v /usr/bin/hostname:/usr/bin/hostname \
    --net=host \
    --shm-size=128g \
    --privileged \
    "${IMAGE}" /bin/bash
```

> 说明：设备 `--device` 与挂载 `-v` 为 Ascend 训练容器的通用配置；NPU 卡数按实际硬件增删 `davinci*` 设备行。

#### 3) 进入容器

```bash
docker exec -it "${CONTAINER_NAME}" /bin/bash
```

#### 4) 容器内初始化环境变量

> 以下为镜像内 CANN 环境变量脚本路径，具体路径以镜像实际安装位置为准：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

### Step 1：对齐版本配套

先读取 `docs/user-guides/installation.md` 中的版本配套表，再开始安装。

### Step 2：安装依赖软件

按文档顺序完成以下依赖安装：

1. 昇腾 NPU 驱动 + 固件 （更新频率低，建议用户自行安装）
2. CANN 组件：Toolkit / Kernel / NNAL（Ascend Transformer Boost）

参考文档链接请以 `installation.md`「安装依赖的软件」表格中提供的链接为准。

#### CANN 环境安装约束

CANN 环境相关操作统一使用 `cann-operator-env-config` 技能；若本地缺失，按「概述」中的命令先安装。

### Step 3：下载并安装 torchtitan-npu

先拉取源码（注意命令大小写），再进入仓库根目录安装：

```bash
git clone https://gitcode.com/cann/torchtitan-npu.git
cd torchtitan-npu
pip install -r requirements.txt
pip install -e .
```

> 若已有旧版本，先执行卸载再安装：
>
> ```bash
> pip uninstall torchtitan_npu
> ```

### Step 4：可选特性 — 算子自动融合支持

如需启用 `torch.compile` 编译链路下的 NPU Codegen 后端，按文档执行：

```bash
git clone https://gitcode.com/Ascend/torchair.git
cd torchair/experimental/_inductor_npu_ext/
pip3 install -e ./python/
cd -
```

功能说明参考：
<https://gitcode.com/cann/torchtitan-npu/blob/master/docs/feature_guides/torch_compile.md>

### Step 5：可选安装方式 — PyPI

> 主线暂未提供此安装方式，待 torchtitan 发布稳定版本后提供（以 `installation.md` 的「PyPI安装」章节为准）。

### Step 6：环境验证

安装完成后，必须执行以下命令验证环境是否搭建成功：

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| NPU 状态 | `npu-smi info` | 返回 NPU 当前的面板监控视图，显示设备信息 |
| PyTorch & torch_npu | `python -c "import torch; import torch_npu; print(f'NPU Found: {torch_npu.npu.device_count()}')"` | 无报错，且 `NPU Found: >=1` (取决于实际硬件) |
| torchtitan 导入 | `python -c "import torchtitan; import torchtitan_npu; print(f'torchtitan ready')"` | 无报错，成功导入包 |

## 卸载

```bash
pip uninstall torchtitan_npu
```

## 执行要求

- 不跳步骤，不混用未在 `installation.md` 中声明的替代流程。
- 遇到安装差异时，以 `docs/user-guides/installation.md` 为唯一基准回溯。
- 若安装文档更新，需同步更新本技能内容。
