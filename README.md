<div align="center" markdown="1">

# torchtitan-npu

<h4>基于 torchtitan 的昇腾全流程大模型训练适配插件</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](#特性支持概览)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](https://gitcode.com/cann/torchtitan-npu/tree/override-refactor/LICENSE)
[![contributing](https://img.shields.io/badge/CONTRIBUTING-teal)](https://gitcode.com/cann/torchtitan-npu/blob/master/CONTRIBUTING.md)
[![SIG](https://img.shields.io/badge/SIG-framework--adapter-yellow)](https://gitcode.com/cann/community/tree/master/CANN/sigs/framework-adapter)
[![pypi](https://img.shields.io/badge/pypi-0.2.2.post1-blue)](https://pypi.org/project/torchtitan-npu/)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=plastic&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/torchtitan-npu)

</div>

# 简介

---

`torchtitan-npu`定位为`torchtitan`的昇腾（Ascend）后端扩展插件，通过即插即用的硬件亲和性优化，充分释放NPU算力，助力`PyTorch native`训练在昇腾平台无缝、高效、稳定地运行。

本插件通过 torchtitan 配置级 override 和 PyTorch backend 注册机制扩展上游能力，涵盖 NPU 融合算子、显存管理、分布式并行以及调试维测等能力。

## 社群
[![SIG](https://img.shields.io/badge/SIG-framework--adapter-yellow)](https://gitcode.com/cann/community/tree/master/CANN/sigs/framework-adapter)

SIG 例会：[sig-framework-adapter](https://meeting.osinfra.cn/cann?sig=sig-framework-adapter)

# 最新消息

---
- [Aug. 2026]: ⚠️ **override-refactor分支基于torchtitan override机制重构**。
- [May. 2026]: 🚀 **[DeepSeek-V4-Pro 模型续训练支持](https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md)**：基于纯FSDP + 大EP极简切分，使能AutoFuse特性，达成训练入图。
- [May. 2026]: ⚠️ **配置系统重构**：master 分支对齐 torchtitan main 的 `config_registry.py` / `ConfigManager` 机制，模型训练使用 `--module` 和 `--config` 启动，不再通过 `--job.config_file` 加载 TOML。
- [Apr. 2026]: 🚀 **[DeepSeek-V4-Flash 续训练 0day 支持](https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md)**：基于纯FSDP + 大EP极简切分，使能AutoFuse特性，达成训练入图，开箱即优。
- [Apr. 2026]: 🚀 **【重要特性支持】算子自动融合**：基于AscendC AutoFuse的能力，支持torch.compile + Inductor后端的算子自动融合。
- [Apr. 2026]: 🚀 **torchtitan‑npu 正式开源**：在 NPU 上支持 4D 并行等 torchtitan 原生特性，并引入 Swap Optimizer 等 NPU 亲和优化。

***
* [torchtitan-npu 0day 支持 DeepSeek-V4 续训练，助力训练场景轻松入图，开箱即优](https://gitcode.com/cann/cann-recipes-train/blob/master/docs/llm_pretrain/deepseek-v4_torchtitan_npu_autofuse.md)

# Roadmap

---

当前季度的规划见 `torchtitan-npu` [Roadmap](https://gitcode.com/cann/torchtitan-npu/issues/5)。欢迎访问。

# 安装

源码安装：

```shell
git clone https://gitcode.com/cann/torchtitan-npu.git
cd torchtitan-npu
git checkout override-refactor
pip install -e .
```

详情参见 [安装教程](https://gitcode.com/cann/torchtitan-npu/tree/override-refactor/docs/user-guides/installation.md) 。


# 快速上手
快速启动大语言模型的训练任务，参见
[快速上手文档](https://gitcode.com/cann/torchtitan-npu/tree/override-refactor/docs/user-guides/quickstart.md) 。


<a id="特性支持概览"></a>

# 特性支持概览

---

<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>特性名称</th>
      <th>原生支持</th>
      <th>NPU支持</th>
    </tr>
  </thead>
  <tbody>
    <!-- 并行能力 -->
    <tr>
      <td rowspan="3">并行能力</td>
      <td>4D 并行 (FSDP2/TP/CP/PP)</td>
      <td>✅</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>专家并行 (EP)</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>自定义 CP (DeepSeek-V3.2 CP/SDPA Ulysses CP)</td>
      <td>❌</td>
      <td>✅</td>
    </tr>
    <!-- torch.compile -->
    <tr>
      <td>torch.compile</td>
      <td>torch.compile</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <!-- 训练精度 -->
    <tr>
      <td>训练精度</td>
      <td>MxFP8 量化</td>
      <td>✅</td>
      <td>❌</td>
    </tr>
    <!-- 训练调试与监控 -->
    <tr>
      <td rowspan="2">训练调试与监控</td>
      <td>分布式 Checkpoint</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>调试工具</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <!-- 性能优化 -->
    <tr>
      <td rowspan="2">性能优化</td>
      <td>Swap Optimizer</td>
      <td>❌</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>NPU 融合算子适配</td>
      <td>❌</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

# 项目结构

`torchtitan-npu` 是 `torchtitan` 的 Ascend NPU 适配层，主要通过两类机制扩展上游能力：`override/` 使用配置级 `@override` 替换组件，并通过 `override.imports` 显式启用；`patches/` 补齐 PyTorch NPU backend 缺口，以及当前依赖版本尚未包含的临时上游能力。模型实现与并行化策略放在 `models/`，CANN 和设备专属算子封装放在 `ops/`。

```text
torchtitan-npu/
├── torchtitan_npu/
│   ├── models/                    # 模型与并行化实现等
│   ├── override/
│   │   ├── common/                # 模型无关的 NPU 组件替换
│   │   ├── deepseek_v3_2/         # DeepSeek-V3.2 专属 override
│   │   └── deepseek_v4/           # DeepSeek-V4 专属 override 与数值基线
│   ├── patches/
│   │   ├── torch_npu/             # torch_npu 的临时补丁
│   │   ├── torchtitan/            # 尚未进入当前上游版本的临时补丁
│   │   └── workaround/            # NPU 运行时兼容处理
│   ├── ops/                       # CANN 与 NPU 专属算子封装
│   └── __init__.py                # 导入 package patch
├── scripts/                       # 训练与仓库辅助脚本
├── tests/                         # 单元测试和测试数据
└── docs/                          # 使用指南与设计说明
```

上下游软件栈架构图如下：
![Architecture](docs/assets/Architecture.png)

# 性能基准

---

### 待测试


# 免责声明

---

## 致 torchtitan‑npu 使用者

1. torchtitan‑npu 提供的所有内容仅供您用于非商业目的。
2. 对于 torchtitan‑npu 测试用例以及示例文件中所涉及的各模型和数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集。如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的 License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用 torchtitan‑npu 过程中，发现任何问题（包括但不限于功能问题、合规问题），请在 GitCode 提交 issue，我们将及时审视并解决。

torchtitan‑npu 功能依赖的 PyTorch 等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，torchtitan‑npu 仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。


# License 声明

---

- torchtitan‑npu 产品的使用许可证，具体请参见 [LICENSE](https://gitcode.com/cann/torchtitan-npu/tree/override-refactor/LICENSE)。
- torchtitan‑npu 工具 docs 目录下的文档适用相应许可证，具体请参见根目录下的 LICENSE 文件。

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/torchtitan-npu/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/torchtitan-npu/discussions)参与交流。
- **经验分享**：通过GitCode[【Wiki】](https://gitcode.com/cann/torchtitan-npu/wiki)分享经验总结。
- **加入交流群**：通过扫描下方微信二维码添加torchtitan‑npu小助手微信，加入微信群与我们进一步交流。

<img src="docs/assets/torchtitan_npu_contact.png" alt="contact us" width="50%">
