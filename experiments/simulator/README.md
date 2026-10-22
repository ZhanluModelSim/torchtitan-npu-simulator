# TorchTitan simulator sidecar

DeepSeek V4 优先迁移的独立 distribution。显式运行入口才激活 simulator 适配，
生产 `torchtitan_npu` 包不依赖本包；不编辑 TorchTitan、PyTorch 或 NPU 安装源码。

S0–S5 已完成七个 DSV4 preset × 三种 AC 模式的完整 Meta 仿真验收，覆盖 BF16/MXFP8、MTP=1 和组合并行。真实 NPU 数值、耗时与 allocator 峰值尚未验证。
实现状态与未完成项见 [migration-status.md](docs/migration-status.md)。
S5 的完整执行命令、七个 preset 验收入口和输出契约见 [交付说明](docs/s5-delivery.md)，
各能力的范围与状态见 [能力矩阵](docs/capability-matrix.md)。
当前实现已按原生 dispatch 优先收敛，具体改动和本轮复验见 [收敛记录](docs/dispatch-first-remediation.md)。

上游平台的同进程函数接入、旧入口迁移及全部参数见 [上游接入指南](docs/upstream-integration.md)。

## 安装与运行

先准备与目标主线配套的完整 CANN 官方镜像、torch/torch_npu、TorchTitan 和 torchtitan-npu，
然后从本目录安装：

```bash
python -m pip install --no-deps --no-build-isolation -e .
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torchtitan_simulator \
  --simulation.world-size 4 \
  --parallelism.expert-parallel-degree 2 \
  --training.num-mtp-modules 1 \
  --simulation.output-formats text html dot json csv mem \
  --simulation.output-dir ./simulator_output
```

`--build-only` 只检查模型构建，不执行图/梯度验收。
`--simulation.logical-rank` 选择单个捕获进程承载的逻辑 rank；它与 PP stage 分开。
默认 preset `deepseek_v4_smoketest` 映射到目标原生 debugmodel，而非复刻旧版参数规模。
显式 `module.factory` 也可使用；全规模验收覆盖范围见能力矩阵。

## 当前合同

- BF16/MXFP8，AC none/full/selective，PP=1；使用原生 DSV4、配置 override、parallelize、loss、MTP、clip、fused AdamW 和 LR scheduler。
- 单次捕获一个 microbatch 和一次优化器更新。输入为合成单文档 next-token 序列；EP 显式采用均衡路由假设。
- 默认启用目标原生 sparse attention、LI、mHC、RoPE、RMSNorm 和 NPU token dispatcher overrides。
- 保留目标默认 spmd_types 后端。compile/CUDA graphs 为 dispatch capture 显式关闭。
- PP、FP8 通信开关与未验收 A5 目标明确拒绝。MXFP8 使用原生 `all_mxfp8` recipe，详见 [S4 审计](docs/s4-mxfp8-audit.md)。
- 内存输出是逻辑活跃 Tensor/storage 字节，不含 allocator reserved/cache、碎片及未建模 kernel workspace。
- 未知 cost 单独标记，不能将图存在解释为硬件耗时已知。
- tensor 驻留同时依据实际 storage ownership、autograd 保存槽销毁、native SAC 缓存消费和 FSDP alloc/free；
  CPU 控制元数据不计入逻辑设备字节，注册 buffer 从捕获开始计为常驻。详见 [S3 审计](docs/s3-lifetime-audit.md)。
- `--training.global-batch-size` 保留实际 GA 配置，但仍只捕获一个 microbatch；参数存储 dtype/offload 选项是明确的内存投影假设。

## MXFP8

独立 `torchao_npu` 包需与目标 NPU 主线配套安装。以原生开关启用：

```bash
python -m torchtitan_simulator \
  --extension.quantization.enable-quantized-training \
  --extension.quantization.recipe all_mxfp8 \
  --training.num-mtp-modules 1 \
  --simulation.output-formats json mem \
  --simulation.output-dir ./mxfp8_output
```

不提供旧 `mxfp8_fqns` CLI 复写；选择范围完全沿用原生 recipe，导出实际包装后的参数 FQN。
Flash/Pro/Pro 20T 的 `deepseek_v4_<flavor>_baseline_bf16|mxfp8` 已执行完整原生几何 F/B/O；
Pro 20T 保留旧几何并使用新原生 builder，不是目标主线新增的官方 preset。逐配置结果见能力矩阵。
类型化模型几何选项使用 `--model-overrides.*`，先重建配置树再量化；可用字段与边界见 S4 审计。

## 验证

在上述完整环境中安装本包的 `dev` extra，从本目录运行：

```bash
TORCHTITAN_SOURCE=/path/to/torchtitan \
TORCHTITAN_NPU_SOURCE=/path/to/torchtitan-npu \
bash scripts/validate.sh
```

兼容层按实际接口/schema 和最小梯度探针选择，不按版本号分支。
私有 FSDP/DTensor 观测接口仍是后续升级的审计点，详见状态文档。
独立 distribution 使用本目录的类型检查配置；根项目不再以 `experiments.simulator.*` 重复扫描本包。

## 镜像

基础训练镜像需已包含官方完整 CANN 和固定的 TorchTitan/NPU 依赖。本目录仅安装 simulator：

```bash
docker --context colima-titan-migration-audit build --network=none \
  --build-arg BASE_TRAINING_IMAGE=torchtitan-simulator:titan030-s5 \
  -t torchtitan-simulator:titan030-dispatch-first .
docker --context colima-titan-migration-audit run --rm torchtitan-simulator:titan030-dispatch-first --build-only
```

默认 `CANN_ENV_FILE=/usr/local/Ascend/cann/set_env.sh`；其他完整镜像可显式设置对应路径。
本次验收使用独立 Docker context `colima-titan-migration-audit`。
