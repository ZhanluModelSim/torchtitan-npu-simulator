# MAGI-2-preview 离线 latent 数据流水线

本文档描述 MAGI-2-preview 训练所需的离线预编码数据（latent）的**分片格式**、
**预处理脚本**、**打包与分桶行为**以及**训练配置接入方式**。对应代码：

- 数据集与 dataloader：`torchtitan_npu/models/magi2_preview/latent_dataset.py`
  （`Magi2LatentDataset` / `Magi2LatentDataLoader`）。
- 预处理 CLI：`scripts/magi2_preprocess_latents.py`。
- 合成数据对照实现：`torchtitan_npu/models/magi2_preview/dataset.py`。

合成（synthetic）数据仍可用于冒烟验证（见 `magi2_preview_smoketest`）；真实训练
应使用本文描述的离线 latent 流水线（`magi2_preview_latent_smoketest` 及
`Magi2LatentDataLoader`）。

## 1. 分片格式（shard format）

一个数据集就是一个目录，目录下包含若干分片文件，外加一个可选的
`index.json` 汇总：

```
magi2_latent_shards/
├── shard_0000.safetensors
├── shard_0001.safetensors
├── ...
└── index.json            # 可选，仅供工具浏览，读取不依赖它
```

### 1.1 单个分片（优先 `.safetensors`，兼容 `.pt`）

每个分片承载若干样本。推荐容器是 `safetensors`（仓库已依赖 `safetensors`），
若环境中不可用，可退化为 `torch.save` 的 `.pt` 分片。两种容器表达同一结构：

- 每个样本包含三个张量，键名以样本 id 为前缀：

  | 张量键 | 形状 | 说明 |
  | --- | --- | --- |
  | `{sample_id}.video_latent` | `(48, T, H, W)` | 视频 VAE latent，`float16`/`bfloat16` |
  | `{sample_id}.audio_latent` | `(L_a, 64)` | 音频 latent，可为 0 行（纯视频+文本） |
  | `{sample_id}.text_emb` | `(L_t, 5120)` | 文本编码，宽度固定 5120 |

  其中 `video_latent` 采用通道优先 `(C, T, H, W)`；读取时会被换回
  `(T, H, W, C)` 以对齐合成实现的 token 顺序。

- 每个样本附带属性（`fps`、`num_frames` 等）与形状信息，记录在**分片内嵌的
  样本清单**中：
  - `.safetensors`：写入文件 `metadata`，`samples` 为 JSON 字符串；
  - `.pt`：写入顶层字典的 `samples` 字段。

  清单中每个样本条目包含：`id`、`video_shape`（`[T, H, W]`）、`audio_len`、
  `text_len` 与任意 `attrs`。若清单缺失，读取端会按张量键名与形状自动推导。

- `index.json`（可选）：`{ "format": ..., "shards": [ { "file": ..., "samples": [...] } ] }`，
  由预处理脚本自动生成，仅作目录浏览/校验用途，加载器不依赖它。

读取端会校验张量形状与清单一致，且通道维满足视频 48 / 音频 64 / 文本 5120。

## 2. 预处理脚本

脚本：`scripts/magi2_preprocess_latents.py`。编码器相关依赖全部**延迟导入**，
缺少依赖或权重时给出可操作的报错。

### 2.1 `--dry-run`（无需权重，CPU 可跑）

写入一组随机 latent 的小分片，用于打通整条数据管线：

```bash
python3 scripts/magi2_preprocess_latents.py \
    --dry-run \
    --output-dir ./magi2_latent_shards \
    --num-dry-run-samples 8 \
    --samples-per-shard 4 \
    --seed 0
```

`--dry-run` 会轮流使用几种小的 `(T, H, W)` 形状，以便同时覆盖分桶与多形状打包
路径。生成的目录可直接被 `Magi2LatentDataLoader` 读取（单元测试即如此验证）。

### 2.2 真实编码

真实编码需要官方 MAGI-2 推理仓库在 `PYTHONPATH` 上，且提供视频 VAE、文本
编码器（及可选音频 VAE）权重：

```bash
PYTHONPATH=/path/to/MAGI-2 python3 scripts/magi2_preprocess_latents.py \
    --input ./videos \
    --output-dir ./magi2_latent_shards \
    --vae-ckpt /weights/Wan2.2_VAE.pth \
    --text-encoder-path /weights/qwen3.5 \
    [--audio-vae-ckpt /weights/sa_audio_vae] \
    --device cuda
```

- `--input` 目录中每个 `<name>.mp4` 需配一个 `<name>.txt` 字幕；提供
  `--audio-vae-ckpt` 时还需 `<name>.wav` 波形。
- 编码器导入是惰性的；未提供权重或依赖缺失时会明确报错并提示如何补齐。
- 未提供 `--audio-vae-ckpt` 时，音频张量写为 0 行，加载端按纯视频+文本处理。

## 3. 分桶与打包行为

`Magi2LatentDataset` 是一个**无限迭代**的 `IterableDataset`：

- **按视频形状分桶**：桶键为视频 latent 形状 `(T, H, W)`。同一个包（pack）内
  所有样本共享同一视频形状；音频/文本长度可以不同。
- **打包**：在 `max_tokens_per_pack` 预算内，把同桶样本按 token 数贪心拼接。
  单个样本超过预算时会单独成包。
- **`cu_seqlens` 契约**：每个样本在包内对应一个段，`cu_seqlens` 为各段累计
  边界；`input`/`coords_mapping`/`modality_mapping`/`time_embedding` 与
  `labels` 与合成实现完全一致（`labels` 为 `(T, 64)`，文本行为 0）。这与
  官方 `SimplePackedData` 的多段布局一致。
- **噪声**：每个样本独立采样 `sigma` 与 `eps`（由 `seed`、epoch 与样本位置
  确定性推导），因此流可精确复现、可断点续训。

## 4. 数据并行与断点续训

- **dp 分片**：分片文件列表按 `files[dp_rank::dp_world_size]` 切分，各数据并行
  rank 读取互不重叠的分片子集。若 rank 数多于分片数会报错。
- **断点续训**：`Magi2LatentDataLoader` 实现 `state_dict()` /
  `load_state_dict()`，保存 `epoch` 与样本游标；加载后从断点继续，噪声与顺序
  完全一致。空状态视为从头开始。

## 5. 训练配置接入

配置工厂 `magi2_preview_latent_smoketest()`（见
`torchtitan_npu/models/magi2_preview/config_registry.py`）与
`magi2_preview_smoketest()` 完全一致，仅把数据源换成
`Magi2LatentDataLoader`：

```python
dataloader=Magi2LatentDataLoader.Config(
    data_path="./magi2_latent_shards",   # 指向你的分片目录
    max_tokens_per_pack=4096,
    seed=0,
)
```

- 把 `data_path` 指向你自己的分片目录（可先用 `--dry-run` 目录跑通流程）。
- 该配置**刻意不注册进 simulator registry**：构建它需要一个真实存在的分片
  目录，不适合无数据的模拟场景。
- `Magi2LatentDataLoader.Config` 继承自 `BaseDataLoader.Config`，字段
  `data_path` / `max_tokens_per_pack` / `seed` 均可通过 CLI 覆盖。
