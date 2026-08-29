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
缺少仓库、依赖或权重时给出可操作的报错。三种模式：

- `--dry-run`：随机 latent 冒烟（无需权重，CPU 可跑）；
- `--self-test`：stub 编码器打通完整管线（无需权重/视频文件，CPU 确定性）；
- 真实编码：对接官方 MAGI-2-preview 推理仓库与权重（`--magi2-repo` + manifest）。

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

### 2.2 权重目录（官方 `hf download` 布局）

真实编码需要三组权重，目录名与官方仓库 README 的 `ckpt/` 布局一致
（`hf download sand-ai/MAGI-2-preview --local-dir ckpt`；完整仓库约 307 GB，
预处理只需要其中三个子目录）：

```
ckpt/
├── text_encoder/                # Qwen3.5 文本编码器（--text-encoder-path）
├── vae/                         # 视频 VAE（--vae-ckpt）
│   └── Wan2.2_VAE.pth
└── stable-audio-open-1.0/       # 音频 VAE（--audio-vae-ckpt）
    ├── model_config.json
    └── model.safetensors
```

| 参数 | 目录 | 官方内容 | 大小 |
| --- | --- | --- | --- |
| `--text-encoder-path` | `ckpt/text_encoder` | Qwen/Qwen3.5-27B 文本编码器 | ~56 GB |
| `--vae-ckpt` | `ckpt/vae` | Wan2.2 视频 VAE（`Wan2.2_VAE.pth`） | ~3 GB |
| `--audio-vae-ckpt` | `ckpt/stable-audio-open-1.0` | Stable Audio Open 音频 VAE | ~5 GB |

只下载预处理所需子目录：

```bash
pip install huggingface_hub
hf download sand-ai/MAGI-2-preview --local-dir ckpt \
    --include "vae/*" "text_encoder/*" "stable-audio-open-1.0/*"
```

（`preview/`、`refiner/`、`turbo_vae/` 为推理/精修权重，预处理不需要。）

### 2.3 真实编码（官方编码器）

需要官方推理仓库的本地克隆，经 `--magi2-repo` 传入（脚本会把它插入
`sys.path` 并惰性导入：`inference/model/vae2_2.py` 的 `get_vae2_2`、
`inference/model/qwen35.py` 的 `Qwen35TextEncoder`、
`inference/pipeline/audio_decoder.py` 的 `SAAudioFeatureExtractor`）。
输入用 JSON Lines manifest，每行一个样本：

```jsonl
{"video": "videos/clip0001.mp4", "caption": "A red fox runs through the snow.", "audio": "audio/clip0001.wav", "id": "clip0001"}
{"video": "videos/clip0002.mp4", "caption": "Waves crash on a black-sand beach."}
```

- `video`/`caption` 必填；`audio` 可选（提供 `--audio-vae-ckpt` 时才编码）；
  `id` 可选，缺省取视频文件名主干（确定性；id 不得含 `.`、不得重复）。
- 相对路径按 manifest 所在目录解析。
- 兼容旧的目录模式 `--input`：`<name>.mp4` + `<name>.txt`
  （提供 `--audio-vae-ckpt` 时还需 `<name>.wav`）。

完整示例：

```bash
git clone https://github.com/SandAI-org/MAGI-2-preview

python3 scripts/magi2_preprocess_latents.py \
    --magi2-repo ./MAGI-2-preview \
    --input-manifest ./data/train_manifest.jsonl \
    --output-dir ./magi2_latent_shards \
    --vae-ckpt ./ckpt/vae \
    --text-encoder-path ./ckpt/text_encoder \
    --audio-vae-ckpt ./ckpt/stable-audio-open-1.0 \
    --device cuda \
    --samples-per-shard 64
```

#### 帧采样与张量约定

与官方推理配置一致（`inference/common/magi2_config.py` 的 `video_fps=25` /
`vae_stride=(8,16,16)` / `audio_latent_fps=25`）：

- **时间**：按 25 fps 抽帧（源帧率更高时按 `round(fps_src / 25)` 抽稀，
  不做上采样），再裁剪到 `T ≡ 1 (mod 8)`；VAE 时间下采样 8×（因果，首帧
  自成一段），得 `T_latent = (T - 1) / 8 + 1` 帧 latent。过短（不足一个
  时间窗）的视频报错。
- **空间**：中心裁剪到 16 的倍数（patchify 2× + 三次 2× 下采样）。
- **视频**：像素归一化到 `[-1, 1]`，`Wan2_2_VAE.encode`（float32）输出
  `(48, T, H, W)`，以 `float16` 写入分片。
- **文本**：`Qwen35TextEncoder`（bf16、`skip_layer=2`，与官方推理管线一致），
  输出 `(L_t, 5120)`。
- **音频**：波形经 `SAAudioFeatureExtractor` 编码后，沿时间轴重采样到
  25 行/秒（`scipy.signal.resample`，与官方 `resample_audio_sinc` 同一实现），
  长度与视频时长对齐（行数 = 抽帧后的像素帧数），输出 `(L_a, 64)`；
  无音频的样本写 0 行，加载端按纯视频+文本处理。
- 视频解码惰性导入：优先 `torchvision.io.read_video`，否则 `imageio` +
  `imageio-ffmpeg`（官方仓库 requirements 自带 torchvision）；音频波形经
  `scipy.io.wavfile` 读取（wav）。

### 2.4 `--self-test`（stub 编码器，CPU 确定性）

无权重、无视频文件：用确定性的无权重 stub 编码器（以均值池化近似各编码器
的下采样行为）在合成张量上跑通 解码 → 编码 → 分片 → `Magi2LatentDataset`
加载 全流程，适合 CI 冒烟：

```bash
python3 scripts/magi2_preprocess_latents.py \
    --self-test \
    --output-dir /tmp/magi2_self_test \
    --samples-per-shard 2
```

`--num-self-test-samples` 控制样本数（轮流使用几种合法形状以覆盖分桶），
不指定 `--output-dir` 时写入临时目录。

### 2.5 编码器注册与环境变量

预处理脚本在内部采用**可插拔的三阶段编码器管线**：视频 VAE、文本编码器、
音频 VAE 各自由一个注册在 `EncoderRegistry` 中的类承载（见
`scripts/magi2_preprocess_latents.py` 的 `BaseEncoder` / `EncoderRegistry`）。
默认注册的三个编码器分别是 `Wan22VideoEncoder`（`video_vae`）、
`Qwen35TextEncoderWrapper`（`text`）和 `StableAudioEncoder`（`audio_vae`），
对应官方 MAGI-2-preview 推理仓库中的 `get_vae2_2`、`Qwen35TextEncoder` 和
`SAAudioFeatureExtractor`。每个编码器类通过 `importlib` 延迟导入其依赖，
只有在实际使用到该编码器时才会触发导入，缺少仓库或依赖时报出可操作的错误。

#### CLI 参数别名

三组编码器参数同时支持新名与旧名（旧名仍然有效）：

| 新名 | 旧名（别名） | 环境变量 |
| --- | --- | --- |
| `--video-ckpt` | `--vae-ckpt` | `MAGI2_VIDEO_CKPT` |
| `--text-ckpt` | `--text-encoder-path` | `MAGI2_TEXT_CKPT` |
| `--audio-ckpt` | `--audio-vae-ckpt` | `MAGI2_AUDIO_CKPT` |

优先级：新 CLI 参数 > 旧 CLI 参数 > 环境变量 > 无（`None`）。

#### 环境变量用法

当 CLI 参数未传入时，脚本会从环境变量读取。适合在 shell profile 或 CI 配置
中一次设定：

```bash
export MAGI2_VIDEO_CKPT=/weights/ckpt/vae
export MAGI2_TEXT_CKPT=/weights/ckpt/text_encoder
export MAGI2_AUDIO_CKPT=/weights/ckpt/stable-audio-open-1.0

# 无需重复传入 --video-ckpt / --text-ckpt / --audio-ckpt
python3 scripts/magi2_preprocess_latents.py \
    --magi2-repo ./MAGI-2-preview \
    --input-manifest ./data/train_manifest.jsonl \
    --output-dir ./magi2_latent_shards
```

#### 各编码器单独使用示例

只编码视频+文本（跳过音频）：

```bash
python3 scripts/magi2_preprocess_latents.py \
    --magi2-repo ./MAGI-2-preview \
    --input-manifest ./data/manifest.jsonl \
    --output-dir ./shards \
    --video-ckpt /weights/ckpt/vae \
    --text-ckpt /weights/ckpt/text_encoder
```

编码含音频的完整三模态样本：

```bash
python3 scripts/magi2_preprocess_latents.py \
    --magi2-repo ./MAGI-2-preview \
    --input-manifest ./data/manifest.jsonl \
    --output-dir ./shards \
    --video-ckpt /weights/ckpt/vae \
    --text-ckpt /weights/ckpt/text_encoder \
    --audio-ckpt /weights/ckpt/stable-audio-open-1.0
```

#### 替换编码器

第三方或自定义编码器可以通过在导入脚本前注册来替换默认实现。编码器类必须
继承 `BaseEncoder`、设置 `name` 属性、实现 `from_config` 和 `encode`：

```python
from magi2_preprocess_latents import BaseEncoder, EncoderRegistry

@EncoderRegistry.register
class MyCustomVideoEncoder(BaseEncoder):
    name = "video_vae"   # 覆盖默认的视频编码器

    @classmethod
    def from_config(cls, *, ckpt, device="cpu", **kw):
        ...  # 自定义加载逻辑

    def encode(self, video):
        ...  # 返回 (48, T, H, W) 的视频 latent
```

#### 缺依赖时的排错

每个编码器的依赖只在被实际使用时导入，因此缺失某个依赖不会阻止其他编码器
（或 `--dry-run` / `--self-test`）运行。常见报错及对应处理：

| 报错关键字 | 原因 | 解决 |
| --- | --- | --- |
| `Cannot import inference.model.vae2_2` | `--magi2-repo` 未传或路径不对 | `git clone https://github.com/SandAI-org/MAGI-2-preview` 并传 `--magi2-repo` |
| `Wan2.2_VAE.pth` not found | 视频 VAE 权重缺失 | `hf download sand-ai/MAGI-2-preview --include 'vae/*'` |
| `--text-encoder-path not found` | 文本编码器目录不存在 | `hf download sand-ai/MAGI-2-preview --include 'text_encoder/*'` |
| `model_config.json` / `model.safetensors` | 音频 VAE 目录不完整 | `hf download sand-ai/MAGI-2-preview --include 'stable-audio-open-1.0/*'` |
| `torchvision unavailable` | 视频解码缺 torchvision | `pip install torchvision` 或安装 `imageio` + `imageio-ffmpeg` |
| `scipy is required` | 音频波形读取缺 scipy | `pip install scipy` |
| `Qwen3_5TextModel support` | transformers 版本太旧 | 升级 transformers 到支持 Qwen3.5 的版本 |

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
