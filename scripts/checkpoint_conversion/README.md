# DeepSeek-V4 推理对齐权重准备

这些 CPU 工具将 DeepSeek-V4 官方 HF 权重反量化为 BF16，再通过本仓现有 `DeepSeekV4StateDictAdapter` 转成 DCP。转换时可裁剪层数和专家数，用于固定权重的数值对齐。

## 反量化

输入目录包含 `model-*.safetensors` 分片。每个量化 weight 及其 `.scale` 必须位于同一分片。FP8 E4M3FN 使用 128×128 block scale；packed FP4 int8 使用每行 32 元素 scale，低 nibble 在前、高 nibble 在后。输出保持 HF tensor 名称，移除 scale tensor，并生成新的权重索引和去除量化配置的 `config.json`。

```bash
python scripts/checkpoint_conversion/dequant_save_hf_dsv4.py \
    "$HF_QUANTIZED" "$HF_BF16" --num-threads 8
```

输出目录必须为空且与输入不同。该命令逐分片处理，需要 CPU 内存及 BF16 输出磁盘空间；不要直接覆盖原始权重。已反量化的 BF16 输入可跳过此步骤。

## HF → DCP

```bash
python scripts/checkpoint_conversion/convert_dsv4_hf_to_dcp.py \
    "$HF_BF16" "$TT_DCP" \
    --n-layers 4 --num-experts 4 --top-k 3
```

省略 `--num-experts`、`--top-k` 时保留原模型专家配置；缩减后的配置必须与加载该 DCP 的模型一致。

用于 Transformers 推理对齐入口时，必须设置 `--num-experts 4 --top-k 3`，层数也须与推理入口一致。该入口固定四专家、top-3；裁剪专家时会按固定种子重新生成 `tid2eid`，其他专家数或 top-k 的 DCP 不能直接用于该入口。

HF → DCP 会保留全部选中张量，`from_hf` 还会为 routed experts 堆叠额外副本。CPU 内存预算可按「保留的 BF16 权重总大小 + routed expert 权重大小」估算，并预留 DCP 保存开销；专家权重占主体时，预算接近保留权重的两倍。转换完整 43 层时需按全模型规模准备 CPU 内存和磁盘空间，不能只依据单个分片大小估算。

转换后使用当前 key，例如 `layers.N.moe.routed_experts.inner_experts.w1_EFD`、`w2_EDF`、`w3_EFD`，并校验 attention 输出投影形状。输出目录必须为空。
