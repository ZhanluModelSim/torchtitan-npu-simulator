#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dequantize DSV4 FP8/FP4 Hugging Face shards to BF16 in HF layout.

Tensor names are copied unchanged. In particular, per-expert weights retain
their Hugging Face ``layers.N.ffn.experts.E.*`` names; the separate HF-to-DCP
converter owns all TorchTitan key mapping and expert stacking.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_BLOCK_SIZE = 128
FP4_BLOCK_SIZE = 32
FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _expand_block_scales(scales: torch.Tensor, rows: int, cols: int, block_size: int) -> torch.Tensor:
    expected = (
        (rows + block_size - 1) // block_size,
        (cols + block_size - 1) // block_size,
    )
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"invalid scale grid {tuple(scales.shape)} for weight "
            f"{(rows, cols)} and block size {block_size}; expected {expected}"
        )
    return scales.float().repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:rows, :cols]


def dequant_fp8_weight(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    expanded_scales = _expand_block_scales(scales, rows, cols, FP8_BLOCK_SIZE)
    return (weight.float() * expanded_scales).bfloat16()


def dequant_fp4_weight(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.int8:
        raise TypeError(f"packed FP4 weight must use int8 storage, got {weight.dtype}")
    rows, packed_cols = weight.shape
    cols = packed_cols * 2
    expected = (rows, (cols + FP4_BLOCK_SIZE - 1) // FP4_BLOCK_SIZE)
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"invalid FP4 scale grid {tuple(scales.shape)} for packed weight {tuple(weight.shape)}; expected {expected}"
        )

    packed = weight.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values = torch.stack((FP4_TABLE[low.long()], FP4_TABLE[high.long()]), dim=-1).flatten(1)
    expanded_scales = scales.float().repeat_interleave(FP4_BLOCK_SIZE, dim=1)[:, :cols]
    return (values * expanded_scales).bfloat16()


def dequant_weight_to_bf16(name: str, weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        return dequant_fp8_weight(weight, scales)
    if weight.dtype == torch.int8:
        return dequant_fp4_weight(weight, scales)
    raise TypeError(
        f"{name} has a scale tensor but unsupported weight dtype {weight.dtype}; expected FP8 E4M3FN or packed FP4 int8"
    )


def _copy_metadata(input_dir: Path, output_dir: Path) -> None:
    # The original weight index still references consumed scale tensors. A new
    # index is written after conversion. Remove quantization metadata from the
    # copied config so Transformers does not reinterpret BF16 weights as FP8.
    config_path = input_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.pop("quantization_config", None)
        if config.get("expert_dtype") in ("fp8", "fp4"):
            config["expert_dtype"] = None
        config["torch_dtype"] = "bfloat16"
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for name in ("tokenizer.json", "tokenizer_config.json"):
        source = input_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def dequantize_checkpoint(input_dir: Path, output_dir: Path) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("input and output directories must be different")

    shards = sorted(input_dir.glob("model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no model-*.safetensors under {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_metadata(input_dir, output_dir)

    weight_map: dict[str, str] = {}
    total_size = 0
    total_tensors = 0
    total_dequantized = 0
    started = time.monotonic()

    for shard in shards:
        shard_started = time.monotonic()
        output: dict[str, torch.Tensor] = {}
        dequantized = 0
        with safe_open(shard, framework="pt", device="cpu") as reader:
            keys = set(reader.keys())
            scale_keys = {key for key in keys if key.endswith(".scale")}
            orphan_scales = sorted(key for key in scale_keys if key.removesuffix(".scale") + ".weight" not in keys)
            if orphan_scales:
                raise ValueError(f"{shard.name} contains scale tensors without weights: {orphan_scales[:3]}")

            for name in sorted(keys - scale_keys):
                tensor = reader.get_tensor(name)
                scale_name = name.removesuffix(".weight") + ".scale"
                if name.endswith(".weight") and scale_name in scale_keys:
                    tensor = dequant_weight_to_bf16(name, tensor, reader.get_tensor(scale_name))
                    dequantized += 1
                if name.endswith(".weight") and tensor.dtype in (
                    torch.float8_e4m3fn,
                    torch.int8,
                ):
                    raise RuntimeError(f"{name} is still quantized after BF16 conversion")
                output[name] = tensor.contiguous()

        destination = output_dir / shard.name
        save_file(output, destination)
        for name, tensor in output.items():
            weight_map[name] = shard.name
            total_size += tensor.numel() * tensor.element_size()
        total_tensors += len(output)
        total_dequantized += dequantized
        elapsed = time.monotonic() - shard_started
        print(
            f"[{shard.name}] {len(output)} tensors, {dequantized} dequantized, {elapsed:.1f}s",
            flush=True,
        )
        del output

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    elapsed = time.monotonic() - started
    print(
        f"Converted {total_tensors} tensors ({total_dequantized} dequantized) in {elapsed:.1f}s -> {output_dir}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="official DSV4 checkpoint")
    parser.add_argument("output_dir", type=Path, help="BF16 HF-layout output")
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"input directory not found: {args.input_dir}")
    if args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_default_device("cpu")
    torch.set_num_threads(args.num_threads)
    dequantize_checkpoint(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
