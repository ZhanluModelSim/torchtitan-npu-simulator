# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert dequantized DeepSeek-V4 Hugging Face weights to TorchTitan DCP.

The input must use Hugging Face tensor names and BF16 Safetensors. The current
``DeepSeekV4StateDictAdapter`` performs all HF-to-TorchTitan key conversion and
stacks per-expert weights. The resulting DCP uses the current routed-expert
schema::

    layers.N.moe.routed_experts.inner_experts.w1_EFD
    layers.N.moe.routed_experts.inner_experts.w2_EDF
    layers.N.moe.routed_experts.inner_experts.w3_EFD

Run from the repository root. Tensor conversion and DCP writing use CPU memory;
the Python environment must still be able to import TorchTitan and torchtitan-npu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

from scripts.numerical_alignment.deepseek_v4 import config as dsv4_config

_CURRENT_MOE_PREFIX = ".moe.routed_experts.inner_experts."
_LEGACY_MOE_PREFIX = ".moe.experts."
_EXPERT_WEIGHTS = ("w1_EFD", "w2_EDF", "w3_EFD")


def _keep_hf_key(key: str, n_layers: int) -> bool:
    if key.startswith("layers."):
        return int(key.split(".")[1]) < n_layers
    return not key.startswith("mtp.")


def _validate_converted_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_config,
) -> None:
    n_layers = model_config.n_layers
    legacy = sorted(key for key in state_dict if _LEGACY_MOE_PREFIX in key)
    if legacy:
        raise RuntimeError(f"adapter produced legacy MoE keys; expected routed_experts schema: {legacy[:3]}")

    missing = []
    for layer_id in range(n_layers):
        prefix = f"layers.{layer_id}{_CURRENT_MOE_PREFIX}"
        missing.extend(prefix + weight for weight in _EXPERT_WEIGHTS if prefix + weight not in state_dict)
    if missing:
        raise RuntimeError(f"converted state dict is missing current MoE weights: {missing[:3]}")

    for layer_id, layer in enumerate(model_config.layers):
        key = f"layers.{layer_id}.attention.wo_a.weight"
        cfg = layer.attention.wo_a
        # ``wo_a`` uses the flattened BatchedLinear layout ``[n_heads * out, in]``
        # (torchtitan_npu/patches/torchtitan/models/common/linear.py); HF stores
        # the same per-group matrices already flattened across group and output.
        expected_shape = (cfg.n_heads * cfg.out_features, cfg.in_features)
        actual_shape = tuple(state_dict[key].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(f"converted {key} has shape {actual_shape}; expected {expected_shape}")


def _read_hf_state_dict(
    input_dir: Path,
    n_layers: int,
    *,
    num_experts: int | None = None,
    top_k: int | None = None,
) -> dict[str, torch.Tensor]:
    single_file = input_dir / "model.safetensors"
    shards = sorted(input_dir.glob("model-*.safetensors"))
    if single_file.is_file():
        if shards:
            raise ValueError(f"both model.safetensors and sharded model files exist under {input_dir}")
        shards = [single_file]
    if not shards:
        raise FileNotFoundError(f"no model Safetensors found under {input_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for key in reader.keys():  # noqa: SIM118  (safe_open is not iterable)
                keep = (
                    _keep_hf_key(key, n_layers)
                    if num_experts is None
                    else dsv4_config.keep_hf_key(key, n_layers, num_experts)
                )
                if not keep:
                    continue
                if key in state_dict:
                    raise ValueError(f"duplicate checkpoint key {key!r} in {shard.name}")
                tensor = reader.get_tensor(key)
                if key.endswith(".scale") or tensor.dtype in (
                    torch.float8_e4m3fn,
                    torch.int8,
                ):
                    raise ValueError(f"{key} is still quantized; run dequant_save_hf_dsv4.py first")
                if num_experts is not None and top_k is not None:
                    resized = dsv4_config.resize_hf_router_tensor(
                        key,
                        tensor,
                        num_experts=num_experts,
                        top_k=top_k,
                    )
                    if resized is not None:
                        tensor = resized
                state_dict[key] = tensor
    return state_dict


def convert_hf_to_dcp(
    input_dir: Path,
    output_dir: Path,
    n_layers: int,
    model_flavor: str = "deepseek_v4_flash",
    num_experts: int | None = None,
    top_k: int | None = None,
) -> None:
    from torchtitan_npu.models.deepseek_v4.state_dict_adapter import (
        DeepSeekV4StateDictAdapter,
    )

    config = dsv4_config.build_torchtitan_config(model_flavor)
    model_config = config.model_spec.model
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    dsv4_config.truncate_model_config(model_config, n_layers)
    if (num_experts is None) != (top_k is None):
        raise ValueError("num_experts and top_k must be set together")
    if num_experts is not None and top_k is not None:
        dsv4_config.configure_reduced_moe(
            model_config,
            num_experts=num_experts,
            top_k=top_k,
        )
    adapter = DeepSeekV4StateDictAdapter(
        model_config,
        hf_assets_path=None,
    )

    hf_state_dict = _read_hf_state_dict(
        input_dir,
        n_layers,
        num_experts=num_experts,
        top_k=top_k,
    )
    print(
        f"[convert] read {len(hf_state_dict)} HF tensors from {input_dir}",
        flush=True,
    )

    state_dict = adapter.from_hf(hf_state_dict)
    del hf_state_dict
    _validate_converted_state_dict(state_dict, model_config)
    moe_keys = sum(_CURRENT_MOE_PREFIX in key for key in state_dict)
    print(
        f"[convert] from_hf -> {len(state_dict)} TorchTitan tensors ({moe_keys} routed-expert tensors)",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dcp.save(state_dict, checkpoint_id=str(output_dir))
    print(f"[convert] saved DCP checkpoint -> {output_dir}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="dequantized BF16 Hugging Face checkpoint directory",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="new TorchTitan DCP checkpoint directory",
    )
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument(
        "--model-flavor",
        choices=dsv4_config.MODEL_FLAVORS,
        default="deepseek_v4_flash",
    )
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"input directory not found: {args.input_dir}")
    if args.n_layers <= 0:
        parser.error("--n-layers must be positive")
    if (args.num_experts is None) != (args.top_k is None):
        parser.error("--num-experts and --top-k must be set together")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_default_device("cpu")
    convert_hf_to_dcp(
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        args.n_layers,
        args.model_flavor,
        args.num_experts,
        args.top_k,
    )


if __name__ == "__main__":
    main()
