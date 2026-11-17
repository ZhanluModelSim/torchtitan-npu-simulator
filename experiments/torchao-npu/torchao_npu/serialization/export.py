# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Save a mixed-precision state dict as a HuggingFace-layout safetensors checkpoint."""

__all__ = ["save_hf_safetensors", "to_bytes"]

import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
from torchao.prototype.safetensors.safetensors_utils import ALLOWED_TENSORS_SUBCLASSES

_SINGLE_SHARD_NAME = "model.safetensors"
_INDEX_NAME = "model.safetensors.index.json"
_SHARD_NAME = "model-{:05d}-of-{:05d}.safetensors"

_SIZE_MULTIBYTES = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_SIZE_RE = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)\s*B?\s*", re.IGNORECASE)


def to_bytes(size: int | str) -> int:
    """Parse ``size`` into bytes; accepts ``int`` bytes or strings like ``"5GB"`` / ``"512MB"`` (1024-based)."""
    if isinstance(size, bool) or not isinstance(size, (int, str)):
        raise TypeError(f"size must be an int (bytes) or a string, got {type(size).__name__}")
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        return size
    match = _SIZE_RE.fullmatch(size)
    if match is None:
        raise ValueError(f"Cannot parse size {size!r}; expected e.g. 5368709120, '5GB' or '512MB'.")
    value = int(float(match.group(1)) * _SIZE_MULTIBYTES[match.group(2).upper()])
    if value <= 0:
        raise ValueError(f"size {size!r} rounds to {value} bytes; must be positive.")
    return value


def _inner_tensors(tensor: Any):
    # The torchao flatten protocol: tensor subclasses list their storage
    # tensors in ``tensor_data_names``; plain tensors have no such attribute.
    names = getattr(tensor, "tensor_data_names", None)
    if names is None:
        yield tensor
        return
    for name in names:
        inner = getattr(tensor, name, None)
        if inner is not None:
            yield inner


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Validate the entries and normalize ``nn.Parameter`` to plain detached tensors."""
    normalized: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if isinstance(tensor, nn.Parameter):
            normalized[name] = tensor.detach()
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}: expected torch.Tensor or an allowed tensor subclass, got {type(tensor).__name__}")
        elif getattr(tensor, "tensor_data_names", None) is None:
            normalized[name] = tensor
        elif tensor.__class__.__name__ in ALLOWED_TENSORS_SUBCLASSES:
            if "." not in name:
                raise ValueError(
                    f"tensor name {name!r} must be a dotted FQN (e.g. 'model.embed_tokens.weight'); "
                    "flattening a tensor subclass derives inner-tensor names from the dotted path"
                )
            normalized[name] = tensor
        else:
            raise TypeError(
                f"{name}: unsupported tensor subclass {type(tensor).__name__}; the writer accepts "
                "plain tensors and the registered torchao subclasses (MXTensor) -- convert quantized "
                "weights to MXTensor before export"
            )
        for inner in _inner_tensors(normalized[name]):
            if inner.device.type != "cpu":
                raise ValueError(
                    f"{name}: expected CPU tensors, found device={inner.device}; "
                    "move the checkpoint to CPU before saving"
                )
    return normalized


def _pack_shards(tensors_data: dict[str, torch.Tensor], max_bytes: int):
    """Greedily pack tensors into shards of at most ``max_bytes`` (transformers-style)."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = 0
    total_size = 0
    for name, tensor in tensors_data.items():
        tensor_size = tensor.nelement() * tensor.element_size()
        total_size += tensor_size
        if current_size + tensor_size > max_bytes and current_size > 0:
            shards.append(current)
            current = {}
            current_size = 0
        current[name] = tensor
        current_size += tensor_size
    if current:
        shards.append(current)
    return shards, total_size


def save_hf_safetensors(
    state_dict: dict[str, torch.Tensor],
    output_dir: str | Path,
    max_shard_size: int | str = "5GB",
) -> list[str]:
    """Flatten, shard and save ``state_dict`` as an HF-layout safetensors checkpoint.

    Args:
        state_dict: mapping of full tensor FQNs to ``torch.Tensor`` or
            ``MXTensor`` values. Tensor-subclass (quantized) entries must have
            a dotted FQN -- flattening derives their inner-tensor names via
            ``rsplit(".")`` -- while plain tensors accept any key (the DSv4
            HF layout has top-level dotless keys such as ``hc_head_fn``).
            Every tensor (or inner tensor) must live on CPU.
        output_dir: directory to write into (created if missing).
        max_shard_size: largest shard size in bytes, or a string like ``"5GB"``.
            A tensor larger than the limit still gets a shard of its own when
            the current shard is empty (matching transformers' packing).

    Returns:
        The shard file names (basenames) in shard order.
    """
    if not state_dict:
        raise ValueError("state_dict is empty; nothing to save")
    max_bytes = to_bytes(max_shard_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_state_dict(state_dict)

    # MXTensor -> qdata/scale + metadata; plain tensors pass through.
    tensors_data, metadata = flatten_tensor_state_dict(normalized)
    # save_file rejects non-contiguous tensors.
    tensors_data = {name: (t if t.is_contiguous() else t.contiguous()) for name, t in tensors_data.items()}

    shards, total_size = _pack_shards(tensors_data, max_bytes)

    if len(shards) == 1:
        shard_names = [_SINGLE_SHARD_NAME]
    else:
        shard_names = [_SHARD_NAME.format(i, len(shards)) for i in range(1, len(shards) + 1)]

    # Full metadata in every shard header so any shard is self-describing.
    for shard, name in zip(shards, shard_names, strict=True):
        save_file(shard, str(output_dir / name), metadata=metadata)

    if len(shards) > 1:
        weight_map = {
            tensor_name: name for shard, name in zip(shards, shard_names, strict=True) for tensor_name in shard
        }
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        (output_dir / _INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")

    return shard_names
