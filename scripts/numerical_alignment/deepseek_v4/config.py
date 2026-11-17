# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSV4 model and reduced-MoE configuration for numerical checks."""

from __future__ import annotations

import re

import torch

NUM_EXPERTS = 4
TOP_K = 3
N_HASH_LAYERS = 3
HASH_ROUTING_SEED = 0
DEFAULT_MODEL_FLAVOR = "deepseek_v4_flash"
DEFAULT_N_LAYERS = 4
VOCAB_SIZE = 129280

_MODEL_CONFIG_FACTORIES = {
    "debugmodel": "deepseek_v4_debugmodel",
    "deepseek_v4_flash": "deepseek_v4_flash",
}
MODEL_FLAVORS = tuple(_MODEL_CONFIG_FACTORIES)
_EXPERT_KEY = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")
_ROUTER_KEY = re.compile(r"^layers\.(\d+)\.ffn\.gate\.(weight|bias|tid2eid)$")

__all__ = [
    "DEFAULT_MODEL_FLAVOR",
    "DEFAULT_N_LAYERS",
    "MODEL_FLAVORS",
    "NUM_EXPERTS",
    "N_HASH_LAYERS",
    "TOP_K",
    "VOCAB_SIZE",
    "build_torchtitan_config",
    "configure_reduced_moe",
    "keep_hf_key",
    "make_hash_routing_table",
    "resize_hf_router_tensor",
    "truncate_model_config",
]


def build_torchtitan_config(model_flavor: str):
    """Build the registered TorchTitan config for one DSV4 flavor."""

    from torchtitan_npu.models.deepseek_v4 import config_registry

    factory_name = _MODEL_CONFIG_FACTORIES.get(model_flavor)
    if factory_name is None:
        raise ValueError(
            f"unknown DSV4 model flavor {model_flavor!r}; expected one of {sorted(_MODEL_CONFIG_FACTORIES)}"
        )
    # The alignment reference config disables multi-token prediction. The HF
    # view built by ``hf_model`` also sets ``num_nextn_predict_layers=0``;
    # keeping both sides on the same main decoder avoids constructing an
    # unused extra MoE layer and keeps the checkpoint/output contracts equal.
    return getattr(config_registry, factory_name)(num_mtp_layers=0)


def truncate_model_config(model_config, n_layers: int) -> None:
    """Truncate a DSV4 model config to its first ``n_layers`` layers."""

    max_layers = len(model_config.layers)
    if not 1 <= n_layers <= max_layers:
        raise ValueError(f"n_layers must be in [1, {max_layers}], got {n_layers}")
    model_config.n_layers = n_layers
    model_config.layers = model_config.layers[:n_layers]
    model_config.compress_ratios = model_config.compress_ratios[:n_layers]


def configure_reduced_moe(
    model_config,
    *,
    num_experts: int = NUM_EXPERTS,
    top_k: int = TOP_K,
    n_hash_layers: int | None = None,
) -> None:
    """Resize every MoE config while preserving the remaining model shape."""

    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

    if n_hash_layers is None:
        n_hash_layers = min(N_HASH_LAYERS, len(model_config.layers))
    if not 0 <= n_hash_layers <= len(model_config.layers):
        raise ValueError(f"n_hash_layers must be in [0, {len(model_config.layers)}], got {n_hash_layers}")

    for layer_id, layer in enumerate(model_config.layers):
        moe = layer.moe
        if moe is None:
            continue
        moe.num_experts = num_experts
        moe.router.num_experts = num_experts
        moe.router.gate.out_features = num_experts
        moe.router.top_k = top_k
        moe.router.hash = layer_id < n_hash_layers
        moe.routed_experts.inner_experts.num_experts = num_experts
        moe.routed_experts.token_dispatcher.num_experts = num_experts
        moe.routed_experts.token_dispatcher.top_k = top_k


def keep_hf_key(key: str, n_layers: int, num_experts: int = NUM_EXPERTS) -> bool:
    """Keep the selected model layers and the first ``num_experts`` experts."""

    if key.startswith("layers.") and int(key.split(".")[1]) >= n_layers:
        return False
    if key.startswith("mtp."):
        return False
    match = _EXPERT_KEY.match(key)
    return match is None or int(match.group(2)) < num_experts


def resize_hf_router_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    num_experts: int = NUM_EXPERTS,
    top_k: int = TOP_K,
) -> torch.Tensor | None:
    """Return a resized router tensor, or ``None`` for non-router tensors."""

    match = _ROUTER_KEY.match(key)
    if match is None:
        return None
    layer_id = int(match.group(1))
    field = match.group(2)
    if field == "tid2eid":
        return make_hash_routing_table(
            tensor.shape[0],
            num_experts,
            top_k,
            seed=HASH_ROUTING_SEED + layer_id,
        )
    return tensor[:num_experts].contiguous()


def make_hash_routing_table(
    vocab_size: int,
    num_experts: int,
    top_k: int,
    *,
    seed: int,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Build one deterministic hash-routing table without a large score buffer."""

    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    table = torch.empty((vocab_size, top_k), dtype=torch.int64, device="cpu")
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        table[start:end] = (
            torch.rand(
                (end - start, num_experts),
                dtype=torch.float32,
                device="cpu",
                generator=generator,
            )
            .topk(top_k, dim=-1)
            .indices
        )
    return table
