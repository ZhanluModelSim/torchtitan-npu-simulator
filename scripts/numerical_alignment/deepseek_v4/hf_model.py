# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transformers adapter for the DSV4 numerical-alignment reference."""

from __future__ import annotations

import gc
import json
import tempfile
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import torch

from . import config as dsv4_config

EXPECTED_TRANSFORMERS_VERSION = "5.13.0"

__all__ = [
    "EXPECTED_TRANSFORMERS_VERSION",
    "ensure_transformers_version",
    "forward_logits",
    "prepare_checkpoint_view",
]


_COMPRESS_RATIOS = (1, 1, 4, 128)
_LAYER_TYPE = {
    1: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


def ensure_transformers_version() -> str:
    """Load Transformers and reject unpinned versions.

    DSV4 numerical alignment depends on the model implementation and tensor
    execution order from one exact Transformers release. Failing early keeps
    a silently upgraded/downgraded package from being mistaken for a model
    regression.
    """

    import transformers

    actual = getattr(transformers, "__version__", "unknown")
    if actual != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "DSV4 numerical alignment requires "
            f"transformers=={EXPECTED_TRANSFORMERS_VERSION}, got {actual!r}. "
            "Install scripts/numerical_alignment/requirements.txt in an isolated environment before comparing."
        )
    return actual


def _config(n_layers: int):
    ensure_transformers_version()
    from transformers import DeepseekV4Config

    if not 1 <= n_layers <= len(_COMPRESS_RATIOS):
        raise ValueError(f"n_layers must be in [1, {len(_COMPRESS_RATIOS)}], got {n_layers}")

    config_path = Path(__file__).parent.parent / "configs" / "dsv4_4l.json"
    config = DeepseekV4Config.from_pretrained(config_path)
    config.num_hidden_layers = n_layers
    config.layer_types = [_LAYER_TYPE[ratio] for ratio in _COMPRESS_RATIOS[:n_layers]]
    config.mlp_layer_types = [
        "hash_moe" if layer_id < dsv4_config.N_HASH_LAYERS else "moe" for layer_id in range(n_layers)
    ]
    return config


@contextmanager
def prepare_checkpoint_view(path: Path, n_layers: int):
    """Create a temporary HF view of a layer-limited BF16 checkpoint."""

    from safetensors import safe_open
    from safetensors.torch import save_file

    single_file = path / "model.safetensors"
    shards = sorted(path.glob("model-*.safetensors"))
    if single_file.is_file():
        if shards:
            raise ValueError(f"both model.safetensors and sharded model files exist under {path}")
        shards = [single_file]
    if not shards:
        raise FileNotFoundError(f"no model Safetensors found under {path}")

    with tempfile.TemporaryDirectory(prefix="dsv4-hf-") as tmp:
        view = Path(tmp)
        weight_map: dict[str, str] = {}
        for shard in shards:
            selected: dict[str, torch.Tensor] = {}
            with safe_open(shard, framework="pt", device="cpu") as reader:
                keys = reader.keys()
                for key in keys:
                    if not dsv4_config.keep_hf_key(key, n_layers):
                        continue
                    if key in weight_map:
                        raise ValueError(f"duplicate checkpoint key {key!r} in {shard.name}")
                    tensor = reader.get_tensor(key)
                    if key.endswith(".scale") or tensor.dtype in (torch.float8_e4m3fn, torch.int8):
                        raise ValueError(f"{key} is still quantized; use a dequantized BF16 HF checkpoint")
                    resized = dsv4_config.resize_hf_router_tensor(key, tensor)
                    selected[key] = tensor if resized is None else resized
            if selected:
                save_file(selected, view / shard.name)
                weight_map.update(dict.fromkeys(selected, shard.name))

        config = _config(n_layers)
        config.save_pretrained(view)
        index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        (view / "model.safetensors.index.json").write_text(
            json.dumps(index),
            encoding="utf-8",
        )
        yield view, config


@torch.no_grad()
def forward_logits(
    checkpoint: Path,
    inputs: torch.Tensor,
    n_layers: int,
    device: torch.device,
    *,
    apply_patch: bool = False,
    trace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Load the Transformers reference, run it once, and return CPU logits."""

    ensure_transformers_version()
    from transformers import DeepseekV4ForCausalLM
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as modeling

    tf_patch = import_module(f"{__package__}.transformers_dsv4_patch") if apply_patch else None
    trace_adapter = import_module(f"{__package__}.trace") if trace is not None else None
    if trace is not None:
        from ..utils import trace_hooks

    with prepare_checkpoint_view(checkpoint, n_layers) as (view, config):
        print("[golden] loading Transformers checkpoint on CPU", flush=True)
        model, loading_info = DeepseekV4ForCausalLM.from_pretrained(
            view,
            config=config,
            dtype=torch.bfloat16,
            local_files_only=True,
            output_loading_info=True,
        )
        missing = loading_info.get("missing_keys", [])
        mismatched = loading_info.get("mismatched_keys", [])
        if missing or mismatched:
            raise RuntimeError(
                f"HF checkpoint does not match the inference reference: "
                f"missing={missing[:3]}, mismatched={mismatched[:3]}"
            )
        model.eval()

        original_attention = modeling.eager_attention_forward
        original_rope_apply = modeling.apply_rotary_pos_emb
        attention_forward = original_attention
        if tf_patch is not None:
            print(
                "[golden] applying Transformers numerical-alignment patch",
                flush=True,
            )
            tf_patch.patch_transformers_for_tt(model)
            attention_forward = tf_patch.tt_transformers_attention_forward

        if trace is not None:

            def traced_attention_forward(module, query, key, value, *args, **kwargs):
                prefix = f"layers.{module.layer_idx}.attention"
                trace_hooks.record_trace(
                    trace,
                    f"{prefix}.q",
                    query.transpose(1, 2),
                )
                trace_hooks.record_trace(
                    trace,
                    f"{prefix}.kv",
                    key.squeeze(1),
                )
                output = attention_forward(
                    module,
                    query,
                    key,
                    value,
                    *args,
                    **kwargs,
                )
                trace_hooks.record_trace(trace, f"{prefix}.core", output[0])
                return output

            modeling.eager_attention_forward = traced_attention_forward
        elif tf_patch is not None:
            modeling.eager_attention_forward = attention_forward
        if tf_patch is not None:
            modeling.apply_rotary_pos_emb = tf_patch.tt_transformers_apply_rotary_pos_emb

        hook_registry = None
        try:
            hook_registry = trace_adapter.register_transformers(model, trace) if trace_adapter is not None else None
            if tf_patch is not None:
                tf_patch.set_rope_debug_trace(trace)
            model.to(device)
            logits = model(
                inputs.to(device),
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :]
            result = logits.cpu()
            if trace is not None:
                trace_hooks.record_trace(trace, "logits", result)
        finally:
            if tf_patch is not None:
                tf_patch.set_rope_debug_trace(None)
            modeling.eager_attention_forward = original_attention
            modeling.apply_rotary_pos_emb = original_rope_apply
            if hook_registry is not None:
                hook_registry.remove()
        del model, logits

    gc.collect()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return result
