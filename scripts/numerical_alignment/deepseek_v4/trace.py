# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Map DSV4 implementation modules to stable numerical trace keys.

Model field names are intentionally isolated in this adapter. The comparison
entry points and shared hook utilities only consume the canonical trace keys.
"""

from __future__ import annotations

from typing import Any

import torch

from ..utils import trace_hooks

__all__ = ["register_torchtitan", "register_transformers"]


def _require_attr(root: Any, path: str) -> Any:
    value = root
    for name in path.split("."):
        if not hasattr(value, name):
            raise RuntimeError(f"unsupported DSV4 model layout: missing {path!r} at {type(value).__name__}.{name}")
        value = getattr(value, name)
    return value


def _record_hc(trace, prefix: str, output) -> None:
    post, comb, collapsed = output
    trace_hooks.record_trace(trace, f"{prefix}.post", post)
    trace_hooks.record_trace(trace, f"{prefix}.comb", comb)
    trace_hooks.record_trace(trace, f"{prefix}.collapsed", collapsed)


def register_transformers(model, trace) -> trace_hooks.TraceHookRegistry:
    """Register canonical trace hooks on a Transformers DSV4 model."""
    return _register(_require_attr(model, "model"), trace, transformers=True)


def register_torchtitan(model, trace) -> trace_hooks.TraceHookRegistry:
    """Register canonical trace hooks on a TorchTitan DSV4 model."""
    return _register(model, trace, transformers=False)


def _register(core, trace, *, transformers: bool) -> trace_hooks.TraceHookRegistry:
    registry = trace_hooks.TraceHookRegistry(trace)
    registry.forward(_require_attr(core, "embed_tokens" if transformers else "tok_embeddings"), "embedding")
    registry.forward(_require_attr(core, "hc_head"), "hc_head")
    registry.forward(_require_attr(core, "norm"), "final_norm")
    layers = _require_attr(core, "layers")
    for layer_id, layer in enumerate(layers.values() if hasattr(layers, "values") else layers):
        prefix = f"layers.{layer_id}"
        attention = _require_attr(layer, "self_attn" if transformers else "attention")
        moe = _require_attr(layer, "mlp" if transformers else "moe")

        def router_hook(_module, _args, output, *, prefix=prefix):
            weights, indices = output[1:] if transformers else output[:2]
            for name, value in (("weights", weights), ("indices", indices)):
                trace_hooks.record_trace(trace, f"{prefix}.moe.router.{name}", value.flatten(0, -2))

        for name, tf_attr, tt_attr in (
            ("attn_hc", "attn_hc", "hc_attn_pre"),
            ("ffn_hc", "ffn_hc", "hc_ffn_pre"),
        ):
            hc = _require_attr(layer, tf_attr if transformers else tt_attr)
            registry.callback(
                hc,
                lambda _module, _args, output, key=f"{prefix}.{name}": _record_hc(
                    trace, key, output if transformers else (output[1], output[2], output[0])
                ),
            )
            if name == "ffn_hc":
                # The FFN input is the attention residual boundary on both models.
                registry.pre(hc, f"{prefix}.attn_post")
        for name, tf_attr, tt_attr in (
            ("attn_norm", "input_layernorm", "attention_norm"),
            ("ffn_norm", "post_attention_layernorm", "ffn_norm"),
        ):
            registry.forward(_require_attr(layer, tf_attr if transformers else tt_attr), f"{prefix}.{name}")
        registry.forward(attention, f"{prefix}.attention")
        for name, tf_attr, tt_attr in (
            ("q_a_proj", "q_a_proj", "wq_a"),
            ("q_a_norm", "q_a_norm", "q_norm"),
            ("q_b_proj", "q_b_proj", "wq_b"),
            ("q_b_norm", "q_b_norm", "q_head_norm"),
            ("kv_proj", "kv_proj", "wkv"),
            ("kv_norm", "kv_norm", "kv_norm"),
            ("o_b_proj", "o_b_proj", "wo_b"),
        ):
            module = _require_attr(attention, tf_attr if transformers else tt_attr)
            if transformers and name == "q_b_norm":
                registry.forward(module, f"{prefix}.attention.{name}", lambda output: output.transpose(1, 2))
            else:
                registry.forward(module, f"{prefix}.attention.{name}")
        registry.pre(_require_attr(attention, "o_b_proj" if transformers else "wo_b"), f"{prefix}.attention.o_a")
        if not transformers:
            inner_attention = _require_attr(attention, "compressed_sparse_attention.inner_attention")

            def attention_core_pre_hook(_module, args, *, prefix=prefix):
                trace_hooks.record_trace(trace, f"{prefix}.attention.q", args[0])
                batch = args[0].shape[0]
                kv = args[1].reshape(batch, -1, args[1].shape[-1])
                if len(args) > 2 and args[2] is not None and args[2].numel() > 0:
                    kv = torch.cat([kv, args[2].reshape(batch, -1, args[2].shape[-1])], dim=1)
                trace_hooks.record_trace(trace, f"{prefix}.attention.kv", kv)

            registry.callback(inner_attention, attention_core_pre_hook, pre=True)
            registry.forward(inner_attention, f"{prefix}.attention.core")
        registry.forward(moe, f"{prefix}.moe")
        registry.callback(_require_attr(moe, "gate" if transformers else "router"), router_hook)
        # Record the routed output after token combine, rather than the expanded
        # [tokens * top_k, dim] inner-expert implementation data.
        registry.forward(
            _require_attr(moe, "experts" if transformers else "routed_experts"),
            f"{prefix}.moe.experts",
            lambda output: output.flatten(0, -2),
        )
        shared = getattr(moe, "shared_experts", None)
        if shared is not None:
            registry.forward(shared, f"{prefix}.moe.shared", lambda output: output.flatten(0, -2))
        registry.forward(layer, f"{prefix}.output")
        compressor = getattr(attention, "compressor", None)
        if compressor is not None:
            if transformers:
                registry.forward(
                    compressor,
                    f"{prefix}.attention.compressor",
                    lambda output: (output[0] if isinstance(output, tuple) else output).squeeze(1).flatten(0, -2),
                )
            else:
                registry.forward(compressor, f"{prefix}.attention.compressor")
            for name, tf_attr, tt_attr in (
                ("kv_proj", "kv_proj", "wkv"),
                ("gate_proj", "gate_proj", "wgate"),
                ("kv_norm", "kv_norm", "norm"),
            ):
                module = _require_attr(compressor, tf_attr if transformers else tt_attr)
                if transformers and name == "kv_norm":
                    registry.forward(module, f"{prefix}.compressor.{name}", lambda output: output.flatten(0, -2))
                else:
                    registry.forward(module, f"{prefix}.compressor.{name}")
    return registry
