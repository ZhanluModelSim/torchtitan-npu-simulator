# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Install Qwen3-Next shape-only kernel bindings in simulator mode.

Binds the MindSpeed ``Qwen3NextMoeExperts`` production op stream as virtual
fused operators (``npu.npu_moe_token_permute`` / ``npu.npu_swiglu`` /
``npu.npu_moe_token_unpermute``, mirroring ``mindspeed.ops`` and
``torch_npu``), plus the fused ``fusion_attention`` virtual operator for the
gated attention core (mirroring MindSpeed ``torch_npu.npu_fusion_attention``).
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate

from torchtitan_npu.models.qwen3_next import loss as qwen3_next_loss
from torchtitan_npu.models.qwen3_next.attention import GatedAttention, RMSNormGated
from torchtitan_npu.models.qwen3_next.feed_forward import Qwen3NextGroupedExperts
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
from torchtitan_npu.simulator.hardware_shims.rms_norm_shim import run_meta_rms_norm
from torchtitan_npu.simulator.synthetic_ac import run_synthetic_op

_FA_SHIM_MARKER = "_simulator_qwen_fa_shim_installed"
_EXPERTS_SHIM_MARKER = "_simulator_qwen_experts_shim_installed"
_GATED_NORM_SHIM_MARKER = "_simulator_qwen_gated_norm_shim_installed"
_NORM_SHIM_MARKER = "_simulator_qwen_norm_shim_installed"


def _local_param(param):
    return param.to_local() if isinstance(param, DTensor) else param


def _sim_qwen_rms_norm(module, x: torch.Tensor) -> torch.Tensor:
    """MindSpeed ``Qwen3NextRMSNorm`` fused sequence.

    ``npu_rms_norm(x.float(), 1.0 + weight.float(), eps)[0].type_as(x)``
    (``use_fused_rmsnorm=True``): the float casts, the ``1.0 + weight``
    offset add, and the output cast are real eager ops in the measured
    stream, so they are emitted here instead of hiding inside the fused
    boundary.
    """
    weight = _local_param(module.weight)
    resolved_eps = (
        module.eps if module.eps is not None else torch.finfo(x.dtype).eps
    )
    out = run_meta_rms_norm(x.float(), 1.0 + weight.float(), resolved_eps)
    return out.type_as(x)


def _sim_qwen_rms_norm_gated(
    module: RMSNormGated, x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """MindSpeed ``Qwen3NextRMSNormGated`` fused sequence.

    ``npu_rms_norm(x.float(), weight.float(), eps)[0] * silu(gate.float())``
    then ``.to(input_dtype)`` — weight has no ``1.0 +`` offset in the gated
    variant.
    """
    weight = _local_param(module.weight)
    normed = run_meta_rms_norm(x.float(), weight.float(), module.eps)
    return (normed * F.silu(gate.float())).to(x.dtype)


def _suspend():
    capture = get_active_capture()
    if capture is None:
        import contextlib

        return contextlib.nullcontext()
    return capture.suspend_recording()


def _module_path() -> str:
    capture = get_active_capture()
    if capture is None or capture.module_path_tracker is None:
        return ""
    return capture.module_path_tracker.current_path()


class _SimMoeTokenPermute(torch.autograd.Function):
    """Virtual ``npu_moe_token_permute``: single fused op in the capture."""

    @staticmethod
    def forward(ctx, tokens: torch.Tensor, indices: torch.Tensor, path: str):
        def factory():
            with _suspend():
                flat = indices.reshape(-1).to(torch.int64)
                row_ids = torch.argsort(flat, stable=True)
                permuted = tokens.index_select(0, row_ids)
                return permuted, row_ids.to(torch.int32)

        permuted, row_ids_map = run_synthetic_op(
            "npu.npu_moe_token_permute.default",
            inputs=[tokens, indices],
            output_factory=factory,
            module_path=path,
        )
        ctx.save_for_backward(row_ids_map)
        ctx.path = path
        ctx.num_tokens = tokens.shape[0]
        return permuted, row_ids_map

    @staticmethod
    def backward(ctx, grad_permuted: torch.Tensor, grad_map: torch.Tensor | None):
        (row_ids_map,) = ctx.saved_tensors
        row_ids = row_ids_map.to(torch.int64)

        def factory():
            with _suspend():
                grad_tokens = torch.zeros(
                    (ctx.num_tokens, grad_permuted.shape[-1]),
                    dtype=grad_permuted.dtype,
                    device=grad_permuted.device,
                )
                grad_tokens.index_add_(0, row_ids, grad_permuted)
                return (grad_tokens,)

        (grad_tokens,) = run_synthetic_op(
            "npu.npu_moe_token_permute_grad.default",
            inputs=[grad_permuted],
            output_factory=factory,
            module_path=ctx.path,
        )
        return grad_tokens, None, None


class _SimNpuSwiglu(torch.autograd.Function):
    """Virtual ``torch_npu.npu_swiglu(x, dim=-1)``: single fused op."""

    @staticmethod
    def forward(ctx, fc1_output: torch.Tensor, path: str):
        def factory():
            with _suspend():
                gate, up = fc1_output.chunk(2, dim=-1)
                return (F.silu(gate) * up,)

        (out,) = run_synthetic_op(
            "npu.npu_swiglu.default",
            inputs=[fc1_output],
            output_factory=factory,
            module_path=path,
        )
        ctx.save_for_backward(fc1_output)
        ctx.path = path
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (fc1_output,) = ctx.saved_tensors

        def factory():
            with _suspend():
                gate, up = fc1_output.chunk(2, dim=-1)
                sig = torch.sigmoid(gate)
                grad_gate = grad_out * up * sig
                grad_up = grad_out * sig
                return (torch.cat([grad_gate, grad_up], dim=-1),)

        (grad_fc1,) = run_synthetic_op(
            "npu.npu_swiglu_backward.default",
            inputs=[grad_out],
            output_factory=factory,
            module_path=ctx.path,
        )
        return grad_fc1, None


class _SimMoeTokenUnpermute(torch.autograd.Function):
    """Virtual ``npu_moe_token_unpermute`` (probs pre-folded into tokens)."""

    @staticmethod
    def forward(
        ctx,
        routed_output: torch.Tensor,
        row_ids_map: torch.Tensor,
        num_tokens: int,
        top_k: int,
        path: str,
    ):
        row_ids = row_ids_map.to(torch.int64)

        def factory():
            with _suspend():
                token_ids = row_ids // top_k
                output = torch.zeros(
                    (num_tokens, routed_output.shape[-1]),
                    dtype=routed_output.dtype,
                    device=routed_output.device,
                )
                output.index_add_(0, token_ids, routed_output)
                return (output,)

        (out,) = run_synthetic_op(
            "npu.npu_moe_token_unpermute.default",
            inputs=[routed_output, row_ids_map],
            output_factory=factory,
            module_path=path,
        )
        ctx.save_for_backward(row_ids_map)
        ctx.path = path
        ctx.num_tokens = num_tokens
        ctx.top_k = top_k
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (row_ids_map,) = ctx.saved_tensors
        row_ids = row_ids_map.to(torch.int64)

        def factory():
            with _suspend():
                token_ids = row_ids // ctx.top_k
                grad_tokens = grad_out.index_select(0, token_ids)
                return (grad_tokens,)

        (grad_routed,) = run_synthetic_op(
            "npu.npu_moe_token_unpermute_grad.default",
            inputs=[grad_out],
            output_factory=factory,
            module_path=ctx.path,
        )
        return grad_routed, None, None, None, None


def _sim_moe_token_permute(
    tokens: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _SimMoeTokenPermute.apply(tokens, indices, _module_path())


def _sim_npu_swiglu(fc1_output: torch.Tensor) -> torch.Tensor:
    return _SimNpuSwiglu.apply(fc1_output, _module_path())


def _sim_moe_token_unpermute(
    routed_output: torch.Tensor,
    row_ids_map: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    return _SimMoeTokenUnpermute.apply(
        routed_output, row_ids_map, num_tokens, top_k, _module_path()
    )


def _sim_qwen_grouped_experts(
    module: Qwen3NextGroupedExperts,
    hidden_states: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """MindSpeed expert core: GMM(gate_up) -> npu_swiglu -> GMM(down)."""
    if isinstance(hidden_states, DTensor):
        hidden_states = hidden_states.to_local()
    gate_up_proj, down_proj = module._view_experts_weight()

    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    fc1_output = torch._grouped_mm(hidden_states, gate_up_proj, offs=offsets)
    fc1_activation = _sim_npu_swiglu(fc1_output)
    fc2_out = torch._grouped_mm(fc1_activation, down_proj, offs=offsets)
    return fc2_out


def _virtual_token_permute(
    hidden_states: torch.Tensor, selected_experts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _sim_moe_token_permute(hidden_states, selected_experts)


def _virtual_token_unpermute(
    routed_output: torch.Tensor,
    row_ids_map: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    return _sim_moe_token_unpermute(routed_output, row_ids_map, num_tokens, top_k)


def sim_gated_attention(
    module: GatedAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Simulator-only fused attention core (GQA head-major q/k/v)."""
    attrs = {"num_heads": int(query.shape[1]), "layout": "BNSD"}
    return _SimFusionAttention.apply(query, key, value, _module_path(), attrs)


class _SimFusionAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, path, attrs):  # noqa: ANN001
        def factory():
            with _suspend():
                return _empty_like(query)

        output = run_synthetic_op(
            "fusion_attention",
            inputs=[query, key, value],
            output_factory=factory,
            module_path=path,
            attrs=attrs,
        )
        ctx.save_for_backward(query, key, value)
        ctx.path = path
        ctx.attrs = attrs
        return output

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        saved = ctx.saved_tensors

        def factory():
            with _suspend():
                return tuple(_empty_like(t) for t in saved)

        grads = run_synthetic_op(
            "fusion_attention_grad",
            inputs=[*saved, grad_output],
            output_factory=factory,
            module_path=ctx.path,
            attrs=ctx.attrs,
        )
        return (*grads, None, None)


def _empty_like(tensor: torch.Tensor) -> torch.Tensor:
    capture = get_active_capture()
    if capture is None:
        return torch.empty_like(tensor)
    with capture.suspend_recording():
        return torch.empty_like(tensor)


def apply_qwen3_next_fusion_shims(model) -> None:
    """Bind fused virtual ops while preserving module hooks."""
    # NOTE: `_token_permute`/`_token_unpermute` in feed_forward.py are eager
    # (argsort + index_select / zeros + index_add_), mirroring the measured
    # MindSpeed run (ep_dispatcher=eager records index_copy_/index_select).
    # Do NOT rebind them to virtual fused npu_moe_token_permute ops — that
    # cut the eager op stream and hid the data-movement semantics.
    _apply_qwen3_next_module_shims(model)


def _apply_qwen3_next_module_shims(model) -> None:
    """Bind Qwen3-Next fused virtual ops onto an already-built model.

    The plain-norm rebinding (MindSpeed ``Qwen3NextRMSNorm`` sequence) must
    not leak into other models sharing the ``NPURMSNorm`` converter class,
    so it is gated on detecting a Qwen3-Next-specific module.
    """
    from types import MethodType

    from torchtitan_npu.converters.kernels.rms_norm import NPURMSNorm

    is_qwen3_next = any(
        isinstance(m, (GatedAttention, RMSNormGated)) for m in model.modules()
    )
    for module in model.modules():
        if isinstance(module, GatedAttention):
            if not getattr(module, _FA_SHIM_MARKER, False):
                module._attention_core = MethodType(sim_gated_attention, module)
                setattr(module, _FA_SHIM_MARKER, True)
        elif isinstance(module, RMSNormGated):
            if not getattr(module, _GATED_NORM_SHIM_MARKER, False):
                module.forward = MethodType(_sim_qwen_rms_norm_gated, module)
                setattr(module, _GATED_NORM_SHIM_MARKER, True)
        elif is_qwen3_next and isinstance(module, NPURMSNorm):
            if not getattr(module, _NORM_SHIM_MARKER, False):
                module.forward = MethodType(_sim_qwen_rms_norm, module)
                setattr(module, _NORM_SHIM_MARKER, True)
        elif isinstance(module, Qwen3NextGroupedExperts):
            if getattr(module, "_ep_dispatcher", False):
                # The MindSpeed eager dispatcher owns this module's op
                # stream (permute/a2a/per-expert matmul loop); the fused
                # gmm virtual ops only apply to the ep=1 path.
                continue
            if not getattr(module, _EXPERTS_SHIM_MARKER, False):
                module.forward = MethodType(_sim_qwen_grouped_experts, module)
                setattr(module, _EXPERTS_SHIM_MARKER, True)

    # MindSpeed chunked loss: the wrapper's forward calls the LM head
    # TWICE unconditionally — first ``self.lm_head(h)`` for full logits
    # (the result is discarded when ``loss_ctx`` is set), then
    # ``self.lm_head(h, loss_ctx=loss_ctx)`` which runs the chunked loss
    # inside the module (so its FSDP unit all-gathers on each call), and
    # the model returns the precomputed loss (the trainer loss fn is a
    # passthrough). Each FSDP unit re-unshards before backward
    # (reshard_after_forward=True), so the head sees three all-gathers.
    if qwen3_next_loss.is_active() and getattr(model, "output", None) is not None:
        model.output.forward = MethodType(_lm_head_forward, model.output)
        model.forward = MethodType(_causal_lm_forward, model)


def _lm_head_forward(self, x: torch.Tensor, loss_ctx=None):
    """MindSpeed ``LMHead.forward``: full-logits linear or the chunked
    loss delegation (``(None, loss_ctx(hidden, weight, bias))``)."""
    weight = _local_param(self.weight)
    if loss_ctx is None:
        return F.linear(x, weight)
    return None, loss_ctx(x, weight)


def _causal_lm_forward(
    self,
    tokens: torch.Tensor,
    attention_masks=None,
    positions=None,
):
    """Mirror MindSpeed ``Qwen3NextForCausalLM.forward`` (chunked-loss
    path) on top of the torchtitan Decoder layout."""
    h = self.tok_embeddings(tokens)
    for layer in self.layers.values():
        h = layer(h, self.freqs_cis, attention_masks, positions)
    h = self.norm(h)
    # Under TP + sequence parallelism the head input is sequence-sharded;
    # the head stays replicated (no measured tp>1 vocab-parallel baseline,
    # and the chunked loss needs full-vocab local logits), so gather the
    # stream once (all-gather) before the two head calls.
    if isinstance(h, DTensor):
        h = h.full_tensor()
    # logits_to_keep=0 -> slice(None): the wrapper slices before each
    # head call; the first head call computes full logits whose result
    # is overwritten (dead) by the loss_ctx call.
    h_slice = h[:, slice(0, None), :]
    self.output(h_slice)
    _, loss = self.output(
        h_slice,
        loss_ctx=qwen3_next_loss.make_chunk_loss_ctx(self._sim_labels),
    )
    return loss
