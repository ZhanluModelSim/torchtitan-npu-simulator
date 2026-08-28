# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview NPU converter for the MHC stream-mixing path.

Replaces each ``TransformerLayer``'s pure-torch MHC mixes with the fused
triton ops of ``torchtitan_npu.ops.triton.mhc_triton``, bound to the SAME
layer parameters (no state-dict key changes):

- pre-attn / pre-mlp mix (``_mhc_project`` + ``sigmoid_affine`` +
  ``apply_hpre`` in models/magi2_preview/mhc.py) -> ``MHCPreTriton``
  (RMSNorm + phi projection + sigmoid/sinkhorn coefficients + pre bmm);
- post/res stream update (``hyper_connect``) -> ``MHCPostTriton``.

Argument mapping notes:
- ``mhc_phi_fused_*`` is stored as ``(num_stream * hidden, phi_out)``; the
  triton op transposes its weight argument internally, so it receives
  ``phi.t()`` and its returned gradient lands back in the stored layout.
- The six alphas/biases of one mix are packed into the triton
  ``hc_scale (3,)`` / ``hc_base ((2 + n) * n,)`` layout (pre, post, res
  order) with the layer's ``mhc_matmul_scale`` folded into the scales. The
  triton op bakes in the 2.0 post sigmoid scale, matching
  ``sigmoid_affine(sigmoid_scale=2.0)``.
- The MHC norm gain (``mhc_norm.weight + weight_bias``) is passed as the
  triton RMSNorm gamma; ``num_modality > 1`` layers run the op once per
  modality segment (rows are modality-sorted and every step is row-local).
- MAGI-2's ``hyper_connect`` indexes ``new_i = sum_j h_res[i, j] * s_j``
  while ``MHCPostTriton`` follows the DeepSeek-V4 comb convention (the
  transpose of that), so the sinkhorn coefficients are transposed when
  handed to the post op.
- ``MHCPreOnlyTriton`` (the HcHead-style aggregation) has no MAGI-2
  counterpart: the PostAdapter normalizes the full flattened stream, so it
  is not bound here.

Numerical fidelity: the triton sinkhorn kernel runs a slightly different
schedule than the phase-1 torch ``sinkhorn_knopp`` (row softmax with an
additive eps first, then alternating row/column norms; it also reuses the
norm eps as the sinkhorn smoothing eps). At MHC res-logit magnitudes near
initialization the fp32 outputs/grads agree to ~1e-5; very large logits
slow both schedules' convergence and widen the gap (same caveat as the
DeepSeek-V4 triton path, whose torch reference shares this schedule).

The triton kernels hardcode 4 streams; layers with another ``num_stream``
are skipped and keep the pure-torch path.
"""

import logging

import torch
from torch import nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.magi2_preview.model import TransformerLayer

try:
    from torchtitan_npu.ops.triton import MHCPostTriton, MHCPreTriton
except ImportError:
    # triton / triton-ascend unavailable (e.g. CPU-only environments): the
    # converter can still be imported; converting raises a clear error.
    MHCPostTriton = None
    MHCPreTriton = None

logger = logging.getLogger(__name__)

# Torch path default of ``sinkhorn_knopp``; matches the official inference
# ``MHCHandler`` (num_sk_iters=20).
_SINKHORN_ITERS = 20
# The triton pre/post/bmm kernels hardcode 4 streams.
_SUPPORTED_NUM_STREAM = 4


def _as_int_splits(m_splits) -> list[int]:
    splits = m_splits.tolist() if isinstance(m_splits, torch.Tensor) else m_splits
    return [int(s) for s in splits]


class NpuMagi2TransformerLayer(TransformerLayer):
    """TransformerLayer with its MHC mixes bound to the fused triton ops.

    Shares the parent's parameters, children and plain attributes via a
    shallow ``__dict__`` update (same pattern as ``mhc_prepost.NpuHcPre``),
    so state-dict keys are unchanged. The attention and MLP children keep
    their pure-torch implementations; only the MHC mix math is replaced.
    """

    def __init__(self, parent: TransformerLayer) -> None:
        self.__dict__.update(parent.__dict__)

    def _mhc_pre_mix(
        self,
        s_flat: torch.Tensor,
        phi: torch.Tensor,
        alpha_pre: torch.Tensor,
        alpha_post: torch.Tensor,
        alpha_res: torch.Tensor,
        bias_pre: torch.Tensor,
        bias_post: torch.Tensor,
        bias_res: torch.Tensor,
        m_splits,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused norm + phi + sigmoid/sinkhorn + pre-bmm for one sublayer.

        Returns ``(mixed_in, h_post, h_res)`` with a leading batch dim of 1:
        ``(1, T, hidden)``, ``(1, T, n)``, ``(1, T, n, n)``. ``mixed_in`` is
        the sublayer input; ``h_post``/``h_res`` are the already-gated
        coefficients consumed by ``_mhc_post_mix``.
        """
        hc_scale = (
            torch.cat([alpha_pre, alpha_post, alpha_res]) * self.mhc_matmul_scale
        )
        hc_base = torch.cat([bias_pre, bias_post, bias_res.reshape(-1)])
        x = s_flat.unsqueeze(0)
        norm = self.mhc_norm
        num_modality = norm.num_modality
        if num_modality == 1:
            gain = norm.weight + norm.weight_bias
            return MHCPreTriton.apply(
                x,
                phi.t(),
                hc_scale,
                hc_base,
                gain,
                True,
                self.num_stream,
                _SINKHORN_ITERS,
                norm.eps,
            )

        # num_modality > 1: the triton RMSNorm carries one gamma per call, so
        # run the fused op per modality segment. Rows are modality-sorted and
        # every step is row-local, so this is mathematically identical.
        splits = _as_int_splits(m_splits)
        if len(splits) != num_modality:
            raise ValueError(
                f"Expected {num_modality} m_splits entries, got {len(splits)}"
            )
        ys, posts, ress = [], [], []
        for seg, weight_chunk in zip(
            torch.split(x, splits, dim=1),
            norm.weight.chunk(num_modality),
            strict=True,
        ):
            if seg.shape[1] == 0:
                ys.append(seg.new_empty((1, 0, self.hidden_size)))
                posts.append(seg.new_empty((1, 0, self.num_stream)))
                ress.append(
                    seg.new_empty((1, 0, self.num_stream, self.num_stream))
                )
                continue
            gain = weight_chunk + norm.weight_bias
            y_i, post_i, res_i = MHCPreTriton.apply(
                seg,
                phi.t(),
                hc_scale,
                hc_base,
                gain,
                True,
                self.num_stream,
                _SINKHORN_ITERS,
                norm.eps,
            )
            ys.append(y_i)
            posts.append(post_i)
            ress.append(res_i)
        return (
            torch.cat(ys, dim=1),
            torch.cat(posts, dim=1),
            torch.cat(ress, dim=1),
        )

    def _mhc_post_mix(
        self,
        sublayer_out: torch.Tensor,
        s_flat: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        """Fused ``hyper_connect``; returns the updated stream ``(T, n*C)``."""
        # MAGI-2 indexes new_i = sum_j h_res[i, j] * s_j while MHCPostTriton
        # follows the DeepSeek-V4 comb convention (the transpose of that),
        # hence the transpose of the sinkhorn coefficients.
        new_flat = MHCPostTriton.apply(
            sublayer_out.unsqueeze(0),
            s_flat.unsqueeze(0),
            h_post,
            h_res.transpose(-1, -2),
        )
        return new_flat.squeeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        sort_idx: torch.Tensor,
        inv_sort_idx: torch.Tensor,
        m_splits: list[int],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-attn mix -> attention -> post/res attn stream update.
        attn_in, h_post_attn, h_res_attn = self._mhc_pre_mix(
            hidden_states,
            self.mhc_phi_fused_attn,
            self.mhc_alpha_pre_attn,
            self.mhc_alpha_post_attn,
            self.mhc_alpha_res_attn,
            self.mhc_bias_pre_attn,
            self.mhc_bias_post_attn,
            self.mhc_bias_res_attn,
            m_splits,
        )
        attn_out = self.attention(
            attn_in.squeeze(0), rope, m_splits, sort_idx, inv_sort_idx, cu_seqlens
        )
        s_flat = self._mhc_post_mix(attn_out, hidden_states, h_post_attn, h_res_attn)

        # Pre-mlp mix -> MLP -> post/res mlp stream update.
        mlp_in, h_post_mlp, h_res_mlp = self._mhc_pre_mix(
            s_flat,
            self.mhc_phi_fused_mlp,
            self.mhc_alpha_pre_mlp,
            self.mhc_alpha_post_mlp,
            self.mhc_alpha_res_mlp,
            self.mhc_bias_pre_mlp,
            self.mhc_bias_post_mlp,
            self.mhc_bias_res_mlp,
            m_splits,
        )
        mlp_out = self.mlp(mlp_in.squeeze(0), m_splits)
        return self._mhc_post_mix(mlp_out, s_flat, h_post_mlp, h_res_mlp)


class NpuMagi2MHCConverter(ModelCustomConverter):
    """Swap MAGI-2-preview TransformerLayers for the fused-MHC variants.

    Idempotent: only exact ``TransformerLayer`` instances are replaced, so a
    second pass finds nothing to do. Layers that do not fit the triton ops
    (``num_stream != 4``) are skipped and keep the pure-torch path.
    """

    def convert(self, model: nn.Module) -> None:
        if MHCPreTriton is None or MHCPostTriton is None:
            raise RuntimeError(
                "npu_magi2_mhc requires torchtitan_npu.ops.triton "
                "(triton / triton-ascend must be importable)"
            )
        converted = 0
        skipped = 0
        for name, module in list(model.named_modules()):
            if type(module) is not TransformerLayer:
                continue
            if module.num_stream != _SUPPORTED_NUM_STREAM:
                skipped += 1
                continue
            replace_module_with_name(model, name, NpuMagi2TransformerLayer(module))
            converted += 1
        if skipped:
            logger.warning(
                "[npu_magi2_mhc] Skipped %d TransformerLayer(s) with "
                "num_stream != %d (the triton MHC kernels hardcode 4 streams)",
                skipped,
                _SUPPORTED_NUM_STREAM,
            )
        logger.info(
            "[npu_magi2_mhc] Converted %d TransformerLayer(s) to fused MHC triton ops",
            converted,
        )


@register_model_converter("npu_magi2_mhc")
class Magi2MHCModelConfig(ModelCustomConfig):
    model_converter = NpuMagi2MHCConverter
