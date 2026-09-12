# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Route einsum on MXFP8 weight wrappers to the NPU FP8 path.

torchao's ``MXFP8TrainingWeightWrapperTensor.__torch_function__`` intercepts
only ``_grouped_mm``/``linear``/``mm``/``matmul``/``addmm``; an einsum whose
weight operand is a wrapped 3D ``[n, out, in]`` parameter falls through with
the subclass disabled and silently computes in BF16. ar_llm's
``LatentExpertMLP`` (shared experts) expresses its projections as batched
einsums (``bsd,nld->nbsl`` and friends), so without this patch the shared
experts stay in BF16 under MXFP8 quantization.

The patch lowers such an einsum into ``n`` per-expert ``NpuMXFP8MM`` calls
(the same dense-MXFP8-linear op used elsewhere) and stacks the results.
The lowering applies when the two-operand equation satisfies:

- exactly one operand is the wrapper, 3-D ``[n, out, in]``;
- the shared labels between the two terms are only the contraction label
  (last on both terms) and optionally the wrapper's leading batch label;
- the wrapper's out label does not appear in the activation term;
- the activation's contracted label is its last label (linear-like
  ``A @ W.t()`` per expert);
- the wrapper batch label leads the output, and the remaining output labels
  match the activation-derived per-expert result.

Non-matching einsums fall through to torchao's original behavior, and
baseline (non-quantized) runs are untouched -- the patch only activates when
a wrapped operand is present.
"""

import torch

from torchtitan_npu.patches.torchao_npu.mx_linear import NpuMXFP8MM

_PATCHED_FLAG = "_npu_einsum_patch_applied"


def _try_lower_batched_weight_einsum(cls, args):
    """Return the FP8 result for a supported batched-weight einsum, else None."""
    if len(args) != 3 or not isinstance(args[0], str):
        return None
    equation, *operands = args
    wrapped = [k for k, operand in enumerate(operands) if isinstance(operand, cls)]
    if len(wrapped) != 1:
        return None
    k = wrapped[0]
    wrapper, act = operands[k], operands[1 - k]
    if wrapper.ndim != 3 or not isinstance(act, torch.Tensor):
        return None

    lhs, sep, rhs = equation.replace(" ", "").partition("->")
    if not sep:
        return None
    terms = lhs.split(",")
    if len(terms) != 2:
        return None
    w_term, a_term = terms[k], terms[1 - k]
    if len(w_term) != 3 or len(a_term) < 2:
        return None

    batch, out_label, in_label = w_term[0], w_term[1], w_term[2]
    shared = set(w_term) & set(a_term)
    if shared - {batch, in_label} or in_label not in shared:
        return None
    if out_label in a_term or (batch in a_term and a_term[0] != batch):
        return None
    if not rhs.startswith(batch):
        return None
    expected_tail = (
        a_term[1:-1] + out_label if batch in a_term else a_term[:-1] + out_label
    )
    if rhs[1:] != expected_tail:
        return None

    from torchao.prototype.moe_training.utils import unwrap_weight

    weight = unwrap_weight(wrapper)
    has_batch = batch in a_term
    n = wrapper.shape[0]
    outs = []
    for i in range(n):
        act_i = act[i] if has_batch else act
        outs.append(NpuMXFP8MM.apply(act_i, weight[i]))
    return torch.stack(outs, dim=0)


def _apply_einsum_patch() -> None:
    from torchao.prototype.moe_training.tensor import (
        MXFP8TrainingWeightWrapperTensor,
    )

    if getattr(MXFP8TrainingWeightWrapperTensor, _PATCHED_FLAG, False):
        return

    orig_torch_function = MXFP8TrainingWeightWrapperTensor.__torch_function__

    def _torch_function(cls, func, types, args, kwargs=None):
        if func in (torch.einsum, torch.ops.aten.einsum.default):
            result = _try_lower_batched_weight_einsum(cls, args)
            if result is not None:
                return result
        return orig_torch_function(func, types, args, kwargs or {})

    MXFP8TrainingWeightWrapperTensor.__torch_function__ = classmethod(_torch_function)
    setattr(MXFP8TrainingWeightWrapperTensor, _PATCHED_FLAG, True)


# Apply patches when this module is imported (skip if torchao is not installed)
try:
    _apply_einsum_patch()
except ModuleNotFoundError:
    from torchtitan.tools.logging import logger

    logger.warning(
        "torchao is not installed, and the MXFP8 wrapper einsum patch is skipped. "
        "einsum-based modules (e.g. ar_llm shared experts) will stay in BF16 "
        "under MXFP8 quantization."
    )
