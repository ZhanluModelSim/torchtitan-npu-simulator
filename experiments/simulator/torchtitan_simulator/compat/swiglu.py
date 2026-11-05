# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-capture bridge for ``cann_ops_nn.swiglu_group``.

Meta capture needs the fused vendor op names and shapes in the graph, but
the capture container may not carry the vendor library at all and the
vendor autograd (when present) may have the wrong gradient arity. The
bridge registers shape-only stand-ins when the vendor library is missing
and wraps the activation in an autograd Function whose backward emits the
``swiglu_group_backward`` node — matching the device profile's fused
kernel pair. Real-device forwards keep calling the vendor activation.

The autograd Function takes only tensor inputs (an optional weight is
passed as a 0-dim placeholder): torch's autograd engine rejects non-None
gradients for non-tensor apply arguments, and returning positional None
gradients for None inputs trips the engine's meta-mode validation.
"""

import importlib
import importlib.util

import torch

_VENDOR_OP = "cann_ops_nn.swiglu_group.default"
_VENDOR_BACKWARD_OP = "cann_ops_nn.swiglu_group_backward.default"

_FAKE_OPS_LIB = None


def _load_cann_ops() -> bool:
    """Load the vendor ``cann_ops_nn`` ops; register shape-only stand-ins when
    the capture container has no vendor library (meta capture only needs the
    op names and shapes to line up with the device profile)."""
    global _FAKE_OPS_LIB
    if importlib.util.find_spec("cann_ops_nn") is not None:
        importlib.import_module("cann_ops_nn.ops")
        return True
    if _FAKE_OPS_LIB is not None:
        return False
    lib = torch.library.Library("cann_ops_nn", "FRAGMENT")
    lib.define("swiglu_group(Tensor x, Tensor? weight, Tensor? group_index, *, float clamp_limit) -> Tensor")
    lib.impl(
        "swiglu_group",
        lambda x, weight=None, group_index=None, *, clamp_limit=0.0: torch.empty(
            (*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device
        ),
        "Meta",
    )
    lib.define(
        "swiglu_group_backward(Tensor grad, Tensor x, Tensor? weight=None, Tensor? y_origin=None,"
        " Tensor? group_index=None, *, float clamp_limit) -> (Tensor, Tensor)"
    )
    lib.impl(
        "swiglu_group_backward",
        lambda grad, x, weight=None, y_origin=None, group_index=None, *, clamp_limit=0.0: (
            torch.empty_like(x),
            torch.empty((), dtype=torch.float32, device=x.device) if weight is None else torch.empty_like(weight),
        ),
        "Meta",
    )
    _FAKE_OPS_LIB = lib
    return False


class _SwigluMetaAutograd(torch.autograd.Function):
    """Fused swiglu forward + ``swiglu_group_backward`` node pair.

    All apply arguments are tensors: ``weight`` is the routed-score vector
    or a 0-dim placeholder, ``clamp_limit`` a 0-dim scalar tensor.
    """

    @staticmethod
    def forward(ctx, x, weight, clamp_limit):
        # clamp_limit rides in save_for_backward as a 0-dim tensor: the spmd
        # type checker invokes backward with a bare ctx that only exposes
        # standard attributes (saved_tensors).
        ctx.save_for_backward(x, weight, clamp_limit)
        return torch.ops.cann_ops_nn.swiglu_group.default(x, weight, None, clamp_limit=clamp_limit.item())

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        x, weight, clamp_limit = ctx.saved_tensors
        dx, dw = torch.ops.cann_ops_nn.swiglu_group_backward.default(
            grad,
            x,
            weight=weight,
            y_origin=None,
            group_index=None,
            clamp_limit=clamp_limit.item(),
        )
        return dx, dw, None


def install_meta_swiglu_autograd(scope):
    import spmd_types

    from torchtitan_npu.override.common.swiglu_group import ascendc

    loaded = _load_cann_ops()
    op = torch.ops.cann_ops_nn.swiglu_group.default
    schema = op._schema
    if [(argument.name, argument.kwarg_only) for argument in schema.arguments] != [
        ("x", False),
        ("weight", False),
        ("group_index", False),
        ("clamp_limit", True),
    ]:
        raise RuntimeError(f"SwiGLU autograd contract requires review: {schema}")
    if loaded:
        with torch.enable_grad():
            x = torch.empty(2, 64, device="meta", requires_grad=True)
            try:
                torch.autograd.grad(op(x).sum(), x)
            except RuntimeError as error:
                if "incorrect number of gradients" not in str(error):
                    raise
            else:
                return "native"
    spmd_types.register_local_autograd_function(_SwigluMetaAutograd)
    original = ascendc.swiglu_group_activation

    def activation(h, swiglu_limit=None, routed_scores=None):
        if h.device.type != "meta":
            return original(h, swiglu_limit, routed_scores)
        weight = (
            torch.zeros((), dtype=torch.float32, device=h.device)
            if routed_scores is None
            else routed_scores.to(dtype=torch.float32).contiguous()
        )
        limit = torch.tensor(-1.0 if swiglu_limit is None else float(swiglu_limit))
        return _SwigluMetaAutograd.apply(h.contiguous(), weight, limit)

    scope.replace(ascendc, "swiglu_group_activation", activation)
    return "bridge_fake_vendor_ops" if not loaded else "bridge_vendor_gradient_arity"
