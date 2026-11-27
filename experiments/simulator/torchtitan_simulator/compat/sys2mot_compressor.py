"""sys2mot fused compressor gated-pooling virtual boundary.

The reference compressor (``model.py`` lines 767-771) computes the ratio>1
token pooling as an eager chain: ``prob = softmax(masked_fill(score)) * vm``
followed by a clamp-min renormalization and the fp32 weighted sum over the
window.  The boundary folds the whole chain into one virtual op; the kv/gate
projections and the output RMSNorm stay eager, and the ratio==1 squeeze path
is untouched.
"""

from __future__ import annotations

import torch


def _pool_math(y: torch.Tensor, score: torch.Tensor, vm: torch.Tensor) -> torch.Tensor:
    prob = score.masked_fill(~vm[..., None], -1e9).softmax(2) * vm[..., None]
    prob = prob / prob.sum(2, keepdim=True).clamp_min(1e-9)
    return (y.float() * prob).sum(2).to(y.dtype)


@torch.library.custom_op("sys2mot_sim::compress_pool", mutates_args=())
def compress_pool(y: torch.Tensor, score: torch.Tensor, vm: torch.Tensor) -> torch.Tensor:
    return _pool_math(y, score, vm)


@torch.library.custom_op("sys2mot_sim::compress_pool_backward", mutates_args=())
def compress_pool_backward(
    y: torch.Tensor, score: torch.Tensor, vm: torch.Tensor, upstream: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    d_y, d_score = torch.autograd.grad(
        _pool_math(y.requires_grad_(), score.requires_grad_(), vm),
        (y, score),
        grad_outputs=upstream,
        create_graph=False,
    )
    return d_y, d_score


def _pool_meta(y: torch.Tensor, score: torch.Tensor, vm: torch.Tensor) -> torch.Tensor:
    if y.dim() != 4 or score.shape != y.shape:
        raise ValueError("compress pool reads (batch, count, window, width) streams")
    if vm.dtype != torch.bool or vm.shape != y.shape[:3]:
        raise ValueError("compress pool validity mask matches the window grid")
    if score.dtype != torch.float32:
        raise ValueError("compress pool scores run in float32")
    if y.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("compress pool reads the model stream dtype")
    batch, count, _, width = y.shape
    return torch.empty((batch, count, width), dtype=y.dtype, device=y.device)


def _pool_backward_meta(
    y: torch.Tensor, score: torch.Tensor, vm: torch.Tensor, upstream: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _pool_meta(y, score, vm)
    if upstream.shape != y.shape[:2] + y.shape[3:] or upstream.dtype != y.dtype:
        raise ValueError("compress pool backward upstream matches the pooled shape")
    return torch.empty_like(y), torch.empty_like(score)


def _pool_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2])


def _pool_backward(ctx, grad_output):
    y, score, vm = ctx.saved_tensors
    d_y, d_score = compress_pool_backward(y, score, vm, grad_output)
    return d_y, d_score, None


def install_sys2mot_compressor_meta_support(scope) -> dict[str, str]:
    """Route the sys2mot compressor's gated pooling through the boundary."""
    torch.library.impl_abstract("sys2mot_sim::compress_pool")(_pool_meta)
    torch.library.impl_abstract("sys2mot_sim::compress_pool_backward")(_pool_backward_meta)
    torch.library.register_autograd(
        "sys2mot_sim::compress_pool", _pool_backward, setup_context=_pool_setup
    )
    import torchtitan_npu.models.sys2mot.model as sys2mot_model

    def compressor_forward(self, x: torch.Tensor, valid: torch.Tensor):
        r, length = self.ratio, x.shape[1]
        count = length // r
        y = self.kv(x[:, : count * r]).view(x.shape[0], count, r, -1)
        vm = valid[:, : count * r].view(x.shape[0], count, r)
        if r > 1:
            score = self.gate(x[:, : count * r]).view_as(y).float()
            pooled = compress_pool(y, score, vm)
        else:
            pooled = y.squeeze(2)
        return self.norm(pooled), vm.any(-1)

    scope.replace(sys2mot_model.Compressor, "forward", compressor_forward)
    return {
        "compress_pool_capture_boundary": "sys2mot_sim.compress_pool",
        "compress_pool_backward_capture_boundary": "sys2mot_sim.compress_pool_backward",
    }


__all__ = ["install_sys2mot_compressor_meta_support"]
