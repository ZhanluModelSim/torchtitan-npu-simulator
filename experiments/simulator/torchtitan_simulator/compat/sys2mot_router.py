"""sys2mot fused router-score virtual boundary.

The reference router computes ``raw = sqrt(softplus(W x))`` in float32
(``model.py`` line 352, eight MoE layers). The projection GEMM stays eager
and the ``softplus`` + ``sqrt`` elementwise chain folds into one virtual op
whose math matches ``F.softplus(x, beta=1, threshold=20).sqrt()`` exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("sys2mot_sim::router_score", mutates_args=())
def router_score(score: torch.Tensor, beta: float, threshold: float) -> torch.Tensor:
    return F.softplus(score, beta, threshold).sqrt()


@torch.library.custom_op("sys2mot_sim::router_score_backward", mutates_args=())
def router_score_backward(
    score: torch.Tensor, beta: float, threshold: float, upstream: torch.Tensor
) -> torch.Tensor:
    (d_score,) = torch.autograd.grad(
        F.softplus(score.requires_grad_(), beta, threshold).sqrt(),
        (score,),
        grad_outputs=upstream,
        create_graph=False,
    )
    return d_score


def _router_score_meta(score: torch.Tensor, beta: float, threshold: float) -> torch.Tensor:
    if score.dtype != torch.float32:
        raise ValueError("router score math runs in float32")
    if beta <= 0:
        raise ValueError("router score softplus beta must be positive")
    if threshold < 0:
        raise ValueError("router score softplus threshold must be nonnegative")
    return F.softplus(score, beta, threshold).sqrt().contiguous()


def _router_score_backward_meta(
    score: torch.Tensor, beta: float, threshold: float, upstream: torch.Tensor
) -> torch.Tensor:
    _router_score_meta(score, beta, threshold)
    if upstream.dtype != torch.float32 or upstream.shape != score.shape:
        raise ValueError("router score backward upstream matches the fp32 score shape")
    return torch.empty_like(score)


def _score_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(inputs[0])
    ctx.beta = inputs[1]
    ctx.threshold = inputs[2]


def _score_backward(ctx, grad_output):
    (score,) = ctx.saved_tensors
    return router_score_backward(score, ctx.beta, ctx.threshold, grad_output), None, None


def install_sys2mot_router_meta_support(scope) -> dict[str, str]:
    """Route the sys2mot router's sqrt(softplus) chain through the boundary."""
    torch.library.impl_abstract("sys2mot_sim::router_score")(_router_score_meta)
    torch.library.impl_abstract("sys2mot_sim::router_score_backward")(_router_score_backward_meta)
    torch.library.register_autograd(
        "sys2mot_sim::router_score", _score_backward, setup_context=_score_setup
    )
    import torchtitan_npu.models.sys2mot.model as sys2mot_model

    def router_forward(self, flat: torch.Tensor, image_mask_flat: torch.Tensor):
        raw = router_score(F.linear(flat.float(), self.weight.float()), 1.0, 20.0)
        bias = torch.where(image_mask_flat[:, None], self.bias_image.float(), self.bias_text.float())
        ids = (raw + bias).topk(self.topk, -1).indices
        if self._debug_force_load_balance:
            n_tokens = ids.shape[0]
            ids = (
                torch.arange(n_tokens * self.topk, device=raw.device, dtype=torch.int64).reshape(n_tokens, self.topk)
                % self.num_experts
            )
        weights = raw.gather(1, ids)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20) * self.route_scale
        return ids, weights

    scope.replace(sys2mot_model.Router, "forward", router_forward)
    return {
        "router_score_capture_boundary": "sys2mot_sim.router_score",
        "router_score_backward_capture_boundary": "sys2mot_sim.router_score_backward",
    }


__all__ = ["install_sys2mot_router_meta_support"]
