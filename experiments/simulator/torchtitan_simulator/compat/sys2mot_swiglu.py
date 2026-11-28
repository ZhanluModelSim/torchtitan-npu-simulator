"""sys2mot fused clamped-SwiGLU virtual boundary.

The reference runs both activation sites (the dense ``SwiGLU`` MLP and the
MoE ``RoutedExperts`` activation, ``model.py`` lines 193-196 and 443) as
``F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)`` eager chains.
The boundary keeps the three projection GEMMs (or grouped GEMMs) eager and
folds the clamp/silu/multiply chain into one virtual op whose math matches
the reference placement exactly: gate clamped from above before silu, up
clamped on both sides, product unclamped.  ``limit=0`` disables clamping
(dense ``SwiGLU.Config.limit`` default), matching the reference
``if self.limit:`` gate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("sys2mot_sim::swiglu", mutates_args=())
def swiglu(gate: torch.Tensor, up: torch.Tensor, clamp_limit: float) -> torch.Tensor:
    g = gate.clamp(max=clamp_limit) if clamp_limit else gate
    u = up.clamp(-clamp_limit, clamp_limit) if clamp_limit else up
    return F.silu(g) * u


@torch.library.custom_op("sys2mot_sim::swiglu_backward", mutates_args=())
def swiglu_backward(
    gate: torch.Tensor, up: torch.Tensor, clamp_limit: float, upstream: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    def _swiglu_fwd(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        gc = g.clamp(max=clamp_limit) if clamp_limit else g
        uc = u.clamp(-clamp_limit, clamp_limit) if clamp_limit else u
        return F.silu(gc) * uc

    d_gate, d_up = torch.autograd.grad(
        _swiglu_fwd(gate.requires_grad_(), up.requires_grad_()),
        (gate, up),
        grad_outputs=upstream,
        create_graph=False,
    )
    return d_gate, d_up


def _swiglu_meta(gate: torch.Tensor, up: torch.Tensor, clamp_limit: float) -> torch.Tensor:
    if gate.dtype != up.dtype:
        raise ValueError("swiglu gate and up share the model dtype")
    if gate.shape != up.shape:
        raise ValueError("swiglu gate and up share one shape")
    if clamp_limit < 0:
        raise ValueError("swiglu clamp limit is non-negative (0 disables clamping)")
    g = gate.clamp(max=clamp_limit) if clamp_limit else gate
    u = up.clamp(-clamp_limit, clamp_limit) if clamp_limit else up
    return (F.silu(g) * u).contiguous()


def _swiglu_backward_meta(
    gate: torch.Tensor, up: torch.Tensor, clamp_limit: float, upstream: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _swiglu_meta(gate, up, clamp_limit)
    if upstream.shape != gate.shape:
        raise ValueError("swiglu backward upstream matches the activation shape")
    return torch.empty_like(gate), torch.empty_like(up)


def _swiglu_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.clamp_limit = inputs[2]


def _swiglu_backward(ctx, grad_output):
    gate, up = ctx.saved_tensors
    d_gate, d_up = swiglu_backward(gate, up, ctx.clamp_limit, grad_output)
    return d_gate, d_up, None


def _install_swiglu(scope) -> dict[str, str]:
    from torch.distributed.tensor import DTensor

    from torchtitan_npu.models.sys2mot.model import GroupedExperts, SwiGLU

    def swiglu_forward(self, x: torch.Tensor) -> torch.Tensor:
        act = swiglu(self.gate(x), self.up(x), self.limit)
        return self.down(act)

    def routed_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w_gate_EFD, DTensor):
            w_gate_EFD = self.w_gate_EFD.to_local()
            w_up_EFD = self.w_up_EFD.to_local()
            w_down_EDF = self.w_down_EDF.to_local()
        else:
            w_gate_EFD = self.w_gate_EFD
            w_up_EFD = self.w_up_EFD
            w_down_EDF = self.w_down_EDF

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        gate_RF = self._grouped_mm(A=x_RD.bfloat16(), B_t=w_gate_EFD.bfloat16().transpose(-2, -1), offs=offsets_E)
        up_RF = self._grouped_mm(A=x_RD.bfloat16(), B_t=w_up_EFD.bfloat16().transpose(-2, -1), offs=offsets_E)
        act_RF = swiglu(gate_RF, up_RF, self.swiglu_limit)
        return self._grouped_mm(
            A=act_RF.bfloat16(), B_t=w_down_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
        ).type_as(x_RD)

    scope.replace(SwiGLU, "forward", swiglu_forward)
    scope.replace(GroupedExperts, "forward", routed_forward)
    return {
        "swiglu_capture_boundary": "sys2mot_sim.swiglu",
        "swiglu_backward_capture_boundary": "sys2mot_sim.swiglu_backward",
    }


def install_sys2mot_swiglu_meta_support(scope) -> dict[str, str]:
    torch.library.impl_abstract("sys2mot_sim::swiglu")(_swiglu_meta)
    torch.library.impl_abstract("sys2mot_sim::swiglu_backward")(_swiglu_backward_meta)
    torch.library.register_autograd(
        "sys2mot_sim::swiglu", _swiglu_backward, setup_context=_swiglu_setup
    )
    return _install_swiglu(scope)


__all__ = ["install_sys2mot_swiglu_meta_support"]
