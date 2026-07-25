"""Swap-memory optimizer for Ascend NPU.

Allocates Adam/AdamW optimizer states (exp_avg, exp_avg_sq) in NPU swap memory
via ``torch_npu.empty_with_swapped_memory``, offloading ~2x parameter memory
from HBM to host while keeping the fused optimizer kernel on the fast path.

Swap tensor ``.cpu()`` requires Ascend HDK >= 26.0.rc1 (VA normalization),
letting DCP serialize swap tensors directly during checkpointing.

A ``register_load_state_dict_post_hook`` replaces loaded regular tensors with
swap-memory copies after PyTorch's standard ``load_state_dict`` completes.
"""

from dataclasses import dataclass

import torch
import torch_npu
from torch.distributed._tensor import DTensor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import derive, override

_STATE_KEYS = ("exp_avg", "exp_avg_sq")


def _make_swap(t: torch.Tensor, *, copy_data: bool = False) -> torch.Tensor:
    local = t.to_local() if isinstance(t, DTensor) else t
    out = torch_npu.empty_with_swapped_memory(local.size(), device=local.device)
    if copy_data:
        out.copy_(local)
    if isinstance(t, DTensor):
        return DTensor.from_local(
            out,
            t.device_mesh,
            t.placements,
            shape=t.size(),
            stride=t.stride(),
            run_check=False,
        )
    return out


def _swap_state_init_hook(optimizer, args, kwargs):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros(
                    (), dtype=torch.float32, device=p.device,
                )
                state["exp_avg"] = _make_swap(p).zero_()
                state["exp_avg_sq"] = _make_swap(p).zero_()


def _swap_load_post_hook(optimizer):
    for state in optimizer.state.values():
        for k in _STATE_KEYS:
            if k in state:
                state[k] = _make_swap(state[k], copy_data=True)


class SwapOptimizersContainer(OptimizersContainer):

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts):
        super().__init__(config=config, model_parts=model_parts)
        for opt in self.optimizers:
            opt.register_step_pre_hook(_swap_state_init_hook)
            opt.register_load_state_dict_post_hook(_swap_load_post_hook)


@override(
    target=OptimizersContainer.Config,
    description="Allocate Adam/AdamW states in NPU swap memory (host-offload)",
)
def swap(
    cfg: OptimizersContainer.Config,
) -> SwapOptimizersContainer.Config:
    return derive(cfg, SwapOptimizersContainer.Config)
