# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims

from torchtitan.experiments.ft import FTManager

logger = logging.getLogger("torchtitan")

_MUON_EXCLUDED_KEYWORDS = ("embed", "lm_head", "output")


def _should_use_muon(p: nn.Parameter, name: str) -> bool:
    """Check if parameter should be optimized by Muon.

    Rules:
    - 2D params go to Muon, except embeddings, lm_head, and output layers
    - Non-2D params go to AdamW
    """
    if p.ndim != 2:
        return False
    return not any(kw in name for kw in _MUON_EXCLUDED_KEYWORDS)


def _split_parameters_for_muon(
    model_parts: list[nn.Module],
) -> tuple[list[nn.Parameter], list[str], list[nn.Parameter], list[str]]:
    """Split parameters into Muon (2D) and AdamW (non-2D) groups.

    Returns:
        Tuple of (muon_params, muon_param_names, adamw_params, adamw_param_names)
    """
    muon_params = []
    muon_param_names = []
    adamw_params = []
    adamw_param_names = []

    for model in model_parts:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _should_use_muon(p, name):
                muon_params.append(p)
                muon_param_names.append(name)
            else:
                adamw_params.append(p)
                adamw_param_names.append(name)

    return muon_params, muon_param_names, adamw_params, adamw_param_names


def _build_muon_kwargs(
    muon_lr: float,
    weight_decay: float,
    config: "MuonHybridOptimizersContainer.Config",
    muon_adjust_lr_fn: str | None,
) -> dict[str, Any]:
    muon_kwargs = {
        "lr": muon_lr,
        "weight_decay": weight_decay,
        "momentum": config.muon_momentum,
        "nesterov": config.muon_enable_nesterov,
        "ns_steps": config.muon_ns_steps,
    }
    if muon_adjust_lr_fn:
        muon_kwargs["adjust_lr_fn"] = muon_adjust_lr_fn
    return muon_kwargs


def _build_adamw_kwargs(
    lr: float,
    weight_decay: float,
    config: "MuonHybridOptimizersContainer.Config",
) -> dict[str, Any]:
    optim_implementation = config.implementation
    if optim_implementation not in ["fused", "foreach", "for-loop"]:
        raise ValueError(
            f"Invalid implementation '{optim_implementation}'. "
            f"Must be one of: 'fused', 'foreach', 'for-loop'"
        )
    return {
        "lr": lr,
        "betas": (config.beta1, config.beta2),
        "eps": config.eps,
        "weight_decay": weight_decay,
        "fused": optim_implementation == "fused",
        "foreach": optim_implementation == "foreach",
    }


class MuonHybridOptimizersContainer(OptimizersContainer):
    """Container for Muon + AdamW hybrid optimizers.

    Key difference from upstream OptimizersContainer:
    - Upstream: model_parts[i] <-> optimizers[i] (1:1 pairing)
    - This class: each optimizer manages a subset of params from all model_parts

    When muon_adjust_lr_fn == "original":
    - Muon and AdamW use different base_lr
    - Must be used with MuonLRSchedulersContainer

    When muon_adjust_lr_fn == "match_rms_adamw":
    - Muon and AdamW use the same base_lr
    - Can use standard LRSchedulersContainer

    state_dict/load_state_dict use double loop over each optimizer and model_part.
    DCP APIs automatically filter to only process params managed by each optimizer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        """Config for Muon hybrid optimizer.

        Extends OptimizersContainer.Config with Muon-specific parameters.
        When name == "Muon", enables Muon + AdamW hybrid optimization.
        """

        _owner: ClassVar[type | None] = None

        # Muon-specific parameters
        muon_lr: float | None = None
        muon_momentum: float = 0.9
        muon_enable_nesterov: bool = False
        muon_ns_steps: int = 5
        muon_adjust_lr_fn: str = "match_rms_adamw"

        def build(self, **kwargs):
            """Build MuonHybridOptimizersContainer instance.

            Note: Unlike standard OptimizersContainer, Muon requires parallel_dims
            for validation (single-device check). This is passed via kwargs.
            """
            if self._owner is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no owner class. "
                    "Define Config inside a Configurable subclass."
                )
            if self.name != "Muon":
                raise ValueError(
                    f"MuonHybridOptimizersContainer.Config.name must be 'Muon', "
                    f"got '{self.name}'"
                )
            config_fields = {f.name for f in fields(self)}
            overlap = config_fields & kwargs.keys()
            if overlap:
                raise ValueError(
                    f"build() kwargs {overlap} overlap with config fields. "
                    "Put these values in the Config, not in build() kwargs."
                )
            return self._owner(config=replace(self), **kwargs)

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims | None = None,
        ft_manager: FTManager | None = None,
    ) -> None:
        # Validate single-device constraint
        is_distributed = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if is_distributed else 1
        if world_size > 1:
            raise NotImplementedError(
                "Muon optimizer currently only support single device"
            )

        lr = config.lr
        weight_decay = config.weight_decay
        muon_lr, muon_adjust_lr_fn = self._get_muon_lr_config(config, lr)

        (
            muon_params,
            muon_param_names,
            adamw_params,
            adamw_param_names,
        ) = _split_parameters_for_muon(model_parts)

        logger.info(
            f"[MuonAdamW] Muon optimizer parameters ({len(muon_param_names)}): {muon_param_names}"
        )
        logger.info(
            f"[MuonAdamW] AdamW optimizer parameters ({len(adamw_param_names)}): {adamw_param_names}"
        )

        muon_kwargs = _build_muon_kwargs(
            muon_lr, weight_decay, config, muon_adjust_lr_fn
        )
        adamw_kwargs = _build_adamw_kwargs(lr, weight_decay, config)

        muon = torch.optim.Muon(muon_params, **muon_kwargs)
        adamw = torch.optim.AdamW(adamw_params, **adamw_kwargs)

        self.model_parts = model_parts
        self.optimizers = [muon, adamw]
        self.muon_adjust_lr_fn = muon_adjust_lr_fn

        all_params = []
        for model in model_parts:
            all_params.extend(p for p in model.parameters() if p.requires_grad)
        Optimizer.__init__(self, all_params, {})

    def __iter__(self) -> Iterator[Optimizer]:
        """Return iterator over sub-optimizers for MuonLRSchedulersContainer."""
        return iter(self.optimizers)

    def __len__(self) -> int:
        """Return number of optimizers (Muon + AdamW = 2)."""
        return len(self.optimizers)

    @property
    def muon_optimizer(self) -> Optimizer:
        """Get the Muon optimizer."""
        return self.optimizers[0]

    @property
    def adamw_optimizer(self) -> Optimizer:
        """Get the AdamW optimizer."""
        return self.optimizers[1]

    @staticmethod
    def _get_muon_lr_config(
        config: Config,
        base_lr: float,
    ) -> tuple[float, str | None]:
        """Calculate Muon's effective learning rate and adjustment mode.

        Returns:
            Tuple of (muon_lr, muon_adjust_lr_fn)
        """
        muon_adjust_lr_fn = config.muon_adjust_lr_fn
        muon_lr = config.muon_lr

        if muon_adjust_lr_fn == "original" and muon_lr is not None:
            return float(muon_lr), muon_adjust_lr_fn

        if muon_adjust_lr_fn == "match_rms_adamw" and muon_lr is not None:
            logger.warning(
                "[Muon] muon_lr=%s is ignored when "
                "muon_adjust_lr_fn='match_rms_adamw'. Using base lr=%s instead.",
                muon_lr,
                base_lr,
            )
        return base_lr, muon_adjust_lr_fn

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Save state for all optimizers using double loop over optimizer x model_part."""
        merged = {}
        for opt in self.optimizers:
            for model in self.model_parts:
                sd = get_optimizer_state_dict(
                    model,
                    opt,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )
                merged.update(sd)
        return merged

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state for all optimizers using double loop over optimizer x model_part."""
        for opt in self.optimizers:
            for model in self.model_parts:
                set_optimizer_state_dict(
                    model,
                    opt,
                    optim_state_dict=state_dict,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )


class MuonLRSchedulersContainer:
    """LR Scheduler container for Muon hybrid optimizers.

    Creates independent LambdaLR schedulers for Muon and AdamW,
    ensuring each maintains its own base_lr.

    Key difference from upstream LRSchedulersContainer:
    - Upstream: assumes all optimizers use the same base_lr
    - This class: allows Muon and AdamW to have different base_lr

    Note: state_dict only saves the first scheduler's state (last_epoch),
    consistent with upstream behavior since Muon and AdamW share the same
    lr curve, only differing in base_lr.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LRSchedulersContainer.Config):
        _owner: ClassVar[type | None] = None

        def build(self, *, optimizers, training_steps):
            """Build LR scheduler for Muon hybrid optimizers.

            Routes to different scheduler types based on optimizer's muon_adjust_lr_fn:
            - "original": MuonLRSchedulersContainer (different base_lr for Muon and AdamW)
            - Other: Standard LRSchedulersContainer (same base_lr for both)
            """
            if self._owner is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no owner class. "
                    "Define Config inside a Configurable subclass."
                )

            total_steps = (
                self.total_steps if self.total_steps is not None else training_steps
            )

            warmup_steps = int(self.warmup_steps)

            if warmup_steps > total_steps:
                logger.warning(
                    f"Warmup steps ({warmup_steps}) exceed total steps ({total_steps}). "
                    f"Adjusting warmup steps to {total_steps}."
                )
                warmup_steps = total_steps

            if self.decay_ratio is not None:
                decay_steps = round(total_steps * self.decay_ratio)
                if warmup_steps + decay_steps > total_steps:
                    decay_steps = total_steps - warmup_steps
            else:
                decay_steps = total_steps - warmup_steps

            stable_steps = total_steps + 1 - warmup_steps - decay_steps
            lr_decay_type = self.decay_type
            min_lr_factor = self.min_lr_factor

            def linear_warmup_stable_decay(
                current_step: int,
                warmup_steps: int,
                stable_steps: int,
                decay_steps: int,
                lr_decay_type: str,
                min_lr_factor: float,
            ):
                warmup_stable_steps = warmup_steps + stable_steps
                if current_step < warmup_steps:
                    current_step += 1
                    curr_adjustment = float(current_step / warmup_steps)
                elif current_step < warmup_stable_steps:
                    curr_adjustment = 1.0
                else:
                    current_step += 1
                    progress = float(current_step - warmup_stable_steps) / decay_steps
                    if lr_decay_type == "linear":
                        curr_adjustment = 1 - progress
                    elif lr_decay_type == "sqrt":
                        curr_adjustment = 1 - math.sqrt(progress)
                    elif lr_decay_type == "cosine":
                        curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
                    else:
                        raise ValueError(f"Unknown lr_decay_type: {lr_decay_type}")
                    curr_adjustment = (
                        min_lr_factor + (1 - min_lr_factor) * curr_adjustment
                    )
                return curr_adjustment

            lr_lambda = functools.partial(
                linear_warmup_stable_decay,
                warmup_steps=warmup_steps,
                stable_steps=stable_steps,
                decay_steps=decay_steps,
                lr_decay_type=lr_decay_type,
                min_lr_factor=min_lr_factor,
            )

            if (
                isinstance(optimizers, MuonHybridOptimizersContainer)
                and optimizers.muon_adjust_lr_fn == "original"
            ):
                return self._owner(optimizers, lr_lambda)
            else:
                return LRSchedulersContainer(optimizers, lr_lambda)

    def __init__(
        self,
        optimizers: MuonHybridOptimizersContainer,
        lr_lambda: Callable,
    ) -> None:
        if len(optimizers) != 2:
            raise ValueError(
                f"MuonHybridOptimizersContainer must have 2 optimizers, got {len(optimizers)}"
            )

        self.schedulers = [
            LambdaLR(optimizers.muon_optimizer, lr_lambda),
            LambdaLR(optimizers.adamw_optimizer, lr_lambda),
        ]

        logger.info("[MuonLRSchedulersContainer] Created 2 schedulers")
        logger.info(f"  Muon scheduler base_lrs: {self.schedulers[0].base_lrs}")
        logger.info(f"  AdamW scheduler base_lrs: {self.schedulers[1].base_lrs}")
        logger.info(
            f"  Muon param_groups lr: {[pg['lr'] for pg in optimizers.muon_optimizer.param_groups]}"
        )
        logger.info(
            f"  AdamW param_groups lr: {[pg['lr'] for pg in optimizers.adamw_optimizer.param_groups]}"
        )

    def __iter__(self):
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        last_epoch = state_dict["last_epoch"]
        for scheduler in self.schedulers:
            scheduler.last_epoch = last_epoch
            scheduler._step_count = last_epoch + 1
            scheduler._last_lr = [
                scheduler.base_lrs[i] * scheduler.lr_lambdas[i](last_epoch)
                for i in range(len(scheduler.base_lrs))
            ]


_OWNER_ATTR = "_owner"
setattr(
    MuonHybridOptimizersContainer.Config, _OWNER_ATTR, MuonHybridOptimizersContainer
)
setattr(MuonLRSchedulersContainer.Config, _OWNER_ATTR, MuonLRSchedulersContainer)


def build_muon_hybrid_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizersContainer.Config,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> MuonHybridOptimizersContainer:
    """Build Muon hybrid optimizer (backward-compatible entry point).

    This function provides backward compatibility for the old API.
    New code should use MuonHybridOptimizersContainer.Config.build() instead.
    """
    config = MuonHybridOptimizersContainer.Config(
        name="Muon",
        lr=optimizer_config.lr,
        beta1=optimizer_config.beta1,
        beta2=optimizer_config.beta2,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
        implementation=optimizer_config.implementation,
        muon_lr=getattr(optimizer_config, "muon_lr", None),
        muon_momentum=getattr(optimizer_config, "muon_momentum", 0.9),
        muon_enable_nesterov=getattr(optimizer_config, "muon_enable_nesterov", False),
        muon_ns_steps=getattr(optimizer_config, "muon_ns_steps", 5),
        muon_adjust_lr_fn=getattr(
            optimizer_config, "muon_adjust_lr_fn", "match_rms_adamw"
        ),
    )
    return MuonHybridOptimizersContainer(
        config=config,
        model_parts=model_parts,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
    )
