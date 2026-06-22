# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import fields, replace
from typing import Any

from torchtitan.components.optimizer import OptimizersContainer

logger = logging.getLogger(__name__)


def _build_muon_config_kwargs(config: Any) -> dict[str, Any]:
    return {
        "name": config.name,
        "lr": config.lr,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "eps": config.eps,
        "weight_decay": config.weight_decay,
        "implementation": config.implementation,
        "muon_lr": getattr(config, "muon_lr", None),
        "muon_momentum": getattr(config, "muon_momentum", 0.95),
        "muon_enable_nesterov": getattr(config, "muon_enable_nesterov", True),
        "muon_ns_steps": getattr(config, "muon_ns_steps", 5),
        "muon_adjust_lr_fn": getattr(config, "muon_adjust_lr_fn", "match_rms_adamw"),
        "muon_hybrid_ns": getattr(config, "muon_hybrid_ns", False),
        "extra_param_group_split_rules": getattr(config, "extra_param_group_split_rules", None),
    }


class NpuOptimizerDispatcher:
    _virtual_patched = False
    _swap_patched = False

    @staticmethod
    def dispatch_build(self: Any, **kwargs) -> OptimizersContainer:
        NpuOptimizerDispatcher._check_build_kwargs(self, kwargs)

        is_virtual = getattr(self, "virtual_optimizer", False)
        is_swap = getattr(self, "swap_optimizer", False)
        optimizer_name = getattr(self, "name", None)

        if is_virtual and is_swap:
            raise ValueError("Cannot enable both virtual_optimizer and swap_optimizer at the same time.")

        # Muon optimizer routing
        if optimizer_name == "Muon":
            if is_virtual:
                raise ValueError("Muon does not support virtual_optimizer. Use swap_optimizer for Muon.")
            return NpuOptimizerDispatcher._build_muon_optimizer(self, is_swap, kwargs)

        if is_virtual:
            return NpuOptimizerDispatcher._build_virtual_optimizer(self, kwargs)

        if is_swap:
            return NpuOptimizerDispatcher._build_swap_optimizer(self, kwargs)

        return NpuOptimizerDispatcher._build_standard_optimizer(self, kwargs)

    @staticmethod
    def _check_build_kwargs(config: Any, kwargs) -> None:
        config_fields = {f.name for f in fields(config)}
        overlap = config_fields & kwargs.keys()
        if overlap:
            raise ValueError(f"build() kwargs {overlap} overlap with config fields.")

    @staticmethod
    def _resolve_parallel_dims(kwargs):
        from torchtitan_npu.patches.torchtitan._trainer_config_stash import (
            get_active_parallel_dims,
        )

        parallel_dims = kwargs.get("parallel_dims") or get_active_parallel_dims()
        if parallel_dims is None:
            raise RuntimeError(
                "parallel_dims is required for Muon optimizer but not available. "
                "Ensure Trainer.init_distributed() has been called before "
                "config.optimizer.build()."
            )
        return parallel_dims

    @staticmethod
    def _build_muon_optimizer(config: Any, is_swap: bool, kwargs):
        parallel_dims = NpuOptimizerDispatcher._resolve_parallel_dims(kwargs)
        if is_swap:
            return NpuOptimizerDispatcher._build_swap_muon_optimizer(config, parallel_dims, kwargs)

        logger.info("[OptimizerDispatcher] Using MuonOptimizer")
        from .muon_optimizer import MuonHybridOptimizersContainer

        cfg = MuonHybridOptimizersContainer.Config(**_build_muon_config_kwargs(config))
        return cfg.build(
            model_parts=kwargs["model_parts"], parallel_dims=parallel_dims, ft_manager=kwargs.get("ft_manager")
        )

    @staticmethod
    def _build_swap_muon_optimizer(config: Any, parallel_dims, kwargs):
        logger.info("[OptimizerDispatcher] Using SwapMuonOptimizer")
        from .swap_optimizer import SwapMuonHybridOptimizersContainer

        cfg_kwargs = _build_muon_config_kwargs(config)
        cfg_kwargs.update(
            swap_optimizer_times=getattr(config, "swap_optimizer_times", 16),
            swap_merge_buckets=getattr(config, "swap_merge_buckets", 1),
        )
        cfg = SwapMuonHybridOptimizersContainer.Config(
            **cfg_kwargs,
        )
        return cfg.build(
            model_parts=kwargs["model_parts"], parallel_dims=parallel_dims, ft_manager=kwargs.get("ft_manager")
        )

    @staticmethod
    def _build_virtual_optimizer(config: Any, kwargs):
        virtual_size = getattr(config, "virtual_optimizer_size", None)
        if virtual_size is None:
            raise ValueError("virtual_optimizer_size must be specified when virtual_optimizer is enabled.")

        logger.info("[OptimizerDispatcher] Using VirtualOptimizer")

        from .virtual_optimizer import VirtualOptimizersContainer

        NpuOptimizerDispatcher._apply_virtual_patch()

        container_kwargs = {k: v for k, v in kwargs.items() if k in ("model_parts",)}
        return VirtualOptimizersContainer(config=replace(config), **container_kwargs)

    @staticmethod
    def _build_swap_optimizer(config: Any, kwargs):
        logger.info("[OptimizerDispatcher] Using SwapOptimizer")

        from .swap_optimizer import SwapOptimizersContainer

        NpuOptimizerDispatcher._apply_swap_patch()

        container_kwargs = {k: v for k, v in kwargs.items() if k in ("model_parts",)}
        return SwapOptimizersContainer(config=replace(config), **container_kwargs)

    @staticmethod
    def _build_standard_optimizer(config: Any, kwargs):
        logger.info("[OptimizerDispatcher] Using standard Optimizer")
        base_config = OptimizersContainer.Config(
            name=config.name,
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
            implementation=config.implementation,
        )
        return OptimizersContainer(config=base_config, **kwargs)

    @classmethod
    def _apply_virtual_patch(cls):
        if cls._virtual_patched:
            return

        import torch

        from .virtual_optimizer import (
            _make_patched_load,
            patched_state_dict,
            swap_tensor_copy_wrapper,
            swap_tensor_func_wrapper,
            virtual_optimizer_step,
        )

        torch.Tensor.copy_ = swap_tensor_copy_wrapper(torch.Tensor.copy_)
        torch.Tensor.cpu = swap_tensor_func_wrapper(torch.Tensor.cpu, "cpu")
        torch.Tensor.clone = swap_tensor_func_wrapper(torch.Tensor.clone, "clone")
        torch.Tensor.detach = swap_tensor_func_wrapper(torch.Tensor.detach, "detach")

        for cls_opt in [torch.optim.AdamW, torch.optim.Adam]:
            if not hasattr(cls_opt, "_original_state_dict"):
                cls_opt._original_state_dict = cls_opt.state_dict
                cls_opt.state_dict = patched_state_dict
                cls_opt.step = virtual_optimizer_step
                cls_opt._original_load_state_dict = cls_opt.load_state_dict
                cls_opt.load_state_dict = _make_patched_load(cls_opt.load_state_dict)

        cls._virtual_patched = True
        logger.info("[VirtualOptimizer] Patched Adam/AdamW successfully")

    @classmethod
    def _apply_swap_patch(cls):
        if cls._swap_patched:
            return

        import torch

        from .swap_optimizer import swap_optimizer_step

        torch.optim.AdamW.step = swap_optimizer_step
        torch.optim.Adam.step = swap_optimizer_step
        cls._swap_patched = True
        logger.info("[SwapOptimizer] Patched Adam/AdamW successfully")


def patch_npu_optimizer_framework():
    try:
        from torchtitan_npu.config.configs import OptimizerConfig

        OptimizerConfig.build = NpuOptimizerDispatcher.dispatch_build
        logger.info("[OptimizerFramework] Patch successful")
    except Exception as e:
        logger.error(f"Patch failed: {e}")
        raise
