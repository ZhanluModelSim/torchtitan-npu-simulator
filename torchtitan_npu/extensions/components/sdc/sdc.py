# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Initialize compiled SDC and submit accumulated gradients to torch-npu."""

import os
from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.config import Configurable
from torchtitan.trainer import Trainer

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


class SDC(Configurable):
    """Configure SDC once after the standard NPU Trainer has built its models."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        gradient_enabled: bool = False
        with_checksum: bool = False
        hccl_mode: int = 0
        cooldown: int = 5
        strikes_num: int = 3
        strikes_window: int = 480
        checksum_cooldown: int = 180
        upper_thresh1: int = 1_000_000
        upper_thresh2: int = 100
        grad_sample_interval: int = 3

        def __post_init__(self) -> None:
            if type(self.hccl_mode) is not int or self.hccl_mode not in (0, 1, 2, 3):
                raise ValueError("--sdc.hccl-mode must be 0, 1, 2 or 3")
            for option, value, minimum in (
                ("--sdc.cooldown", self.cooldown, 1),
                ("--sdc.strikes-num", self.strikes_num, 1),
                ("--sdc.strikes-window", self.strikes_window, 1),
                ("--sdc.checksum-cooldown", self.checksum_cooldown, 1),
                ("--sdc.upper-thresh1", self.upper_thresh1, 3),
                ("--sdc.upper-thresh2", self.upper_thresh2, 3),
                ("--sdc.grad-sample-interval", self.grad_sample_interval, 1),
            ):
                if type(value) is not int or value < minimum:
                    raise ValueError(f"{option} must be an integer greater than or equal to {minimum}")
            if self.with_checksum and not self.gradient_enabled:
                raise ValueError("--sdc.with-checksum requires --sdc.gradient-enabled=true")
            if not self.gradient_enabled and any(
                getattr(self, option.name) != option.default
                for option in fields(self)
                if option.name not in ("gradient_enabled", "with_checksum", "hccl_mode")
            ):
                raise ValueError("non-default --sdc gradient tuning requires --sdc.gradient-enabled=true")

    def __init__(
        self,
        config: Config,
        trainer_config: Trainer.Config,
        *,
        model_parts: list[nn.Module],
        gradient_accumulation_steps: int,
    ) -> None:
        self._checker: Any = None
        self._targets: list[tuple[torch.Tensor, str]] = []
        self._accumulation_steps = gradient_accumulation_steps
        self._microbatch = 0
        if not config.gradient_enabled and config.hccl_mode == 0:
            return
        self._validate_compiled_sdc_config(trainer_config)
        # Import before writing NPU_ASD_ENABLE so torch-npu does not install eager hooks.
        import torch_npu

        if config.hccl_mode != 0:
            os.environ["NPU_ASD_ENABLE"] = str(config.hccl_mode)
            torch_npu._C._npu_set_module_train_state("train")
            torch_npu._C._npu_set_call_state("backward")
        if not config.gradient_enabled:
            return

        checker = torch_npu.asd.asd.matmul_check
        checker.set_matmul_hook_enable(1)
        checker.set_with_checksum(config.with_checksum)
        checker.set_cooldown(config.cooldown)
        checker.set_strikes_num(config.strikes_num)
        checker.set_strikes_window(config.strikes_window)
        checker.set_checksum_cooldown(config.checksum_cooldown)
        checker.set_upper_thresh1(config.upper_thresh1)
        checker.set_upper_thresh2(config.upper_thresh2)
        checker.set_grad_sample_interval(config.grad_sample_interval)
        if config.with_checksum:
            # Model compile wrappers exist, but the first training graph is still lazy.
            from torchtitan_npu.extensions.compile.sdc_checksum import install_checksum_pass

            install_checksum_pass()

        checker.init_stream()
        for name, parameter in self._iter_gradient_candidates(model_parts):
            checker.matmul_dtype_supported = True
            if not checker.parameter_filtering():
                continue
            state_key = f"{name}_backward"
            checker.check_stat[state_key] = {"avg": 0.0, "pre_val": 0.0, "step": 0, "none_zero_step": 0}
            self._targets.append((parameter, state_key))
        self._checker = checker
        checker._startup()

    @staticmethod
    def _validate_compiled_sdc_config(config: Trainer.Config) -> None:
        compile_config = config.compile
        if not (compile_config.enable and "model" in compile_config.components):
            raise ValueError(
                "compiled SDC requires compile.enable=true with 'model' in compile.components; "
                "use NPU_ASD_CONFIG/NPU_ASD_ENABLE for eager SDC"
            )
        if compile_config.backend != "inductor":
            raise ValueError(f"compiled SDC requires compile.backend='inductor'; got {compile_config.backend!r}")
        if int(config.parallelism.pipeline_parallel_degree) > 1:
            raise ValueError("compiled SDC does not support pipeline parallelism")

        # Match torch-npu's enable parsing: default false, last enable entry wins.
        native_gradient = "false"
        for item in os.environ.get("NPU_ASD_CONFIG", "").split(","):
            pair = item.split(":")
            if len(pair) == 2 and pair[0] == "enable":
                native_gradient = pair[1]
        for name, value, disabled, allowed in (
            ("NPU_ASD_CONFIG", native_gradient, "false", ("false", "true")),
            ("NPU_ASD_ENABLE", os.environ.get("NPU_ASD_ENABLE", "0"), "0", ("0", "1", "2", "3")),
        ):
            if value not in allowed:
                raise ValueError(f"invalid native SDC switch {name}: {value!r}; expected one of {allowed}")
            if value != disabled:
                raise ValueError(
                    f"typed compiled SDC cannot be combined with enabled {name}; "
                    "disable the native switch before importing torch-npu or disable typed SDC"
                )

    @torch.no_grad()
    def finalize_sdc_step(self) -> None:
        """Submit a complete accumulation window after a successful backward step."""

        if self._checker is None:
            return
        self._microbatch += 1
        if self._microbatch < self._accumulation_steps:
            return
        self._microbatch = 0
        for parameter, state_key in self._targets:
            grad = parameter.grad
            if grad is None:
                continue
            grad = grad.to_local() if isinstance(grad, DTensor) else grad
            self._checker._detect_grad(grad.detach(), state_key)

    def _iter_gradient_candidates(self, model_parts: list[nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield unique, nonempty trainable matrices supported by the detector."""

        seen_parameters: set[int] = set()
        for part_index, part in enumerate(model_parts):
            for name, parameter in part.named_parameters(remove_duplicate=True):
                if id(parameter) in seen_parameters or not parameter.requires_grad or parameter.dim() < 2:
                    continue
                if parameter.dtype not in _SUPPORTED_DTYPES:
                    continue
                seen_parameters.add(id(parameter))
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                if local.numel() == 0:
                    continue
                canonical = name if len(model_parts) == 1 else f"model_parts.{part_index}.{name}"
                yield canonical, parameter
