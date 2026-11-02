# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Initialize native training state before installing capture observers."""

from collections.abc import Callable

from torchtitan_simulator.meta_env import _pp_context, patch_device_type_to_meta


def warmup_training_iteration(
    *,
    gradient_accumulation: int,
    forward_backward: Callable[[], object],
    clip_gradients: Callable[[], object],
    optimizer_step: Callable[[], object],
    zero_grad: Callable[[], object],
) -> None:
    """Execute one structural iteration without advancing the LR scheduler.

    The caller owns the native model/runtime and installs observers afterwards.
    Gradients and capture context are cleaned even if native execution fails;
    model/optimizer updates are intentionally not rolled back.
    """
    if type(gradient_accumulation) is not int or gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    patch_device_type_to_meta()
    previous_context = dict(_pp_context)
    try:
        zero_grad()
        for microbatch in range(gradient_accumulation):
            _pp_context.update(stage=0, mb_idx=microbatch, comp_type="F", phase="forward")
            forward_backward()
        _pp_context.update(comp_type="OPTIMIZER", phase="optimizer")
        clip_gradients()
        optimizer_step()
    finally:
        try:
            zero_grad()
        finally:
            _pp_context.clear()
            _pp_context.update(previous_context)
