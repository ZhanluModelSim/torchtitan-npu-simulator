# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-safe shadow of torch._fused_adamw_ for optimizer step capture.

In real NPU training, torch._fused_adamw_ dispatches to
npu.npu_apply_adam_w (a fused NPU kernel). Under meta simulation,
fused=True raises RuntimeError (meta device not in supported list).

This shim records npu.npu_apply_adam_w.default via record_synthetic_op and
skips the numerical parameter update. Optimizer values are irrelevant to the
single-step meta simulation, and executing the in-place update would require
DTensor optimizer kernels to support every simulated mesh layout.

The optimizer OpNode uses logical DTensor global shapes so uneven HSDP shards
do not leak into operator modeling. Tensor dependencies and memory events keep
using per-rank local tensors.

See meta_env._patch_fused_adamw_for_meta for installation.
"""

from __future__ import annotations

import torch


def _meta_safe_fused_adamw(
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
    state_steps, *, amsgrad, beta1, beta2, lr,
    weight_decay, eps, maximize, **_kwargs,
):
    """Replacement for torch._fused_adamw_ under meta simulation.

    Record the NPU fused op name without executing numerical optimizer math.
    """
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    cap = get_active_capture()

    # Record the fused NPU op name (one per param group)
    if cap is not None:
        inputs = [
            *params,
            *grads,
            *exp_avgs,
            *exp_avg_sqs,
            *max_exp_avg_sqs,
            *state_steps,
        ]
        mutated = [
            *params,
            *exp_avgs,
            *exp_avg_sqs,
            *max_exp_avg_sqs,
            *state_steps,
        ]
        cap.record_synthetic_op(
            "npu.npu_apply_adam_w.default",
            inputs or [torch.empty(1, device="meta")],
            mutated,
            logical_dtensor_shapes=True,
        )
