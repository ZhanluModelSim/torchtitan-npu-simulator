# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-safe shadow of torch._fused_adamw_ for optimizer step capture.

In real NPU training, torch._fused_adamw_ dispatches to
npu.npu_apply_adam_w (a fused NPU kernel). Under meta simulation,
fused=True raises RuntimeError (meta device not in supported list).

This shim:
1. Records npu.npu_apply_adam_w.default via record_synthetic_op
2. Executes standard foreach AdamW math for shape inference
3. Suppresses individual _foreach_* sub-op capture during the math

See meta_env._patch_fused_adamw_for_meta for installation.
"""

from __future__ import annotations

import torch


def _meta_safe_fused_adamw(
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
    state_steps, *, amsgrad, beta1, beta2, lr,
    weight_decay, eps, maximize,
):
    """Replacement for torch._fused_adamw_ under meta simulation.

    Records the NPU fused op name, then runs standard AdamW math
    using foreach ops (with L0 capture temporarily disabled so
    only the fused op name appears in the IR).
    """
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    cap = get_active_capture()

    # Temporarily disable L0 capture so foreach sub-ops are not recorded
    old_capture = cap._capture_l0 if cap is not None else None
    if cap is not None:
        cap._capture_l0 = False

    # Record the fused NPU op name (one per param group)
    if cap is not None:
        rep = params[0] if params else torch.empty(1, device="meta")
        cap._capture_l0 = True  # re-enable briefly for the synthetic op
        cap.record_synthetic_op(
            "npu.npu_apply_adam_w.default",
            [rep],
            [rep],
        )
        cap._capture_l0 = False  # disable again for foreach sub-ops

    # Standard AdamW foreach math (shape inference only, no data)
    bias_correction1 = 1 - beta1 ** state_steps[0].item() if state_steps else 1.0
    bias_correction2 = 1 - beta2 ** state_steps[0].item() if state_steps else 1.0
    step_size = lr * (bias_correction1 / (bias_correction2 + eps))

    # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

    # denom = sqrt(exp_avg_sq) + eps
    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_add_(denom, eps)

    # step_size / denom
    step_sizes = [torch.full_like(d, step_size) for d in denom]
    torch._foreach_div_(step_sizes, denom)

    # param = param - step_size * exp_avg / denom
    if weight_decay != 0:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
    torch._foreach_addcdiv_(params, exp_avgs, step_sizes)

    # Restore capture state
    if cap is not None:
        cap._capture_l0 = old_capture
