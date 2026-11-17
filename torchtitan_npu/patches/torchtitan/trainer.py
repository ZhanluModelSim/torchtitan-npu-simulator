# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3985
# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4421

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torchtitan.trainer
from torchtitan.trainer import Trainer

from torchtitan_npu.patches.torchtitan.components.checkpoint import EMACheckpointManager
from torchtitan_npu.patches.torchtitan.components.ema import EMAOptimizersContainer
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss

logger = logging.getLogger(__name__)

original_post_dataloading_process = Trainer.post_dataloading_process
original_pp_forward_backward_step = Trainer.pp_forward_backward_step


@functools.wraps(original_post_dataloading_process)
def patched_post_dataloading_process(self, input_dict, labels):
    """Delegate the dataloader-to-forward preprocessing to the model.

    Upstream #4116 gives the model the seam (``Decoder.preprocess_inputs``:
    mask/metadata build, CP shard, SPMD annotation), so the trainer does not
    keep its own copy of it — the validator's duplicate disappears with it.
    A model-level ``build_attention_masks`` override (transitional, the
    pre-#4116 shape of the same seam) is honored inside ``preprocess_inputs``.
    """
    model = self.model_parts[0]
    preprocess_inputs = getattr(model, "preprocess_inputs", None)
    if preprocess_inputs is None:
        return original_post_dataloading_process(self, input_dict, labels)
    inputs, labels, extra_kwargs = preprocess_inputs(
        input_dict,
        labels,
        parallel_dims=self.parallel_dims,
        device=self.device,
        parallelism=self.config.parallelism,
    )
    # ``preprocess_inputs`` rejects the full_dtensor input path, so ``labels``
    # is always a plain local tensor here: ``numel()`` is the post-CP-shard
    # token count this counter recorded before #4116 moved the step into the model.
    self.ntokens_seen += labels.numel()
    return inputs, labels, extra_kwargs


def _run_presplit_pipeline_schedule(
    schedule,
    *,
    arg_mbs,
    kwarg_mbs,
    target_mbs,
    losses,
    loss_kwargs,
    return_outputs,
):
    """Bridge the NPU Torch pipeline schedule's private microbatch API."""
    if (
        schedule._has_backward
        and getattr(schedule, "_backward_requires_autograd", True)
        and not torch.is_grad_enabled()
    ):
        raise RuntimeError("pipeline backward requires gradients")

    stages = getattr(schedule, "_stages", None)
    if stages is None:
        stages = [schedule._stage]
    for stage in stages:
        stage.has_backward = schedule._has_backward
        stage.clear_runtime_states()
    return schedule._step_microbatches(
        arg_mbs=arg_mbs,
        kwarg_mbs=kwarg_mbs,
        target_mbs=target_mbs,
        losses=losses,
        loss_kwargs=loss_kwargs,
        return_outputs=return_outputs,
    )


@functools.wraps(original_pp_forward_backward_step)
def patched_pp_forward_backward_step(self, *args, **kwargs):
    """Adapt older NPU pipeline schedules without changing Trainer semantics."""
    schedule = self.pp_schedule
    parameters = inspect.signature(schedule.step).parameters
    if {"arg_mbs", "kwarg_mbs", "target_mbs"}.issubset(parameters):
        return original_pp_forward_backward_step(self, *args, **kwargs)
    if not hasattr(schedule, "_step_microbatches"):
        raise RuntimeError("installed pipeline schedule has no pre-split API")

    original_step = schedule.step
    had_instance_step = "step" in vars(schedule)
    schedule.step = functools.partial(_run_presplit_pipeline_schedule, schedule)
    try:
        return original_pp_forward_backward_step(self, *args, **kwargs)
    finally:
        if had_instance_step:
            schedule.step = original_step
        else:
            del schedule.step


# Backport TorchTitan #4421 for the DataLoader batch H2D copies, plus the
# aux-loss denominator of #3864: both land in the same method.
def patch_train_step() -> None:
    src = textwrap.dedent(inspect.getsource(Trainer.train_step))
    assert src.count("value.to(self.device)") == src.count("labels.to(self.device)") == 1
    src = src.replace("value.to(self.device)", "value.to(self.device, non_blocking=True)")
    src = src.replace("labels.to(self.device)", "labels.to(self.device, non_blocking=True)")

    # Auxiliary losses normalize by the same per-step token count as the main
    # loss, so their scale is independent of parallelism degrees.  The token
    # count is a local of this method, so the call is inserted into its source,
    # right after the last branch that assigns it.
    # ``textwrap.dedent`` strips the method's own indentation, so the body sits at four spaces.
    anchor = "    # Process each gradient accumulation step, then free its inputs."
    assert src.count(anchor) == 1
    # The inserted statement must sit at the *anchor's own* indentation: a comment line does not
    # close the preceding suite, so a deeper insertion silently becomes that suite's last statement
    # -- which is how this call once ended up inside the ``else:`` branch of the dp check and never
    # ran when data parallelism was enabled.
    indent = " " * (len(anchor) - len(anchor.lstrip()))
    statement = f"{indent}AuxLoss.set_step_denominator(global_valid_tokens)"
    src = src.replace(anchor, f"{statement}\n\n{anchor}")
    assert f"\n{statement}\n" in src

    scope = dict(Trainer.train_step.__globals__)
    scope["AuxLoss"] = AuxLoss
    exec(src, scope)
    Trainer.train_step = scope["train_step"]


class EMATrainer(Trainer):
    """Trainer with upstream EMA configuration and optimizer wiring."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        ema_weights: EMAOptimizersContainer.Config = field(default_factory=EMAOptimizersContainer.Config)
        checkpoint: EMACheckpointManager.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=EMACheckpointManager.Config
        )

    def __init__(self, config: Config) -> None:
        config.checkpoint.ema_weights = config.ema_weights
        super().__init__(config)

        self.ema_optimizer = cast("Any", self.checkpointer).ema_optimizer
        self.optimizers.register_step_pre_hook(lambda *_args, **_kwargs: self.ema_optimizer.wait_for_param_reads())
        self.optimizers.register_step_post_hook(lambda *_args, **_kwargs: self.ema_optimizer.step(self.step))


def apply() -> None:
    logger.info("[PATCH] Trainer.post_dataloading_process -> patched_post_dataloading_process")
    Trainer.post_dataloading_process = patched_post_dataloading_process
    logger.info("[PATCH] Trainer.train_step -> async H2D and aux-loss denominator (upstream #4421, #3864)")
    patch_train_step()
    logger.info("[PATCH] Trainer.pp_forward_backward_step -> patched_pp_forward_backward_step")
    Trainer.pp_forward_backward_step = patched_pp_forward_backward_step
    torchtitan.trainer.Trainer = EMATrainer


apply()
