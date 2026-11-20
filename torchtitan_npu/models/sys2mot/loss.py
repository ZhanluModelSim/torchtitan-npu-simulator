# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Joint VLA training loss for sys2mot capture.

The reference joint objective (``train_dummy``) is
``flow + 0.5*KD_KL + 0.1*hidden + 0.1*CE + 0.01*indexer``. The KD/hidden terms
distill from a frozen teacher forward: the runner mounts a frozen meta-model
on the loss (``set_teacher``) and the loss runs the teacher VLM backbone per
step under ``no_grad``, replicating the reference ``Distiller`` formulas
(temperature-2 logits KL and normalized hidden MSE over the text mask). The
teacher is not FSDP-wrapped and never joins the optimizer, matching the
reference ``teacher.eval().requires_grad_(False)``. The model forward returns
the velocity, the final prefix hidden states and the indexer aux as separate
tensors, so no gradient injection is needed — the composite loss backward
flows through all of them.

CE semantics follow the capture convention: logits over the text span only,
targets are the next-token labels with the final column nulled
(``build_attention_masks``), normalized by the runner's ``global_valid_tokens``
(the BaseLoss token normalization convention; the reference means over its
shifted-target mask, a constant-factor difference in loss scale only).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.components.loss import BaseLoss
from torchtitan.config import CompileConfig

from .model import synthetic_action_batch


class Sys2MotJointLoss(BaseLoss):
    """flow MSE (velocity target ``noise - actions``) + text CE + indexer aux."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        text_offset: int
        """Number of vision prefix tokens in front of the text span."""
        action_dim: int
        horizon: int
        ce_weight: float = 0.1
        indexer_weight: float = 0.01
        distill: bool = False
        """Run the frozen teacher forward and the KD/hidden distillation terms.

        Current capture form is offline distillation (student-only training
        step). The archived in-loop sharded form (tag sys2mot-distill-sharded)
        re-enables the in-loop teacher by setting True here and in the preset.
        """

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        del compile_config  # the composite loss is not compiled
        self.text_offset = config.text_offset
        self.action_dim = config.action_dim
        self.horizon = config.horizon
        self.ce_weight = config.ce_weight
        self.indexer_weight = config.indexer_weight
        self.distill = config.distill
        self.lm_head: torch.nn.Module | None = None
        self.teacher: torch.nn.Module | None = None
        self.kd_temperature: float = 2.0

    def set_lm_head(self, lm_head: torch.nn.Module) -> None:
        """Set the LM head module. Must be called before the first __call__."""
        self.lm_head = lm_head

    def set_teacher(self, teacher: torch.nn.Module) -> None:
        """Mount the frozen teacher (eval, no grad); must run before the first call.

        The teacher lives on the loss so it stays outside the optimizer,
        exactly like the reference distillation setup; it MAY carry its own
        FSDP2+EP sharding (the mounter binds the instance forward to the
        backbone path, so the loss calls it as a module). Frozen: no
        gradients, no optimizer states, forward comm only.
        """
        if not self.distill:
            raise ValueError("set_teacher requires the distill loss knob")
        teacher.eval()
        teacher.requires_grad_(False)
        self.teacher = teacher

    def _distill_terms(
        self,
        pred: dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Reference ``Distiller.forward``: temperature-2 logits KL + hidden MSE.

        The teacher forward runs here (frozen, ``no_grad``); only the student
        side carries gradients, and both terms normalize over the text mask
        exactly like the reference. ``logits`` is the same student text-span
        tensor the CE consumes, and the hidden term compares the text-span
        hidden against the teacher's, matching ``student["text_hidden"]``.
        """
        assert self.teacher is not None, "set_teacher must run before the first loss call"
        images = pred["distill_images"]
        tokens = pred["distill_tokens"]
        text_mask = pred["distill_text_mask"]
        with torch.no_grad():
            # Module call, not ``backbone_forward``: the sharded teacher is an
            # FSDP tree and the root ``__call__`` pre-hook unshards the vision
            # stem the root unit owns before the backbone path runs.
            tf = self.teacher(images, tokens, text_mask=text_mask)
        temperature = self.kd_temperature
        s = logits.float()
        t = tf["logits"].detach().float()
        kl = F.kl_div(
            (s / temperature).log_softmax(-1),
            (t / temperature).softmax(-1),
            reduction="none",
        ).sum(-1)
        kd = (kl * text_mask).sum() / text_mask.sum().clamp_min(1) * temperature**2
        sh = F.normalize(pred["hidden"][:, self.text_offset :].float(), dim=-1)
        th = F.normalize(tf["text_hidden"].detach().float(), dim=-1)
        hidden = ((sh - th).square().sum(-1) * text_mask).sum() / text_mask.sum().clamp_min(1)
        return 0.5 * kd + 0.1 * hidden

    def __call__(
        self,
        pred: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs  # synthetic rows carry no extra loss inputs
        velocity, indexer_aux = pred["velocity"], pred["indexer_aux"]
        _, actions, noise, _ = synthetic_action_batch(velocity.shape[0], self.action_dim, self.horizon, velocity.device)
        flow = F.mse_loss(velocity.float(), (noise - actions).float())
        if "logits" in pred:
            logits = pred["logits"]
        else:
            head = self.lm_head
            assert head is not None, "set_lm_head must run before the first loss call"
            logits = head(pred["hidden"][:, self.text_offset :])
        ce = F.cross_entropy(
            logits.flatten(0, 1).float(),
            labels.flatten(0, 1),
            reduction="sum",
            ignore_index=-100,
        )
        if global_valid_tokens is not None:
            ce = ce / global_valid_tokens
        loss = flow + self.ce_weight * ce + self.indexer_weight * indexer_aux
        if self.distill:
            loss = loss + self._distill_terms(pred, logits)
        return loss, {}
