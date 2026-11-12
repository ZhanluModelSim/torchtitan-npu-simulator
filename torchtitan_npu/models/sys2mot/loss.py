# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Joint VLA training loss for sys2mot capture.

The reference joint objective (``train_dummy``) is
``flow + 0.5*KD_KL + 0.1*hidden + 0.1*CE + 0.01*indexer``. The KD/hidden terms
distill from a frozen teacher forward, which the single-model capture stack
does not run; the captured loss keeps the three student-only terms with the
reference weights. The model forward returns the velocity, the final prefix
hidden states and the indexer aux as separate tensors, so no gradient
injection is needed — the composite loss backward flows through all of them.

CE semantics follow the reference: logits over the text span only, targets are
the next-token labels with the final column nulled (``build_attention_masks``),
normalized by the runner's ``global_valid_tokens`` (the BaseLoss token
normalization convention; the reference means over its shifted-target mask, a
constant-factor difference in loss scale only).
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

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        del compile_config  # the composite loss is not compiled
        self.text_offset = config.text_offset
        self.action_dim = config.action_dim
        self.horizon = config.horizon
        self.ce_weight = config.ce_weight
        self.indexer_weight = config.indexer_weight
        self.lm_head: torch.nn.Module | None = None

    def set_lm_head(self, lm_head: torch.nn.Module) -> None:
        """Set the LM head module. Must be called before the first __call__."""
        self.lm_head = lm_head

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
        return loss, {}
