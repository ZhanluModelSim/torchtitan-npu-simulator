# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 dynamic expert selection."""

from dataclasses import dataclass

import torch
from torchtitan.models.common.moe import TokenChoiceTopKRouter


class DeepSeekV41Router(TokenChoiceTopKRouter):
    """Router with optional vision bias and sorted dynamic selection."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        vision_enabled: bool = False
        sorted_topk: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.sorted_topk = config.sorted_topk
        self.bias_vl = (
            torch.nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32)) if config.vision_enabled else None
        )

    def reset_parameters(self) -> None:
        # ``bias_vl`` starts at zero; it is not part of the score gate, so no gate
        # initialization applies to it.
        bias_vl = getattr(self, "bias_vl", None)
        if bias_vl is not None:
            torch.nn.init.zeros_(bias_vl)

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        image_mask: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        # Image tokens use the vision bias instead of the balancing bias. The
        # replicated bias and token-sharded mask follow the gate's DTensor layout.
        if image_mask is not None and self.bias_vl is not None:
            text_bias_E = torch.zeros_like(self.bias_vl) if expert_bias_E is None else expert_bias_E
            choice_bias_TE = torch.where(image_mask.unsqueeze(-1), self.bias_vl, text_bias_E)
            scores_for_choice_TE = scores_TE + choice_bias_TE
        else:
            scores_for_choice_TE = scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        return self._topk(scores_for_choice_TE)

    def _topk(self, scores_for_choice_TE: torch.Tensor) -> torch.Tensor:
        """Select experts with an optional kernel override."""
        return scores_for_choice_TE.topk(self.top_k, dim=-1, sorted=self.sorted_topk).indices
