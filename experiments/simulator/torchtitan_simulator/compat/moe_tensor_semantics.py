# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Semantic contracts for main-package ascendc MoE custom wrapper boundaries.

Declarations live in the sidecar so torchtitan_npu never depends on simulator.
Source: torchtitan_npu/ops/ascendc/moe_{re_routing,token_permute,token_unpermute}.py.
"""

from typing import Any

from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)

register_tensor_semantics(
    "torchtitan_npu.npu_moe_re_routing.default",
    OperatorTensorSemantics(
        inputs=(
            ("routed_tokens", TensorMeaning("rank_major_tokens", ("rank_major_row", "feature"))),
            (
                "expert_token_num_per_rank",
                TensorMeaning("expert_token_counts", ("ep_rank", "local_expert")),
            ),
            ("per_token_scales", TensorMeaning("rank_major_scales", ("rank_major_row",), optional=True)),
        ),
        outputs=(
            TensorMeaning("expert_major_tokens", ("expert_major_row", "feature")),
            # The wrapper returns a real empty tensor when no scales were passed.
            TensorMeaning("expert_major_scales", ("expert_major_row",)),
            TensorMeaning("restore_indices", ("rank_major_row",)),
            TensorMeaning("global_tokens_per_local_expert", ("local_expert",)),
        ),
    ),
)

register_tensor_semantics(
    "torchtitan_npu.npu_moe_token_permute.default",
    OperatorTensorSemantics(
        inputs=(
            ("tokens", TensorMeaning("tokens", ("token", "feature"))),
            (
                "indices",
                TensorMeaning("expert_indices", ("token", "topk"), rank_alternatives=(("token",),)),
            ),
        ),
        outputs=(
            TensorMeaning("permuted_tokens", ("routed_row", "feature")),
            TensorMeaning("sorted_indices", ("routed_row",)),
        ),
    ),
)
register_tensor_semantics(
    "torchtitan_npu.npu_moe_token_permute_backward.default",
    OperatorTensorSemantics(
        inputs=(
            ("grad", TensorMeaning("permuted_tokens_gradient", ("routed_row", "feature"))),
            ("sorted_indices", TensorMeaning("sorted_indices", ("routed_row",))),
        ),
        outputs=(TensorMeaning("tokens_gradient", ("token", "feature")),),
    ),
)


def _unpermute_semantics(attrs: dict[str, Any]) -> OperatorTensorSemantics:
    # Empty probabilities are still present: only the native None binding
    # selects the unweighted row-restoration contract.
    weighted = dict(attrs["input_tensor_bindings"])["probs"] is not None
    return OperatorTensorSemantics(
        inputs=(
            ("permuted_tokens", TensorMeaning("permuted_tokens", ("routed_row", "feature"))),
            ("sorted_indices", TensorMeaning("sorted_indices", ("routed_row",))),
            ("probs", TensorMeaning("routing_probabilities", ("token", "topk"), optional=True)),
        ),
        outputs=(
            TensorMeaning(
                "combined_tokens" if weighted else "restored_rows", ("token" if weighted else "routed_row", "feature")
            ),
        ),
    )


register_tensor_semantics("torchtitan_npu.npu_moe_token_unpermute.default", _unpermute_semantics)
register_tensor_semantics(
    "torchtitan_npu.npu_moe_token_unpermute_grad_unweighted.default",
    OperatorTensorSemantics(
        inputs=(
            ("grad_output", TensorMeaning("restored_rows_gradient", ("routed_row", "feature"))),
            ("sorted_indices", TensorMeaning("sorted_indices", ("routed_row",))),
        ),
        outputs=(TensorMeaning("permuted_tokens_gradient", ("routed_row", "feature")),),
    ),
)
