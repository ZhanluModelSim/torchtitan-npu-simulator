# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused attention no longer owns an indexer-loss accumulator.

The selector's distillation loss moved to the separate ``lightning_indexer.asc``
autograd.Function, which derives the teacher from SMLAG's backward by-product.  The old
per-rank ``_indexer_loss_acc`` buffer (and its ``_init_self_buffers`` hook) must not
come back: it would accumulate a metric nobody feeds.
"""

import pytest

from torchtitan_npu.override.deepseek_v4.lightning_indexer.ascendc import AscInnerIndexer
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
    AscCompressedSparseInnerAttention,
)

pytestmark = pytest.mark.usefixtures("dsv4")


def _config():
    return AscCompressedSparseInnerAttention.Config(
        window_size=8,
        compress_ratio=4,
        softmax_scale=0.5,
        index_topk=4,
        num_heads=2,
        head_dim=8,
        index_num_heads=2,
        index_head_dim=4,
    )


def test_the_fused_attention_has_no_indexer_loss_accumulator():
    attention = _config().build()
    assert "_indexer_loss_acc" not in attention._buffers
    assert not hasattr(attention, "_indexer_loss_acc")


def test_the_selector_has_no_loss_accumulator_either():
    selector = AscInnerIndexer(AscInnerIndexer.Config(index_topk=4))
    assert "_indexer_loss_acc" not in selector._buffers
    assert not hasattr(selector, "_indexer_loss_acc")
