# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 MFU FLOPs accounting.

The parameter half follows the pinned TorchTitan 0.3.0 ``6 * active_params``
convention, with Engram's sparse lookup table removed from the dense-matmul
term. The attention half replaces the helper's dense-attention estimate with
V4.1's model topology:

* sliding-window attention on every layer;
* selected compressed entries on every layer with ``compress_ratio > 0``;
* Full indexers scoring their complete compressed container;
* hierarchical Reindex layers scoring at most the candidate pool;
* Reuse indexers doing no scoring.

The tests use parameter-free stand-ins where only the attention terms are
under test. This keeps the full 1M geometry cheap enough for CPU CI.
"""

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1 import model_registry


class _NoParameters:
    """Parameter-free model stand-in for attention-only arithmetic tests."""

    def __init__(self, n_layers):
        self.layers = {str(idx): _NoEngramLayer() for idx in range(n_layers)}

    def named_parameters(self):
        return iter(())


class _NoEngramLayer:
    engram = None


@pytest.mark.parametrize(
    ("flavor", "seq_len", "expected"),
    (
        # 40 x window(6*8*64*2*128) + 38 x selected(6*8*64*2*32)
        #   + 3 x Full(6*8*32*256)
        #   + candidate-source Full(6*8*32*512)
        #   + 4 x Reindex(6*8*32*32), where 4 blocks x 8 entries = 32.
        (
            "deepseek_v4_1_debugmodel",
            512,
            31_457_280 + 7_471_104 + 2_162_688,
        ),
        # Same debug topology at 4K. Reindex remains capped at 32 entries.
        (
            "deepseek_v4_1_debugmodel",
            4096,
            31_457_280 + 7_471_104 + 15_925_248,
        ),
        # At 4K the published Flash candidate pool (2048 x 8) is wider than
        # the sequence, so hierarchical Reindex has not reached its cap yet.
        (
            "deepseek_v4_1_flash_40layers_16experts_text",
            4096,
            2_013_265_920 + 7_650_410_496 + 654_311_424,
        ),
        # At 1M the four Reindex layers are bounded by 16,384 candidates:
        #   window:   40 * 6*64*1024*128
        #   selected: 38 * 6*64*1024*512
        #   indexer:  3 * 6*32*128*(1M/2)
        #           + 1 * 6*32*128*1M
        #           + 4 * 6*32*128*(2048*8)
        (
            "deepseek_v4_1_flash",
            1_048_576,
            2_013_265_920 + 7_650_410_496 + 66_035_122_176,
        ),
    ),
)
def test_flops_are_window_plus_compressed_plus_hierarchical_indexer(
    flavor,
    seq_len,
    expected,
):
    config = model_registry(flavor).model

    nparams, flops = config.get_nparams_and_flops(
        _NoParameters(config.n_layers),
        seq_len=seq_len,
    )

    assert nparams == 0
    assert flops == expected


def test_flash_4k_full_model_flops_keep_engram_as_lookup_storage():
    """Real model coverage: Engram tables count toward size, not dense 6P."""

    config = model_registry("deepseek_v4_1_flash").model
    with torch.device("meta"):
        model = config.build()

    engram_tables = [layer.engram.table.weight for layer in model.layers.values() if layer.engram is not None]
    engram_table_nparams = sum(table.numel() for table in engram_tables)

    nparams, flops = config.get_nparams_and_flops(model, seq_len=4096)

    assert engram_table_nparams == 196_614_815_744
    assert nparams == sum(parameter.numel() for parameter in model.parameters())
    assert flops == 107_101_428_384
