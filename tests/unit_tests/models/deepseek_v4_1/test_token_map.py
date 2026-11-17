# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenizer normalization and automatic or pre-generated Engram map loading."""

from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from torchtitan_npu.models.deepseek_v4_1 import EngramArgs
from torchtitan_npu.models.deepseek_v4_1.engram.config import _make_engram_configs
from torchtitan_npu.models.deepseek_v4_1.engram.token_map import build_token_id_map, normalize_token

pytestmark = pytest.mark.cpu

_TOKENIZER = Path(__file__).parents[4] / "tests" / "assets" / "deepseek_v3" / "tokenizer.json"


def _generated_map():
    return build_token_id_map(Tokenizer.from_file(str(_TOKENIZER)))


def test_normalization_collapses_case_accents_and_whitespace():
    assert normalize_token("Hello") == normalize_token("hello")
    assert normalize_token("café") == normalize_token("cafe")
    assert normalize_token("a\t b") == "a b"
    # A token that normalizes to a single space keeps it; others are stripped.
    assert normalize_token(" ") == " "
    assert normalize_token("  word  ") == "word"
    assert normalize_token("  Apple\n") == "apple"
    assert normalize_token("\t\n") == " "
    # Normalizing to nothing falls back to the original text.
    assert normalize_token("\u200b") == "\u200b"


@pytest.mark.parametrize("source", ["map", "tokenizer"])
def test_generated_map_loads_into_the_table(tmp_path, source):
    mapping = _generated_map()
    compressed = int(mapping.max()) + 1
    map_path = tmp_path / "engram_token_id_map.npy"
    np.save(map_path, mapping)

    def build(expected_compressed):
        args = EngramArgs(
            layer_ids=(0,),
            vocab_size_per_ngram=(16, 16),
            n_embed_per_ngram=16,
            num_heads_per_ngram=2,
            token_id_map_path=str(map_path) if source == "map" else None,
            require_token_id_map=True,
            compressed_vocab_size=expected_compressed,
        )
        configs = _make_engram_configs(hidden_size=8, hc_mult=1, vocab_size=mapping.shape[0], engram=args)
        if source == "tokenizer":
            configs[0].table.tokenizer_path = str(_TOKENIZER)
        table = configs[0].table.build()
        table.init_states()
        return table

    table = build(compressed)
    assert torch.equal(table.token_id_map.cpu(), torch.from_numpy(mapping))

    with pytest.raises(ValueError, match="Regenerate the map"):
        build(compressed + 1)
