# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for dense block-causal SDPA and position-based packing boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torchtitan.models.common import ScaledDotProductAttention
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.decoder import Decoder

from tests.unit_tests.patches.chat_dataset_test_utils import (
    DATASET_RECORDS,
    SEQ_LEN,
    ChatMLTokenizer,
    make_chat_dataloader_config,
    next_single_sequence,
    sample_content_text,
)


def _qwen3_model_config():
    from torchtitan_npu.models.qwen3.config_registry import (
        sft_qwen3_1_7b_wordle_block_causal_sdpa,
    )

    return sft_qwen3_1_7b_wordle_block_causal_sdpa().model_spec.model


def _packed_chat_batch():
    # EOS appears inside the first two samples, while neither true sample join
    # (indices 6 and 9) is marked by EOS in the shifted model inputs.
    tokens = torch.tensor([[10, 11, 0, 12, 13, 14, 15, 0, 16, 17, 18]])
    positions = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1]])
    return tokens, positions


def _position_mask(tokens, positions):
    return Decoder.get_attention_masks(
        SimpleNamespace(config=_qwen3_model_config()),
        tokens,
        tokenizer=SimpleNamespace(eos_id=0),
        positions=positions,
    )


def test_position_boundaries_ignore_message_eos_and_split_packed_samples():
    tokens, positions = _packed_chat_batch()
    mask = _position_mask(tokens, positions)[0, 0]

    # The internal EOS at index 2 does not prevent later tokens in the same
    # conversation from attending to its prompt and history.
    assert mask[5, :6].all()
    # The true packed-sample join has no EOS token, but the position reset
    # prevents attention across it.
    assert not mask[6, 5]
    # The second sample's internal EOS also remains inside one document.
    assert mask[8, 6:9].all()
    assert not mask[9, 8]


def test_block_causal_sdpa_requires_positions():
    tokens, _ = _packed_chat_batch()

    with pytest.raises(ValueError, match="SDPA attention requires dataloader positions"):
        Decoder.get_attention_masks(
            SimpleNamespace(config=_qwen3_model_config()),
            tokens,
            tokenizer=SimpleNamespace(eos_id=0),
        )


def test_qwen3_block_causal_sdpa_get_attention_masks_returns_dense_mask():
    model_config = _qwen3_model_config()
    fake_decoder = SimpleNamespace(config=model_config)
    tokens, positions = _packed_chat_batch()

    mask = Decoder.get_attention_masks(
        fake_decoder,
        tokens,
        tokenizer=SimpleNamespace(eos_id=0),
        positions=positions,
    )

    assert isinstance(model_config.layers[0].attention.inner_attention, ScaledDotProductAttention.Config)
    assert model_config.layers[0].attention.mask_type == "block_causal"
    assert mask.shape == (1, 1, 11, 11)


def test_qwen3_block_causal_sdpa_recipe_configures_every_layer():
    model_config = _qwen3_model_config()

    assert all(
        isinstance(layer.attention.inner_attention, ScaledDotProductAttention.Config)
        and layer.attention.mask_type == "block_causal"
        for layer in model_config.layers
    )


def test_sdpa_block_causal_attention_config_is_allowed():
    config = BaseAttention.Config(
        n_heads=4,
        inner_attention=ScaledDotProductAttention.Config(),
        mask_type="block_causal",
    )

    assert config.mask_type == "block_causal"
    assert isinstance(config.inner_attention, ScaledDotProductAttention.Config)


def test_sdpa_forwards_dense_mask_to_functional_sdpa():
    from torchtitan.models.common import attention as attention_module

    torch.manual_seed(0)
    q = torch.randn(1, 6, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    tokens = torch.zeros((1, 6), dtype=torch.long)
    mask = _position_mask(tokens, torch.tensor([[0, 1, 2, 0, 1, 2]]))
    attention = ScaledDotProductAttention(ScaledDotProductAttention.Config())
    original_sdpa = attention_module.F.scaled_dot_product_attention

    with patch.object(
        attention_module.F,
        "scaled_dot_product_attention",
        wraps=original_sdpa,
    ) as sdpa:
        actual = attention(q, k, v, attention_masks=mask)

    kwargs = sdpa.call_args.kwargs
    assert kwargs["is_causal"] is False
    torch.testing.assert_close(kwargs["attn_mask"], mask)
    expected = original_sdpa(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        is_causal=False,
        enable_gqa=False,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_trainer_post_dataloading_process_requests_sdpa_mask():
    import torchtitan.trainer as trainer_module

    model_config = _qwen3_model_config()
    tokens, positions = _packed_chat_batch()
    decoder = SimpleNamespace(config=model_config)
    decoder.get_attention_masks = lambda **kwargs: Decoder.get_attention_masks(decoder, **kwargs)
    fake_trainer = SimpleNamespace(
        model_config=model_config,
        parallel_dims=SimpleNamespace(cp_enabled=False),
        tokenizer=SimpleNamespace(eos_id=0),
        model_parts=[decoder],
    )

    inputs, _labels, extra_inputs, extra_kwargs = trainer_module.Trainer.post_dataloading_process(
        fake_trainer,
        {"input": tokens, "positions": positions},
        torch.zeros_like(tokens),
    )

    assert inputs is tokens
    assert "attention_masks" not in extra_inputs
    torch.testing.assert_close(extra_kwargs["positions"], positions)
    torch.testing.assert_close(
        extra_kwargs["attention_masks"],
        _position_mask(tokens, positions),
    )


def test_block_causal_sdpa_keeps_greedy_packing():
    """Dense-mask SDPA + block-causal attention keeps greedy packing enabled."""

    from datasets import Dataset
    from torchtitan.hf_datasets.text_datasets import ChatDataLoader

    from torchtitan_npu.models.qwen3.config_registry import (
        sft_qwen3_1_7b_wordle_block_causal_sdpa,
    )

    trainer_config = sft_qwen3_1_7b_wordle_block_causal_sdpa()
    attention = trainer_config.model_spec.model.layers[0].attention
    assert isinstance(attention.inner_attention, ScaledDotProductAttention.Config)
    assert attention.mask_type == "block_causal"
    mock_ds = Dataset.from_list(DATASET_RECORDS)

    with (
        patch(
            "torchtitan_npu.patches.torchtitan.chat_dataset.get_trainer_config",
            return_value=trainer_config,
        ),
        patch(
            "torchtitan.hf_datasets.text_datasets.load_dataset",
            return_value=mock_ds,
        ),
    ):
        loader = ChatDataLoader(
            make_chat_dataloader_config(),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=ChatMLTokenizer(),
            seq_len=SEQ_LEN,
            local_batch_size=1,
        )

    assert loader.dataset._greedy_packing is True
    input_ids, _label_ids = next_single_sequence(loader)
    assert sample_content_text(input_ids) == "AAAABBBBCCCCDDDD"
