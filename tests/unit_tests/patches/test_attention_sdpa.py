# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract tests for dense block-causal SDPA support.

The cases are deliberately layered so a failure identifies the broken seam:

* ``test_block_causal_sdpa_mask_matches_document_reference`` checks the pure
  mask contract against an explicit two-document reference matrix.  It does
  not involve a model or an attention kernel.
* The Qwen3 recipe and ``Decoder.get_attention_masks`` cases verify that the
  real 1.7B configuration selects SDPA + ``block_causal`` and returns the
  dense ``[batch, 1, seq, seq]`` mask expected by SDPA.
* ``test_sdpa_block_causal_attention_config_is_allowed`` exercises the exact
  ``BaseAttention.Config`` constructor that upstream previously rejected.
* ``test_sdpa_forwards_dense_mask_to_functional_sdpa`` spies on
  ``F.scaled_dot_product_attention`` and also compares its output with an
  independent functional-SDPA reference.  This proves both mask forwarding
  and numerical behavior without requiring an NPU in unit tests.
* ``test_trainer_post_dataloading_process_requests_sdpa_mask`` covers the
  runtime plumbing that supplies the mask to the model; testing only the
  decoder helper would miss this dispatch point.
The separate packing test in ``test_chat_dataset_packing.py`` verifies the
data-loader consequence: the opt-in block-causal recipe keeps greedy packing,
while the ordinary causal SDPA recipe continues to pad instead.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.models.common import ScaledDotProductAttention
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.decoder import Decoder

from torchtitan_npu.patches.torchtitan.attention import (
    build_block_causal_sdpa_mask,
)
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


def test_block_causal_sdpa_mask_matches_document_reference():
    tokens = torch.tensor([[10, 11, 0, 12, 13, 0]])
    mask = build_block_causal_sdpa_mask(tokens, eos_id=0)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    assert mask.shape == (1, 1, 6, 6)
    torch.testing.assert_close(mask[0, 0], expected)


def test_qwen3_block_causal_sdpa_get_attention_masks_returns_dense_mask():
    model_config = _qwen3_model_config()
    fake_decoder = SimpleNamespace(config=model_config)
    tokens = torch.tensor([[10, 11, 0, 12, 13, 0]])

    mask = Decoder.get_attention_masks(
        fake_decoder,
        tokens,
        tokenizer=SimpleNamespace(eos_id=0),
    )

    assert isinstance(model_config.layers[0].attention.inner_attention, ScaledDotProductAttention.Config)
    assert model_config.layers[0].attention.mask_type == "block_causal"
    assert mask.shape == (1, 1, 6, 6)
    torch.testing.assert_close(mask, build_block_causal_sdpa_mask(tokens, eos_id=0))


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
    mask = build_block_causal_sdpa_mask(torch.tensor([[10, 11, 0, 12, 13, 0]]), eos_id=0)
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
    tokens = torch.tensor([[10, 11, 0, 12, 13, 0]])
    fake_model = SimpleNamespace(
        get_attention_masks=lambda **kwargs: build_block_causal_sdpa_mask(
            kwargs["input_batch"], kwargs["tokenizer"].eos_id
        )
    )
    fake_trainer = SimpleNamespace(
        model_config=model_config,
        parallel_dims=SimpleNamespace(cp_enabled=False),
        tokenizer=SimpleNamespace(eos_id=0),
        model_parts=[fake_model],
    )

    inputs, _labels, _extra_inputs, extra_kwargs = trainer_module.Trainer.post_dataloading_process(
        fake_trainer,
        {"input": tokens, "positions": torch.arange(tokens.shape[1]).unsqueeze(0)},
        torch.zeros_like(tokens),
    )

    assert inputs is tokens
    torch.testing.assert_close(extra_kwargs["attention_masks"], build_block_causal_sdpa_mask(tokens, 0))


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

    with patch(
        "torchtitan_npu.patches.torchtitan.chat_dataset.get_trainer_config",
        return_value=trainer_config,
    ), patch(
        "torchtitan.hf_datasets.text_datasets.load_dataset",
        return_value=mock_ds,
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
