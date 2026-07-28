# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end tests for the greedy packing fallback in the ChatDataset patch.

During training, ``ChatDataLoader.__init__`` loads the configured dataset via::

    dataset = datasets.load_dataset(
        config.dataset_path,
        **config.load_dataset_kwargs,
    )

For instance, given a ``train.jsonl`` file containing::

    {"q": "AAAA", "a": "BBBB"}
    {"q": "CCCC", "a": "DDDD"}
    {"q": "EEEE", "a": "FFFF"}

``load_dataset`` returns a ``Dataset`` whose iteration yields the three dictionaries
in the order shown above.

Instead of reading from an actual file, this test patches ``load_dataset`` to return
the dataset json directly. All subsequent steps follow the normal training
pipeline. ``ChatDataLoader`` passes each record to ``_process_sample``, which converts
the first record into chat messages of the form::

    [
        {"role": "user", "content": "AAAA"},
        {"role": "assistant", "content": "BBBB"},
    ]

Instead of a real tokenizer, the test supplies a mock ``ChatMLTokenizer`` to
``ChatDataLoader``. This test tokenizer renders the messages as::

    <|im_start|>user\nAAAA<|im_end|><|im_start|>assistant\nBBBB<|im_end|>

``ChatDataset`` calls its ``encode`` method with ``add_bos=True`` and
``add_eos=False``. The method therefore maps each character to its ASCII
integer, prepends the BOS token (``200``), and does not append an EOS token.
The helper ``sample_content_text`` extracts the original ``AAAABBBB``-like
content from the tokenized output for validation.

The end-to-end test:

With SDPA, the patch pads the sequence to ``SEQ_LEN=200`` using EOS for the
input and ``IGNORE_INDEX`` for the labels, so the first batch contains only the
tokenized representation of ``AAAABBBB``.

With varlen/block-causal attention, the 68‑token sequences remain unpadded; two
such sequences fit into the first 200‑token packed sequence, yielding a combined
sample content of ``AAAABBBBCCCCDDDD``.
"""

from unittest.mock import patch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.common import ScaledDotProductAttention

from torchtitan_npu.models.common.npu_varlen_attention import NPUVarlenAttention
from torchtitan_npu.models.qwen3.config_registry import (
    sft_qwen3_1_7b_wordle,
    sft_qwen3_30ba3b_gsm8k_tnd,
)
from tests.unit_tests.patches.chat_dataset_test_utils import (
    DATASET_RECORDS,
    SEQ_LEN,
    ChatMLTokenizer,
    make_chat_dataloader_config,
    next_single_sequence,
    sample_content_text,
)


class TestSDPADisablesPacking:
    """ChatDataLoaderConfig + SDPA (mask_type=causal) → no greedy packing."""

    def test_sdpa_no_packing(self):
        from datasets import Dataset
        from torchtitan.hf_datasets.text_datasets import ChatDataLoader

        trainer_config = sft_qwen3_1_7b_wordle()
        attention = trainer_config.model_spec.model.layers[0].attention
        assert isinstance(attention.inner_attention, ScaledDotProductAttention.Config)
        assert attention.mask_type == "causal"
        mock_ds = Dataset.from_list(DATASET_RECORDS)

        with patch(
            "torchtitan_npu.patches.torchtitan.chat_dataset.get_trainer_config",
            return_value=trainer_config,
        ), patch(
            "torchtitan.hf_datasets.text_datasets.load_dataset",
            return_value=mock_ds,
        ), patch(
            "torchtitan_npu.patches.torchtitan.chat_dataset.logger.warning"
        ) as warning_mock:
            loader = ChatDataLoader(
                make_chat_dataloader_config(),
                dp_world_size=1,
                dp_rank=0,
                tokenizer=ChatMLTokenizer(),
                seq_len=SEQ_LEN,
                local_batch_size=1,
            )

        assert loader.dataset._greedy_packing is False
        warning_mock.assert_called_once()
        assert "Greedy packing is disabled" in warning_mock.call_args.args[0]
        assert "differs from upstream ChatDataset behavior" in warning_mock.call_args.args[0]

        input_ids, label_ids = next_single_sequence(loader)
        assert len(input_ids) == SEQ_LEN
        assert sample_content_text(input_ids) == "AAAABBBB"

        first_padding_index = input_ids.index(ChatMLTokenizer.eos_id)
        assert first_padding_index == 68
        padding_length = SEQ_LEN - first_padding_index
        # Every input position after the 68-token chat must be padded with EOS.
        assert input_ids[first_padding_index:] == [ChatMLTokenizer.eos_id] * padding_length
        # Labels at those padding positions must not contribute to the loss.
        assert label_ids[first_padding_index:] == [IGNORE_INDEX] * padding_length


class TestVarlenKeepsPacking:
    """ChatDataLoaderConfig + VarlenAttention (mask_type=block_causal) → greedy packing."""

    def test_varlen_packing(self):
        from datasets import Dataset
        from torchtitan.hf_datasets.text_datasets import ChatDataLoader

        trainer_config = sft_qwen3_30ba3b_gsm8k_tnd()
        attention = trainer_config.model_spec.model.layers[0].attention
        assert isinstance(attention.inner_attention, NPUVarlenAttention.Config)
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


class TestPretrainingUnaffected:
    """HuggingFaceTextDataset (pretraining) packs regardless of mask_type."""

    def test_pretraining_always_packs(self):
        from datasets import Dataset
        from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

        text_samples = [
            {"text": "AAAA BBBB"},
            {"text": "CCCC DDDD"},
            {"text": "EEEE FFFF"},
        ]
        mock_ds = Dataset.from_list(text_samples)

        with patch(
            "torchtitan.hf_datasets.text_datasets.split_dataset_by_node",
            return_value=mock_ds,
        ), patch(
            "torchtitan.hf_datasets.text_datasets._validate_dataset",
            return_value=("mock", lambda p: mock_ds, lambda s: s["text"]),
        ):
            ds = HuggingFaceTextDataset(
                dataset_name="mock",
                dataset_path=None,
                tokenizer=ChatMLTokenizer(),
                seq_len=50,
                dp_rank=0,
                dp_world_size=1,
                infinite=False,
            )

        assert not hasattr(ds, "_greedy_packing")

        results = list(iter(ds))
        total_batches = len(results)
        assert total_batches < len(text_samples), (
            f"Pretraining should pack multiple docs into fewer batches, "
            f"got {total_batches} batches from {len(text_samples)} samples"
        )
