# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torchtitan.components.loss import IGNORE_INDEX

from torchtitan_npu.models.deepseek_v4.tnd import (
    build_smla_attention_masks,
    smla_global_tnd_post_dataloading_process,
)


def _model_args(num_mtp_modules: int) -> SimpleNamespace:
    return SimpleNamespace(
        compress_ratios=(1,),
        mtp_layer_compress_ratio=1,
        num_mtp_modules=num_mtp_modules,
        use_global_tnd=True,
    )


def _trainer_config(*, converters) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(seq_len=1024, num_mtp_modules=0),
        parallelism=SimpleNamespace(
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        debug=SimpleNamespace(moe_force_load_balance=False),
        model_converters=SimpleNamespace(converters=converters),
    )


class DeepSeekV4TNDTest(unittest.TestCase):
    def test_global_tnd_rejects_pp_or_cp(self):
        from torchtitan_npu.models.deepseek_v4.parallelize import _validate_deepseek_v4_parallelism

        model_args = SimpleNamespace(use_global_tnd=True)
        for parallel_dims, expected in (
            (SimpleNamespace(pp_enabled=True, cp_enabled=False), "does not support PP"),
            (SimpleNamespace(pp_enabled=False, cp_enabled=True), "does not support CP"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(NotImplementedError, expected):
                _validate_deepseek_v4_parallelism(model_args, parallel_dims)

    def test_a5_smla_requires_both_mhc_converters(self):
        from torchtitan_npu.converters import get_model_converter_config
        from torchtitan_npu.models.deepseek_v4 import deepseekv4_configs

        smla = get_model_converter_config("npu_smla")
        mhc_pre = get_model_converter_config("npu_mhc_pre")
        config = deepseekv4_configs["mini_1b"]()
        with (
            patch("torchtitan_npu.models.deepseek_v4.model.get_npu_device_type", return_value="A5"),
            self.assertRaisesRegex(ValueError, "missing converter.*npu_mhc_post"),
        ):
            config.update_from_config(trainer_config=_trainer_config(converters=[smla, mhc_pre]))

    def test_tnd_attention_masks_cache_compressor_layout(self):
        positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        model_args = SimpleNamespace(
            compress_ratios=(4, 128),
            mtp_layer_compress_ratio=1,
            num_mtp_modules=0,
            use_global_tnd=True,
        )

        attention_masks = build_smla_attention_masks(positions, model_args)

        torch.testing.assert_close(attention_masks.block_starts_by_ratio[4], torch.tensor([0, 4]))
        self.assertEqual(attention_masks.block_starts_by_ratio[128].numel(), 0)
        self.assertEqual(attention_masks.max_seqlen_q, 4)
        self.assertEqual(attention_masks.max_seqlen_cmp_kv, {4: 1, 128: 0})

    def test_tnd_ratio4_reuses_supplied_compressor_block_starts(self):
        from torchtitan_npu.models.deepseek_v4 import model as deepseek_model

        class RecordingCompressor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.use_tnd_metadata = True
                self.block_starts = None

            def forward(self, x, freqs_cis, positions=None, block_starts=None):
                self.block_starts = block_starts
                return x

        class RecordingIndexer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block_starts = None

            def forward(
                self,
                x,
                qr,
                freqs_cis,
                hadamard_mat,
                positions=None,
                block_starts=None,
            ):
                self.block_starts = block_starts
                return x, x, x

        pre_attention = deepseek_model.PreAttention.__new__(deepseek_model.PreAttention)
        torch.nn.Module.__init__(pre_attention)
        pre_attention.n_heads = 1
        pre_attention.head_dim = 4
        pre_attention.rope_head_dim = 2
        pre_attention.eps = 1e-6
        pre_attention.compress_ratio = 4
        pre_attention.wq_a = torch.nn.Identity()
        pre_attention.q_norm = torch.nn.Identity()
        pre_attention.wq_b = torch.nn.Identity()
        pre_attention.wkv = torch.nn.Identity()
        pre_attention.kv_norm = torch.nn.Identity()
        pre_attention.compressor = RecordingCompressor()
        pre_attention.indexer = RecordingIndexer()

        positions = torch.arange(8)
        expected_block_starts = torch.tensor([0, 4])
        with patch.object(deepseek_model, "apply_rotary_emb", side_effect=lambda x, *args, **kwargs: x):
            pre_attention(
                torch.randn(8, 4),
                torch.empty(0),
                torch.eye(4),
                positions=positions,
                block_starts=expected_block_starts,
            )

        self.assertIs(pre_attention.indexer.block_starts, pre_attention.compressor.block_starts)
        torch.testing.assert_close(pre_attention.compressor.block_starts, expected_block_starts)

    def test_mtp_shifts_do_not_cross_packed_request_boundaries(self):
        inputs = torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]])
        labels = torch.tensor([[11, 12, 13, 14, 21, 22, 23, 24]])
        positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])

        inputs_tnd, labels_tnd, extra_inputs, extra_kwargs = smla_global_tnd_post_dataloading_process(
            {"input": inputs, "positions": positions},
            labels,
            _model_args(num_mtp_modules=2),
        )

        torch.testing.assert_close(inputs_tnd, torch.tensor([10, 11, 12, 13, 20, 21]))
        torch.testing.assert_close(extra_kwargs["positions"], torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.int32))
        torch.testing.assert_close(
            extra_kwargs["mtp_inputs"],
            torch.tensor(
                [
                    [11, 12, 13, 13, 21, 22],
                    [12, 13, 12, 13, 22, 23],
                ]
            ),
        )
        torch.testing.assert_close(
            labels_tnd,
            torch.tensor(
                [
                    [11, 12, 13, 14, 21, 22],
                    [12, 13, 14, IGNORE_INDEX, 22, 23],
                    [13, 14, IGNORE_INDEX, IGNORE_INDEX, 23, 24],
                ]
            ),
        )
        torch.testing.assert_close(
            extra_kwargs["attention_masks"].cu_seqlens_q,
            torch.tensor([0, 4, 6], dtype=torch.int32),
        )
        self.assertEqual(extra_inputs, {})

    def test_mtp_shifts_keep_single_request_behavior(self):
        inputs = torch.tensor([[10, 11, 12, 13, 14, 15]])
        labels = torch.tensor([[11, 12, 13, 14, 15, 16]])
        positions = torch.tensor([[0, 1, 2, 3, 4, 5]])

        _, labels_tnd, _, extra_kwargs = smla_global_tnd_post_dataloading_process(
            {"input": inputs, "positions": positions},
            labels,
            _model_args(num_mtp_modules=2),
        )

        torch.testing.assert_close(
            extra_kwargs["mtp_inputs"],
            torch.tensor(
                [
                    [11, 12, 13, 14],
                    [12, 13, 14, 15],
                ]
            ),
        )
        torch.testing.assert_close(
            labels_tnd,
            torch.tensor(
                [
                    [11, 12, 13, 14],
                    [12, 13, 14, 15],
                    [13, 14, 15, 16],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
