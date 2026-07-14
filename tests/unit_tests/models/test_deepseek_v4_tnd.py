# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torchtitan.components.loss import IGNORE_INDEX

from torchtitan_npu.models.deepseek_v4.tnd import smla_global_tnd_post_dataloading_process


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
        config = deepseekv4_configs["smoketest"]()
        with (
            patch("torchtitan_npu.models.deepseek_v4.model.get_npu_device_type", return_value="A5"),
            self.assertRaisesRegex(ValueError, "missing converter.*npu_mhc_post"),
        ):
            config.update_from_config(trainer_config=_trainer_config(converters=[smla, mhc_pre]))

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
