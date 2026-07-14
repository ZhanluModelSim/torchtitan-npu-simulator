# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DSV4_DIR = REPO_ROOT / "torchtitan_npu" / "models" / "deepseek_v4"


class DeepSeekV4ParallelizeStaticTest(unittest.TestCase):
    def test_tp_ratio4_smla_li_compute_plan_uses_mask_dispatch(self):
        parallelize_source = (DSV4_DIR / "parallelize.py").read_text(encoding="utf-8")
        smla_source = (
            REPO_ROOT / "torchtitan_npu" / "converters" / "kernels" / "npu_smla.py"
        ).read_text(encoding="utf-8")
        tnd_source = (DSV4_DIR / "tnd.py").read_text(encoding="utf-8")

        self.assertIn("if parallel_dims.tp_enabled:", parallelize_source)
        self.assertIn("apply_non_moe_tp(", parallelize_source)
        self.assertIn(
            "FlexAttention.Config, VarlenAttention.Config", parallelize_source
        )
        self.assertIn(
            "use_attention_masks = _model_uses_attention_masks(model_args)",
            parallelize_source,
        )
        self.assertIn("if compress_ratio == 4:", parallelize_source)
        self.assertIn(
            "li_compute_smla_plan if use_attention_masks else li_compute_plan",
            parallelize_source,
        )
        self.assertIn(
            '"attention.inner_attention.li_compute": li_compute_parallel_plan',
            parallelize_source,
        )
        self.assertIn("VarlenAttention.Config()", tnd_source)
        self.assertIn("SimpleNamespace(inner_attention=inner_attention", tnd_source)
        self.assertNotIn("__class__.__name__", parallelize_source)
        self.assertNotIn("_smla_metadata_cache", parallelize_source)
        self.assertNotIn("._get_fsdp_state()", parallelize_source)
        self.assertIn("num_heads_q = query.shape[-2]", smla_source)
        self.assertIn("ctx.num_heads_q = num_heads_q", smla_source)
        self.assertIn("indexer_q.shape[-2]", smla_source)
        self.assertNotIn('"num_heads_q": model_args.n_heads', smla_source)


if __name__ == "__main__":
    unittest.main()
