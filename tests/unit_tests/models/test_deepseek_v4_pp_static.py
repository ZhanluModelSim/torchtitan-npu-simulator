# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
DSV4_DIR = REPO_ROOT / "torchtitan_npu" / "models" / "deepseek_v4"


def _trainer_config(*, num_mtp_modules=0, pipeline_parallel_degree=1):
    # Minimal duck-typed Trainer.Config for ``Config.update_from_config``.
    return SimpleNamespace(
        training=SimpleNamespace(seq_len=1024, num_mtp_modules=num_mtp_modules),
        parallelism=SimpleNamespace(
            context_parallel_degree=1,
            pipeline_parallel_degree=pipeline_parallel_degree,
        ),
        debug=SimpleNamespace(moe_force_load_balance=False),
        model_converters=SimpleNamespace(converters=[]),
    )


def _class_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    raise AssertionError(f"{class_name} not found in {path}")


class DeepSeekV4PipelineStaticTest(unittest.TestCase):
    def test_model_registry_uses_upstream_pipeline_llm(self):
        # DeepSeek-V4 registers upstream pipeline_llm directly as its
        # pipelining_fn (no custom wrapper), like deepseek_v3/v32.
        from torchtitan.distributed.pipeline_parallel import pipeline_llm

        from torchtitan_npu.models.deepseek_v4 import model_registry

        self.assertIs(model_registry("mini_1b").pipelining_fn, pipeline_llm)

    def test_config_layers_matches_total_layer_count(self):
        # ``Config.layers`` lets upstream pipeline_llm derive the layer count via
        # ``len(model_config.layers)`` even though DeepSeek-V4 stores ``n_layers``.
        from torchtitan_npu.models.deepseek_v4 import deepseekv4_configs

        config = deepseekv4_configs["mini_1b"]()
        self.assertEqual(len(config.layers), config.n_layers + config.num_mtp_modules)

    def test_update_from_config_rejects_mtp_with_pp(self):
        from torchtitan_npu.models.deepseek_v4 import deepseekv4_configs

        config = deepseekv4_configs["mini_1b"]()
        with self.assertRaisesRegex(NotImplementedError, r"MTP \+ PP"):
            config.update_from_config(trainer_config=_trainer_config(num_mtp_modules=1, pipeline_parallel_degree=2))

    def test_update_from_config_allows_mtp_without_pp(self):
        from torchtitan_npu.models.deepseek_v4 import deepseekv4_configs

        config = deepseekv4_configs["mini_1b"]()
        config.update_from_config(trainer_config=_trainer_config(num_mtp_modules=1, pipeline_parallel_degree=1))
        self.assertEqual(config.num_mtp_modules, 1)

    def test_model_forward_declares_pp_sidecar_protocol(self):
        path = DSV4_DIR / "model.py"
        source = path.read_text()
        methods = _class_methods(path, "DeepSeekV4Model")

        self.assertIn("_normalize_pp_input_ids", methods)
        self.assertIn("input_ids = self._normalize_pp_input_ids(input_ids)", source)
        self.assertIn("if self.output is None:", source)
        self.assertIn("return h", source)
        self.assertIn('layer_id = cast("Any", layer).layer_id', source)
        self.assertIn("if layer_id < self.model_args.n_layers", source)

    def test_parallelize_uses_dynamic_root_plan_for_pp_chunks(self):
        source = (DSV4_DIR / "parallelize.py").read_text()

        self.assertIn("root_parallelize_plan: dict[str, Any] = {}", source)
        self.assertIn('tok_embeddings = getattr(model, "tok_embeddings", None)', source)
        self.assertIn("if tok_embeddings is not None:", source)
        # hc_head TP now applies to the last layer inside the per-block loop.
        self.assertIn('getattr(transformer_block, "is_last_layer", False)', source)
        self.assertIn('"hc_head": hc_head_plan', source)
        self.assertIn("hc_head_plan = HcHeadParallelStyle()", source)
        self.assertIn("use_local_input=False", source)
        self.assertIn("apply_distributed_indexer_loss_tracking(", source)
        self.assertIn('model_args = cast("Any", model).model_args', source)
        self.assertIn(
            "parallel_dims, model_args.n_layers, model_args.compress_ratios",
            source,
        )

    def test_hc_head_parallel_style_owns_parameter_registration(self):
        source = (DSV4_DIR / "parallelize.py").read_text()
        methods = _class_methods(DSV4_DIR / "parallelize.py", "HcHeadParallelStyle")

        self.assertIn("ParallelStyle", source)
        self.assertIn("class HcHeadParallelStyle(ParallelStyle):", source)
        self.assertIn("partition_fn", methods)
        self.assertIn("_apply", methods)
        self.assertIn("distribute_module(", source)
        self.assertIn("HcHeadParallelStyle(", source)
        self.assertIn('"hc_head": hc_head_plan', source)
        self.assertIn('"mtp_hc_head": hc_head_plan', source)
        self.assertNotIn("for hc_head_param in", source)
        self.assertNotIn('"mtp_hc_head_fn"', source)
        self.assertNotIn("module._parameters", source)
        self.assertNotIn("self._io_plan._apply", source)
        self.assertNotIn("self._partition_fn", source)

    def test_pipelining_patch_injects_deepseek_v4_input_ids(self):
        patch_source = (REPO_ROOT / "torchtitan_npu" / "patches" / "torch" / "pipelining.py").read_text()
        pipeline_source = (DSV4_DIR / "pipeline_parallel.py").read_text()

        self.assertIn(
            "_patch_post_dataloading_process_for_deepseek_v4_pp_input_ids",
            patch_source,
        )
        self.assertIn(
            "from torchtitan_npu.models.deepseek_v4.pipeline_parallel import",
            patch_source,
        )
        self.assertNotIn("def _with_deepseek_v4_pp_input_ids", patch_source)
        self.assertNotIn("def _get_trainer_model_name", pipeline_source)
        self.assertNotIn("job_config", pipeline_source)
        self.assertIn("def _is_deepseek_v4_pp_target", pipeline_source)
        self.assertIn(
            'model_spec = getattr(trainer.config, "model_spec", None)',
            pipeline_source,
        )
        self.assertIn("def _with_deepseek_v4_pp_input_ids", pipeline_source)
        self.assertIn(
            'extra_kwargs["input_ids"] = input_ids.detach().long()',
            pipeline_source,
        )
        self.assertIn(
            'if devices is not None and "device_type" not in kwargs',
            patch_source,
        )
        self.assertIn('kwargs["device_type"] = "npu"', patch_source)

    def test_dsa_tracker_records_zero_based_layers_and_distributed_reduce(self):
        common_source = (REPO_ROOT / "torchtitan_npu" / "models" / "common" / "dsa_indexer_loss.py").read_text()
        parallelize_source = (DSV4_DIR / "parallelize.py").read_text()

        self.assertIn('tracker["values"][layer_number]', common_source)
        self.assertIn("valid = das_indexer_losses != 0", common_source)
        self.assertNotIn('tracker["present"]', common_source)
        self.assertIn("valid_indices = [", parallelize_source)
        self.assertIn("compress_ratios[i] == 4", parallelize_source)
        self.assertIn("dist.all_reduce(dsa_indexer_losses", parallelize_source)
        self.assertIn(
            "norm_factor = dist.get_world_size() // max(parallel_dims.pp, 1)",
            parallelize_source,
        )


if __name__ == "__main__":
    unittest.main()
