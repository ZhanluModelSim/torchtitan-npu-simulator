# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host Engram joint-graph training with equal FSDP and EP degrees."""

import math
import os
from dataclasses import replace
from unittest.mock import patch

import torch
from torchtitan.config.parallelism import ParallelismConfig

from torchtitan_npu.models.deepseek_v4_1 import _flash_engram_geometry
from torchtitan_npu_recipes.graph_trainer.deepseek_v4_1 import _make_graph_trainer_config
from torchtitan_npu_recipes.models.deepseek_v4_1 import _overrides, _set_context, _set_muon, _set_schedule


def _recipe(*, ep_overlap, auto_overlap):
    # Identity token mapping matches the committed miniature tokenizer fixture.
    geometry = replace(
        _flash_engram_geometry(),
        vocab_size_per_ngram=(1024, 1024, 1024),
        require_token_id_map=False,
        compressed_vocab_size=None,
    )
    seq_len = 512
    config = _make_graph_trainer_config(
        engram=geometry,
        hf_assets_path="tests/assets/deepseek_v3",
        seq_len=seq_len,
        parallelism=ParallelismConfig(data_parallel_shard_degree=8, expert_parallel_degree=8),
    )
    # Include the first compressed source and its cross-layer consumer.
    config.model.layers = config.model.layers[:4]
    config.model.n_layers = 4
    config.model.compress_ratios = config.model.compress_ratios[:4]
    config.dataloader.dataset.dataset.source.load_dataset_kwargs["data_files"] = "tests/assets/c4_test/data.json"
    config.compile.ep_overlap.enabled = ep_overlap
    config.compile.pass_pipeline = "mutation-functionalization" + ("+npu_auto_overlap" if auto_overlap else "")
    _set_context(config, sequence_length=seq_len, token_budget=-1, steps=3)
    _set_muon(config)
    _set_schedule(config)
    config.override.imports = _overrides(a5=False, debug=True)
    config.checkpointer = None
    config.debug.seed = config.dataloader.seed = 42
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.training.extension.allow_hf32 = False
    config.metrics.enable_tensorboard = True
    config.metrics.log_freq = 1
    config.metrics.save_tb_folder = f"tb_phase_{os.environ.get('TT_NPU_INTEGRATION_PHASE', '0')}"
    return config


def basic():
    return _recipe(ep_overlap=False, auto_overlap=False)


def auto():
    return _recipe(ep_overlap=True, auto_overlap=True)


def main():
    from torchtitan.observability.metrics import MetricsProcessor

    from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer
    from torchtitan_npu.train import main as train_main

    original_step = HostSparseOptimizersContainer.step
    original_log = MetricsProcessor.log

    def verified_log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        assert all(math.isfinite(value) for value in (global_avg_loss, global_max_loss, grad_norm)), (
            f"Non-finite training metrics at step {step}: loss={global_avg_loss}, norm={grad_norm}"
        )
        return original_log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics)

    def verified_step(self, closure=None):
        tables = list(self._iter_host_tables())
        assert tables, "Host Engram was not enabled"
        previous = []
        for table in tables:
            assert table.weight.device.type == "cpu" and table.graph_training
            assert table.lookup_mesh.size() == 8 and table._replica_size == 1
            pending = table.pending_sparse_grad()
            assert pending is not None and pending.is_sparse and torch.isfinite(pending.values()).all()
            previous.append(table.weight.detach().clone())
        result = original_step(self, closure)
        for table, old in zip(tables, previous, strict=True):
            assert not torch.equal(old, table.weight), "Host table did not update"
            assert table.pending_sparse_grad() is None
        return result

    with (
        patch.object(HostSparseOptimizersContainer, "step", verified_step),
        patch.object(MetricsProcessor, "log", verified_log),
    ):
        train_main()


if __name__ == "__main__":
    main()
