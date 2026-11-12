# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native DCP continuation with Host Engram and mixed dense/sparse optimizers."""

import copy
import json

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torchtitan.components.optim import AdamW
from torchtitan.components.optim.lr_scheduler import LRSchedulersContainer

from tests.unit_tests.models.common.text_data_utils import IntegerTextTokenizer, aligned_text_config
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import (
    HostSparseOptimizersContainer,
    SparseAdam,
)
from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable


class _SparseTrainingState:
    """Completed-step and consumed-microbatch counters saved as extra state."""

    def __init__(self):
        self.step = 0
        self.cursor = 0

    def state_dict(self):
        return {"step": self.step, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.cursor = state["cursor"]


def _build_training_components(data_path):
    model = torch.nn.Module()
    model.table = HostEngramTable.Config(
        vocab_size=16,
        layer_id=0,
        ngram_orders=(2,),
        num_heads=1,
        head_vocab_sizes=(11,),
        embedding_dim=4,
        num_embeddings=12,
        require_token_id_map=False,
        pin_memory=False,
    ).build()
    model.register_parameter("dense", torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.7, 1.1])))
    # Never used before the save. DCP must restore its zero, lazily created moments too.
    model.register_parameter("late_dense", torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.6, 0.1])))
    with torch.no_grad():
        model.table.weight.copy_(torch.arange(48).reshape(12, 4) / 16)
    optimizers = HostSparseOptimizersContainer.Config(
        optimizers=[
            SparseAdam.Config(pattern=r"table\.weight$", lr=0.05),
            AdamW.Config(pattern=r".*", lr=0.02, fused=False, foreach=False),
        ],
    ).build(model_parts=[model])
    schedulers = LRSchedulersContainer.Config(warmup_steps=0, total_steps=4, min_lr_factor=0.1).build(
        optimizers=optimizers, training_steps=4
    )
    loader = aligned_text_config(data_path, alignment=2).build(
        tokenizer=IntegerTextTokenizer(),
        dp_world_size=1,
        dp_rank=0,
        max_context_length=4,
        num_tokens_per_microbatch=4,
    )
    return model, optimizers, schedulers, _SparseTrainingState(), loader


def _train_step(model, optimizers, schedulers, state, loader):
    optimizers.zero_grad(set_to_none=True)
    # Use the checkpointed production loader's cursor, not the step counter,
    # to choose the next input. Repeated IDs still exercise sparse coalescing.
    rows = next(iter(loader)).input
    output = model.table._distributed_lookup(rows)
    dense = model.dense if state.step < 2 else model.dense + model.late_dense
    loss = (output * dense).square().mean()
    loss.backward()
    assert model.table.pending_sparse_grad() is not None
    norm = optimizers.clip_grad_norm(model.parameters(), 1.0, clipper=torch.nn.utils.clip_grad_norm_)
    optimizers.step()
    schedulers.step()
    state.step += 1
    state.cursor += 1
    assert model.table.pending_sparse_grad() is None
    return loss.detach().clone(), norm.detach().clone(), rows.clone()


def _assert_exact_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        wanted = expected[key]
        if isinstance(value, torch.Tensor):
            assert value.dtype == wanted.dtype
            torch.testing.assert_close(value, wanted, rtol=0, atol=0)
        else:
            assert value == wanted


def test_native_dcp_resumes_host_sparse_adam_and_late_dense_state_exactly(tmp_path):
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    managers = []
    loaders = []
    try:
        data_path = tmp_path / "text.jsonl"
        texts = ("0 7 7", "1 7 1", "2 7 2", "0 4 4", "11 3 10")
        data_path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
        continuous = _build_training_components(data_path)
        loaders.append(continuous[-1])
        continuous_metrics = [_train_step(*continuous) for _ in range(4)]
        for index, text in enumerate(texts[:4]):
            torch.testing.assert_close(
                continuous_metrics[index][2], torch.tensor([1, *map(int, text.split())]), rtol=0, atol=0
            )
        interrupted = _build_training_components(data_path)
        loaders.append(interrupted[-1])
        for step in range(2):
            actual = _train_step(*interrupted)
            for value, wanted in zip(actual, continuous_metrics[step], strict=True):
                torch.testing.assert_close(value, wanted, rtol=0, atol=0)
        model, optimizers, schedulers, state, loader = interrupted
        manager = CheckpointManager.Config(interval=2, last_save_model_only=False).build(
            model_parts=[model],
            optimizers=optimizers,
            lr_schedulers=schedulers,
            ema=None,
            extra_states={"train_state": state, "dataloader": loader},
            sd_adapter=None,
            base_folder=str(tmp_path),
        )
        managers.append(manager)
        assert manager.save(2)
        saved_keys = dcp.FileSystemReader(tmp_path / "checkpoint" / "step-2").read_metadata().state_dict_metadata
        assert {
            "table.weight",
            "optimizer.state.table.weight.step",
            "optimizer.state.table.weight.exp_avg",
            "optimizer.state.table.weight.exp_avg_sq",
        } <= saved_keys.keys()
        assert any(key.startswith("dataloader.") for key in saved_keys)
        saved_optimizer = copy.deepcopy(optimizers.state_dict())
        assert saved_optimizer["state.late_dense.step"].item() == 0
        expected_metrics = continuous_metrics[2:]

        resumed = _build_training_components(data_path)
        loaders.append(resumed[-1])
        restored_model, restored_optimizers, restored_schedulers, restored_state, restored_loader = resumed
        restore = CheckpointManager.Config(interval=2, last_save_model_only=False).build(
            model_parts=[restored_model],
            optimizers=restored_optimizers,
            lr_schedulers=restored_schedulers,
            ema=None,
            extra_states={"train_state": restored_state, "dataloader": restored_loader},
            sd_adapter=None,
            base_folder=str(tmp_path),
        )
        managers.append(restore)
        assert restore.load(2)
        assert restored_state.step == restored_state.cursor == 2
        _assert_exact_state(restored_optimizers.state_dict(), saved_optimizer)
        actual_metrics = [_train_step(*resumed) for _ in range(2)]

        for actual, expected in zip(actual_metrics, expected_metrics, strict=True):
            for value, wanted in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, wanted, rtol=0, atol=0)
        model, optimizers, schedulers, state, loader = continuous
        _assert_exact_state(restored_model.state_dict(), model.state_dict())
        _assert_exact_state(restored_optimizers.state_dict(), optimizers.state_dict())
        assert restored_schedulers.state_dict() == schedulers.state_dict()
        assert restored_state.state_dict() == state.state_dict() == {"step": 4, "cursor": 4}
        torch.testing.assert_close(restored_loader.state_dict(), loader.state_dict(), rtol=0, atol=0)
        next_expected = torch.tensor([1, 11, 3, 10])
        torch.testing.assert_close(next(iter(loader)).input, next_expected, rtol=0, atol=0)
        torch.testing.assert_close(next(iter(restored_loader)).input, next_expected, rtol=0, atol=0)
    finally:
        for manager in managers:
            manager.close()
        for loader in loaders:
            loader.close()
        dist.destroy_process_group()
