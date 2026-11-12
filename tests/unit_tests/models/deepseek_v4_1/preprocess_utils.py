# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Drive a V4.1 model through its real preprocessing seam, on CPU.

``Decoder.preprocess_inputs`` is the seam under test: it builds the per-layer
attention-metadata map out of the attention backends' builders and, under context
parallelism, shards the inputs.  A CPU unit test has no process group, so the framework's
The geometry map is produced by the model's shared metadata seam. Context parallelism is deferred
for this upgrade.
"""

import importlib
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import DefaultDeviceType
from torchtitan.distributed.spmd_types import spmd_dense_mesh

from tests.unit_tests.models.mtp_test_utils import build_cpu_model

# ``preprocess_inputs`` takes the parallelism context; these models run a single process
# with no CP and the default SPMD backend.
_NO_CP_DIMS = SimpleNamespace(cp_enabled=False, get_mesh=lambda name: None, spmd_dense_mesh=spmd_dense_mesh)
_NO_SPMD = SimpleNamespace(
    spmd_backend="partial_dtensor",
    context_parallel_load_balancer=None,
    context_parallel_ptrr_mask_key=None,
)


def tiny_v41_model(monkeypatch, *, configure=None):
    """The registered debug flavor at CPU-sized widths, entered through the registry.

    ``configure`` runs on the config tree before the build, which is where a test swaps
    the attention backend the way an override factory does.
    """
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    make_config = registry._make_v41_config

    def tiny_config(**kwargs):
        kwargs.update(
            dim=8,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=4,
            n_groups=1,
            index_n_heads=2,
            index_head_dim=4,
            moe_inter_dim=16,
            vision_dim=8,
            vision_heads=2,
            vision_inter_dim=16,
        )
        return make_config(**kwargs)

    monkeypatch.setattr(registry, "_make_v41_config", tiny_config)
    from torchtitan.models.common import decoder as decoder_module

    monkeypatch.setattr(decoder_module, "annotate_input_spmd_types", lambda _ctx, batch, _layouts: batch)
    config = registry.model_registry("deepseek_v4_1_debugmodel_text")
    config.vocab_size = config.tok_embeddings.num_embeddings = config.lm_head.out_features = 64
    if configure is not None:
        configure(config)
    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config)
    return model, config


def preprocess(model, tokens, labels, positions, *, parallel_dims=None, parallelism=None, **batch):
    """Run the model's own input preprocessing and return its forward arguments.

    The metadata is the seam's product: the model no longer builds one object for every
    layer, so what a migrated stack must reach its forward through is the **map**
    (``attention_metadata``), not the pinned single ``attention_masks``.

    ``batch`` carries the dataloader's other keys (the modality tensors, say) straight into
    ``input_dict``, exactly as the trainer hands them over.
    """
    inputs, labels, extra_kwargs, _ = model.preprocess_inputs(
        {"input": tokens, "labels": labels, "positions": positions, **batch},
        parallelism_context=_NO_CP_DIMS if parallel_dims is None else parallel_dims,
        parallelism=_NO_SPMD if parallelism is None else parallelism,
    )
    assert "attention_metadata" in extra_kwargs, "expected the attention backends to build the metadata map"
    return inputs, labels, extra_kwargs


def layer_metadata(layer, extra_kwargs):
    """The map entry one layer reads, selected by its own attention backend key.

    The same selection the forward makes, so a test that hands a layer its metadata
    exercises the dispatch rather than guessing which entry belongs to which stack.
    """
    return extra_kwargs["attention_metadata"][layer.attention.attention_metadata_key]


class FakeCpMesh:
    """The two ``cp_mesh`` methods the seam reads, without a real process group."""

    def __init__(self, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


def cp_parallel_dims(cp_size: int, cp_rank: int) -> SimpleNamespace:
    """A parallelism context with CP on, over :class:`FakeCpMesh`."""
    mesh = FakeCpMesh(cp_size, cp_rank)
    return SimpleNamespace(cp_enabled=True, get_mesh=lambda name: mesh, spmd_dense_mesh=spmd_dense_mesh)
