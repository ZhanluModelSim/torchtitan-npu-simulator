# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torchtitan.config import OverrideConfig, apply_overrides
from torchtitan.config.parallelism import ParallelismConfig

import torchtitan_npu.override.deepseek_v4.lightning_indexer.pypto_pro as pypto_pro
from torchtitan_npu.models.deepseek_v4 import build_model_config
from torchtitan_npu.ops.pypto_pro.li_lig_wrapper import _check_compact_guard
from torchtitan_npu.override.deepseek_v4.metadata import AscV4Metadata

pytestmark = pytest.mark.usefixtures("dsv4")


def _debug_model(*, use_pro):
    config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    # The same geometry can first be seen on an ordinary layer.
    csa = next(layer for layer in config.layers if layer.attention.compress_ratio == 4)
    config.layers = [deepcopy(csa), deepcopy(csa), csa]
    indexer = config.layers[0].attention.indexer.inner_indexer
    selector = "pypto_pro" if use_pro else "asc"
    apply_overrides(
        OverrideConfig(
            imports=[
                f"torchtitan_npu.override.deepseek_v4.lightning_indexer.{selector}",
                "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
            ]
        ),
        config,
    )
    config.layers[0].attention.indexer.inner_indexer = indexer
    with torch.device("meta"):
        return config.build()


def _entry(model):
    metadata = model._get_attention_metadata(torch.arange(16).repeat(2))
    return metadata[model.layers["1"].attention.attention_metadata_key]


def test_bf16_metadata_has_no_compact_frame_state():
    metadata = _entry(_debug_model(use_pro=False))
    assert type(metadata) is AscV4Metadata
    assert all("compact" not in field.name and "version" not in field.name for field in fields(metadata))
    assert metadata.aicpu_metadata.lightning_indexer.fwd is not None


def test_inference_tensors_are_supported_by_the_ordinary_backend():
    ordinary = _debug_model(use_pro=False)
    pro = _debug_model(use_pro=True)
    with torch.inference_mode():
        assert type(_entry(ordinary)) is AscV4Metadata
        with pytest.raises(ValueError, match="tracked versions"):
            _entry(pro)


def test_pro_checks_each_geometry_once_before_tracing(monkeypatch):
    model = _debug_model(use_pro=True)
    checks = []
    validate = pypto_pro._validate_compact_frames

    def count_validation(metadata):
        checks.append(metadata)
        return validate(metadata)

    monkeypatch.setattr(pypto_pro, "_validate_compact_frames", count_validation)
    metadata = _entry(model)
    assert len(checks) == 1
    selector = model.layers["1"].attention.indexer.inner_indexer

    def forward_and_backward(q, k, w, entry):
        indices, scores = selector(q, k, w, attention_metadata=entry)
        return indices, torch.autograd.grad(scores, (q, k, w), torch.ones_like(scores))

    q = torch.zeros(32, 64, 128, dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(8, 128, dtype=torch.bfloat16, requires_grad=True)
    w = torch.zeros(32, 64, requires_grad=True)
    graph = make_fx(forward_and_backward, tracing_mode="fake")(q, k, w, metadata)
    operators = {node.target for node in graph.graph.nodes if node.op == "call_function"}
    assert torch.ops.torchtitan_npu.pypto_pro_lightning_indexer_indices.default in operators
    assert torch.ops.torchtitan_npu.pypto_pro_sparse_lightning_indexer_kl_loss_grad.default in operators
    assert len(checks) == 1
    _entry(model)
    assert len(checks) == 2


def test_pro_refuses_noncompact_or_changed_frames_before_launch():
    model = _debug_model(use_pro=True)
    metadata = _entry(model)
    key = metadata.cmp_k
    short = replace(key, seqused=key.seqused - 1)
    unverified = replace(_entry(_debug_model(use_pro=False)), cmp_k=short)
    with pytest.raises(ValueError, match="compact"):
        pypto_pro._validate_compact_frames(unverified)
    with pytest.raises(ValueError, match="replac"):
        pypto_pro._pro_options(replace(metadata, cmp_k=short), metadata.ratio)
    options = pypto_pro._pro_options(metadata, metadata.ratio)
    key.seqused.sub_(1)
    with pytest.raises(ValueError, match="changed in place"):
        _check_compact_guard(options["guard_tensors"], options["guard_versions"])


@pytest.mark.parametrize(("a5", "rank"), [(False, 0), (True, 0), (True, 1)])
def test_cp_rechecks_the_local_operands_only_for_pro(monkeypatch, a5, rank):
    from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc
    from torchtitan_npu.patches.torchtitan.models.common import attention_metadata as native

    monkeypatch.setattr(ascendc, "_IS_A5", a5)
    mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: rank)
    monkeypatch.setattr(
        native,
        "cp_shard_context",
        lambda: SimpleNamespace(cp_mesh=mesh, positions=torch.arange(16).repeat(2), load_balancer_type="doc_chunk"),
    )
    cut = ascendc.AscV4ContextParallelSparseAttention.prepare_cp_metadata
    ordinary = cut(_entry(_debug_model(use_pro=False)), permutation=None)
    assert type(ordinary) is AscV4Metadata
    torch.testing.assert_close(ordinary.q.seqused, torch.tensor([8, 8], dtype=torch.int32))
    source = _entry(_debug_model(use_pro=True))
    if a5 and rank == 0:
        # A5 leaves a short valid prefix inside each whole-document key frame.
        with pytest.raises(ValueError, match="compact"):
            cut(source, permutation=None)
    else:
        local = cut(source, permutation=None)
        options = pypto_pro._pro_options(local, local.ratio)
        assert options["cu_seqlens_q"] is local.q.cu_seqlens
        assert options["cu_seqlens_q"] is not source.q.cu_seqlens
        _check_compact_guard(options["guard_tensors"], options["guard_versions"])


def test_pro_rejects_forward_only_geometry_before_training():
    metadata = _entry(_debug_model(use_pro=False))
    records = metadata.aicpu_metadata
    records = replace(records, geometry=replace(records.geometry, index_num_heads=8))
    with pytest.raises(ValueError, match=r"64 .*128"):
        pypto_pro._validate_compact_frames(replace(metadata, aicpu_metadata=records))
