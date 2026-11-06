# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contracts of ``patches.torchtitan.models.common.decoder``: #4897's map build and per-layer
dispatch, #4116's ``preprocess_inputs`` and #4617's ``_cp_shard``. The seam's own defaults
(``attention_metadata.py``) live in ``test_attention_metadata``.
"""

from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import context_parallel
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.attention import BaseAttention, VarlenInnerAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import Module

import torchtitan_npu  # noqa: F401  imported for its patch chain: the seam under test
from torchtitan_npu.patches.torchtitan.distributed import context_parallel as cp_patch
from torchtitan_npu.patches.torchtitan.models.common import decoder as decoder_patch
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import (
    cp_shard_context,
)

_TOKENS = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
# Positions [0, 1, 2, 0, 1] pack two documents: [0, 3) and [3, 5).
_POSITIONS = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
_LEGACY_MASKS = object()
_PARALLELISM = ParallelismConfig()
_NO_CP = SimpleNamespace(cp_enabled=False)
_CP_MESH = object()
_CP = SimpleNamespace(cp_enabled=True, get_mesh=lambda name: _CP_MESH)
# One kernel geometry of one backend class (what a DSV4 stack varies on).
_Geometry = namedtuple("_Geometry", "ratio")
_SEQ_LEN, _SHARD_LEN = 8, 4
# What the headtail balancer rearranges a length-8 row over 2 ranks with.
_HEADTAIL_PERMUTATION = [[0, 1, 6, 7, 2, 3, 4, 5]]
_MESH = SimpleNamespace(size=lambda: 2, device_type="cpu")


class _Backend(Module):
    """Inner-attention double: it builds the record its config carries (``None`` = no metadata)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        record: object | None = None
        builds: list | None = None
        geometry: object | None = None

        @property
        def attention_metadata_key(self):
            return type(self)._owner if self.geometry is None else self.geometry

        def build_attention_metadata(self, positions, **kwargs):
            if self.builds is not None:  # the test-owned call recorder
                self.builds.append(self.attention_metadata_key)
            return self.record

    def __init__(self, config) -> None:
        super().__init__()


class _Layer(BaseAttention):
    """Layer double: it dispatches by its inner attention's key and keeps what it is handed."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        attention = property(lambda self: self)  # ``Decoder.Config.first_attention`` reads this

    attention = property(lambda self: self)  # the layer's dispatch slot is its own attention

    def __init__(self, config) -> None:
        super().__init__()
        self.inner_attention = config.inner_attention.build()
        self.received: list = []

    def forward(self, x, attention_metadata=None, positions=None, **kwargs):
        self.received.append(attention_metadata)
        return x


class _CPMetadata:
    """Metadata double that owns the transform (upstream #4617's hook, on the rebuilt object)."""

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        _CPBackend.events.append("prepare")
        _CPBackend.calls.append((attention_metadata, permutation, cp_shard_context()))
        return "rank-local"


class _RefusingMetadata:
    """A CP metadata whose transform refuses this rank's copy (the V3.2 reference core's shape)."""

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        del attention_metadata, permutation
        _RefusingBackend.events.append("prepare")
        raise RuntimeError("this backend cannot be sharded for CP")


class _CPBackend(Module):
    """Inner-attention double whose built metadata is the one that shards itself."""

    calls: list = []
    events: list = []

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        record: object | None = None

        def build_attention_metadata(self, positions, **kwargs):
            del positions, kwargs
            return self.record

    def __init__(self, config) -> None:
        super().__init__()


class _RefusingBackend(_CPBackend):
    """Same, for the refusal: the metadata it builds is the one that raises."""

    events: list = []


class _LegacyBackend(_CPBackend):
    """Doubles that still host the transform on their class, as upstream does: inert here."""

    events: list = []

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        _LegacyBackend.events.append("prepare")
        return "rank-local"


def _layer(record=None, builds=None, geometry=None) -> _Layer.Config:
    return _Layer.Config(n_heads=1, inner_attention=_Backend.Config(record=record, builds=builds, geometry=geometry))


def _model(layers) -> Decoder:
    return Decoder(
        Decoder.Config(
            max_context_length=8,
            dim=8,
            vocab_size=16,
            layers=layers,
            tok_embeddings=Embedding.Config(num_embeddings=16, embedding_dim=8),
            norm=RMSNorm.Config(normalized_shape=8),
            lm_head=Linear.Config(in_features=8, out_features=16),
        )
    )


def _preprocess(model, parallelism_context=_NO_CP, parallelism=_PARALLELISM):
    return model.preprocess_inputs(
        {"input": _TOKENS, "labels": _TOKENS, "positions": _POSITIONS},
        parallelism_context=parallelism_context,
        parallelism=parallelism,
    )


@pytest.fixture(autouse=True)
def _isolate_spmd_annotation(monkeypatch):
    # The metadata routing tests isolate annotation, which has its own consumer check below.
    monkeypatch.setattr(decoder_patch, "annotate_input_spmd_types", lambda context, batch, layouts: batch)


def _shard(monkeypatch, events: list, entries=None, positions=None, load_balancer_type="headtail", shard=None):
    """Run metadata routing with native load balancing and an isolated shard boundary."""
    batch = {
        "input": torch.zeros(_SEQ_LEN, dtype=torch.int64),
        "labels": torch.zeros(_SEQ_LEN, dtype=torch.int64),
        "positions": torch.arange(_SEQ_LEN) if positions is None else positions,
    }
    if entries is not None:
        batch["attention_metadata"] = entries

    def fake_shard(input_dict, *, input_shardings, permutation):
        del input_shardings, permutation
        events.append("shard")
        return {
            name: value[:_SHARD_LEN] if name in {"input", "labels", "positions"} else value
            for name, value in input_dict.items()
        }

    monkeypatch.setattr(context_parallel, "shard_tensors", fake_shard if shard is None else shard)
    monkeypatch.setattr(context_parallel, "spmd_mesh_group", lambda axis: SimpleNamespace(size=lambda: 2))
    monkeypatch.setattr(context_parallel, "current_spmd_mesh", lambda: _MESH)
    monkeypatch.setattr(cp_patch, "spmd_mesh_group", lambda axis: SimpleNamespace(size=lambda: 2))
    balancer = context_parallel.HeadTailCPLoadBalancer.Config() if load_balancer_type == "headtail" else None
    output = Decoder._cp_shard(
        SimpleNamespace(),
        batch,
        input_shardings=decoder_input_sharding(),
        parallelism_context=SimpleNamespace(get_mesh=lambda name: _MESH),
        parallelism=ParallelismConfig(context_parallel_load_balancer=balancer),
    )
    return output.pop("input"), output.pop("labels"), output


def test_forward_dispatches_per_layer_and_broadcasts_without_a_map():
    """One microbatch builds one entry per key and each layer reads its own; no map broadcasts one mask."""
    builds: list = []
    r1, r4 = object(), object()
    model = _model(
        [
            _layer(r1, builds, _Geometry(1)),
            _layer(r1, builds, _Geometry(1)),  # the same geometry, not the same config instance
            _layer(r4, builds, _Geometry(4)),
            _layer(None, builds, _Geometry(8)),  # a builder that contributes no entry
        ]
    )
    attention_metadata = model._get_attention_metadata(_POSITIONS)

    assert builds == [_Geometry(1), _Geometry(4), _Geometry(8)]  # one build per geometry, not per layer
    assert attention_metadata == {_Geometry(1): r1, _Geometry(4): r4}
    model(_TOKENS, _POSITIONS, attention_metadata=attention_metadata)
    handed = [layer.received[-1] for layer in model.layers.values()]

    assert handed[0] is handed[1] is r1  # the shared geometry reads one record
    assert handed[2] is r4 and handed[3] is None  # the keyless geometry reads nothing
    model(_TOKENS, _POSITIONS, _LEGACY_MASKS)  # with no map, the pinned single mask is broadcast
    assert [layer.received[-1] for layer in model.layers.values()] == [_LEGACY_MASKS] * 4


def test_preprocess_inputs_hands_metadata_masks_or_neither():
    """A metadata builder wins, a Flex/Varlen stack falls back to one mask, a maskless one gets neither."""
    record = object()
    _, _, extra = _preprocess(_model([_layer(record, geometry=_Geometry(4))]))
    assert extra["attention_metadata"] == {_Geometry(4): record} and "attention_masks" not in extra

    _, _, extra = _preprocess(_model([_Layer.Config(n_heads=1, inner_attention=VarlenInnerAttention.Config())]))
    assert extra["attention_masks"].cu_seq_q.tolist() == [0, 3, 5] and "attention_metadata" not in extra

    _, _, extra = _preprocess(_model([_layer(geometry=_Geometry(1))]))
    assert "attention_masks" not in extra and "attention_metadata" not in extra


def test_preprocess_inputs_passes_named_layouts_to_spmd_annotator(monkeypatch):
    seen = []

    def annotate(context, batch, layouts):
        seen.append((context, batch["input"], layouts))
        return {**batch, "input": "annotated"}

    monkeypatch.setattr(decoder_patch, "annotate_input_spmd_types", annotate)
    output = _preprocess(_model([_layer()]))

    assert output[0] == "annotated"
    assert seen[0][0] is _NO_CP and seen[0][1] is _TOKENS
    assert seen[0][2] == decoder_input_sharding()


def test_preprocess_inputs_delegates_metadata_and_layouts_to_cp_shard():
    seen = []
    record = object()
    model = _model([_layer(record, geometry=_Geometry(4))])

    def delegate(batch, **kwargs):
        seen.append((kwargs, batch["attention_metadata"]))
        return batch

    model._cp_shard = delegate
    _, _, extra = _preprocess(model, parallelism_context=_CP)

    assert seen == [
        (
            {"input_shardings": decoder_input_sharding(), "parallelism_context": _CP, "parallelism": _PARALLELISM},
            {_Geometry(4): record},
        )
    ]
    assert extra["attention_metadata"] == {_Geometry(4): record}


def test_cp_shard_owns_the_rank_local_map(monkeypatch):
    """Metadata is prepared from the global row before the native typed tensor shard."""
    events: list = []
    _CPBackend.calls, _CPBackend.events = [], events
    record, untouched = _CPMetadata(), object()
    positions = torch.zeros(_SEQ_LEN, dtype=torch.int64)
    inputs, labels, out = _shard(monkeypatch, events, {_CPBackend: record, "untouched": untouched}, positions)

    assert events == ["prepare", "shard"]  # metadata is prepared before native typed sharding
    assert out["attention_metadata"][_CPBackend] == "rank-local"
    assert out["attention_metadata"]["untouched"] is untouched  # a key without the hook passes through
    assert out["positions"].shape[-1] == inputs.shape[-1] == _SHARD_LEN and labels.shape[-1] == _SHARD_LEN
    # The transform is the metadata's own static method, handed the permutation alone
    # (upstream #4617's hook, hosted by the object being rebuilt).
    handed, permutation, context = _CPBackend.calls[0]
    assert handed is record
    assert permutation.tolist() == _HEADTAIL_PERMUTATION
    # The pinned stand-in for upstream's ambient parallelism context carries what the
    # permutation alone cannot: the mesh, the balancer and the *pre-shard* row.
    assert context.cp_mesh is _MESH and context.load_balancer_type == "headtail"
    assert context.positions is positions
    # Only the headtail balancer shards by a permutation the seam can re-derive.
    _CPBackend.calls, _CPBackend.events = [], events
    _shard(monkeypatch, events, {_CPBackend: record}, positions, load_balancer_type=None)
    assert _CPBackend.calls[0][1] is None and _CPBackend.calls[0][2].load_balancer_type is None

    plain_events: list = []
    _, _, plain_out = _shard(monkeypatch, plain_events)  # no map: the pinned shard runs alone
    assert plain_events == ["shard"] and "attention_metadata" not in plain_out


def test_an_unnamed_layout_is_the_one_the_metadata_declares(monkeypatch):
    """A stack declares the layout its frames describe; an unnamed run is sharded by it, and by the row."""
    seen: dict = {}

    def shard(batch, *, input_shardings, permutation):
        seen["layouts"], seen["positions"], seen["permutation"] = input_shardings, batch["positions"], permutation
        return batch

    class _DeclaredMetadata:
        cp_layout = "doc_chunk"

    positions = torch.arange(_SEQ_LEN)
    _shard(monkeypatch, [], {_CPBackend: _DeclaredMetadata()}, positions, load_balancer_type=None, shard=shard)
    assert seen["layouts"] == decoder_input_sharding()
    assert seen["positions"] is positions
    assert seen["permutation"].tolist() == [list(range(_SEQ_LEN))]

    # An incompatible explicit balancer cannot silently misinterpret document-local metadata.
    with pytest.raises(ValueError, match="Document-chunk"):
        _shard(
            monkeypatch, [], {_CPBackend: _DeclaredMetadata()}, positions, load_balancer_type="headtail", shard=shard
        )


def test_cp_shard_propagates_a_backend_that_refuses(monkeypatch):
    """A backend that cannot prepare rank-local metadata fails the step."""
    events: list = []
    _RefusingBackend.events = events
    with pytest.raises(RuntimeError, match="cannot be sharded for CP"):
        _shard(monkeypatch, events, {_RefusingBackend: _RefusingMetadata()})

    assert events == ["prepare"]


def test_a_hook_on_the_backend_class_is_not_consulted(monkeypatch):
    """The seam asks the metadata, never the key: a class-hosted hook cannot fire."""
    events: list = []
    _LegacyBackend.events = []

    class _PlainMetadata:
        pass

    _, _, out = _shard(monkeypatch, events, {_LegacyBackend: _PlainMetadata()})

    assert events == ["shard"]  # no "prepare": the class' hook was never reached
    assert _LegacyBackend.events == []
    assert isinstance(out["attention_metadata"][_LegacyBackend], _PlainMetadata)
