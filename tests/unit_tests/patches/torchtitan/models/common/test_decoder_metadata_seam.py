# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contracts of ``patches.torchtitan.models.common.decoder``: #4897's map build and per-layer
dispatch, #4116's ``preprocess_inputs`` and #4617's ``_cp_shard``, plus the trainer step that consumes
them; the seam's own defaults (``attention_metadata.py``) live in ``test_attention_metadata``.
"""

from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torchtitan.config import ParallelismConfig
from torchtitan.models.common.attention import BaseAttention, VarlenAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import Module

import torchtitan_npu  # noqa: F401  imported for its patch chain: the seam under test
from torchtitan_npu.patches.torchtitan.models.common import decoder as decoder_patch
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import (
    cp_shard_context,
)
from torchtitan_npu.patches.torchtitan.trainer import patched_post_dataloading_process

_TOKENS = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
# Positions [[0, 1, 2, 0, 1]] pack two documents: [0, 3) and [3, 5).
_POSITIONS = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.int64)
_LEGACY_MASKS = object()
# ``partial_dtensor`` takes neither SPMD tail branch in ``preprocess_inputs``.
_NO_SPMD = ParallelismConfig(spmd_backend="partial_dtensor")
_SPMD_TYPES, _FULL_DTENSOR = (ParallelismConfig(spmd_backend=name) for name in ("spmd_types", "full_dtensor"))
_NO_CP = SimpleNamespace(cp_enabled=False, get_mesh=lambda name: None)
# ``preprocess_inputs`` only forwards the CP mesh, so a sentinel stands in for it.
_CP_MESH = object()
_CP = SimpleNamespace(cp_enabled=True, get_mesh=lambda name: _CP_MESH)
_LB, _PTRR = _NO_SPMD.context_parallel_load_balancer, _NO_SPMD.context_parallel_ptrr_mask_key
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
            dim=8,
            vocab_size=16,
            layers=layers,
            tok_embeddings=Embedding.Config(num_embeddings=16, embedding_dim=8),
            norm=RMSNorm.Config(normalized_shape=8),
            lm_head=Linear.Config(in_features=8, out_features=16),
        )
    )


def _preprocess(model, parallel_dims=_NO_CP, parallelism=_NO_SPMD):
    return model.preprocess_inputs(
        {"input": _TOKENS, "positions": _POSITIONS},
        _TOKENS,
        parallel_dims=parallel_dims,
        device=_TOKENS.device,
        parallelism=parallelism,
    )


def _shard(monkeypatch, events: list, entries=None, positions=None, load_balancer_type="headtail"):
    """Run the patched ``_cp_shard`` on a bare receiver, with the pinned shard stubbed out."""
    extra_kwargs = {"positions": torch.zeros(1, _SEQ_LEN, dtype=torch.int64) if positions is None else positions}
    if entries is not None:
        extra_kwargs["attention_metadata"] = entries

    def fake_prepare(inputs, labels, extra_kwargs, *args):
        events.append("shard")
        extra_kwargs["positions"] = extra_kwargs["positions"][:, :_SHARD_LEN]
        return inputs[:, :_SHARD_LEN], labels, extra_kwargs

    monkeypatch.setattr(decoder_patch, "prepare_context_parallel_input", fake_prepare)
    tokens = torch.zeros(1, _SEQ_LEN, dtype=torch.int64)
    return Decoder._cp_shard(
        SimpleNamespace(),
        tokens,
        tokens,
        extra_kwargs,
        cp_mesh=_MESH,
        device=torch.device("cpu"),
        load_balancer_type=load_balancer_type,
    )


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

    _, _, extra = _preprocess(_model([_Layer.Config(n_heads=1, inner_attention=VarlenAttention.Config())]))
    assert extra["attention_masks"].cu_seq_q.tolist() == [0, 3, 5] and "attention_metadata" not in extra

    _, _, extra = _preprocess(_model([_layer(geometry=_Geometry(1))]))
    assert "attention_masks" not in extra and "attention_metadata" not in extra


def test_input_annotation_is_spmd_types_only_and_full_dtensor_is_rejected(monkeypatch):
    """Upstream's input half: ``spmd_types`` annotates, and the removed ``full_dtensor`` fails fast."""
    monkeypatch.setattr(decoder_patch, "annotate_input_spmd_types", lambda *args: ("annotated", *args[2:]))
    assert _preprocess(_model([_layer()]), parallelism=_SPMD_TYPES)[0] == "annotated"  # the annotated path
    assert _preprocess(_model([_layer()]))[0] is _TOKENS  # partial_dtensor: neither branch

    inputs = _TOKENS.clone()
    input_dict = {"input": inputs, "positions": _POSITIONS}
    # A bare receiver survives no step at all, so the ValueError is the proof that the guard runs first.
    with pytest.raises(ValueError, match="full_dtensor"):
        decoder_patch.preprocess_inputs(
            SimpleNamespace(), input_dict, inputs, parallel_dims=_NO_CP, device=inputs.device, parallelism=_FULL_DTENSOR
        )
    assert list(input_dict) == ["input", "positions"] and input_dict["input"] is inputs  # nothing was wrapped


def test_the_step_counts_the_local_labels_the_seam_hands_back(monkeypatch):
    """With no full_dtensor wrapping, the trainer's counter adds this rank's CP shard, not the batch."""

    def shard(inputs, labels, extra_kwargs, *args):  # the pinned shard, stubbed: the first rows stay here
        return inputs[:, :_SHARD_LEN], labels[:, :_SHARD_LEN], extra_kwargs

    monkeypatch.setattr(decoder_patch, "prepare_context_parallel_input", shard)
    trainer = SimpleNamespace(
        model_parts=[_model([_layer()])],
        ntokens_seen=16,
        device=_TOKENS.device,
        parallel_dims=_CP,
        config=SimpleNamespace(parallelism=_NO_SPMD),
    )

    patched_post_dataloading_process(trainer, {"input": _TOKENS, "positions": _POSITIONS}, _TOKENS)

    assert trainer.ntokens_seen == 16 + _SHARD_LEN  # this rank's 4 rows, not the 5 the loader delivered


def test_preprocess_inputs_delegates_the_step_instead_of_building_it():
    """A model-level hook owns the whole step; otherwise the CP step is the model's ``_cp_shard``."""
    seen: list = []
    record = object()
    model = _model([_layer(record, geometry=_Geometry(4))])

    def delegate(inputs, labels, extra_kwargs, **kwargs):
        seen.append((kwargs, extra_kwargs.get("attention_metadata")))
        return inputs, labels, extra_kwargs

    model.build_attention_masks = delegate
    _preprocess(model, parallel_dims=_CP)
    assert seen == [({"cp_mesh": _CP_MESH, "load_balancer_type": _LB}, None)]  # the hook owns the metadata step

    del model.build_attention_masks
    model._cp_shard = delegate
    _, _, extra = _preprocess(model, parallel_dims=_CP)
    cp_context = {"cp_mesh": _CP_MESH, "device": _TOKENS.device, "load_balancer_type": _LB, "ptrr_mask_key": _PTRR}
    assert seen[1] == (cp_context, {_Geometry(4): record})  # the seam's own map is what the CP step receives
    assert extra["attention_metadata"] == {_Geometry(4): record}  # ... and what comes back out


def test_cp_shard_owns_the_rank_local_map(monkeypatch):
    """The map is prepared after the shard, from the row the shard started from; no map is left alone."""
    events: list = []
    _CPBackend.calls, _CPBackend.events = [], events
    record, untouched = _CPMetadata(), object()
    positions = torch.zeros(1, _SEQ_LEN, dtype=torch.int64)
    inputs, labels, out = _shard(monkeypatch, events, {_CPBackend: record, "untouched": untouched}, positions)

    assert events == ["shard", "prepare"]  # this rank's rows exist first
    assert out["attention_metadata"][_CPBackend] == "rank-local"
    assert out["attention_metadata"]["untouched"] is untouched  # a key without the hook passes through
    assert out["positions"].shape[-1] == inputs.shape[-1] == _SHARD_LEN and labels.shape[-1] == _SEQ_LEN
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


def test_cp_shard_propagates_a_backend_that_refuses(monkeypatch):
    """A backend that cannot prepare rank-local metadata fails the step."""
    events: list = []
    _RefusingBackend.events = events
    with pytest.raises(RuntimeError, match="cannot be sharded for CP"):
        _shard(monkeypatch, events, {_RefusingBackend: _RefusingMetadata()})

    assert events == ["shard", "prepare"]


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
