# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss

from tests.unit_tests.models.mtp_test_utils import (
    PACKED_POSITIONS,
    SEQ_LEN,
    VOCAB_SIZE,
    assert_packed_mtp_training,
    build_cpu_model,
    preprocess,
)
from torchtitan_npu.models.common.linear import SharedLinear
from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel
from torchtitan_npu.models.deepseek_v4.mtp import MTPChunkedLossWrapper
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import (
    DeepSeekV4StateDictAdapter,
)

pytestmark = pytest.mark.usefixtures("single_rank_spmd")


def _build_model_config(
    *,
    compress_ratios=(1, 1, 1),
    rope_head_dim=4,
    original_seq_len=SEQ_LEN,
    num_mtp_layers=1,
):
    return _make_v4_config(
        dim=32,
        n_layers=3,
        vocab_size=VOCAB_SIZE,
        n_heads=4,
        head_dim=8,
        rope_head_dim=rope_head_dim,
        q_lora_rank=16,
        o_lora_rank=8,
        n_groups=2,
        compress_ratios=compress_ratios,
        window_size=SEQ_LEN,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        moe_inter_dim=16,
        num_experts=2,
        num_shared_experts=1,
        top_k=1,
        n_hash_layers=0,
        route_norm=False,
        route_scale=1.0,
        load_balance_coeff=1e-3,
        hc_mult=2,
        sinkhorn_iters=2,
        hc_eps=1e-6,
        max_seq_len=SEQ_LEN,
        compress_rope_theta=10000.0,
        original_seq_len=original_seq_len,
        num_mtp_layers=num_mtp_layers,
    )


def test_deepseek_v4_mtp_model_supports_packed_sequences():
    assert_packed_mtp_training(build_cpu_model(_build_model_config()))


def test_deepseek_v4_mtp_metadata_includes_uncompressed_plan():
    model = build_cpu_model(
        _build_model_config(
            compress_ratios=(128, 128, 4),
            rope_head_dim=8,
            original_seq_len=65536,
        )
    )
    tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
    labels = (tokens + 1) % VOCAB_SIZE
    positions = torch.arange(SEQ_LEN)

    _, _, extra_kwargs = preprocess(model, tokens, labels, positions)

    # One entry per geometry: the stack's ratios -- the MTP depths' uncompressed one included --
    # are one map key each, and each entry carries only its own ratio's plan.
    entries = extra_kwargs["attention_metadata"]
    assert {ratio for _, ratio in entries} == {1, 4, 128}
    assert all(entry.ratio == ratio for (_, ratio), entry in entries.items())
    output = model(tokens, **extra_kwargs)
    assert isinstance(output, list) and len(output) == 2


def test_deepseek_v4_mtp_model_supports_chunked_loss():
    model = build_cpu_model(_build_model_config())
    tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
    positions = PACKED_POSITIONS.clone()
    labels = (tokens + 1) % VOCAB_SIZE
    tokens, labels, extra_kwargs = preprocess(model, tokens, labels, positions)
    assert extra_kwargs["mtp_batch"].labels.shape == (SEQ_LEN, 1)
    object.__setattr__(model, "_skip_lm_head", True)
    output = model(tokens, **extra_kwargs)

    assert isinstance(output, list) and len(output) == 2
    assert output[0].shape[0] == SEQ_LEN

    loss_fn = MTPChunkedLossWrapper(
        MTPChunkedLossWrapper.Config(
            num_chunks=2,
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE),
        )
    )
    loss_fn.set_lm_head(model.lm_head)
    loss, _ = loss_fn(
        output,
        labels,
        torch.tensor(labels.numel()),
        positions=positions,
    )
    assert torch.isfinite(loss).item()

    loss.backward()
    assert any(parameter.grad is not None for parameter in model.mtp_layers.parameters())


@pytest.fixture
def single_threaded_cpu():
    # Keep scatter/reduction scheduling fixed for bitwise gradient comparison.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_deepseek_v4_mtp_loss_flag_reaches_model():
    trainer_config = deepseek_v4_debugmodel(num_mtp_layers=1)
    assert not trainer_config.loss.mtp_isolate_gradients
    trainer_config.loss.mtp_isolate_gradients = True
    model_config = trainer_config.model
    model_config.update_from_config(config=trainer_config)
    with torch.device("meta"):
        model = model_config.build()

    assert model.mtp_isolate_gradients
    assert isinstance(model.lm_head, SharedLinear)


@pytest.mark.parametrize("missing_head", [False, True], ids=["ordinary_linear", "missing_head"])
def test_deepseek_v4_mtp_isolation_rejects_incompatible_loss_head(missing_head):
    with torch.random.fork_rng():
        loss_fn = MTPChunkedLossWrapper(
            MTPChunkedLossWrapper.Config(
                num_chunks=2,
                loss_fn=CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE),
                mtp_isolate_gradients=True,
            )
        )
        head = torch.nn.Linear(4, VOCAB_SIZE, bias=False)
        if not missing_head:
            loss_fn.set_lm_head(head)
        hidden = [torch.ones(4, 4, requires_grad=True) for _ in range(2)]
        labels = torch.zeros(4, dtype=torch.long)

        with pytest.raises(ValueError, match="mtp_isolate_gradients requires a SharedLinear LM head"):
            loss_fn(hidden, labels, positions=torch.arange(4))

        assert head.weight.grad is None
        assert all(item.grad is None for item in hidden)


def test_deepseek_v4_mtp_isolation_rejects_replaced_model_head():
    with torch.random.fork_rng():
        config = _build_model_config()
        config.mtp_isolate_gradients = True
        model = build_cpu_model(config)
        model.lm_head = torch.nn.Linear(32, VOCAB_SIZE, bias=False)
        tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
        tokens, _, extra_kwargs = preprocess(model, tokens, (tokens + 1) % VOCAB_SIZE, PACKED_POSITIONS.clone())

        with pytest.raises(ValueError, match="mtp_isolate_gradients requires a SharedLinear LM head"):
            model(tokens, **extra_kwargs)


@pytest.mark.parametrize("num_mtp_layers", [1, 2], ids=["depth1", "depth2"])
def test_deepseek_v4_mtp_detaches_all_shared_parameters(num_mtp_layers):
    with torch.random.fork_rng():
        config = _build_model_config(num_mtp_layers=num_mtp_layers)
        config.mtp_isolate_gradients = True
        model = build_cpu_model(config)
        tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
        labels = (tokens + 1) % VOCAB_SIZE
        tokens, _, extra_kwargs = preprocess(model, tokens, labels, PACKED_POSITIONS.clone())

        output = model(tokens, **extra_kwargs)
        sum(item.square().sum() for item in output[1:]).backward()

        for name, parameter in model.named_parameters():
            if not name.startswith("mtp_layers."):
                assert parameter.grad is None or not parameter.grad.count_nonzero(), name
        for layer in model.mtp_layers:
            for projection in (layer.e_proj, layer.h_proj):
                assert projection.weight.grad is not None
                assert projection.weight.grad.count_nonzero()


@pytest.mark.parametrize("chunked", [False, True], ids=["logits", "chunked"])
def test_deepseek_v4_mtp_isolation_preserves_main_gradients(chunked, single_threaded_cpu):
    with torch.random.fork_rng():
        config = _build_model_config()
        reference = build_cpu_model(config)
        isolated_config = deepcopy(config)
        isolated_config.mtp_isolate_gradients = True
        model = build_cpu_model(isolated_config)
        model.load_state_dict(reference.state_dict(), strict=True)
        tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
        labels = (tokens + 1) % VOCAB_SIZE
        positions = PACKED_POSITIONS.clone()
        tokens, labels, extra_kwargs = preprocess(model, tokens, labels, positions)
        if chunked:
            object.__setattr__(model, "_skip_lm_head", True)
            object.__setattr__(reference, "_skip_lm_head", True)

        output = model(tokens, **extra_kwargs)
        expected = reference(tokens, **extra_kwargs)
        for actual, original in zip(output, expected, strict=True):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        loss_config = CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE)
        if chunked:
            main_loss_fn = ChunkedLossWrapper(ChunkedLossWrapper.Config(num_chunks=2, loss_fn=loss_config))
            main_loss_fn.set_lm_head(reference.lm_head)
            loss_fn = MTPChunkedLossWrapper(
                MTPChunkedLossWrapper.Config(num_chunks=2, loss_fn=loss_config, mtp_isolate_gradients=True)
            )
            loss_fn.set_lm_head(model.lm_head)
            main_loss, _ = main_loss_fn(expected[0], labels, torch.tensor(labels.numel()))
            loss, _ = loss_fn(output, labels, torch.tensor(labels.numel()), positions=positions)
        else:
            main_loss, _ = CrossEntropyLoss(loss_config)(expected[0], labels)
            loss = CrossEntropyLoss(loss_config)(output[0], labels)[0] + output[1].square().mean()
        main_loss.backward()
        loss.backward()
        assert model.mtp_layers[0].e_proj.weight.grad.count_nonzero()

        for name, parameter in model.named_parameters():
            if name.startswith("mtp_layers."):
                continue
            expected_grad = dict(reference.named_parameters())[name].grad
            assert (parameter.grad is None) == (expected_grad is None), name
            if expected_grad is not None:
                torch.testing.assert_close(
                    parameter.grad, expected_grad, rtol=0, atol=0, msg=lambda message: f"{name}: {message}"
                )


def test_deepseek_v4_mtp_state_dict_round_trip():
    model_config = _build_model_config()
    model = build_cpu_model(model_config)
    local_state_dict = {
        key: value.detach().clone() for key, value in model.state_dict().items() if key.startswith("mtp_layers.0.")
    }
    required_local_prefixes = (
        "mtp_layers.0.attention.",
        "mtp_layers.0.hc_attn_pre.",
        "mtp_layers.0.hc_ffn_pre.",
        "mtp_layers.0.hc_head.",
        "mtp_layers.0.moe.router.",
        "mtp_layers.0.moe.shared_experts.",
        "mtp_layers.0.moe.routed_experts.",
    )
    assert all(any(key.startswith(prefix) for key in local_state_dict) for prefix in required_local_prefixes)

    adapter = DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)
    hf_state_dict = adapter.to_hf(local_state_dict)
    required_hf_keys = {
        "mtp.0.e_proj.weight",
        "mtp.0.h_proj.weight",
        "mtp.0.enorm.weight",
        "mtp.0.hnorm.weight",
        "mtp.0.norm.weight",
        "mtp.0.hc_head_fn",
        "mtp.0.attn.wq_a.weight",
        "mtp.0.ffn.gate.weight",
        "mtp.0.ffn.shared_experts.w1.weight",
        "mtp.0.ffn.experts.0.w1.weight",
    }
    assert required_hf_keys <= hf_state_dict.keys()
    assert all(key.startswith("mtp.0.") for key in hf_state_dict)

    restored_state_dict = adapter.from_hf(hf_state_dict)
    assert restored_state_dict.keys() == local_state_dict.keys()
    for key, expected in local_state_dict.items():
        torch.testing.assert_close(
            restored_state_dict[key],
            expected,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("method_name", "key"),
    (
        ("from_hf", "mtp.1.e_proj.weight"),
        ("to_hf", "mtp_layers.1.e_proj.weight"),
    ),
)
def test_deepseek_v4_mtp_state_dict_rejects_invalid_depth(method_name, key):
    model_config = _build_model_config()
    adapter = DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)

    with pytest.raises(ValueError, match="not present"):
        getattr(adapter, method_name)({key: torch.ones(1)})


def test_preprocess_inputs_builds_the_mtp_batch_on_the_global_row(monkeypatch):
    """The MTP depths' rolled tokens are built before the CP shard, then cut with it.

    A shift across a shard boundary is not a shift, so the batch must be built
    from the whole row and the same ``cp_shard`` cut applied afterwards.  The
    shard here really slices rank 1's half of a two-rank run, so the test pins
    the alignment, not only the call order.
    """
    from types import SimpleNamespace

    from torchtitan_npu.models.deepseek_v4 import model as model_mod

    model = build_cpu_model(_build_model_config())
    tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
    labels = (tokens + 1) % VOCAB_SIZE
    positions = PACKED_POSITIONS.clone()

    real_prepare = model_mod.prepare_mtp_batch
    built = []

    def record_prepare(inputs, prepared_labels, prepared_positions, num_layers):
        batch = real_prepare(inputs, prepared_labels, prepared_positions, num_layers)
        built.append((inputs, prepared_labels, prepared_positions, batch))
        return batch

    shard_calls = []
    rank, cp_size = 1, 2
    shard_len = SEQ_LEN // cp_size

    def fake_cp_shard(input_dict, *, input_shardings, parallelism_context, parallelism):
        shard_calls.append((dict(input_dict), input_dict["positions"], input_shardings))
        assert parallelism_context.get_mesh("cp") is mesh
        return {key: tensor.narrow(0, rank * shard_len, shard_len) for key, tensor in input_dict.items()}

    monkeypatch.setattr(model_mod, "prepare_mtp_batch", record_prepare)
    monkeypatch.setattr(model, "_cp_shard", fake_cp_shard)

    # Isolate the decoder's metadata build; both calls still exercise the model's named shard boundary.
    from torchtitan.models.common.decoder import Decoder
    from torchtitan.models.common.decoder_sharding import decoder_input_sharding

    from torchtitan_npu.models.deepseek_v4.metadata import build_v4_metadata, doc_chunk_entry

    base_calls = []
    row_entry = build_v4_metadata(positions, 1)
    local_entry = doc_chunk_entry(row_entry, cp_size=cp_size, cp_rank=rank, slab_len=shard_len)

    def fake_base(self, input_dict, *, parallelism_context, parallelism, **kwargs):
        del kwargs
        base_calls.append((input_dict["input"].shape, input_dict["labels"].shape))
        cut = self._cp_shard(
            input_dict,
            input_shardings=decoder_input_sharding(),
            parallelism_context=parallelism_context,
            parallelism=parallelism,
        )
        return (
            cut["input"],
            cut["labels"],
            {"positions": cut["positions"], "attention_metadata": {("debug", 1): local_entry}},
        )

    monkeypatch.setattr(Decoder, "preprocess_inputs", fake_base)
    mesh = SimpleNamespace(size=lambda *_: cp_size, get_local_rank=lambda: rank, device_type="cpu")
    parallelism_context = SimpleNamespace(cp_enabled=True, get_mesh=lambda name: mesh)
    parallelism = SimpleNamespace(context_parallel_load_balancer=None)
    inputs, out_labels, extra_kwargs = model.preprocess_inputs(
        {"input": tokens, "labels": labels, "positions": positions},
        parallelism_context=parallelism_context,
        parallelism=parallelism,
    )

    # The batch was built from the global row, once.
    assert len(built) == 1
    global_inputs, global_labels, global_positions, batch = built[0]
    assert global_inputs.shape == tokens.shape
    assert torch.equal(global_labels, labels)
    assert torch.equal(global_positions, positions)

    # Two cuts, both fed the full row: the base inputs, then the MTP batch.
    assert base_calls == [(tokens.shape, labels.shape)]
    assert len(shard_calls) == 2
    for name, expected in (("input", tokens), ("labels", labels), ("positions", positions)):
        assert torch.equal(shard_calls[0][0][name], expected)
    for index, expected in enumerate(batch):
        assert torch.equal(shard_calls[1][0][f"mtp_{index}"], expected)
    # The row reaches both calls, so the layout cannot be derived from anything else.
    for _, handed_positions, _ in shard_calls:
        assert torch.equal(handed_positions, positions)
    # Every MTP tensor uses the same token layout that the decoder's native shard consumes.
    assert set(shard_calls[1][2]) == {"mtp_0", "mtp_1", "mtp_2", "positions"}
    assert all(layout == decoder_input_sharding()["input"] for layout in shard_calls[1][2].values())

    # What the forward reads is the shard's answer, not the global row.
    slab = slice(rank * shard_len, (rank + 1) * shard_len)
    assert torch.equal(inputs, tokens[slab]) and torch.equal(out_labels, labels[slab])
    assert torch.equal(extra_kwargs["positions"], positions[slab])
    # The mask is the global row's own (a shift across a shard boundary is not a shift): the
    # local copy is exactly its slab, so the row's last token still has no successor.
    local_batch = extra_kwargs["mtp_batch"]
    assert torch.equal(local_batch.input_valid_mask, batch.input_valid_mask[slab])
    assert not bool(local_batch.input_valid_mask[-1].all())
    assert extra_kwargs["attention_metadata"][("debug", 1)] is local_entry
