# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from inspect import signature
from types import MethodType

import pytest
import torch
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.models.deepseek_v4.mtp import MTPBlock
from torchtitan.protocols.module import ModuleList

from tests.unit_tests.models.mtp_test_utils import (
    PACKED_POSITIONS,
    SEQ_LEN,
    VOCAB_SIZE,
    assert_packed_mtp_training,
    build_cpu_model,
    build_v4_model_config,
    preprocess,
)
from torchtitan_npu.models.common.linear import SharedLinear
from torchtitan_npu.models.deepseek_v4.config_registry import (
    deepseek_v4_debugmodel,
    deepseek_v4_flash_43layers_16experts,
)
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4TransformerBlock
from torchtitan_npu.models.deepseek_v4.mtp import MTPChunkedLossWrapper
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import (
    DeepSeekV4StateDictAdapter,
)

pytestmark = pytest.mark.usefixtures("single_rank_spmd")


@pytest.mark.parametrize(
    "recipe", [deepseek_v4_debugmodel, deepseek_v4_flash_43layers_16experts], ids=["debug", "flash43"]
)
@pytest.mark.parametrize("num_mtp_layers", [0, 1, 2])
def test_v4_recipe_builds_upstream_mtp_contract(recipe, num_mtp_layers):
    trainer_config = recipe(num_mtp_layers=num_mtp_layers)
    model_config = trainer_config.model
    if num_mtp_layers:
        assert len(model_config.mtp_layers) == num_mtp_layers
        assert all(type(layer) is MTPBlock.Config for layer in model_config.mtp_layers)
        assert isinstance(trainer_config.loss, MTPChunkedLossWrapper.Config)
    else:
        assert model_config.mtp_layers == []
        assert type(trainer_config.loss) is ChunkedLossWrapper.Config
    with torch.device("meta"):
        model = model_config.build()
    assert isinstance(model.mtp_layers, ModuleList)
    assert len(model.mtp_layers) == num_mtp_layers
    assert all(type(layer) is MTPBlock for layer in model.mtp_layers)
    assert bool([name for name, _ in model.named_parameters() if name.startswith("mtp_layers.")]) == bool(
        num_mtp_layers
    )


def test_upstream_mtp_block_accepts_sequence_parallel_valid_mask():
    assert "mtp_input_valid_mask" in signature(MTPBlock.forward).parameters


def test_deepseek_v4_mtp_model_supports_packed_sequences():
    assert_packed_mtp_training(build_cpu_model(build_v4_model_config()))


def test_deepseek_v4_mtp_metadata_includes_uncompressed_plan():
    model = build_cpu_model(
        build_v4_model_config(
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
    model = build_cpu_model(build_v4_model_config())
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
        config = build_v4_model_config()
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
        config = build_v4_model_config(num_mtp_layers=num_mtp_layers)
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
        config = build_v4_model_config()
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
                    parameter.grad, expected_grad, rtol=0, atol=0, msg=lambda message, name=name: f"{name}: {message}"
                )


def test_deepseek_v4_mtp_state_dict_round_trip():
    model_config = build_v4_model_config()
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
    model_config = build_v4_model_config()
    adapter = DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)

    with pytest.raises(ValueError, match="not present"):
        getattr(adapter, method_name)({key: torch.ones(1)})


@pytest.mark.parametrize("rank", [0, 1])
def test_preprocess_mtp_batch_uses_the_main_input_cp_layout(monkeypatch, rank, single_rank_spmd):
    from types import SimpleNamespace

    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed import context_parallel

    from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata, doc_chunk_entry
    from torchtitan_npu.patches.torchtitan.distributed import context_parallel as cp_patch

    with torch.random.fork_rng():
        model = build_cpu_model(build_v4_model_config(num_mtp_layers=2))
    tokens = torch.arange(SEQ_LEN)
    labels = tokens + 1000
    positions = PACKED_POSITIONS.clone()
    shard_calls = []
    mesh = SimpleNamespace(size=lambda *_: 2, get_local_rank=lambda: rank, device_type="cpu")
    indices = torch.cat(
        [torch.arange(rank * 64, (rank + 1) * 64), torch.arange(128 + rank * 64, 128 + (rank + 1) * 64)]
    )

    def device_shard(batch, *, input_shardings, permutation):
        assert permutation is not None
        permutation = permutation.flatten()
        shard_calls.append((set(input_shardings) & batch.keys(), permutation))
        return {
            key: value.index_select(0, permutation).chunk(mesh.size(), dim=0)[rank] if key in input_shardings else value
            for key, value in batch.items()
        }

    def cpu_metadata(entry, *, permutation):
        del permutation
        return doc_chunk_entry(entry, cp_size=2, cp_rank=rank, slab_len=SEQ_LEN // 2)

    monkeypatch.setattr(DeepSeekV4Metadata, "prepare_cp_metadata", staticmethod(cpu_metadata))
    monkeypatch.setattr(cp_patch, "spmd_mesh_group", lambda name: mesh)
    monkeypatch.setattr(context_parallel, "shard_tensors", device_shard)
    inputs, out_labels, extra = model.preprocess_inputs(
        {"input": tokens, "labels": labels, "positions": positions},
        parallelism_context=SimpleNamespace(
            cp_enabled=True, get_mesh=lambda name: mesh, spmd_dense_mesh=single_rank_spmd.spmd_dense_mesh
        ),
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
    )
    assert [keys for keys, _ in shard_calls] == [
        {"input", "labels", "positions"},
        {"mtp_0", "mtp_1", "mtp_2", "positions"},
    ]
    torch.testing.assert_close(shard_calls[0][1], shard_calls[1][1])
    batch = extra["mtp_batch"]
    torch.testing.assert_close(inputs, tokens[indices])
    torch.testing.assert_close(out_labels, labels[indices])
    torch.testing.assert_close(extra["positions"], positions[indices])
    for depth in (1, 2):
        valid = indices.remainder(128) < 128 - depth
        torch.testing.assert_close(batch.input_tokens[..., depth - 1], torch.where(valid, indices + depth, 0))
        torch.testing.assert_close(batch.input_valid_mask[..., depth - 1], valid)
        torch.testing.assert_close(batch.labels[..., depth - 1], torch.where(valid, indices + depth + 1000, -100))


# Reference computation from dev_0.4.0 commit 029756d, before upstream-block reuse.
# Keep the two original computation paths independent of the inherited forward methods.
def _baseline_v4_block_forward(self, hidden, input_ids, metadata, positions=None, *, padding_mask=None):
    residual = hidden
    hidden, post, comb = self.hc_attn_pre(hidden)
    hidden = self.attention(self.attention_norm(hidden), metadata, positions)
    hidden = self.hc_post(hidden, residual, post, comb)
    residual = hidden
    hidden, post, comb = self.hc_ffn_pre(hidden)
    hidden = self.moe(self.ffn_norm(hidden), input_ids_T=input_ids, padding_mask_T=padding_mask)
    return self.hc_post(hidden, residual, post, comb)


def _baseline_mtp_forward(
    self, embedding, prev_hidden, input_ids, valid_mask, metadata, positions=None, *, padding_mask=None
):
    prev_hidden = prev_hidden * valid_mask.unsqueeze(-1).unsqueeze(-1).to(prev_hidden.dtype)
    hidden = self.e_proj(self.enorm(embedding)).unsqueeze(1) + self.h_proj(self.hnorm(prev_hidden))
    mtp_padding_mask = ~valid_mask if padding_mask is None else ~valid_mask | padding_mask
    next_hidden = _baseline_v4_block_forward(
        self, hidden, input_ids, metadata, positions, padding_mask=mtp_padding_mask
    )
    prediction = self.mtp_norm(self.hc_head(next_hidden))
    return next_hidden, prediction


@pytest.mark.parametrize(
    ("chunked", "isolate", "prepared", "n_hash_layers", "num_mtp_layers"),
    [
        (False, False, True, 0, 2),
        (True, False, False, 2, 2),
        (True, True, False, 0, 2),
        (True, False, True, 0, 2),
        (False, False, False, 2, 0),
    ],
    ids=["logits-learned", "chunked-hash", "chunked-isolated", "chunked-prepared-labels", "mtp-disabled"],
)
def test_upstream_mtp_block_matches_baseline_outputs_loss_and_gradients(
    chunked, isolate, prepared, n_hash_layers, num_mtp_layers, single_threaded_cpu
):
    with torch.random.fork_rng():
        config = build_v4_model_config(num_mtp_layers=num_mtp_layers, n_hash_layers=n_hash_layers)
        config.mtp_isolate_gradients = isolate
        model = build_cpu_model(config)
        reference = deepcopy(model)
        assert all(isinstance(layer, DeepSeekV4TransformerBlock) for layer in model.layers.values())
        assert all(type(layer) is MTPBlock for layer in model.mtp_layers or ())
        for layer in reference.layers.values():
            layer.forward = MethodType(_baseline_v4_block_forward, layer)
        for layer in reference.mtp_layers or ():
            layer.forward = MethodType(_baseline_mtp_forward, layer)
        model._skip_lm_head = reference._skip_lm_head = chunked
        tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
        labels = (tokens + 1) % VOCAB_SIZE
        inputs, labels, extra_kwargs = preprocess(model, tokens, labels, PACKED_POSITIONS.clone())
        if not prepared:
            extra_kwargs.pop("mtp_batch", None)

        actual_outputs = model(inputs, **extra_kwargs)
        expected_outputs = reference(inputs, **extra_kwargs)
        actual_branches = [actual_outputs] if isinstance(actual_outputs, torch.Tensor) else actual_outputs
        expected_branches = [expected_outputs] if isinstance(expected_outputs, torch.Tensor) else expected_outputs
        for actual, expected in zip(actual_branches, expected_branches, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if chunked:
            loss_config = MTPChunkedLossWrapper.Config(
                num_chunks=2,
                mtp_isolate_gradients=isolate,
                loss_fn=CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE),
            )
            actual_fn = loss_config.build()
            expected_fn = deepcopy(loss_config).build()
            actual_fn.set_lm_head(model.lm_head)
            expected_fn.set_lm_head(reference.lm_head)
        elif num_mtp_layers:
            actual_fn = MTPLoss(MTPLoss.Config(global_vocab_size=VOCAB_SIZE))
            expected_fn = MTPLoss(MTPLoss.Config(global_vocab_size=VOCAB_SIZE))
        else:
            actual_fn = CrossEntropyLoss(CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE))
            expected_fn = CrossEntropyLoss(CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE))
        denominator = torch.tensor(labels.numel())
        if chunked:
            actual_loss, _ = actual_fn(actual_outputs, labels, denominator, positions=PACKED_POSITIONS)
            expected_loss, _ = expected_fn(expected_outputs, labels, denominator, positions=PACKED_POSITIONS)
        elif num_mtp_layers:
            aligned_labels = (labels, *extra_kwargs["mtp_batch"].labels.unbind(dim=-1))
            actual_loss, _ = actual_fn(tuple(actual_outputs), aligned_labels, denominator)
            expected_loss, _ = expected_fn(tuple(expected_outputs), aligned_labels, denominator)
        else:
            actual_loss, _ = actual_fn(actual_outputs, labels, denominator)
            expected_loss, _ = expected_fn(expected_outputs, labels, denominator)
        actual_loss.backward()
        expected_loss.backward()

        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        expected_parameters = dict(reference.named_parameters())
        assert dict(model.named_parameters()).keys() == expected_parameters.keys()
        for name, parameter in model.named_parameters():
            expected_grad = expected_parameters[name].grad
            assert (parameter.grad is None) == (expected_grad is None), name
            if expected_grad is not None:
                torch.testing.assert_close(
                    parameter.grad,
                    expected_grad,
                    rtol=0,
                    atol=0,
                    msg=lambda message, name=name: f"{name}: {message}",
                )
