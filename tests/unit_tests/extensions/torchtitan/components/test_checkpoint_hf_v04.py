# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Full tiny-model HF safetensors loading through the real checkpoint manager."""

import json

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.checkpoint import CheckpointException, HuggingFaceStorageReader
from torchtitan.components.optim import AdamW, LRSchedulersContainer, OptimizersContainer

from torchtitan_npu.extensions.mx_storage_reader.engram import EngramHuggingFaceStorageReader
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import (
    HostSparseOptimizersContainer,
    SparseAdam,
)


def _model_config(monkeypatch, family):
    if family == "v4":
        import torchtitan_npu.models.deepseek_v4 as registry

        make = registry._make_v4_config

        def tiny(**kwargs):
            kwargs.update(
                dim=16,
                n_layers=2,
                vocab_size=32,
                n_heads=2,
                head_dim=8,
                rope_head_dim=4,
                q_lora_rank=8,
                o_lora_rank=4,
                n_groups=1,
                compress_ratios=(4, 1),
                window_size=8,
                index_n_heads=2,
                index_head_dim=8,
                index_topk=2,
                moe_inter_dim=16,
                num_experts=2,
                num_shared_experts=1,
                top_k=1,
                n_hash_layers=1,
                hc_mult=2,
                sinkhorn_iters=2,
                original_seq_len=16,
            )
            return make(**kwargs)

        monkeypatch.setattr(registry, "_make_v4_config", tiny)
        return registry.model_registry("debugmodel", seq_len=16, num_mtp_layers=1)

    import torchtitan_npu.models.deepseek_v4_1 as registry

    make = registry._make_v41_config

    def tiny(**kwargs):
        kwargs.update(
            dim=16,
            n_layers=4,
            vocab_size=32,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=4,
            n_groups=1,
            compress_ratios=(0, 2, 2, 2),
            kv_source_layers=(1,),
            index_source_layers=(1, 2),
            candidate_source_layer=1,
            window_size=8,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            moe_inter_dim=16,
            num_experts=2,
            top_k=1,
            hc_mult=2,
            sinkhorn_iters=2,
            original_seq_len=16,
            max_seq_len=16,
        )
        return make(**kwargs)

    monkeypatch.setattr(registry, "_make_v41_config", tiny)
    monkeypatch.setattr(
        registry,
        "_ENGRAM_DEBUG_GEOMETRY",
        registry.EngramArgs(
            layer_ids=(1,),
            vocab_size_per_ngram=(17,),
            n_embed_per_ngram=8,
            ngram_orders=(2,),
            num_heads_per_ngram=2,
            table_padding_multiple=8,
            require_token_id_map=False,
        ),
    )
    return registry.model_registry("deepseek_v4_1_debugmodel_text")


def _build_model(config, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = config.build()
        model.init_states(buffer_device=torch.device("cpu"))
    return model


@pytest.fixture(params=["v4", "v41"])
def hf_case(tmp_path, monkeypatch, request):
    family = request.param
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    manager = None
    try:
        config = _model_config(monkeypatch, family)
        source = _build_model(config, 11)
        target = _build_model(config, 23)
        if family == "v41":
            table = source.layers["1"].engram.table
            with torch.no_grad():
                table.weight[table.logical_num_embeddings:].zero_()
        expected = {name: value.detach().clone() for name, value in source.state_dict().items()}
        adapter = config._owner.state_dict_adapter_cls(config, None)
        hf = adapter.to_hf(source.state_dict(), to_quantized_layout=False)
        assert "embed.weight" in hf and "head.weight" in hf
        if family == "v4":
            assert "layers.0.ffn.gate.tid2eid" in hf
            assert any(name.startswith("mtp.0.") for name in hf)
        else:
            assert "layers.1.engram.embed.weight" in hf
            assert hf["layers.1.engram.embed.weight"].shape == (table.logical_num_embeddings, table.embedding_dim)
            assert "layers.1.attn.indexer.wk.weight" in hf
            assert "layers.2.attn.indexer.wq_b.weight" in hf
        tensors = {name: value.detach().clone().contiguous() for name, value in hf.items()}
        assets = tmp_path / "hf"
        assets.mkdir()
        save_file(tensors, assets / "model.safetensors")
        (assets / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": dict.fromkeys(tensors, "model.safetensors")})
        )
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.zero_()
            for name, buffer in target.named_buffers():
                if name.endswith("tid2eid"):
                    buffer.zero_()
        reader = adapter.get_hf_storage_reader(str(assets), from_quantized=False)
        assert isinstance(reader, HuggingFaceStorageReader)
        assert isinstance(reader, EngramHuggingFaceStorageReader) == (family == "v41")
        optimizer_config = (
            HostSparseOptimizersContainer.Config(
                optimizers=[
                    SparseAdam.Config(pattern=r".*\.engram\.table\.weight$"),
                    AdamW.Config(pattern=r".*", fused=False),
                ]
            )
            if family == "v41"
            else OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", fused=False)])
        )
        optimizers = optimizer_config.build(model_parts=[target])
        schedulers = LRSchedulersContainer.Config(warmup_steps=0).build(optimizers=optimizers, training_steps=2)
        manager = CheckpointManager.Config(
            initial_load_path=str(assets),
            initial_load_in_hf=True,
            initial_load_model_only=True,
            interval=1,
            keep_latest_k=0,
        ).build(
            model_parts=[target],
            optimizers=optimizers,
            lr_schedulers=schedulers,
            ema=None,
            extra_states={},
            sd_adapter=adapter,
            base_folder=str(tmp_path / "run"),
        )
        yield manager, target, expected, assets, tensors
    finally:
        if manager is not None:
            manager.close()
        dist.destroy_process_group()


def test_manager_loads_full_hf_weights_into_registered_tiny_model(hf_case):
    manager, target, expected, _, _ = hf_case
    assert manager.load()
    actual = target.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)


def test_manager_rejects_hf_checkpoint_missing_head(hf_case):
    manager, _, _, assets, tensors = hf_case
    tensors.pop("head.weight")
    save_file(tensors, assets / "model.safetensors")
    (assets / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    with pytest.raises(CheckpointException, match=r"head\.weight"):
        manager.load()
