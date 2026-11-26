# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import DTensor
from torchtitan.config import derive
from torchtitan.models.common.linear import Linear

from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import tiny_v41_model
from torchtitan_npu.models.deepseek_v4_1.parallelize import _chunk_v41_moe_forward
from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import _GrainV41Collator
from torchtitan_npu_recipes.graph_trainer.deepseek_v4_1 import (
    graph_trainer_deepseek_v4_1_flash_40layers_16experts_multimodal,
    graph_trainer_deepseek_v4_1_flash_40layers_16experts_text,
)


@pytest.fixture(autouse=True)
def restore_graph_trainer_global_annotations(monkeypatch):
    from torchtitan.experiments.graph_trainer import common_utils
    from torchtitan.models.common.moe import MoE
    from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher, LocalTokenDispatcher

    # Upstream annotates class methods once per process. Keep the real
    # annotation path, but restore it before unrelated dispatcher tests run.
    monkeypatch.setattr(common_utils, "_MOE_EP_REGIONS_ANNOTATED", common_utils._MOE_EP_REGIONS_ANNOTATED)
    for cls, names in (
        (LocalTokenDispatcher, ("dispatch", "combine")),
        (
            AllToAllTokenDispatcher,
            (
                "dispatch",
                "combine",
                "_token_count_exchange",
                "_sync_token_count_exchange",
                "_dispatch_token_exchange",
                "_combine_token_exchange",
            ),
        ),
        (MoE, ("forward",)),
    ):
        for name in names:
            monkeypatch.setattr(cls, name, getattr(cls, name))
    monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", torch._dynamo.config.capture_scalar_outputs)
    monkeypatch.setattr(
        torch._inductor.config, "reorder_for_peak_memory", torch._inductor.config.reorder_for_peak_memory
    )


@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
@pytest.mark.parametrize("with_padding", [False, True], ids=["dense", "padding"])
def test_moe_chunks_preserve_masks_gradients_and_global_denominator(with_image, with_padding):
    denominator = torch.tensor(16.0)
    seen = []

    def forward(x, *, image_mask, padding_mask_T, aux_loss_denominator):
        seen.append(aux_loss_denominator)
        image = 0 if image_mask is None else image_mask[:, None].float()
        padding = 0 if padding_mask_T is None else padding_mask_T[:, None].float()
        return (x + image - padding) / aux_loss_denominator

    x = torch.arange(4.0).reshape(4, 1).requires_grad_()
    image = torch.tensor([False, True, True, False]) if with_image else None
    padding = torch.tensor([False, False, True, True]) if with_padding else None
    result = _chunk_v41_moe_forward(forward, "layers.0.feed_forward")(
        x, image_mask=image, padding_mask_T=padding, aux_loss_denominator=denominator
    )
    expected = [0.0, 1.0, 2.0, 3.0]
    if with_image:
        expected = [0.0, 2.0, 3.0, 3.0]
    if with_padding:
        expected[2] -= 1
        expected[3] -= 1
    torch.testing.assert_close(result, torch.tensor(expected).reshape(4, 1) / 16)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.full((4, 1), 1 / 16))
    assert len(seen) == 2
    assert all(value is denominator for value in seen)


@pytest.mark.parametrize(
    "ep_overlap,module_fqn",
    [(False, "layers.*.feed_forward"), (False, "layers.*"), (True, "layers.*.feed_forward")],
    ids=["base", "base-unused-overlap-options", "ep-overlap"],
)
def test_graph_parallelize_initializes_v040_model_parallel_axes(
    tmp_path, monkeypatch, single_rank_spmd, ep_overlap, module_fqn
):
    # Shared dispatcher annotations outlive the model; this test checks sharding.
    monkeypatch.setattr("torchtitan.experiments.graph_trainer.common_utils.annotate_moe_ep_regions", lambda: None)
    monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", torch._dynamo.config.capture_scalar_outputs)
    monkeypatch.setattr(
        torch._inductor.config, "reorder_for_peak_memory", torch._inductor.config.reorder_for_peak_memory
    )
    config = graph_trainer_deepseek_v4_1_flash_40layers_16experts_text(hf_assets_path="tests/assets/deepseek_v3")
    config.training.mixed_precision_param = "float32"
    config.compile.ep_overlap.enabled = ep_overlap
    config.compile.ep_overlap.module_fqn = module_fqn
    config.model.layers = config.model.layers[:2]
    config.model.n_layers = 2
    config.model.compress_ratios = (1, 1)
    config.model.vocab_size = config.model.tok_embeddings.num_embeddings = config.model.lm_head.out_features = 64
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        model = config.model.build()
    model = model.parallelize(
        parallelism_context=single_rank_spmd,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=config.compile,
        activation_remat=config.activation_remat,
        dump_folder=str(tmp_path),
    )
    projection = model.layers["0"].attention.wq_a
    assert projection._parallelized
    assert isinstance(projection._parameters["weight"], DTensor)
    assert not isinstance(projection.weight, DTensor)
    table = model.layers["1"].engram.table
    assert not isinstance(table.weight, DTensor)
    assert table.weight._engram_host_offload
    projection.weight.square().sum().backward()
    assert isinstance(projection._parameters["weight"].grad, DTensor)


@pytest.mark.parametrize(
    "invalid_input",
    [None, "zero-height", "negative-width", "extra-patches", "missing-patches", "feature-oob", "invalid-sentinel"],
    ids=[
        "valid",
        "zero-height",
        "negative-width",
        "extra-patches",
        "missing-patches",
        "feature-oob",
        "invalid-sentinel",
    ],
)
def test_multimodal_graph_preprocessing_preserves_v040_loss_kwargs_and_image_gradients(
    monkeypatch, single_rank_spmd, invalid_input
):
    config = graph_trainer_deepseek_v4_1_flash_40layers_16experts_multimodal(hf_assets_path="tests/assets/deepseek_v3")
    _, tiny_config = tiny_v41_model(monkeypatch, flavor="deepseek_v4_1_debugmodel")
    model_config = derive(tiny_config, type(config.model))
    model_config.layers = model_config.layers[:3]
    model_config.n_layers = 3
    model_config.compress_ratios = model_config.compress_ratios[:3]
    model_config.vocab_size = model_config.tok_embeddings.num_embeddings = model_config.lm_head.out_features = 64
    vision = model_config.vision_encoder
    vision.dim, vision.inter_dim, vision.num_layers, vision.num_heads = 8, 16, 1, 2
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = model_config.build()
        model.init_states(buffer_device=torch.device("cpu"))
    vision_linears = [module for module in model.vision_encoder.modules() if isinstance(module, Linear)]
    assert vision_linears
    assert all(not module.inplace_wgrad_accum for module in vision_linears)
    tokens = torch.arange(128) % 64
    patches = torch.linspace(-1, 1, 9 * 588).reshape(1, 9, 588).requires_grad_()
    collator = _GrainV41Collator.Config().build(
        context=SimpleNamespace(num_tokens_per_microbatch=128, tokenizer=SimpleNamespace(get_vocab_size=lambda: 64))
    )
    microbatch = collator(
        [
            {
                "input_ids": tokens,
                "labels": tokens.clone(),
                "positions": torch.arange(128),
                "pixel_values": [(patches.squeeze(0), torch.tensor([3, 3]))],
            }
        ]
    )
    batch = microbatch.as_input_dict()
    batch["loss_kwargs"] = {"loss_mask": torch.ones(128)}
    if invalid_input is not None:
        if invalid_input == "zero-height":
            batch["image_grid"][0, 0] = 0
        elif invalid_input == "negative-width":
            batch["image_grid"][0, 1] = -1
        elif invalid_input == "extra-patches":
            batch["pixel_values"] = torch.cat([batch["pixel_values"], patches[:, :1]], dim=1)
        elif invalid_input == "missing-patches":
            batch["pixel_values"] = batch["pixel_values"][:, :-1]
        else:
            batch["image_feature_indices"][0] = 1 if invalid_input == "feature-oob" else -2
        with pytest.raises(ValueError, match=r"Image|image_feature_indices"):
            model.preprocess_inputs(
                batch,
                parallelism_context=single_rank_spmd,
                parallelism=config.parallelism,
                global_routing_token_counts=microbatch.routing_token_counts,
            )
        return
    inputs, labels, kwargs, loss_kwargs = model.preprocess_inputs(
        batch,
        parallelism_context=single_rank_spmd,
        parallelism=config.parallelism,
        global_routing_token_counts=microbatch.routing_token_counts,
    )
    # The real collator must provide one count per routing depth, not a scalar.
    torch.testing.assert_close(kwargs["aux_loss_denominators"], torch.tensor([128]))
    torch.testing.assert_close(kwargs["aux_loss_denominators"][0], torch.tensor(128))
    assert labels.shape == (128,)
    torch.testing.assert_close(loss_kwargs["loss_mask"], torch.ones(128))
    torch.testing.assert_close(kwargs["image_patch_indices"], torch.arange(9).reshape(1, 9))
    from torch.utils._pytree import tree_flatten, tree_unflatten

    from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Metadata

    leaves, spec = tree_flatten(kwargs["attention_metadata"])
    assert not any(isinstance(leaf, DeepSeekV41Metadata) for leaf in leaves)
    kwargs["attention_metadata"] = tree_unflatten(leaves, spec)
    output = model(inputs, **kwargs)
    assert output.shape == (128, 64)
    output.square().mean().backward()
    assert patches.grad is not None and torch.isfinite(patches.grad).all()
    assert patches.grad.abs().sum() > 0


@pytest.mark.parametrize("branch", ["hc_attn_pre", "hc_ffn_pre"], ids=["attention", "ffn"])
def test_graph_bfloat16_forward_backward_preserves_hc_fp32_parameters(monkeypatch, single_rank_spmd, tmp_path, branch):
    from torchtitan_npu.models.deepseek_v4_1.model import GraphTrainerDeepSeekV41Model

    recipe = graph_trainer_deepseek_v4_1_flash_40layers_16experts_text(
        hf_assets_path="tests/assets/deepseek_v3", engram_enabled=False
    )
    _, tiny = tiny_v41_model(monkeypatch)
    config = derive(tiny, GraphTrainerDeepSeekV41Model.Config)
    config.fp32_param_patterns = recipe.model.fp32_param_patterns
    with torch.device("cpu"):
        model = config.build()
        model.init_states(buffer_device=torch.device("cpu"))
    reference = deepcopy(getattr(model.layers["0"], branch))
    # Values that would change under a BF16 round-trip expose premature casts.
    with torch.no_grad():
        reference.hc_fn.fill_(0.1234567)
        getattr(model.layers["0"], branch).hc_fn.copy_(reference.hc_fn)
    recipe.compile.ep_overlap.enabled = False
    model.parallelize(
        parallelism_context=single_rank_spmd,
        training=recipe.training,
        parallelism=recipe.parallelism,
        compile_config=recipe.compile,
        activation_remat=recipe.activation_remat,
        dump_folder=str(tmp_path),
    )
    hc = getattr(model.layers["0"], branch)
    torch.testing.assert_close(hc.hc_fn, reference.hc_fn, rtol=0, atol=0)
    assert model.layers["0"].attention.attn_sink.dtype == torch.float32
    assert model.layers["0"].attention.wq_a.weight.dtype == torch.bfloat16
    x = torch.randn(4, hc.hc_mult, config.dim, dtype=torch.bfloat16, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    pre = torch.full((4, hc.hc_mult), 1 / hc.hc_mult)
    actual, expected = hc(x, pre), reference(expected_x, pre)
    for result, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, oracle, rtol=0, atol=0)
    sum(value.float().square().sum() for value in actual).backward()
    sum(value.float().square().sum() for value in expected).backward()
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(hc._parameters["hc_fn"].grad.to_local(), reference.hc_fn.grad, rtol=0, atol=0)


@pytest.mark.parametrize("axis", ["tensor", "context", "pipeline"], ids=["tp", "cp", "pp"])
def test_graph_parallelize_rejects_unsupported_axes_before_mesh_initialization(axis):
    recipe = graph_trainer_deepseek_v4_1_flash_40layers_16experts_text(
        hf_assets_path="tests/assets/deepseek_v3", engram_enabled=False
    )
    setattr(recipe.parallelism, f"{axis}_parallel_degree", 2)
    with torch.device("meta"):
        model = recipe.model.build()
    from torchtitan_npu.models.deepseek_v4_1.parallelize import parallelize_graph_trainer_deepseek_v4_1

    with pytest.raises(NotImplementedError, match="TP=CP=PP=1"):
        parallelize_graph_trainer_deepseek_v4_1(
            model,
            parallelism_context=None,
            training=recipe.training,
            parallelism=recipe.parallelism,
            compile_config=recipe.compile,
            activation_remat=recipe.activation_remat,
            dump_folder="unused",
        )


def test_graph_parallelize_rejects_dispatcher_capacity_before_buffer_initialization():
    from torchtitan.models.common.moe.token_dispatcher import DeepEPTokenDispatcher

    from torchtitan_npu.models.deepseek_v4_1.parallelize import parallelize_graph_trainer_deepseek_v4_1

    recipe = graph_trainer_deepseek_v4_1_flash_40layers_16experts_text(
        hf_assets_path="tests/assets/deepseek_v3", engram_enabled=False
    )
    recipe.parallelism.expert_parallel_degree = 8
    experts = recipe.model.layers[0].feed_forward.routed_experts
    experts.token_dispatcher = derive(experts.token_dispatcher, DeepEPTokenDispatcher.Config, num_max_tokens_per_rank=1)
    # Config validation must run before attempting to construct the DeepEP buffer.
    model = SimpleNamespace(config=recipe.model)
    with pytest.raises(ValueError, match="num_max_tokens_per_rank must be at least"):
        parallelize_graph_trainer_deepseek_v4_1(
            model,
            parallelism_context=None,
            training=recipe.training,
            parallelism=recipe.parallelism,
            compile_config=recipe.compile,
            activation_remat=recipe.activation_remat,
            dump_folder="unused",
        )
