# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import torch
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)

from tests.unit_tests.models.common.text_data_utils import IntegerTextTokenizer
from torchtitan_npu.models.common.alignment import AlignedTextProcessor
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel


def test_debugmodel_uses_the_documented_per_layer_compression_ratios():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_spec = model_registry("debugmodel")

    assert model_spec.compress_ratios == (1, 1, 4, 128)
    assert len(model_spec.layers) == len(model_spec.compress_ratios)


def test_flash_mtp_4k_model_flops():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_config = model_registry(
        "deepseek_v4_flash",
        num_mtp_layers=1,
    )

    with torch.device("meta"):
        model = model_config.build()

    assert model_config.get_nparams_and_flops(model, seq_len=4096) == (
        290_942_278_866,
        92_762_352_876,
    )


def test_debugmodel_uses_cann_profiler_extension():
    from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel()

    assert isinstance(config.profiler, CANNProfiler.Config)
    assert isinstance(config.profiler.build(), CANNProfiler)


def test_every_layer_pins_the_core_config_the_overrides_target():
    """The sparse core's config type is the anchor the fused overrides replace.

    ``asc``/``pypto`` are registered against exactly this type, so a model that
    moved to another Config class would quietly run the reference core instead of the fused
    one.  The core's implementation may change; this anchor may not.
    """
    from torchtitan_npu.models.deepseek_v4 import model_registry
    from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention

    model_config = model_registry("debugmodel")
    cores = [layer.attention.inner_attention for layer in model_config.layers]

    assert len(cores) == len(model_config.compress_ratios) == 4
    for core in cores:
        assert type(core) is CompressedSparseInnerAttention.Config
        assert isinstance(core, CompressedSparseInnerAttention.Config)


def test_flash_rope_configs_pin_split_per_site():
    """Every rope site's config carries the split matching its tensor width."""
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_config = model_registry("deepseek_v4_flash", num_mtp_layers=0)
    rd = 64

    for layer in model_config.layers:
        attn = layer.attention
        # Attention q/kv/o rotate the tail of a head_dim-wide (512) tensor.
        assert attn.rope.dim == rd
        assert attn.rope.split == attn.head_dim - rd
        if attn.compress_ratio > 1:
            assert attn.compressor.rope.split == attn.compressor.head_dim - rd
        if attn.compress_ratio == 4:
            # Indexer q is index_head_dim-wide (128): a different split even
            # though it reuses the same base rope flavor.
            indexer = attn.indexer
            assert indexer.rope.dim == rd
            assert indexer.rope.split == indexer.index_head_dim - rd
            return
    raise AssertionError("flash spec must contain compress_ratio == 4 layers")


def test_dataloader_pads_documents_to_the_compression_alignment():
    """Every pooled document must be a whole number of blocks at every ratio.

    The loader pads each document to ``lcm(compress_ratios)``, so a compressed
    block never straddles a document edge -- the property the compressor's
    "complete blocks only" contract and the kernels' ``cmp_residual_k`` rely on.
    """
    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel()
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    assert isinstance(config.dataloader.dataset, ConcatThenSplitPackingConfig)
    assert isinstance(config.dataloader.dataset.dataset, SingleDatasetConfig)
    assert isinstance(config.dataloader.dataset.dataset.processor, AlignedTextProcessor.Config)
    alignment = config.dataloader.dataset.dataset.processor.per_doc_alignment
    assert alignment == 128
    assert config.training.max_context_length % alignment == 0


def test_aligned_dataloader_only_inserts_pads(tmp_path):
    path = tmp_path / "text.jsonl"
    path.write_text('{"text":"3 4"}\n{"text":"5"}\n')
    recipe = deepseek_v4_debugmodel()
    assert isinstance(recipe.dataloader, GrainDataLoader.Config)
    recipe.dataloader.shuffle = False
    recipe.dataloader.repeat = False
    packing = recipe.dataloader.dataset
    assert isinstance(packing, ConcatThenSplitPackingConfig)
    assert isinstance(packing.dataset, SingleDatasetConfig)
    recipe.dataloader.dataset = replace(
        packing,
        dataset=replace(
            packing.dataset,
            source=HuggingFaceRandomAccessSource.Config(
                path="json", split="train", load_dataset_kwargs={"data_files": str(path)}
            ),
        ),
    )
    loader = recipe.dataloader.build(
        tokenizer=IntegerTextTokenizer(),
        dp_world_size=1,
        dp_rank=0,
        max_context_length=128,
        num_tokens_per_microbatch=256,
    )
    try:
        batches = list(loader)
    finally:
        loader.close()

    assert len(batches) == 1
    inputs = batches[0].as_input_dict()
    labels = inputs["labels"]
    assert inputs["input"].tolist() == [1, 3, 4] + [0] * 125 + [1, 5] + [0] * 126
    assert labels.tolist() == [3, 4, 2] + [-100] * 125 + [5, 2] + [-100] * 126
    assert inputs["positions"].tolist() == list(range(128)) * 2
    assert batches[0].num_valid_tokens == 5
