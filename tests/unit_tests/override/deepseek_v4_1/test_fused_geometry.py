# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The port's kernel geometry: stated by the factory, never defaulted on the config.

The port's config *is* one layer's geometry -- it builds the metadata its own kernels read -- so
the five widths live on it and are required, and nothing else does: the candidate pool belongs to
the model (the reference selector builds and searches it), so the MXFP4 selector's metadata takes
its numbers from the model config and the port states none of them.  What ``sparse_attn.asc``
adds is the one place a recipe states the widths: its arguments default to the released
V4.1-Flash shape, which is what makes the shipped Flash recipes need no arguments at all, and a
recipe that runs other widths states its own.  The facts this file pins are therefore: the
*reference* core config carries no kernel geometry (the AscendC attention's metadata is the only consumer),
the port config has no defaults for it, the factory's defaults are exactly the Flash numbers, the
port states no quantization, and every geometry the shipped
flavors describe is the one their recipes state -- or, for the debug flavor, the one they must
state, since its widths are not the released ones.
"""

from dataclasses import fields

import pytest

from torchtitan_npu.models.deepseek_v4_1 import config_registry
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.override.deepseek_v4_1.metadata import AicpuGeometry
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import asc
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention

# What the metadata is sized with, in the order the port states them.
REFERENCE_GEOMETRY_FIELDS = {
    "num_heads",
    "head_dim",
    "index_topk",
    "index_num_heads",
    "index_head_dim",
    "compress_ratios",
}
# The five the port's own config declares (``compress_ratios`` is the stack-wide table the
# reference model carries, which no single geometry's metadata reads).
PORT_GEOMETRY_FIELDS = REFERENCE_GEOMETRY_FIELDS - {"compress_ratios"}
# The released V4.1-Flash shape: heads, head dim, selection width, indexer heads, indexer head dim.
RELEASED = (64, 512, 512, 32, 128)
# The registered debug flavor's own widths, which is the shape the CPU tests run.
DEBUG = (8, 64, 32, 8, 32)


def _port_numbers(port) -> tuple[int, ...]:
    """The five numbers the port would size this config's metadata with."""
    return (port.num_heads, port.head_dim, port.index_topk, port.index_num_heads, port.index_head_dim)


def _geometry_numbers(geometry: AicpuGeometry) -> tuple[int, ...]:
    """The same five, read off the geometry the AI_CPU operators are called with."""
    return (
        geometry.num_heads,
        geometry.head_dim,
        geometry.index_topk,
        geometry.index_num_heads,
        geometry.index_head_dim,
    )


def _model_numbers(config) -> tuple[int, ...]:
    """The same five, read off every layer of the model the config describes."""
    numbers = {
        (
            layer.attention.n_heads,
            layer.attention.head_dim,
            layer.attention.indexer.index_topk,
            layer.attention.indexer.num_index_heads,
            layer.attention.indexer.index_head_dim,
        )
        for layer in config.model_spec.model.layers
    }
    assert len(numbers) == 1, f"the stack mixes geometries, which one port config cannot describe: {numbers}"
    return numbers.pop()


def _registered(name: str):
    """A published config, or a skip where this environment cannot build its vision tower."""
    try:
        return getattr(config_registry, name)()
    except RuntimeError as error:  # pragma: no cover - the vision tower needs torchvision
        if "torchvision" in str(error):
            pytest.skip("this environment cannot build the vision tower (torchvision mismatch)")
        raise


def test_the_reference_config_carries_no_kernel_geometry() -> None:
    """Only the AscendC attention's metadata needs the widths, so the reference core never states them."""
    reference = CompressedSparseInnerAttention2.Config(window_size=2, softmax_scale=0.5, compress_ratio=2)
    assert REFERENCE_GEOMETRY_FIELDS.isdisjoint({item.name for item in fields(reference) if item.init})
    # Not merely absent-but-accepted: a recipe that tries to state one is rejected by the type.
    with pytest.raises(TypeError, match="num_heads"):
        CompressedSparseInnerAttention2.Config(window_size=2, softmax_scale=0.5, compress_ratio=2, num_heads=1)


def test_the_port_config_has_no_defaults_for_the_geometry() -> None:
    """The config *is* one layer's geometry, so all five are required and none is guessed."""
    # A missing number is a build-time error, not a defaults-to-Flash: the released shape belongs
    # to the factory a recipe names, which is the one place a whole stack states it.
    with pytest.raises(TypeError, match="num_heads"):
        AscV41SparseAttention.Config(window_size=128, softmax_scale=0.5, compress_ratio=2)
    # And they are init fields of the type, not properties derived from somewhere else.
    declared = {item.name for item in fields(AscV41SparseAttention.Config) if item.init}
    assert declared >= PORT_GEOMETRY_FIELDS
    # The geometry the operators are called with is read off exactly those fields.
    port = AscV41SparseAttention.Config(
        window_size=128,
        softmax_scale=0.5,
        compress_ratio=2,
        num_heads=1,
        head_dim=4,
        index_topk=3,
        index_num_heads=2,
        index_head_dim=4,
    )
    assert port.geometry == AicpuGeometry(
        num_heads=1,
        head_dim=4,
        window_size=128,
        index_topk=3,
        index_num_heads=2,
        index_head_dim=4,
    )


def test_the_port_states_no_quantization() -> None:
    """The pool's numbers are the model's, so neither the factory nor the geometry takes them.

    The reference selector builds and searches the candidate pool in torch, so its capacity and
    block size are model facts; the MXFP4 selector's metadata reads them off the model config where
    the pool is declared (through the adaptor), and the AscendC attention -- whose kernels never see the
    pool -- states nothing about it.
    """
    base = CompressedSparseInnerAttention2.Config(window_size=128, softmax_scale=0.5, compress_ratio=2)

    port = asc(base)

    assert not hasattr(port, "candidate_topk_blocks")
    assert not hasattr(port.geometry, "candidate_topk_blocks")
    assert not hasattr(port.geometry, "has_candidate_pool")
    with pytest.raises(TypeError):
        asc(base, candidate_topk_blocks=2048, candidate_block_size=8)
    # One entry, one geometry: the widths, and nothing that belongs to a consumer.
    assert _geometry_numbers(port.geometry) == RELEASED


def test_the_override_lands_the_numbers_it_is_given() -> None:
    """``asc`` states each field once: the defaults, or the recipe's own numbers."""
    base = CompressedSparseInnerAttention2.Config(window_size=128, softmax_scale=0.5, compress_ratio=2)

    defaulted = asc(base)
    assert type(defaulted) is AscV41SparseAttention.Config
    assert _port_numbers(defaulted) == RELEASED
    # The reference core's own fields survive the swap, and the key names the swapped class.
    assert (defaulted.window_size, defaulted.softmax_scale, defaulted.compress_ratio) == (128, 0.5, 2)
    assert defaulted.attention_metadata_key == (AscV41SparseAttention.Config, 2)
    # The one datum the builder hands the operators restates the same five: a drift between the
    # config and the geometry would size the metadata from one shape and call the operator with another.
    assert _geometry_numbers(defaulted.geometry) == RELEASED
    assert defaulted.geometry.window_size == defaulted.window_size

    stated = asc(
        base,
        num_heads=DEBUG[0],
        head_dim=DEBUG[1],
        index_topk=DEBUG[2],
        index_num_heads=DEBUG[3],
        index_head_dim=DEBUG[4],
    )
    assert _port_numbers(stated) == DEBUG
    assert _geometry_numbers(stated.geometry) == DEBUG
    # The window and the ratio are the layer's, not the kernel geometry's: they ride along.
    assert (stated.geometry.window_size, stated.compress_ratio) == (128, 2)


def test_a_flavor_of_other_widths_states_them() -> None:
    """The debug flavor is not the released shape, which is why its recipes state the numbers."""
    config = _registered("deepseek_v4_1_debugmodel_text")

    assert _model_numbers(config) == DEBUG
    assert DEBUG != RELEASED, "the defaults cannot serve this flavor, and this is the test that says so"

    # Stated, the port matches the model it runs: the metadata is sized from the same numbers the
    # layer reads, which is what makes a mismatch a build-time error rather than a lookup miss.
    inner = config.model_spec.model.layers[0].attention.inner_attention
    port = asc(
        inner,
        num_heads=DEBUG[0],
        head_dim=DEBUG[1],
        index_topk=DEBUG[2],
        index_num_heads=DEBUG[3],
        index_head_dim=DEBUG[4],
    )
    assert _port_numbers(port) == DEBUG
    assert _geometry_numbers(port.geometry) == DEBUG


def test_the_flash_flavor_is_the_default() -> None:
    """The one published flavor the defaults describe, pinned against the model's own numbers."""
    config = _registered("deepseek_v4_1_flash_text")
    assert _model_numbers(config) == RELEASED
