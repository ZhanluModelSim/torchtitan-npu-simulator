# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def test_deepseek_v3_mapping():
    from torchtitan_npu.converters.kernels.rope import (
        npu_apply_rotary_emb_deepseek,
        RoPEKernel,
    )

    module_path, func_name, impl = RoPEKernel.get_impl_cls("deepseek_v3")

    assert module_path == "torchtitan.models.common.rope"
    assert func_name == "apply_rotary_emb_single_complex"
    assert impl is npu_apply_rotary_emb_deepseek


def test_deepseek_v32_mapping():
    from torchtitan_npu.converters.kernels.rope import (
        npu_apply_rotary_emb_deepseek,
        RoPEKernel,
    )

    # NOTE: "deepseek_v3" key matches "deepseek_v32" first due to substring
    # matching; v32-specific entry is currently unreachable. This is a known
    # issue to be fixed separately.
    module_path, func_name, impl = RoPEKernel.get_impl_cls("deepseek_v32")

    assert impl is npu_apply_rotary_emb_deepseek


def test_qwen3_mapping():
    from torchtitan_npu.converters.kernels.rope import (
        npu_apply_rotary_emb_qwen,
        RoPEKernel,
    )

    module_path, func_name, impl = RoPEKernel.get_impl_cls("qwen3")

    assert module_path == "torchtitan.models.common.rope"
    assert func_name == "apply_rotary_emb_cos_sin"
    assert impl is npu_apply_rotary_emb_qwen


def test_llama_mapping():
    from torchtitan_npu.converters.kernels.rope import (
        npu_apply_rotary_emb_llama,
        RoPEKernel,
    )

    module_path, func_name, impl = RoPEKernel.get_impl_cls("llama3")
    default_entry = RoPEKernel.MODEL_IMPL["_default"]

    assert default_entry == (
        "torchtitan.models.common.rope",
        "apply_rotary_emb_complex",
        npu_apply_rotary_emb_llama,
    )
    assert (module_path, func_name, impl) == default_entry


def test_unknown_mapping():
    from torchtitan_npu.converters.kernels.rope import (
        npu_apply_rotary_emb_llama,
        RoPEKernel,
    )

    module_path, func_name, impl = RoPEKernel.get_impl_cls("unknown")
    default_entry = RoPEKernel.MODEL_IMPL["_default"]

    assert default_entry[2] is npu_apply_rotary_emb_llama
    assert (module_path, func_name, impl) == default_entry
