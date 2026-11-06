# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared pytest fixtures for the CPU unit-test suite.

The plugin's model-dir and override host logic import against the real
torchtitan checkout (env ``TORCHTITAN_DIR`` or the default) with the
plugin's patches applied by the package chain.  The only faked surface is
``cann_ops_transformer`` (the NPU op boundary): the ``dsv4`` fixture
installs the call recorder and lazily imports the model-dir/override
modules. Tooling tests also run below this directory, but remain separate from
product UT accounting.
"""

import os
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("TORCHTITAN_DIR", os.path.expanduser("~/workspace/torchtitan")))
# Spawned workers must resolve this repository's tests package before upstream's.
sys.path.insert(0, str(_REPO))

# These features still use the pre-fold-batch trainer/optimizer interfaces.
# Keep their tests out of collection until the feature itself is migrated;
# skip marks cannot prevent imports of removed upstream symbols.
collect_ignore = [
    "ema",
    "extensions/experiment/cv_parallel",
    "extensions/torchtitan/experiments/graph_trainer",
    "patches/torchtitan/experiments/graph_trainer",
    # Qwen runtime migration is deferred. Upstream eagerly imports CUDA-only
    # CuTeDSL code, which is not installed in the NPU test image.
    "override/qwen3_5/test_gated_delta.py",
    # Keep model-compute suites deferred; migrated vision loaders run in CPU UT.
    "models/deepseek_v4_1/test_training_contract.py",
    "models/deepseek_v4_1/test_vision_attention.py",
    "models/deepseek_v4_1/test_vision_layout.py",
]


# Mixed modules retain their text cases; entries may name a whole vision class.
_DEFERRED_VISION_TESTS = {
    "models/qwen3_5/test_qwen3_5_torchvision_dependency.py": {
        "test_multimodal_dependency_and_package_discovery_are_explicit",
        "test_torchvision_native_image_ops_and_v2_transforms_are_available",
        "test_qwen3_5_adapter_uses_real_torchvision_without_compat_modules",
    },
    "models/deepseek_v4_1/test_baseline_contract.py": {
        "test_v41_moe_rides_the_common_stack_with_an_opt_in_vision_bias",
        "test_vision_bias_follows_the_factory_switch",
    },
    "models/deepseek_v4_1/test_engram.py": {
        "test_image_tokens_reset_hash_history_and_receive_no_residual_gradient",
    },
    "models/deepseek_v4_1/test_grain_metadata.py": {
        "test_multimodal_preprocess_annotates_vision_inputs",
    },
    "models/deepseek_v4_1/test_model_split.py": {
        "test_only_the_multimodal_half_carries_vision_parameters",
        "test_multimodal_rejects_context_parallel",
        "test_parallelize_asks_the_multimodal_stack_for_both_hooks",
        "test_multimodal_ac_hook_wraps_every_vision_block",
    },
    "models/deepseek_v4_1/test_rms_norm.py": {
        "test_override_reaches_vision_and_preserves_parameter_names",
    },
    "models/deepseek_v4_1/test_state_dict_adapter.py": {
        "TestVisionAdapterOwnership",
        "TestComposedAdapterPartition",
        "TestAdapterPartitionIsTotal",
    },
    "models/deepseek_v4_1/test_v41_lora.py": {
        "test_multimodal_lora_updates_adapters_with_frozen_vision_and_engram",
    },
    "patches/torchtitan/models/common/test_hash_moe_counts.py": {
        "test_moe_forwards_image_mask_to_router",
    },
}


# DeepSeek-V3.2 still forwards the pre-fold ``[B, L, ...]`` layout against the
# folded pinned upstream. This test crosses the unmigrated seam: the decoder
# config's context-length knob. Match the integration runner's dsv3_2 deferral
# until the folded-token migration lands.
_DEFERRED_DSV32_FOLD_TESTS = {
    "models/test_mtp_cp_chunk_loss.py": {
        "test_dsv32_mtp_keeps_community_cp_restriction",
    },
}


def pytest_collection_modifyitems(items):
    for item in items:
        if not item.path.is_relative_to(_REPO / "tests/unit_tests"):
            continue
        relative_path = item.path.relative_to(_REPO / "tests/unit_tests").as_posix()
        vision_names = _DEFERRED_VISION_TESTS.get(relative_path, ())
        is_vision = any(getattr(node, "originalname", node.name) in vision_names for node in item.listchain())
        if relative_path.startswith("models/deepseek_v4_1/"):
            params = getattr(getattr(item, "callspec", None), "params", {})
            is_vision |= (
                params.get("vision") is True
                or params.get("multimodal") is True
                or params.get("modality") == "multimodal"
                or (
                    relative_path == "models/deepseek_v4_1/test_v41_lora.py"
                    and params.get("flavor") == "deepseek_v4_1_debugmodel"
                )
                or getattr(params.get("factory"), "__name__", "").endswith("_multimodal")
                or params.get("image_mask") is not None
            )
            if relative_path == "models/deepseek_v4_1/test_router.py":
                settings = params.get("settings", {})
                # Hash routing discards the vision flag and uses token IDs.
                is_vision |= settings.get("vision_enabled", False) and not settings.get("hash", False)
            if (relative_path, getattr(item, "originalname", item.name)) in {
                (
                    "models/deepseek_v4_1/test_migration_config.py",
                    "test_debug_recipe_builds_with_token_budget_and_preserves_embedding_weights",
                ),
                (
                    "models/deepseek_v4_1/test_v41_lora.py",
                    "test_lora_recipe_preserves_vision_and_uses_standard_adamw_and_peft",
                ),
            }:
                # These config-only nodes use the migrated target contracts.
                is_vision = False
        if is_vision:
            item.add_marker(pytest.mark.skip(reason="vision migration is outside the current text migration"))
        if getattr(item, "originalname", item.name) in _DEFERRED_DSV32_FOLD_TESTS.get(relative_path, ()):
            item.add_marker(
                pytest.mark.skip(reason="DeepSeek-V3.2 model boundary still requires the folded-token migration")
            )
        # These legacy suites still consume removed optimizer interfaces.
        # Do not hide new Muon behavior tests by matching their names globally.
        migrated_muon_config_test = (
            relative_path == "models/common/test_lora_muon_profile.py"
            and getattr(item, "originalname", item.name)
            == "test_actual_lora_recipe_muon_groups_cover_adapters_with_layouts_and_unique_buckets"
        )
        if (
            not relative_path.startswith("config/")
            and not migrated_muon_config_test
            and (
                (
                    relative_path
                    in {
                        "extensions/cpu_offload/test_no_optimizer_state.py",
                        "override/common/test_swap_optimizer.py",
                        "models/common/test_lora_muon_profile.py",
                    }
                    and "muon" in item.name.lower()
                )
                or item.name.startswith(
                    (
                        "test_lora_training_freezes_engram_and_native_dcp_resumes",
                        "test_v41_adapter_only_resume_restores_engram_and_next_step",
                    )
                )
            )
        ):
            item.add_marker(pytest.mark.skip(reason="checkpoint/Muon migration is pending"))


# ---------------------------------------------------------------------------
# The NPU-bound seam: a fake ``cann_ops_transformer`` call recorder.
# Everything else in the plugin imports against the real torchtitan checkout
# with the patches applied; only the CANN op surface is untestable on CPU.
# ``install()`` replaces the module (and its ``ops`` submodule) in
# ``sys.modules`` and injects the missing ``torch.ops.cann_ops_transformer``
# attributes with the recorder: every call is appended to ``ct.calls`` as
# ``(fn_name, args, kwargs)``.  No real ``cann_ops_transformer`` package is
# required. Normal product collection installs the recorder in
# ``pytest_configure``; tooling-only collection skips it so repository tooling
# sees its real optional imports.
# ---------------------------------------------------------------------------

_FAKE_FUNCTIONS = (
    "sparse_flash_mla",
    "sparse_flash_mla_grad",
    "sparse_flash_mla_metadata",
    "sparse_flash_mla_grad_metadata",
    "lightning_indexer",
    "lightning_indexer_metadata",
    "sparse_lightning_indexer_kl_loss_grad",
    "sparse_lightning_indexer_kl_loss_grad_metadata",
)


# Resolved at import time via ``torch.ops.cann_ops_transformer.*`` (the
# ``_ASC_SPARSEATTN_HOOK`` bundle); they are never invoked on CPU.
_TORCH_OPS_FUNCTIONS = (
    "lightning_indexer",
    "lightning_indexer_metadata",
    "sparse_flash_mla",
    "sparse_flash_mla_metadata",
    "sparse_flash_mla_grad",
    "sparse_flash_mla_grad_metadata",
    "sparse_lightning_indexer_kl_loss_grad",
    "sparse_lightning_indexer_kl_loss_grad_metadata",
)

# The two partial-RoPE mutator ops: unlike the recorder functions above they
# carry ``(Tensor(a!)) -> ()`` schemas and really run on CPU during the rope
# tests (the test module registers CPU kernels emulating the CANN math), so
# the mock must define them with the exact native signatures whenever the
# real package did not.  The mHC ops follow the same rule: the fused mHC
# modules register autograd formulas for them, which requires the ops to
# exist even though the CPU tests never execute them.
_TORCH_OPS_MUTATOR_SCHEMAS = (
    (
        "inplace_partial_rotary_mul",
        "inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, *, "
        'str rotary_mode="interleave", int[2] partial_slice=[0, 0]) -> ()',
    ),
    (
        "inplace_partial_rotary_mul_backward",
        "inplace_partial_rotary_mul_backward(Tensor(a!) grad_output, Tensor r1, Tensor r2, *, "
        'str rotary_mode="interleave", int[2] partial_slice=[0, 0]) -> ()',
    ),
    (
        "mhc_post",
        "mhc_post(Tensor x, Tensor? h_res, Tensor h_out, Tensor h_post) -> Tensor",
    ),
    (
        "mhc_pre_sinkhorn",
        "mhc_pre_sinkhorn(Tensor x, Tensor phi, Tensor alpha, Tensor bias, int hc_mult, "
        "int num_iters, float hc_eps, float norm_eps, bool out_flag) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
    ),
)


def _fake_cann_ops():
    import types

    ct = types.ModuleType("cann_ops_transformer")
    ct.calls = []

    def _make(fn_name):
        def _call(*args, **kwargs):
            ct.calls.append((fn_name, args, kwargs))
            return torch.empty((1024,), dtype=torch.int32)

        _call.__name__ = fn_name
        return _call

    for fn_name in _FAKE_FUNCTIONS:
        setattr(ct, fn_name, _make(fn_name))

    ct.ops = types.ModuleType("cann_ops_transformer.ops")
    for fn_name in _FAKE_FUNCTIONS:
        setattr(ct.ops, fn_name, _make(fn_name))

    # Registration submodule for the fused partial-RoPE mutator: the ops
    # module imports ``cann_ops_transformer.ops.inplace_partial_rotary_mul``
    # at module level, so a bare importable module keeps the CPU suite green.
    ct.inplace_partial_rotary_mul_module = types.ModuleType("cann_ops_transformer.ops.inplace_partial_rotary_mul")
    ct.inplace_partial_rotary_mul_module.__path__ = []
    # ``torchtitan_npu.ops.ascendc.mhc`` imports this submodule for its op
    # registration side effect; an empty module satisfies that import.
    ct.mhc_post_backward_module = types.ModuleType("cann_ops_transformer.ops.mhc_post_backward")
    ct.mhc_post_backward_module.__path__ = []
    return ct


_INSTALLED = False


def install():
    """Replace ``cann_ops_transformer`` with the call recorder (once).

    Covers the Python module surface (``from cann_ops_transformer import
    ...`` and ``from cann_ops_transformer.ops import ...``) and the
    ``torch.ops.cann_ops_transformer`` namespace that fused modules resolve
    at import time, so the CPU tests need no real CANN dependency.
    Attributes already present in the ``torch.ops`` namespace (e.g. after a
    real package import earlier in the process) are left untouched.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    recorder = _fake_cann_ops()
    sys.modules["cann_ops_transformer"] = recorder
    sys.modules["cann_ops_transformer.ops"] = recorder.ops
    sys.modules["cann_ops_transformer.ops.inplace_partial_rotary_mul"] = recorder.inplace_partial_rotary_mul_module
    sys.modules["cann_ops_transformer.ops.mhc_post_backward"] = recorder.mhc_post_backward_module
    ns = torch.ops.cann_ops_transformer
    for fn_name in _TORCH_OPS_FUNCTIONS:
        if not hasattr(ns, fn_name):
            setattr(ns, fn_name, getattr(recorder, fn_name))
    # Define the partial-RoPE mutator ops (real signatures) when the real
    # package has not registered them; keep the library handle alive on the
    # recorder so the schemas stay registered for the process lifetime.
    missing = [name for name, _ in _TORCH_OPS_MUTATOR_SCHEMAS if not hasattr(ns, name)]
    if missing:
        lib = torch.library.Library("cann_ops_transformer", "FRAGMENT")
        for name, schema in _TORCH_OPS_MUTATOR_SCHEMAS:
            if name in missing:
                lib.define(schema)
        recorder._fragment_lib = lib


def _requested_only_tooling(config):
    """Return whether pytest was asked to collect only tooling tests.

    A tooling-only invocation should not install the product-suite CANN fake:
    importing a repository parser must see the real optional dependencies and
    must not inherit product test state.  The normal ``pytest tests/unit_tests``
    invocation still installs the fake before collection of product modules.
    """
    args = [str(arg).split("::", 1)[0] for arg in config.args if not str(arg).startswith("-")]
    return bool(args) and all("tests/unit_tests/tooling" in Path(arg).as_posix() for arg in args)


def pytest_configure(config):
    if not _requested_only_tooling(config):
        install()


@pytest.fixture(scope="module")
def single_rank_spmd():
    """Provide real CPU mesh axes only to model tests that explicitly request them."""
    from tests.unit_tests.models.spmd_test_utils import single_rank_spmd_context

    with single_rank_spmd_context() as context:
        yield context


@pytest.fixture(scope="module")
def dsv4():
    """Install the ``cann_ops_transformer`` recorder and import the
    model-dir/override modules (real torchtitan + the applied patches).

    Module-scoped so the recorder and the imports are shared within a test
    module; ``ct.calls`` isolation is the test modules' own concern.
    """
    install()
    import importlib

    ns = types.SimpleNamespace()
    ns.metadata = importlib.import_module("torchtitan_npu.models.deepseek_v4.metadata")
    ns.attention = importlib.import_module("torchtitan_npu.models.deepseek_v4.attention")
    ns.compressor = importlib.import_module("torchtitan_npu.models.deepseek_v4.compressor")
    ns.cann_ops = importlib.import_module("cann_ops_transformer")
    return ns
