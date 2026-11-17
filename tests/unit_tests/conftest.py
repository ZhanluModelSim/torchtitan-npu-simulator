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
sys.path.insert(0, str(_REPO))
sys.path.insert(0, os.environ.get("TORCHTITAN_DIR", os.path.expanduser("~/workspace/torchtitan")))


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
    ns.token_dispatcher = importlib.import_module("torchtitan_npu.models.deepseek_v4.token_dispatcher")
    ns.attention = importlib.import_module("torchtitan_npu.models.deepseek_v4.attention")
    ns.compressor = importlib.import_module("torchtitan_npu.models.deepseek_v4.compressor")
    ns.golden = importlib.import_module("torchtitan_npu.override.deepseek_v4.sparse_attn.golden")
    spec = importlib.util.spec_from_file_location(
        "varlen_cp_backport",
        _REPO / "torchtitan_npu" / "patches" / "torchtitan" / "distributed" / "varlen_cp.py",
    )
    varlen_cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(varlen_cp)
    ns.CPVarlenMetadata = varlen_cp.CPVarlenMetadata
    ns.cann_ops = importlib.import_module("cann_ops_transformer")
    return ns
