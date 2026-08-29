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

# Imported via ``cann_ops_transformer.ops`` (the compile-pattern module).
_OPS_SUBMODULE_FUNCTIONS = (
    "inplace_partial_rotary_mul",
    *_FAKE_FUNCTIONS,
)

# Resolved at import time via ``torch.ops.cann_ops_transformer.*`` (the
# ``_ASC_SPARSEATTN_HOOK`` bundle); they are never invoked on CPU.
_TORCH_OPS_FUNCTIONS = (
    "lightning_indexer",
    "sparse_flash_mla",
    "sparse_flash_mla_grad",
    "sparse_lightning_indexer_kl_loss_grad",
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
    for fn_name in _OPS_SUBMODULE_FUNCTIONS:
        setattr(ct.ops, fn_name, _make(fn_name))
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
    ns = torch.ops.cann_ops_transformer
    for fn_name in _TORCH_OPS_FUNCTIONS:
        if not hasattr(ns, fn_name):
            setattr(ns, fn_name, getattr(recorder, fn_name))


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
    ns.reference = importlib.import_module("torchtitan_npu.models.deepseek_v4.reference")
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
