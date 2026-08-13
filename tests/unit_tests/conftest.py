"""Shared pytest fixtures for the DSV4 CPU unit tests.

The plugin's model-dir and override host logic import against the real
torchtitan checkout (env ``TORCHTITAN_DIR`` or the default) with the
plugin's patches applied by the package chain.  The only faked surface is
``cann_ops_transformer`` (the NPU op boundary): the ``dsv4`` fixture
installs the call recorder and lazily imports the model-dir/override
modules.
"""

import os
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(
    0, os.environ.get("TORCHTITAN_DIR", os.path.expanduser("~/workspace/torchtitan"))
)


# ---------------------------------------------------------------------------
# The NPU-bound seam: a fake ``cann_ops_transformer`` call recorder.
# Everything else in the plugin imports against the real torchtitan checkout
# with the patches applied; only the CANN op surface is untestable on CPU.
# ``install()`` replaces the module in ``sys.modules`` with a recorder:
# every call is appended to ``ct.calls`` as ``(fn_name, args, kwargs)``.
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
    return ct


_INSTALLED = False


def install():
    """Replace ``cann_ops_transformer`` with the call recorder (once)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    sys.modules["cann_ops_transformer"] = _fake_cann_ops()


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
    ns.token_dispatcher = importlib.import_module(
        "torchtitan_npu.models.deepseek_v4.token_dispatcher"
    )
    ns.reference = importlib.import_module(
        "torchtitan_npu.models.deepseek_v4.reference"
    )
    ns.attention = importlib.import_module(
        "torchtitan_npu.models.deepseek_v4.attention"
    )
    ns.compressor = importlib.import_module(
        "torchtitan_npu.models.deepseek_v4.compressor"
    )
    ns.golden = importlib.import_module(
        "torchtitan_npu.override.deepseek_v4.sparse_attn.golden"
    )
    spec = importlib.util.spec_from_file_location(
        "varlen_cp_backport",
        _REPO
        / "torchtitan_npu"
        / "patches"
        / "torchtitan"
        / "distributed"
        / "varlen_cp.py",
    )
    varlen_cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(varlen_cp)
    ns.CPVarlenMetadata = varlen_cp.CPVarlenMetadata
    ns.cann_ops = importlib.import_module("cann_ops_transformer")
    return ns
