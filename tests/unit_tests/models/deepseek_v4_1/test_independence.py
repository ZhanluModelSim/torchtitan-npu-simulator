# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 dependency and import-side-effect boundaries."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


# The isolated subprocess must be self-contained without a CANN stack:
# ``override.common.rope`` loads the fused partial-RoPE wrapper lazily on
# first call, so the model packages import with ``cann_ops_transformer``
# entirely absent.  A blocker hook in each subprocess *rejects* that import
# (no fake package), proving the boundary instead of merely running on a
# machine that happens to lack CANN.
_ISOLATED_PRELUDE = """\
import sys


class _NoCannOps:
    def find_spec(self, name, path=None, target=None):
        if name == "cann_ops_transformer" or name.startswith("cann_ops_transformer."):
            raise ImportError(f"cann_ops_transformer must not be imported by the model packages: {name}")
        return None


sys.meta_path.insert(0, _NoCannOps())
"""


def _run_isolated(code):
    # Pass a torchtitan source checkout through explicitly when the host uses
    # one (the pytest conftest would otherwise do it); never import the
    # product conftest here — it installs the fake CANN recorder.
    torchtitan_dir = os.environ.get("TORCHTITAN_DIR", "")
    pythonpath = os.pathsep.join(part for part in (str(REPO), torchtitan_dir) if part)
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATED_PRELUDE + code],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OMP_NUM_THREADS": "1", "PYTHONPATH": pythonpath},
        timeout=180,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"


def test_v41_builds_and_runs_without_v4():
    _run_isolated(r"""
import sys

class Blocker:
    blocked = ("torchtitan_npu.models.deepseek_v4", "torchtitan_npu.override.deepseek_v4")

    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in self.blocked):
            raise ImportError(f"independence blocker: {name}")
        return None

sys.meta_path.insert(0, Blocker())
import importlib
import torch
registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
make_config = registry._make_v41_config
def tiny_config(**kwargs):
    kwargs.update(dim=8, n_heads=2, head_dim=8, rope_head_dim=4,
        q_lora_rank=8, o_lora_rank=4, n_groups=1, index_n_heads=2,
        index_head_dim=4, moe_inter_dim=16, vision_dim=8, vision_heads=2, vision_inter_dim=16)
    return make_config(**kwargs)
registry._make_v41_config = tiny_config

cfg = registry.model_registry("deepseek_v4_1_debugmodel").model
cfg.vocab_size = cfg.tok_embeddings.num_embeddings = cfg.lm_head.out_features = 64
from torchtitan_npu.models.deepseek_v4_1.state_dict_adapter import DeepSeekV41StateDictAdapter
from torchtitan_npu.models.deepseek_v4_1.config_registry import deepseek_v4_1_debugmodel_text

adapter = DeepSeekV41StateDictAdapter(cfg, None)
recipe = deepseek_v4_1_debugmodel_text()
assert recipe.model_spec.model.layers
assert not any(name == blocked or name.startswith(blocked + ".")
               for name in sys.modules for blocked in Blocker.blocked)
from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import preprocess
from tests.unit_tests.models.mtp_test_utils import build_cpu_model
model = build_cpu_model(cfg)
tokens = torch.arange(128).remainder(32).unsqueeze(0)
_, _, kwargs = preprocess(model, tokens, tokens, torch.arange(128).unsqueeze(0))
out = model(tokens, **kwargs)
assert torch.isfinite(out).all()
out.square().mean().backward()
assert len(model.layers) == 40
for layer in model.layers.values():
    assert layer.moe.tokens_per_expert_E.sum() == 128 * 6
    assert layer.moe.routed_experts.inner_experts.w1_EFD.grad is not None
""")


def test_v41_import_does_not_patch_v4():
    _run_isolated(r"""
import importlib
import sys
import torch.nn.functional as functional
v4 = importlib.import_module("torchtitan_npu.models.deepseek_v4")
moe = importlib.import_module("torchtitan.models.common.moe")
assert "torchtitan_npu.models.deepseek_v4_1" not in sys.modules

def identities():
    return (v4.attention.Attention, v4.attention.Attention.forward,
            v4.model.DeepSeekV4Model, v4.model.DeepSeekV4Model.forward,
            moe.MoE, moe.MoE.forward, moe.TokenChoiceTopKRouter,
            moe.TokenChoiceTopKRouter.forward, functional.cross_entropy)

def config_contract():
    cfg = v4.model_registry("debugmodel").model
    return [(type(layer.attention), type(layer.moe), type(layer.moe.router),
             layer.attention.compress_ratio, layer.moe.router.score_func)
            for layer in cfg.layers]

before, config_before = identities(), config_contract()
v41 = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
v41.model_registry("deepseek_v4_1_debugmodel")
assert identities() == before
assert config_contract() == config_before
""")


def test_common_rope_imports_without_cann():
    """``override.common`` stays importable (and constructible) with the CANN
    op package rejected: the fused partial-RoPE wrapper loads its CANN
    dependency lazily, so only *calling* the asc_partial variant requires the
    package — and that failure is loud, never a silent math fallback."""
    _run_isolated(
        "import torch\n"
        "from torchtitan_npu.override.common.rope import (\n"
        "    AscPartialComplexRoPE, WorkaroundComplexRoPE,\n"
        ")\n"
        "workaround = WorkaroundComplexRoPE(WorkaroundComplexRoPE.Config(dim=8, max_seq_len=16))\n"
        "assert workaround.cache.shape == (2, 16, 8)\n"
        "fused = AscPartialComplexRoPE(AscPartialComplexRoPE.Config(dim=8, max_seq_len=16, split=4))\n"
        "assert fused.cache.shape == (2, 16, 8)\n"
        "try:\n"
        "    fused(torch.randn(1, 3, 2, 12), positions=torch.arange(3).unsqueeze(0))\n"
        "except ImportError as error:\n"
        "    assert 'cann_ops_transformer' in str(error), error\n"
        "else:\n"
        "    raise AssertionError('asc_partial forward must fail without the CANN package')\n"
    )
