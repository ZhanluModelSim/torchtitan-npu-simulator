# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise launcher argument and process-environment contracts without training."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tooling
ROOT = Path(__file__).resolve().parents[3]


def _startup(tmp_path):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_train.sh", "run_train_multinodes.sh"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    binary = tmp_path / "bin"
    binary.mkdir()
    capture = tmp_path / "capture.json"
    program = tmp_path / "capture.py"
    program.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "keys = ['MODULE', 'CONFIG', 'NGPU', 'NNODES', 'WORLD_SIZE', "
        "'CPU_AFFINITY_CONF', 'HCCL_CONNECT_TIMEOUT', 'CANN_TEST_VALUE']\n"
        "Path(os.environ['CAPTURE_PATH']).write_text(json.dumps({"
        "'argv': sys.argv[1:], 'env': {key: os.environ.get(key) for key in keys}}))\n"
        "sys.exit(int(os.environ.get('CAPTURE_EXIT', '0')))\n"
    )
    for name in ("python3", "torchrun"):
        executable = binary / name
        executable.write_text(
            f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(program))} "$@"\n'
        )
        executable.chmod(0o755)
    cann = tmp_path / "cann.sh"
    cann.write_text("export CANN_TEST_VALUE=active\n")
    env = os.environ.copy()
    for name in (
        "NODE_IPS",
        "NNODES",
        "NODE_RANK",
        "MASTER_ADDR",
        "WORLD_SIZE",
        "COMM_BACKEND",
        "COMM_MODE",
        "LOCAL_HOST",
        "Network_Interface",
    ):
        env.pop(name, None)
    env.update(
        PATH=str(binary) + os.pathsep + env["PATH"],
        ASCEND_SET_ENV_PATH=str(cann),
        CAPTURE_PATH=str(capture),
        MODULE="my_recipes",
        CONFIG="my_config",
        NGPU="2",
    )
    return project, env, capture, program


@pytest.mark.parametrize("exit_code", [0, 19], ids=["completed", "training-failed"])
def test_single_node_launcher_preserves_native_arguments_environment_and_exit(tmp_path, exit_code):
    project, env, capture, _ = _startup(tmp_path)
    env.update(CAPTURE_EXIT=str(exit_code), CPU_AFFINITY_CONF="custom-affinity", HCCL_CONNECT_TIMEOUT="777")
    extra = ["--output-dir", "an output folder", "--override", 'my_override.factory={"pattern":"a b"}']

    result = subprocess.run(["bash", str(project / "scripts/run_train.sh"), *extra], env=env, capture_output=True)

    assert result.returncode == exit_code
    observed = json.loads(capture.read_text())
    assert observed["argv"][-len(extra) :] == extra
    assert "--nproc_per_node=2" in observed["argv"]
    assert observed["argv"][observed["argv"].index("--module") + 1] == "my_recipes"
    assert observed["argv"][observed["argv"].index("--config") + 1] == "my_config"
    assert observed["env"]["CPU_AFFINITY_CONF"] == "custom-affinity"
    assert observed["env"]["HCCL_CONNECT_TIMEOUT"] == "777"
    assert observed["env"]["CANN_TEST_VALUE"] == "active"


def test_single_node_launcher_forwards_print_config_without_switching_to_multinode(tmp_path):
    project, env, capture, _ = _startup(tmp_path)
    env.update(NGPU="8", NNODES="16", NODE_IPS="192.0.2.1,192.0.2.2")

    subprocess.run(["bash", str(project / "scripts/run_train.sh"), "--print-config"], env=env, check=True)

    observed = json.loads(capture.read_text())
    assert "--nproc_per_node=8" in observed["argv"]
    assert not any(arg.startswith("--nnodes=") for arg in observed["argv"])
    assert observed["argv"][-1] == "--print-config"
    assert observed["env"]["WORLD_SIZE"] is None


@pytest.mark.parametrize("arguments", [[], ["--print-config"]], ids=["training", "print-config"])
def test_fake_communication_uses_the_native_backend_option(tmp_path, arguments):
    project, env, capture, _ = _startup(tmp_path)
    env["COMM_BACKEND"] = "fake"

    subprocess.run(["bash", str(project / "scripts/run_train.sh"), *arguments], env=env, check=True)

    observed = json.loads(capture.read_text())
    assert observed["argv"] == [
        "-m",
        "torchtitan_npu.train",
        "--module",
        "my_recipes",
        "--config",
        "my_config",
        "--comm-backend",
        "fake",
        *arguments,
    ]


def test_multi_node_startup_keeps_topology_separate_from_the_recipe(tmp_path):
    project, env, capture, _ = _startup(tmp_path)
    env.update(
        NGPU="4", NNODES="2", MASTER_ADDR="192.0.2.1", NODE_RANK="1", LOCAL_HOST="192.0.2.2", Network_Interface="eth0"
    )

    subprocess.run(
        ["bash", str(project / "scripts/run_train_multinodes.sh"), "--resume-step", "2"], env=env, check=True
    )

    observed = json.loads(capture.read_text())
    for option in ("--nnodes=2", "--node_rank=1", "--nproc_per_node=4", "--master_addr=192.0.2.1"):
        assert option in observed["argv"]
    assert observed["argv"][-2:] == ["--resume-step", "2"]
    assert observed["env"]["MODULE"] == "my_recipes"
    assert observed["env"]["CONFIG"] == "my_config"


def test_multi_node_rank_detection_uses_the_selected_network_interface(tmp_path):
    project, env, capture, _ = _startup(tmp_path)
    interface = tmp_path / "bin/ip"
    interface.write_text('#!/usr/bin/env bash\necho "    inet 192.0.2.2/24 scope global training"\n')
    interface.chmod(0o755)
    env.update(Network_Interface="training", NODE_IPS="192.0.2.1,192.0.2.2")

    subprocess.run(["bash", str(project / "scripts/run_train_multinodes.sh")], env=env, check=True)

    observed = json.loads(capture.read_text())
    assert "--nnodes=2" in observed["argv"]
    assert "--node_rank=1" in observed["argv"]


@pytest.mark.parametrize(
    "script", sorted((ROOT / "examples").rglob("*.sh")), ids=lambda path: str(path.relative_to(ROOT / "examples"))
)
def test_example_wrapper_selects_its_recipe_and_only_forwards_native_launch_arguments(tmp_path, script):
    project, env, capture, program = _startup(tmp_path)
    target = project / script.relative_to(ROOT)
    target.parent.mkdir(parents=True)
    shutil.copy2(script, target)
    launcher_name = "run_train_multinodes.sh" if "run_train_multinodes.sh" in script.read_text() else "run_train.sh"
    launcher = project / "scripts" / launcher_name
    launcher.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(program))} "$@"\n')
    env.pop("MODULE")
    env.pop("CONFIG")
    extra = ["--output-dir", "a folder with spaces", "--override", 'my_override.factory={"pattern":"a b"}']

    subprocess.run(["bash", str(target), *extra], env=env, check=True, cwd=tmp_path)

    observed = json.loads(capture.read_text())
    assert observed["argv"] == extra
    assert observed["env"]["MODULE"].startswith("torchtitan_npu_recipes.examples.")
    assert observed["env"]["CONFIG"] in {script.stem, f"{script.stem}_eager"}


@pytest.mark.parametrize(
    "script",
    sorted(path for path in (ROOT / "examples").rglob("*.sh") if "NODE_IPS" in path.read_text()),
    ids=lambda path: str(path.relative_to(ROOT / "examples")),
)
def test_multi_node_examples_derive_default_node_count_from_addresses(tmp_path, script):
    project, env, capture, program = _startup(tmp_path)
    target = project / script.relative_to(ROOT)
    target.parent.mkdir(parents=True)
    shutil.copy2(script, target)
    launcher_name = "run_train_multinodes.sh" if "run_train_multinodes.sh" in script.read_text() else "run_train.sh"
    launcher = project / "scripts" / launcher_name
    launcher.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(program))} "$@"\n')
    env.update(NODE_IPS="192.0.2.1,192.0.2.2")

    subprocess.run(["bash", str(target), "--print-config"], env=env, check=True)

    assert json.loads(capture.read_text())["env"]["NNODES"] == "2"
