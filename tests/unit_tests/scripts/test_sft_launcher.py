# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "examples/deepseek_v4/sft_deepseek_v4_flash_16k_A3.sh"


@pytest.mark.parametrize("ngpu", ["8", "16"])
def test_sft_launcher_executes_real_script_and_isolates_subcommands(tmp_path, ngpu):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_args = tmp_path / "args.txt"
    capture_ngpu = tmp_path / "ngpu.txt"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
        'printf "%s" "${NGPU-}" > "$CAPTURE_NGPU"\n',
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "CAPTURE_ARGS": str(capture_args),
            "CAPTURE_NGPU": str(capture_ngpu),
            "NGPU": ngpu,
            "NODE_IPS": "127.0.0.1",
            "LOCAL_HOST": "127.0.0.1",
            "NODE_RANK": "0",
            "NNODES": "1",
            "Network_Interface": "lo",
            "DATASET_PATH": "/tmp/tau",
            "DATA_FILES": "demo.parquet",
            "DATASET_CONFIG_NAME": "default",
            "CHAT_PROCESSOR": "torchtitan_npu.hf_datasets.chat_processors.process_tau_sample",
            "HF_ASSETS_PATH": "/tmp/assets",
            "CHECKPOINT_INITIAL_LOAD_PATH": "/tmp/checkpoint",
            "ENCODING_MODULE_PATH": "/tmp/encoding_dsv4.py",
        }
    )

    subprocess.run(
        ["/bin/bash", str(LAUNCHER), "--training.steps", "7"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    captured_args = capture_args.read_text(encoding="utf-8").splitlines()
    assert captured_args[0] == "scripts/run_train_multinodes.sh"
    assert capture_ngpu.read_text(encoding="utf-8") == ngpu

    trainer_override_index = max(
        index for index, argument in enumerate(captured_args) if argument == "--training.steps"
    )
    dataloader_subcommand_index = captured_args.index("dataloader:chat_data_loader_config")
    assert trainer_override_index < dataloader_subcommand_index
    assert captured_args[trainer_override_index + 1] == "7"
    assert "--dataloader.dataset_config_name" in captured_args
    assert "default" in captured_args
    assert "--dataloader.chat_processor" in captured_args
    assert "torchtitan_npu.hf_datasets.chat_processors.process_tau_sample" in captured_args
    assert "dataloader.chat_encoder:dsv4_encoder_config" in captured_args

    source_lines = LAUNCHER.read_text(encoding="utf-8").splitlines()
    at_index = source_lines.index('  "$@"')
    next_nonempty = next(line.strip() for line in source_lines[at_index + 1 :] if line.strip())
    assert next_nonempty.startswith("#")
    assert "top-level cli" in next_nonempty.lower()
    assert "subcommand" in next_nonempty


def test_all_multinode_example_launchers_use_ngpu():
    callers = sorted(Path(REPO_ROOT / "examples").rglob("*.sh"))
    multi_node_callers = [
        path for path in callers if "run_train_multinodes.sh" in path.read_text(encoding="utf-8")
    ]

    assert multi_node_callers
    for path in multi_node_callers:
        source = path.read_text(encoding="utf-8")
        assert "NPUS_PER_NODE=" not in source, path
        assert "NGPU=" in source, path


def test_multinode_runner_maps_ngpu_to_nproc_per_node(tmp_path):
    capture_args = tmp_path / "torchrun_args.txt"
    fake_torchrun = tmp_path / "torchrun"
    fake_ps = tmp_path / "ps"
    fake_kill = tmp_path / "kill"
    fake_torchrun.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$CAPTURE_TORCHRUN_ARGS"\n',
        encoding="utf-8",
    )
    fake_ps.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_kill.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_torchrun.chmod(0o755)
    fake_ps.chmod(0o755)
    fake_kill.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}{os.pathsep}{env['PATH']}",
            "CAPTURE_TORCHRUN_ARGS": str(capture_args),
            "NGPU": "16",
            "NODE_IPS": "127.0.0.1",
            "LOCAL_HOST": "127.0.0.1",
            "NODE_RANK": "0",
            "NNODES": "1",
            "Network_Interface": "lo",
        }
    )
    env.pop("NPUS_PER_NODE", None)

    subprocess.run(
        ["/bin/bash", str(REPO_ROOT / "scripts/run_train_multinodes.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--nproc_per_node=16" in capture_args.read_text(encoding="utf-8").splitlines()
