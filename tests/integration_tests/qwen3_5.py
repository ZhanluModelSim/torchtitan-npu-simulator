# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.integration_tests import OverrideDefinitions

# The Qwen3.5 launcher supplies the tokenizer and cc12m-test defaults.  Keep
# only the common metrics switches here so the suite exercises that launcher
# exactly as users do.
QWEN3_5_TRAIN_ARGS = (
    "--metrics.enable_tensorboard",
    "--metrics.log_freq=1",
    "--metrics.save_tb_folder=tb",
    "--training.disable-cuda-graphs",
)


def _build_case(
    *,
    test_name: str,
    test_descr: str,
    ngpu: int,
    extra_args: tuple[str, ...],
    disabled: bool = False,
) -> OverrideDefinitions:
    return OverrideDefinitions(
        override_args=[extra_args],
        test_descr=test_descr,
        test_name=test_name,
        ngpu=ngpu,
        disabled=disabled,
        env_vars={
            "MODULE": "torchtitan_npu_recipes.models.qwen3_5",
            "CONFIG": "qwen35_debugmodel",
        },
        use_golden=False,
        check_loss=False,
        train_script="examples/qwen3_5/qwen3_5_debugmodel_1p_cpt_512.sh",
        train_args=QWEN3_5_TRAIN_ARGS,
    )


def build_qwen3_5_test_list() -> list[OverrideDefinitions]:
    """Return the runnable Qwen3.5 NPU smoke case.

    The multimodal stack (torchvision and the torchtitan tokenizer and
    cc12m-test assets) is installed by .ci/smoke_test.sh in the smoke
    runner environment.
    """
    common = (
        "--training.steps=1",
        "--training.num-tokens-per-microbatch-per-dp-rank=512",
        "--training.num-tokens-per-train-step=512",
        "--training.max-context-length=512",
    )
    return [
        _build_case(
            test_name="qwen3_5_debugmodel_1rank",
            test_descr="Qwen3.5 debug model 1rank",
            ngpu=1,
            extra_args=common,
        ),
        OverrideDefinitions(
            override_args=[()],
            test_descr="Qwen packed text SFT on CP2 and FSDP2",
            test_name="qwen3_5_packed_sft_cp2",
            ngpu=4,
            env_vars={"MODULE": "tests.integration_tests.qwen3_5", "CONFIG": "packed_sft_debug"},
            use_golden=False,
            check_loss=False,
            python_config=True,
            expected_steps=((1,),),
        ),
    ]


def packed_sft_debug():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from torchtitan.components.data import FirstFitPackingConfig
    from transformers import PreTrainedTokenizerFast

    from torchtitan_npu.models.qwen3_5.dataloader import SFTGrainDataLoader, raw_sft_dataset
    from torchtitan_npu_recipes.models.qwen3_5 import _text_sft_config

    assets = TemporaryDirectory(prefix="qwen-sft-smoke-")
    atexit.register(assets.cleanup)
    root = Path(assets.name)
    vocab = {
        token: index
        for index, token in enumerate(("[UNK]", "<|im_end|>", "<|im_start|>", "user", "assistant", "answer"))
    }
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="<|im_end|>")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    tokenizer.chat_template = (
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
        "{%- endfor %}"
    )
    tokenizer.save_pretrained(root)
    data = root / "train.jsonl"
    row = {"messages": [{"role": "user", "content": "answer"}, {"role": "assistant", "content": "answer"}]}
    data.write_text((json.dumps(row) + "\n") * 8)
    dataset, contract = raw_sft_dataset(dataset_path=str(data))
    loader = SFTGrainDataLoader.Config(
        dataset=FirstFitPackingConfig(dataset=dataset, num_packing_bins=1), input_contract=contract, max_num_documents=8
    )
    config = _text_sft_config(
        flavor="debugmodel", length=128, cp=2, dp_shard=2, steps=1, load_checkpoint=False, dataloader=loader
    )
    config.hf_assets_path = str(root)
    config.checkpointer = None
    config.metrics.enable_tensorboard = True
    config.metrics.save_tb_folder = "tb_phase_0"
    return config
