# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import grain
import pytest
from torchtitan.components.data import DatasetBuildContext

from tests.unit_tests.models.common.text_data_utils import IntegerTextTokenizer
from torchtitan_npu.models.common.alignment import validate_alignment


@pytest.mark.parametrize("context_length,budget,valid", [(128, 256, True), (127, 256, False), (128, 255, False)])
def test_alignment_reads_native_microbatch_context(context_length, budget, valid):
    context = DatasetBuildContext(
        tokenizer=IntegerTextTokenizer(),
        max_context_length=context_length,
        num_tokens_per_microbatch=budget,
        read_options=grain.ReadOptions(),
    )
    if valid:
        validate_alignment(128, context=context)
    else:
        with pytest.raises(ValueError, match="multiples of per_doc_alignment"):
            validate_alignment(128, context=context)
