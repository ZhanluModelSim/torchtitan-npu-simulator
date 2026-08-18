# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The metadata extension: the single overridable post-process of the
model-built attention masks/metadata.

The model owns its per-batch metadata construction (``build_attention_masks``
on the Decoder — including its own context-parallel handling).  The
``metadata_extension`` config field carries the vendor-specific post-process
(the default is the identity): e.g. the AscendC override injects the
``*_metadata`` kernel tensors.  This replaces the removed ``mask_handler``
pattern: the model's build is the only metadata-handling seam, and the
extension only adds what the model cannot know (vendor kernel metadata).
"""

from dataclasses import dataclass

from torchtitan.config.configurable import Configurable


class MetadataExtension(Configurable):
    """The default identity post-process of the built attention masks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        window_size: int = 0
        """The sliding-window size (model-config constant), consumed by the
        vendor extension's kernel metadata fills."""

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, metadata):
        return metadata
