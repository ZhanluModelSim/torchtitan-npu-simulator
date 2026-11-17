# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchao.prototype.moe_training.utils import unwrap_weight

from torchao_npu.ops.block_mx_ops import (
    to_block_mx_then_bmm,
    to_block_mx_then_bmm_from_prequantized,
    to_block_mx_then_grouped_mm,
    to_block_mx_then_grouped_mm_from_prequantized,
    to_block_mx_then_linear_from_prequantized,
    to_block_mx_then_mm,
    to_block_mx_then_mm_from_prequantized,
)
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantization.quant_primitives.block_mx import (
    block_mx_quantize,
    from_block_quant_layout,
    mxfp4_to_block_mx,
    quantize_mxfp4_payload,
)
from torchao_npu.quantization.transform import register_parameter_swap_handler
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor
from torchao_npu.wrapper_tensors.base_wrapper_tensor import (
    BaseTrainingWeightWrapperTensor,
    _ops_to_preserve_subclass,
)

# Block MX quantizes in 32x32 blocks and packs two adjacent block scales along
# the quantized dim (scale trailing pack dim of 2). When FSDP shards the
# quantized dim, local-quantize + all-gather reconstructs the global quantized
# weight correctly only when each shard is a multiple of the scale packing
# granularity (2 * 32 = 64): a shard aligned to 32 but not 64 rounds its own
# scale count up, so the gathered K-dim scale has more rows than the
# globally-quantized one and the ``npu_quant_matmul`` meta check fails with
# "k dimension of scale and pertoken_scale must be equal". Sharding another
# dim (e.g. the expert axis under Shard(0)) keeps each local shard's quantized
# dim whole and cannot misalign, so the 64-alignment only applies to the dim
# that is actually sharded.

# MX quantization block size. The unsharded quantized dim (last dim) of a
# pre-quantized weight must be a multiple of it.
_BLOCK_SIZE = 32
_SCALE_PACK = 2

# The alignment an FSDP shard's quantized dim must satisfy for pre-quantize +
# all-gather to reconstruct the global scales exactly. The FP8 block path
# quantizes axis=-2; the MXFP4 communication path quantizes the stored last
# axis. The guard below is the single enforcement point: the shard-alignment
# contract depends on the actual FSDP sharding (dp_shard, cp folding, expert
# meshes), which config-stage validation deliberately does not model.
_PREQUANTIZE_SHARD_ALIGNMENT = _BLOCK_SIZE * _SCALE_PACK


class BlockMXTrainingWeightWrapperTensor(BaseTrainingWeightWrapperTensor):
    """Applies block MX quantized matmul on NPU.

    Performs a **real** block MX matmul using
    ``to_block_mx_then_mm`` / ``to_block_mx_then_grouped_mm``.

    When ``weight_config.mxfp4_fake_quantize_config`` is set, weights are
    pre-quantized to MXFP4 before the block MX matmul.

    When ``weight_config.fsdp_prequantize`` is set, the weight is quantized to
    block MX in :meth:`fsdp_pre_all_gather` (before all_gather) and the
    quantized weight + scales are all-gathered, so forward/backward reuse the
    pre-quantized data instead of re-quantizing on the fly. This reduces
    communication bandwidth and saves quantization compute. The wrapper's
    logical dtype stays BF16 (for correct gradient routing) while ``_data``
    holds the FP8 quantized weight.
    """

    # Pre-quantized payload carried by the wrapper: FP8 weight plus its N-dim /
    # K-dim block MX scales. Assigned in ``_from_prequantized`` /
    # ``_write_prequantized_to_out`` rather than ``__init__``.
    _data: torch.Tensor
    _scale_s1: torch.Tensor
    _scale_s2: torch.Tensor
    _data: torch.Tensor
    weight_config: BlockMXQuantizeConfig
    activation_config: MXQuantizeConfig

    def __init__(
        self,
        tensor: torch.Tensor,
        weight_config: BlockMXQuantizeConfig | None = None,
        activation_config: MXQuantizeConfig | None = None,
    ):
        if weight_config is None:
            raise ValueError(f"`weight_config` is required for {type(self).__name__}.")

        if activation_config is None:
            raise ValueError(f"`activation_config` is required for {type(self).__name__}.")

        if type(weight_config) is not BlockMXQuantizeConfig:
            raise ValueError(f"Only `BlockMXQuantizeConfig` is supported for `weight_config` in {type(self).__name__}.")

        if type(activation_config) is not MXQuantizeConfig:
            raise ValueError(f"Only `MXQuantizeConfig` is supported for `activation_config` in {type(self).__name__}.")

        super().__init__(tensor, weight_config=weight_config, activation_config=activation_config)

    # ------------------------------------------------------------------
    # Pre-quantized data helpers
    # ------------------------------------------------------------------

    def _has_prequantized_data(self) -> bool:
        """True if this wrapper carries pre-quantized FP8 data + scales."""
        return (
            hasattr(self, "_scale_s1")
            and hasattr(self, "_scale_s2")
            and self._data.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        )

    @classmethod
    def _from_prequantized(
        cls,
        B_q: torch.Tensor,
        B_s1: torch.Tensor,
        B_s2: torch.Tensor,
        weight_config: BlockMXQuantizeConfig,
        activation_config: MXQuantizeConfig,
        requires_grad: bool = False,
    ) -> "BlockMXTrainingWeightWrapperTensor":
        """Build a wrapper whose ``_data`` is the FP8 weight but whose logical
        dtype is BF16 (so autograd/FSDP produce a BF16 gradient).

        ``_make_wrapper_subclass`` is called directly with ``dtype=bfloat16``
        (rather than reusing ``__new__``, which would inherit the FP8 dtype),
        avoiding a full-size BF16 placeholder allocation.
        """
        obj = torch.Tensor._make_wrapper_subclass(
            cls,
            B_q.size(),
            strides=B_q.stride(),
            storage_offset=0,
            memory_format=torch.contiguous_format,
            dtype=torch.bfloat16,
            layout=B_q.layout,
            device=B_q.device,
            pin_memory=False,
            requires_grad=requires_grad,
        )
        obj._data = B_q
        obj._scale_s1 = B_s1
        obj._scale_s2 = B_s2
        obj.weight_config = weight_config  # pyrefly: ignore [missing-attribute]
        obj.activation_config = activation_config  # pyrefly: ignore [missing-attribute]
        return obj

    def _write_prequantized_to_out(
        self,
        out: torch.Tensor,
        B_q: torch.Tensor,
        B_s1: torch.Tensor,
        B_s2: torch.Tensor,
    ) -> None:
        """Write pre-quantized data into a pre-allocated ``out`` (bare wrapper
        or ``DTensor`` with a wrapper local tensor)."""
        local = out._local_tensor if isinstance(out, DTensor) else out
        if not isinstance(local, BlockMXTrainingWeightWrapperTensor):
            raise RuntimeError(
                f"expected out to be {type(self).__name__} or DTensor with "
                f"local_tensor={type(self).__name__}, but got {type(out)}"
            )
        if not all(hasattr(local, attr) for attr in ("_data", "_scale_s1", "_scale_s2")):
            raise RuntimeError("FSDP prequantized output wrapper is missing its payload tensors")
        if not all(isinstance(getattr(local, attr), torch.Tensor) for attr in ("_data", "_scale_s1", "_scale_s2")):
            raise RuntimeError("FSDP prequantized output wrapper has invalid payload tensors")

        # FSDP2 reuses a shallow copy of the unsharded wrapper on subsequent
        # iterations. Restore the current configs just like the base wrapper
        # path does, so a reused ``out`` cannot retain stale quantization state.
        local.weight_config = self.weight_config
        local.activation_config = self.activation_config
        targets = (local._data, local._scale_s1, local._scale_s2)
        sources = (B_q, B_s1, B_s2)
        for target, source in zip(targets, sources, strict=True):
            if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
                raise RuntimeError("FSDP prequantized output shape, dtype and device must remain unchanged")
        # FSDP retains the inner tensors returned by the first all-gather. Keep
        # their storage aliases and the Parameter's inner requires_grad flags:
        # replacing _data would discard those flags on subsequent gathers.
        # FP8 outputs normally already share the exact destination view; MXFP4
        # conversion produces separate buffers that must be copied back. This
        # restores the same parameter for backward, not an optimizer update.
        with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(targets):
            for target, source in zip(targets, sources, strict=True):
                if not target.is_set_to(source):
                    target.copy_(source)

    @staticmethod
    def _transpose_prequantized_scales(
        B_q: torch.Tensor,
        B_s1: torch.Tensor,
        B_s2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transpose the last two dims of a pre-quantized weight and its scales.

        Block MX quantizes a ``[..., K, N]`` weight into ``B_q`` plus an N-dim
        scale ``B_s1`` and a K-dim scale ``B_s2``. Transposing the weight to
        ``[..., N, K]`` swaps the roles of the two scales and transposes them:

        - new ``B_q``  = ``B_q.transpose(-1, -2)``
        - new ``B_s1`` (N-dim) = old K-dim scale ``B_s2.transpose(-2, -3)``
        - new ``B_s2`` (K-dim) = old N-dim scale ``B_s1.transpose(-2, -3)``
        """
        return (
            B_q.transpose(-1, -2),
            B_s2.transpose(-2, -3),
            B_s1.transpose(-2, -3),
        )

    @staticmethod
    def _reshape_prequantized_scales(
        B_s1: torch.Tensor,
        B_s2: torch.Tensor,
        orig_shape: torch.Size,
        new_shape: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape the block MX scales when a 2D pre-quantized weight is viewed
        as 3D by adding a leading batch dim (e.g. ``wo_a.weight.view(G, R, -1)``).

        A 2D weight ``[R, K]`` has scales:
        - ``B_s1`` (N-dim): ``[R, ceil(ceil(K/32)/2), 2]``
        - ``B_s2`` (K-dim): ``[ceil(ceil(R/32)/2), K, 2]``

        Viewing it as ``[G, R, K]`` (with ``G*R == R``) adds a leading batch dim,
        so the scales are reshaped to ``[G, R, ...]`` and ``[G, ...]`` respectively.
        """
        if len(orig_shape) != 2 or len(new_shape) != 3:
            raise RuntimeError(
                f"pre-quantized scale reshape only supports 2D->3D view, got {orig_shape} -> {new_shape}"
            )
        if orig_shape[0] != new_shape[0] * new_shape[1]:
            raise RuntimeError(f"2D->3D view must split the first dim: {orig_shape} -> {new_shape}")
        g = new_shape[0]
        r = new_shape[1]
        # B_s1: [R, ceil(ceil(K/32)/2), 2] -> [G, R, ceil(ceil(K/32)/2), 2]
        new_s1 = B_s1.view(g, r, *B_s1.shape[1:])
        # B_s2: [ceil(ceil(R/32)/2), K, 2] -> [G, ceil(ceil(R/32)/2)//G, K, 2]
        new_s2 = B_s2.view(g, -1, *B_s2.shape[1:])
        return new_s1, new_s2

    def _can_prequantize(self, mesh: DeviceMesh | None, outer_size: torch.Size | None = None) -> bool:
        """Whether the local shard can be pre-quantized before all_gather.

        Returns False (structural no-op, the BF16 communication path simply
        applies) when:
        - ``fsdp_prequantize`` is disabled;
        - there is no FSDP sharding (``mesh.size()==1``, e.g. EFSDP=1 MoE);
        - the local storage is not allocated (backward reconstruction phase);
        - ``outer_size`` equals the local shape (nothing is sharded here).

        Raises ``ValueError`` when the shard violates the quantization/packing
        alignment: the configuration asked for pre-quantization but the weight
        shape and parallelism cannot satisfy it, so silently falling back would
        make the training behavior differ from the configuration.
        """
        if not self.weight_config.fsdp_prequantize:  # pyrefly: ignore [missing-attribute]
            return False
        # A mesh that is not a DeviceMesh (unit-test path) is treated as
        # sharded so the alignment guards below still run, mirroring a real
        # FSDP run.
        if isinstance(mesh, DeviceMesh):
            if mesh.size() == 1:
                return False
            mesh_size = mesh.size()
        else:
            mesh_size = 1
        if self._data.data_ptr() == 0:
            return False
        # Which dim FSDP actually shards is decided by the placement, not by
        # the tensor rank: 3D expert weights use Shard(0) (expert axis) unless
        # the EFSDP degree exceeds the expert count (extensions/distributed/
        # fsdp.py). The 2-block scale packing alignment (64) is only required
        # when a sharded dim IS the quantized dim: a shard of the quantized dim
        # that is 32- but not 64-aligned rounds its own scale count up, so the
        # gathered ``B_s2`` has more scale rows than the
        # global one and the dgrad ``npu_quant_matmul`` meta check fails. When
        # another dim is sharded (e.g. experts under Shard(0)) and the quantized
        # dimension is left whole, packing cannot misalign; when several dims are sharded at
        # once (e.g. a 2D mesh splitting both the expert axis and the quantized
        # dim), the guard checks every sharded dim, so a split quantized dim
        # still fails fast.
        shard_shape = tuple(self._data.shape)
        quantized_dim = -1 if self.weight_config.fsdp_prequantize_dtype == "mxfp4" else -2
        if outer_size is not None:
            dims = len(shard_shape)
            sharded = [d - dims for d in range(dims) if shard_shape[d] != outer_size[d]]
            if not sharded:
                return False  # Local shape == global shape: nothing to shard.
            check_quant_dim = quantized_dim in sharded
        else:
            # No outer_size (e.g. unit-test path): assume the quantized dim may
            # be sharded so the alignment guard runs, mirroring a real FSDP run.
            check_quant_dim = True
        if check_quant_dim and self._data.shape[quantized_dim] % _PREQUANTIZE_SHARD_ALIGNMENT != 0:
            raise ValueError(
                f"fsdp_prequantize requires the FSDP-sharded quantized dim of "
                f"local shard {shard_shape} (global {tuple(outer_size) if outer_size is not None else 'unknown'}) "
                f"to be a multiple of {_PREQUANTIZE_SHARD_ALIGNMENT} (block size {_BLOCK_SIZE} x scale "
                f"pack {_SCALE_PACK}) per FSDP shard (mesh size {mesh_size}); "
                f"remove this weight from fsdp_prequantize_fqns or adjust the "
                f"parallelism."
            )
        if self._data.shape[-1] % _BLOCK_SIZE != 0:
            raise ValueError(
                f"fsdp_prequantize requires the last dim of local shard "
                f"{shard_shape} to be a multiple of the block size "
                f"{_BLOCK_SIZE}; remove this weight from fsdp_prequantize_fqns."
            )
        return True

    # ------------------------------------------------------------------
    # FSDP hooks (pre-quantization)
    # ------------------------------------------------------------------

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ):
        if self.weight_config.fsdp_prequantize and self.weight_config.fsdp_prequantize_dtype == "mxfp4":
            self._validate_mxfp4_communication(mesh, outer_size)
        if not self._can_prequantize(mesh, outer_size):
            return super().fsdp_pre_all_gather(mesh, outer_size, outer_stride, module, mp_policy)

        # Cast the local shard to the mixed-precision param dtype before
        # quantizing (mirrors the base class, which casts to ``param_dtype``
        # prior to all_gather). The block MX kernel only accepts FP16/BF16
        # input, so a FP32 local shard (e.g. optimizer param dtype) must be
        # cast first.
        hp_local = self._data.to(mp_policy.param_dtype)
        if self.weight_config.fsdp_prequantize_dtype == "mxfp4":
            # The stored MoE layout is [E, N, K].  The grouped matmul
            # transposes it to [E, K, N], so MXFP4 QAT quantizes the stored
            # K dimension (the last axis) before communication.
            fp4, scale, permutation = quantize_mxfp4_payload(hp_local, self.weight_config, axis=-1)
            metadata = (
                "mxfp4_v1",
                tuple(hp_local.shape),
                tuple(fp4.shape),
                tuple(scale.shape),
                tuple(permutation),
                mesh.size(),
            )
            return (fp4.contiguous().view(-1), scale.contiguous().view(-1)), metadata
        # QAT quantizes K in the stored [..., N, K] layout, before the
        # matmul transpose. Direct Block FP8 ignores this axis argument.
        # pyrefly: ignore [missing-attribute]
        B_q, B_s1, B_s2 = block_mx_quantize(hp_local, config=self.weight_config, axis=-1)
        # FSDP flattens each payload with view(-1); QAT may return permuted views.
        return (B_q.contiguous(), B_s1.contiguous(), B_s2.contiguous()), ()

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        if metadata and metadata[0] == "mxfp4_v1":
            return self._post_mxfp4_all_gather(all_gather_outputs, metadata, out)
        # FSDP reconstructs the three FP8 payload tensors after its grouped collective.
        if len(all_gather_outputs) == 3:
            B_q, B_s1, B_s2 = all_gather_outputs
        else:
            # Not our format (fallback path): delegate to the base class.
            return super().fsdp_post_all_gather(all_gather_outputs, metadata, param_dtype, out=out)

        if out is None:
            # Training step 0: create a new pre-quantized wrapper.
            output = type(self)._from_prequantized(
                B_q,
                B_s1,
                B_s2,
                self.weight_config,  # pyrefly: ignore [missing-attribute]
                self.activation_config,  # pyrefly: ignore [missing-attribute]
                requires_grad=self.requires_grad,  # pyrefly: ignore [missing-attribute]
            )
            inner_tensors = (B_q, B_s1, B_s2)
            return output, inner_tensors
        else:
            # Training step 1+: write pre-quantized data into the pre-allocated out.
            self._write_prequantized_to_out(out, B_q, B_s1, B_s2)
            return None

    def _validate_mxfp4_communication(self, mesh, outer_size):
        """Reject unsupported layouts on every rank before entering a collective."""
        if self.weight_config.mxfp4_fake_quantize_config is None:
            raise ValueError("MXFP4 FSDP communication requires MXFP4 QAT")
        if mesh.size() == 1:
            return
        if len(outer_size) not in (2, 3):
            raise ValueError("MXFP4 FSDP communication supports only 2D/3D weights")
        if outer_size[0] % mesh.size() or tuple(self._data.shape) != (outer_size[0] // mesh.size(), *outer_size[1:]):
            raise ValueError("MXFP4 FSDP communication requires even Shard(0); Shard(1) is unsupported")
        config = self.weight_config.mxfp4_fake_quantize_config
        if config.block_size != 32:
            raise ValueError("MXFP4 FSDP communication currently requires block_size=32")

    def _post_mxfp4_all_gather(self, outputs, metadata, out):
        _, local_shape, fp4_shape, scale_shape, permutation, world_size = metadata
        fp4, scale = outputs
        # MXFP4 communication quantizes the last stored dimension (K). FSDP
        # shards the leading dimension, so rank-major payloads are restored by
        # concatenating that leading dimension for both 2D and 3D weights.
        fp4 = fp4.view(world_size, *fp4_shape).reshape(world_size * fp4_shape[0], *fp4_shape[1:])
        scale = scale.view(world_size, *scale_shape).reshape(world_size * scale_shape[0], *scale_shape[1:])
        q_p, s1_p, s2_p = mxfp4_to_block_mx(fp4, scale, self.weight_config)
        logical_shape = list(local_shape)
        logical_shape[0] *= world_size
        q, s1, s2 = from_block_quant_layout(tuple(logical_shape), list(permutation), q_p, s1_p, s2_p)
        # FSDP constructs its unsharded parameter with the original contiguous stride.
        q, s1, s2 = q.contiguous(), s1.contiguous(), s2.contiguous()
        if out is None:
            output = type(self)._from_prequantized(
                q,
                s1,
                s2,
                self.weight_config,
                self.activation_config,
                requires_grad=self.requires_grad,  # pyrefly: ignore [missing-attribute]
            )
            return output, (q, s1, s2)
        self._write_prequantized_to_out(out, q, s1, s2)
        return None

    def to_inference_weight(self) -> MXTensor:
        """Quantize the master weight into its inference-time :class:`MXTensor`.

        Training quantizes the stored weight's last axis (``block_mx_quantize``
        runs on the transposed operand along axis -2, i.e. the stored last axis),
        so quantizing axis -1 here reproduces the quantized weight used in
        training's forward.

        With mxfp4-QAT enabled the weight is served as MXFP4: training's QAT
        pre-pass quantizes the same stored last axis, so an
        :class:`MXTensor` built from the nested FP4 config matches it exactly.
        Otherwise the block-MX FP8 quantization is converted to the equivalent
        single-axis :class:`MXTensor` along the stored last dim (see
        :meth:`~torchao_npu.quantized_tensors.block_mx_tensor.BlockMXTensor.to_mx_tensor`).
        """
        mxfp4_config = self.weight_config.mxfp4_fake_quantize_config
        if mxfp4_config is not None:
            return MXTensor.from_hp(
                self._data,
                quant_config=mxfp4_config,
                axis=-1,
                act_quant_config=self.activation_config,
            )
        else:
            return BlockMXTensor.from_hp(
                self._data,
                quant_config=self.weight_config,
                act_quant_config=self.activation_config,
            ).to_mx_tensor(quant_axis=-1)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in (torch.mm, torch.matmul):
            # 2D matmul: A @ B where B is the wrapped weight [K, N]
            return cls._block_mx_mm(args, kwargs)

        elif func is torch._grouped_mm:
            # Grouped matmul: A @ B[E,K,N] with offs=group_list
            return cls._block_mx_grouped_mm(args, kwargs)

        elif func is torch.nn.functional.linear:
            # Linear: A @ weight.T + bias, weight is [N, K]
            return cls._block_mx_linear(args, kwargs)

        elif func is torch.addmm:
            # addmm: bias + A @ B where B is the wrapped weight
            return cls._block_mx_addmm(args, kwargs)

        elif func is torch.bmm:
            # bmm: A @ B where both are 3D and B is the wrapped weight
            return cls._block_mx_bmm(args, kwargs)

        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Override the base ``__torch_dispatch__`` to (1) route the matmul ops
        # through the pre-quantized path and (2) preserve the pre-quantized
        # scales (``_scale_s1``/``_scale_s2``) when a pre-quantized wrapper is
        # re-wrapped through a subclass-preserving op.
        #
        # (1) In ``torch.compile`` mode Dynamo bypasses ``__torch_function__``
        # and dispatches the matmul ops (``aten.linear``/``mm``/``matmul``/
        # ``addmm``/``bmm``/``_grouped_mm``) here. Without intercepting them,
        # the raw ``aten`` op would run on the FP8 ``_data`` directly, which the
        # NPU Triton backend cannot compile (``KeyError: 'r'``). We route them
        # through the same per-op helpers used by ``__torch_function__``.
        #
        # (2) FSDP2's ``init_unsharded_param`` applies ``torch.as_strided`` to
        # the wrapper returned from ``fsdp_post_all_gather``; without preserving
        # the scales, the re-wrap would drop them and leave ``_data`` as FP8
        # with no scales (which then fails the on-the-fly ``block_mx_quantize``).
        #
        # NOTE: For shape-changing ops (view/slice/transpose) the scales are
        # preserved as-is, which is only correct when the op is a no-op reshape
        # (as in the FSDP2 all-gather path). General strided slicing of a
        # pre-quantized wrapper is not supported.
        if kwargs is None:
            kwargs = {}

        # --- (1) Route matmul ops through the pre-quantized path (compile mode) ---
        # The per-op helpers expect the wrapper ``B`` to be passed directly, so
        # we dispatch on the original (un-unwrapped) args. Only intercept when a
        # wrapper is actually among the args (``__torch_dispatch__`` is only
        # entered when a subclass is involved, but the wrapper may be in either
        # operand position).
        if func in (
            torch.ops.aten.linear.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.matmul.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten._grouped_mm.default,
        ) and any(isinstance(a, cls) for a in args):
            if func == torch.ops.aten.linear.default:
                return cls._block_mx_linear(args, kwargs)
            if func in (torch.ops.aten.mm.default, torch.ops.aten.matmul.default):
                return cls._block_mx_mm(args, kwargs)
            if func == torch.ops.aten.addmm.default:
                return cls._block_mx_addmm(args, kwargs)
            if func == torch.ops.aten.bmm.default:
                return cls._block_mx_bmm(args, kwargs)
            if func == torch.ops.aten._grouped_mm.default:
                return cls._block_mx_grouped_mm(args, kwargs)

        prequantized_scales = None
        weight_config = None
        activation_config = None
        unique_weight_config = True
        unique_activation_config = True

        def unwrap(t):
            nonlocal prequantized_scales, weight_config, activation_config
            nonlocal unique_weight_config, unique_activation_config
            if t._has_prequantized_data():
                prequantized_scales = (t._scale_s1, t._scale_s2)
            if weight_config is None:
                weight_config = t.weight_config
            else:
                unique_weight_config = unique_weight_config and (t.weight_config == weight_config)
            if activation_config is None:
                activation_config = t.activation_config
            else:
                unique_activation_config = unique_activation_config and (t.activation_config == activation_config)
            return t._data

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            BaseTrainingWeightWrapperTensor, unwrap, (args, kwargs or {})
        )

        if func == torch.ops.aten.detach.default:
            src = args[0]
            if src._has_prequantized_data():
                return cls._from_prequantized(
                    args_unwrapped[0].detach(),
                    src._scale_s1,
                    src._scale_s2,
                    src.weight_config,
                    src.activation_config,
                )
            return cls(
                args_unwrapped[0].detach(),
                activation_config=activation_config,
                weight_config=weight_config,
            )

        out = func(*args_unwrapped, **kwargs_unwrapped)

        if func not in _ops_to_preserve_subclass:
            return out

        assert unique_activation_config, (
            f"In {func}, all BaseTrainingWeightWrapperTensor instances must have the same activation_config"
        )
        assert unique_weight_config, (
            f"In {func}, all BaseTrainingWeightWrapperTensor instances must have the same weight_config"
        )

        if prequantized_scales is not None:
            s1, s2 = prequantized_scales
            # A transpose of the last two dims swaps the roles of the two block
            # MX scales, so they must be transformed to stay consistent with
            # the transposed ``_data`` (e.g. ``wo_a.transpose(-1, -2)`` in the
            # DeepSeek-V4 output projection bmm).
            if cls._is_last_two_dim_transpose(func, args_unwrapped, kwargs_unwrapped):
                s1, s2 = cls._transpose_prequantized_scales(args_unwrapped[0], s1, s2)[1:]
            # A 2D->3D view (adding a leading batch dim) reshapes the scales to
            # add the batch dim (e.g. ``wo_a.weight.view(G, R, -1)`` in the
            # DeepSeek-V4 output projection bmm).
            elif cls._is_2d_to_3d_view(func, args_unwrapped, kwargs_unwrapped):
                s1, s2 = cls._reshape_prequantized_scales(s1, s2, args_unwrapped[0].shape, out.shape)
            return pytree.tree_map_only(
                torch.Tensor,
                lambda x: cls._from_prequantized(x, s1, s2, weight_config, activation_config),
                out,
            )
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: cls(
                x,
                activation_config=activation_config,
                weight_config=weight_config,
            ),
            out,
        )

    @staticmethod
    def _is_last_two_dim_transpose(func, args_unwrapped, kwargs_unwrapped) -> bool:
        """True if ``func`` transposes the last two dims of its first argument."""
        if func == torch.ops.aten.t.default:
            return True
        if func == torch.ops.aten.transpose.int:
            dim0 = args_unwrapped[1]
            dim1 = args_unwrapped[2]
            ndim = args_unwrapped[0].ndim
            return {dim0 % ndim, dim1 % ndim} == {ndim - 2, ndim - 1}
        if func == torch.ops.aten.permute.default:
            dims = list(args_unwrapped[1])
            ndim = len(dims)
            return dims == [*list(range(ndim - 2)), ndim - 1, ndim - 2]
        return False

    @staticmethod
    def _is_2d_to_3d_view(func, args_unwrapped, kwargs_unwrapped) -> bool:
        """True if ``func`` is a view/reshape that adds a leading batch dim.

        A 2D pre-quantized weight ``[R, K]`` viewed as ``[G, R, K]`` (with
        ``G * R == R``) adds a leading batch dim, so the block MX scales must
        be reshaped to match (e.g. ``wo_a.weight.view(G, R, -1)`` in the
        DeepSeek-V4 output projection bmm).
        """
        if func in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
            orig = args_unwrapped[0]
            new_shape = args_unwrapped[1]
            return orig.ndim == 2 and len(new_shape) == 3 and orig.shape[0] == new_shape[0] * new_shape[1]
        return False

    # ------------------------------------------------------------------
    # Per-op helpers
    # ------------------------------------------------------------------

    @classmethod
    def _block_mx_mm(cls, args, kwargs):
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        with torch._C.DisableTorchFunctionSubclass():
            if B._has_prequantized_data():
                return to_block_mx_then_mm_from_prequantized(A, B, B.activation_config)  # type: ignore
            B_data = unwrap_weight(B)
            return to_block_mx_then_mm(A, B_data, B.activation_config, B.weight_config)  # type: ignore

    @classmethod
    def _block_mx_grouped_mm(cls, args, kwargs):
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        group_list = args[2] if len(args) > 2 else kwargs.get("offs")

        with torch._C.DisableTorchFunctionSubclass():
            if B._has_prequantized_data():
                return to_block_mx_then_grouped_mm_from_prequantized(A, B, group_list, B.activation_config)  # type: ignore
            B_data = unwrap_weight(B)
            return to_block_mx_then_grouped_mm(A, B_data, group_list, B.activation_config, B.weight_config)  # type: ignore

    @classmethod
    def _block_mx_linear(cls, args, kwargs):
        # F.linear(A, weight, bias) — weight is [N, K], A @ weight.T + bias
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        bias = args[2] if len(args) > 2 else kwargs.get("bias")

        with torch._C.DisableTorchFunctionSubclass():
            if B._has_prequantized_data():
                result = to_block_mx_then_linear_from_prequantized(A, B, B.activation_config)  # type: ignore
            else:
                B_data = unwrap_weight(B)
                result = to_block_mx_then_mm(A, B_data.T, B.activation_config, B.weight_config)  # type: ignore
            if bias is not None:
                result = result + bias
            return result

    @classmethod
    def _block_mx_addmm(cls, args, kwargs):
        # addmm(bias, A, B) — bias + A @ B
        bias, A, B = args[0], args[1], args[2]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        with torch._C.DisableTorchFunctionSubclass():
            if B._has_prequantized_data():
                result = to_block_mx_then_mm_from_prequantized(A, B, B.activation_config)  # type: ignore
            else:
                B_data = unwrap_weight(B)
                result = to_block_mx_then_mm(A, B_data, B.activation_config, B.weight_config)  # type: ignore
            result = result + bias
            return result

    @classmethod
    def _block_mx_bmm(cls, args, kwargs):
        # bmm(A, B) — both 3D batched matmul, B is the wrapped weight
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        with torch._C.DisableTorchFunctionSubclass():
            if B._has_prequantized_data():
                return to_block_mx_then_bmm_from_prequantized(A, B, B.activation_config)  # type: ignore
            B_data = unwrap_weight(B)
            return to_block_mx_then_bmm(A, B_data, B.activation_config, B.weight_config)  # type: ignore

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __tensor_flatten__(self):
        if self._has_prequantized_data():
            return ["_data", "_scale_s1", "_scale_s2"], {
                "activation_config": self.activation_config,  # pyrefly: ignore [missing-attribute]
                "weight_config": self.weight_config,  # pyrefly: ignore [missing-attribute]
            }
        return super().__tensor_flatten__()

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride):
        if "_scale_s1" in tensor_data_dict:
            return cls._from_prequantized(
                tensor_data_dict["_data"],
                tensor_data_dict["_scale_s1"],
                tensor_data_dict["_scale_s2"],
                tensor_attributes["weight_config"],
                tensor_attributes["activation_config"],
            )
        return cls(
            tensor_data_dict["_data"],
            activation_config=tensor_attributes["activation_config"],
            weight_config=tensor_attributes["weight_config"],
        )


@register_parameter_swap_handler(BlockMXQuantizeConfig)
def _(
    module: nn.Module,
    param_fqn: str,
    param: nn.Parameter,
    extra_args: tuple[Any, ...] = (),
):
    from torchao_npu.configs import ParamSwapConfig

    config: ParamSwapConfig = extra_args[0]

    if not isinstance(config, ParamSwapConfig):
        raise ValueError(f"extra_args[0] must be a ParamSwapConfig, got {type(config).__name__}.")

    if config.activation_config is not None and type(config.activation_config) is not MXQuantizeConfig:
        raise ValueError(
            f"activation_config must be {MXQuantizeConfig.__name__}, got {type(config.activation_config).__name__}."
        )

    if config.weight_config is not None and type(config.weight_config) is not BlockMXQuantizeConfig:
        raise ValueError(
            f"weight_config must be {BlockMXQuantizeConfig.__name__}, got {type(config.weight_config).__name__}."
        )

    return nn.Parameter(
        data=BlockMXTrainingWeightWrapperTensor(
            param.data,
            activation_config=config.activation_config,
            weight_config=config.weight_config,
        ),
        requires_grad=param.requires_grad,
    )


# Safe-unpickling allowlist: DCP loads checkpoints with torch.load(weights_only=True).
torch.serialization.add_safe_globals([BlockMXTrainingWeightWrapperTensor])
