# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch._prims_common import suggest_memory_format
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase
from torchao.utils import TorchAOBaseTensor

"""
ATen ops that should preserve the wrapper subclass identity. When any of these
ops is called on a BaseTrainingWeightWrapperTensor, the output is re-wrapped
in the same subclass with the operated-on ``_data``.

Design: deferred transform. Slicing or indexing a wrapped weight returns
a new wrapper with the sliced ``_data`` — no precision transform is applied
at this point. The transform is deferred until computation time, when
``__torch_function__`` intercepts computation ops (``torch.mm``, ``torch.bmm``,
``torch._grouped_mm``, etc.) and applies it just before the op. This avoids
double application (e.g., double fake-quantization).

Indexing patterns on a 3D weight tensor and their ATen ops (all preserved):

  w[0]                  → aten.select.int
  w[0:5]                → aten.slice.Tensor
  w[ids]                → aten.index.Tensor / aten._unsafe_index.Tensor
  w[[0,2,3]]            → aten.index.Tensor / aten._unsafe_index.Tensor
  w[mask]               → aten.index.Tensor / aten._unsafe_index.Tensor
  w[ids, :, :]          → aten.index.Tensor / aten._unsafe_index.Tensor
  w[ids, [0,1]]         → aten.index.Tensor / aten._unsafe_index.Tensor
  w[...]                → aten.slice.Tensor
  w[None]               → aten.unsqueeze.default

Dimension-manipulation ops follow the same deferred design — they return a new
wrapper with the reshaped ``_data``, no quantization applied:
  permute, squeeze, view, as_strided, transpose, t, split
"""
_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.index.Tensor,
    torch.ops.aten._unsafe_index.Tensor,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,  # for *.to(dtype)
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
    # required for TP - scatter_ is used to distribute weights
    torch.ops.c10d.scatter_.default,
}


class BaseTrainingWeightWrapperTensor(TorchAOBaseTensor):
    """
    Base class for wrapper tensor subclasses that intercept computation ops
    during training to apply a precision-specific transform to weights.

    Wraps a 2D or 3D weight tensor ``_data`` plus optional
    :class:`~torchao.quantization.qat.fake_quantize_config.FakeQuantizeConfigBase`
    instances for weight and activation. Subclasses define the actual
    transform — e.g., fake-quantize with straight-through estimator, or real low-precision
    matmul — by overriding :meth:`__torch_function__`.

    Carries decoupled weight/activation configs (either may be absent) and
    provides deferred-op infrastructure (subclass-preserving slicing/indexing,
    FSDP2 hooks, ``__deepcopy__``, ``to_tensor``, ``requires_grad_`` sync)
    usable by both fake-quantize and real-low-precision subclasses.

    Supports FSDP2 via :meth:`fsdp_pre_all_gather` and :meth:`fsdp_post_all_gather`,
    which handle mixed-precision casting and wrapper reconstruction after all-gather.

    Subclasses may override :meth:`__torch_function__` to intercept computation
    ops and apply their precision-specific transform. The base class's default
    ``__torch_function__`` forwards to the C++ ``_disabled_torch_function_impl``
    sentinel so unwrapped ops fall through to ``__torch_dispatch__`` unchanged.

    Not intended to be used directly.
    """

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        weight_config: FakeQuantizeConfigBase | None = None,
        activation_config: FakeQuantizeConfigBase | None = None,
    ):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned() if not isinstance(tensor, DTensor) else False,
            requires_grad=tensor.requires_grad,
        )
        return self

    def __init__(
        self,
        tensor: torch.Tensor,
        weight_config: FakeQuantizeConfigBase | None = None,
        activation_config: FakeQuantizeConfigBase | None = None,
    ):
        self._data = tensor
        self.weight_config = weight_config
        self.activation_config = activation_config

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: Iterable[type],
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        # Forward to the C++ sentinel ``_disabled_torch_function_impl`` so
        # PyTorch's identity check (``__torch_function__ is _disabled_torch_function_impl``)
        # semantics are preserved for any code that inspects this attribute.
        # Defined as a classmethod (rather than assigned as the bare C function)
        # so subclasses can override with a matching signature.
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs or {})

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Operands may not all be wrapped — e.g., activations can be plain
        # tensors with no config — so we check to ensure at least one operand
        # carries a config.
        weight_config = None
        activation_config = None
        unique_weight_config = True
        unique_activation_config = True

        # Uniqueness of weight_config / activation_config is tracked across all
        # wrapped operands, but only enforced on the re-wrap path below — i.e.,
        # when an op's output is wrapped back into the subclass. Ops that don't
        # preserve the subclass (or short-circuit via detach / copy_) never
        # re-wrap, so they tolerate mixed configs.
        def unwrap(t: BaseTrainingWeightWrapperTensor):
            nonlocal weight_config, activation_config, unique_weight_config, unique_activation_config
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

        # To align with the semantics of "detach" and avoid the "dual-nature"
        # problem of a wrapper, we also detach _data. The config is shared
        # since in the "detach" of torch.nn.Tensor, most of metadata is also
        # shared except the metadata related to autograd.
        #
        # NOTE: Configs are assumed immutable. If configs gain trainable parameters,
        #       a newly-created config should be used instead of sharing the reference.
        #       The actual logic depends on the design of these quantization parameters in the future.
        if func == torch.ops.aten.detach.default:
            return cls(
                args_unwrapped[0].detach(),
                activation_config=activation_config,
                weight_config=weight_config,
            )

        # Perform op
        out = func(*args_unwrapped, **kwargs_unwrapped)

        # Return regular tensors for ops that don't preserve subclass
        if func not in _ops_to_preserve_subclass:
            return out

        # Return the original wrapper to maintain in-place semantics.
        # Unlike copy_ where the wrapper is both input and output, scatter_
        # writes into pre-allocated output buffers that did not exist as
        # wrappers before the call — so scatter_ falls through to the generic
        # re-wrap path below to create new wrappers from the result tensors.
        if func == torch.ops.aten.copy_.default:
            return args[0]

        assert unique_activation_config, (
            f"In {func}, all BaseTrainingWeightWrapperTensor instances must have the same activation_config"
        )
        assert unique_weight_config, (
            f"In {func}, all BaseTrainingWeightWrapperTensor instances must have the same weight_config"
        )

        # Wrap outputs back into the same subclass for the remaining preserved ops.
        # Configs are captured during unwrapping (above), which handles both single-input
        # ops (select, slice, view, etc.) and multi-input ops (scatter_) correctly.
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: cls(
                x,
                activation_config=activation_config,
                weight_config=weight_config,
            ),
            out,
        )

    def __deepcopy__(self, memo):
        result = type(self)(
            self._data.clone(),
            activation_config=copy.deepcopy(self.activation_config),
            weight_config=copy.deepcopy(self.weight_config),
        )

        # self.requires_grad triggers __torch_function__ of self, use `DisableTorchFunctionSubclass`
        # to avoid the NotImplementedError error if BaseTrainingWeightWrapperTensor is used.
        with torch._C.DisableTorchFunctionSubclass():
            result.requires_grad = self.requires_grad  # pyrefly: ignore [missing-attribute]

        return result

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"data={self._data}, "
            f"activation_config={self.activation_config}, "
            f"weight_config={self.weight_config})"
        )

    def __tensor_flatten__(self):
        metadata = {
            "activation_config": self.activation_config,
            "weight_config": self.weight_config,
        }
        return ["_data"], metadata

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride):
        return cls(
            tensor_data_dict["_data"],
            activation_config=tensor_attributes["activation_config"],
            weight_config=tensor_attributes["weight_config"],
        )

    def requires_grad_(self, requires_grad: bool = True):
        # requires_grad_ bypasses both __torch_function__ and __torch_dispatch__,
        # so it only sets the flag on the wrapper. Need to keep _data in sync, otherwise
        # Dynamo sees _data.requires_grad=False and may drop custom autograd.Function
        # backward during tracing, causing eager/compile gradient mismatches.
        super().requires_grad_(requires_grad)
        self._data.requires_grad_(requires_grad)
        return self

    def to_tensor(self) -> torch.Tensor:
        """Return the underlying raw tensor, unwrapping the subclass."""
        return self._data

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ):
        # Cast to mixed precision dtype prior to all-gather
        all_gather_inputs = (self._data.to(mp_policy.param_dtype),)
        all_gather_metadata = ()
        return all_gather_inputs, all_gather_metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        (data,) = all_gather_outputs

        # For training step 0, out=None, create a new wrapper.
        if out is None:
            output = type(self)(
                data,
                activation_config=self.activation_config,
                weight_config=self.weight_config,
            )
            inner_tensors = (data,)
            return output, inner_tensors
        else:
            # For training step 1+, out=unsharded param. FSDP2 creates a shallow copy
            # of the wrapper; we restore configs from self. out may be a bare subclass
            # or wrapped in DTensor.
            if isinstance(out, BaseTrainingWeightWrapperTensor):
                out_data = out._data
                out.activation_config = self.activation_config
                out.weight_config = self.weight_config
            elif isinstance(out, DTensor) and isinstance(out._local_tensor, BaseTrainingWeightWrapperTensor):
                # Bind to a local: pyrefly doesn't propagate isinstance narrowing
                # through attribute accesses reliably, so `out._local_tensor.X`
                # below would lose the BaseTrainingWeightWrapperTensor narrowing.
                local_tensor: BaseTrainingWeightWrapperTensor = out._local_tensor
                out_data = local_tensor._data
                local_tensor.activation_config = self.activation_config
                local_tensor.weight_config = self.weight_config
            else:
                raise RuntimeError(
                    f"expected out to be {type(self).__name__} or DTensor with "
                    f"local_tensor={type(self).__name__}, but got {type(out)}"
                )

            # If `data` (all-gather outputs) is already in the mixed precision policy param_dtype,
            # verify it has underlying storage as `out` (pre-allocated unsharded param),
            # and then we can just return directly.
            if data.dtype == param_dtype:
                assert data.untyped_storage().data_ptr() == out_data.untyped_storage().data_ptr()
            else:
                # Otherwise, verify that `out` (pre-allocated unsharded param) has the
                # mixed precision policy param_dtype, then copy `data` to `out`.
                assert out_data.dtype == param_dtype, (
                    f"`out`(dtype={out_data.dtype}) does not match the mixed precision policy param_dtype {param_dtype}"
                )
                out_data.copy_(data)

            return None
