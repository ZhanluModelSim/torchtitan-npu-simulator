# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common base for the frozen, quantized NPU tensor subclasses."""

import torch
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten

# Logical dtypes a quantized tensor may be presented as; they are the high-precision
# dtypes, since the stored ``qdata``/``scale`` are never cast.
_SUPPORTED_TO_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class BaseQuantizedTensor(TorchAOBaseTensor):
    """Common base for frozen, quantized NPU tensor subclasses.

    Subclasses declare their components and metadata through ``tensor_data_names``
    / ``tensor_attribute_names``, plus their optional counterparts
    ``optional_tensor_data_names`` / ``optional_tensor_attribute_names``, and
    inherit the serialization and dispatch machinery from ``TorchAOBaseTensor``.
    Not intended to be instantiated directly.

    Behavior inherited by all subclasses:
        * Values are frozen: ``__torch_dispatch__`` rejects every in-place
          (mutable-schema) op with a ``RuntimeError``.

        * ``__torch_function__`` is disabled, so every op routes straight to
          ``__torch_dispatch__``, bypassing the Python function layer.

        * Op support is a whitelist: ``TorchAOBaseTensor`` leaves most ops
          unimplemented, and each subclass registers the ops it supports
          explicitly. ``detach`` / ``clone`` / ``alias`` / ``contiguous``
          dispatch handlers and ``_apply_fn_to_data`` come free from
          ``TorchAOBaseTensor``; every other unimplemented op raises
          ``NotImplementedError``.

        * ``.to`` moves the tensor between devices and additionally accepts
          ``float16`` / ``bfloat16`` / ``float32`` as its logical dtype: only the
          dtype the stored quantized data is presented as changes, never the data
          itself (``qdata``/``scale`` keep their dtypes). Any other target dtype
          raises ``ValueError``.
    """

    # Disable the torch-function layer: ops route straight to __torch_dispatch__.
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Frozen values: no in-place op may run on a quantized tensor. This also
        # shadows the ``aten.copy_.default`` handler ``TorchAOBaseTensor``
        # auto-registers; every other mutating op was already unregistered and
        # would otherwise fall through to the NotImplementedError base fallback.
        if func._schema.is_mutable:
            raise RuntimeError(
                f"{cls.__name__} is immutable; in-place op {func} is not supported. "
                "Quantized tensors are frozen -- dequantize to a plain tensor first."
            )

        if func is aten._to_copy.default:
            args = _args_with_requested_dtype_(args, kwargs)

        return super().__torch_dispatch__(func, types, args, kwargs)


def _args_with_requested_dtype_(args: tuple, kwargs: dict | None) -> tuple:
    """Update ``args`` so that ``args[0]`` carries the requested ``.to`` dtype.

    Why the operand has to be replaced: a wrapper's dtype is fixed when the tensor is
    constructed, so it cannot be written onto the tensor the inherited handler rebuilds
    -- that handler rebuilds the tensor from ``args[0]``'s components and attributes.
    The requested dtype therefore has to travel inside ``args[0]``: it is replaced by a
    copy of the tensor -- same components, device and other metadata, only ``orig_dtype``
    differing -- which the handler rebuilds into the tensor the caller receives.

    Warning:
        ``kwargs`` is modified in place: the requested ``dtype`` is popped from it, since
        the dtype is handled here and should not also trigger the parent class's dtype
        handling.

    Args:
        args: The dispatched ``aten._to_copy.default`` args, ``args[0]`` being the
            tensor.
        kwargs: Its keyword args.

    Returns:
        ``args`` with ``args[0]`` carrying the requested dtype, or ``args`` unchanged
        when no other dtype is requested.

    Raises:
        ValueError: If the requested dtype is not a supported logical dtype.
        RuntimeError: If the tensor's class does not declare ``orig_dtype`` among its
            ``tensor_attribute_names``.
    """
    x = args[0]
    dtype = (kwargs or {}).get("dtype")
    if dtype is None or dtype is x.dtype:
        return args

    if dtype not in _SUPPORTED_TO_DTYPES:
        raise ValueError(
            f"{type(x).__name__} only supports the logical dtypes {_SUPPORTED_TO_DTYPES} for ``.to``, got {dtype}."
        )

    if "orig_dtype" not in x.tensor_attribute_names:
        raise RuntimeError(
            f"{type(x).__name__} must declare ``orig_dtype`` in ``tensor_attribute_names``; "
            "its logical dtype cannot be changed by ``.to``."
        )

    # ``kwargs`` is a dict here: the dtype checked above was read off it, so it cannot be
    # ``None``; pyrefly just cannot narrow ``dict | None`` through that read.
    kwargs.pop("dtype")  # pyrefly: ignore [missing-attribute]

    data = [getattr(x, name) for name in x.tensor_data_names]
    attributes = [dtype if name == "orig_dtype" else getattr(x, name) for name in x.tensor_attribute_names]
    optional_data = [getattr(x, name) for name in getattr(x, "optional_tensor_data_names", [])]
    optional_attributes = [getattr(x, name) for name in getattr(x, "optional_tensor_attribute_names", [])]
    return (type(x)(*data, *attributes, *optional_data, *optional_attributes), *args[1:])
