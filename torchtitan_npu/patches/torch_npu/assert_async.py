import torch.library


@torch.library.impl("aten::_assert_async.msg", "PrivateUse1")
def _(self: torch.Tensor, assert_msg: str) -> None:
    return
