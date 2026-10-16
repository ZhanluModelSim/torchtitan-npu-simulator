# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
"""Chunked LM loss mirroring MindSpeed fsdp2 ``ChunkLoss``.

The measured fsdp2 run computes the language-model loss in
``optimization.chunk_loss_size`` (1024) token chunks inside a custom
autograd function: per chunk it projects the hidden states through the
LM head, evaluates the sum-reduced cross entropy normalized by the
valid-token count, and derives the input/weight gradients **in the
forward pass** (``torch.func.grad_and_value``), back-filling
pre-allocated gradient buffers with ``copy_``/``add_``. The backward
pass only rescales the saved gradients.

The simulator mirrors that op stream: the model returns the final
hidden states (the LM-head module is bound to an identity forward so
its FSDP unit still all-gathers), and this loss function performs the
per-chunk matmul + cross-entropy with in-forward gradients. Under meta
capture every recorded op is value-free; the chunk count follows from
the static sequence length.
"""

import torch
import torch.nn.functional as F

CHUNK_LOSS_SIZE = 1024

_active = False


def is_active() -> bool:
    return _active


class _SimChunkLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, alpha):
        device = hidden.device
        accumulated_loss = torch.zeros((), dtype=torch.float32, device=device)
        grad_inputs = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight)
        hidden_chunks = torch.split(hidden, CHUNK_LOSS_SIZE, dim=1)
        label_chunks = torch.split(labels, CHUNK_LOSS_SIZE, dim=1)
        grad_input_chunks = torch.split(grad_inputs, CHUNK_LOSS_SIZE, dim=1)
        with torch.enable_grad():
            for hidden_chunk, label_chunk, grad_input_chunk in zip(
                hidden_chunks, label_chunks, grad_input_chunks
            ):
                chunk_input = hidden_chunk.detach()
                chunk_input.requires_grad_(True)
                # calculate_lm_loss: flatten -> F.linear -> .float() ->
                # cross_entropy(sum) / alpha
                logits = F.linear(
                    chunk_input.reshape(-1, chunk_input.size(-1)), weight
                ).float()
                chunk_loss = F.cross_entropy(
                    logits,
                    label_chunk.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                chunk_loss = chunk_loss / alpha
                chunk_grad_input, chunk_grad_weight = torch.autograd.grad(
                    chunk_loss, (chunk_input, weight)
                )
                accumulated_loss.add_(chunk_loss)
                grad_input_chunk.copy_(chunk_grad_input)
                grad_weight.add_(chunk_grad_weight)
        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        # MindSpeed skips the rescale only when grad_output == 1; the
        # trainer's loss/global_valid_tokens pre-division makes it != 1.
        return grad_output * grad_input, grad_output * grad_weight, None, None


def make_chunk_loss_ctx(labels: torch.Tensor):
    """Build the measured ``loss_ctx`` closure.

    MindSpeed's trainer injects ``loss_ctx`` into the model forward; the
    LM-head module invokes it as ``loss_ctx(hidden_states, weight, bias)``
    and the returned tensor IS the model loss (the torchtitan loss fn is
    therefore a passthrough on the precomputed loss).
    """
    def loss_ctx(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        loss_mask = labels != -100
        alpha = loss_mask.sum()
        return _SimChunkLoss.apply(hidden, weight, labels, alpha)

    return loss_ctx


def build_qwen3_next_chunk_loss(compile_config=None, **kwargs):
    """``build_loss_fn`` factory for the qwen3_next model spec.

    The model forward computes the chunked loss internally (mirroring
    MindSpeed's ``loss_ctx`` flow), so the trainer-side loss function is
    a passthrough on the precomputed loss sum.
    """
    global _active
    del compile_config, kwargs
    _active = True

    def passthrough(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        return pred

    return passthrough
