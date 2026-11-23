# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Differentiable, zero-initial-state chunked GDN backed by the Neuron SDK."""

import torch
from nkilib.experimental import gdn
from torch_neuronx import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

if not hasattr(gdn, "gdn_cte_bwd"):
    raise ImportError(
        "Qwen3.8-27B training requires nkilib.experimental.gdn.gdn_cte_bwd "
        "from the NKI Library. Install the revision in the Qwen3.8 recipe README; "
        "Neuron SDK 2.33.1 alone does not provide it."
    )

CHUNK = gdn.CHUNK
_grid = (int(get_logical_neuron_cores()),)
_forward = wrap_nki(gdn.gdn_cte)
_backward = wrap_nki(gdn.gdn_cte_bwd)


@nki_op("torchtitan_neuron::gdn_chunk_forward", mutates_args={})
def chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward[_grid](
        q, k, v, g, beta, scale=scale, algorithm=gdn.ALGORITHM_SCHUR, mask_pack=mask
    )


@nki_op("torchtitan_neuron::gdn_chunk_backward", mutates_args={})
def chunk_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    grad_output: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv, dg, db, _ = _backward[_grid](
        q, k, v, g, beta, grad_output, mask, scale=scale
    )
    return dq, dk, dv, dg, db


def _setup(ctx, inputs, output):
    *tensors, scale = inputs
    ctx.save_for_backward(*tensors)
    ctx.scale = scale
    # Training consumes only the outputs at token positions, not the terminal state.
    ctx.mark_non_differentiable(output[1])


def _vjp(ctx, grad_output, _grad_state):
    q, k, v, g, beta, mask = ctx.saved_tensors
    grads = chunk_backward(
        q, k, v, g, beta, grad_output.float().contiguous(), mask, ctx.scale
    )
    return (*grads, None, None)


chunk_forward.register_autograd(_vjp, setup_context=_setup)
