# Copyright (c) Amazon.com, Inc. or its affiliates
# Licensed under the BSD-3-Clause license.

"""Contiguous state-based GDN CP. B=batch, H=value heads, K/V=head widths."""

import torch
from torch.distributed._functional_collectives import all_gather_single


def _reference(q, k, v, g, beta, state, mask, scale):
    repeats = v.shape[1] // k.shape[1]
    q, k = (x.float().repeat_interleave(repeats, dim=1) for x in (q, k))
    outputs = []
    for t in range(q.shape[2]):
        key = k[:, :, t, :, None]
        state = state * g[:, :, t, None, None].exp()
        delta = v[:, :, t, None, :].float() - key.mT @ state
        state = state + key @ (beta[:, :, t, None, None] * delta)
        outputs.append(((q[:, :, t, None, :] * scale) @ state).squeeze(-2))
    return torch.stack(outputs, dim=2).to(v.dtype), state


def _reference_backward(q, k, v, g, beta, state, dy, ds, mask, scale):
    with torch.enable_grad():
        q, k, v, g, beta, state = (
            x.detach().requires_grad_() for x in (q, k, v, g, beta, state)
        )
        args = (q, k, v, g, beta, state)
        outputs = _reference(*args, mask, scale)
        return torch.autograd.grad(outputs, args, (dy, ds))


class StateParallelGDN(torch.autograd.Function):
    """One affine-summary gather forward and one cotangent gather backward."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, q, k, v, g, beta, mask, scale, mesh, rank):
        batch, heads, _, dv = v.shape
        dk = k.shape[-1]
        zero = v.new_zeros((batch, heads, dk, dv), dtype=torch.float32)
        eye = (
            torch.eye(dk, device=v.device, dtype=torch.float32)
            .expand(batch, heads, dk, dk)
            .contiguous()
        )
        # In NKI's K-major convention each shard maps S -> A @ S + B.
        # B is the terminal state from zero. A is the terminal state from I
        # with the additive update disabled by supplying zero values.
        if q.device.type == "neuron":
            from .nki_gdn import chunk_forward, state_forward as run

            _, bias = chunk_forward(q, k, v, g, beta, mask, scale)
        else:
            run = _reference
            _, bias = run(q, k, v, g, beta, zero, mask, scale)
        zero_values = v.new_zeros((*v.shape[:-1], dk))
        _, transition = run(q, k, zero_values, g, beta, eye, mask, scale)
        summaries = all_gather_single(torch.cat((transition, bias), -1), 0, mesh)
        matrices, biases = summaries.reshape(
            mesh.size(), batch, heads, dk, dk + dv
        ).split((dk, dv), -1)
        # All peers must compile identical collective graphs on Trainium. The
        # mesh coordinate is a runtime tensor, including for the boundary ranks.
        entry = zero
        for peer in range(mesh.size() - 1):
            entry = torch.where(
                rank > peer, matrices[peer] @ entry + biases[peer], entry
            )
        out, _ = run(q, k, v, g, beta, entry, mask, scale)
        ctx.save_for_backward(q, k, v, g, beta, mask, entry, matrices, rank)
        ctx.scale, ctx.mesh = scale, mesh
        return out

    @staticmethod
    def backward(ctx, dy):  # pyrefly: ignore [bad-override]
        q, k, v, g, beta, mask, entry, matrices, rank = ctx.saved_tensors
        if q.device.type == "neuron":
            from .nki_gdn import state_backward as vjp
        else:
            vjp = _reference_backward
        args = (q, k, v, g, beta, entry, dy.float().contiguous())
        ds = torch.zeros_like(entry)
        local = vjp(*args, ds, mask, ctx.scale)
        biases = all_gather_single(local[-1].contiguous(), 0, ctx.mesh).reshape(
            ctx.mesh.size(), *entry.shape
        )
        for peer in range(ctx.mesh.size() - 1, 0, -1):
            ds = torch.where(rank < peer, matrices[peer].mT @ ds + biases[peer], ds)
        grads = vjp(*args, ds.contiguous(), mask, ctx.scale)
        return (*grads[:5], None, None, None, None)
