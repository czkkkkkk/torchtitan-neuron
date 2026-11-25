# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Gated delta rule for training. B: batch, L: tokens, H: local heads, D: head width."""

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from torchtitan.models.qwen3_5.model import _torch_native_gated_delta, GatedDeltaKernel


class _ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):  # pyrefly: ignore [bad-override]
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):  # pyrefly: ignore [bad-override]
        # The local_map boundary returns this to a DTensor view backward.
        return grad.contiguous()


class NeuronGatedDeltaKernel(GatedDeltaKernel):
    supports_state_cp = True
    mask_pack: torch.Tensor
    cp_rank: torch.Tensor

    @dataclass(kw_only=True, slots=True)
    class Config(GatedDeltaKernel.Config):
        def build(self, **kwargs):
            instance = super(NeuronGatedDeltaKernel.Config, self).build(**kwargs)
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import (
                    dense_param_placement,
                )

                instance._sharding_config = replace(
                    self.sharding_config,
                    state_shardings={
                        name: dense_param_placement(tp=spmd.R)
                        for name in ("mask_pack", "cp_rank")
                    },
                )
            return instance

    def __init__(self, config: Config):
        super().__init__(config)
        self.chunk = 1
        self.register_buffer(
            "cp_rank", torch.empty((), dtype=torch.int32), persistent=False
        )
        self.register_buffer("mask_pack", torch.empty(0), persistent=False)
        self._init_self_buffers()

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = self.mask_pack.device if buffer_device is None else buffer_device
        rank = self.cp_mesh.get_local_rank() if self.cp_mesh is not None else 0
        self.cp_rank = torch.tensor(rank, device=device, dtype=torch.int32)
        if device.type == "neuron":
            from nkilib.experimental.gdn import build_mask_pack

            from . import nki_gdn  # register the operators before compiling the block

            self.chunk = nki_gdn.CHUNK
            self.mask_pack = torch.from_numpy(build_mask_pack()).to(device)

    def local_cp_rank(self):
        from torch.distributed.tensor import DTensor

        return (
            self.cp_rank.to_local()
            if isinstance(self.cp_rank, DTensor)
            else self.cp_rank
        )

    def forward(self, q, k, v, g, beta):
        q, k, v, g, beta = (_ContiguousGrad.apply(x) for x in (q, k, v, g, beta))
        if q.device.type == "cpu" and self.cp_mesh is None:
            if q.shape[2] != v.shape[2]:
                repeats = v.shape[2] // q.shape[2]
                q = q.repeat_interleave(repeats, dim=2)
                k = k.repeat_interleave(repeats, dim=2)
            return _torch_native_gated_delta(q, k, v, g, beta)
        if q.device.type not in ("neuron", "cpu"):
            raise ValueError(f"Neuron GDN requires Trainium, got {q.device}")
        # The Schur kernel implements grouped heads and accumulates dQ and
        # dK within each key head. Expanding heads outside the custom op
        # leaves this reduction to the compiler, which miscompiles its VJP.
        from torch.distributed.tensor import DTensor

        from .context_parallel import StateParallelGDN

        width = q.shape[1]
        pad = (-width) % self.chunk

        def heads_first(x, *, normalize=False):
            x = x.transpose(1, 2)
            if normalize:
                value = x.float()
                x = (
                    value * torch.rsqrt(value.square().sum(-1, keepdim=True) + 1e-6)
                ).to(x.dtype)
            return F.pad(x, (0, 0, 0, pad)).contiguous() if pad else x.contiguous()

        def gate(x):
            x = x.transpose(1, 2).float()
            return F.pad(x, (0, pad)).contiguous() if pad else x.contiguous()

        mask = self.mask_pack
        if isinstance(mask, DTensor):
            mask = mask.to_local()
        args = (
            heads_first(q, normalize=True),
            heads_first(k, normalize=True),
            heads_first(v),
            gate(g),
            gate(beta),
            mask,
            q.shape[-1] ** -0.5,
        )
        if self.cp_mesh is None:
            from .nki_gdn import chunk_forward

            out, _ = chunk_forward(*args)
        else:
            out = StateParallelGDN.apply(*args, self.cp_mesh, self.local_cp_rank())
        return out[:, :, :width, :].transpose(1, 2).contiguous().to(q.dtype)
