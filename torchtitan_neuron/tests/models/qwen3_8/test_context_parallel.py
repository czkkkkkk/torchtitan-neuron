# Copyright (c) Amazon.com, Inc. or its affiliates
# Licensed under the BSD-3-Clause license.

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed.context_parallel import causal_conv_halo
from torchtitan.models.qwen3_5.model import GatedDeltaKernel
from torchtitan_neuron.models.qwen3_8.gdn import NeuronGatedDeltaKernel


def compare(mesh, device, *, width):
    rank, peers = mesh.get_local_rank(), mesh.size()
    dim = 128 if device == "neuron" else 8
    length = peers * width
    rng = torch.Generator().manual_seed(817)
    inputs = [torch.randn(1, length, h, dim, generator=rng) for h in (1, 1, 3)]
    # Weak decay makes losses on the last shard depend on the first shard.
    inputs += [
        -torch.rand(1, length, 3, generator=rng) * 0.01,
        torch.rand(1, length, 3, generator=rng),
    ]
    upstream = torch.randn(1, length, 3, dim, generator=rng)
    upstream[:, :-width] = 0
    weights = [torch.randn(h * dim, 1, 4, generator=rng) / 4 for h in (1, 1, 3)]
    region = slice(rank * width, (rank + 1) * width)

    def evaluate(args, weights, kernel, partition):
        values = []
        for x, w in zip(args[:3], weights, strict=True):
            channels = x.flatten(2).transpose(1, 2)
            padded = (
                F.pad(channels, (3, 0))
                if partition is None
                else causal_conv_halo(channels, 3, partition, kernel.local_cp_rank())
            )
            values.append(
                F.silu(F.conv1d(padded, w, groups=w.size(0)))
                .transpose(1, 2)
                .reshape_as(x)
            )
        return kernel(*values, *args[3:])

    ref_args = [x.detach().requires_grad_() for x in inputs]
    ref_weights = [w.detach().requires_grad_() for w in weights]
    ref = evaluate(
        ref_args,
        ref_weights,
        GatedDeltaKernel.Config(backend="torch_native").build(),
        None,
    )
    ref.backward(upstream)
    with torch.device(device):
        kernel = NeuronGatedDeltaKernel.Config().build()
    if peers > 1:
        kernel.cp_mesh = mesh
        kernel._init_self_buffers()
    local_args = [x[:, region].contiguous().to(device).requires_grad_() for x in inputs]
    local_weights = [w.to(device).detach().requires_grad_() for w in weights]

    def forward(args, weights):
        return evaluate(args, weights, kernel, mesh if peers > 1 else None)

    if device == "neuron":
        forward = torch.compile(
            forward, backend="neuron", fullgraph=True, dynamic=False
        )
    out = forward(local_args, local_weights)
    out.backward(upstream[:, region].contiguous().to(device))
    for x in local_weights:
        dist.all_reduce(x.grad, group=mesh.get_group())
    mismatches = []
    for name, actual, expected in zip(
        ("output", "dq", "dk", "dv", "dg", "dbeta", "conv_q", "conv_k", "conv_v"),
        [out, *(x.grad for x in local_args), *(w.grad for w in local_weights)],
        [
            ref[:, region],
            *(x.grad[:, region] for x in ref_args),
            *(w.grad for w in ref_weights),
        ],
        strict=True,
    ):
        value = actual.detach().cpu()
        try:
            torch.testing.assert_close(value, expected.detach(), atol=4e-4, rtol=4e-3)
        except AssertionError as exc:
            mismatches.append(
                f"{name}: {exc}; norm={value.norm().item():.5f}, reference={expected.norm().item():.5f}"
            )
    assert not mismatches, "\n".join(mismatches)
    if rank == 0:
        assert local_args[2].grad.abs().max().item() > 1e-6


def _cpu_rank(rank, peers, width, store):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=peers,
        timeout=timedelta(minutes=2),
    )
    try:
        compare(
            DeviceMesh("cpu", list(range(peers)), mesh_dim_names=("cp",)),
            "cpu",
            width=width,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("peers,width", [(1, 5), (2, 5), (4, 5), (4, 1)])
def test_contiguous_cp(tmp_path, peers, width):
    torch.multiprocessing.start_processes(
        _cpu_rank,
        (peers, width, f"file://{tmp_path}/store"),
        nprocs=peers,
        start_method="fork",
    )


if __name__ == "__main__":
    dist.init_process_group("cpu:gloo,neuron:neuron")
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    try:
        compare(
            DeviceMesh(
                "neuron", list(range(dist.get_world_size())), mesh_dim_names=("cp",)
            ),
            "neuron",
            width=133,
        )
    finally:
        dist.destroy_process_group()
