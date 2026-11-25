# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

import os

import pytest
import torch

from torchtitan.models.qwen3_5.model import GatedDeltaKernel
from torchtitan_neuron.models.qwen3_8 import model_registry, NeuronGatedDeltaKernel


def test_27b_checkpoint_layout():
    spec = model_registry()
    config = spec.model
    assert config.vision_encoder is None
    assert len(config.layers) == 64
    assert config.vocab_size == 248320
    for index, layer in enumerate(config.layers):
        if (index + 1) % 4 == 0:
            assert layer.delta_net is None
            assert layer.attention.n_heads == 24
        else:
            assert isinstance(layer.delta_net.kernel, NeuronGatedDeltaKernel.Config)
            assert layer.delta_net.in_proj_q.out_features == 16 * 128
            assert layer.delta_net.in_proj_v.out_features == 48 * 128


@pytest.mark.parametrize("device", ["cpu", "neuron"])
@pytest.mark.parametrize("length,head_dim,key_heads", [(133, 32, 1), (256, 128, 4)])
def test_grouped_heads_forward_backward(device, length, head_dim, key_heads):
    if device == "neuron":
        if not os.path.exists("/dev/neuron0"):
            pytest.skip("requires Trainium")
        import torch_neuronx  # noqa: F401

    # Grouped heads (3:1) and a partial second 128-token chunk.
    generator = torch.Generator().manual_seed(17)
    value_heads = 3 * key_heads
    inputs = [
        torch.randn(1, length, heads, head_dim, generator=generator)
        for heads in (key_heads, key_heads, value_heads)
    ]
    inputs.extend(
        [
            -torch.rand(1, length, value_heads, generator=generator),
            torch.rand(1, length, value_heads, generator=generator),
        ]
    )
    upstream = torch.randn(1, length, value_heads, head_dim, generator=generator)

    def run(module, target):
        args = [x.to(target).detach().requires_grad_() for x in inputs]
        result = module(*args)
        with torch.autograd.set_multithreading_enabled(False):
            grads = torch.autograd.grad(result, args, grad_outputs=upstream.to(target))
        return [x.detach().cpu() for x in (result, *grads)]

    reference = run(GatedDeltaKernel.Config(backend="torch_native").build(), "cpu")
    with torch.device(device):
        candidate = NeuronGatedDeltaKernel.Config().build()
    if device == "neuron":
        candidate = torch.compile(
            candidate, backend="neuron", fullgraph=True, dynamic=False
        )
    for actual, expected in zip(run(candidate, device), reference, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-3)


def test_neuron_schur_nonzero_entry_and_terminal_gradient():
    if not os.path.exists("/dev/neuron0"):
        pytest.skip("requires Trainium")
    import torch_neuronx  # noqa: F401
    from nkilib.experimental.gdn import build_mask_pack
    from torchtitan_neuron.models.qwen3_8.context_parallel import _reference
    from torchtitan_neuron.models.qwen3_8.nki_gdn import state_backward, state_forward

    rng = torch.Generator().manual_seed(311)
    batch, heads, dim, length = 1, 3, 128, 256
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(batch, 1, length, dim, generator=rng), dim=-1
        )
        for _ in range(2)
    ]
    v = torch.randn(batch, heads, length, dim, generator=rng)
    g = -torch.rand(batch, heads, length, generator=rng) * 0.01
    beta = torch.rand(batch, heads, length, generator=rng)
    state = torch.randn(batch, heads, dim, dim, generator=rng)
    inputs = [x.requires_grad_() for x in (q, k, v, g, beta, state)]
    expected = _reference(*inputs, None, dim**-0.5)
    dy, ds = [torch.randn(x.shape, generator=rng) for x in expected]
    gradients = torch.autograd.grad(expected, inputs, (dy, ds))
    mask = torch.from_numpy(build_mask_pack()).to("neuron")
    args = [x.detach().to("neuron") for x in inputs]

    def run(*args):
        return (
            *state_forward(*args, mask, dim**-0.5),
            *state_backward(*args, dy_device, ds_device, mask, dim**-0.5),
        )

    dy_device, ds_device = dy.to("neuron"), ds.to("neuron")
    actual = torch.compile(run, backend="neuron", fullgraph=True)(*args)
    torch.testing.assert_close(args[-1].cpu(), state.detach(), rtol=0, atol=0)
    for got, reference in zip(actual, (*expected, *gradients), strict=True):
        torch.testing.assert_close(got.cpu(), reference.detach(), atol=4e-4, rtol=4e-3)
