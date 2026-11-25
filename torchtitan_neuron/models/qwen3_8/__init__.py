# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

"""Text decoder of Qwen3.8-27B for training on Trainium."""

import os
from typing import Any, cast

from torchtitan.distributed import ParallelDims
from torchtitan.models.qwen3_5 import (
    model_registry as core_model_registry,
    parallelize_qwen3_5,
)
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.protocols.model_spec import ModelSpec

from .gdn import NeuronGatedDeltaKernel


def parallelize_qwen38(model, *, parallel_dims: ParallelDims, **kwargs):
    if parallel_dims.tp_enabled and parallel_dims.cp >= 4:
        # The native all_to_all_v kernel requires aligned, contiguous groups.
        # SDPA ring backward permutes equal-sized buffers across strided CP peers.
        os.environ.setdefault("NEURON_USE_NKI_ALLTOALLV", "0")
    return parallelize_qwen3_5(model, parallel_dims=parallel_dims, **kwargs)


def model_registry(flavor: str = "27B", *, attn_backend: str = "sdpa") -> ModelSpec:
    """Build the text decoder from its stock checkpoint layout, without vision or MTP."""
    if flavor != "27B":
        raise ValueError(
            f"Qwen3.8-27B text training does not support flavor {flavor!r}"
        )
    spec = core_model_registry(flavor, attn_backend=attn_backend)
    spec.name = "qwen3_8"
    spec.parallelize_fn = parallelize_qwen38
    model = cast(Qwen35Model.Config, spec.model)
    model.vision_encoder = None
    for layer in model.layers:
        if layer.delta_net is not None:
            layer.delta_net.kernel = NeuronGatedDeltaKernel.Config()
    return spec


def from_hf_config(config: Any, *, attn_backend: str = "sdpa") -> ModelSpec:
    """Resolve a Qwen3.8-27B text checkpoint or a layer prefix of it."""
    text = getattr(config, "text_config", None) or config
    model_type = getattr(text, "model_type", None)
    if model_type != "qwen3_5_text":
        raise ValueError(
            f"Qwen3.8-27B requires HF text model_type='qwen3_5_text', got {model_type!r}"
        )
    shape = {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 248320,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
    }
    for field, expected in shape.items():
        if getattr(text, field, None) != expected:
            raise ValueError(f"Qwen3.8-27B requires {field}={expected}")
    rope = getattr(text, "rope_parameters", None) or {}
    if (
        rope.get("rope_type", "default"),
        rope.get("rope_theta"),
        rope.get("partial_rotary_factor", getattr(text, "partial_rotary_factor", None)),
    ) != ("default", 10_000_000, 0.25):
        raise ValueError(
            "The 27B text decoder requires default partial RoPE with theta=1e7"
        )
    layers = text.num_hidden_layers
    if not 4 <= layers <= 64 or list(text.layer_types) != [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(layers)
    ]:
        raise ValueError("Expected a prefix of the 27B GDN/full-attention layer layout")
    spec = model_registry(attn_backend=attn_backend)
    model = cast(Qwen35Model.Config, spec.model)
    model.layers = model.layers[:layers]
    return spec
