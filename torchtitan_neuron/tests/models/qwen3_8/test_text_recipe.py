# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

from types import SimpleNamespace

import pytest
import torch

from torchtitan.models.qwen3_5 import model_registry as core_model_registry
from torchtitan_neuron.models.qwen3_8 import from_hf_config, model_registry
from torchtitan_neuron.models.qwen3_8.config_registry import qwen38_27b


def _text_config():
    from transformers import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        hidden_size=5120,
        intermediate_size=17408,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_hidden_layers=4,
        vocab_size=248320,
        tie_word_embeddings=False,
        attn_output_gate=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        },
    )


def test_text_checkpoint_configuration():
    text = _text_config()
    prefix = from_hf_config(SimpleNamespace(text_config=text)).model
    assert len(prefix.layers) == 4 and prefix.vision_encoder is None
    assert prefix.layers[-1].delta_net is None
    text.layer_types = list(reversed(text.layer_types))
    with pytest.raises(ValueError, match="layer layout"):
        from_hf_config(text)


def test_text_recipe_keeps_core_vision_default():
    recipe = qwen38_27b()
    assert recipe.model_spec.name == "qwen3_8"
    assert recipe.optimizer.implementation == "for-loop"
    assert recipe.model_spec.model.vision_encoder is None
    assert (
        core_model_registry("27B", attn_backend="sdpa").model.vision_encoder is not None
    )


def test_text_checkpoint_filters_vision_and_preserves_untied_head():
    spec = model_registry()
    adapter = spec.state_dict_adapter(spec.model, None)
    embed = torch.arange(8).view(2, 4)
    head = embed + 1
    qkv = torch.arange((16 + 16 + 48) * 128).view(-1, 1)
    hf = {
        "lm_head.weight": head,
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": qkv,
        "model.visual.blocks.0.attn.qkv.weight": torch.ones(3, 1),
        "mtp.fc.weight": torch.ones(1),
    }
    exported = adapter.to_hf(adapter.from_hf(hf))
    assert set(exported) == set(hf) - {
        "model.visual.blocks.0.attn.qkv.weight",
        "mtp.fc.weight",
    }
    for name, value in exported.items():
        torch.testing.assert_close(value, hf[name], rtol=0, atol=0)
    del hf["lm_head.weight"]
    with pytest.raises(ValueError, match="Untied text checkpoint"):
        adapter.from_hf(hf)
