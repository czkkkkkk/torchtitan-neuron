# Copyright (c) Amazon.com, Inc. or its affiliates
# All rights reserved.
#
# Licensed under the BSD-3-Clause license.

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import CompileConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.qwen3_5.config_registry import qwen35_27b
from torchtitan.trainer import Trainer

from . import model_registry


def qwen38_27b() -> Trainer.Config:
    """Text-only TP4 training with the Neuron implementation of GDN."""
    config = qwen35_27b()
    config.model_spec = model_registry()
    config.compile = CompileConfig(enable=True, backend="neuron")
    config.optimizer.implementation = "for-loop"
    config.parallelism.enable_sequence_parallel = False
    config.parallelism.context_parallel_load_balancer = None
    config.training.local_batch_size = 1
    config.training.seq_len = 512
    config.hf_assets_path = "./assets/hf/Qwen3.8-27B"
    config.tokenizer = HuggingFaceTokenizer.Config()
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="c4")
    return config
