# Qwen3.8-27B text decoder

The Neuron registry and training logs use `qwen3_8` for Qwen3.8-27B. Its
Hugging Face checkpoint declares `model_type="qwen3_5"` and
`text_config.model_type="qwen3_5_text"`. Those are architecture identifiers:
the Neuron recipe reuses the decoder and weight mapping in
[`torchtitan.models.qwen3_5`](../../../torchtitan/models/qwen3_5).

Build the text model with
`torchtitan_neuron.models.qwen3_8.from_hf_config(hf_config)`, or select
`qwen38_27b` from this model's trainer configuration registry. The text
adapter ignores vision and MTP weights on import. It does not train or export
those components. A four-layer prefix of the 27B model is supported for tests.

Training requires both `gdn_cte` and `gdn_cte_bwd` from
`nkilib.experimental.gdn`. SDK 2.33.1 bundles only the forward
kernel. Install the [public NKI Library](https://github.com/aws-neuron/nki-library/tree/ad2f1b921d8e5949fa3b9a186e5f9130269353dc)
in the trainer environment without changing the SDK packages:

```bash
python -m pip install --no-deps 'https://github.com/aws-neuron/nki-library/archive/ad2f1b921d8e5949fa3b9a186e5f9130269353dc.tar.gz'
python -c 'from nkilib.experimental.gdn import gdn_cte, gdn_cte_bwd'
```

For a TorchTitan training run, put the tokenizer and model assets in
`./assets/hf/Qwen3.8-27B` and configure the C4 text dataset. Select the
recipe with

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=1 NGPU=32 \
  MODULE=torchtitan_neuron.models.qwen3_8 CONFIG=qwen38_27b \
  ./run_train.sh --checkpoint.enable --checkpoint.initial_load_in_hf
```

Use a fresh dump folder to initialize from HF; an existing TorchTitan checkpoint
takes precedence. The default is one 512-token sequence per data-parallel rank,
with TP4 and full activation checkpointing. The C4 loader packs documents:
both causal SDPA and the GDN recurrence carry context across document boundaries
within a sequence. Use one document per sequence if you need isolation. See the
[checkpoint guide](../../../docs/checkpoint.md) for overriding the HF checkpoint
location or resuming a previous run.

On SDK 2.33.1, Neuron process groups keep their initial timeout; PyTorch's
post-warmup timeout update only affects backends that implement it.

Use `python -m pytest torchtitan_neuron/tests/models/qwen3_8` on a
Trainium node with a visible `/dev/neuron0`.

[`nki_gdn.py`](nki_gdn.py) registers the TorchTitan operators and their
autograd rule. Both Schur kernels come from the installed nkilib.

## Context-parallel training

Set `--parallelism.context_parallel_degree 2` (or a higher divisor of the
sequence length) with the `qwen38_27b` recipe. The recipe selects contiguous
CP shards. The GDN layers all-gather projected convolution tails, then an FP32
affine transition and additive state per local value head. The backward pass
combines terminal cotangents in reverse sequence order. Local grouping of key
and value heads follows TP. CP1 retains the zero-entry Schur kernel.

The pinned NKI backward kernel accepts up to 49,152 local tokens at head width
128. For example, CP2 at 65,536 global tokens fits that kernel limit. Total
model memory and the full-attention backend also determine usable length.

The Neuron port implements the contiguous, text-only case of [upstream
state-based CP](https://github.com/pytorch/torchtitan/blob/b490c9ddcae2a1e6f4ee425c385e92ddaae43282/torchtitan/models/common/attention/cp_linear_attention.py)
with the NKI Schur kernels. The full-attention layers use TorchTitan SDPA CP.
At head width 256 the Neuron SDK uses its rank-dependent ring fallback. The
trainer compiles the GDN blocks and dense FFNs inside the activation-checkpoint
boundary and executes this ring eagerly.
For TP with CP4 or greater the recipe defaults `NEURON_USE_NKI_ALLTOALLV` to
`0`. The SDK's padded all-to-all supports these strided CP groups in ring backward.
