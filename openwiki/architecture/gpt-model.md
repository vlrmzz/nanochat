---
type: architecture
title: GPT model architecture
description: The Transformer in nanochat/gpt.py - config, meta-device construction and init_weights, rotary embeddings, QK norm, value embeddings, smear and backout, per-layer lambdas, sliding windows, logit softcap, FLOP and KV-cache accounting, and how parameters are grouped for the optimizer.
tags: [model, transformer, gpt, architecture, initialization, rotary, value-embeddings, optimizer-groups]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-a8aef4f65d816b1c96ddcc9e
    resource: repo://nanochat/checkpoint_manager.py
  - id: openwiki-source-cbecc005733ae5979fd4c639
    resource: repo://nanochat/gpt.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# GPT model architecture

`nanochat/gpt.py` defines the only model in the repository. It is a decoder-only
Transformer with a small number of deliberate deviations from the textbook
design, most of them borrowed from modded-nanogpt. The file also owns the
cost-accounting helpers (FLOPs, KV-cache bytes, scaling parameter counts) and
the optimizer construction, so it is the place where architectural choices
meet training and inference budgets.

## Configuration

`GPTConfig` has six fields: `sequence_len` (default 2048), `vocab_size` (32768),
`n_layer`, `n_head`, `n_kv_head` (grouped-query attention), `n_embd`, and a
`window_pattern` string (default `"SSSL"`). The pattern is tiled across layers:
`L` means full context, `S` means a quarter-context sliding window rounded up
to a multiple of 128. The last layer is always forced to `L`. Details of how
windows reach the kernels are in [attention backends](attention-backends.md).

Pretraining does not set most of these directly. `scripts/base_train.py`
derives `n_embd` from `depth * aspect_ratio` rounded up to a multiple of
`head_dim`, sets `n_head = n_kv_head = n_embd // head_dim`, and passes the
window pattern through; see [pretraining](../training/pretraining.md).

## Construction on the meta device

`GPT.__init__` is written to run under `torch.device("meta")`, so it may only
compute shapes and dtypes. Every real value, including the rotary tables, is
produced later by `init_weights()`. The two-step protocol is
`GPT(config)` under meta, then `model.to_empty(device=...)`, then
`model.init_weights()`. Both `scripts/base_train.py` and
`checkpoint_manager.build_model` follow it; the latter still calls
`init_weights` before `load_state_dict` because the rotary buffers are
non-persistent and would otherwise be uninitialised.

The vocabulary is padded up to a multiple of 64 (`pad_vocab_size_to`) for
tensor-core efficiency. Padding is purely internal: `forward` slices logits
back to `config.vocab_size` before the softcap and loss.

## Components

- **`Linear`** subclasses `nn.Linear` and casts its fp32 weight to the input
  dtype at every call. This is the mechanism that replaces autocast; see
  [precision and FP8](precision-and-fp8.md). `num_matmul_params` counts
  parameters structurally by summing over `Linear` modules, so any new
  matmul must go through this class or FLOP estimates will be wrong.
- **Token embedding** (`wte`) is followed by an RMS norm (parameter-free
  `F.rms_norm`) before entering the trunk.
- **Smear**: a gated mix of the previous token's normalised embedding into
  the current position, gate computed from the first 24 channels. In KV-cache
  decode the previous embedding is read from `kv_cache.prev_embedding`, which
  the model also updates, so the [Engine](../inference/engine.md) has to
  carry that state when it replicates a cache.
- **Per-layer scalars**: `resid_lambdas[i]` scales the residual stream and
  `x0_lambdas[i]` blends the post-norm embedding `x0` back in at every layer.
  Initial values decay with depth (1.15 to 1.05 and 0.20 to 0.05).
- **Attention** (`CausalSelfAttention`): separate `c_q`, `c_k`, `c_v`, `c_proj`
  projections with no bias, rotary embeddings applied to q and k (rotating by
  `-theta`, the transpose of the textbook convention, kept for checkpoint
  compatibility), QK norm, then both q and k scaled by 1.2 for sharper
  attention. Layers selected by `has_ve` (alternating, always including the
  last layer) add a **value embedding**: a per-layer `nn.Embedding` of the
  input token ids, mixed into `v` through a per-KV-head sigmoid gate in
  `(0, 3)` computed from the first 12 channels of the input.
- **MLP**: 4x expansion with `relu(x)^2` activation, no bias.
- **Backout**: the residual after the middle layer (`n_layer // 2`) is cached
  and `backout_lambda * x_backout` is subtracted before the final norm.
- **Head**: untied `lm_head`, logits cast to fp32 and squashed with
  `15 * tanh(logits / 15)`. With targets, `forward` returns cross entropy with
  `ignore_index=-1` and the requested reduction; without, it returns logits.

## Initialisation

`init_weights` documents the full scheme in one place: `wte` normal with std
0.8, `lm_head` normal with std 0.001, attention and MLP input matrices uniform
with bound `sqrt(3)/sqrt(n_embd)` (the MLP `c_fc` at 0.4x that), all output
projections (`c_proj`) zero, value embeddings like `c_v`, gates uniform in
`[0, 0.02]`, and the scalar schedules above. After initialisation, `wte` and
the value embeddings are cast to `COMPUTE_DTYPE` unless it is fp16 (GradScaler
cannot unscale fp16 gradients), which is why the optimizer must tolerate bf16
parameters; see [MuonAdamW](../training/optimizer.md).

Rotary tables are precomputed for `10 * sequence_len` positions with base
100000 and stored in `COMPUTE_DTYPE`; `forward` asserts the sequence fits and
that the dtype matches.

## Cost accounting

The model carries the formulas that the training and benchmarking scripts rely
on:

- `estimate_flops()`: training FLOPs per token as `6 * matmul_params` plus an
  attention term of `12 * n_head * head_dim * effective_seq` per layer, where
  `effective_seq` is capped by that layer's window.
- `estimate_decode_flops(context_len)` and `estimate_prefill_flops(n)`: the
  forward-only analogues used by the [inference benchmark](../evaluation/inference-benchmark.md).
- `kv_bytes_per_token()` and `kv_read_bytes(context_len)`: KV-cache storage
  and per-step read traffic across layers, window-aware, in `COMPUTE_DTYPE`.
- `num_scaling_params()`: a dict of counts for `wte`, `value_embeds`,
  `lm_head`, `transformer_matrices` and `scalars`, asserted to sum to the total.
  Pretraining uses `transformer_matrices + lm_head` as the "scaling params"
  that set the token horizon.

## Optimizer grouping

`setup_optimizer` partitions every parameter into groups and asserts that
nothing is left out. AdamW groups (with per-group betas, eps `1e-10` and
weight decay) cover `lm_head`, `wte`, value embeddings, `resid_lambdas`,
`x0_lambdas`, and the smear/backout scalars; their learning rates are scaled
by `(n_embd / 768) ** -0.5` relative to a d12 reference. Muon groups are formed
from the transformer block matrices, one group per distinct tensor shape so the
optimizer can stack them. The result is a `MuonAdamW` instance with
`initial_lr` recorded on each group for the schedules in the training scripts.

## Inference entry points

`forward(idx, kv_cache=...)` offsets the rotary tables by the cache position
and lets attention write into the cache; the last layer advances the cache.
`GPT.generate` is a deliberately naive batch-1 sampler used as the reference
that the Engine is checked against in `nanochat/engine.py`'s `__main__`.

## Checkpoint compatibility

`checkpoint_manager` patches older checkpoints: a missing `window_pattern`
config key defaults to `"L"`, and missing `resid_lambdas`/`x0_lambdas`
tensors are filled with 1.0 and 0.0. Any new parameter added to the model
needs the same treatment or old checkpoints will fail the strict
`load_state_dict`. See [runtime and checkpoints](../operations/runtime-and-checkpoints.md).
