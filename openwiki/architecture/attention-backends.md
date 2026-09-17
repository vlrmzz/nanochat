---
type: architecture
title: Flash Attention 3 and SDPA fallback
description: How nanochat/flash_attention.py picks Flash Attention 3 or PyTorch SDPA at import time, the shared (B, T, H, D) API, sliding-window and GQA handling, KV-cache semantics in both backends, and what the fallback costs.
tags: [attention, flash-attention, sdpa, kv-cache, sliding-window, gqa]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-227fa48b83a8a21de101560a
    resource: repo://nanochat/flash_attention.py
  - id: openwiki-source-cbecc005733ae5979fd4c639
    resource: repo://nanochat/gpt.py
  - id: openwiki-source-71e0a7d95ee58ab9cadfb5e6
    resource: repo://scripts/base_train.py
  - id: openwiki-source-afe930b9c8fd4b271b7d6406
    resource: repo://tests/test_attention_fallback.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Flash Attention 3 and SDPA fallback

`nanochat/flash_attention.py` is the single attention entry point for the whole
codebase. It exports a `flash_attn` namespace with two functions that mirror the
Flash Attention 3 (FA3) interface exactly, `flash_attn_func` for training and
`flash_attn_with_kvcache` for inference, and decides once at import time whether
calls go to the real FA3 kernel or to a PyTorch `scaled_dot_product_attention`
(SDPA) re-implementation. The GPT model only ever imports this module, never
FA3 directly, so every other part of nanochat is backend-agnostic.

## Backend selection

Selection happens in two steps when the module is imported:

1. **Availability (`HAS_FA3`).** `_load_flash_attention_3` returns `None` unless
   CUDA is available. On a Hopper GPU (compute capability major 9) it loads the
   `varunneal/flash-attention-3` kernel from the Hugging Face `kernels` hub,
   which the code notes performs better on H100. On other CUDA GPUs it loads
   `kernels-community/flash-attn3` if that kernel exists for the device. Any
   exception during loading is swallowed and treated as "not available", so a
   missing `kernels` package or an unsupported architecture (Blackwell is called
   out as needing the fallback) silently degrades to SDPA.
2. **Use decision (`USE_FA3`).** `_resolve_use_fa3` honours a test-only
   `_override_impl` (`'fa3'` asserts FA3 is present, `'sdpa'` forces the
   fallback) and otherwise uses FA3 only when `HAS_FA3` is true **and**
   `COMPUTE_DTYPE` is bfloat16. The Hopper kernels support bf16 and fp8 only,
   so fp16 and fp32 runs fall back to SDPA even on an H100. See
   [precision and FP8](precision-and-fp8.md) for how `COMPUTE_DTYPE` is chosen.

Because `USE_FA3` is a module global evaluated once, changing it later requires
re-running `_resolve_use_fa3`, which is exactly what the test helper
`set_impl` in `tests/test_attention_fallback.py` does.

## API contract

Both public functions take and return tensors in FA3's native `(B, T, H, D)`
layout, which is why `CausalSelfAttention` in `nanochat/gpt.py` reshapes
projections directly into that layout with no transposes. `window_size` is a
`(left, right)` tuple: `left` is how many previous tokens may be attended
(`-1` meaning unlimited) and `right` is always `0` for causal models. The GPT
model builds a per-layer `window_sizes` list from the `window_pattern` string,
rounds the short window up to a multiple of 128 (FA3's tile size), and forces
the final layer to full context. See [GPT model architecture](gpt-model.md).

Grouped-query attention (GQA) is supported by both backends: FA3 handles fewer
KV heads natively, while the SDPA path detects `q.size(1) != k.size(1)` after
transposing and passes `enable_gqa=True`.

## SDPA fallback semantics

The fallback transposes to SDPA's `(B, H, T, D)` layout, calls
`_sdpa_attention`, and transposes back. `_sdpa_attention` has three cases:

- **Full context, equal lengths** (window unlimited or wider than `Tq`, and
  `Tq == Tk`): plain `is_causal=True` SDPA, the fast path used for training
  with `--window-pattern L`.
- **Single-token decode** (`Tq == 1`): no mask is needed; if the window is
  smaller than the cache length the key/value tensors are sliced to the last
  `window + 1` positions. A dedicated regression test
  (`test_kvcache_single_token_sliding_window`) exists because an earlier
  version ignored the window here and attended to the whole cache.
- **Everything else** (sliding window during training, or chunked prefill
  where `Tq != Tk`): an explicit boolean mask is materialised. Row indices are
  offset by `Tk - Tq` so causality aligns with the cache position, and the
  window constraint `(row - col) <= window` is ANDed in when it binds.

The explicit-mask path is why pretraining warns loudly when FA3 is unavailable
and the window pattern is not `L`: `scripts/base_train.py` prints that SDPA has
no native sliding-window support and that GPU utilisation "will be terrible",
recommending `--window-pattern L`. `scripts/chat_sft.py` prints a milder
efficiency warning.

## KV cache semantics

`flash_attn_with_kvcache` receives pre-allocated `k_cache`/`v_cache` tensors of
shape `(B, T_max, H_kv, D)`, the new `k`/`v` for the current step, and an int32
`cache_seqlens` tensor holding the current position per batch row. FA3 writes
the new keys and values into the cache in place. The SDPA fallback reproduces
that side effect: it reads `pos` from `cache_seqlens[0]` (assuming all rows are
at the same position), copies `k`/`v` into `[pos, pos + T_new)`, then attends
over the cache prefix `[:pos + T_new]`. Neither backend advances
`cache_seqlens`; the `KVCache` object in `nanochat/engine.py` owns the
position and the model advances it once after the last layer runs, so that
every layer in one forward pass sees the same offset. See
[Inference Engine](../inference/engine.md).

## Testing

`tests/test_attention_fallback.py` is split by hardware constraint:

- `TestFA3VsSDPA` is skipped without FA3. It runs the same bf16 inputs through
  both backends via `run_both_impls` and asserts closeness for causal, full
  context, sliding window, GQA, larger shapes, prefill, single-token decode,
  windowed decode, and backward gradients.
- `TestSDPAOnly` runs on any device with the device-appropriate dtype and
  checks forward shapes, gradient flow, and `KVCache` interaction.
- `TestOverrideMechanism` checks that the override knob actually flips
  `USE_FA3`.

See [Test suite](../testing/test-suite.md) for how to run these.

## Consequences for contributors

- Any new attention feature must work in both backends or the CPU/MPS path
  breaks; the comparison tests are the contract.
- The SDPA decode path assumes a uniform position across the batch, which
  holds because the Engine prefills once and replicates the cache.
- Import-time selection means dtype and device decisions must be final before
  `nanochat.flash_attention` is first imported; the `NANOCHAT_DTYPE`
  environment variable is read in `nanochat/common.py` before that point.
