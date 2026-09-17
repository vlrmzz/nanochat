---
type: architecture
title: Precision, COMPUTE_DTYPE and FP8 training
description: nanochat's explicit mixed-precision design without autocast - how COMPUTE_DTYPE is detected and overridden, fp32 master weights with a casting Linear, bf16 embeddings, the fp16 GradScaler path, and the tensorwise FP8 matmul implementation in nanochat/fp8.py with its enable/disable flow in pretraining.
tags: [precision, dtype, bfloat16, float16, fp8, gradscaler, mixed-precision]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-b4c36c62f3071600ed4e6487
    resource: repo://nanochat/common.py
  - id: openwiki-source-d22f3da3a0fceb916e861104
    resource: repo://nanochat/engine.py
  - id: openwiki-source-f40a995ba25dde0ecb06fbf5
    resource: repo://nanochat/fp8.py
  - id: openwiki-source-cbecc005733ae5979fd4c639
    resource: repo://nanochat/gpt.py
  - id: openwiki-source-b9751542d4064fb72ca59f83
    resource: repo://nanochat/optim.py
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-71e0a7d95ee58ab9cadfb5e6
    resource: repo://scripts/base_train.py
  - id: openwiki-source-ca457235ed5b9f351ff341ee
    resource: repo://scripts/chat_sft.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Precision, COMPUTE_DTYPE and FP8 training

nanochat does not use `torch.amp.autocast`. Precision is governed by one module
global, `COMPUTE_DTYPE` in `nanochat/common.py`, and by two small mechanisms
that consume it: the casting `Linear` layer in the model and, optionally, the
FP8 matmul wrapper in `nanochat/fp8.py`. `README.md` ("Precision / dtype") is
the authoritative user-facing description; this page explains how the code
implements it and what changes must respect.

## COMPUTE_DTYPE

`_detect_compute_dtype()` runs once at import of `nanochat.common`:

1. If `NANOCHAT_DTYPE` is set, it must be one of `bfloat16`, `float16` or
   `float32`; any other value raises a `KeyError`.
2. Otherwise on CUDA, compute capability 8.0 or higher (Ampere and newer) gets
   `bfloat16`; older GPUs get `float32`, because fp16 needs a GradScaler and
   the code prefers a safe default. Users can still force `float16`.
3. Without CUDA (CPU or MPS) the default is `float32`. The README notes that
   MPS on recent macOS also runs `bfloat16` when opted in.

The result is stored with a human-readable `COMPUTE_DTYPE_REASON` that the
training scripts print at start-up. Because it is decided at import time,
`NANOCHAT_DTYPE` has to be in the environment before any nanochat module is
imported; downstream import-time decisions such as FA3 selection depend on it
(see [attention backends](attention-backends.md)).

## Where the dtype is applied

- **Master weights stay fp32.** Matrix parameters are created in fp32 and the
  custom `Linear` in `nanochat/gpt.py` casts `self.weight` to the activation
  dtype inside `forward`. Optimizer math therefore sees fp32 parameters and
  gradients for all matrices.
- **Embeddings live in COMPUTE_DTYPE.** `init_weights` casts `wte` and the
  value embeddings to `COMPUTE_DTYPE` to save memory, except under fp16 where
  GradScaler cannot unscale fp16 gradients. The fused AdamW kernel in
  `nanochat/optim.py` upcasts to fp32 internally so these bf16 parameters
  update correctly, including on MPS which refuses mixed-dtype ops; see
  [MuonAdamW](../training/optimizer.md).
- **Activations.** The model explicitly casts the embedded tokens to
  `COMPUTE_DTYPE` at the start of `forward` (a no-op except on the fp16 path),
  precomputes rotary tables in `COMPUTE_DTYPE` and asserts they match, and
  casts logits back to fp32 before the softcap and cross entropy.
- **Inference.** The Engine allocates the KV cache in `COMPUTE_DTYPE` so it
  matches what attention emits.
- **Muon.** The orthogonalisation iteration is run in bf16 when
  `COMPUTE_DTYPE` is bf16 and otherwise stays in the parameter dtype, because
  fp16's exponent range is too narrow for it.

## The fp16 path

When `COMPUTE_DTYPE` is `float16`, `scripts/base_train.py` and
`scripts/chat_sft.py` create a `torch.amp.GradScaler`, scale the loss before
`backward`, unscale before stepping, and in the distributed case all-reduce the
scaler's `found_inf` flags with MAX so every rank agrees on whether to skip the
step. `scripts/chat_rl.py` has no scaler, which is why the README says RL does
not support fp16 training. Inference in fp16 needs none of this.

## FP8 training

`nanochat/fp8.py` is a roughly 150-line replacement for torchao's
`Float8Linear`, supporting only the tensorwise recipe (one scale per tensor).
Its module docstring is a good primer on FP8 training; the mechanics are:

- `_to_fp8(x, dtype)` computes `scale = FP8_MAX / amax` (division done in
  float64 so eager and compiled numerics agree), saturates with `clamp`
  because PyTorch's default cast would wrap, and returns the FP8 tensor and
  the inverse scale that `torch._scaled_mm` expects.
- `_Float8Matmul` is an `autograd.Function` decorated with
  `torch._dynamo.allow_in_graph`, so `torch.compile` treats it as one opaque
  node. Forward quantises input and weight to `float8_e4m3fn` and calls
  `_scaled_mm` with `use_fast_accum=True`. Backward quantises the incoming
  gradient to `float8_e5m2` (wider range) and performs the two remaining GEMMs
  with `use_fast_accum=False`, re-laying out operands to satisfy cuBLAS's
  row-major/column-major requirements via `_to_col_major`.
- `Float8Linear` casts its input to `COMPUTE_DTYPE`, flattens to 2-D, applies
  the function and reshapes back. `from_float` builds the shell on the meta
  device and shares the original weight tensor, so converting allocates no
  new parameter memory and the optimizer keeps pointing at the same tensors.
- `Float8LinearConfig.from_recipe_name` rejects anything but `"tensorwise"`,
  so passing `--fp8-recipe rowwise` to pretraining fails at start-up.
- `convert_to_float8_training` walks the module tree post-order and swaps
  each `nn.Linear` passing the filter for a `Float8Linear`.

### Enable/disable flow in pretraining

`--fp8` is only honoured on CUDA. `scripts/base_train.py` converts before
`torch.compile` and only for linears whose in and out features are multiples
of 16 with the smaller dimension at least 128 (a hardware requirement), then
prints how many layers were converted. Small layers such as the smear and
value-embedding gates are skipped.

Evaluation and sampling must not run in FP8, so `disable_fp8(model)` is a
context manager that temporarily replaces every `Float8Linear` with the
model's own casting `Linear` sharing the same weight (built on the meta device
to avoid a VRAM spike) and restores the FP8 modules afterwards. It wraps the
bits-per-byte evaluation, the CORE evaluation and periodic sampling. The
uncompiled `orig_model` is used for CORE and sampling because their input
shapes vary.

## Consequences for contributors

- New matmuls should go through the model's `Linear` (or be converted by the
  FP8 filter) or they will run in fp32 and be excluded from FLOP counts.
- Anything that inspects `p.dtype` must handle bf16 embeddings alongside fp32
  matrices; `infer_bench` reports a per-dtype parameter breakdown for this
  reason.
- Under `torch.compile`, the custom FP8 path produces slightly different
  rounding from torchao (different fused graphs) but identical eager numerics,
  as the module docstring explains.
