---
type: component
title: MuonAdamW optimizer
description: The combined optimizer in nanochat/optim.py - a fused AdamW kernel for embeddings and scalars, a fused Muon kernel (Polar Express orthogonalisation with MuonEq, Muon+ and NorMuon variance reduction and cautious weight decay) for matrices, parameter groups stacked by shape, the three-phase asynchronous communication that shards state ZeRO-2 style across ranks, single-rank degeneration, and what all of this means for checkpoints.
tags: [optimizer, muon, adamw, distributed, zero-2, torch-compile, sharding]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-a8aef4f65d816b1c96ddcc9e
    resource: repo://nanochat/checkpoint_manager.py
  - id: openwiki-source-b9751542d4064fb72ca59f83
    resource: repo://nanochat/optim.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# MuonAdamW optimizer

`nanochat/optim.py` defines one `torch.optim.Optimizer` subclass,
`MuonAdamW`, that applies AdamW to some parameter groups and Muon to others
and also performs **all gradient communication** for distributed training.
nanochat does not use `DistributedDataParallel`: gradients are averaged
inside `step()`, and optimizer state is sharded across ranks. A recent commit
unified what were two implementations (single-GPU and distributed) into this
one class so that the code paths cannot drift apart; on a single rank the
communication calls are simply skipped.

Parameter grouping is decided by the model in `GPT.setup_optimizer`: AdamW
for `lm_head`, `wte`, value embeddings and the per-layer scalars, Muon for
transformer block matrices with one group per distinct shape. See
[GPT model](../architecture/gpt-model.md). Each group dict carries
`kind='adamw'` or `kind='muon'` and its own hyperparameters.

## Fused kernels

Both update rules are single `torch.compile(dynamic=False, fullgraph=True)`
functions. Hyperparameters are passed as 0-D CPU tensors that the optimizer
fills in before each call, so changing the learning rate, momentum or weight
decay during training does not trigger recompilation.

### AdamW

`adamw_step_fused` performs decoupled weight decay (`p *= 1 - lr * wd`),
`lerp_`-based moment updates, bias correction and the parameter update in one
graph. It upcasts `p`, the gradient and both moments to fp32 and copies back at
the end, so bf16 parameters such as `wte` and the value embeddings update
correctly; the comment notes that MPS refuses mixed-dtype ops and that
`1 - beta2` loses all precision in bf16. `test_adamw_matches_torch_reference`
checks agreement with `torch.optim.AdamW`.

### Muon

`muon_step_fused` operates on a stack of same-shaped matrices `(K, m, n)`:

1. Nesterov momentum via two `lerp_`s.
2. Cast to bf16 when `COMPUTE_DTYPE` is bf16 (fp16 is unstable here).
3. **MuonEq** row equilibration: rescale each row to the mean row norm.
4. **Polar Express** orthogonalisation: normalise by the Frobenius norm, then
   `ns_steps` iterations with precomputed quintic coefficients, using the
   `X^T X` form for tall matrices and `X X^T` for wide ones.
5. **Muon+** renormalisation: snap the Frobenius norm to `sqrt(min(m, n))`.
6. **NorMuon** variance reduction: a factored second-moment buffer of shape
   `(K, m, 1)` or `(K, 1, n)` (reducing over the longer dimension) rescales
   per row or column while preserving the overall update norm.
7. **Cautious weight decay**: decay is applied only where the update and the
   parameter have the same sign, then `p -= lr * g + lr * wd * p * mask`.

The module docstring cites the papers for each component. The Muon learning
rate for a group is additionally scaled by `sqrt(max(1, m / n))` at call time.
`test_muon_update_is_orthogonalized` checks the first update of a random
gradient has singular values within a 4x band.

## Distributed algorithm

`step()` runs three phases so that communication overlaps computation:

1. **Launch all reduces.** For each group, kick off asynchronous reduce ops
   and keep their futures.
2. **Wait, compute, launch gathers.** For each group in order: wait for its
   reduce, run the fused kernel on the slice this rank owns, and start the
   asynchronous all-gather of the updated slice. Later groups' computation
   overlaps earlier groups' gathers.
3. **Wait for gathers and copy back.** Muon results are copied from the
   gathered stack back into the individual parameter tensors with
   `torch._foreach_copy_`.

### AdamW groups (ZeRO-2 style)

Parameters with fewer than 1024 elements are all-reduced (`AVG`) and updated
in full on every rank; their optimizer state is replicated but tiny. Larger
parameters are `reduce_scatter`ed along dimension 0, so each rank receives and
updates a `1/world_size` slice and keeps `exp_avg`/`exp_avg_sq` only for that
slice; the updated slices are then `all_gather`ed back into the full parameter.
This requires `shape[0]` to be divisible by the world size, which is asserted;
the padded vocabulary (a multiple of 64) satisfies it for typical world sizes.

### Muon groups (stacked and chunked)

All parameters in a Muon group share a shape. Their gradients are stacked
into one `(K, m, n)` tensor, zero-padded to `ceil(K / world_size) * world_size`,
and `reduce_scatter`ed so each rank owns a contiguous chunk of whole matrices.
Each rank runs the fused kernel only on its owned matrices (possibly none),
keeping `momentum_buffer` and `second_momentum_buffer` for that chunk, then
`all_gather`s the updated matrices. The stacked-gradient buffer is reused as
the gather output to save memory, and padding is ignored on copy-back.

### Single rank

With no multi-rank process group, `_reduce_*` return the full gradients with
no futures, the owned chunk is the whole stack, and no gathers are issued; the
same code produces plain full-tensor updates. This is what makes single-GPU,
CPU and MPS training identical in semantics to multi-GPU training.

## State layout and checkpoints

Because state is sharded, each rank's `state_dict()` differs, which is why
`save_checkpoint` writes one `optim_<step>_rank<r>.pt` per rank and resume or
warm start must load the shard for the same rank (see
[runtime and checkpoints](../operations/runtime-and-checkpoints.md)).
Consequences:

- Resuming or warm-starting requires the **same world size** as the run that
  saved the state; a different rank count changes the slicing.
- Muon state is keyed on the *first* parameter of each group (`self.state[p]`
  for `params[0]`), so group composition and ordering must match at load time.
- `load_state_dict` overwrites group hyperparameters (learning rates, betas,
  momentum); `chat_sft` explicitly restores its own learning rates after
  loading the pretraining shard. See [chat SFT](chat-sft.md).

## Schedules driven from outside

The optimizer itself has no schedule. Training scripts mutate
`group["lr"]`, `group["momentum"]` and `group["weight_decay"]` each step
(warm-up/warm-down for LR, momentum ramps, cosine weight-decay decay in
pretraining) and the 0-D tensors pick up the new values without recompiling.
See [pretraining](pretraining.md).

## Tests

`tests/test_optim.py` is CUDA-only (kernel compilation dominates). It covers
reference agreement for AdamW, bitwise determinism across runs, convergence on
a distance-to-target objective for all group kinds and shapes, and
orthogonality of the Muon update. The distributed paths have no unit test and
are validated by multi-GPU training runs.
