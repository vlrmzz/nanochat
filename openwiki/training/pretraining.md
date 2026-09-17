---
type: workflow
title: Pretraining (base_train)
description: End-to-end trace of scripts/base_train.py - the CLI, how --depth determines model width and heads, the scaling-law derivation of the token horizon, batch size, learning-rate and weight-decay scaling from a d12 reference, the LR, momentum and weight-decay schedules, gradient accumulation, periodic bits-per-byte, CORE and sampling, checkpointing with resume, and logging.
tags: [pretraining, base-train, scaling-laws, hyperparameters, schedules, checkpoints, resume]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-71e0a7d95ee58ab9cadfb5e6
    resource: repo://scripts/base_train.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Pretraining (base_train)

`scripts/base_train.py` is the heart of nanochat and the stage where almost all
compute goes. Its defining idea, stated in `README.md`, is one dial:
`--depth`. Everything else (width, heads, batch size, learning rates, weight
decay, training horizon) is derived so that any depth comes out roughly
compute-optimal, which is what lets `runs/miniseries.sh` sweep depths and
`runs/speedrun.sh` reach GPT-2 capability with a d24. This page follows the
script top to bottom.

## Launch

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
python -m scripts.base_train --depth=6 --window-pattern=L --max-seq-len=512 --num-iterations=5000   # laptop
```

Compute and wandb setup follow the shared pattern in
[runtime and checkpoints](../operations/runtime-and-checkpoints.md); the
script prints the `COMPUTE_DTYPE` reason and a loud warning block if Flash
Attention 3 is unavailable, recommending `--window-pattern L` for the SDPA
fallback (see [attention backends](../architecture/attention-backends.md)).
`dev/LEADERBOARD.md` documents the flag combination used for leaderboard runs
and explains each flag.

## Model sizing from depth

`build_model_meta(depth)` sets `model_dim = depth * --aspect-ratio` (default
64) rounded up to a multiple of `--head-dim` (default 128), `n_head =
n_kv_head = model_dim // head_dim`, `sequence_len = --max-seq-len` (2048), the
tokenizer's vocabulary size, and `--window-pattern` (default `SSSL`). The model
is built on the meta device, moved with `to_empty`, then `init_weights()`;
see [GPT model](../architecture/gpt-model.md). The same function builds a
d12 reference model whose parameter count anchors the scaling rules below.

Optional FP8 conversion happens here, before compilation; see
[precision and FP8](../architecture/precision-and-fp8.md). The model is then
`torch.compile`d with `dynamic=False` (training shapes never change) while
`orig_model` is kept for evaluation, sampling and saving.

## Deriving the training recipe

The script computes, in order:

1. **Scaling parameters.** `num_scaling_params = transformer_matrices + lm_head`
   (embeddings excluded), the convention `dev/LOG.md` found gives the cleanest
   scaling laws.
2. **Token horizon.** `target_tokens = --target-param-data-ratio * num_scaling_params`.
   The default ratio is 12; the README and leaderboard docs discuss lowering it
   (e.g. 8 for the speedrun) to slightly undertrain a larger depth.
3. **Batch size.** Unless `--total-batch-size` is given, it follows the Power
   Lines rule `B ∝ D^0.383` relative to the d12 reference (`B_REF = 2^19`
   tokens at the d12 horizon `D_REF`), rounded to the nearest power of two.
4. **Learning-rate scaling.** All LRs are multiplied by `sqrt(B / B_REF)`; the
   comment notes this is standard for AdamW and assumed for Muon.
5. **Weight decay.** `--weight-decay` (default 0.28, Muon matrices only) is
   scaled by `sqrt(B / B_REF) * (D_REF / D)` following the constant-`T_epoch`
   argument, again borrowed from AdamW theory.
6. **Optimizer.** `model.setup_optimizer(...)` with the scaled values; see
   [MuonAdamW](optimizer.md). On resume the per-rank optimizer shard is loaded
   here.
7. **Iterations.** Exactly one horizon source is used, in precedence order:
   `--num-iterations` if positive, else `--target-flops` (used by
   `runs/scaling_laws.sh`), else the token horizon divided by the batch size.
   At least one must be set.

The script prints the parameter breakdown, FLOPs per token, total tokens,
the realised tokens-to-params ratio and the total FLOPs estimate; the sweep
scripts scrape these lines.

## Schedules

Three step-indexed schedules mutate optimizer groups every step:

- **LR multiplier**: linear warm-up over `--warmup-steps` (40), constant,
  then linear warm-down over the final `--warmdown-ratio` (0.65) of training to
  `--final-lr-frac` (0.05).
- **Muon momentum**: 0.85 to 0.97 over the first 400 steps, held, then
  decayed to 0.90 across the warm-down.
- **Muon weight decay**: cosine decay from the scaled value to zero over the
  whole run, which is why SFT starts from zero weight decay.

## The step

Gradient accumulation is derived, not configured:
`grad_accum_steps = total_batch_size / (device_batch_size * max_seq_len * world_size)`,
asserted to divide exactly. This is how a single GPU reproduces an 8-GPU
run's optimisation trajectory at one eighth the speed, and how the sweep
scripts lower `--device-batch-size` at large depths without changing the
recipe. Each micro-step calls the compiled model, scales the loss by the
accumulation count, backpropagates (through a GradScaler under fp16), and
prefetches the next batch from the [BOS-aligned dataloader](../data/pretraining-dataset.md)
while the GPU is busy. Then the schedules are applied, `optimizer.step()`
synchronises gradients and updates, and gradients are freed.

## Periodic evaluation and sampling

At intervals, and always on the final step (the loop runs
`num_iterations + 1` times for this reason):

- validation bits per byte every `--eval-every` steps over `--eval-tokens`;
- CORE every `--core-metric-every` steps on `--core-metric-max-per-task`
  examples, on the uncompiled model;
- greedy samples for a fixed prompt list every `--sample-every` steps on rank 0.

All three run inside `disable_fp8`. See
[base model evaluation](../evaluation/base-model-evaluation.md).

## Checkpointing and resume

Checkpoints go to `<base_dir>/base_checkpoints/<tag>/` where the tag is
`--model-tag` or `d<depth>`. They are written on the last step and, if
`--save-every` is positive, every that many steps except step 0 and the step
just resumed from. Each save stores the uncompiled model state, every rank's
optimizer shard, and JSON metadata: `step`, `val_bpb`, `model_config`,
`user_config`, batch settings, the dataloader position and a `loop_state`
with the running minimum bpb, smoothed loss and accumulated training time.

`--resume-from-step N` reloads all of that from the same tag directory,
overwrites the freshly initialised weights, restores the optimizer shard for
this rank, seeds the dataloader from the saved position and continues the
schedules from step N. The world size must match the saved run because
optimizer state is sharded.

## Logging

Every step prints loss (an EMA with bias correction), LR multiplier, step
time, tokens per second, bf16 MFU against the GPU peak table, the dataloader
position and an ETA; timing excludes the first ten steps. Every 100 steps the
same values go to wandb, along with cumulative FLOPs and training time, which
is what the leaderboard reports as `total_training_time`. Python garbage
collection is frozen and disabled after the first step to avoid periodic
pauses, with a manual collection every 5000 steps.

## Changing the recipe

Because the derivation chain is explicit, most research changes land in one
of three places: the architecture in `nanochat/gpt.py` (must work for every
depth, per `README.md`), the scaling rules in the "Scaling laws" block of this
script, or the schedules. `dev/LOG.md` records ideas already tried and
rejected; check it before re-running an experiment.
