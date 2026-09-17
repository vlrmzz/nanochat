---
type: workflow
title: Reinforcement learning on GSM8K (chat_rl)
description: How scripts/chat_rl.py fine-tunes the SFT model with a simplified GRPO that reduces to REINFORCE - rollouts sampled through the Engine, binary rewards from the GSM8K task, masking of prompt and forced tool tokens, mean-subtracted advantages with token-level normalisation, per-rank example sharding, pass@k evaluation and checkpointing.
tags: [rl, grpo, reinforce, gsm8k, rollouts, policy-gradient, post-training]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-b657b01e69409a21afb5d521
    resource: repo://scripts/chat_rl.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Reinforcement learning on GSM8K (chat_rl)

`scripts/chat_rl.py` is the optional last training stage. It loads the SFT
checkpoint and optimises it directly for GSM8K correctness with a
policy-gradient method the script calls "GRPO" in quotes, because its
docstring lists four simplifications that make it closer to REINFORCE: no
trust region or KL penalty against a reference model, no PPO ratio clipping
(training is on-policy), DAPO-style token-level rather than sequence-level
normalisation, and mean-subtracted rather than z-scored advantages.
`runs/speedrun.sh` does not run this stage.

## Setup

The script loads `load_model("sft", ...)` in eval mode and wraps it in an
[Engine](../inference/engine.md) for sampling. Training data is
`GSM8K(main, train)`, evaluation data `GSM8K(main, test)`. The number of
optimisation steps is `len(train) // examples_per_step * num_epochs`. The
optimizer is the model's standard `setup_optimizer` with RL-specific learning
rates (defaults: embedding 0.2, unembedding 0.004, matrix 0.02, weight decay
0), every group scaled by `--init-lr-frac` (default 0.05) and then decayed
linearly to zero over the run. There is no `torch.compile` and no GradScaler,
so fp16 is unsupported here.

## Rollouts

`get_batch` is a generator that cycles over the training examples assigned to
this rank (`range(rank, len(train), world_size)`), one example per yield:

1. Render the prompt with `render_for_completion` (question plus
   `<|assistant_start|>`), recording the prefix length.
2. Sample `--num-samples` completions (default 16) in chunks of
   `--device-batch-size` (default 8) with `engine.generate_batch` at
   `--temperature` 1.0 and `--top-k` 50, up to `--max-new-tokens` 256. The seed
   mixes the step, example index and chunk index so chunks differ.
3. Score each completion by decoding the generated suffix and calling
   `GSM8K.reward`, which is the same `####`-answer comparison as evaluation,
   returned as a float (see [tasks](../data/tasks.md)).
4. Right-pad sequences with `<|assistant_end|>` and masks with 0, build
   `inputs = ids[:, :-1]` and `targets = ids[:, 1:]`, and set targets to `-1`
   wherever the Engine's mask is 0. Because the Engine marks both prompt
   tokens and forced tool-output tokens as 0, neither the question nor
   calculator results receive gradient; only tokens the policy actually
   sampled do.
5. Compute advantages as `rewards - rewards.mean()` over the samples of that
   one example.

## Objective

For each of the `examples_per_rank = examples_per_step / world_size` examples
per step, the rollout batch is processed in passes of `device_batch_size`
rows. Per pass, `logp = -model(inputs, targets, loss_reduction='none')` gives
per-token log-probabilities (zero at ignored positions), the objective is
`sum(logp * advantage)` divided by the number of valid tokens, the number of
passes and `examples_per_rank`, and `loss = -objective` is backpropagated.
Gradients thus accumulate across all passes and examples before a single
`optimizer.step()`; gradient synchronisation across ranks happens inside the
optimizer as in every other stage (see [MuonAdamW](optimizer.md)). Mean
reward and mean sequence length are all-reduced for logging.

## Evaluation

Every `--eval-every` steps (default 60) the script runs `run_gsm8k_eval` on
`--eval-examples` (default 400) test problems with `device_batch_size`
samples each at temperature 1.0, strided across ranks, and computes pass@k for
`k = 1..device_batch_size`: a problem counts for pass@k if any of its first k
samples is correct. Counts are all-reduced and logged as `pass@k`. This is
distinct from the greedy single-sample GSM8K number in
[chat evaluation](../evaluation/chat-evaluation.md), which is the comparable
metric after training via `scripts.chat_eval -i rl`.

## Checkpointing

Rank 0 saves the model (no optimizer state) to
`<base_dir>/chatrl_checkpoints/<tag>/` every `--save-every` steps (default 60,
skipping step 0) and at the last step, with metadata holding only the model
config. The tag defaults to `d<depth>`; the source SFT tag is not carried
over, so name RL runs explicitly with `--model-tag` when several depths
coexist. `scripts/chat_cli.py -i rl` and `scripts/chat_eval.py -i rl` load
these.

## Launch

```bash
python -m scripts.chat_rl                       # 1 GPU
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
```

`--examples-per-step` (default 16) must be divisible by the world size, and
`--num-samples` should be a multiple of `--device-batch-size` because sampling
is chunked by integer division.

## Relationship to SFT

RL starts from the SFT model and its chat format; it does not revisit the
SmolTalk or MMLU data, so it sharpens math with tool use while leaving general
conversation to whatever SFT produced. See [chat SFT](chat-sft.md).
