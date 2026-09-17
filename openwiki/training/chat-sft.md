---
type: workflow
title: Supervised fine-tuning (chat_sft)
description: How scripts/chat_sft.py turns a base checkpoint into a chat model - hyperparameter inheritance from the pretraining metadata, optimizer warm start with LR reset, the SmolTalk/MMLU/GSM8K mixture, BOS-aligned best-fit packing that pads instead of cropping, loss masking to assistant tokens, the progress-based LR schedule and distributed stop synchronisation, in-training ChatCORE, and the checkpoint it writes.
tags: [sft, fine-tuning, chat, packing, loss-mask, mixture, post-training]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-ca457235ed5b9f351ff341ee
    resource: repo://scripts/chat_sft.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Supervised fine-tuning (chat_sft)

`scripts/chat_sft.py` teaches the pretrained model the conversation format:
the user/assistant special tokens, multiple-choice answering, and calculator
tool use. It is the stage that produces `chatsft_checkpoints`, which the chat
CLI, chat evaluation and RL consume. Structurally it mirrors
[pretraining](pretraining.md) but replaces the document dataloader with a
conversation packer and the CORE metric with ChatCORE.

## Loading and hyperparameter inheritance

The script loads the base checkpoint in train mode (largest tag, last step, or
`--model-tag`/`--model-step`). Six hyperparameters default to the values the
pretraining run used, read from the checkpoint metadata: `max_seq_len`,
`device_batch_size` and `total_batch_size` from the top-level meta, and
`embedding_lr`, `unembedding_lr` and `matrix_lr` from the saved
`user_config`. Passing a flag explicitly overrides and logs a note; when the
metadata lacks a value a hard-coded fallback applies. This keeps SFT
consistent with whatever depth was trained without repeating flags.

The optimizer is built with `weight_decay=0.0` because pretraining already
decays weight decay to zero by its end. With `--load-optimizer=1` (default) the
per-rank optimizer shard from the base checkpoint is loaded to warm-start the
momentum buffers. Since `load_state_dict` also overwrites the param-group
learning rates with the pretraining values, which have decayed to nearly zero,
the script saves the fresh LRs before loading and restores them afterwards. If
the shard is missing it warns and continues with a fresh optimizer. All groups
are then scaled by `--init-lr-frac` (default 0.8) and recorded as `initial_lr`.
The model is compiled with `torch.compile(dynamic=False)`; the uncompiled
`orig_model` is kept for evaluation and saving.

## Data mixture

Training data is a `TaskMixture` of `SmolTalk(train)`, `--mmlu-epochs`
(default 3) copies of `MMLU(auxiliary_train)` and `--gsm8k-epochs` (default 4)
copies of `GSM8K(train)`; repeating a task oversamples it, as explained in
[tasks](../data/tasks.md). Validation is `SmolTalk(test)` plus `MMLU(test)`
truncated to 5200 rows and `GSM8K(test)` truncated to 420 rows so the ratios
match training.

## Packing and masking

`sft_data_generator_bos_bestfit(split)` renders each conversation with the
tokenizer's `render_conversation` into `(ids, mask)` (see
[tokenizer](../data/tokenizer.md)) and packs rows of `max_seq_len + 1`
tokens with the same best-fit strategy as the pretraining loader, with one
difference: when no buffered conversation fits, the row is **padded with BOS
tokens** rather than cropping a conversation, so no SFT tokens are ever
discarded. Each rank walks the dataset with a stride of the world size.

Targets are the row shifted by one. Two masks are applied by setting targets
to `-1` (the cross-entropy ignore index): positions where the render mask is 0
(BOS, user turns, `<|assistant_start|>`, tool outputs) and padding positions.
Only assistant text, tool calls and `<|assistant_end|>` contribute to the
loss.

The generator also owns the stopping logic through module globals. It sets
`last_step` when `--num-iterations` (if positive) is reached, or when the
number of conversations *consumed* reaches the dataset size (one epoch);
consumption is tracked separately from buffering so prefetching does not end
training early. It exposes `approx_progress` in `[0, 1]` for the schedule.

## Schedule and step

Because the iteration count is not known in advance in the epoch-driven case,
the LR multiplier is a function of progress rather than step: linear warm-up
over `--warmup-ratio` (default 0), constant, then linear warm-down over the
last `--warmdown-ratio` (default 0.5) to `--final-lr-frac` (default 0).
Progress is taken as the running maximum of the generator's estimate so it
never decreases. Muon momentum warms from 0.85 to 0.95 over the first 300
steps. Each step accumulates `total_batch_size / (device_batch_size *
max_seq_len * world_size)` micro-batches, exactly as in pretraining, with the
optional fp16 GradScaler.

At the top of every step, in the distributed case, `last_step` is all-reduced
with MAX so that if any rank has exhausted its data all ranks stop together
and no rank hangs waiting on a collective.

## Evaluation during training

- Validation bits per byte every `--eval-every` steps (default 200) over
  `--eval-tokens`, using the same packer on the validation mixture.
- ChatCORE every `--chatcore-every` steps (default 200) via `run_chat_eval`
  on the uncompiled model, with categorical tasks capped by
  `--chatcore-max-cat` and generative tasks by `--chatcore-max-sample`
  (default 24). See [chat evaluation](../evaluation/chat-evaluation.md).

Both also run on the last step. Metrics go to the `nanochat-sft` wandb project
unless `--run` is `dummy`.

## Output

On the last step every rank participates in `save_checkpoint` to
`<base_dir>/chatsft_checkpoints/<tag>/` (tag `d<depth>` unless
`--model-tag`), writing the uncompiled model state, each rank's optimizer
shard, and metadata with `step`, `val_bpb`, the reconstructed `model_config`
(including `window_pattern`) and the SFT `user_config`. Downstream stages load
it with `load_model("sft", ...)`.

## Launch

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
python -m scripts.chat_sft --num-iterations=1500 --eval-every=200   # laptop-scale, see runs/runcpu.sh
```

Garbage collection is frozen and disabled after the first step, as in
pretraining, with a manual collection every 5000 steps.
