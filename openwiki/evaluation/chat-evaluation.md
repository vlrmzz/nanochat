---
type: evaluation
title: Chat evaluation and ChatCORE
description: How scripts/chat_eval.py grades SFT and RL checkpoints - the generative loop that samples through the Engine and calls each task's evaluate, the categorical loop that scores logits restricted to the allowed answer letters, distributed aggregation, the ChatCORE centred mean, and how SFT reuses the same code during training.
tags: [evaluation, chatcore, chat-eval, generative, categorical, benchmark]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-c138cdd9025e78e6dd8cd26f
    resource: repo://scripts/chat_eval.py
  - id: openwiki-source-ca457235ed5b9f351ff341ee
    resource: repo://scripts/chat_sft.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Chat evaluation and ChatCORE

`scripts/chat_eval.py` measures a chat model (source `sft` or `rl`) on the
five tasks ARC-Easy, ARC-Challenge, MMLU, GSM8K and HumanEval and combines
them into **ChatCORE**, the chat analogue of the base model's CORE score. The
generic loops live in this script; everything task-specific (prompt format,
grading) lives in the [tasks package](../data/tasks.md).

## Two evaluation loops

`run_chat_eval(task_name, model, tokenizer, engine, ...)` looks the task up in
a fixed table (`HumanEval`, `MMLU` test, `ARC-Easy` test, `ARC-Challenge`
test, `GSM8K` main test), instantiates it, and dispatches on
`task.eval_type`:

### Generative (GSM8K, HumanEval)

`run_generative_eval` handles one problem at a time. It renders the prompt with
`render_for_completion` (the conversation minus its final assistant turn,
primed with `<|assistant_start|>`), draws `num_samples` completions from the
[Engine](../inference/engine.md) with the given `max_new_tokens`,
`temperature` (default 0, greedy) and `top_k`, decodes only the tokens after
the prompt, calls `task.evaluate` on each, and counts the problem as passed if
**any** sample passes. With the default `num_samples=1` this is plain
accuracy; with more samples it is pass@k. Problems are strided across ranks
and the pass/total counts are summed with `all_reduce` at the end.

Because the Engine's tool-use state machine runs during sampling, GSM8K
completions can invoke the calculator exactly as they would in the chat CLI,
and HumanEval completions are executed in the [sandbox](../inference/code-execution-sandbox.md)
by the task's `evaluate`.

### Categorical (ARC, MMLU)

`run_categorical_eval` needs no sampling, so it processes `batch_size`
problems per forward pass. Prompts are rendered with `render_for_completion`,
right-padded with BOS to equal length, and forwarded once. For each problem it
takes the logits at the last real prompt position (where the answer letter
would be produced), restricts them to the token ids of that problem's allowed
letters (asserting each letter is a single token, and caching the lookups),
takes the argmax, and passes the predicted letter to `task.evaluate`. The code
comments are explicit that restricting to the allowed letters makes the task
easier than free generation and that this mirrors how such evaluations are
typically run. Batches are strided across ranks and counts all-reduced.

## ChatCORE

When all five tasks are evaluated, the script reports
`ChatCORE = mean over tasks of (acc - baseline) / (1 - baseline)`, with
baselines of 0.25 for the three four-way multiple-choice tasks and 0.0 for
GSM8K and HumanEval. This makes 0 the random-guess level and 1 perfect, the
same convention as CORE (see [base model evaluation](base-model-evaluation.md)).

## Command line

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

`-i/--source` (`sft` or `rl`) is required. `-a/--task-name` selects tasks
(`|`-separated; default all five), `-x/--max-problems` caps each task,
`-b/--batch-size` sets the categorical batch, and `-n`, `-t`, `-k`, `-m`
control sampling. `-g/--model-tag` and `-s/--step` pick a specific checkpoint.
`runs/speedrun.sh` runs it on 8 GPUs after SFT.

## Reuse inside SFT

`scripts/chat_sft.py` imports `run_chat_eval` and, every `--chatcore-every`
steps and on the last step, evaluates all five tasks with the uncompiled model
(prompt shapes vary, so the compiled graph is avoided). Categorical tasks are
capped by `--chatcore-max-cat` (default unlimited) and generative tasks by
`--chatcore-max-sample` (default 24 problems) to keep the in-training estimate
cheap. It logs `chatcore_metric`, a categorical-only `chatcore_cat`, and per
task accuracies to wandb. See [chat SFT](../training/chat-sft.md).

## Notes for contributors

- Adding a task to the table requires it to declare `eval_type` and implement
  `evaluate`; categorical tasks must also carry `letters` in each conversation.
- The categorical loop assumes each answer letter tokenises to exactly one
  token in the trained vocabulary, which holds for single capital letters
  without a leading space; this is why `render_mc` avoids a space before the
  letter.
- Reported percentages are per-rank until the final all-reduce; only the
  `Final:` line is global.
