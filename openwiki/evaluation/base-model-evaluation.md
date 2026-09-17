---
type: "Reference"
title: "Base model evaluation: bits per byte and CORE"
openwiki_generated: true
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-c004e4158ed02ea4034feaa3
    resource: repo://nanochat/core_eval.py
  - id: openwiki-source-f48c99143175820f617754c2
    resource: repo://nanochat/loss_eval.py
  - id: openwiki-source-7dc0e414d68d7af33da53015
    resource: repo://scripts/base_eval.py
  - id: openwiki-source-71e0a7d95ee58ab9cadfb5e6
    resource: repo://scripts/base_train.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---


# Base model evaluation: bits per byte and CORE

Two numbers describe a pretrained nanochat model: **validation bits per byte
(bpb)**, a loss that does not depend on the tokenizer, and the **CORE score**
from the DCLM paper, an average of centred accuracies over 22 in-context
learning tasks. GPT-2's CORE score of 0.256525 is the target of the
Time-to-GPT-2 leaderboard described in `README.md` and `dev/LEADERBOARD.md`.
Both metrics are computed during pretraining and again, more thoroughly, by
`scripts/base_eval.py`.

## Bits per byte

`evaluate_bpb(model, batches, steps, token_bytes)` in `nanochat/loss_eval.py`
replaces mean token loss. For each batch it runs the model with
`loss_reduction='none'`, sums the per-token losses in nats, and independently
sums the byte lengths of the target tokens using the `token_bytes` table from
the tokenizer (see [tokenizer](../data/tokenizer.md)). The result is
`total_nats / (ln 2 * total_bytes)`. Because special tokens have zero bytes
they are excluded from both sums, and a slower code path handles targets equal
to `-1` (the ignore index used by SFT) without indexing the table with a
negative id; the fast path is taken when no target is negative. Under
`torchrun` both sums are all-reduced before the division, so every rank returns
the same value. If no bytes were counted the function returns infinity.

Callers decide the amount of data through `steps`: `base_train` uses
`--eval-tokens` (default about 42M tokens) divided by tokens per step, and
`base_eval` uses `--split-tokens` per split, rounding down to a multiple of
the per-step token count.

## CORE

### Data and configuration

`evaluate_core` in `scripts/base_eval.py` downloads `eval_bundle.zip` from a
public S3 bucket on first use via `download_file_with_lock` (so concurrent
ranks download once) and unpacks it to `<base_dir>/eval_bundle/`. The bundle
holds `core.yaml` (the task list with type, data file, few-shot count and
continuation delimiter), `eval_meta_data.csv` (random-baseline accuracy per
task) and one JSONL data file per task. Each task's examples are shuffled with
a fixed seed (1337) before optional truncation to `max_per_task`, so partial
evaluations are comparable across runs.

### Scoring one example

`nanochat/core_eval.py` implements the three DCLM task types:

- **multiple_choice**: prompts share a context and differ in the continuation.
  `render_prompts_mc` renders one prompt per choice (with few-shot examples
  prepended), `batch_sequences_mc` tokenises them and finds the common token
  prefix so the continuation span is known, and the choice with the lowest
  mean loss over its span is the prediction.
- **schema**: contexts differ and the continuation is shared, so the common
  *suffix* is located instead; scoring is the same lowest-mean-loss rule.
- **language_modeling**: two prompts, without and with the continuation; the
  example is correct only if the argmax prediction at every continuation
  position equals the actual token.

Few-shot examples are sampled per item with `random.Random(1234 + idx)`,
excluding the item itself. Sequences are right-padded with BOS to a batch,
forwarded once, and per-position losses and argmax predictions come out of
`forward_model`. A `max_seq_len` attribute on the model, if present, triggers
left-cropping with index adjustment; the nanochat GPT has no such attribute,
so its rotary cache limit applies instead.

### Aggregation and centering

`evaluate_task` strides examples across ranks (`range(rank, n, world_size)`),
fills a per-example correctness tensor, all-reduces it, and returns the mean.
`evaluate_core` then centres each task as
`(accuracy - baseline/100) / (1 - baseline/100)` and averages the centred
values into `core_metric`, so 0 is chance and 1 is perfect. A docstring TODO
notes SQuAD scores below the reference implementation.

## The base_eval script

`python -m scripts.base_eval [--eval core,bpb,sample]` loads a base checkpoint
(largest tag and last step by default, or `--model-tag`/`--step`) and runs any
subset of three modes:

- **sample**: on rank 0 only, greedy completions of a fixed prompt list plus
  eight temperature-1 unconditioned samples, through the
  [Engine](../inference/engine.md).
- **bpb**: train and validation bits per byte with the same BOS-aligned
  dataloader used for training.
- **core**: the full CORE evaluation, written on rank 0 to
  `<base_dir>/base_eval/base_model_<step>.csv` with per-task accuracy and
  centred score plus the final `CORE` line.

It runs under `torchrun` for speed (`runs/speedrun.sh` uses 8 GPUs) or on a
single device with `--max-per-task` and `--split-tokens` reduced.

## Use inside pretraining

`scripts/base_train.py` evaluates validation bpb every `--eval-every` steps and
on the last step, and runs CORE every `--core-metric-every` steps with
`--core-metric-max-per-task` examples (default 500). Both are wrapped in
`disable_fp8` so evaluation runs in bf16, and CORE uses the uncompiled model
because prompt shapes vary. The leaderboard convention documented in
`dev/LEADERBOARD.md` is `--core-metric-every=999999` with
`--core-metric-max-per-task=-1`, so the full CORE runs exactly once at the end.
See [pretraining](../training/pretraining.md).

## Practical notes

- CORE prompts are short and evaluated one example at a time, so the metric is
  slow relative to bpb; that is why training samples it at intervals.
- bpb is comparable across tokenizers only if `token_bytes.pt` was produced
  by the same `tok_train.py` logic; the raw-bytes fix mentioned in git history
  makes new runs read slightly higher than older ones.
- The chat-model analogue of CORE is ChatCORE; see
  [chat evaluation](chat-evaluation.md).
