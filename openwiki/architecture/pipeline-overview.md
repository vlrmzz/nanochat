---
type: architecture
title: End-to-end pipeline
description: The nanochat stages (data download, tokenizer, pretraining, base evaluation, SFT, chat evaluation, RL, chat) as one system - which script owns each stage, what each reads and writes under NANOCHAT_BASE_DIR, and how the run scripts chain them.
tags: [pipeline, stages, artifacts, run-scripts, speedrun, checkpoints, base-dir]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-a8aef4f65d816b1c96ddcc9e
    resource: repo://nanochat/checkpoint_manager.py
  - id: openwiki-source-b4c36c62f3071600ed4e6487
    resource: repo://nanochat/common.py
  - id: openwiki-source-28ef039256ff8e08c2b60c1d
    resource: repo://nanochat/dataset.py
  - id: openwiki-source-7f5ef7612eaa1b8c8aeefce8
    resource: repo://nanochat/tokenizer.py
  - id: openwiki-source-758ecdaafa4f5ef736843fd3
    resource: repo://runs/miniseries.sh
  - id: openwiki-source-a86a54a8af3b6833b22de039
    resource: repo://runs/runcpu.sh
  - id: openwiki-source-1cc9bd905c6653d5f0d9711d
    resource: repo://runs/scaling_laws.sh
  - id: openwiki-source-d4b89b7c31c1565c1461a5a4
    resource: repo://runs/speedrun.sh
  - id: openwiki-source-7dc0e414d68d7af33da53015
    resource: repo://scripts/base_eval.py
  - id: openwiki-source-71e0a7d95ee58ab9cadfb5e6
    resource: repo://scripts/base_train.py
  - id: openwiki-source-c138cdd9025e78e6dd8cd26f
    resource: repo://scripts/chat_eval.py
  - id: openwiki-source-b657b01e69409a21afb5d521
    resource: repo://scripts/chat_rl.py
  - id: openwiki-source-ca457235ed5b9f351ff341ee
    resource: repo://scripts/chat_sft.py
  - id: openwiki-source-cadf1570739aed4547654b15
    resource: repo://tasks/common.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# End-to-end pipeline

nanochat is a sequence of standalone Python entry points that communicate only
through files under one base directory. No stage imports another stage's
in-memory state; each one loads what the previous stage wrote. This page maps
the stages, their owners, and their on-disk contract. `runs/speedrun.sh` is the
canonical chaining of all of them and, per `README.md`, always reflects the
reference way to reach GPT-2 capability.

## The base directory

Every artifact lives under `get_base_dir()` in `nanochat/common.py`: the
`NANOCHAT_BASE_DIR` environment variable if set, otherwise
`~/.cache/nanochat`. All run scripts export it explicitly to that default. The
layout that the stages agree on:

| Path under base dir | Written by | Read by |
|---|---|---|
| `base_data_climbmix/shard_NNNNN.parquet` | `python -m nanochat.dataset` | tokenizer training, pretraining and BPB dataloaders |
| `tokenizer/tokenizer.pkl`, `tokenizer/token_bytes.pt` | `scripts/tok_train.py` | every stage via `get_tokenizer()` / `get_token_bytes()` |
| `eval_bundle/` | first CORE evaluation (auto-downloaded zip) | `evaluate_core` in `scripts/base_eval.py` |
| `task_data/<repo>/<subset>/<split>/*.parquet` | first use of a task (Hugging Face parquet export) | `tasks/*` via `load_hub_dataset` |
| `base_checkpoints/<tag>/` | `scripts/base_train.py` | `base_eval`, `chat_sft`, `infer_bench` |
| `base_eval/<slug>.csv` | `scripts/base_eval.py` | humans |
| `chatsft_checkpoints/<tag>/` | `scripts/chat_sft.py` | `chat_eval`, `chat_rl`, `chat_cli`, `infer_bench` |
| `chatrl_checkpoints/<tag>/` | `scripts/chat_rl.py` | `chat_eval`, `chat_cli` |

Each checkpoint directory holds `model_<step>.pt`, `meta_<step>.json` and, when
the optimizer is saved, one `optim_<step>_rank<r>.pt` per rank because
optimizer state is sharded. `checkpoint_manager.load_model(source, ...)` maps
the source names `base`, `sft` and `rl` onto the three checkpoint roots and,
absent an explicit tag or step, picks the largest `d<depth>` tag and the highest
step. Details are in [runtime and checkpoints](../operations/runtime-and-checkpoints.md).

## Stages

1. **Data download** (`python -m nanochat.dataset -n N`): fetches the first N
   ClimbMix training shards plus the pinned validation shard. The speedrun
   grabs 8 shards for tokenizer training, then 170 more in the background while
   the tokenizer trains, and waits for that download before pretraining. See
   [pretraining data](../data/pretraining-dataset.md).
2. **Tokenizer** (`scripts/tok_train.py`, then `scripts/tok_eval.py`): trains a
   32768-token BPE vocabulary on about 2B characters and writes the encoding
   plus the per-token byte table used for bits-per-byte. See
   [tokenizer](../data/tokenizer.md).
3. **Pretraining** (`scripts/base_train.py`): builds the model from `--depth`,
   derives every other hyperparameter, trains, evaluates periodically and writes
   `base_checkpoints`. The speedrun trains d24 with `--target-param-data-ratio=8`
   and `--fp8` on 8 GPUs. See [pretraining](../training/pretraining.md).
4. **Base evaluation** (`scripts/base_eval.py`): CORE score, train/val bits per
   byte and samples for a base checkpoint. See
   [base model evaluation](../evaluation/base-model-evaluation.md).
5. **SFT** (`scripts/chat_sft.py`): loads the base checkpoint (and, by default,
   its optimizer shards), fine-tunes on the SmolTalk/MMLU/GSM8K mixture, and
   writes `chatsft_checkpoints`. See [chat SFT](../training/chat-sft.md).
6. **Chat evaluation** (`scripts/chat_eval.py -i sft|rl`): ARC, MMLU, GSM8K and
   HumanEval accuracies and the ChatCORE mean. See
   [chat evaluation](../evaluation/chat-evaluation.md).
7. **RL** (`scripts/chat_rl.py`, optional): loads the SFT checkpoint and runs
   the simplified GRPO loop on GSM8K into `chatrl_checkpoints`. The speedrun
   does not run it. See [chat RL](../training/chat-rl.md).
8. **Chat** (`scripts/chat_cli.py`): interactive or single-prompt chat against
   the `sft` or `rl` checkpoint through the [Engine](../inference/engine.md).
   `scripts/infer_bench.py` measures the same checkpoints' latency and
   throughput.

The multi-GPU stages are launched with `torchrun --standalone --nproc_per_node=8`;
the same scripts run on one GPU, CPU or MPS without `torchrun`, falling back to
gradient accumulation to hit the same total batch size.

## Run scripts

- **`runs/speedrun.sh`**: the reference pipeline for an 8xH100 node. Installs
  `uv`, creates the venv with the `gpu` extra, downloads data, trains the
  tokenizer, pretrains d24, evaluates, runs SFT and chat eval. `WANDB_RUN`
  defaults to `dummy`, which disables wandb logging.
- **`runs/runcpu.sh`**: the same shape shrunk for a laptop: 8 data shards, a
  depth-6 model with `--window-pattern=L`, short sequences, and reduced
  evaluation, followed by a short SFT. It is a demo of the code paths, not a
  way to get a good model.
- **`runs/miniseries.sh`**: sweeps depths 12 through 26 at the default
  compute-optimal ratio, one `base_train` per depth with CORE only at the end,
  and scrapes the training logs into a results CSV under the base dir.
- **`runs/scaling_laws.sh`**: sweeps FLOP budgets times depths using
  `--target-flops` with `--target-param-data-ratio=-1`, skipping runs already
  present in the CSV so the sweep is resumable, and records detailed parameter
  counts for scaling-law fits.

Both sweep scripts reduce `--device-batch-size` at larger depths to avoid
out-of-memory failures; `base_train` compensates with gradient accumulation.

## Invariants worth knowing

- The tokenizer is shared by everything downstream. `build_model` asserts that
  the loaded tokenizer's vocabulary size equals the checkpoint's `vocab_size`,
  so retraining the tokenizer invalidates existing checkpoints.
- Stage outputs are keyed by model tag (`d<depth>` unless `--model-tag` is
  given); SFT and RL default their output tag to the depth of the loaded model,
  so a base d24 becomes an SFT d24 in a different root.
- Downloads are guarded by file locks so that multiple `torchrun` ranks do not
  fetch the same file concurrently.
