---
type: guide
title: Quickstart and task routing
description: Orientation for a coding agent working on nanochat - what the project is and optimises for, the end-to-end pipeline and its artifacts, how to run things at laptop and 8-GPU scale, the repository's non-negotiable conventions, and which wiki page to read for each kind of task.
tags: [quickstart, overview, routing, onboarding, conventions]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-9c8e92f5ee4bbde60422b4b4
    resource: repo://dev/LEADERBOARD.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-a86a54a8af3b6833b22de039
    resource: repo://runs/runcpu.sh
  - id: openwiki-source-d4b89b7c31c1565c1461a5a4
    resource: repo://runs/speedrun.sh
  - id: openwiki-source-71e0a7d95ee58ab9cadfb5e6
    resource: repo://scripts/base_train.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Quickstart and task routing

nanochat is a minimal, single-node harness for training a ChatGPT-style
language model end to end: tokenizer training, pretraining, supervised
fine-tuning, optional reinforcement learning, evaluation and KV-cached
inference, in roughly 8,000 lines of Python on PyTorch. `README.md` is the
authoritative human-written overview; this page tells you where to go next.

## What the project optimises for

- **One dial.** `--depth` determines every other hyperparameter so that any
  size comes out roughly compute-optimal. Changes must be principled enough
  to hold across the whole depth "miniseries", not just one model.
- **Time to GPT-2.** The headline metric is wall-clock time on an 8xH100 node
  to exceed GPT-2's DCLM CORE score of 0.256525; `runs/speedrun.sh` is the
  reference recipe and `dev/LEADERBOARD.md` explains how runs are reported.
- **Minimal, forkable code.** No configuration objects or model factories.
  Recent commits removed the web UI, the report generator and the Hugging
  Face `transformers`/`datasets` dependencies. Adding code is a cost that has
  to be justified; `dev/LOG.md` records ideas that were tried and rejected.

## The pipeline in one glance

```
nanochat.dataset -n N  ->  tok_train / tok_eval  ->  base_train  ->  base_eval
                                                        |
                                              chat_sft  ->  chat_eval  ->  chat_cli
                                                        |
                                              chat_rl (optional)     infer_bench
```

Every stage is a module under `scripts/` run with `python -m` or `torchrun`,
and stages communicate only through files under `NANOCHAT_BASE_DIR`
(default `~/.cache/nanochat`): data shards, `tokenizer/`, `eval_bundle/`,
`task_data/`, and `base_checkpoints/`, `chatsft_checkpoints/`,
`chatrl_checkpoints/` keyed by model tag (`d<depth>` by default). The full
table is in [End-to-end pipeline](architecture/pipeline-overview.md).

## Running it

```bash
uv sync --extra gpu        # or --extra cpu for CPU / Apple MPS
source .venv/bin/activate
bash runs/speedrun.sh      # 8xH100, ~1.5 h: tokenizer, d24 pretrain, eval, SFT, chat eval
bash runs/runcpu.sh        # laptop demo of every code path with a tiny model
python -m pytest tests -v  # unit tests (some need CUDA / FA3)
```

For research iteration the README recommends a d12 pretraining run (about
five minutes on 8 GPUs) with CORE only at the end, watching `val_bpb`,
`core_metric`, MFU and tokens per second in wandb. Any script runs without
`torchrun` on one device; gradient accumulation keeps the recipe identical.

## Conventions you must respect

- **Precision is explicit.** No autocast. `COMPUTE_DTYPE` (auto-detected or
  `NANOCHAT_DTYPE`) governs everything; matrices stay fp32 and are cast in the
  model's `Linear`. See [precision and FP8](architecture/precision-and-fp8.md).
- **Attention goes through one module.** Import `flash_attn` from
  `nanochat/flash_attention.py`; never call FA3 directly. Features must work
  in the SDPA fallback too. See [attention backends](architecture/attention-backends.md).
- **No DDP wrapper.** Gradient sync and optimizer-state sharding live in
  `MuonAdamW`; optimizer checkpoints are per rank. See [optimizer](training/optimizer.md).
- **Meta-device construction.** `GPT.__init__` may only compute shapes; real
  values come from `init_weights()`. New parameters need legacy patches in
  `checkpoint_manager` to load old checkpoints. See [GPT model](architecture/gpt-model.md).
- **The tokenizer is shared state.** Retraining it invalidates every
  checkpoint (vocabulary assertion on load). See [tokenizer](data/tokenizer.md).
- **Chat format is defined by the tokenizer's `render_conversation`** and
  mirrored by the Engine's tool-use state machine; change them together.

## Where to look for each task

| If you want to... | Read |
|---|---|
| Understand or change the Transformer architecture | [GPT model architecture](architecture/gpt-model.md) |
| Touch attention, sliding windows, GQA or the KV cache kernels | [Flash Attention 3 and SDPA fallback](architecture/attention-backends.md) |
| Work on dtypes, bf16/fp16 behaviour or FP8 | [Precision, COMPUTE_DTYPE and FP8](architecture/precision-and-fp8.md) |
| Change how a stage reads or writes artifacts, or chain stages | [End-to-end pipeline](architecture/pipeline-overview.md) |
| Modify pretraining hyperparameters, scaling rules or schedules | [Pretraining (base_train)](training/pretraining.md) |
| Change the optimizer or debug multi-GPU training | [MuonAdamW optimizer](training/optimizer.md) |
| Adjust the SFT mixture, packing or loss masking | [Supervised fine-tuning](training/chat-sft.md) |
| Work on the RL loop or rewards | [Reinforcement learning on GSM8K](training/chat-rl.md) |
| Change pretraining data, sharding, packing or resume | [Pretraining data and dataloader](data/pretraining-dataset.md) |
| Change the vocabulary, special tokens or chat rendering | [Tokenizer and conversation rendering](data/tokenizer.md) |
| Add a dataset or benchmark task | [Tasks package](data/tasks.md) |
| Understand bpb or CORE, or evaluate a base model | [Base model evaluation](evaluation/base-model-evaluation.md) |
| Evaluate a chat model or change ChatCORE | [Chat evaluation and ChatCORE](evaluation/chat-evaluation.md) |
| Measure inference latency, throughput, MBU/MFU | [Inference benchmark](evaluation/inference-benchmark.md) |
| Change generation, sampling, tool use or the chat CLI | [Inference Engine, KV cache and tool use](inference/engine.md) |
| Run or harden LLM-generated code | [Sandboxed Python execution](inference/code-execution-sandbox.md) |
| Set up an environment, launch with torchrun, or load checkpoints | [Runtime setup, distributed init and checkpoints](operations/runtime-and-checkpoints.md) |
| Add or run tests | [Test suite](testing/test-suite.md) |

## Contribution norms

The README asks contributors to disclose substantial LLM-generated code they
do not fully understand, and to weigh code bloat against demonstrated gains.
Leaderboard entries need a CORE score above GPT-2's, the reported
`total_training_time`, the validation bpb, and the commit hash.
