# OpenWiki brief for nanochat

nanochat is a minimal, hackable, single-node LLM training harness in Python
and PyTorch: tokenizer training, pretraining, supervised fine-tuning,
reinforcement learning, evaluation and KV-cached inference, all driven by
one complexity dial (`--depth`). Document it as a system for coding agents
that will change it, not as a directory inventory.

## Authoritative human-written docs (read these first, never contradict them)

- README.md — project goals, the Time-to-GPT-2 leaderboard, setup, the
  precision/dtype contract and the file structure. Treat as ground truth.
- dev/LEADERBOARD.md — how leaderboard runs are configured and reported.
- dev/LOG.md — experiment log; negative results and why ideas were rejected.
  Cite, do not restate.
- runs/speedrun.sh — the reference end-to-end pipeline; always reflects the
  current state of the art on the leaderboard.

## Priorities

1. The pretraining path: scripts/base_train.py, the depth-driven
   hyperparameter derivation (width, heads, batch size, learning rates,
   weight decay, training horizon), the GPT architecture in nanochat/gpt.py,
   the MuonAdamW optimizer with its distributed ZeRO-2 style sharding, FP8
   training, and the Flash Attention 3 / SDPA fallback.
2. Data: ClimbMix parquet shards, on-demand download, the BOS-aligned
   best-fit dataloader and its resumable state, the rustbpe/tiktoken
   tokenizer and its special tokens.
3. Evaluation: bits per byte, the DCLM CORE metric, ChatCORE over the tasks
   package, and the inference benchmark.
4. Post-training: chat SFT with the task mixture and loss masking, GSM8K RL,
   the tool-use state machine in the inference Engine, and the sandboxed
   Python executor.
5. Operations: uv extras (cpu vs gpu), NANOCHAT_BASE_DIR layout and
   checkpoint directories, torchrun launches, the run scripts, wandb, and the
   CPU/MPS path.

## Conventions

- Never reproduce secret values or the contents of any .env file.
- Ignore build output, caches, notebooks, images and lock files.
- The repository is a fork of karpathy/nanochat; describe the code as it is
  in this checkout, not as upstream documentation describes it.
