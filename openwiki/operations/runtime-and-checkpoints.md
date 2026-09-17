---
type: operations
title: Runtime setup, distributed init and checkpoints
description: Operational plumbing shared by every nanochat script - environment and uv extras, compute_init and torchrun-based DDP detection, device autodetection, the NANOCHAT_BASE_DIR layout, checkpoint saving and loading with tag and step discovery and legacy patching, wandb conventions, the CPU/MPS path, and the GPU peak tables.
tags: [operations, setup, uv, torchrun, ddp, checkpoints, wandb, base-dir, cpu, mps]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-a8aef4f65d816b1c96ddcc9e
    resource: repo://nanochat/checkpoint_manager.py
  - id: openwiki-source-b4c36c62f3071600ed4e6487
    resource: repo://nanochat/common.py
  - id: openwiki-source-b9751542d4064fb72ca59f83
    resource: repo://nanochat/optim.py
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-71e0a7d95ee58ab9cadfb5e6
    resource: repo://scripts/base_train.py
  - id: openwiki-source-b657b01e69409a21afb5d521
    resource: repo://scripts/chat_rl.py
  - id: openwiki-source-ca457235ed5b9f351ff341ee
    resource: repo://scripts/chat_sft.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Runtime setup, distributed init and checkpoints

Every script in `scripts/` starts the same way: parse arguments, call
`compute_init`, load or build a model, and (for training) open a wandb run.
The shared pieces live in `nanochat/common.py` and
`nanochat/checkpoint_manager.py`. This page is the operational reference for
running nanochat on a laptop, a single GPU or an 8-GPU node.

## Environment

- **Python and dependencies.** `pyproject.toml` requires Python 3.10 or newer
  and pins `torch==2.9.1`. Two mutually exclusive extras select the wheel
  index: `uv sync --extra gpu` (CUDA 12.8) or `uv sync --extra cpu` (CPU and
  Apple MPS). The `dev` group adds pytest, matplotlib, ipykernel and
  python-dotenv. Runtime dependencies are deliberately few: `rustbpe` and
  `tiktoken` for the tokenizer, `pyarrow` for parquet, `kernels` for Flash
  Attention 3, `wandb`, `filelock`, `numpy`, `psutil`. There is no
  `transformers` or `datasets` dependency; recent commits removed them.
- **Environment variables.** `NANOCHAT_BASE_DIR` sets the artifact root
  (default `~/.cache/nanochat`), `NANOCHAT_DTYPE` overrides the compute dtype
  (see [precision](../architecture/precision-and-fp8.md)), `WANDB_RUN` is read
  by the run scripts and passed as `--run`, and `OMP_NUM_THREADS=1` is exported
  by the run scripts to stop CPU thread oversubscription under `torchrun`.
- **Secrets.** wandb credentials come from `wandb login`; nothing in the
  repository reads API keys, and `.env` is git-ignored.

## compute_init and distributed detection

`compute_init(device_type)` is called once per process:

1. Validates `device_type` (`cuda`, `mps` or `cpu`) against what PyTorch can
   actually use.
2. Seeds `torch.manual_seed(42)` (and CUDA), noting that most code uses
   explicit RNG objects so the global seed mainly affects weight init.
3. On CUDA, sets `torch.set_float32_matmul_precision("high")` (TF32 matmuls).
4. Detects a distributed launch via `is_ddp_requested()`, which is true when
   `RANK`, `LOCAL_RANK` and `WORLD_SIZE` are all in the environment, i.e. the
   process was started by `torchrun`. In that case (CUDA only) it binds the
   process to `cuda:LOCAL_RANK`, initialises an NCCL process group and
   barriers. Otherwise the device is plain `cuda`, `mps` or `cpu` and the
   world size is 1.
5. Returns `(ddp, rank, local_rank, world_size, device)`.

`autodetect_device_type()` prefers CUDA, then MPS, then CPU; every script
accepts `--device-type` to override it. `compute_cleanup()` destroys the
process group if one exists. `print0` prints only on rank 0, and
`get_dist_info()` lets library code (dataloader, evaluators) learn the rank
layout without being passed it.

nanochat does **not** wrap the model in `DistributedDataParallel`; gradient
synchronisation happens inside the optimizer (see
[MuonAdamW](../training/optimizer.md)). Running the same command without
`torchrun` therefore works unchanged and uses gradient accumulation to reach
the configured total batch size.

## Launch conventions

```bash
# 8 GPUs
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --run=$WANDB_RUN
# single device (GPU, MPS or CPU)
python -m scripts.base_train --depth=6 --window-pattern=L ...
```

Scripts are modules run with `-m` from the repository root. `README.md`
notes that GPUs with less than 80GB need a smaller `--device-batch-size`, and
`runs/runcpu.sh` shows the scaled-down flags for a laptop (depth 6,
`--window-pattern=L`, short sequences, few evaluation tokens). On CPU/MPS the
attention fallback is SDPA and MFU is reported as meaningless.

## wandb

Training scripts create `wandb.init(project=..., name=args.run)` on rank 0
unless `--run` is `dummy`, in which case a `DummyWandb` with no-op `log` and
`finish` is used. Projects are `nanochat` (pretraining), `nanochat-sft` and
`nanochat-rl`. The user config is attached to the run.

## Base directory and checkpoints

`get_base_dir()` creates and returns the artifact root. The directory layout is
tabulated in [pipeline overview](../architecture/pipeline-overview.md).

**Saving.** `save_checkpoint(dir, step, model_data, optimizer_data, meta, rank)`
writes `model_<step>.pt` and `meta_<step>.json` on rank 0 and
`optim_<step>_rank<r>.pt` on every rank when optimizer data is given, because
optimizer state is sharded across ranks. Metadata is JSON and includes the
model config plus whatever the script adds (user config, dataloader state,
loop state).

**Loading.** `load_checkpoint` reads the model, optional optimizer shard for a
given rank, and metadata. `build_model` then reconstructs the `GPT` from
`meta["model_config"]` on the meta device, `to_empty`s it onto the target
device, initialises weights (needed for rotary buffers), and loads the state
dict strictly. It converts bf16 tensors to fp32 on CPU/MPS, strips the
`_orig_mod.` prefix that `torch.compile` adds to keys, applies legacy patches
(default `window_pattern`, missing per-layer lambdas), sets train or eval
mode, loads the tokenizer, and asserts its vocabulary matches the config.

**Discovery.** `load_model(source, device, phase, model_tag=None, step=None)`
maps `base`/`sft`/`rl` to their roots; with no tag it picks the highest
`d<depth>` directory (falling back to the most recently modified), and with no
step the highest `model_<step>.pt`. `load_optimizer_state` fetches just one
rank's optimizer shard and returns `None` with a log line if it is missing,
which is how SFT tolerates base checkpoints saved without optimizer state.

**Resuming pretraining.** `base_train --resume-from-step N` loads model,
per-rank optimizer shard, dataloader state and loop state from
`base_checkpoints/<tag>/` and continues; see [pretraining](../training/pretraining.md).

## Downloads

`download_file_with_lock(url, filename, postprocess_fn)` fetches into the base
dir under a `FileLock`, rechecks existence after acquiring it, and runs an
optional post-processing step; the CORE eval bundle uses it. Dataset shards
and task parquet files have their own lock or temp-file protocols.

## GPU peak tables

`get_peak_flops` and `get_peak_bandwidth` return hard-coded BF16 peak FLOPs and
memory bandwidth for known device-name substrings, checked most specific
first, and infinity (with a warning) for unknown devices so MFU/MBU display as
0% rather than a wrong value. Extend both when adding a GPU.

## Logging

`setup_default_logging` installs a colourised formatter at INFO level; it is
called at import of `nanochat.common` and again by `checkpoint_manager`.
Training scripts print step lines with `print0` rather than the logger.
