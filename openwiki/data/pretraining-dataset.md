---
type: data
title: Pretraining data and dataloader
description: How nanochat stores and streams pretraining text - ClimbMix parquet shards, on-demand download with retries, the train/val split convention, the BOS-aligned best-fit packing dataloader with DDP row-group sharding and resumable state, and its coupling to the tokenizer.
tags: [data, dataset, dataloader, parquet, climbmix, packing, ddp, resume]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-a7dade439134fd10c38ddabe
    resource: repo://nanochat/dataloader.py
  - id: openwiki-source-28ef039256ff8e08c2b60c1d
    resource: repo://nanochat/dataset.py
  - id: openwiki-source-71e0a7d95ee58ab9cadfb5e6
    resource: repo://scripts/base_train.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Pretraining data and dataloader

Two modules own pretraining text. `nanochat/dataset.py` knows where the
parquet shards come from and how to list and download them.
`nanochat/dataloader.py` turns those shards into `(inputs, targets)` token
batches for `scripts/base_train.py` and the bits-per-byte evaluators. Neither
module knows anything about the model.

## The shard set

The dataset is `karpathy/climbmix-400b-shuffle` on the Hugging Face Hub, a
pre-shuffled repackaging of NVIDIA ClimbMix into 6543 parquet files named
`shard_00000.parquet` through `shard_06542.parquet`, each roughly 100MB after
zstd compression with row groups of 1024 documents. `dev/repackage_data_reference.py`
documents how the shards were produced (it is reference code only and imports
`datasets`, which is not a project dependency). Shards are stored locally under
`<base_dir>/base_data_climbmix`.

**Split convention.** The last parquet file on disk is the validation set;
everything before it is training. This is implemented identically in
`parquets_iter_batched` (used by tokenizer training) and in the dataloader's
`_document_batches`. The download command therefore always fetches shard 6542
in addition to the first `-n` training shards, so that the validation shard is
pinned and does not shift as more training shards are added.

**Legacy fallback.** If `base_data_climbmix` does not exist,
`list_parquet_files` prints a loud "DATASET UPGRADE REQUIRED" banner (on rank 0
for the train split) and falls back to the old FinewebEdu directory
`base_data`. The banner tells users to re-download and retrain the tokenizer;
mixing an old tokenizer with new data is not detected automatically.

## Downloading

`python -m nanochat.dataset -n N [-w workers]` downloads shards 0..N-1 plus the
validation shard with a multiprocessing pool (4 workers by default).
`download_single_file` skips files already present, streams each shard to a
`.tmp` file and renames it into place only on success, retries up to five times
with exponential backoff, and deletes partial files on failure.
`list_parquet_files` ignores `.tmp` files, so a training run can start while a
background download is still adding shards. `runs/speedrun.sh` relies on that:
it downloads 8 shards, starts a 170-shard download in the background, trains
the tokenizer, then waits for the download before pretraining.

## The dataloader

`tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, B, T, split, ...)`
is an infinite generator. The module docstring summarises the design as
"BOS-aligned bestfit": every row starts with a BOS token, rows are 100% full
with no padding, and about 35% of tokens are discarded to cropping at
T=2048 in exchange for every token being able to attend back to its document's
BOS. The docstring points to the older non-aligned loader in git history for
users with very little data and long documents.

**Document stream and DDP sharding.** `_document_batches` walks the parquet
files in order and, within each file, reads row groups `rank, rank+world,
rank+2*world, ...`, so ranks partition each shard by row group without any
communication. Each row group's `text` column is yielded in lists of
`tokenizer_batch_size` documents together with `(pq_idx, rg_idx, epoch)`. When
the files are exhausted the generator wraps to the first file and increments
`epoch`.

**Tokenisation and packing.** `refill_buffer` encodes a document batch with
the tokenizer in parallel threads, prepending BOS to each document, and keeps a
buffer of about 1000 tokenised documents. For each of the `B` rows of capacity
`T + 1`, it repeatedly picks the largest buffered document that fits entirely;
when nothing fits it crops the shortest buffered document to fill the remaining
space exactly. Rows are assembled in a preallocated tensor, copied into a
pinned CPU buffer laid out as `[inputs | targets]` (targets being the row
shifted by one), and moved to the device in a single transfer into a persistent
device buffer. Callers must consume a batch before requesting the next one,
because the yielded tensors are views into that reused buffer.

**Resumable state.** Every yield returns
`{"pq_idx", "rg_idx", "epoch"}` describing the position of the most recent
document batch. Pretraining stores that dict in each checkpoint's
`meta_<step>.json` as `dataloader_state_dict` and passes it back as
`resume_state_dict` when `--resume-from-step` is used. On resume the loader
advances one row-group stride past the recorded position so data is not
repeated; the resume is approximate, since the in-memory buffer at save time is
not reconstructed. The same triple is what the training log prints as
`epoch: E pq: P rg: R`.

`tokenizing_distributed_data_loader_bos_bestfit` is a thin wrapper that drops
the state dict; `base_train` uses it for the validation loader and
`scripts/base_eval.py` uses it for both splits.

## Coupling to the tokenizer

The loader depends on the tokenizer only through `encode(..., prepend=bos)`
and `get_bos_token_id()`. Bits-per-byte evaluation additionally needs the
`token_bytes.pt` table written by `scripts/tok_train.py`, which maps every
token id to its byte length (0 for special tokens); see
[tokenizer](tokenizer.md) and [base model evaluation](../evaluation/base-model-evaluation.md).
Tokenizer training itself reads raw text through `parquets_iter_batched`,
cropping each document to `--doc-cap` characters and stopping after
`--max-chars`.

## Practical notes

- Changing `B`, `T` or the world size changes which documents land in which
  row, so runs are not bit-reproducible across hardware configurations even
  with the same seed.
- The best-fit search is a linear scan over the buffer for every placement;
  it runs on the CPU while the GPU is busy with the previous step, which is
  why `base_train` prefetches the next batch immediately after `backward`.
- With the default 2048-token context, a useful mental model is that 170
  shards suffice for a GPT-2 grade run, per the comments in `runs/speedrun.sh`.
