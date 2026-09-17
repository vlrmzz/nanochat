---
type: evaluation
title: Inference benchmark (infer_bench)
description: What scripts/infer_bench.py measures and how - prefill versus decode regimes, time to first token and time per output token, model bandwidth utilisation and model FLOPs utilisation rooflines built from the GPT model's cost accounting and the GPU peak tables, the batch-size sweep, and the machine-readable JSON last line.
tags: [benchmark, inference, latency, throughput, mbu, mfu, kv-cache]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-b4c36c62f3071600ed4e6487
    resource: repo://nanochat/common.py
  - id: openwiki-source-e316d09d86c6778699271c53
    resource: repo://scripts/infer_bench.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Inference benchmark (infer_bench)

`scripts/infer_bench.py` treats inference cost as an evaluation in its own
right: CORE and ChatCORE say nothing about what a model costs to serve, and
architecture choices such as grouped-query attention or sliding windows show
up here rather than in accuracy. It runs a trained checkpoint through the
[Engine](../inference/engine.md) and reports latency, throughput, memory and
how close the implementation gets to the GPU's physical ceilings.

## Regimes and rooflines

The module docstring lays out the two regimes the script separates:

- **Prefill** processes the prompt in parallel with large matmuls, so it is
  compute-bound; its distance from the roofline is **MFU** (achieved FLOPs
  over peak FLOPs).
- **Decode** produces one token per step and must re-read all weights plus the
  KV cache to do little arithmetic, so it is memory-bandwidth-bound; its
  roofline metric is **MBU** (achieved bytes moved over peak bandwidth), the
  decode counterpart of training MFU. Batching is close to free until compute
  saturates, so sweeping the batch size traces the latency/throughput curve.

The rooflines come from two sources. Per-GPU peaks are the lookup tables
`get_peak_flops` and `get_peak_bandwidth` in `nanochat/common.py`, which match
device-name substrings (Blackwell, Hopper, Ampere, Ada, AMD CDNA, consumer RTX)
and return infinity for unknown GPUs so utilisation reads 0% instead of a wrong
number. Per-model costs come from the [GPT model](../architecture/gpt-model.md):
`estimate_decode_flops`, `estimate_prefill_flops`, `kv_bytes_per_token` and
`kv_read_bytes`, all window-aware, plus `weight_bytes` computed in the script
from each parameter's actual storage size.

## What it does

1. **Setup.** Requires CUDA and a single process (no `torchrun`), loads the
   checkpoint by source (`-i base|sft|rl`, tag, step), and clamps the prompt so
   prompt plus decode fits in the training `sequence_len`. The prompt is real
   English text repeated and cut to exactly `--prompt-tokens` tokens, because
   random ids would make greedy decoding degenerate.
2. **Static card.** Before measuring anything it prints the model shape, GPU
   peaks and VRAM, parameter count with a per-dtype breakdown, weight bytes,
   KV bytes stored per token and read per step at a representative mid-decode
   context, the theoretical batch-1 decode ceiling
   `peak_bandwidth / (weight_bytes + kv_read)`, and how many full-context KV
   rows fit next to the weights.
3. **Prefill measurement.** One warm-up, then a batch-1 generation of two
   tokens; the time to the first yielded token is taken as prefill time and
   converted to tokens per second and MFU.
4. **Batch sweep.** For each size in `--batch-sizes` (default 1, 8, 32, 128):
   a short warm-up run, then a timed run of `--decode-tokens` steps.
   `bench_generate` times the first `next()` on the generator as **TTFT**
   (which includes the batch-1 prefill, the KV replication to the batch and
   the first sample) and every later `next()` as one decode step, synchronising
   the device after each. **TPOT** is the median step time; throughput is
   `batch * steps / total step time`; MBU divides `(weight_bytes + batch * kv_read) / TPOT`
   by peak bandwidth; MFU divides `batch * decode_flops / TPOT` by peak FLOPs;
   peak VRAM is read from the allocator. Early termination (all rows hit a
   stop token) is flagged in the table.

## Output contract

Everything is printed human-readably first, and then **the last line of
stdout is a single JSON document** with the static card, the prefill result
and the sweep rows (`batch_size`, `ttft_sec`, `tpot_sec`, `tok_per_sec`,
`mbu_percent`, `mfu_percent`, `peak_vram_bytes`, `decode_steps`). Unknown GPU
peaks are emitted as `null` rather than infinity so the line stays valid JSON.
The docstring shows the intended consumption:

```bash
python -m scripts.infer_bench -i base -g d12 | tail -1 | jq .sweep
```

This lets scripts compare architecture variants on both intelligence and cost
axes without parsing the pretty table.

## Practical notes

- The KV cache and rooflines assume `COMPUTE_DTYPE` for cache storage, which
  is what the Engine allocates; see [precision](../architecture/precision-and-fp8.md).
- Because the Engine prefills at batch 1 and replicates, TTFT grows with batch
  size only through the replication copy and the first sampled step.
- A GPU missing from the peak tables still produces latency and throughput
  numbers; only MBU and MFU degrade to zero. Adding a device means extending
  both tables in `nanochat/common.py`, most specific name pattern first.
