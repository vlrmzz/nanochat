---
type: testing
title: Test suite
description: Map of nanochat's pytest suite - what each of the six test files verifies, which tests need CUDA or Flash Attention 3, how the tokenizer and engine tests stay hermetic with in-process training and mock components, the pytest configuration, and how to run subsets.
tags: [testing, pytest, ci, hermetic, cuda, mocks]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-afe930b9c8fd4b271b7d6406
    resource: repo://tests/test_attention_fallback.py
  - id: openwiki-source-0e41257659137c7a2c27de1c
    resource: repo://tests/test_engine.py
  - id: openwiki-source-c43ee0ac0d95949a6d1178d8
    resource: repo://tests/test_execution.py
  - id: openwiki-source-7a749b3195d09a969723143e
    resource: repo://tests/test_optim.py
  - id: openwiki-source-6343060992f3d87399d31b3b
    resource: repo://tests/test_tasks.py
  - id: openwiki-source-bdc49f626c4101be45e0e66a
    resource: repo://tests/test_tokenizer.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Test suite

nanochat ships six pytest files under `tests/`, each guarding one subsystem
with focused, mostly hermetic tests. There is no CI configuration in the
repository; tests are run manually. The suite is small by design, matching
the project's preference for minimal code, so each file is also the best
executable specification of its module's contract.

## Configuration

`pyproject.toml` configures pytest: `testpaths = ["tests"]`, discovery of
`test_*.py` files, `Test*` classes and `test_*` functions, and one registered
marker, `slow`, documented as deselectable with `-m "not slow"`. No test in
the current tree uses the marker; it is reserved. Install with
`uv sync --extra gpu --group dev` (or `--extra cpu`) and run:

```bash
python -m pytest tests -v
```

## The files

| File | Subsystem | Hardware | Hermetic? |
|---|---|---|---|
| `test_attention_fallback.py` | FA3 vs SDPA attention, KV cache | FA3 comparison tests need an FA3-capable CUDA GPU; SDPA tests run anywhere | yes |
| `test_engine.py` | `Engine`, `KVCache`, sampling | CPU | yes (mock model and tokenizer) |
| `test_execution.py` | Python sandbox | any (memory limit relies on rlimits, skipped inside the guard on macOS) | yes |
| `test_optim.py` | `MuonAdamW` | CUDA required, whole file skipped otherwise | yes |
| `test_tasks.py` | `Task`, `TaskMixture`, `HubDataset`, `render_mc` | CPU, no network | yes |
| `test_tokenizer.py` | BPE round-trips, special tokens, chat rendering | CPU | yes (trains a tiny tokenizer in-process) |

### test_attention_fallback.py

Three classes. `TestFA3VsSDPA` is skipped without `HAS_FA3`; it runs identical
bf16 inputs through both backends by flipping the module-level override with
`set_impl` and asserts closeness (looser tolerances for gradients) across
causal, full-context, sliding-window, GQA, larger-shape, prefill, single-token
decode and windowed decode cases. `TestSDPAOnly` uses CUDA/bf16 if available
and CPU/fp32 otherwise, checking forward shapes, gradients and `KVCache`
interaction. `TestOverrideMechanism` checks the override knob itself. See
[attention backends](../architecture/attention-backends.md).

### test_engine.py

Defines `MockModel` (uniform logits, advances the cache) and `ByteTokenizer`
(bytes 0-255 plus six special ids) so the Engine runs on CPU with no
checkpoint. Tests cover cache advance/reset/views, `prefill` copying, per-row
first-token diversity, seed reproducibility, temperature-0 determinism,
`max_tokens` and `num_samples` limits, and seed-driven variation. See
[Engine](../inference/engine.md).

### test_execution.py

One adversarial property per test: happy path, exception capture, timeout
killing an infinite loop promptly, the memory limit, five destructive calls
being disabled, stdin disabled, writes confined to the temp directory,
environment scrubbing, tricky string embedding, and fresh globals. See
[sandbox](../inference/code-execution-sandbox.md).

### test_optim.py

Module-level `pytestmark` skips everything without CUDA because the fused
kernels are `torch.compile`d (compilation dominates runtime). Shapes are kept
small and shared so kernels compile once: a sub-1024-element AdamW parameter,
a large AdamW parameter, and wide and tall Muon stacks. Tests: fused AdamW
matches `torch.optim.AdamW` after 10 steps, two identical runs are bitwise
identical, optimising distance-to-target halves the distance for every
parameter, and the first Muon update of a full-rank gradient is close to
semi-orthogonal (singular-value spread under 4x). See
[MuonAdamW](../training/optimizer.md).

### test_tasks.py

Uses an in-memory `ToyTask` and pyarrow tables, so no Hub access. Verifies
full and sliced views (including step slicing with ceil-division length),
that a mixture covers every example exactly once, deterministically and
interleaved, that repeating a task oversamples it, `HubDataset` row access, that
`shuffle(seed)` equals a numpy `default_rng` permutation, and the
letter-binding format of `render_mc`. See [tasks](../data/tasks.md).

### test_tokenizer.py

A module-scoped fixture trains a throwaway tokenizer on a 32-line corpus with
`256 + 9 + 35` tokens, so nothing under `~/.cache/nanochat` is needed. Tests
check vocabulary size, round-trips including unseen text and emoji, unique
special ids and that special strings in plain text do not collapse,
prepend/append, batch encoding, the exact supervised token set of a multi-turn
conversation, system-message merging, tool-part masking, truncation and
completion priming. See [tokenizer](../data/tokenizer.md).

## What is not covered

There are no tests for the training scripts, the dataloader, bits-per-byte or
CORE evaluation, FP8, or checkpoint round-trips. Those paths are exercised
end to end by `runs/runcpu.sh` and `runs/speedrun.sh` rather than by unit
tests, so changes there should be validated with a short real run (the README
suggests a d12 run for iteration).

## Running subsets

```bash
python -m pytest tests/test_tokenizer.py tests/test_tasks.py tests/test_engine.py tests/test_execution.py -v
```

is the CPU-only set. On a GPU node add `tests/test_optim.py`, and on an
H100 (with the `kernels` package able to fetch FA3) the comparison class in
`tests/test_attention_fallback.py` becomes active. Several files also run
directly with `python -m pytest <file> -v -s`, and `test_attention_fallback.py`
prints hardware detection when executed as a script.
