---
type: component
title: Inference Engine, KV cache and tool use
description: How nanochat/engine.py generates text - the KVCache layout and batch-1 prefill replicated to many samples, batched sampling, the per-row state machine that detects python tool calls and injects calculator results as forced tokens, the sampled-versus-forced mask, termination rules, generate versus generate_batch, and the chat CLI loop built on it.
tags: [inference, engine, kv-cache, sampling, tool-use, calculator, chat-cli]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-d22f3da3a0fceb916e861104
    resource: repo://nanochat/engine.py
  - id: openwiki-source-9575b8f00a3027e9afa1b2f7
    resource: repo://scripts/chat_cli.py
  - id: openwiki-source-0e41257659137c7a2c27de1c
    resource: repo://tests/test_engine.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Inference Engine, KV cache and tool use

`nanochat/engine.py` is the fast generation path used by everything that
samples from a model: the chat CLI, sampling during pretraining, chat
evaluation, RL rollouts and the inference benchmark. It works purely on token
id sequences and knows nothing about text, except that it needs the tokenizer
to encode calculator results and to look up special token ids. The naive
reference is `GPT.generate` in `nanochat/gpt.py`; the module's `__main__`
block checks that the Engine reproduces it token for token under greedy
decoding.

## KVCache

`KVCache(batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype)`
preallocates `k_cache` and `v_cache` as `(n_layers, B, T_max, H_kv, D)` zeros,
the layout Flash Attention 3's `flash_attn_with_kvcache` expects, plus an
int32 `cache_seqlens` per row. Attention layers get their slice through
`get_layer_cache(layer_idx)` and the model calls `advance(T)` once after the
last layer, so all layers in a forward pass see the same position;
`get_pos()` assumes every row is at the same position. The cache also carries
`prev_embedding`, the previous token's normalised embedding that the model's
smear mechanism needs during single-token decode (see
[GPT model](../architecture/gpt-model.md)).

`prefill(other)` copies a smaller cache's contents into an empty larger one,
requires matching layer/head geometry and `max_seq_len >= other.max_seq_len`,
sets the position, and expands `prev_embedding` from batch 1 to the new batch.
Backend details of how the cache is written are in
[attention backends](../architecture/attention-backends.md).

## Generation algorithm

`Engine.generate(tokens, num_samples, max_tokens, temperature, top_k, seed)`
is a generator:

1. **Prefill once.** The prompt is run at batch 1 into a cache sized exactly
   to the prompt. The last-position logits are expanded to `num_samples`.
2. **Replicate.** A decode cache of `num_samples` rows is allocated for
   `len(prompt) + max_tokens` positions (or `sequence_len` if `max_tokens` is
   `None`), filled from the prefill cache with `prefill`, and the prefill cache
   is freed. This is why `test_multi_sample_first_token_diversity` exists:
   the first token is sampled independently per row from the expanded logits
   rather than broadcast.
3. **Loop.** Each iteration samples one token per row with
   `sample_next_token` (argmax when `temperature == 0`, otherwise softmax
   sampling with optional top-k, driven by a seeded `torch.Generator`), then
   walks the rows to decide the actual next token, yields the column, and
   forwards the batch of chosen tokens through the model with the decode
   cache to get the next logits.

Stopping conditions are `max_tokens` reached or every row `completed`. A row
completes when its chosen token is `<|assistant_end|>` or `<|bos|>`.

## Tool-use state machine

Each row has a `RowState`: the token history, a `forced_tokens` deque, an
`in_python_block` flag, and the tokens of the current expression. Per step:

- If `forced_tokens` is non-empty, the next token is popped from it instead of
  the sampled one, and the mask for that position is `0`; sampled tokens get
  mask `1`.
- `<|python_start|>` opens a block; subsequent tokens accumulate; on
  `<|python_end|>` the expression is decoded and passed to `use_calculator`.
  If it returns a result, `<|output_start|>`, the encoded result and
  `<|output_end|>` are queued as forced tokens, so the model "sees" the
  answer on the following steps exactly as the SFT data rendered it (see
  [tokenizer](../data/tokenizer.md)). A `None` result queues nothing and
  generation simply continues.

`use_calculator` strips commas and either evaluates a pure arithmetic
expression (digits, `+-*/.()`, no `**`) or a restricted string expression that
must contain `.count(` and no dangerous substrings, both under a 3 second
`SIGALRM` timeout with empty builtins. It is deliberately tiny; the heavier
[execution sandbox](code-execution-sandbox.md) is only for HumanEval.

The yielded `(token_column, token_masks)` pairs are what make RL possible:
`scripts/chat_rl.py` uses the masks to exclude both prompt tokens and forced
tool outputs from the policy-gradient loss. See [chat RL](../training/chat-rl.md).

## generate_batch

`generate_batch(tokens, num_samples, **kwargs)` drains `generate`, appends
each row's tokens until that row's terminator, drops the terminator itself,
and returns `(results, masks)` with the prompt prefixed to each row and prompt
positions masked `0`. It stops as soon as all rows have completed. Evaluation
and RL use it; the CLI uses streaming `generate`.

## The chat CLI

`scripts/chat_cli.py` is a single-process loop over one conversation held as a
token list starting with `<|bos|>`. Each user line is wrapped in
`<|user_start|>`/`<|user_end|>`, `<|assistant_start|>` is appended, and the
Engine streams up to 256 tokens which are printed as they arrive. If
generation stops on the token limit rather than `<|assistant_end|>`, the CLI
appends `<|assistant_end|>` itself so the transcript stays well formed for the
next turn. `-p` sends one prompt and exits; `clear` resets the conversation.
It loads the `sft` source by default (`-i rl` for the RL checkpoint) with
temperature 0.6 and top-k 50.

## Tests

`tests/test_engine.py` runs entirely on CPU with a `MockModel` returning
uniform logits and a `ByteTokenizer`, and checks cache advance/reset/prefill,
per-row first-token diversity, seed reproducibility, temperature-0
determinism, `max_tokens` and `num_samples` being respected, and that
different seeds produce variation.

## Constraints to keep in mind

- Rows share one position counter, so the Engine cannot serve prompts of
  different lengths in one batch; evaluation pads instead (categorical) or
  runs one problem at a time (generative).
- The decode cache is sized from `max_tokens`; callers wanting long outputs
  must pass it explicitly or accept a cache of `sequence_len` positions.
- The Engine holds no text state; conversation management belongs to callers
  such as the CLI.
