---
type: component
title: Tokenizer and conversation rendering
description: The rustbpe/tiktoken BPE tokenizer in nanochat/tokenizer.py - split pattern, the nine special tokens, training and saving, the token_bytes table for bits-per-byte, render_conversation supervision masks and tool parts, render_for_completion, and the tokenizer/model vocabulary invariant.
tags: [tokenizer, bpe, tiktoken, rustbpe, special-tokens, chat-format, loss-mask]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-7f5ef7612eaa1b8c8aeefce8
    resource: repo://nanochat/tokenizer.py
  - id: openwiki-source-c138cdd9025e78e6dd8cd26f
    resource: repo://scripts/chat_eval.py
  - id: openwiki-source-b657b01e69409a21afb5d521
    resource: repo://scripts/chat_rl.py
  - id: openwiki-source-4aba25bb33a353ea9fcb2b2a
    resource: repo://scripts/tok_eval.py
  - id: openwiki-source-7b9ec272a07afbd4da7c7f51
    resource: repo://scripts/tok_train.py
  - id: openwiki-source-bdc49f626c4101be45e0e66a
    resource: repo://tests/test_tokenizer.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Tokenizer and conversation rendering

`nanochat/tokenizer.py` wraps a GPT-4 style byte-level BPE tokenizer and is
also where the chat format is defined: which special tokens delimit turns and
tool calls, and which tokens the model is trained to predict. Training uses
the `rustbpe` library; inference uses a `tiktoken.Encoding` built from the
trained merges.

## Vocabulary and split pattern

`SPLIT_PATTERN` follows GPT-4's regex with one documented change: numbers are
split into runs of at most two digits (`\p{N}{1,2}`) instead of three, which
the author found to be the sweet spot for a 32K vocabulary. There are nine
`SPECIAL_TOKENS`: `<|bos|>` delimits documents in pretraining; the other eight
(`<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`,
`<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`)
are only used from fine-tuning onward to render conversations.

`train_from_iterator(text_iterator, vocab_size)` trains `vocab_size - 9`
ordinary tokens with rustbpe (asserting at least 256 remain for the raw bytes),
then constructs a `tiktoken.Encoding` whose special tokens occupy the last nine
ids. Special-token strings appearing in ordinary text are *not* collapsed to a
single token by `encode`, because it uses `encode_ordinary`; only
`encode_special` produces special ids. `tests/test_tokenizer.py` checks this.

## Training, saving, loading

`scripts/tok_train.py` streams training-split documents through
`parquets_iter_batched`, crops each to `--doc-cap` characters (default 10,000)
and stops after `--max-chars` (default 2B), trains a `--vocab-size` (default
32768) tokenizer, saves it, and round-trips a Unicode test string as a sanity
check. `save` pickles the `tiktoken.Encoding` to `<base_dir>/tokenizer/tokenizer.pkl`
and `get_tokenizer()` unpickles it from the same place.

The script also writes `token_bytes.pt`: a `(vocab_size,)` int32 tensor giving
each token's length in bytes, with 0 for special tokens. It uses
`decode_single_token_bytes` rather than decoding to a string first, because
tokens that are not valid stand-alone UTF-8 would otherwise be miscounted (a
recent fix noted in git history shifted bpb by about 0.05%). `get_token_bytes()`
loads this tensor; it is the denominator for bits-per-byte in
[base model evaluation](../evaluation/base-model-evaluation.md).

`RustBPETokenizer.from_pretrained(name)` wraps a stock tiktoken encoding
(`gpt2`, `cl100k_base`) with `<|endoftext|>` as BOS; `scripts/tok_eval.py`
uses it to compare compression ratios (bytes per token) of the trained
tokenizer against GPT-2 and GPT-4 on news, Korean, code, LaTeX, science and
ClimbMix train/val text.

## Encoding API

`encode(text, prepend=None, append=None, num_threads=8)` accepts a string or a
list of strings (the batch form runs in parallel threads) and optionally
prepends or appends a special token given as an id or as its string. The
pretraining dataloader relies on `prepend=bos`. `decode(ids)` returns text;
`__call__` aliases `encode`; `encode_special` is `lru_cache`d.

## The vocabulary invariant

The model's `vocab_size` must equal the tokenizer's. `checkpoint_manager.build_model`
asserts this when loading any checkpoint, so retraining the tokenizer
invalidates every existing checkpoint, and `base_train` reads the vocabulary
size from the tokenizer rather than from a flag.

## Rendering conversations for SFT

`render_conversation(conversation, max_tokens=2048)` returns `(ids, mask)` where
`mask[i] == 1` marks tokens the assistant is trained to produce. Rules:

- A leading `system` message is merged into the following user message as
  `system + "\n\n" + user` (on a deep copy) and the roles must then strictly
  alternate user, assistant, ...; violations raise assertions.
- The sequence starts with `<|bos|>` (mask 0). User turns are
  `<|user_start|> text <|user_end|>`, all mask 0.
- Assistant turns are `<|assistant_start|>` (mask 0), content, then
  `<|assistant_end|>` (mask 1). String content is mask 1. List content is
  rendered part by part: `text` parts mask 1; `python` parts as
  `<|python_start|> code <|python_end|>` all mask 1, so the model learns to
  invoke the tool; `python_output` parts as `<|output_start|> result <|output_end|>`
  all mask 0, because at inference those tokens come from the interpreter.
- The result is truncated to `max_tokens` to bound memory.

`chat_sft` aligns `mask[1:]` with targets and sets masked targets to `-1`, the
cross-entropy ignore index; see [chat SFT](../training/chat-sft.md).
`visualize_tokenization` prints a colourised view for debugging.

## Rendering prompts for completion

`render_for_completion(conversation)` pops the final assistant message, renders
the rest, and appends `<|assistant_start|>` so the model is primed to answer.
It is used by RL rollouts and by both evaluation loops in `scripts/chat_eval.py`.
The [Engine](../inference/engine.md) then treats `<|assistant_end|>` or
`<|bos|>` as terminators and watches for `<|python_start|>`/`<|python_end|>`
to run the calculator.

## Tests

`tests/test_tokenizer.py` trains a tiny throwaway tokenizer in-process (bytes
plus specials plus 35 merges) so it needs no cached artifacts, and checks
round-trips, special-token uniqueness, prepend/append, batch encoding, the
exact supervised token set of a multi-turn conversation, system-message
merging equivalence, tool-part masking, truncation and completion priming.
