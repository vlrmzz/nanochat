# Files

- [Pretraining data and dataloader](pretraining-dataset.md) - How nanochat stores and streams pretraining text - ClimbMix parquet shards, on-demand download with retries, the train/val split convention, the BOS-aligned best-fit packing dataloader with DDP row-group sharding and resumable state, and its coupling to the tokenizer.
- [Tasks package: datasets, mixtures and evaluation](tasks.md)
- [Tokenizer and conversation rendering](tokenizer.md) - The rustbpe/tiktoken BPE tokenizer in nanochat/tokenizer.py - split pattern, the nine special tokens, training and saving, the token_bytes table for bits-per-byte, render_conversation supervision masks and tool parts, render_for_completion, and the tokenizer/model vocabulary invariant.
