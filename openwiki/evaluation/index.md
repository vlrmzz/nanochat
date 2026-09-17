# Files

- [Base model evaluation: bits per byte and CORE](base-model-evaluation.md)
- [Chat evaluation and ChatCORE](chat-evaluation.md) - How scripts/chat_eval.py grades SFT and RL checkpoints - the generative loop that samples through the Engine and calls each task's evaluate, the categorical loop that scores logits restricted to the allowed answer letters, distributed aggregation, the ChatCORE centred mean, and how SFT reuses the same code during training.
- [Inference benchmark (infer_bench)](inference-benchmark.md) - What scripts/infer_bench.py measures and how - prefill versus decode regimes, time to first token and time per output token, model bandwidth utilisation and model FLOPs utilisation rooflines built from the GPT model's cost accounting and the GPU peak tables, the batch-size sweep, and the machine-readable JSON last line.
