# Files

- [Sandboxed Python execution](code-execution-sandbox.md) - How nanochat/execution.py runs LLM-generated Python - a fresh subprocess with a guard prelude that applies rlimits and disables destructive functions, a scrubbed environment and temporary working directory, the ExecutionResult contract, the explicit non-goals of the sandbox, and its use by HumanEval grading.
- [Inference Engine, KV cache and tool use](engine.md) - How nanochat/engine.py generates text - the KVCache layout and batch-1 prefill replicated to many samples, batched sampling, the per-row state machine that detects python tool calls and injects calculator results as forced tokens, the sampled-versus-forced mask, termination rules, generate versus generate_batch, and the chat CLI loop built on it.
