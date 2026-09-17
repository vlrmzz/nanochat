---
type: component
title: Sandboxed Python execution
description: How nanochat/execution.py runs LLM-generated Python - a fresh subprocess with a guard prelude that applies rlimits and disables destructive functions, a scrubbed environment and temporary working directory, the ExecutionResult contract, the explicit non-goals of the sandbox, and its use by HumanEval grading.
tags: [sandbox, execution, subprocess, humaneval, safety, tool-use]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-d22f3da3a0fceb916e861104
    resource: repo://nanochat/engine.py
  - id: openwiki-source-4a52de821bd3f443cabe20fe
    resource: repo://nanochat/execution.py
  - id: openwiki-source-bf593cc009dff7376bbe3723
    resource: repo://tasks/humaneval.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---

# Sandboxed Python execution

`nanochat/execution.py` exposes one function, `execute_code(code, timeout,
maximum_memory_bytes) -> ExecutionResult`, for running Python that came out of
a language model. Its only production caller is `HumanEval.evaluate` in
`tasks/humaneval.py`, which executes a generated function against the
benchmark's tests. The module docstring credits the OpenAI HumanEval
execution code as inspiration and is explicit about what the sandbox is and is
not.

Note that the calculator used during chat generation is a different, much
narrower mechanism: `use_calculator` in `nanochat/engine.py` evaluates
arithmetic or `.count()` expressions in-process with a character allowlist and
an alarm-based timeout. It never calls `execute_code`. See
[Inference Engine](engine.md).

## What the sandbox does

1. **Fresh interpreter.** The code runs in a new `sys.executable -c` process,
   so it cannot touch the parent's memory. `subprocess.run` handles the
   lifetime; on timeout it kills the child.
2. **Guard prelude.** The child first executes `GUARD`, a string template that:
   - applies `RLIMIT_AS`, `RLIMIT_DATA` and `RLIMIT_STACK` equal to
     `maximum_memory_bytes` (default 256MB), skipped on macOS where the calls
     fail;
   - disables `faulthandler` and the `exit`, `quit` and `help` builtins;
   - sets `OMP_NUM_THREADS=1`;
   - replaces destructive `os` functions (`kill`, `system`, `fork`, `remove`,
     `rmdir`, `rename`, `chmod`, `chown`, `chdir`, `getcwd` and others),
     `shutil.rmtree`/`move`/`chown` and `subprocess.Popen` with `None`;
   - blocks imports of `ipdb`, `joblib`, `resource`, `psutil` and `tkinter`
     by setting their `sys.modules` entries to `None`.
3. **Fresh globals.** The user code is embedded as a `repr` literal and run
   with `exec(compile(code, '<llm>', 'exec'), {'__name__': '__main__'})`, so it
   does not see the guard's own imports (`test_fresh_globals` checks this) and
   arbitrary quotes, backslashes and braces survive embedding.
4. **Isolation of side effects.** The working directory is a
   `tempfile.TemporaryDirectory` that is deleted afterwards, the environment
   is reduced to `PATH=/usr/bin:/bin`, stdin is `DEVNULL`, and stdout and
   stderr are captured as text.

## Result contract

`ExecutionResult` has `success` (exit code 0), `stdout`, `stderr`, `error`,
`timeout` and `memory_exceeded`. On timeout the result has empty output,
`timeout=True` and an explanatory `error`. Otherwise `error` is the last line
of stderr (the exception summary such as `ZeroDivisionError: ...`) or
`"Execution failed"` when stderr is empty, and `memory_exceeded` is set when
`MemoryError` appears in stderr. HumanEval only consults `success`.

## Explicit non-goals

The docstring states that this is **not a true security sandbox**: network
access is not blocked, Python's dynamic features such as `ctypes` can bypass
the monkey-patching, and there is no kernel-level isolation (no seccomp,
containers or virtualisation). It protects against *accidental* destructive
behaviour from generated code, which is the threat model for benchmark
evaluation, and should not be used to run adversarial code.

## Use by HumanEval

`HumanEval.evaluate` assembles a program from the imports at the top of the
prompt, the extracted completion, the dataset's test code and a final
`check(entry_point)` call, then treats `execute_code(program).success` as the
pass/fail signal. See [tasks](../data/tasks.md). Because HumanEval is a
generative task, this runs once per sampled completion inside
`scripts/chat_eval.py` (and during SFT's ChatCORE estimate), so the 5 second
default timeout bounds evaluation time per problem.

## Tests

`tests/test_execution.py` is adversarial by design, one property per test:
happy path, exception capture, prompt killing of an infinite loop, the memory
limit, each disabled destructive call, disabled stdin, writes landing in the
temp directory and not leaking, environment scrubbing of a fake secret,
tricky string content, and fresh globals. These run on any platform, though
the memory-limit test depends on rlimits, which the guard skips on macOS.
