---
type: "Reference"
title: "Tasks package: datasets, mixtures and evaluation"
openwiki_generated: true
verified:
  - by: openwiki/0.5.2
    at: 2026-09-17T20:15:00.211Z
sources:
  - id: openwiki-source-ca457235ed5b9f351ff341ee
    resource: repo://scripts/chat_sft.py
  - id: openwiki-source-24626c556790549d3c492821
    resource: repo://tasks/arc.py
  - id: openwiki-source-cadf1570739aed4547654b15
    resource: repo://tasks/common.py
  - id: openwiki-source-33310ed5715270657c3342a7
    resource: repo://tasks/gsm8k.py
  - id: openwiki-source-bf593cc009dff7376bbe3723
    resource: repo://tasks/humaneval.py
  - id: openwiki-source-e297faa82133045e0e6a5b3a
    resource: repo://tasks/mmlu.py
  - id: openwiki-source-6343060992f3d87399d31b3b
    resource: repo://tests/test_tasks.py
generated: { by: "claude-code", at: "2026-09-17T20:15:00.211Z" }
---


# Tasks package: datasets, mixtures and evaluation

`tasks/` is the post-training data layer. A **Task** is a dataset of
conversations plus, usually, a way to grade a model's answer. The same objects
feed the SFT mixture in `scripts/chat_sft.py`, the evaluation loops in
`scripts/chat_eval.py`, and the reward in `scripts/chat_rl.py`, so a task's
conversation format is the contract that ties training and evaluation
together.

## Conversation schema

Every `get_example` returns a dict with a `messages` list and optional
task-specific extras. Messages alternate `user`, `assistant`, ... (an optional
leading `system` message is allowed and merged into the first user turn by the
tokenizer). A user `content` is a string. An assistant `content` is either a
string or a list of parts, each `{"type": "text" | "python" | "python_output", "text": ...}`,
which is how tool calls are represented. The tokenizer's `render_conversation`
turns this into token ids and a supervision mask; see
[tokenizer](tokenizer.md).

## Loading data without `datasets`

`load_hub_dataset(repo_id, subset, split)` replaces
`datasets.load_dataset`. It asks the Hub API for the auto-generated parquet
export of that subset and split, downloads every shard once into
`<base_dir>/task_data/<owner--repo>/<subset>/<split>/`, and writes a
`manifest.json` last so that its presence signals a complete download. The
download runs under a `FileLock` on the manifest path, so under `torchrun` only
one rank fetches while the others block and then reuse the files. Shards are
read with pyarrow and concatenated into one table.

`HubDataset` wraps that table: `len`, row access by index (converted to Python
values column by column), and `shuffle(seed)`, which builds a
`np.random.default_rng(seed).permutation` view. The docstring and
`tests/test_tasks.py` assert this reproduces `datasets.Dataset.shuffle(seed)`
exactly, so row order matches earlier nanochat versions that used the
`datasets` library. Every concrete task shuffles with seed 42.

## The Task base class

`Task(start=0, stop=None, step=1)` is a logical view over an underlying
dataset. `len` is `ceil((stop - start) / step)` and `task[i]` maps to physical
index `start + i * step`, then calls the subclass's `get_example`. This is how
`chat_sft` trims the validation sets (`MMLU(..., stop=5200)`,
`GSM8K(..., stop=420)`) to match training ratios without copying data.
Subclasses implement `num_examples`, `get_example`, `eval_type` (`'generative'`
or `'categorical'`) and `evaluate(conversation, response)`.

- **`TaskMixture(tasks)`** concatenates several tasks and shuffles the
  combined index map with `random.Random(42)`, so examples from all tasks are
  interleaved deterministically. Passing a task twice oversamples it; that is
  how `--mmlu-epochs` and `--gsm8k-epochs` work in SFT.
- **`TaskSequence(tasks)`** concatenates without shuffling, for curricula.

## Multiple-choice rendering

`render_mc(question, letters, choices)` formats a prompt as the question, one
`- <choice>=<letter>` line per option, and the instruction to respond with only
the letter. Two deliberate choices are documented in its docstring: the letter
comes *after* the choice because small models bind better that way, and there
is no whitespace between `=` and the letter because the tokenizer has
different ids for `"A"` and `" A"`; the assistant answer is the bare letter, so
the prompt must present the same token. `test_render_mc_letter_binding`
guards this.

## Concrete tasks

| Task | Source | Type | Answer / grading |
|---|---|---|---|
| `ARC(subset, split)` | `allenai/ai2_arc`, `ARC-Easy` or `ARC-Challenge` | categorical | Prompt via `render_mc`; the conversation carries `letters`; `evaluate` asserts the response is one of them and compares to the gold letter. |
| `MMLU(subset="all", split)` | `cais/mmlu` | categorical | Same as ARC with fixed letters A-D and 4 choices asserted; also records `subject`. `auxiliary_train` is the SFT split, `test` the eval split. |
| `GSM8K(subset, split)` | `openai/gsm8k` | generative | The reference answer's `<<expr=result>>` calculator spans are split into `python` and `python_output` parts so SFT teaches tool use; `evaluate` extracts the number after `####` from both reference and response and compares; `reward` returns that as a float for RL. |
| `HumanEval()` | `openai/openai_humaneval` | generative | User message is the function stub; `evaluate` extracts the first code block (or whole completion), prepends the prompt's imports, appends the test and `check(entry_point)`, and runs it in the [sandbox](../inference/code-execution-sandbox.md); success is the process exit status. |
| `SmolTalk(split)` | `HuggingFaceTB/smol-smoltalk` | (no eval) | General conversations; asserts strict user/assistant alternation and string contents, allows a leading system message. |

The categorical/generative distinction drives `scripts/chat_eval.py`:
categorical tasks are scored by comparing logits at the answer position over
the allowed letter tokens, generative tasks by sampling and calling
`evaluate`. See [chat evaluation](../evaluation/chat-evaluation.md).

## Where tasks are used

- SFT trains on `SmolTalk(train)` plus repeated `MMLU(auxiliary_train)` and
  `GSM8K(train)`, and validates on trimmed test splits. See
  [chat SFT](../training/chat-sft.md).
- RL rolls out on `GSM8K(train)` and evaluates pass@k on `GSM8K(test)` using
  `reward`. See [chat RL](../training/chat-rl.md).
- ChatCORE evaluates ARC-Easy, ARC-Challenge, MMLU, GSM8K and HumanEval test
  splits.

## Adding a task

Subclass `Task`, load data with `load_hub_dataset(...).shuffle(seed=42)`,
return conversations in the schema above, declare `eval_type`, and implement
`evaluate`. Register it in the task table in `scripts/chat_eval.py` if it should
be evaluated, and add it to the mixture in `scripts/chat_sft.py` if it should be
trained on. `tests/test_tasks.py` shows the in-memory `ToyTask` pattern for
unit-testing slicing and mixtures without network access.
