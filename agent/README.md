# Autonomous agent internals

See the root [README](../README.md) for setup and the complete testing/runbook.

The agent has three proposal layers:

1. `seed_pipeline.py` reproduces the immutable pointwise FM baseline.
2. `bpr_pipeline.py` is the built-in first knowledge-guided operator.
3. An OpenAI model proposes later single-change candidates through the
   Responses API, using the contract in `context.py` and reference notes in
   `knowledge_base.md`.

Candidates write only an aligned `scores.npy` plus optional diagnostics. The
orchestrator independently validates and evaluates validation scores, records
the tree and code diff, repairs runtime failures, applies a paired second-seed
promotion check to sub-`0.002` gains, and checkpoints after every durable log
entry. That two-seed check is a search heuristic, not a significance test.

The bootstrap candidate uses the textbook BPR pairwise loss with
benchmark-adapted observed-negative sampling, so it is best described as an
observed-negative BPR-style FM rather than a canonical implicit-feedback BPR
sampler. Each eligible positive is paired with a uniformly sampled logged
zero-label impression from the same user; positives from all-positive users
cannot form pairs and are skipped. In the recorded paired check, seed 1 raised
primary by `0.000460` while nDCG@5 declined by `0.000177`; the two-seed mean
primary gain was `0.001194`. Promotion records a search decision, not a
statistically confirmed improvement. This is also a pipeline comparison rather
than a clean loss-only ablation: the pointwise and pairwise epochs contain
different numbers of optimizer steps, changing the effective update and dense
regularization schedules.

Research candidates are passed a generated data view with no test rows.
One-time finalization uses the exact saved winner and a separate view in which
every test outcome is masked. It writes a submission but never calculates test
metrics. The AST/subprocess layer is best-effort; use a container or VM that
mounts only the generated view if generated code must be treated as hostile.

Quick validation-only run with no API key (iterations 0 and 1 only):

```bash
python agent/orchestrator.py --max_iterations 2 --run_dir /tmp/woof-bpr-run
```

Full autonomous run:

```bash
cp .env.example .env
chmod 600 .env
# In .env, replace OPENAI_API_KEY with your real key.
# OPENAI_MODEL defaults to gpt-5.6-terra; change it if unavailable.
python agent/preflight.py --require_llm
python agent/orchestrator.py --max_iterations 50 --wall_clock_hours 6
```

The repository-root `.env` is loaded automatically, and `--model <model-id>`
overrides `OPENAI_MODEL` for one command. Never put the real key in source,
chat, logs, screenshots, or a commit. Nodes after iteration 1 require OpenAI;
`--max_iterations 2` does not call the API. See the
[official OpenAI Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).

Resume:

```bash
python agent/orchestrator.py --resume agent/runs/<run-id>
```

This is defense-in-depth execution, not a hard security sandbox. The API key is
removed from child environments, imports/calls are AST-checked, subprocesses
have timeouts and process-group termination, and Linux gets resource limits.
For high-assurance execution, place the whole runner inside a disposable
container or VM.
